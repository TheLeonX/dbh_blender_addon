"""Settings-only edits: patch reflected scalars, never replace native topology.

Pins, state transitions, operators, bind poses and constraint rest lengths
remain native. Elasticity is a link-compliance approximation; bounciness is
an inverse-damping proxy, NOT collision restitution.
"""
import math
import struct
from .cloth_native import resource_tag
from .cloth_constraints import link_coefficient


def patch_settings(payload, overrides):
    if not overrides: return payload
    tag = resource_tag(payload)
    def value(obj, field): return tag.value(tag.field(obj, field))
    def pointers(obj, field): return [tag.array(p)[0] for p in tag.array(tag.field(obj, field))]
    def scalar(obj, field, number):
        if not math.isfinite(number): raise ValueError('Non-finite cloth setting')
        tag.write_float(tag.field(obj, field), number)
    def integer(obj, field, number):
        typ, at = tag.field(obj, field); base = tag.types[tag.base(typ)]
        if base.subtype & 255 not in (2, 4): raise ValueError('Expected native cloth integer')
        bits = next(n for flag,n in ((0x2000,8),(0x4000,16),(0x8000,32),(0x10000,64)) if base.subtype & flag)
        tag.raw[at:at+bits//8] = int(number).to_bytes(bits//8,'little',signed=bool(base.subtype & 512))
    for key, number in overrides.items():
        if key not in ('mass','radius','stiffness','bend','elasticity','bounciness','movement','max_distance','damping','substeps','iterations','collisions'):
            raise ValueError('Unknown cloth setting '+key)
        if not isinstance(number,(int,float)) or not math.isfinite(number): raise ValueError('Invalid cloth '+key)
        if key in ('stiffness','bend','elasticity','bounciness','movement','damping') and not 0<=number<=1:
            raise ValueError(key+' must be 0..1')
        if key in ('mass','radius','max_distance') and (number<0 or key=='mass' and number==0):
            raise ValueError('Invalid cloth '+key)
        if key in ('iterations','substeps') and (type(number) is not int or not 1<=number<=16):
            raise ValueError('Cloth solver counts must be integers 1..16')
        if key=='collisions' and type(number) is not bool: raise ValueError('Cloth collisions must be a boolean')
    cloth = next(tag.objects('hclClothData'))
    for sim in pointers(cloth,'simClothDatas'):
        particles = tag.array(tag.field(sim,'particleDatas'))
        old_inverse = [value(p,'invMass') for p in particles]
        for particle, inverse in zip(particles,old_inverse):
            # Never unpin a native fixed particle when mass/radius changes.
            if inverse<=0: continue
            if 'mass' in overrides:
                scalar(particle,'mass',overrides['mass']); scalar(particle,'invMass',1/overrides['mass'])
            if 'radius' in overrides: scalar(particle,'radius',overrides['radius'])
        if 'mass' in overrides: scalar(sim,'totalMass',sum(value(p,'mass') for p in particles))
        if 'radius' in overrides: scalar(sim,'maxParticleRadius',max(value(p,'radius') for p in particles))
        info = tag.field(sim,'simulationInfo')
        if 'bounciness' in overrides: scalar(info,'globalDampingPerSecond',1-overrides['bounciness'])
        elif 'damping' in overrides: scalar(info,'globalDampingPerSecond',overrides['damping'])
        for cs in pointers(sim,'staticConstraintSets'):
            kind = tag.types[cs[0]].name
            if kind=='hclStandardLinkConstraintSet':
                bend = 'bend' in value(cs,'name').lower()
                key = 'bend' if bend else 'stiffness'
                for link in tag.array(tag.field(cs,'links')):
                    a,b = value(link,'particleA'),value(link,'particleB')
                    if not 0<=a<len(particles) or not 0<=b<len(particles): raise ValueError('Invalid cloth link index')
                    effective = value(link,'stiffness')*(old_inverse[a]+old_inverse[b])
                    if key in overrides: effective = overrides[key]
                    if not bend and 'elasticity' in overrides:
                        # An absolute compliance target, not a multiplier of an
                        # already edited link (reimport/export must not compound).
                        effective = overrides.get('stiffness',1.)*(1-overrides['elasticity'])
                    if 'mass' in overrides or key in overrides or not bend and 'elasticity' in overrides:
                        scalar(link,'stiffness',link_coefficient(min(1.,max(0.,effective)),value(particles[a],'invMass'),value(particles[b],'invMass')))
            elif kind=='hclBendStiffnessConstraintSet' and 'bend' in overrides:
                for link in tag.array(tag.field(cs,'links')):
                    scalar(link,'bendStiffness',overrides['bend'])
            elif kind=='hclLocalRangeConstraintSet' and 'max_distance' in overrides:
                distance = overrides['max_distance']
                if not distance: scalar(cs,'stiffness',0.)
                else:
                    for constraint in tag.array(tag.field(cs,'localConstraints')):
                        old = value(constraint,'maximumDistance')
                        if old==struct.unpack('<f',struct.pack('<f',distance))[0]:continue
                        scalar(constraint,'maximumDistance',distance)
                        for field in ('maxNormalDistance','minNormalDistance'):
                            scalar(constraint,field,value(constraint,field)*(distance/old) if old else distance*(1 if field.startswith('max') else -1))
    for op in pointers(cloth,'operators'):
        if tag.types[op[0]].name!='hclSimulateOperator': continue
        for config in tag.array(tag.field(op,'simulateOpConfigs')):
            for key,field in (('substeps','subSteps'),('iterations','numberOfSolveIterations')):
                if key in overrides: integer(config,field,overrides[key])
            if 'collisions' in overrides:
                if value(config,'instanceCollidablesUsed'): raise ValueError('Explicit collision subsets require a separate native editor')
                integer(config,'useAllInstanceCollidables',overrides['collisions'])
    result = bytes(tag.raw)
    resource_tag(result)
    if len(result)!=len(payload): raise ValueError('Scalar cloth edit changed native layout')
    return result


def apply_settings(index, requests, payloads, resources):
    planned={}
    for binding, overrides in requests:
        values={k:v for k,v in overrides.items() if k!='movement'}
        if not values: continue
        if binding.resource in planned and planned[binding.resource][1]!=values:
            raise ValueError('Meshes sharing one cloth resource have conflicting physics settings')
        planned[binding.resource]=binding,values
    for ri,(binding,values) in planned.items():
        rec=index.package.container.records[ri]; target=resources if rec.external else payloads
        original=target.get(ri,bytes(binding.tag.raw)); result=patch_settings(original,values)
        if result!=original: target[ri]=result
