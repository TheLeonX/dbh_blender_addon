"""Native landscape selection, not a runtime world-geometry provider.

SDK setup RVA 0x1652700 enables the flag, marks free particles with mask bit
31 and counts them. Actual contacts still require runtime landscape buffers.
"""
import math
from .cloth_native import resource_tag

WORLD_BIT = 0x80000000


def patch(payload, enabled):
    if enabled is None:return payload
    if type(enabled) is not bool:raise ValueError('World collisions must be a boolean or native inheritance')
    tag=resource_tag(payload)
    def value(obj,name):return tag.value(tag.field(obj,name))
    def integer(field,number):
        typ,at=field;base=tag.types[tag.base(typ)]
        if base.subtype&255 not in (2,4):raise ValueError('World collision field is not a native integer')
        bits=next((n for flag,n in ((0x2000,8),(0x4000,16),(0x8000,32),(0x10000,64)) if base.subtype&flag),None)
        if not bits:raise ValueError('World collision integer width is unsupported')
        tag.raw[at:at+bits//8]=int(number).to_bytes(bits//8,'little',signed=bool(base.subtype&512))
    cloth=next(tag.objects('hclClothData'))
    for ptr in tag.array(tag.field(cloth,'simClothDatas')):
        sim,=tag.array(ptr)
        particles=tag.array(tag.field(sim,'particleDatas'));masks=tag.array(tag.field(sim,'staticCollisionMasks'))
        cols=tag.array(tag.field(sim,'perInstanceCollidables'));old_enabled=bool(value(sim,'landscapeCollisionEnabled'))
        if (enabled or old_enabled) and len(cols)>31:
            raise ValueError('World collision reserves mask bit 31; select at most 31 body colliders')
        if len(particles)!=len(masks):raise ValueError('World collision requires one native collision mask per particle')
        count=0
        for particle,field in zip(particles,masks):
            inv=value(particle,'invMass')
            if not math.isfinite(inv) or inv<0:raise ValueError('Invalid particle inverse mass for world collisions')
            mask=tag.value(field)
            if not 0<=mask<=0xffffffff:raise ValueError('Invalid native collision mask')
            if enabled:
                selected=inv>0;mask=(mask&~WORLD_BIT)|(WORLD_BIT if selected else 0);count+=selected
            elif old_enabled or len(cols)<32:mask&=~WORLD_BIT
            # With 32 ordinary shapes and landscape off, bit 31 belongs to
            # the last body collider. An explicit off must not delete it.
            integer(field,mask)
        integer(tag.field(sim,'landscapeCollisionEnabled'),enabled)
        integer(tag.field(sim,'numLandscapeCollidableParticles'),count)
    result=bytes(tag.raw);check=resource_tag(result)
    if len(result)!=len(payload) or check.types!=tag.types or check.items!=tag.items:
        raise ValueError('World collision edit changed native layout')
    return result


def plan(requests):
    planned={}
    for binding,enabled in requests:
        if enabled is None:continue
        if binding.resource in planned and planned[binding.resource][1]!=enabled:
            raise ValueError('Meshes sharing one cloth resource have conflicting world-collision flags')
        planned[binding.resource]=binding,enabled
    return planned


def apply(index,requests,payloads,resources,reports):
    planned=plan(requests)
    for ri,(binding,enabled) in planned.items():
        rec=index.package.container.records[ri];target=resources if rec.external else payloads
        raw=target.get(ri,bytes(binding.tag.raw));result=patch(raw,enabled)
        if result!=raw:target[ri]=result
        for report in reports:
            if report['resource_id']==binding.asset:report['world_collision']=enabled
