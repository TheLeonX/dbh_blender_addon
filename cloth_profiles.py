"""Read native per-state profiles; regenerate topology without global defaults.

Unedited vanilla cloth remains byte-for-byte native. New topology cannot copy
the old spring topology: it retains per-state gravity/damping/configuration,
projects particle properties from donor poses and preserves total free mass.
Generated distance/bend links are still a different constraint model; native
constraint names/types are displayed rather than falsely treated as scalars.
"""
import copy
import math
import statistics
import struct
from .cloth_native import resource_tag
from .havok_encode import Graph, Writer, Node, array_item_flags
from .cloth_wind import read_wind, write_wind


def scalar_rows(tag,array,fields):
    """Decode only UI scalars using the file's reflected layout, not a graph.

    The ITEM reader validates bounds first. Field widths, offsets, signedness,
    element stride and contiguity are checked; no hard-coded Havok offsets.
    """
    objects=tag.array(array)
    if not objects:return []
    typ,start=objects[0];size=tag.types[tag.base(typ)].size
    if any(obj!=(typ,start+i*size) for i,obj in enumerate(objects)):
        raise ValueError('Noncontiguous native cloth scalar array')
    layout=[]
    for name in fields:
        field,offset=tag.field((typ,0),name);base=tag.types[tag.base(field)]
        kind=base.subtype&255
        if kind==5:fmt='f';width=4
        elif kind in (2,4):
            bits=next(n for flag,n in ((0x2000,8),(0x4000,16),(0x8000,32),(0x10000,64)) if base.subtype&flag)
            width=bits//8;fmt={1:'b',2:'h',4:'i',8:'q'}[width]
            if not base.subtype&512:fmt=fmt.upper()
        else:raise ValueError('Native cloth scalar field is not numeric')
        layout.append((offset,name,fmt,width))
    layout.sort();fmt='<';at=0;names=[]
    for offset,name,storage,width in layout:
        if offset<at or offset+width>size:raise ValueError('Overlapping/out-of-bounds native cloth scalar field')
        if offset>at:fmt+=str(offset-at)+'x'
        fmt+=storage;at=offset+width;names.append(name)
    if size>at:fmt+=str(size-at)+'x'
    rows=struct.iter_unpack(fmt,memoryview(tag.raw)[start:start+len(objects)*size])
    return [Node(0,dict(zip(names,row))) for row in rows]


def state_simulations(cloth, tag):
    result=[]
    for state in cloth.value['clothStateDatas']:
        sims=sorted({cloth.value['operators'][i].value['simClothIndex'] for i in state.value['operators']
                     if tag.types[cloth.value['operators'][i].type].name=='hclSimulateOperator'})
        result.append(sims)
    return result


def read_profiles(payload):
    # Inspect only physics fields. Do not materialize skin/deformer vertex graphs
    # just to display float values in a window or import a painted envelope.
    tag=resource_tag(payload);cloth=next(tag.objects('hclClothData'))
    def pointers(obj,field):return [tag.array(ptr)[0] for ptr in tag.array(tag.field(obj,field))]
    def value(obj,field):return tag.value(tag.field(obj,field))
    operators=pointers(cloth,'operators');states=pointers(cloth,'clothStateDatas')
    assignments=[sorted({value(operators[i],'simClothIndex') for i in value(s,'operators')
                        if tag.types[operators[i][0]].name=='hclSimulateOperator'}) for s in states]
    profiles=[]
    for i,node in enumerate(pointers(cloth,'simClothDatas')):
        sets=[]
        for cs in pointers(node,'staticConstraintSets'):
            kind=tag.types[cs[0]].name;values={'name':value(cs,'name')}
            if kind=='hclStandardLinkConstraintSet':
                values['links']=scalar_rows(tag,tag.field(cs,'links'),('particleA','particleB','stiffness'))
            elif kind=='hclBendStiffnessConstraintSet':
                values['links']=scalar_rows(tag,tag.field(cs,'links'),('bendStiffness',))
            elif kind=='hclLocalRangeConstraintSet':
                values['localConstraints']=scalar_rows(tag,tag.field(cs,'localConstraints'),('particleIndex','maximumDistance'))
            else:continue
            sets.append(Node(cs[0],values))
        sim=dict(name=value(node,'name'),particleDatas=scalar_rows(tag,tag.field(node,'particleDatas'),('mass','invMass','radius')),
                 simulationInfo=Node(0,value(node,'simulationInfo')),
                 landscapeCollisionEnabled=value(node,'landscapeCollisionEnabled'),
                 perInstanceCollidables=tag.array(tag.field(node,'perInstanceCollidables')),
                 staticConstraintSets=sets)
        free=[p.value for p in sim['particleDatas'] if p.value['invMass']>0]
        configs=[cfg for op in operators if tag.types[op[0]].name=='hclSimulateOperator' and value(op,'simClothIndex')==i
                 for cfg in value(op,'simulateOpConfigs')]
        def span(key):return [min(p[key] for p in free),max(p[key] for p in free)] if free else [0.,0.]
        controls=native_controls(Node(0,sim),Node(0,{'simulateOpConfigs':[Node(0,configs[0])]}),tag) if free and configs else None
        profiles.append(dict(index=i,name=sim['name'],states=[value(s,'name') for s,ids in zip(states,assignments) if i in ids],
            mass_range=span('mass'),radius_range=span('radius'),free_particles=len(free),
            total_mass=sum(p['mass'] for p in free),gravity=list(sim['simulationInfo'].value['gravity'][:3]),
            damping=sim['simulationInfo'].value['globalDampingPerSecond'],
            configs=[dict(name=c['name'],substeps=c['subSteps'],iterations=c['numberOfSolveIterations'],adapt_stiffness=bool(c['adaptConstraintStiffness'])) for c in configs],
            constraints=[dict(name=value(cs,'name'),type=tag.types[cs[0]].name) for cs in pointers(node,'staticConstraintSets')],
            colliders=len(sim['perInstanceCollidables']),world_collision=bool(value(node,'landscapeCollisionEnabled')),controls=controls))
    default=next((ids[0] for s,ids in zip(states,assignments) if value(s,'name').lower()=='dynamic' and len(ids)==1),0)
    return dict(cloth=value(cloth,'name'),profiles=profiles,default_index=default,wind=read_wind(payload))


class Nearest:
    """Small dependency-free KD tree for donor particle-property projection."""
    def __init__(self,points):
        self.points=points
        def tree(ids,depth=0):
            if not ids:return None
            axis=depth%3;ids.sort(key=lambda i:points[i][axis]);mid=len(ids)//2
            return ids[mid],axis,tree(ids[:mid],depth+1),tree(ids[mid+1:],depth+1)
        self.root=tree(list(range(len(points))))
    def query(self,point):
        best=[float('inf'),-1]
        def visit(node):
            if node is None:return
            i,axis,left,right=node;distance=sum((a-b)**2 for a,b in zip(point,self.points[i]))
            if (distance,i)<tuple(best):best[:]=distance,i
            gap=point[axis]-self.points[i][axis]
            visit(left if gap<0 else right)
            if gap*gap<=best[0]:visit(right if gap<0 else left)
        visit(self.root)
        if best[1]<0:raise ValueError('No native cloth particles for property projection')
        return best[1]


def project_particles(sim,vertices,weights):
    original=sim.value;free=[i for i,p in enumerate(original['particleDatas']) if p.value['invMass']>0]
    if not free:raise ValueError('Native donor has no free particles')
    positions=original['simClothPoses'][0].value['positions']
    if len(positions)!=len(original['particleDatas']):raise ValueError('Native donor pose/particle count mismatch')
    # Project the complete donor, including its anchors. Searching only free
    # particles incorrectly made native pinned regions simulate in every state.
    tree=Nearest([p[:3] for p in positions])
    ids=[tree.query(v['position']) for v in vertices]
    inherited=[copy.deepcopy(original['particleDatas'][i].value) for i in ids]
    total=sum(original['particleDatas'][i].value['mass'] for i in free)
    projected=sum(p['mass'] for p,w in zip(inherited,weights) if w>1e-6 and p['invMass']>0)
    if not math.isfinite(total) or total<=0 or not math.isfinite(projected) or projected<=0:
        raise ValueError('Invalid native cloth mass distribution')
    scale=total/projected
    for p,w in zip(inherited,weights):
        p['mass']=p['mass']*scale if w>1e-6 and p['invMass']>0 else 0.
        p['invMass']=1/p['mass'] if p['mass'] else 0.
    limits={}
    for cs in original['staticConstraintSets']:
        if 'localConstraints' in cs.value:
            for n in cs.value['localConstraints']:
                v=n.value;limits.setdefault(v['particleIndex'],[]).append(v)
    mapped=[]
    for i,old in enumerate(ids):
        # Multiple native limit operators cannot be folded into one safely.
        values=limits.get(old,[])
        if len(values)>1:raise ValueError('Multiple donor distance limits per particle need a separate topology encoder')
        mapped.append(copy.deepcopy(values[0]) if values else None)
    return inherited,mapped


def native_controls(sim,simulate,tag):
    free=[p.value for p in sim.value['particleDatas'] if p.value['invMass']>0]
    config=simulate.value['simulateOpConfigs'][0].value
    stiffness=[];bend=[]
    for cs in sim.value['staticConstraintSets']:
        if tag.types[cs.type].name=='hclStandardLinkConstraintSet':
            for node in cs.value['links']:
                l=node.value;total=sum(sim.value['particleDatas'][l[k]].value['invMass'] for k in ('particleA','particleB'))
                if total:(bend if 'bend' in cs.value.get('name','').lower() else stiffness).append(l['stiffness']*total)
        elif tag.types[cs.type].name=='hclBendStiffnessConstraintSet':
            bend.extend(node.value['bendStiffness'] for node in cs.value['links'])
    effective=min(1.,max(0.,statistics.median(stiffness))) if stiffness else 1.
    bending=min(1.,max(0.,statistics.median(bend))) if bend else 1.
    distances=[n.value['maximumDistance'] for cs in sim.value['staticConstraintSets'] if 'localConstraints' in cs.value
               for n in cs.value['localConstraints'] if sim.value['particleDatas'][n.value['particleIndex']].value['invMass']>0]
    return dict(mass=statistics.median(p['mass'] for p in free),radius=max(p['radius'] for p in free),
                stiffness=effective,bend=bending,elasticity=0.,movement=1.,bounciness=1-sim.value['simulationInfo'].value['globalDampingPerSecond'],
                max_distance=statistics.median(distances) if distances else 0.,damping=sim.value['simulationInfo'].value['globalDampingPerSecond'],
                substeps=config['subSteps'],iterations=config['numberOfSolveIterations'],collisions=bool(sim.value['perInstanceCollidables']),
                world_collision=bool(sim.value['landscapeCollisionEnabled']),_inherit_profile=True)


def build_profiles(payload,vertices,faces,inverses,name,settings,type_payload=None):
    """One rebuilt sim/buffer per native sim, preserving named state assignments."""
    from .cloth_build import build
    from .cloth_integrity import validate_collider_metadata
    from .cloth_constraints import repair_authored_constraints
    tag=resource_tag(payload);original_graph=Graph(tag);original=original_graph.read(next(tag.objects('hclClothData')))
    assignments=state_simulations(original,tag)
    if any(len(ids)>1 for ids in assignments):raise ValueError('A native state uses simultaneous simulations; preserving it requires a verified multi-simulation topology mapping')
    if original.value['actions'] or any(s.value['actions'] for s in original.value['simClothDatas']):
        raise ValueError('Serialized native cloth actions need their own new-topology remapping')
    variants=[];reports=[]
    for index,sim in enumerate(original.value['simClothDatas']):
        candidates=[op for op in original.value['operators'] if tag.types[op.type].name=='hclSimulateOperator' and op.value['simClothIndex']==index]
        if len(candidates)!=1 or len(candidates[0].value['simulateOpConfigs'])!=1:
            raise ValueError('Multiple native solver configurations need a verified state/configuration remapping')
        simulate=candidates[0]
        controls=native_controls(sim,simulate,tag)
        # User solver overrides are explicit per mesh; inheritance is default.
        controls.update(settings.get('overrides',{}));controls['_source_sim']=index
        if settings.get('world_collision') is not None:controls['world_collision']=settings['world_collision']
        if 'damping' in settings.get('overrides',{}) and 'bounciness' not in settings['overrides']:
            controls['bounciness']=1-controls['damping']
        for field in ('mass','radius','max_distance'):
            controls['_override_'+field]=field in settings.get('overrides',{})
        raw,report=build(payload,vertices,faces,inverses,name,controls,type_payload=type_payload)
        t=resource_tag(raw);g=Graph(t);root=g.read(next(t.objects('hclClothData')))
        variants.append((t,g,root));reports.append(report)
    if not variants:raise ValueError('No native simulation profiles')
    t,g,root=variants[0];merged_buffers=list(root.value['bufferDefinitions'][:2]);merged_ops=list(root.value['operators'][:2]);merged_sims=[];maps=[]
    for index,(other_t,other_g,other) in enumerate(variants):
        part=g.rebase(other,other_g) if index else other
        buffers={0:0,1:1};operators={0:0,1:1}
        for i,buffer in enumerate(part.value['bufferDefinitions'][2:],2):
            buffers[i]=len(merged_buffers)
            if buffer.value['type']==1:buffer.value['subType']=index
            buffer.value['name']+=f' [{index}]';merged_buffers.append(buffer)
        for i,op in enumerate(part.value['operators'][2:],2):
            operators[i]=len(merged_ops);v=op.value;v['operatorID']=operators[i]
            if 'simClothIndex' in v:v['simClothIndex']=index
            for field in ('refBufferIdx','inputBufferIdx','outputBufferIdx','bufferIdx_A','bufferIdx_B','bufferIdx_C'):
                if field in v:v[field]=buffers[v[field]]
            for access in v['usedBuffers']:
                for field in ('bufferIndex','shadowBufferIndex'):access.value[field]=buffers[access.value[field]]
            merged_ops.append(op)
        sim=part.value['simClothDatas'][0]
        sim.value['simOpIds']=[operators[i] for i in sim.value['simOpIds']]
        sim.value['name']=original.value['simClothDatas'][index].value['name']
        merged_sims.append(sim);maps.append((part,buffers,operators))
    states=[]
    for si,(old,indices) in enumerate(zip(original.value['clothStateDatas'],assignments)):
        if indices:
            part,buffers,operators=maps[indices[0]];state=copy.deepcopy(part.value['clothStateDatas'][si])
            state.value['operators']=[operators[i] for i in state.value['operators']]
            for access in state.value['usedBuffers']:
                for field in ('bufferIndex','shadowBufferIndex'):access.value[field]=buffers[access.value[field]]
            state.value['usedSimCloths']=indices
        else:state=copy.deepcopy(root.value['clothStateDatas'][si])
        # Dependency graph indexes are state-local, not global operator IDs.
        states.append(state)
    root.value.update(bufferDefinitions=merged_buffers,operators=merged_ops,simClothDatas=merged_sims,clothStateDatas=states)
    encoded=Writer(g).pack(root);label=name.encode('utf8')
    result=payload[:16]+struct.pack('<I',len(label))+label+struct.pack('<I',len(encoded))+encoded+payload[tag.root[2]:]
    result=write_wind(result,settings.get('wind',{}))
    check=resource_tag(result);readback=Graph(check).read(next(check.objects('hclClothData')))
    validate_collider_metadata(readback);assert array_item_flags(check)[1]==[]
    assert repair_authored_constraints(result)==(result,[])
    if state_simulations(readback,check)!=assignments:raise ValueError('Native cloth state/simulation assignment readback failed')
    baseline=[(check.field(e,'blendWeight')[1],1.) for op in check.objects('hclBlendSomeVerticesOperator') for e in check.array(check.field(op,'blendEntries'))]
    report=dict(reports[0],baseline=baseline,settings=copy.deepcopy(settings),native_profiles=read_profiles(result),
                profile_count=len(merged_sims),profile_settings_inherited=not bool(settings.get('overrides')))
    return result,report
