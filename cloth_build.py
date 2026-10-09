"""Experimental new-topology Havok cloth authoring, using native donor types.

No Havok SDK or converter is required. The native TYPE table and state names
come from the target character; particles, links, poses and skinning blocks
are rebuilt from the exported mesh. Never use an unrelated rig as a donor.
"""
import copy
import math
import struct

from .cloth_native import resource_tag
from .havok_encode import Graph, Writer, Node
from .havok_tag import TagFile
from .cloth_constraints import link_coefficient, LINK_NAMES


def val(value): return value.value if isinstance(value, Node) else value
def length(v): return math.sqrt(sum(x*x for x in v))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def packed_vector(graph, vector):
    """Three signed mantissas plus a shared float exponent (native SSE format)."""
    maximum = max(abs(x) for x in vector)
    if not math.isfinite(maximum) or maximum > 1e10: raise ValueError('Invalid cloth vector')
    exponent = math.ceil(math.log2(maximum / 32767.)) if maximum else -30
    unit = 2.**exponent
    mantissas = [int(round(x/unit)) for x in vector]
    if any(not -32768 <= x <= 32767 for x in mantissas): raise ValueError('Cloth packed-vector overflow')
    return graph.new('hkPackedVector3', values=mantissas+[(127+exponent-16)<<7])


def skin_blocks(graph, skin, vertices, inverse_binds):
    groups = {n: [] for n in range(1,9)}
    influences = []
    for i,v in enumerate(vertices):
        weights = sorted([(b-1,w) for b,w in zip(v['bones'],v['weights']) if w>0],key=lambda x:-x[1])[:8]
        if not weights or any(b<0 or b>=len(inverse_binds) or not math.isfinite(w) for b,w in weights):
            raise ValueError('Every cloth vertex needs valid native bone weights (not synthetic Root)')
        total=sum(w for b,w in weights)
        weights=[(b,w/total) for b,w in weights]
        influences.append(weights); groups[len(weights)].append(i)
    subset=sorted({b for ws in influences for b,w in ws}); lookup={b:i for i,b in enumerate(subset)}
    skin.value['transformSubset']=subset
    skin.value['boneFromSkinMeshTransforms']=[inverse_binds[b] for b in subset]
    deform=skin.value['objectSpaceDeformer']
    deform.value.update({k:[] for k in deform.value if k.endswith('BlendEntries')})
    controls=[]; local=[]
    names={1:'One',2:'Two',3:'Three',4:'Four',5:'Five',6:'Six',7:'Seven',8:'Eight'}
    opcodes={1:3,2:2,3:1,4:0,5:8,6:7,7:6,8:5}
    for count,indices in groups.items():
        for start in range(0,len(indices),16):
            batch=indices[start:start+16]; batch+= [batch[-1]]*(16-len(batch))
            bones=[]; weights=[]
            for vertex in batch:
                entries=influences[vertex]
                bones.extend(lookup[b] for b,w in entries)
                # Preserve a unit sum after byte quantization, including zero tails.
                quant=[int(math.floor(w*255)) for b,w in entries]
                remainder=255-sum(quant)
                order=sorted(range(count),key=lambda n:entries[n][1]*255-quant[n],reverse=True)
                for n in order[:remainder]:quant[n]+=1
                # Native cases 5..8 mask each 16-bit value with 0xff00
                # and multiply by 1/(255*256). Low-byte weights become zero.
                weights.extend(q << 8 if count > 4 else q for q in quant)
            block=graph.new('hclObjectSpaceDeformer::'+names[count]+'BlendEntryBlock',vertexIndices=batch,boneIndices=bones)
            if count>1:block.value['boneWeights']=weights
            deform.value[names[count].lower()+'BlendEntries'].append(block)
            local.append(graph.new('hclObjectSpaceDeformer::LocalBlockPNTB',
                localPosition=[packed_vector(graph,vertices[i]['position']) for i in batch],
                localNormal=[packed_vector(graph,vertices[i]['normal']) for i in batch],
                localTangent=[packed_vector(graph,vertices[i].get('tangent',(1,0,0,1))[:3]) for i in batch],
                localBiTangent=[packed_vector(graph,tuple(x*vertices[i].get('tangent',(1,0,0,1))[3] for x in
                               cross(vertices[i]['normal'],vertices[i].get('tangent',(1,0,0,1))[:3]))) for i in batch]))
            controls.append(opcodes[count])
    deform.value.update(controlBytes=controls,startVertexIndex=0,endVertexIndex=len(vertices)-1,partialWrite=0)
    skin.value.update(localPNTBs=local,localUnpackedPNTBs=[])
    return subset


def connected_pieces(count,edges,seams):
    neighbors=[set() for _ in range(count)]
    for a,b in (*edges,*seams):
        neighbors[a].add(b);neighbors[b].add(a)
    unseen=set(range(count));pieces=[]
    while unseen:
        pending=[min(unseen)];unseen.remove(pending[0]);component=[]
        while pending:
            i=pending.pop();component.append(i)
            add=neighbors[i]&unseen;unseen-=add;pending.extend(sorted(add))
        pieces.append(component)
    return pieces


def geometry(vertices,faces,require_pins=True):
    if not 3<=len(vertices)<=65535 or not faces: raise ValueError('Cloth needs 3..65535 exported vertices and triangles')
    positions=[v['position'] for v in vertices]
    if any(len(p)!=3 or not all(math.isfinite(x) for x in p) for p in positions):raise ValueError('Non-finite cloth position')
    weights=[v['cloth_weight'] for v in vertices]
    if any(not math.isfinite(w) or not 0<=w<=1 for w in weights):raise ValueError('Invalid cloth weight')
    if not any(w>0 for w in weights):raise ValueError('Paint a moving area above 0 in CLOTH_SIMULATION')
    edges={}
    for face in faces:
        if len(set(face))!=3 or any(i<0 or i>=len(vertices) for i in face):raise ValueError('Invalid cloth triangle')
        a,b,c=(positions[i] for i in face)
        if length(cross(tuple(y-x for x,y in zip(a,b)),tuple(y-x for x,y in zip(a,c))))<1e-10:
            raise ValueError('Remove zero-area triangles before exporting cloth')
        for a,b,opposite in ((face[0],face[1],face[2]),(face[1],face[2],face[0]),(face[2],face[0],face[1])):
            edge=tuple(sorted((a,b)));edges.setdefault(edge,[]).append(opposite)
    # Render vertices split at UV/normal seams (including on reimport). Weld
    # coincident positions only when their normalized bone attachment agrees.
    seam_groups={}
    for i,v in enumerate(vertices):
        attachments=sorted((b,w) for b,w in zip(v['bones'],v['weights']) if w>0)
        total=sum(w for b,w in attachments)
        if not total or not math.isfinite(total):raise ValueError('Invalid cloth bone weights')
        key=(tuple((b,round(w/total,6)) for b,w in attachments),tuple(round(x,7) for x in v['position']))
        seam_groups.setdefault(key,[]).append(i)
    seams=[]
    for indices in seam_groups.values():
        for i in indices[1:]:
            seams.append((indices[0],i))
    if require_pins:
        missing=[c for c in connected_pieces(len(vertices),edges,seams) if not any(weights[i]<=1e-6 for i in c)]
        if missing:
            raise ValueError(f'{len(missing)} disconnected cloth piece(s) have no pinned area painted exactly 0 '
                             f'(first: {len(missing[0])} exported vertices). Use Select Unpinned Pieces or Pin Unpinned Pieces in Cloth Simulation.')
    return weights,edges,seams


def weld(vertices,faces,seams):
    """One particle per source vertex, keeping UV/normal splits in render data."""
    parents=list(range(len(vertices)))
    for a,b in seams:parents[b]=parents[a]
    lookup={};sim=[];refs=[];mapping=[]
    for i,v in enumerate(vertices):
        root=parents[i]
        if root not in lookup:lookup[root]=len(sim);sim.append(v);refs.append(i)
        elif abs(v['cloth_weight']-sim[lookup[root]]['cloth_weight'])>1e-6:
            raise ValueError('Cloth paint differs between copies of the same source vertex')
        mapping.append(lookup[root])
    return sim,[tuple(mapping[i] for i in f) for f in faces],refs,mapping


def triangle_inverse(a,b,c):
    """Native triangle frame: centroid, a-centroid, b-centroid, unit normal.

    Verified against native TIE_WRAPPER: its vertices map to (1,0), (0,1),
    and (-1,-1), not the conventional (0,0)/(1,0)/(0,1) frame.
    """
    from .skeleton import inverse,mv
    center=tuple((x+y+z)/3 for x,y,z in zip(a,b,c))
    u=tuple(x-y for x,y in zip(a,center));v=tuple(x-y for x,y in zip(b,center))
    normal=cross(tuple(x-y for x,y in zip(b,a)),tuple(x-y for x,y in zip(c,a)))
    size=length(normal)
    if size<1e-10:raise ValueError('Degenerate cloth mapping triangle')
    normal=tuple(x/size for x in normal)
    matrix=inverse(tuple(tuple(column[r] for column in (u,v,normal)) for r in range(3)))
    translation=tuple(-x for x in mv(matrix,center))
    rows=[list(row)+[translation[r]] for r,row in enumerate(matrix)]+[[0,0,0,1]]
    return [rows[r][c] for c in range(4) for r in range(4)]


def mesh_map(graph,source,skin,vertices,sim_vertices,sim_faces,mapping):
    tag=graph.tag
    prototype=next((o for o in source.value['operators'] if tag.types[o.type].name=='hclObjectSpaceMeshMeshDeformPNOperator'),None)
    op=copy.deepcopy(prototype) if prototype else graph.new('hclObjectSpaceMeshMeshDeformPNOperator')
    incidence={}
    # Havok uses the raw game index-buffer winding.
    for n,face in enumerate(sim_faces):
        for vertex in face:incidence.setdefault(vertex,n)
    triangles=sorted({incidence[i] for i in mapping});subset={n:i for i,n in enumerate(triangles)}
    matrices=[triangle_inverse(*(sim_vertices[i]['position'] for i in reversed(sim_faces[n]))) for n in triangles]
    temp=copy.deepcopy(skin)
    mapped=[dict(v,bones=(subset[incidence[mapping[i]]]+1,),weights=(1.,)) for i,v in enumerate(vertices)]
    skin_blocks(graph,temp,mapped,matrices)
    locals=[graph.new('hclObjectSpaceDeformer::LocalBlockPN',localPosition=b.value['localPosition'],localNormal=b.value['localNormal'])
            for b in temp.value['localPNTBs']]
    op.value.update(name='DBH Welded Mesh Map',inputBufferIdx=2,outputBufferIdx=3,inputTrianglesSubset=triangles,
                    triangleFromMeshTransforms=matrices,objectSpaceDeformer=temp.value['objectSpaceDeformer'],
                    customSkinDeform=0,localPNs=locals,localUnpackedPNs=[])
    return op


def build(payload,vertices,faces,inverse_binds,name,settings=None,type_payload=None):
    settings=dict(mass=.04,radius=.002,stiffness=.8,bend=.15,elasticity=0.,movement=1.,max_distance=.25,damping=.95,substeps=2,iterations=4,collisions=True) | (settings or {})
    if 'bounciness' in settings:settings['damping']=1-settings['bounciness']
    if type(settings['collisions']) is not bool: raise ValueError('Cloth collisions must be enabled or disabled')
    if (not math.isfinite(settings['mass']) or settings['mass']<=0 or not math.isfinite(settings['max_distance']) or
            (settings['max_distance']<0 if settings.get('_inherit_profile') else settings['max_distance']<=0)):
        raise ValueError('Cloth mass and maximum distance must be positive')
    if not math.isfinite(settings['radius']) or settings['radius']<0:
        raise ValueError('Cloth radius must be nonnegative')
    if any(not math.isfinite(settings[k]) or not 0 <= settings[k] <= 1 for k in ('stiffness','bend','damping','elasticity','movement')):
        raise ValueError('Cloth stiffness and damping must be 0..1')
    if any(not isinstance(settings[k], int) or not 1 <= settings[k] <= 16 for k in ('substeps','iterations')):
        raise ValueError('Cloth solver substeps/iterations must be 1..16')
    weights,edges,seams=geometry(vertices,faces)
    sim_vertices,sim_faces,references,mapping=weld(vertices,faces,seams)
    sim_weights,sim_edges,_=geometry(sim_vertices,sim_faces)
    tag=resource_tag(payload);graph=Graph(tag)
    source=graph.read(next(tag.objects('hclClothData')))
    wrapper_end=tag.root[2]
    original_name=source.value['name']
    type_name=original_name
    if type_payload is not None:
        # Discard only operator prototypes that this compiler always replaces.
        # Preserve the source's native states/colliders/settings, not those of
        # the table donor. Schemas are checked before any class IDs are remapped.
        keep={'hclObjectSpaceSkinPNTBOperator','hclCopyVerticesOperator','hclMoveParticlesOperator',
              'hclSimulateOperator','hclBlendSomeVerticesOperator','hclObjectSpaceMeshMeshDeformPNOperator'}
        source.value['operators']=[o for o in source.value['operators'] if tag.types[o.type].name in keep]
        target=resource_tag(type_payload);target_graph=Graph(target)
        type_name=target_graph.read(next(target.objects('hclClothData'))).value['name']
        source=target_graph.rebase(source,graph);tag=target;graph=target_graph
    source_sim=settings.get('_source_sim',0)
    if not 0<=source_sim<len(source.value['simClothDatas']):raise ValueError('Native source simulation is unavailable')
    collision_source = source.value['simClothDatas'][source_sim]
    inherited,source_limits=(None,None)
    if settings.get('_inherit_profile'):
        from .cloth_profiles import project_particles
        inherited,source_limits=project_particles(collision_source,sim_vertices,sim_weights)
        sim_weights=[w if p['invMass']>0 else 0. for w,p in zip(sim_weights,inherited)]
    from .cloth_integrity import validate_collider_metadata
    # Native initialization needs this array even when pinch detection is off.
    validate_collider_metadata(source)
    colliders = copy.deepcopy(collision_source.value['perInstanceCollidables']) if settings['collisions'] else []
    collision_metadata = copy.deepcopy(collision_source.value['collidablePinchingDatas']) if settings['collisions'] else []
    if len(colliders)>32: raise ValueError('Native cloth collision masks support at most 32 donor colliders')
    collision_mask = (1<<len(colliders))-1
    # Keep the target character's registered shapes/names and transform map.
    # Retain externally bound (-1) maps/names for the game's original binding;
    # assigning unrelated shapes or guessing new bone maps would be unsafe.
    collision_map = copy.deepcopy(collision_source.value['collidableTransformMap'])
    if colliders and (collision_map.value['transformSetIndex'] != -1 or
                      collision_map.value['transformIndices'] or collision_map.value['offsets']):
        raise ValueError('Internal donor collision transform maps need another verified encoder; only original external maps are supported')
    # Every reflected class must already be registered in the game's donor.
    skin=next((copy.deepcopy(o) for o in source.value['operators'] if tag.types[o.type].name=='hclObjectSpaceSkinPNTBOperator'),None)
    if skin is None:raise ValueError('Native cloth donor lacks PNTB skinning')
    num_bones=len(inverse_binds)
    if len(source.value['transformSetDefinitions'])!=1 or val(source.value['transformSetDefinitions'][0])['numTransforms']!=num_bones:
        raise ValueError('Cloth donor and native skeleton have different bone counts')
    subset=skin_blocks(graph,skin,vertices,inverse_binds)
    # Actual names are obtained from the donor, not assumed SDK namespace aliases.
    access_type=skin.value['usedBuffers'][0].type
    usage_type=val(skin.value['usedBuffers'][0])['bufferUsage'].type
    def access(index,flags,triangles=0):
        usage=Node(usage_type,graph.zero(usage_type));usage.value.update(perComponentFlags=flags,trianglesRead=triangles)
        item=Node(access_type,graph.zero(access_type));item.value.update(bufferIndex=index,shadowBufferIndex=index,bufferUsage=usage)
        return item
    transform_usage=copy.deepcopy(skin.value['usedTransformSets'])
    for item in transform_usage:
        usage=val(val(item)['transformSetUsage'])
        for n,tracker in enumerate(usage['perComponentTransformTrackers']):
            for key,bitfield in val(tracker).items():
                storage=val(val(bitfield)['storage']);words=[0]*((num_bones+31)//32)
                if n==0 and key in ('read','readBeforeWrite'):
                    for b in subset:words[b//32]|=1<<(b%32)
                storage.update(words=words,numBits=num_bones)
    skin.value.update(name='DBH Skin',operatorID=0,outputBufferIndex=0,transformSetIndex=0,
                      usedBuffers=[access(0,[6,6,6,6])],usedTransformSets=transform_usage)
    def original_op(kind):
        return copy.deepcopy(next(o for o in source.value['operators'] if tag.types[o.type].name==kind and
                             (kind!='hclSimulateOperator' or o.value['simClothIndex']==source_sim)))
    copy_op=original_op('hclCopyVerticesOperator')
    copy_op.value.update(name='DBH Bind Copy',operatorID=1,inputBufferIdx=0,outputBufferIdx=1,startVertexIn=0,startVertexOut=0,
                         numberOfVertices=len(vertices),copyNormals=1,usedBuffers=[access(0,[1,1,0,0]),access(1,[6,6,0,0])],usedTransformSets=[])
    move=original_op('hclMoveParticlesOperator');pair_type=move.value['vertexParticlePairs'][0].type
    pairs=[]
    for i,w in enumerate(sim_weights):
        if w>1e-6:continue
        pair=Node(pair_type,graph.zero(pair_type));pair.value.update(vertexIndex=references[i],particleIndex=i);pairs.append(pair)
    move.value.update(name='DBH Pins',operatorID=2,simClothIndex=0,refBufferIdx=1,vertexParticlePairs=pairs,
                      usedBuffers=[access(1,[1,0,0,0]),access(2,[2,0,0,0])],usedTransformSets=[])
    simulate=original_op('hclSimulateOperator');config=copy.deepcopy(simulate.value['simulateOpConfigs'][0])
    config.value.update(constraintExecution=[0,1,2,-1],instanceCollidablesUsed=[],subSteps=settings['substeps'],
                        numberOfSolveIterations=settings['iterations'],useAllInstanceCollidables=int(bool(colliders)),
                        adaptConstraintStiffness=config.value['adaptConstraintStiffness'] if settings.get('_inherit_profile') else 0)
    simulate.value.update(name='DBH Simulate',operatorID=3,simClothIndex=0,simulateOpConfigs=[config],
                          usedBuffers=[access(1,[1,1,0,0]),access(2,[15,6,0,0],1)],usedTransformSets=[])
    blend=original_op('hclBlendSomeVerticesOperator');blend_type=blend.value['blendEntries'][0].type
    entries=[]
    for i,w in enumerate(weights):
        entry=Node(blend_type,graph.zero(blend_type));entry.value.update(vertexIndex=i,blendWeight=w*settings['movement']);entries.append(entry)
    blend.value.update(name='DBH Cloth Paint',operatorID=5 if seams else 4,bufferIdx_A=3 if seams else 2,bufferIdx_B=0,bufferIdx_C=0,blendEntries=entries,
                       blendNormals=1,blendTangents=0,blendBitangents=0,dynamicBlend=0,
                       usedBuffers=[access(0,[11,11,0,0]),access(3 if seams else 2,[1,1,0,0])],usedTransformSets=[])
    blend.value['blendVertices']=Node(blend.value['blendVertices'].type,graph.zero(blend.value['blendVertices'].type))
    buffers=source.value['bufferDefinitions']
    user=copy.deepcopy(next(b for b in buffers if val(b)['type']==4 and val(b)['bufferLayout'].value['elementsLayout'][2].value['vectorSize']))
    scratch=copy.deepcopy(next(b for b in buffers if val(b)['type']==6))
    simbuffer=copy.deepcopy(next(b for b in buffers if val(b)['type']==1))
    user.value.update(name=name,numVertices=len(vertices),numTriangles=len(faces))
    scratch.value.update(name=name+' Bind',numVertices=len(vertices),numTriangles=0,triangleIndices=[])
    simbuffer.value.update(name=name+' Simulation',numVertices=len(sim_vertices),numTriangles=len(sim_faces),subType=0)
    operators=[skin,copy_op,move,simulate]
    definitions=[user,scratch,simbuffer]
    if seams:
        deformed=copy.deepcopy(scratch);deformed.value['name']=name+' Deformed'
        definitions.append(deformed)
        deform=mesh_map(graph,source,skin,vertices,sim_vertices,sim_faces,mapping)
        deform.value.update(operatorID=4,usedBuffers=[access(2,[1,0,0,0],1),access(3,[2,2,0,0])],usedTransformSets=[])
        operators.append(deform)
    operators.append(blend)
    sim=copy.deepcopy(collision_source)
    particle_type=sim.value['particleDatas'][0].type
    sim.value['particleDatas']=[]
    for w in sim_weights:
        particle=Node(particle_type,graph.zero(particle_type));particle.value.update(mass=settings['mass'] if w>1e-6 else 0,
                         invMass=1/settings['mass'] if w>1e-6 else 0,radius=settings['radius'],friction=1.)
        if inherited is not None:particle.value.update(inherited[len(sim.value['particleDatas'])])
        if inherited is not None and settings.get('_override_mass'):
            particle.value.update(mass=settings['mass'] if w>1e-6 else 0.,invMass=1/settings['mass'] if w>1e-6 else 0.)
        if inherited is not None and settings.get('_override_radius'):particle.value['radius']=settings['radius']
        sim.value['particleDatas'].append(particle)
    links=graph.new('hclStandardLinkConstraintSet',name=LINK_NAMES['DBH Stretch Links'])
    bends=graph.new('hclStandardLinkConstraintSet',name=LINK_NAMES['DBH Bend Links'])
    ranges=graph.new('hclLocalRangeConstraintSet',name='DBH Paint Limits',referenceMeshBufferIdx=1,stiffness=1.,shapeType=0,applyNormalComponent=1)
    def link(a,b,stiffness):
        return graph.new('hclStandardLinkConstraintSet::Link',particleA=a,particleB=b,
            restLength=length(tuple(x-y for x,y in zip(sim_vertices[a]['position'],sim_vertices[b]['position']))),
            stiffness=link_coefficient(stiffness,sim.value['particleDatas'][a].value['invMass'],sim.value['particleDatas'][b].value['invMass']))
    links.value['links']=[link(a,b,settings['stiffness']*(1-settings['elasticity'])) for a,b in sorted(sim_edges)]
    opposite={tuple(sorted(pair)) for pair in sim_edges.values() if len(pair)==2 and len(set(pair))==2}
    bends.value['links']=[link(a,b,settings['bend']) for a,b in sorted(opposite) if (a,b) not in sim_edges and
                         length(tuple(x-y for x,y in zip(sim_vertices[a]['position'],sim_vertices[b]['position'])))>1e-8]
    ranges.value['localConstraints']=[graph.new('hclLocalRangeConstraintSet::LocalConstraint',particleIndex=i,referenceVertex=references[i],
        maximumDistance=settings['max_distance']*w,maxNormalDistance=settings['max_distance']*w,minNormalDistance=-settings['max_distance']*w) for i,w in enumerate(sim_weights) if w>1e-6 and settings['max_distance']>0]
    if source_limits is not None and not settings.get('_override_max_distance'):
        native_ranges=[cs for cs in collision_source.value['staticConstraintSets'] if 'localConstraints' in cs.value]
        if len(native_ranges)>1:raise ValueError('Multiple native distance-limit operators cannot be flattened safely')
        if native_ranges:
            for field in ('stiffness','shapeType','applyNormalComponent'):ranges.value[field]=native_ranges[0].value[field]
        ranges.value['localConstraints']=[]
        for i,(limit,w) in enumerate(zip(source_limits,sim_weights)):
            if limit is None or w<=1e-6:continue
            node=graph.new('hclLocalRangeConstraintSet::LocalConstraint',**limit)
            node.value.update(particleIndex=i,referenceVertex=references[i])
            for field in ('maximumDistance','maxNormalDistance','minNormalDistance'):node.value[field]*=w
            ranges.value['localConstraints'].append(node)
    pose=copy.deepcopy(sim.value['simClothPoses'][0]);pose.value.update(name='DBH Bind Pose',positions=[list(v['position'])+[1.] for v in sim_vertices])
    sim.value.update(name=name+' Simulation',fixedParticles=[i for i,w in enumerate(sim_weights) if w<=1e-6],doNormals=1,simOpIds=[3],
        simClothPoses=[pose],staticConstraintSets=[links,bends,ranges],antiPinchConstraintSets=[],perInstanceCollidables=colliders,
        maxParticleRadius=max(p.value['radius'] for p in sim.value['particleDatas']),staticCollisionMasks=[collision_mask if w>1e-6 else 0 for w in sim_weights],actions=[],totalMass=sum(val(p)['mass'] for p in sim.value['particleDatas']),
        transferMotionEnabled=0,landscapeCollisionEnabled=0,numLandscapeCollidableParticles=0,
        triangleIndices=[i for face in sim_faces for i in reversed(face)],triangleFlips=[0]*len(sim_faces),pinchDetectionEnabled=0,
        perParticlePinchDetectionEnabledFlags=[0]*len(sim_vertices),collidablePinchingDatas=collision_metadata,minPinchedParticleIndex=0,maxPinchedParticleIndex=0,maxCollisionPairs=0)
    if colliders: sim.value['collidableTransformMap'] = collision_map
    else: sim.value['collidableTransformMap'].value.update(transformSetIndex=-1,transformIndices=[],offsets=[])
    world=settings.get('world_collision')
    if world is None:world=bool(collision_source.value['landscapeCollisionEnabled'])
    if type(world) is not bool:raise ValueError('World collision flag must be a boolean')
    if world and len(colliders)>31:raise ValueError('World collision reserves mask bit 31; select at most 31 body colliders')
    sim.value['landscapeCollisionEnabled']=int(world)
    if world:
        sim.value['staticCollisionMasks']=[mask|(0x80000000 if particle.value['invMass']>0 else 0)
            for mask,particle in zip(sim.value['staticCollisionMasks'],sim.value['particleDatas'])]
        sim.value['numLandscapeCollidableParticles']=sum(p.value['invMass']>0 for p in sim.value['particleDatas'])
    virtual=sim.value['virtualCollisionPointsData'];virtual.value=graph.zero(virtual.type)
    sim.value['simulationInfo'].value.update(globalDampingPerSecond=settings['damping'])
    states=[]
    for old in source.value['clothStateDatas']:
        if not old.value['operators']:
            # Some packages retain an unused placeholder state with no graph
            # branches. Do not activate a new simulation in that empty state.
            state=copy.deepcopy(old)
            state.value.update(usedBuffers=[],usedTransformSets=[],usedSimCloths=[])
            states.append(state)
            continue
        state=copy.deepcopy(old);is_skin=old.value['name'].lower()=='skin';ops=[0] if is_skin else list(range(len(operators)))
        accesses=[access(0,[15,15,6,6]),access(1,[7,7,0,0]),access(2,[15,7,0,0],1)]
        if seams:accesses.append(access(3,[7,7,0,0]))
        state.value.update(operators=ops,usedBuffers=[access(0,[6,6,6,6])] if is_skin else accesses,
                           usedTransformSets=copy.deepcopy(transform_usage),usedSimCloths=[] if is_skin else [0])
        dep=state.value['dependencyGraph'];branch=copy.deepcopy(dep.value['branches'][0])
        branch.value.update(branchId=0,stateOperatorIndices=list(range(len(ops))),parentBranches=[],childBranches=[])
        dep.value.update(branches=[branch],rootBranchIds=[0],children=[[i+1] if i+1<len(ops) else [] for i in range(len(ops))],
                         parents=[[i-1] if i else [] for i in range(len(ops))],multiThreadable=0)
        states.append(state)
    source.value.update(name=name,bufferDefinitions=definitions,operators=operators,
                        simClothDatas=[sim],clothStateDatas=states,stateTransitions=[],actions=[])
    validate_collider_metadata(source)
    encoded=Writer(graph).pack(source)
    check=TagFile(encoded);roundtrip=Graph(check).read(next(check.objects('hclClothData')))
    if len(roundtrip.value['simClothDatas'][0].value['particleDatas'])!=len(sim_vertices):raise ValueError('Cloth particle readback failed')
    from .havok_encode import Node as TypedNode
    def plain(value):
        if isinstance(value,TypedNode): return (value.type,plain(value.value))
        if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
        if isinstance(value,list): return [plain(v) for v in value]
        return value
    actual = roundtrip.value['simClothDatas'][0].value
    validate_collider_metadata(roundtrip)
    if plain(actual['perInstanceCollidables']) != plain(colliders): raise ValueError('Native collision shape readback failed')
    if plain(actual['collidablePinchingDatas']) != plain(collision_metadata): raise ValueError('Native collider metadata readback failed')
    if actual['staticCollisionMasks'] != sim.value['staticCollisionMasks']: raise ValueError('Native collision mask readback failed')
    native_name=name.encode('utf8')
    result=payload[:16]+struct.pack('<I',len(native_name))+native_name+struct.pack('<I',len(encoded))+encoded+payload[wrapper_end:]
    resource_tag(result)
    from .cloth_skin_weights import repair_authored_skin_weights
    from .cloth_constraints import repair_authored_constraints
    if repair_authored_constraints(result) != (result, []):
        raise ValueError('Fresh cloth constraint encoding requires migration')
    if repair_authored_skin_weights(result) != (result, []):
        raise ValueError('Fresh cloth export contains legacy low-byte weights')
    final=resource_tag(result)
    baseline=[(final.field(entry,'blendWeight')[1],1.) for op in final.objects('hclBlendSomeVerticesOperator')
              for entry in final.array(final.field(op,'blendEntries'))]
    return result,dict(vertices=len(vertices),triangles=len(faces),pins=len(pairs),stretch_links=len(links.value['links']),
                        bend_links=len(bends.value['links']),seams=len(seams),particles=len(sim_vertices),bones=len(subset),settings=settings,baseline=baseline,
                        collision_count=len(colliders),collision_metadata_count=len(collision_metadata),collision_names=[c.value['name'] for c in colliders],
                        physics_donor=original_name,type_donor=type_name)
