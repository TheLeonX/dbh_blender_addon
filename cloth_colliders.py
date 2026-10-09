"""Explicit selection of same-character native shapes, not guessed mesh export."""
import copy
import struct
from .cloth_native import resource_tag
from .havok_encode import Graph,Writer
from .cloth_integrity import validate_collider_metadata,_named


def donors(package,record):
    cached=getattr(package,'_cloth_donor_rows',{})
    if record in cached:return cached[record]
    from .cloth_route import family
    from .native import u32
    catalog=family(package,record);refs=package.mesh(catalog.cloth).references
    lookup={(r.kind,r.asset):r for r in package.container.records};result=[]
    for i in range(u32(refs,0)&255):
        kind,asset=struct.unpack_from('<II',refs,4+8*i)
        if kind!=2150 or (kind,asset) not in lookup:continue
        rec=lookup[kind,asset]
        raw=package.members[package.record_members[rec.index]].unpacked if rec.external else rec.payload
        tag=resource_tag(raw);root=next(tag.objects('hclClothData'));name=tag.value(tag.field(root,'name'))
        result.append((asset,name,raw))
    cached[record]=result;package._cloth_donor_rows=cached
    return result


def sources(package,record):
    return [(asset,name,raw) for asset,name,raw in donors(package,record) if not name.startswith('DBH_CLOTH_')]


def rows(package,record):
    """Cheap reflected names/classes only; no convex-grid decoding for the UI."""
    result={}
    for asset,label,raw in sources(package,record):
        tag=resource_tag(raw);root=next(tag.objects('hclClothData'))
        for ptr in tag.array(tag.field(root,'simClothDatas')):
            sim=tag.array(ptr)[0]
            for ref in tag.array(tag.field(sim,'perInstanceCollidables')):
                col=tag.array(ref)[0];name=tag.value(tag.field(col,'name'))
                shape=tag.array(tag.field(col,'shape'))[0]
                result.setdefault(name,dict(name=name,shape=tag.types[shape[0]].name,donors=[]))['donors'].append(asset)
    return [dict(row,donors=sorted(set(row['donors']))) for name,row in sorted(result.items())]


def mesh_links(package,metadata,record):
    """UI aliases only: unique native bind matrix AND local bounds match.

    Export always uses native external collider names, never the preview mesh.
    Ambiguous/nonidentity helpers remain native-name-only rather than guessed.
    """
    from .auxiliary_mesh import bindings,collision_slots
    from .native import decode_vertices
    from .skeleton import Rig
    rigs=[Rig(r) for r in package.container.records if r.kind==2138]
    rigs=[r for r in rigs if r.matches(metadata.get('bones',[]))]
    if len(rigs)!=1:return {}
    rig=rigs[0];links=bindings(package,metadata);helpers=[]
    for key in collision_slots(package,metadata):
        link=links.get(key)
        if not link or not link['bone']:continue
        ri,mi=map(int,key.split(':'));md=package.mesh(ri)
        pts=[v['position'] for v in decode_vertices(md,md.flat()[mi])]
        if not pts:continue
        bounds=tuple(min(p[i] for p in pts) for i in range(3))+tuple(max(p[i] for p in pts) for i in range(3))
        joint=rig.joints[link['bone']-1]
        matrix=[joint.world_linear[r][c] if r<3 and c<3 else joint.world_position[r]-(metadata.get('y_offset',0.) if r==1 else 0.) if c==3 and r<3 else float(r==c) for c in range(4) for r in range(4)]
        helpers.append((key,bounds,matrix))
    result={}
    for asset,label,raw in sources(package,record):
        tag=resource_tag(raw);root=next(tag.objects('hclClothData'))
        for ptr in tag.array(tag.field(root,'simClothDatas')):
            sim=tag.array(ptr)[0]
            for ref in tag.array(tag.field(sim,'perInstanceCollidables')):
                col=tag.array(ref)[0];shape=tag.array(tag.field(col,'shape'))[0];kind=tag.types[shape[0]].name
                value=lambda n:tag.value(tag.field(shape,n))
                if kind=='hclConvexGeometryShape':
                    box=tag.field(shape,'objAabb');bounds=tuple(tag.value(tag.field(box,'min'))[:3])+tuple(tag.value(tag.field(box,'max'))[:3])
                elif kind in ('hclCapsuleShape','hclTaperedCapsuleShape'):
                    a,b=(value('start'),value('end')) if kind=='hclCapsuleShape' else (value('small'),value('big'))
                    ra,rb=(value('radius'),)*2 if kind=='hclCapsuleShape' else (value('smallRadius'),value('bigRadius'))
                    bounds=tuple(min(a[i]-ra,b[i]-rb) for i in range(3))+tuple(max(a[i]+ra,b[i]+rb) for i in range(3))
                else:continue
                matrix=tag.value(tag.field(col,'transform'));name=tag.value(tag.field(col,'name'))
                # hkTransform's SIMD rotation-column W lanes are padding, not
                # the affine bottom row (the leg shape has nonzero lane 3).
                lanes=(0,1,2,4,5,6,8,9,10,12,13,14)
                matches=[key for key,box,mat in helpers if max(abs(x-y) for x,y in zip(bounds,box))<2e-5 and max(abs(matrix[i]-mat[i]) for i in lanes)<2e-4]
                if len(matches)==1:result[name]=matches[0]
    return result


def default_names(payload):
    from .cloth_profiles import read_profiles
    profile=read_profiles(payload);tag=resource_tag(payload);root=next(tag.objects('hclClothData'))
    sim=tag.array(tag.array(tag.field(root,'simClothDatas'))[profile['default_index']])[0]
    return [tag.value(tag.field(tag.array(ref)[0],'name')) for ref in tag.array(tag.field(sim,'perInstanceCollidables'))]


def select(payload,selection,pool):
    """Reuse native shapes/metadata and external names; remap masks by identity.

    None is byte-exact inheritance. [] deliberately removes collisions. Existing
    per-state shape variants/masks stay intact. New selected shapes are enabled
    only for free particles. No particle, spring, pose or state graph is rebuilt.
    """
    if selection is None:return payload
    if not isinstance(selection,list) or any(not isinstance(n,str) or not n for n in selection):
        raise ValueError('Invalid native collider selection')
    if len(set(selection))!=len(selection) or len(selection)>32:
        raise ValueError('Choose at most 32 different native colliders per cloth mesh')
    tag=resource_tag(payload);graph=Graph(tag);cloth=graph.read(next(tag.objects('hclClothData')))
    validate_collider_metadata(cloth)
    if all([c.value['name'] for c in s.value['perInstanceCollidables']]==selection for s in cloth.value['simClothDatas']):return payload
    available={};preferred={};own={}
    # Prefer the selected garment's own native variants, then same-family sources.
    for source_index,raw in enumerate([payload]+list(pool)):
        # Usually all chosen shapes already exist in this garment's states.
        # Do not decode unrelated full deformer graphs in that common case.
        if source_index and set(selection)<=own.keys():break
        if source_index==0:t,g,root=tag,graph,cloth
        else:
            t=resource_tag(raw);g=Graph(t);root=g.read(next(t.objects('hclClothData')))
        validate_collider_metadata(root)
        if source_index==0:
            from .cloth_profiles import read_profiles
            default=read_profiles(raw)['default_index']
        for sim_index,s in enumerate(root.value['simClothDatas']):
            for col,meta in zip(s.value['perInstanceCollidables'],s.value['collidablePinchingDatas']):
                name=col.value['name'];candidate=(col,meta,g)
                available.setdefault(name,[]).append(candidate)
                if source_index==0:
                    own.setdefault(name,[]).append(candidate)
                    if sim_index==default:preferred[name]=candidate
    missing=set(selection)-available.keys()
    if missing:raise ValueError('Selected collider is not registered by this character: '+', '.join(sorted(missing)))
    changed=False
    for sim in cloth.value['simClothDatas']:
        v=sim.value;mapping=v['collidableTransformMap'].value
        landscape=bool(v['landscapeCollisionEnabled'])
        if landscape and len(selection)>31:
            raise ValueError('Native world collision reserves mask bit 31; choose at most 31 body colliders')
        if mapping['transformSetIndex']!=-1 or mapping['transformIndices'] or mapping['offsets']:
            raise ValueError('Selecting internally mapped collision shapes is not supported; native external binding is required')
        old=v['perInstanceCollidables'];old_names=[x.value['name'] for x in old]
        if len(set(old_names))!=len(old_names):raise ValueError('Duplicate native collider names cannot be selected safely')
        if old_names==selection:continue
        changed=True;cols=[];metas=[]
        for name in selection:
            if name in old_names:
                i=old_names.index(name);cols.append(old[i]);metas.append(v['collidablePinchingDatas'][i])
            else:
                # Use this garment's dynamic variant when promoting one of
                # its shapes into another state. Other garments can legitimately
                # carry different geometry under the same external name.
                candidates=[preferred[name]] if name in preferred else own.get(name,available[name])
                # Different bind transforms/geometry under one external name
                # are not interchangeable. Reject ambiguous additions.
                fingerprints={repr((_named(c,g.tag),_named(m,g.tag))) for c,m,g in candidates}
                if len(fingerprints)!=1:raise ValueError('Native collider has ambiguous state variants: '+name)
                col,meta,g=candidates[0]
                cols.append(graph.rebase(copy.deepcopy(col),g));metas.append(graph.rebase(copy.deepcopy(meta),g))
        masks=v['staticCollisionMasks'];particles=v['particleDatas']
        if len(masks)!=len(particles):raise ValueError('Native particle/collision-mask counts differ')
        new_masks=[]
        for mask,particle in zip(masks,particles):
            # Havok's landscape selection uses the high mask bit. Remapping
            # body colliders must not silently remove existing world collisions.
            new=mask&0x80000000 if landscape else 0
            for i,name in enumerate(selection):
                if name in old_names:
                    enabled=bool(mask&(1<<old_names.index(name)))
                else:enabled=particle.value['invMass']>0
                if enabled:new|=1<<i
            new_masks.append(new)
        v.update(perInstanceCollidables=cols,collidablePinchingDatas=metas,staticCollisionMasks=new_masks)
    if not changed:return payload
    for op in cloth.value['operators']:
        if tag.types[op.type].name!='hclSimulateOperator':continue
        for cfg in op.value['simulateOpConfigs']:
            if cfg.value['instanceCollidablesUsed']:
                raise ValueError('Explicit native collider subsets require a verified remapper')
            cfg.value['useAllInstanceCollidables']=int(bool(selection))
    encoded=Writer(graph).pack(cloth);label=payload[20:20+struct.unpack_from('<I',payload,16)[0]]
    result=payload[:16]+struct.pack('<I',len(label))+label+struct.pack('<I',len(encoded))+encoded+payload[tag.root[2]:]
    check=resource_tag(result);actual=Graph(check).read(next(check.objects('hclClothData')))
    validate_collider_metadata(actual)
    if _named(actual,check)!=_named(cloth,tag):raise ValueError('Native collider selection round-trip failed')
    return result


def blend_fields(payload):
    tag=resource_tag(payload);result={}
    for op in tag.objects('hclBlendSomeVerticesOperator'):
        identity=tag.value(tag.field(op,'operatorID'));buffer=tag.value(tag.field(op,'bufferIdx_C'))
        for i,e in enumerate(tag.array(tag.field(op,'blendEntries'))):
            key=identity,buffer,i,tag.value(tag.field(e,'vertexIndex'))
            if key in result:raise ValueError('Ambiguous native cloth blend identity')
            result[key]=tag.field(e,'blendWeight')[1]
    return result


def apply(index,requests,payloads,resources,reports):
    planned={}
    for binding,selection,record in requests:
        if selection is None:continue
        if binding.resource in planned and planned[binding.resource][1]!=selection:
            raise ValueError('Meshes sharing one native cloth resource have different collider selections')
        planned[binding.resource]=binding,selection,record
    for ri,(binding,selection,record) in planned.items():
        rec=index.package.container.records[ri];target=resources if rec.external else payloads
        raw=target.get(ri,bytes(binding.tag.raw))
        result=select(raw,selection,[raw for asset,name,raw in sources(index.package,record)])
        old_fields=blend_fields(raw);new_fields=blend_fields(result)
        if old_fields.keys()!=new_fields.keys():raise ValueError('Collider selection changed cloth blends')
        offsets={old_fields[k]:new_fields[k] for k in old_fields}
        for report in reports:
            if report['resource_id']!=binding.asset:continue
            report['baseline']=[(offsets[at],value) for at,value in report['baseline']]
            report['collider_selection']=list(selection)
        if result!=raw:target[ri]=result
