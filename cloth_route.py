"""Attach a new Havok simulation to an existing, verified character family."""
import copy
import hashlib
import math
import struct
from .native import Record,VB,Submesh,pack_attribute,u32
from .runtime_mesh import catalogs,_ordinary_surface_matches,_clone_draw
from .cloth_native import resource_tag
from .cloth_build import build,cross,geometry
from .skeleton import decode_inverse


def family(package,record):
    matches=[c for c in catalogs(package) if record in c.lods and c.cloth is not None]
    if len(matches)!=1:raise ValueError('New game cloth requires a target character with one existing native cloth companion')
    return matches[0]


def inverse_binds(package,catalog,bones):
    rec=package.container.records[catalog.record];raw=rec.payload;at=raw.find(b'SKELETON')
    if at<0 or raw.find(b'SKELETON',at+8)>=0 or u32(raw,at+8)!=15 or u32(raw,at+16)!=len(bones)-1:
        raise ValueError('New cloth requires the character catalog\'s matching SKELETON v15 inverse binds')
    return [[matrix[r][c] for c in range(4) for r in range(4)]
            for matrix in (decode_inverse(raw,at+20+64*i) for i in range(len(bones)-1))]


def donor(package,catalog,old,collisions=False,asset=None):
    md=package.mesh(catalog.cloth);selector=old.descriptor[5]&63
    candidates=[s for s in md.groups[0].meshes if s.material==old.material and s.descriptor[5]==selector]
    source=candidates[0] if len(candidates)==1 else next((s for s in md.groups[0].meshes if md.vbs[s.vb].flag==1),None)
    if source is None:raise ValueError('No verified cloth output declaration in target character')
    references=md.references
    if asset is not None:
        matches=[]
        for candidate in md.groups[0].meshes:
            index=candidate.descriptor[5]-1
            if not 0<=index<(u32(references,0)&255) or md.vbs[candidate.vb].flag!=1:continue
            kind,identity=struct.unpack_from('<II',references,4+8*index)
            if kind==2150 and identity==asset:matches.append(candidate)
        if not matches:raise ValueError('Selected physics donor is not registered by this character cloth companion')
        source=matches[0]
    def resource(sub):
        index=sub.descriptor[5]-1
        if not 0<=index<(u32(references,0)&255):raise ValueError('Invalid donor cloth selector')
        kind,asset=struct.unpack_from('<II',references,4+8*index)
        recs=[r for r in package.container.records if r.kind==kind and r.asset==asset]
        if len(recs)!=1 or kind!=2150:raise ValueError('Missing native cloth donor')
        rec=recs[0]
        raw=package.members[package.record_members[rec.index]].unpacked if rec.external else rec.payload
        from .cloth_integrity import repair_authored_cloth
        raw, _ = repair_authored_cloth(raw, package, rec)
        tag=resource_tag(raw)
        root=next(tag.objects('hclClothData'))
        refs=tag.array(tag.field(root,'simClothDatas'))
        sim=tag.array(refs[0])[0]
        count=len(tag.array(tag.field(sim,'perInstanceCollidables')))
        name=tag.value(tag.field(root,'name'))
        return raw,count,name
    raw,count,name=resource(source)
    if collisions and not count and name.startswith('DBH_CLOTH_'):
        # v0.9.3-v0.9.6 authored graphs stripped colliders. Their original
        # input draws/resources remain in group zero. Recover only a unique
        # same-material donor, never a guessed unrelated collision map.
        alternatives={}
        for candidate in md.groups[0].meshes:
            if candidate.material!=old.material or md.vbs[candidate.vb].flag!=1:continue
            value,n,label=resource(candidate)
            if n and not label.startswith('DBH_CLOTH_'): alternatives[value]=candidate
        if len(alternatives)!=1:
            raise ValueError('Cannot uniquely restore stripped cloth colliders; reimport the original model package and export its edited cloth again')
        raw,source=next(iter(alternatives.items()))
    return md,source,raw


def append_output(target,donor_md,donor_sub,draw,vertices,faces,selector):
    source=donor_md.vbs[donor_sub.vb]
    if source.flag!=1 or source.strides[1]!=48 or any(a[0]>4 for a in source.attributes()):
        raise ValueError('Only verified 48-byte native cloth outputs are supported')
    if len(target.vbs)>=256:raise ValueError('Too many cloth vertex buffers')
    streams=[bytearray() for _ in range(4)]
    for v in vertices:
        template=min(donor_sub.first_vertex+donor_sub.count-1,max(donor_sub.first_vertex,v.get('template_index',donor_sub.first_vertex)))
        records=[bytearray(s[template*n:(template+1)*n]) for s,n in zip(source.streams,source.strides)]
        uv=list(v.get('uvs',())) or [(0,0)];uv+=[uv[0]]*4
        # Native auxiliary channels remain donor values; they are not skin weights.
        attrs={2:v.get('color',(1,1,1,1)),4:(*uv[0],*uv[1]),5:(*uv[2],*uv[3])}
        for stream,offset,fmt,semantic in source.attributes():
            if stream!=0 or semantic not in attrs:continue
            encoded=pack_attribute(fmt,attrs[semantic]);records[0][offset:offset+len(encoded)]=encoded
        tangent=v.get('tangent',(1,0,0,1));bitangent=tuple(x*tangent[3] for x in cross(v['normal'],tangent[:3]))
        records[1]=bytearray(struct.pack('<12f',*v['position'],*v['normal'],*tangent[:3],*bitangent))
        for out,record in zip(streams,records):out.extend(record)
    vi=len(target.vbs);target.vbs.append(VB(len(vertices),1,source.layout,source.strides,streams,{1:bytes(streams[1])}));target.padding.append(b'')
    ib=donor_sub.ib;first=len(target.indices[ib])//2
    for face in faces:target.indices[ib].extend(struct.pack('<3H',*reversed(face)))
    desc=bytearray(donor_sub.descriptor[:donor_sub.detail_offset]);desc+=draw.detail_table
    desc[5]=selector
    for offset,value in ((6,vi),(10,0),(14,len(vertices)),(18,ib),(22,first),(26,len(faces)*3)):
        struct.pack_into('<I',desc,offset,value)
    struct.pack_into('<II',desc,38+u32(desc,34)*4,*draw.material)
    size=sum(len(data) for vb in target.vbs for data in vb.inline_streams.values())
    # Deliberate reallocation of the previously opaque reservation. Keep its
    # bytes as a prefix; use exact new size so native allocation bounds agree.
    target.inline_reservation=(getattr(target,'inline_reservation',b'')+bytes(size))[:size]
    target.inline_reservation_size=size
    radius=max(math.sqrt(sum(x*x for x in v['position'])) for v in vertices)+1
    suffix=donor_sub.suffix[:-16]+struct.pack('<4f',0,0,0,radius)
    return Submesh(bytes(desc),suffix)


def compile_cloth(package,catalog,payload,vertices,faces,inverses,name,settings):
    """Use same-family registered types while preserving the physics donor."""
    if settings.get('world_collision') is False:
        from .cloth_world import patch
        payload=patch(payload,False)
    if settings.get('colliders') is not None:
        from .cloth_colliders import select,sources
        payload=select(payload,settings['colliders'],[raw for asset,label,raw in sources(package,catalog.lods[0])])
    _,_,seams=geometry(vertices,faces)
    needed=('hclObjectSpaceMeshMeshDeformPNOperator','hclObjectSpaceDeformer::LocalBlockPN')
    def capable(raw):
        tag=resource_tag(raw)
        return all(sum(t.name==n for t in tag.types)==1 for n in needed)
    from .cloth_profiles import build_profiles
    compiler=build_profiles if settings.get('inherit_source') else build
    if not seams or capable(payload):return compiler(payload,vertices,faces,inverses,name,settings)
    refs=package.mesh(catalog.cloth).references
    lookup={(r.kind,r.asset):r for r in package.container.records}
    attempted=set();errors=[]
    for i in range(u32(refs,0)&255):
        kind,asset=struct.unpack_from('<II',refs,4+8*i)
        if kind!=2150 or asset in attempted:continue
        attempted.add(asset);rec=lookup.get((kind,asset))
        if rec is None:continue
        raw=package.members[package.record_members[rec.index]].unpacked if rec.external else rec.payload
        if raw==payload or not capable(raw):continue
        try:
            return compiler(payload,vertices,faces,inverses,name,settings,type_payload=raw)
        except ValueError as error:
            if not str(error).startswith(('Havok type ','Incompatible native Havok')):raise
            errors.append(str(error))
    detail=errors[-1] if errors else 'No native PN deformation type table in this character family'
    raise ValueError('Cannot compile welded cloth with compatible same-character Havok types: '+detail)


def route(package,meshes,requests,metadata):
    """requests hold original catalog/draw identities, extracted mesh and fallback."""
    reports=[];prepared=set();touched=set()
    for request in requests:
        ri,gi,old,draw,vertices,faces,settings,original=request
        before=family(package,ri)
        key=before.record,old.material,old.descriptor[5]
        if key in touched:raise ValueError('Export only one cloth topology per original surface, not separate edited LODs')
        touched.add(key)
        donor_md,donor_sub,payload=donor(package,original,old,collisions=settings.get('collisions',True),asset=settings.get('donor_asset'))
        cloth=meshes.setdefault(before.cloth,package.mesh(before.cloth))
        refs=cloth.references;flags=u32(refs,0);count=flags&255
        if flags&256 or count>=32:raise ValueError('Cloth reference table has no free verified selector')
        selector=count+1
        name=f'DBH_CLOTH_{package.container.records[ri].asset:X}_{len(reports)}_{draw.material[1]:X}'
        try:
            raw,report=compile_cloth(package,original,payload,vertices,faces,inverse_binds(package,before,metadata['bones']),name,settings)
        except ValueError as error:
            source_draws=package.mesh(ri).flat()
            mi=source_draws.index(old)
            label=metadata.get('slot_objects',{}).get(f'{ri}:{mi}',{}).get('name',f'DBH_{ri}_{mi:03d}')
            raise ValueError(f'{label}: {error}') from error
        assets={r.asset for r in package.container.records}
        asset=0x60000000|(int.from_bytes(hashlib.sha256(name.encode()+raw).digest()[:4],'little')&0xfffffff)
        while asset in assets:asset=0x60000000|((asset+1)&0xfffffff)
        record=Record(len(package.container.records),2150,asset,struct.pack('<IBII',len(raw),0,0,0),raw)
        package.container.records.append(record)
        cloth.references=struct.pack('<I',flags+1)+refs[4:4+8*count]+struct.pack('<II',2150,asset)+refs[4+8*count:]
        if before.record not in prepared:
            if before.group not in (0,1):raise ValueError('Unsupported cloth render group')
            for target_ri in (*before.lods,before.cloth):
                target=meshes.setdefault(target_ri,package.mesh(target_ri))
                if before.group==0:
                    if len(target.groups)!=1:raise ValueError('Unknown native cloth group layout')
                    target.groups.append(copy.deepcopy(target.groups[0]))
                    if target_ri!=before.cloth:
                        target.groups[0]=copy.deepcopy(package.mesh(target_ri).groups[0])
                elif len(target.groups)!=2:raise ValueError('Unknown previously routed cloth group layout')
            catalog=package.container.records[before.record];data=bytearray(catalog.payload)
            struct.pack_into('<I',data,before.group_offset,1);catalog.payload=bytes(data)
            prepared.add(before.record)
        source=meshes[ri]
        for target_ri in before.lods:
            target=meshes[target_ri];group=target.groups[1]
            candidates=_ordinary_surface_matches(package,ri,gi,old,target_ri,old.descriptor[5])
            for slot,previous in candidates:
                fallback=_clone_draw(target,source,draw)
                desc=bytearray(fallback.descriptor);desc[5]=0x40|selector
                group.meshes[slot]=Submesh(bytes(desc),fallback.suffix)
            if candidates:group.suffix=source.groups[1].suffix
        render=cloth.groups[1]
        for i,s in enumerate(render.meshes):
            if s.material==old.material and (0x40|s.descriptor[5])==old.descriptor[5]:
                desc=bytearray(s.descriptor);struct.pack_into('<I',desc,26,0)
                render.meshes[i]=Submesh(bytes(desc),s.suffix)
        if len(render.meshes)!=len(cloth.groups[0].meshes):raise ValueError('Cloth simulation/render draw indexes disagree')
        output=append_output(cloth,donor_md,donor_sub,draw,vertices,faces,selector)
        cloth.groups[0].meshes.append(copy.deepcopy(output));render.meshes.append(output)
        radius=struct.unpack('<4f',output.suffix[-16:])[3]
        for group in cloth.groups:
            previous=struct.unpack('<4f',group.suffix[-16:])
            safe=max(radius,math.sqrt(sum(x*x for x in previous[:3]))+abs(previous[3]))
            group.suffix=group.suffix[:-16]+struct.pack('<4f',0,0,0,safe)
        report.update(record=ri,mesh=0,
            resource_record=record.index,resource_id=asset,buffer=0,name=name,cloth_record=before.cloth,cloth_draw=len(render.meshes)-1,
            selector=selector,render_group=1,new_topology=True,weights=[v['cloth_weight'] for v in vertices],
            catalog=before.record,lods=list(before.lods),simulation_inputs_preserved=True)
        if settings.get('colliders') is not None:report['collider_selection']=list(settings['colliders'])
        if settings.get('world_collision') is not None:report['world_collision']=settings['world_collision']
        # Find the surviving active source slot, not the last visited LOD.
        slots=_ordinary_surface_matches(package,ri,gi,old,ri,old.descriptor[5])
        report['mesh']=sum(len(g.meshes) for g in source.groups[:1])+slots[0][0]
        reports.append(report)
    return reports
