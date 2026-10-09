"""Blender-facing replacement workflow; binary packing is in native.py."""
import hashlib
import json
import math
import re
import struct
import tempfile
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

from .native import Package, decode_vertices, decode_faces, append_mesh, split_mesh
from .legacy import _create_armature, _to_blender, _from_blender, _resolve_segs
from .xps import Bone
from .textures import decode_qd, decode_dds, png_bytes
from .metadata_text import write_metadata


def digest(data):
    return hashlib.sha256(data).hexdigest()


def signature(obj, local=False):
    """Detect edits without rewriting untouched game data / morphs."""
    mesh = obj.data
    from .cloth_native import GROUP, PIN_GROUP
    simulation_groups = {g.index for g in obj.vertex_groups if g.name in (GROUP, PIN_GROUP)}
    values = [tuple(tuple(row) for row in (obj.matrix_basis if local else obj.matrix_world)),
              [(tuple(v.co), tuple((g.group, g.weight) for g in v.groups if g.group not in simulation_groups)) for v in mesh.vertices],
              [(tuple(p.vertices), p.use_smooth) for p in mesh.polygons],
              [[tuple(uv.uv) for uv in layer.data] for layer in mesh.uv_layers],
              [tuple(n.vector) for n in mesh.corner_normals],
              [(g.index, g.name) for g in obj.vertex_groups if g.index not in simulation_groups],
              [(a.name, a.domain, [tuple(c.color) for c in a.data]) for a in mesh.color_attributes]]
    return digest(repr(values).encode())


def material_refs(package, material_id):
    record = next((r for r in package.container.records if r.kind == 0x855 and r.asset == material_id), None)
    if record is None: return []
    from .materials import bindings
    return list(dict.fromkeys(b.texture for b in bindings(record.payload) if b.texture))


def make_material(package, material_id, folders, cache):
    if material_id in cache: return cache[material_id]
    material = bpy.data.materials.new(f'DBH_m_{material_id:X}')
    material.use_nodes = True
    refs = material_refs(package, material_id)
    material['dbh_material_id'] = material_id
    material['dbh_texture_ids'] = json.dumps(refs)
    # Never mistake the first (usually normal/mask) texture for base colour.
    # Connor's base shirt was visually verified; general materials rank RGB
    # QD resources by resolution. Complex shader composition remains approximate.
    verified = {0x15855:0x3AB4D, 0x14B0D:0x3AB4D, 0x14B0F:0x3AB68, 0x14B1B:0x3AB99}
    if material_id in verified:material['dbh_preferred_diffuse']=f'{verified[material_id]:X}'
    # Self-contained files never use loose images as an initial diffuse source.
    if package.texture_attachments:folders=()
    candidates=[]
    for resource in refs:
        for folder in folders:
            path=Path(folder)/f'{resource:X}'
            if path.is_file():
                with path.open('rb') as stream: head=stream.read(10)
                if len(head)==10:
                    version,fmt,quality,w,h=struct.unpack('<5H',head)
                    if version==16 and fmt in (1,3,7,19):
                        candidates.append((w*h,resource))
                        break
    candidates.sort(reverse=True)
    ids=([verified[material_id]] if verified.get(material_id) in refs else [])+[r for _,r in candidates]
    # Explicit converted resource files are also valid user-selected previews.
    names = [f'm_{material_id:X}', f'{material_id:X}'] + [f'{r:X}' for r in dict.fromkeys(ids)]
    chosen = None
    for name in names:
        for folder in folders:
            for ext in ('.png', '.tga', '.tif', '.tiff', '.dds', '.jpg', '.exr', ''):
                path = Path(folder) / (name + ext)
                if path.is_file():
                    try:
                        load_path=path
                        if ext in ('','.dds'):
                            raw=path.read_bytes()
                            preview=Path(tempfile.gettempdir())/'dbh_preview_v3'
                            preview.mkdir(exist_ok=True)
                            load_path=preview/(path.stem+'_'+digest(raw)[:16]+'.png')
                            if not load_path.is_file():
                                dimensions=(decode_dds if raw[:4]==b'DDS ' else decode_qd)(raw)
                                load_path.write_bytes(png_bytes(*dimensions))
                        image = bpy.data.images.load(str(load_path), check_existing=True)
                        if image.size[0] == 0: continue
                        image.pack()
                        chosen = image; break
                    except (RuntimeError,ValueError,struct.error) as error:
                        print(f'DBH texture {path.name}: {error}',flush=True)
                        continue
            if chosen: break
        if chosen: break
    if chosen:
        node = material.node_tree.nodes.new('ShaderNodeTexImage')
        node.name = 'DBH Diffuse'; node.label = 'Diffuse preview (game shader approximation)'
        node.image = chosen
        uv_node=material.node_tree.nodes.new('ShaderNodeUVMap')
        uv_node.uv_map='UV1'
        material.node_tree.links.new(uv_node.outputs['UV'],node.inputs['Vector'])
        material.node_tree.links.new(node.outputs['Color'], material.node_tree.nodes.get('Principled BSDF').inputs['Base Color'])
        material['dbh_diffuse_status'] = 'LOADED'
        material['dbh_diffuse_source'] = name
    else:
        material['dbh_diffuse_status'] = 'MISSING_OR_PROPRIETARY' if refs else 'NO_REFERENCE'
    cache[material_id] = material
    return material


def reference_metadata(path, package, prefs):
    sidecar = path.with_suffix('.dbh.json')
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text(encoding='utf-8'))
            if isinstance(meta, dict) and meta.get('package_sha256') == digest(package.source_data):
                return meta
        except (ValueError, UnicodeError):
            pass
        # Older releases wrote companions. A stale JSON beside a newly
        # exported SEGS must not prevent native, sidecar-free reimport.
    from .native_metadata import metadata
    return metadata(path, package)


def import_file(context, path, prefs):
    path = _resolve_segs(path).resolve()
    package = Package(path.read_bytes())
    metadata = reference_metadata(path, package, prefs)
    folders = list(metadata.get('texture_folders', []))
    if prefs.texture_folder:
        folders.insert(0, bpy.path.abspath(prefs.texture_folder))
    metadata['texture_folders'] = folders
    from .game_paths import resolve_index
    index = resolve_index(context, metadata, preferences=prefs)
    metadata['game_index'] = str(index) if index is not None else ''
    metadata['texture_override_folder']=bpy.path.abspath(prefs.texture_folder) if prefs.texture_folder else ''
    return import_native(context, path, package, metadata)


def import_native(context, path, package, metadata):
    from .material_ui import TextureLoader
    from .game_paths import resolve_index
    index = resolve_index(context, metadata)
    metadata['game_index'] = str(index) if index is not None else ''
    loader=TextureLoader(package,metadata.get('texture_folders',[]),index,
                         metadata.get('texture_override_folder',''),defer_hashes=True)
    try:return _import_native(context,path,package,metadata,loader)
    finally:loader.close()


def _import_native(context, path, package, metadata, texture_loader):
    started = time.perf_counter()
    stage_start = started
    timings = {}
    def mark(stage):
        nonlocal stage_start
        now = time.perf_counter()
        timings[stage] = timings.get(stage, 0.) + now - stage_start
        stage_start = now
    if context.object and context.object.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    for obj in context.selected_objects: obj.select_set(False)
    from .bone_names import corrected_metadata
    metadata = corrected_metadata(metadata)
    bones = tuple(Bone(**bone) for bone in metadata.get('bones', []))
    armature, bone_names = _create_armature(context, f"DBH_{metadata['code']}", bones)
    y_offset = metadata.get('y_offset', 0.0)
    metadata = dict(metadata, package_sha256=digest(package.source_data),
                    source_segs=str(path), armature_name=armature.name if armature else '', slot_objects={})
    from .scene_organization import create_collection,record_collection,collision_display,wire_colors
    import_collection = create_collection(context,metadata)
    if armature:
        import_collection.objects.link(armature)
        for previous in list(armature.users_collection):
            if previous != import_collection: previous.objects.unlink(armature)
    text = bpy.data.texts.new('DBH Package Metadata')
    write_metadata(text, metadata)
    created, materials = [], {}
    from .cloth_native import ClothIndex
    from .cloth_blender import import_mask
    cloth_index = ClothIndex(package)
    from .auxiliary_mesh import bindings as auxiliary_bindings, setup as setup_auxiliary
    auxiliary = auxiliary_bindings(package, metadata)
    from .auxiliary_mesh import collision_slots
    collision_keys = collision_slots(package,metadata)
    from .material_ui import populate
    mark('setup')
    for record in package.container.records:
        if metadata.get('mesh_records') and record.index not in metadata['mesh_records']: continue
        if record.payload[4:12] != b'MESHDATA' or record.index not in package.record_members: continue
        md = package.mesh(record.index)
        from .runtime_mesh import active_draw_indices
        visible_indices = active_draw_indices(package, record.index)
        textures=[]
        for material_id in dict.fromkeys(sub.material[1] for mi,sub in enumerate(md.flat()) if mi in visible_indices and sub.index_count):
            textures.extend(material_refs(package,material_id))
        texture_loader.prefetch(dict.fromkeys(textures))
        for mi, sub in enumerate(md.flat()):
            if mi not in visible_indices or not sub.index_count:
                continue
            vertices = decode_vertices(md, sub)
            # XPS/native codec faces are clockwise; Blender requires CCW.
            faces = [tuple(reversed(face)) for face in decode_faces(md, sub)]
            mark('decode')
            name = f'DBH_{record.index}_{mi:03d}'
            native_attachment = auxiliary.get(f'{record.index}:{mi}')
            mesh = bpy.data.meshes.new(name)
            positions = [_to_blender((v['position'][0], v['position'][1] + (0 if native_attachment else y_offset), v['position'][2])) for v in vertices]
            mesh.from_pydata(positions, [], faces)
            mesh.validate(clean_customdata=False)
            mesh.update()
            mesh.polygons.foreach_set('use_smooth', np.ones(len(mesh.polygons), dtype=np.bool_))
            mark('mesh')
            normals = [Vector(_to_blender(v['normal'])).normalized() for v in vertices]
            if vertices and mesh.polygons and any(a[3] == 1 for a in md.vbs[sub.vb].attributes()):
                mesh.normals_split_custom_set_from_vertices(normals)
            mark('normals')
            loop_vertices = np.empty(len(mesh.loops), dtype=np.int32)
            mesh.loops.foreach_get('vertex_index', loop_vertices)
            for ui in range(max((len(v['uvs']) for v in vertices), default=0)):
                uv = mesh.uv_layers.new(name=f'UV{ui + 1}')
                values = np.asarray([vertex['uvs'][ui] for vertex in vertices], dtype=np.float32).reshape(-1, 2)
                # Flip in double precision before converting, matching the
                # previous per-corner RNA write exactly (including tiny UVs).
                values[:, 1] = np.asarray([1 - vertex['uvs'][ui][1] for vertex in vertices], dtype=np.float32)
                uv.data.foreach_set('uv', values[loop_vertices].ravel())
            if mesh.uv_layers:
                mesh.uv_layers.active_index=0
                mesh.uv_layers[0].active_render=True
            mark('uvs')
            color = mesh.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='POINT')
            color.data.foreach_set('color', np.asarray([v['color'] for v in vertices], dtype=np.float32).ravel())
            mark('colors')
            obj = bpy.data.objects.new(name, mesh)
            record_collection(import_collection,metadata,record.index).objects.link(obj)
            collision = f'{record.index}:{mi}' in collision_keys
            if collision: collision_display(obj)
            elif sub.material == (0,0):
                # Preserve legacy helper display without making an unclassified
                # material-less surface a physics obstacle (or invisible cloth).
                obj.display_type = 'WIRE'; obj.hide_render = True
            groups={}
            for i, vertex in enumerate(vertices):
                influences={}
                for bi,weight in zip(vertex['bones'],vertex['weights']):
                    if 0<=bi<len(bone_names) and weight>0: influences[bi]=influences.get(bi,0)+weight
                for bi,weight in influences.items():
                    # Keep both group-index order and each vertex's influence
                    # insertion order compatible with old scene signatures.
                    if bi not in groups: groups[bi]=obj.vertex_groups.new(name=bone_names[bi])
                    groups[bi].add([i],weight,'REPLACE')
            if armature:
                modifier = obj.modifiers.new('Detroit Skin', 'ARMATURE'); modifier.object = armature
                obj.parent = armature
                if native_attachment: setup_auxiliary(obj, armature, metadata, package, native_attachment)
            mark('skinning')
            cloth_status = import_mask(obj, package, metadata, record.index, mi, cloth_index)
            mark('cloth')
            material=make_material(package, sub.material[1], metadata.get('texture_folders', []), materials)
            if hasattr(material,'dbh_texture_slots'):populate(material,package,texture_loader)
            settings=metadata.get('material_previews',{}).get(f'{sub.material[1]:X}')
            if settings and not material.get('dbh_preview_restored'):
                from .material_ui import preview
                for role,setting in settings.items():
                    bi=setting['binding']
                    if 0<=bi<len(material.dbh_texture_slots):
                        slot=material.dbh_texture_slots[bi]
                        if slot.image:
                            slot.role=role;slot.channel=setting['channel'];slot.uv=setting['uv']
                            preview(material,slot)
                material['dbh_preview_restored']=True
            obj.data.materials.append(material)
            mark('materials')
            from .scene_export import slot_key
            metadata['slot_objects'].setdefault(slot_key(record.index,mi),{}).update(name=obj.name,signature=signature(obj,local=bool(native_attachment)),replacement=False)
            metadata['slot_objects'][slot_key(record.index,mi)]['collision_mesh'] = collision
            if native_attachment:
                metadata['slot_objects'][slot_key(record.index,mi)]['native_attachment'] = native_attachment
            if cloth_status:
                metadata['slot_objects'][slot_key(record.index,mi)]['cloth_simulation'] = cloth_status
            created.append(obj)
            mark('signatures')
    texture_loader.finish_hashes()
    mark('texture_hash_finish')
    for obj in created: obj.select_set(True)
    wire_colors()
    metadata['imported_slots']=[[int(part) for part in key.split(':')] for key in metadata['slot_objects']]
    write_metadata(text, metadata)
    mark('metadata')
    if created: context.view_layer.objects.active = created[0]
    missing = sum(m.get('dbh_diffuse_status') == 'MISSING_OR_PROPRIETARY' for m in materials.values())
    loaded = sum(m.get('dbh_diffuse_status') == 'LOADED' for m in materials.values())
    warnings = [f'{loaded} diffuse previews; {missing} materials have no automatic base-colour choice'] if missing else []
    context.scene['dbh_import_texture_report'] = json.dumps({m.name: m.get('dbh_diffuse_status') for m in materials.values()})
    timings['total'] = time.perf_counter() - started
    context.scene['dbh_import_timings'] = json.dumps(timings)
    return created, warnings


def object_armature(obj):
    return next((m.object for m in obj.modifiers if m.type == 'ARMATURE' and m.object), None)


def skin_missing_vertices(context):
    """Weight newly authored vertices from the corresponding native mesh slot."""
    from .scene_export import manifest_for_object,slot_for_object
    obj=context.active_object
    found=manifest_for_object(obj)
    if not obj or obj.type!='MESH' or not found:
        raise ValueError('Select a named Detroit mesh slot')
    _,metadata=found
    if not metadata.get('bones'):
        raise ValueError('This package has no imported game skeleton')
    source=Path(metadata.get('source_segs') or obj['dbh_source_segs'])
    data=source.read_bytes()
    if digest(data)!=metadata.get('package_sha256',obj.get('dbh_source_sha256')):
        raise ValueError('Source package changed; reimport it first')
    package=Package(data)
    ri,mi=slot_for_object(obj,metadata)
    md=package.mesh(ri)
    native=decode_vertices(md,md.flat()[mi])
    if not native:
        raise ValueError('Target slot has no reference vertices for automatic skinning')
    arm=object_armature(obj) or bpy.data.objects.get(metadata.get('armature_name',''))
    if not arm or arm.type!='ARMATURE':
        raise ValueError('Imported game armature is missing')
    names=[bone['name'] for bone in metadata['bones']]
    valid={group.index for group in obj.vertex_groups if group.name in names}
    tree=KDTree(len(native))
    offset=metadata.get('y_offset',obj.get('dbh_y_offset',0.0))
    for index,vertex in enumerate(native):
        x,y,z=vertex['position']
        tree.insert(_to_blender((x,y+offset,z)),index)
    tree.balance()
    to_arm=arm.matrix_world.inverted() @ obj.matrix_world
    changed=0
    for vertex in obj.data.vertices:
        if any(group.group in valid and group.weight>0 for group in vertex.groups):
            continue
        nearest=tree.find(to_arm @ vertex.co)[1]
        influences={}
        for bone,weight in zip(native[nearest]['bones'],native[nearest]['weights']):
            if 0<=bone<len(names) and weight>0:
                influences[bone]=influences.get(bone,0)+weight
        total=sum(influences.values())
        for bone,weight in influences.items():
            group=obj.vertex_groups.get(names[bone]) or obj.vertex_groups.new(name=names[bone])
            group.add([vertex.index],weight/total,'REPLACE')
            valid.add(group.index)
        if total:changed+=1
    if not object_armature(obj):
        modifier=obj.modifiers.new('Detroit Skin','ARMATURE')
        modifier.object=arm
    return changed


def simplify_mesh_names(context):
    """Rename existing imported objects/data without changing their game slots."""
    from .scene_export import export_anchor,manifest_for_object,slot_for_object,slot_key
    anchor=export_anchor(context)
    text,metadata=manifest_for_object(anchor)
    renamed=0
    for obj in context.scene.objects:
        if obj.type!='MESH':continue
        if metadata.get('slot_objects'):
            if obj.name not in {entry['name'] for entry in metadata['slot_objects'].values()}:continue
        elif obj.get('dbh_metadata')!=text.name or obj.get('dbh_superseded'):continue
        ri,mi=slot_for_object(obj,metadata)
        desired=f'DBH_{ri}_{mi:03d}'
        if obj.name==desired:continue
        obj.name=desired
        if obj.data.users==1:obj.data.name=obj.name
        if metadata.get('slot_objects'):
            metadata['slot_objects'][slot_key(ri,mi)]['name']=obj.name
        renamed+=1
    if metadata.get('slot_objects') and renamed:
        write_metadata(text, metadata)
    return renamed


def armature_positions(context, active, objects, metadata):
    """Read saved Edit Mode/rest heads, not an animated/evaluated pose."""
    bones=metadata.get('bones', [])
    if not bones:return None
    arms={object_armature(o) for o in objects if object_armature(o)}
    arms.update(o for o in context.scene.objects if o.type=='ARMATURE' and
                (o.name==metadata.get('armature_name') or
                 ('dbh_metadata' in active and o.get('dbh_metadata')==active['dbh_metadata'])))
    if not arms:return None  # Mesh-only workflows can retain the source rig.
    if len(arms)!=1:raise ValueError('Multiple armatures drive this package; keep one export armature')
    arm=next(iter(arms))
    names=[b['name'] for b in bones]
    if len(arm.data.bones)!=len(names) or set(arm.data.bones.keys())!=set(names):
        raise ValueError('Adding, deleting or renaming native bones is not supported; restore the imported bone names')
    for b in bones:
        actual=arm.data.bones[b['name']]
        expected=names[b['parent']] if b['parent']>=0 else None
        if (actual.parent.name if actual.parent else None)!=expected:
            raise ValueError(f'{actual.name}: changing the native bone hierarchy is not supported')
    return [_from_blender(arm.data.bones[name].head_local) for name in names]


def bind_replacement(context):
    from .scene_export import manifest_for_object, slot_for_object, slot_key
    target = context.active_object
    donors = [o for o in context.selected_objects if o != target and o.type == 'MESH']
    found=manifest_for_object(target)
    if not target or target.type != 'MESH' or not found or len(donors) != 1:
        raise ValueError('Select exactly one custom mesh, then Shift-select an imported target LAST.')
    donor = donors[0]
    if manifest_for_object(donor): raise ValueError('Use an unbound custom mesh, not another imported slot.')
    text,metadata=found
    if 'slot_objects' not in metadata:
        if target.get('dbh_superseded'): raise ValueError('This target already has a replacement.')
        for key in ('source_segs', 'package_code', 'record_index', 'mesh_index', 'y_offset', 'metadata', 'source_sha256'):
            donor['dbh_' + key] = target['dbh_' + key]
        donor['dbh_replacement'] = True
    else:
        ri,mi=slot_for_object(target,metadata)
        entry=metadata['slot_objects'][slot_key(ri,mi)]
        if entry.get('replacement'):raise ValueError('This target already has a replacement.')
    armature = object_armature(target)
    # Populate missing weights immediately, so the replacement can be posed
    # in Blender as well as exported. Existing correctly named groups win.
    names={b['name'] for b in metadata.get('bones',[])}
    source_groups={g.index:g.name for g in target.vertex_groups if g.name in names}
    valid_groups={g.index for g in donor.vertex_groups if g.name in names}
    tree=KDTree(len(target.data.vertices))
    for vertex in target.data.vertices: tree.insert(target.matrix_world @ vertex.co,vertex.index)
    tree.balance()
    for vertex in donor.data.vertices:
        if any(g.group in valid_groups and g.weight>0 for g in vertex.groups): continue
        if not target.data.vertices: continue
        near=tree.find(donor.matrix_world @ vertex.co)[1]
        for influence in target.data.vertices[near].groups:
            if influence.group not in source_groups: continue
            name=source_groups[influence.group]
            group=donor.vertex_groups.get(name) or donor.vertex_groups.new(name=name)
            group.add([vertex.index],influence.weight,'REPLACE')
    if armature and not object_armature(donor):
        modifier = donor.modifiers.new('Detroit Skin', 'ARMATURE'); modifier.object = armature
    if not donor.data.materials and target.data.materials:
        donor.data.materials.append(target.data.materials[0])
    # Keep the original object recoverable; it is excluded, not destroyed.
    if 'slot_objects' in metadata:
        original_name=target.name
        target.name='DBH_SUPERSEDED_'+original_name
        donor.name=original_name
        entry.update(name=donor.name,replacement=True)
        write_metadata(text, metadata)
    else:
        target['dbh_superseded'] = True
    target.hide_set(True); target.hide_render = True
    for obj in context.selected_objects: obj.select_set(False)
    donor.select_set(True); context.view_layer.objects.active = donor


def extract_mesh(context, obj, md, sub, metadata, with_materials=False, package=None, cloth_sources=False):
    """Triangulate evaluated bind-pose geometry and split UV/normal seams."""
    source_vertices = decode_vertices(md, sub)
    tree = KDTree(len(source_vertices))
    for i, vertex in enumerate(source_vertices): tree.insert(vertex['position'], i)
    tree.balance()
    bone_names = {bone['name']: i for i, bone in enumerate(metadata.get('bones', []))}
    bone_map = {g.index: bone_names[g.name] for g in obj.vertex_groups if g.name in bone_names}
    from .bone_attachment import active_attachment
    attachment=active_attachment(obj,metadata)
    from .scene_export import slot_entry
    native_attachment = slot_entry(obj,metadata).get('native_attachment')
    if native_attachment:
        from .auxiliary_mesh import native_bind
        if package is None: package = Package(Path(metadata['source_segs']).read_bytes())
    armature = attachment[0].target if attachment else object_armature(obj)
    rest_mode=armature.data.pose_position if attachment else None
    from .cloth_native import PREVIEW
    armatures = [(m, m.show_viewport) for m in obj.modifiers if m.type == 'ARMATURE' or (m.type == 'CLOTH' and m.name == PREVIEW)]
    evaluated = None
    try:
        if attachment:armature.data.pose_position='REST'
        for modifier, _ in armatures: modifier.show_viewport = False
        context.view_layer.update()
        matrix = (armature.matrix_world.inverted() if armature else Matrix.Identity(4)) @ obj.matrix_world
        if native_attachment:
            matrix = native_bind(package,native_attachment,attachment[1] if attachment else 0).inverted() @ matrix
        determinant=matrix.determinant()
        if abs(determinant) < 1e-12: raise ValueError(f'{obj.name}: zero scale is not exportable')
        tangent_matrix=matrix.to_3x3()
        normal_matrix = matrix.to_3x3().inverted().transposed()
        evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=context.evaluated_depsgraph_get())
        mesh.calc_loop_triangles()
        has_tangents = bool(mesh.uv_layers)
        if has_tangents:
            try: mesh.calc_tangents(uvmap=mesh.uv_layers[0].name)
            except RuntimeError: has_tangents = False
        uv_layers = list(mesh.uv_layers)[:4]
        color = mesh.color_attributes.active_color
        vertices, faces, unique,face_materials = [], [], {},[]
        # Geometry Nodes/Join Geometry can reorder evaluated material slots.
        # Resolve material identity, not its temporary evaluated slot number.
        original_materials=list(obj.data.materials)
        evaluated_materials=[mat.original if mat else None for mat in mesh.materials]
        material_map={i:original_materials.index(mat) for i,mat in enumerate(evaluated_materials) if mat in original_materials}
        # Position, nearest template and skin weights depend on a vertex, not
        # its UV/normal corners. Compute once; keep corner splitting unchanged.
        vertex_values=[None]*len(mesh.vertices)
        cloth_group=obj.vertex_groups.get('CLOTH_SIMULATION')
        cloth_index=cloth_group.index if cloth_group else None
        origin=md.vertex_origin(sub)
        y_offset=float(metadata.get('y_offset',obj.get('dbh_y_offset',0.0)))
        for triangle in mesh.loop_triangles:
            face = []
            # Convert Blender CCW back to the codec's XPS/game convention;
            # a reflected object transform contributes the other reversal.
            loops = tuple(reversed(triangle.loops)) if determinant > 0 else triangle.loops
            for li in loops:
                loop = mesh.loops[li]; vertex = mesh.vertices[loop.vertex_index]
                cached=vertex_values[loop.vertex_index]
                if cached is None:
                    position=list(_from_blender(matrix @ vertex.co))
                    if not native_attachment:position[1]-=y_offset
                    near=tree.find(position)[1] if source_vertices else 0
                    source=source_vertices[near] if source_vertices else {}
                    weights=([(attachment[1],1.0)] if attachment else
                             [(bone_map[g.group],g.weight) for g in vertex.groups if g.group in bone_map and g.weight>0])
                    if not weights:weights=list(zip(source.get('bones',(1,)),source.get('weights',(1.0,))))
                    cloth_weight=next((g.weight for g in vertex.groups if g.group==cloth_index),0.) if cloth_index is not None else None
                    cached=(tuple(position),tuple(b for b,w in weights),tuple(w for b,w in weights),origin+near,cloth_weight)
                    vertex_values[loop.vertex_index]=cached
                position,bones,weights,template_index,cloth_weight=cached
                normal = _from_blender((normal_matrix @ mesh.corner_normals[li].vector).normalized())
                uvs = tuple((float(layer.data[li].uv.x), 1 - float(layer.data[li].uv.y)) for layer in uv_layers)
                rgba = tuple(color.data[loop.vertex_index if color.domain == 'POINT' else li].color) if color else (1, 1, 1, 1)
                tangent = (*_from_blender((tangent_matrix @ loop.tangent).normalized()),
                           -loop.bitangent_sign * (-1 if determinant < 0 else 1)) if has_tangents else (1, 0, 0, 1)
                value = dict(position=tuple(position), normal=normal, uvs=uvs, color=rgba, tangent=tangent,
                             bones=bones, weights=weights,template_index=template_index)
                if cloth_index is not None:value['cloth_weight']=cloth_weight
                if cloth_sources: value['cloth_source_vertex']=vertex.index
                key = repr(value)
                if key not in unique:
                    unique[key] = len(vertices); vertices.append(value)
                face.append(unique[key])
            faces.append(tuple(face))
            if with_materials and triangle.material_index not in material_map:
                raise ValueError(f'{obj.name}: modifier-generated material is missing from the object material slots')
            face_materials.append(material_map.get(triangle.material_index,triangle.material_index))
        return (vertices,faces,face_materials) if with_materials else (vertices,faces)
    finally:
        if evaluated: evaluated.to_mesh_clear()
        for modifier, enabled in armatures: modifier.show_viewport = enabled
        if attachment:armature.data.pose_position=rest_mode
        context.view_layer.update()


def export_file(context, destination, bone_skin_cloth=False, experimental_textures=False, embed_textures=True, export_deletions=True, export_bones=True,single_lod=False,single_mip=False):
    from .export_hashes import ExportHashes
    with ExportHashes() as fingerprints:
        return _export_file(context,destination,bone_skin_cloth,experimental_textures,embed_textures,
                            export_deletions,export_bones,fingerprints,single_lod,single_mip)


def _export_file(context,destination,bone_skin_cloth,experimental_textures,embed_textures,export_deletions,export_bones,fingerprints,single_lod,single_mip):
    from .scene_export import export_anchor,collect_objects,manifest_for_object,slot_for_object,slot_entry
    active = export_anchor(context)
    if context.object.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    metadata_text,metadata=manifest_for_object(active)
    # Preferences may have changed since import, or this blend came from a
    # different machine/edition. Resolve once for BOTH donors and attachments.
    from .game_paths import resolve_index
    selected_game_index = resolve_index(context, metadata)
    metadata['game_index'] = str(selected_game_index) if selected_game_index is not None else ''
    source = Path(metadata.get('source_segs') or active['dbh_source_segs'])
    destination = destination.with_suffix('.segs').resolve()
    if destination == source.resolve(): raise ValueError('Export to a NEW filename; the source package is never overwritten.')
    data = source.read_bytes()
    if digest(data) != metadata.get('package_sha256',active.get('dbh_source_sha256')): raise ValueError('Source package changed since import. Reimport it first.')
    package = Package(data)
    objects,missing,recovered = collect_objects(context,active,package,metadata)
    from .preset_shader import PRESETS
    materials={mat.as_pointer():mat for obj in objects for mat in obj.data.materials if mat}
    fingerprints.start(slot.image for mat in materials.values()
                       if mat.dbh_shader_mode not in PRESETS or mat.dbh_shader_mode=='HAIR'
                       for slot in mat.dbh_texture_slots if slot.image is not None and slot.image==slot.original_image)
    from .auxiliary_mesh import export_binding,attachment_needs_bake
    attachment_choices = {}
    attachment_bakes = set()
    for obj in objects:
        binding = slot_entry(obj,metadata).get('native_attachment')
        if not binding: continue
        if attachment_needs_bake(obj,metadata,package,binding): attachment_bakes.add(obj.name)
        payload, bone = export_binding(obj,metadata,package,binding)
        node = binding['node']
        if node in attachment_choices and attachment_choices[node] != payload:
            raise ValueError('Meshes sharing one native NODE have conflicting attachments')
        attachment_choices[node] = payload
    attachment_payloads = {node:payload for node,payload in attachment_choices.items()
                           if payload != package.container.records[node].payload}
    for ri,payload in attachment_payloads.items(): package.container.records[ri].payload = payload
    fingerprints.pump()
    from .cloth_blender import export_masks
    from .cloth_native import PREVIEW
    cloth_payloads, cloth_resources, cloth_report, new_cloth = export_masks(context, package, objects, metadata)
    for ri, payload in cloth_payloads.items(): package.container.records[ri].payload = payload
    fingerprints.pump()
    bone_payloads,bone_report={},[]
    if export_bones:
        positions=armature_positions(context,active,objects,metadata)
        if positions is not None:
            from .skeleton import plan_positions
            bone_payloads,bone_report=plan_positions(package,metadata['bones'],positions,metadata.get('y_offset',0.))
            for ri,payload in bone_payloads.items():package.container.records[ri].payload=payload
    from .preset_export import PresetExport
    presets=PresetExport(package,metadata,experimental_textures,fingerprints,single_mip)
    material_plans={}
    for obj in objects:
        ri,mi=slot_for_object(obj,metadata)
        sub=package.mesh(ri).flat()[mi]
        plan={}
        for index in {p.material_index for p in obj.data.polygons}:
            mat=obj.data.materials[index] if index<len(obj.data.materials) else None
            # Original imported objects always have a game material. An empty
            # slot on a surviving face must not silently keep the vanilla one.
            plan[index]=presets.material(mat,(ri,mi,index))
        material_plans[obj.name]=plan
        fingerprints.pump()
    if missing and not export_deletions:
        raise ValueError(f'{len(missing)} imported meshes were deleted; enable Export Deleted Meshes or restore them')
    seen, meshes, extras = set(), {}, {}
    runtime_edits = []
    cloth_requests=[]
    changed = triangles = 0
    deleted=[]
    from .native import Submesh
    for ri,mi in sorted(missing):
        md=meshes.setdefault(ri,package.mesh(ri)) if ri not in meshes else meshes[ri]
        old=md.flat()[mi]
        for gi,group in enumerate(md.groups):
            for si,sub in enumerate(group.meshes):
                if sub is old:
                    desc=bytearray(old.descriptor);struct.pack_into('<I',desc,26,0)
                    group.meshes[si]=Submesh(bytes(desc),old.suffix)
                    runtime_edits.append((ri,gi,old,[]));deleted.append(dict(record=ri,mesh=mi,material=old.material[1]))
    for obj in objects:
        fingerprints.pump()
        ri, mi = slot_for_object(obj,metadata)
        if (ri, mi) in seen: raise ValueError(f'Duplicate slot {ri}/{mi}; unbind/delete the duplicate object.')
        seen.add((ri, mi))
        from .bone_attachment import active_attachment
        native_attachment = slot_entry(obj,metadata).get('native_attachment')
        from .scene_organization import preview_modifier
        modified = (obj.name in attachment_bakes or
                    any(m.show_viewport and m.type != 'ARMATURE' and not preview_modifier(m) for m in obj.modifiers) or
                    (not native_attachment and active_attachment(obj,metadata) is not None))
        original=package.mesh(ri).flat()[mi]
        plan=material_plans[obj.name]
        reassigned=len(plan)>1 or any(asset!=original.material[1] for asset,_ in plan.values())
        original_signature=slot_entry(obj,metadata).get('signature',obj.get('dbh_signature'))
        replaced=slot_entry(obj,metadata).get('replacement',obj.get('dbh_replacement',False))
        if (ri,mi) not in new_cloth and not reassigned and not replaced and not modified and signature(obj,local=bool(native_attachment)) == original_signature: continue
        if any((r['record'], r['mesh']) == (ri, mi) for r in cloth_report):
            # Material/geometry edits need a new simulation instead of a native mask.
            from .cloth_route import family
            if not context.scene.dbh_cloth_new_topology:
                raise ValueError(f'{obj.name}: material/geometry changes require Experimental New Cloth Topology')
            new_cloth[(ri,mi)]=family(package,ri)
            cloth_report=[r for r in cloth_report if (r['record'],r['mesh'])!=(ri,mi)]
        md = meshes.setdefault(ri, package.mesh(ri)) if ri not in meshes else meshes[ri]
        sub = md.flat()[mi]
        if slot_entry(obj,metadata).get('collision_mesh'):
            if reassigned or (ri,mi) in new_cloth:
                raise ValueError(f'{obj.name}: collision helpers retain their native material and cannot be cloth surfaces')
            from .collision_edit import extract_positions,replace_positions
            positions=extract_positions(context,obj,md,sub,metadata,package)
            replace_positions(md,sub,positions)
            for group in md.groups:
                if any(s is sub for s in group.meshes):
                    old=struct.unpack('<4f',group.suffix[-16:])
                    radius=max(math.sqrt(sum(x*x for x in old[:3]))+abs(old[3]),
                               max(math.sqrt(sum(x*x for x in p)) for p in positions)+1.)
                    group.suffix=group.suffix[:-16]+struct.pack('<4f',0,0,0,radius)
            changed+=1;triangles+=sub.index_count//3
            continue
        if md.vertex_origin(sub) != sub.first_vertex or (md.vbs[sub.vb].attributes() == [(1,0,2,0)] and getattr(md,'vertex_origins',{})):
            raise ValueError(f'{obj.name}: implicit collision geometry can be inspected/attached, but topology export is not yet verified; keep its vertices unchanged')
        vertices, faces,face_materials = extract_mesh(context, obj, md, sub, metadata,with_materials=True,package=package)
        md.preserve_deformations=False
        if not sub.count and faces: raise ValueError('Cannot infer vertex schema defaults from an empty target slot.')
        if (ri,mi) in new_cloth and (not faces or len(set(face_materials))!=1 or len(vertices)>65535):
            raise ValueError(f'{obj.name}: new cloth needs one material, triangles, and at most 65535 corner-split vertices; split large surfaces explicitly')
        if not faces:
            desc=bytearray(sub.descriptor);struct.pack_into('<I',desc,26,0)
            empty=Submesh(bytes(desc),sub.suffix);draws=[]
            deleted.append(dict(record=ri,mesh=mi,material=sub.material[1]))
        else:
            draws=[]
            for material_index in dict.fromkeys(face_materials):
                if material_index not in plan:
                    mat=obj.data.materials[material_index] if material_index<len(obj.data.materials) else None
                    plan[material_index]=presets.material(mat,(ri,mi,material_index))
                asset,carrier=plan[material_index]
                if native_attachment and carrier is not None:
                    raise ValueError(f'{obj.name}: new skinned shader carriers cannot be combined with a static native NODE attachment')
                selected=[f for f,i in zip(faces,face_materials) if i==material_index]
                draws.extend(presets.append(md,sub,v,f,asset,carrier) for v,f in split_mesh(vertices,selected))
        for gi, group in enumerate(md.groups):
            for si, candidate in enumerate(group.meshes):
                if candidate is sub:
                    if (ri,mi) in new_cloth:
                        from .cloth_settings import mesh_settings
                        cloth_requests.append((ri,gi,sub,draws[0],vertices,faces,mesh_settings(context.scene,obj,metadata,package),new_cloth[(ri,mi)]))
                    else:runtime_edits.append((ri, gi, sub, draws))
                    group.meshes[si] = draws[0] if draws else empty
                    extras.setdefault((ri, gi), []).extend(draws[1:])
                    old = struct.unpack('<4f', group.suffix[-16:])
                    radius = max(math.sqrt(sum(x*x for x in old[:3])) + abs(old[3]),
                                 max((math.sqrt(sum(x*x for x in v['position'])) + 1 for v in vertices), default=0))
                    group.suffix = group.suffix[:-16] + struct.pack('<4f', 0, 0, 0, radius)
        changed += 1; triangles += len(faces)
    # Append extras only after all original indices have been resolved.
    for (ri, gi), draws in extras.items(): meshes[ri].groups[gi].meshes.extend(draws)
    lod_report=[]
    if single_lod:
        from .runtime_mesh import keep_single_lod
        lod_report=keep_single_lod(package,{ri for ri,gi,old,draws in runtime_edits} | {r[0] for r in cloth_requests})
    from .runtime_mesh import route_render_edits
    runtime_report = route_render_edits(package, meshes, runtime_edits, enabled=bone_skin_cloth)
    if cloth_requests:
        from .cloth_route import route
        cloth_report.extend(route(package,meshes,cloth_requests,metadata))
    from .texture_export import export_materials, validate_texture
    texture_resources,texture_report=export_materials(package,objects,experimental_textures,fingerprints.finish(),single_mip)
    from .shader_profiles import export_shaders
    shader_resources,shader_report=export_shaders(package,objects,experimental_textures)
    if embed_textures:
        from .texture_bundle import embed_dependencies
        from .materials import ArchiveTextures
        package.texture_attachments=embed_dependencies(package,ArchiveTextures(selected_game_index))
    all_resources={**texture_resources,**shader_resources,**presets.resources,**cloth_resources}
    output = package.rebuild(meshes,force_layout=bool(texture_report or shader_report or presets.reports or bone_payloads or cloth_payloads or attachment_payloads or lod_report),
                             resource_data=all_resources)
    validated = Package(output)
    record_map = package.serialized_record_indices
    for ri,payload in attachment_payloads.items():
        if validated.container.records[record_map[ri]].payload != payload:
            raise ValueError('Native NODE attachment readback failed')
    for report in lod_report:
        ri=report['catalog']
        if validated.container.records[record_map[ri]].payload!=package.container.records[ri].payload:
            raise ValueError('Single Mesh LOD catalog readback failed')
    for ri, payload in cloth_payloads.items():
        if validated.container.records[record_map[ri]].payload != payload: raise ValueError('Native cloth mask readback failed')
    for ri, raw in cloth_resources.items():
        actual = validated.members[validated.record_members[record_map[ri]]].unpacked
        if actual != raw: raise ValueError('Native external cloth mask readback failed')
    for ri in bone_payloads:
        # Cloth/LOD routing can also update the MESH header in the same
        # skeleton entity. Validate the combined final payload, not its earlier
        # bone-only snapshot (the SKELETON tail retains our inverse binds).
        if validated.container.records[record_map[ri]].payload!=package.container.records[ri].payload:
            raise ValueError('Skeleton round-trip validation failed')
    if len(validated.members) != len(package.members)+sum(ri not in package.record_members for ri in all_resources): raise ValueError('SEGS validation failed')
    for ri,raw in texture_resources.items():
        ri=record_map[ri]
        actual=validated.members[validated.record_members[ri]].unpacked
        if actual!=raw:raise ValueError('Texture resource validation failed')
        validate_texture(validated.container.records[ri],actual)
    for ri,raw in shader_resources.items():
        ri=record_map[ri]
        if validated.members[validated.record_members[ri]].unpacked!=raw:
            raise ValueError('Shader resource validation failed')
    for ri,raw in presets.resources.items():
        ri=record_map[ri]
        if validated.members[validated.record_members[ri]].unpacked!=raw:raise ValueError('Preset resource validation failed')
        if validated.container.records[ri].kind==2137:validate_texture(validated.container.records[ri],raw)
    for ri, expected in meshes.items():
        actual = validated.mesh(record_map[ri])
        if actual.pack_geometry() != expected.pack_geometry(): raise ValueError('Geometry validation failed')
        for sub in actual.flat():
            count=sub.count
            if any(i >= count for face in decode_faces(actual, sub) for i in face):
                raise ValueError('Invalid exported triangle index')
    context.scene['dbh_export_texture_count']=len(texture_report)+len(presets.texture_reports)
    context.scene['dbh_export_preset_count']=len(presets.reports)
    context.scene['dbh_export_embedded_count']=len(validated.texture_attachments)
    context.scene['dbh_export_deleted_count']=len(deleted)
    context.scene['dbh_export_bone_count']=len(bone_report)
    context.scene['dbh_export_cloth_count']=len(cloth_report)
    context.scene['dbh_export_bone_ik_warning']=any(edit['ik_controlled'] for edit in bone_report)
    context.scene['dbh_export_recovered_bindings']=json.dumps(recovered)
    destination.write_bytes(output)
    return changed, triangles
