"""Package manifests, mesh-name slot binding, and legacy scene compatibility."""
import json
import bpy
from .runtime_mesh import active_draw_indices


def slot_key(ri, mi):
    return f'{ri}:{mi}'


def manifest_for_object(obj):
    """Return (Text, metadata) for a named slot/rig, or an old ID-property scene."""
    if obj is None:
        return None
    token = obj.get('dbh_metadata')
    if token:
        block = bpy.data.texts.get(token)
        if block:
            return block, json.loads(block.as_string())
    matches = []
    for block in bpy.data.texts:
        if not block.name.startswith('DBH Package Metadata'):
            continue
        try:
            metadata = json.loads(block.as_string())
        except (ValueError, TypeError):
            continue
        if not metadata.get('source_segs') or not metadata.get('slot_objects'):
            continue
        names = {entry['name'] for entry in metadata['slot_objects'].values()}
        if obj.name in names or (obj.type == 'ARMATURE' and obj.name == metadata.get('armature_name')):
            matches.append((block, metadata))
    if len(matches) > 1:
        raise ValueError(f'{obj.name}: mesh name occurs in multiple Detroit imports')
    return matches[0] if matches else None


def slot_for_object(obj, metadata):
    for key, entry in metadata.get('slot_objects', {}).items():
        if obj.name == entry['name']:
            return tuple(int(part) for part in key.split(':'))
    if 'dbh_record_index' in obj and 'dbh_mesh_index' in obj:
        return int(obj['dbh_record_index']), int(obj['dbh_mesh_index'])
    raise ValueError(f'{obj.name}: not bound to a Detroit mesh slot')


def slot_entry(obj, metadata):
    ri, mi = slot_for_object(obj, metadata)
    return metadata.get('slot_objects', {}).get(slot_key(ri, mi), {})


def expected_slots(package, metadata):
    if 'imported_slots' in metadata:
        return {tuple(s) for s in metadata['imported_slots']}
    records = metadata.get('mesh_records')
    if not records:
        raise ValueError('This old scene has no import manifest; reimport before exporting deletions')
    result = set()
    for ri in records:
        if ri not in package.record_members:
            continue
        md = package.mesh(ri)
        for mi in active_draw_indices(package, ri):
            if md.flat()[mi].index_count:
                result.add((ri, mi))
    return result


def export_anchor(context):
    active = context.active_object
    if active and manifest_for_object(active):
        return active
    matches = []
    for obj in context.scene.objects:
        found = manifest_for_object(obj)
        if not found:
            continue
        arm = next((m.object for m in obj.modifiers if m.type == 'ARMATURE' and m.object), None) if obj.type == 'MESH' else None
        if active and (obj.parent == active or arm == active or (active.type == 'MESH' and arm and
                any(m.type == 'ARMATURE' and m.object == arm for m in active.modifiers))):
            matches.append(obj)
    ids = {manifest_for_object(obj)[0].name for obj in matches}
    if len(ids) == 1:
        return matches[0]
    raise ValueError('Select a named Detroit mesh or its imported armature to export this package')


def collect_objects(context, anchor, package, metadata):
    """Find every imported draw by object name; missing names mean deleted draws."""
    if 'slot_objects' in metadata:
        expected = expected_slots(package, metadata)
        objects, missing, recovered = [], set(), []
        scene_objects = set(context.scene.objects)
        for ri, mi in sorted(expected):
            entry = metadata['slot_objects'].get(slot_key(ri, mi))
            if not entry:
                raise ValueError(f'Mesh slot {ri}/{mi} is absent from the package manifest')
            name = entry['name']
            obj = bpy.data.objects.get(name)
            if obj is None:
                missing.add((ri, mi))
            elif obj not in scene_objects:
                raise ValueError(f'{name} exists outside the active scene; delete it explicitly or link it back')
            elif obj.type != 'MESH':
                raise ValueError(f'{name} must remain a mesh object')
            else:
                objects.append(obj)
        arms = {m.object for obj in objects for m in obj.modifiers if m.type == 'ARMATURE' and m.object}
        bound = set(objects)
        unknown = [obj.name for obj in context.scene.objects if obj.type == 'MESH' and obj not in bound
                   and not obj.name.startswith('DBH_SUPERSEDED_') and
                   (obj.parent in arms or any(m.type == 'ARMATURE' and m.object in arms for m in obj.modifiers))]
        if unknown:
            raise ValueError('Meshes attached to this rig are not assigned to named slots: ' +
                             ', '.join(unknown[:8]) + '. Use Selected Mesh as Replacement first.')
        return objects, missing, recovered

    # Old .blend files retain their original object-property bindings.
    token = anchor['dbh_metadata']
    source = anchor['dbh_source_segs']
    expected = expected_slots(package, metadata)
    objects = [o for o in context.scene.objects if o.type == 'MESH' and o.get('dbh_metadata') == token and not o.get('dbh_superseded')]
    present = {(int(o['dbh_record_index']), int(o['dbh_mesh_index'])) for o in objects}
    arms = {m.object for o in objects for m in o.modifiers if m.type == 'ARMATURE' and m.object}
    if anchor.type == 'ARMATURE':
        arms.add(anchor)
    recovery, unknown = [], []
    for o in context.scene.objects:
        if o.type != 'MESH' or 'dbh_source_segs' in o:
            continue
        related = o.parent in arms or any(m.type == 'ARMATURE' and m.object in arms for m in o.modifiers)
        if not related and not o.name.startswith('DBH_'):
            continue
        matches = []
        for ri, mi in expected-present:
            sub = package.mesh(ri).flat()[mi]
            if o.name in (f'DBH_{ri}_{mi:03d}',f'DBH_{ri}_{mi:03d}_m_{sub.material[1]:X}') and any(
                    m and m.get('dbh_material_id') == sub.material[1] for m in o.data.materials):
                matches.append((ri, mi))
        if len(matches) != 1 or not related:
            unknown.append(o.name)
            continue
        ri, mi = matches[0]
        recovery.append((o, ri, mi))
        present.add((ri, mi))
    if unknown:
        raise ValueError('Unbound meshes would be skipped: ' + ', '.join(unknown[:8]) + '. Bind them with Use Selected Mesh as Replacement before export.')
    for o, ri, mi in recovery:
        for k, v in dict(source_segs=source, package_code=metadata['code'], record_index=ri, mesh_index=mi,
                         y_offset=metadata.get('y_offset', 0.0), metadata=token,
                         source_sha256=anchor['dbh_source_sha256'], replacement=True).items():
            o['dbh_' + k] = v
        objects.append(o)
    missing = expected-present
    for o in bpy.data.objects:
        if o.type == 'MESH' and o.get('dbh_metadata') == token and not o.get('dbh_superseded'):
            slot = (int(o['dbh_record_index']), int(o['dbh_mesh_index']))
            if slot in missing:
                raise ValueError(f'{o.name} exists outside the active scene; delete it explicitly or link it back before export')
    return objects, missing, [o.name for o, _, _ in recovery]
