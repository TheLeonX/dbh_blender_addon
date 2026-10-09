"""Blender side labels only; native bone order, hashes and transforms never swap."""
import copy
import json
import re

SCHEMA = 1


def display_name(name):
    return re.sub(r'_([LR])(?=(?:\.\d+)?$)',
                  lambda match: '_R' if match[1] == 'L' else '_L', name)


def corrected_metadata(metadata):
    result = copy.deepcopy(metadata)
    if result.get('bone_side_names', 0) == SCHEMA:
        return result
    if result.get('bone_side_names', 0) != 0:
        raise ValueError('Unsupported bone-name convention')
    rows = result.get('bones', [])
    result['bone_source_names'] = [row['name'] for row in rows]
    for row in rows:
        row['name'] = display_name(row['name'])
    names = [row['name'] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate skeleton names; cannot safely change side labels')
    result['bone_side_names'] = SCHEMA
    return result


def repair(context):
    import bpy
    from .animation_blender import _target
    from .scene_export import manifest_for_object
    from .blender_io import object_armature, signature
    arm, metadata = _target(context)
    text, _ = manifest_for_object(arm)
    if metadata.get('bone_side_names') == SCHEMA:
        return 0
    if context.object.mode != 'OBJECT':
        raise ValueError('Switch to Object Mode before fixing bone side names')
    if arm.library or arm.data.library or arm.data.users != 1:
        raise ValueError('Make the Detroit armature local and single-user before renaming')
    updated = corrected_metadata(metadata)
    names = {row['name']: fixed['name'] for row, fixed in zip(metadata['bones'], updated['bones'])}
    mapping = {old: new for old, new in names.items() if old != new}
    if any(old not in arm.data.bones for old in names):
        raise ValueError('Rig names no longer match its package metadata; no names changed')
    if any(new in arm.data.bones and new not in mapping for new in mapping.values()):
        raise ValueError('A custom bone already uses a corrected name; no names changed')
    # Shared Actions would otherwise also rename animation paths on another rig.
    def actions(obj):
        animation = obj.animation_data
        if not animation:
            return set()
        result = {animation.action} if animation.action else set()
        for track in animation.nla_tracks:
            result.update(strip.action for strip in track.strips if strip.action)
        return result
    owned_actions = actions(arm)
    used_actions = set().union(*(actions(obj) for obj in bpy.data.objects))
    detached = []
    bone_path = re.compile(r'^pose\.bones\[("(?:\\.|[^"\\])*")\]')
    matching_rigs = [obj for obj in bpy.data.objects if obj.type == 'ARMATURE'
                     and set(obj.data.bones.keys()) == set(names)]
    for action in bpy.data.actions:
        from .asset_metadata import get
        if action in used_actions or not get(action,'dbh_animation_space'):
            continue
        owner = get(action,'dbh_armature_name')
        if owner and owner != arm.name:
            continue
        channels = list(action.fcurves)
        curve_names = {json.loads(match[1]) for curve in channels
                       if (match := bone_path.match(curve.data_path))}
        if not owner and curve_names != set(names):
            continue
        if not owner and len(matching_rigs) != 1:
            raise ValueError('A detached legacy Detroit Action matches multiple rigs; assign it to the intended rig before renaming')
        detached.append((action, channels))
    owned_actions.update(action for action, _ in detached)
    if any(action.library for action in owned_actions):
        raise ValueError('A bound Action is linked/read-only; no names changed')
    for other in bpy.data.objects:
        if other != arm and actions(other) & owned_actions:
            raise ValueError('An Action is shared with another object; make it single-user before renaming')
    slots = {entry['name']: entry for entry in updated.get('slot_objects', {}).values()}
    meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH' and
              (object_armature(obj) == arm or obj.name in slots or obj.get('dbh_metadata') == text.name)]
    if any(obj.library for obj in meshes):
        raise ValueError('A bound mesh is linked/read-only; no names changed')
    if any(new in obj.vertex_groups and new not in mapping
           for obj in meshes for old, new in mapping.items() if old in obj.vertex_groups):
        raise ValueError('A custom vertex group already uses a corrected name; no names changed')
    # Only refresh unchanged-mesh signatures. Previously edited meshes stay dirty.
    clean = [obj for obj in meshes if signature(obj,local=bool(slots.get(obj.name,{}).get('native_attachment'))) == slots.get(obj.name, {}).get('signature', obj.get('dbh_signature'))]
    bones = {old: arm.data.bones[old] for old in mapping}
    groups = [(group, old) for obj in meshes for old in mapping
              if (group := obj.vertex_groups.get(old)) is not None]
    occupied = set(arm.data.bones.keys()) | {group.name for obj in meshes for group in obj.vertex_groups}
    temporary = {}
    for i, old in enumerate(mapping):
        name = f'__DBH_SIDE_RENAME_{i}__'
        while name in occupied:
            name += '_'
        temporary[old] = name
        occupied.add(name)
    # Blender's bone rename updates bone parents, constraints, drivers, groups
    # and bound Action paths. Two phases prevent .001 collisions when swapping.
    for old, bone in bones.items():
        bone.name = temporary[old]
    for group, old in groups:
        group.name = temporary[old]
    for old, bone in bones.items():
        bone.name = mapping[old]
    for group, old in groups:
        group.name = mapping[old]
    # Blender updates assigned/NLA Actions itself, but not detached fake-user
    # Actions. Only migrate proven owners or uniquely matching legacy DBH Actions.
    for action, channels in detached:
        for curve in channels:
            match = bone_path.match(curve.data_path)
            if match and (name := json.loads(match[1])) in mapping:
                curve.data_path = 'pose.bones[' + json.dumps(mapping[name]) + ']' + curve.data_path[match.end():]
        action_groups = [(group, mapping[group.name]) for group in action.groups if group.name in mapping]
        for i, (group, _) in enumerate(action_groups):
            group.name = f'__DBH_ACTION_SIDE_{i}__'
        for group, name in action_groups:
            group.name = name
    for action in owned_actions:
        from .asset_metadata import update
        update(action,{'dbh_armature_name':arm.name})
    for obj in clean:
        value = signature(obj,local=bool(slots.get(obj.name,{}).get('native_attachment')))
        if obj.name in slots:
            slots[obj.name]['signature'] = value
        if 'dbh_signature' in obj:
            obj['dbh_signature'] = value
    from .metadata_text import write_metadata
    write_metadata(text, updated)
    context.view_layer.update()
    return len(mapping)
