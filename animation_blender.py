"""Motion graph browser and native Detroit animation import/export UI."""
from pathlib import Path
import textwrap

import bpy
from bpy.props import BoolProperty, CollectionProperty, IntProperty, StringProperty
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Euler, Matrix

from .animation_assets import available_animation_ids, read_animation
from .animation_graph import read_motion_graph


def _target(context):
    from .blender_io import object_armature
    from .scene_export import manifest_for_object
    active = context.active_object
    armature = active if active and active.type == 'ARMATURE' else object_armature(active) if active else None
    if armature is None:
        raise ValueError('Select an imported Detroit armature or one of its meshes')
    found = manifest_for_object(armature)
    if not found:
        raise ValueError('Selected rig has no Detroit package metadata')
    return armature, found[1]


def _nodes_for_rig(metadata):
    return _native_rig_for_animation(metadata).record.payload


def _native_rig_for_animation(metadata):
    from .native import Package
    from .skeleton import Rig
    path = Path(metadata['source_segs'])
    if not path.is_file():
        raise ValueError(f'Original SEGS package is unavailable: {path}')
    package = Package(path.read_bytes())
    matches = []
    for record in package.container.records:
        raw = record.payload
        if record.kind == 0x85A and len(raw) >= 216 and raw[4:12] == b'NODE    ':
            rig = Rig(record)
            if rig.matches(metadata['bones']):
                matches.append(rig)
    if len(matches) != 1:
        raise ValueError(f'Expected one matching NODE skeleton, found {len(matches)}')
    return matches[0]


def _native_bind_matrices(rig):
    """Native NODE bind transforms, expressed in Blender armature coordinates."""
    axis = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    axis_inv = axis.inverted()
    result = [Matrix.Identity(4)]  # The reference exporter adds a synthetic Root.
    for joint in rig.joints:
        linear = joint.world_linear
        x, y, z = joint.world_position
        native = Matrix(((linear[0][0], linear[0][1], linear[0][2], x),
                         (linear[1][0], linear[1][1], linear[1][2], y),
                         (linear[2][0], linear[2][1], linear[2][2], z),
                         (0, 0, 0, 1)))
        result.append(axis @ native @ axis_inv)
    return result


def _apply_clip(context, armature, metadata, clip, name):
    expected = metadata['bones']
    if len(clip.bones) != len(expected):
        raise ValueError(f'Clip has {len(clip.bones)} bones; selected rig has {len(expected)}')
    for index, (source, target) in enumerate(zip(clip.bones, expected)):
        if source.parent != target['parent']:
            raise ValueError(f'Clip skeleton differs from rig at bone {index}')
        if armature.pose.bones.get(target['name']) is None:
            raise ValueError(f'Rig is missing bone {target["name"]}')
    if armature.animation_data is None:
        armature.animation_data_create()
    if armature.animation_data.action is not None:
        armature.animation_data.action.use_fake_user = True
    action = bpy.data.actions.new(name)
    action.use_fake_user = True
    from . import asset_metadata
    asset_metadata.update(action,dict(dbh_animation_space='native_bind_relative_v1',dbh_armature_name=armature.name))
    armature.animation_data.action = action
    armature.data.pose_position = 'POSE'
    pose_bones = [armature.pose.bones[item['name']] for item in expected]
    for bone in pose_bones:
        bone.rotation_mode = 'QUATERNION'
    # Detroit XYZ -> Blender XYZ (the same axis bridge as model import).
    axis = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    axis_inv = axis.inverted()
    rest = [bone.bone.matrix_local.copy() for bone in pose_bones]
    native_bind = _native_bind_matrices(_native_rig_for_animation(metadata))
    if len(native_bind) != len(rest):
        raise ValueError('Native and Blender bind skeletons differ in bone count')
    # Imported Blender bones point toward their children for display, while
    # Detroit's NODE joints have their own axes. Skinning requires the *delta*
    # from each native bind matrix, not the native absolute pose matrix.
    bind_to_display = [matrix.inverted() @ display
                       for matrix, display in zip(native_bind, rest)]
    relative_rest = [rest[index] if source.parent < 0 else rest[source.parent].inverted() @ rest[index]
                     for index, source in enumerate(clip.bones)]
    local = [None] * len(expected)
    # SMD frames are sparse snapshots: omitted bones retain their last pose.
    # A key only when a bone appears again lets Blender interpolate *before*
    # that SMD update, producing severe one-frame distortions on this graph.
    animated = set().union(*(changed for _, changed in clip.frames[1:]))
    previous_quaternions = [None] * len(expected)
    first_frame = clip.frames[0][0]
    for frame, changed in clip.frames:
        for index, pose in changed.items():
            tx, ty, tz, rx, ry, rz = pose
            local[index] = Matrix.Translation((tx, ty, tz)) @ Euler((rx, ry, rz), 'XYZ').to_matrix().to_4x4()
        if any(value is None for value in local):
            raise ValueError('The first SMD frame must initialize every bone')
        world = [None] * len(expected)
        for index, bone in enumerate(clip.bones):
            world[index] = world[bone.parent] @ local[index] if bone.parent >= 0 else local[index]
        target_frame = frame + 1
        blender_world = [axis @ matrix @ axis_inv @ bind_to_display[index]
                         for index, matrix in enumerate(world)]
        for index, bone in enumerate(pose_bones):
            parent = clip.bones[index].parent
            parent_inverse = blender_world[parent].inverted() if parent >= 0 else Matrix.Identity(4)
            bone.matrix_basis = relative_rest[index].inverted() @ parent_inverse @ blender_world[index]
            quaternion = bone.rotation_quaternion.copy()
            previous = previous_quaternions[index]
            if previous is not None and previous.dot(quaternion) < 0:
                quaternion.negate()
                bone.rotation_quaternion = quaternion
            previous_quaternions[index] = quaternion
            if frame == first_frame or index in animated:
                bone.keyframe_insert(data_path='location', frame=target_frame, group=bone.name)
                bone.keyframe_insert(data_path='rotation_quaternion', frame=target_frame, group=bone.name)
                bone.keyframe_insert(data_path='scale', frame=target_frame, group=bone.name)
    for curve in action.fcurves:
        for key in curve.keyframe_points:
            key.interpolation = 'LINEAR'
    context.scene.render.fps = 30
    context.scene.frame_end = max(context.scene.frame_end, clip.frames[-1][0] + 1)
    context.scene.frame_set(1)
    return action


def import_animation(context, animation: bytes, label: str):
    from .animation_native_blender import import_native_animation
    return import_native_animation(context, animation, label)


def export_action_smd(context, armature, metadata, destination):
    """Export edited pose keys to converter-compatible SMD, not ANIM_DATA."""
    if not armature.animation_data or not armature.animation_data.action:
        raise ValueError('Select a Detroit rig with an active animation Action')
    action = armature.animation_data.action
    from .asset_metadata import get as animation_field
    if animation_field(action,'dbh_animation_space') != 'native_bind_relative_v1' and not action.name.endswith('_retargeted'):
        raise ValueError('This Action predates native bind-pose retargeting; reimport it with add-on 0.8.0+')
    expected = metadata['bones']
    pose_bones = [armature.pose.bones[item['name']] for item in expected]
    if any(bone is None for bone in pose_bones):
        raise ValueError('The selected rig is missing animation bones')
    bind = _native_bind_matrices(_native_rig_for_animation(metadata))
    rest = [bone.bone.matrix_local.copy() for bone in pose_bones]
    axis = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    axis_inv = axis.inverted()
    first = round(action.frame_range[0])
    last = round(action.frame_range[1])
    if first < 1 or last < first or last - first > 10000:
        raise ValueError('Unsupported animation frame range')
    lines = ['version 1', 'nodes']
    for index, item in enumerate(expected):
        name = item['name'].replace('"', '_')
        lines.append(f'{index} "{name}" {item["parent"]}')
    lines.extend(('end', 'skeleton'))
    old_frame = context.scene.frame_current
    try:
        for frame in range(first, last + 1):
            context.scene.frame_set(frame)
            lines.append(f'time {frame - first}')
            native_world = [axis_inv @ bone.matrix @ rest[index].inverted() @ bind[index] @ axis
                            for index, bone in enumerate(pose_bones)]
            for index, item in enumerate(expected):
                parent = item['parent']
                local = native_world[parent].inverted() @ native_world[index] if parent >= 0 else native_world[index]
                x, y, z = local.translation
                rx, ry, rz = local.to_euler('XYZ')
                lines.append(f'{index} {x:.9g} {y:.9g} {z:.9g} {rx:.9g} {ry:.9g} {rz:.9g}')
    finally:
        context.scene.frame_set(old_frame)
    lines.append('end')
    Path(destination).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return last - first + 1


class DBHGraphClipItem(bpy.types.PropertyGroup):
    asset_id: StringProperty(name='ANIM_DATA ID')
    references: IntProperty(name='References')
    available: BoolProperty(name='In selected game index')
    labels: StringProperty(name='Direct graph labels')
    named_references: IntProperty(name='Named references')


class DBHGraphLabelItem(bpy.types.PropertyGroup):
    name: StringProperty(name='State / Node Label')
    node_id: IntProperty(name='Graph node ID')
    animation_ids: StringProperty(name='Direct ANIM_DATA IDs')


class DBH_UL_graph_clips(bpy.types.UIList):
    def filter_items(self, context, data, propname):
        query=context.scene.dbh_graph_search.strip().casefold()
        items=getattr(data,propname)
        return [self.bitflag_filter_item if not query or query in (item.asset_id+' '+item.labels).casefold() else 0
                for item in items], []

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        suffix = item.labels.split(' | ', 1)[0] if item.labels else 'graph node has an empty name'
        layout.label(text=f'{item.asset_id}  {suffix}  ({item.references} refs)',
                     icon='ACTION' if item.available else 'ERROR')


def _full_label(layout, context, text):
    width=getattr(context.region,'width',300)
    scale=context.preferences.system.ui_scale or 1.0
    columns=max(20,int((width/scale-55)/7))
    for line in textwrap.wrap(text,width=columns,break_long_words=True,break_on_hyphens=False) or ['']:
        layout.label(text=line)


class DBH_OT_copy_animation_labels(bpy.types.Operator):
    bl_idname='dbh.copy_animation_labels'
    bl_label='Copy ID and Full Names'

    def execute(self,context):
        scene=context.scene
        if not 0<=scene.dbh_graph_selected<len(scene.dbh_graph_clips):
            return {'CANCELLED'}
        entry=scene.dbh_graph_clips[scene.dbh_graph_selected]
        context.window_manager.clipboard=entry.asset_id+'\n'+entry.labels.replace(' | ','\n')
        return {'FINISHED'}


class IMPORT_SCENE_OT_dbh_graph(bpy.types.Operator, ImportHelper):
    bl_idname = 'import_scene.dbh_motion_graph'
    bl_label = 'Browse Detroit Motion Graph'
    filename_ext = '.motion_graph'
    filter_glob: StringProperty(default='*.motion_graph', options={'HIDDEN'})

    def execute(self, context):
        try:
            graph = read_motion_graph(Path(self.filepath))
            scene = context.scene
            from .game_paths import resolve_index
            index_path = resolve_index(context)
            available = available_animation_ids(index_path) if index_path is not None and index_path.is_file() else set()
            scene.dbh_graph_clips.clear()
            scene.dbh_graph_labels.clear()
            for asset_id, count in sorted(graph.clip_counts.items(), key=lambda item: (item[0] not in available, item[0])):
                entry = scene.dbh_graph_clips.add()
                entry.asset_id = f'0x{asset_id:X}'
                entry.references = count
                entry.available = asset_id in available
                entry.labels = ' | '.join(graph.clip_labels.get(asset_id, ()))
                entry.named_references = sum(ref.asset_id == asset_id and ref.label is not None
                                             for ref in graph.references)
            for label in graph.labels:
                entry = scene.dbh_graph_labels.add()
                entry.name = label.name
                entry.node_id = label.node_id
                entry.animation_ids = ', '.join(f'0x{asset_id:X}' for asset_id in
                                                graph.label_clips.get((label.node_id, label.name), ()))
            scene.dbh_graph_selected = 0
            scene.dbh_motion_graph_path = self.filepath
            scene.dbh_graph_schema = 2
            report = bpy.data.texts.get('Detroit Motion Graph Inventory') or bpy.data.texts.new('Detroit Motion Graph Inventory')
            report.clear()
            report.write(f'{Path(self.filepath).name}\n\nAnimation-node names -> ANIM_DATA IDs (typed array indices):\n')
            report.write('\n'.join(f'node {label.node_id}: {label.name} -> ' +
                                   ', '.join(f'0x{asset_id:X}' for asset_id in ids)
                                   for (node_id, name), ids in sorted(graph.label_clips.items())
                                   for label in graph.labels if label.node_id == node_id and label.name == name))
            report.write('\n\nAll referenced ANIM_DATA IDs (graph aliases, not unique asset filenames):\n')
            report.write('\n'.join(f'0x{asset_id:X}  ({count} references; ' +
                                   (' | '.join(graph.clip_labels.get(asset_id, ())) or 'no direct label') + ')'
                                   for asset_id, count in sorted(graph.clip_counts.items())))
            report.write('\n\nAll numbered graph labels:\n')
            report.write('\n'.join(f'node {label.node_id}: {label.name}' for label in graph.labels))
            count = sum(item.available for item in scene.dbh_graph_clips)
            named = sum(ref.label is not None for ref in graph.references)
            self.report({'INFO'}, f'{len(graph.labels)} labels, {len(graph.clip_counts)} clip IDs; {named}/{len(graph.references)} direct name links, {count} standalone')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class IMPORT_SCENE_OT_dbh_anim_id(bpy.types.Operator):
    bl_idname = 'import_scene.dbh_anim_id'
    bl_label = 'Import ANIM_DATA ID'
    bl_options = {'REGISTER', 'UNDO'}
    use_graph_selection: BoolProperty(name='Use selected graph ID', default=False)

    def execute(self, context):
        try:
            scene = context.scene
            text = scene.dbh_anim_id.strip()
            if self.use_graph_selection and scene.dbh_graph_clips and 0 <= scene.dbh_graph_selected < len(scene.dbh_graph_clips):
                selected = scene.dbh_graph_clips[scene.dbh_graph_selected]
                if not selected.available:
                    raise ValueError(f'{selected.asset_id} is not a standalone ANIM_DATA entry in this game index; select another ID or import an extracted ANIM_DATA file')
                text = selected.asset_id
            asset_id = int(text, 0)
            from .game_paths import resolve_index
            data = read_animation(resolve_index(context, required=True), asset_id)
            entry=next((item for item in scene.dbh_graph_clips if int(item.asset_id,0)==asset_id),None)
            names=entry.labels if entry else ''
            label=f'DBH_0x{asset_id:X}'+('_'+names.split(' | ',1)[0] if names else '')
            action, clip, missing = import_animation(context, data, label)
            from .asset_metadata import update
            update(action,dict(dbh_graph_labels=names,dbh_graph_source=scene.dbh_motion_graph_path if names else ''))
            self.report({'INFO'}, f'{action.name}: {clip.frame_count} frames, {len(clip.tracks)} native tracks; {missing} unmapped tracks preserved')
            return {'FINISHED'}
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class IMPORT_SCENE_OT_dbh_anim_file(bpy.types.Operator, ImportHelper):
    bl_idname = 'import_scene.dbh_anim_file'
    bl_label = 'Import Detroit ANIM_DATA File'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.anim_data'
    filter_glob: StringProperty(default='*.anim_data;*.ANIM_DATA;*.anim;*.bin', options={'HIDDEN'})

    def execute(self, context):
        try:
            path = Path(self.filepath)
            action, clip, missing = import_animation(context, path.read_bytes(), f'DBH_{path.stem}')
            self.report({'INFO'}, f'{action.name}: {clip.frame_count} frames, {len(clip.tracks)} native tracks; {missing} unmapped tracks preserved')
            return {'FINISHED'}
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class EXPORT_SCENE_OT_dbh_anim_native(bpy.types.Operator, ExportHelper):
    bl_idname = 'export_scene.dbh_anim_data'
    bl_label = 'Export Detroit ANIM_DATA'
    filename_ext = '.anim_data'
    filter_glob: StringProperty(default='*.anim_data', options={'HIDDEN'})
    use_scene_range: BoolProperty(name='Use Scene Frame Range', default=False,
        description='Bake the scene start/end range instead of the active Action range')

    def invoke(self, context, event):
        import re
        try:
            armature, _ = _target(context)
            action = armature.animation_data.action
            from .asset_metadata import get
            asset = get(action,'dbh_anim_asset_id', '')
            if not asset:
                match = re.search(r'0x([0-9a-f]+)', action.name, re.I)
                asset = '0x' + match.group(1).upper() if match else ''
            if asset and not self.filepath:
                from .game_paths import resolve_index
                index = resolve_index(context)
                if index is None or not index.is_file():
                    return ExportHelper.invoke(self, context, event)
                folder = index.parent / 'data_win32' / 'animations' / asset
                self.filepath = str(folder / (bpy.path.clean_name(action.name) + '.anim_data'))
        except (AttributeError, ValueError):
            pass
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            from .animation_native_blender import export_native_animation
            result = export_native_animation(context, self.filepath, self.use_scene_range)
            self.report({'INFO'}, f'Exported native ANIM_DATA: {result["frames"]} frames, {result["tracks"]} tracks')
            return {'FINISHED'}
        except Exception as error:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class VIEW3D_PT_dbh_animation(bpy.types.Panel):
    bl_label = 'Detroit Animations'
    bl_idname = 'VIEW3D_PT_dbh_animation'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Detroit'

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.operator('dbh.asset_editor',icon='PREFERENCES')
        layout.label(text='Select an imported Detroit mesh or rig.')
        layout.operator(IMPORT_SCENE_OT_dbh_graph.bl_idname)
        if scene.dbh_motion_graph_path:
            layout.label(text=Path(scene.dbh_motion_graph_path).name)
            refresh=layout.operator(IMPORT_SCENE_OT_dbh_graph.bl_idname,text='Refresh Graph Names',icon='FILE_REFRESH')
            refresh.filepath=scene.dbh_motion_graph_path
            if scene.dbh_graph_schema != 2:
                _full_label(layout,context,'This saved list uses the old name mapping. Refresh Graph Names to rebuild it.')
                return
            layout.prop(scene, 'dbh_graph_search', text='Name / ID')
            layout.template_list('DBH_UL_graph_clips', '', scene, 'dbh_graph_clips', scene,
                                 'dbh_graph_selected', rows=6)
            if scene.dbh_graph_clips and 0 <= scene.dbh_graph_selected < len(scene.dbh_graph_clips):
                selected = scene.dbh_graph_clips[scene.dbh_graph_selected]
                box = layout.box()
                box.label(text=f'{selected.asset_id}: all graph animation names')
                for name in selected.labels.split(' | ') if selected.labels else ():
                    _full_label(box,context,name)
                if not selected.labels:
                    box.label(text='The graph node has no stored name.', icon='INFO')
                box.operator(DBH_OT_copy_animation_labels.bl_idname,icon='COPYDOWN')
            available_count = sum(item.available for item in scene.dbh_graph_clips)
            layout.label(text=f'{available_count} standalone clips in selected game index')
            _full_label(layout,context,'Names are graph-node aliases; one clip can have several names.')
            layout.operator(IMPORT_SCENE_OT_dbh_anim_id.bl_idname,
                            text='Import Selected Graph ID').use_graph_selection = True
        layout.prop(scene, 'dbh_anim_id')
        layout.operator(IMPORT_SCENE_OT_dbh_anim_id.bl_idname,
                        text='Import Typed ANIM_DATA ID').use_graph_selection = False
        layout.operator(IMPORT_SCENE_OT_dbh_anim_file.bl_idname)
        layout.operator(EXPORT_SCENE_OT_dbh_anim_native.bl_idname)
        layout.label(text='Native import/export; no animation converter needed.', icon='INFO')


CLASSES = (DBHGraphClipItem, DBHGraphLabelItem, DBH_UL_graph_clips, DBH_OT_copy_animation_labels, IMPORT_SCENE_OT_dbh_graph,
           IMPORT_SCENE_OT_dbh_anim_id, IMPORT_SCENE_OT_dbh_anim_file, EXPORT_SCENE_OT_dbh_anim_native,
           VIEW3D_PT_dbh_animation)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.dbh_graph_clips = CollectionProperty(type=DBHGraphClipItem)
    bpy.types.Scene.dbh_graph_labels = CollectionProperty(type=DBHGraphLabelItem)
    bpy.types.Scene.dbh_graph_selected = IntProperty(default=0)
    bpy.types.Scene.dbh_anim_id = StringProperty(name='ANIM_DATA ID', default='0x267B')
    bpy.types.Scene.dbh_graph_search = StringProperty(name='Graph Label Search')
    bpy.types.Scene.dbh_motion_graph_path = StringProperty(name='Motion Graph', subtype='FILE_PATH')
    bpy.types.Scene.dbh_graph_schema = IntProperty(default=0,options={'HIDDEN'})


def unregister():
    del bpy.types.Scene.dbh_graph_schema
    del bpy.types.Scene.dbh_motion_graph_path
    del bpy.types.Scene.dbh_graph_search
    del bpy.types.Scene.dbh_anim_id
    del bpy.types.Scene.dbh_graph_selected
    del bpy.types.Scene.dbh_graph_labels
    del bpy.types.Scene.dbh_graph_clips
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
