"""Weight-paint workflow and optional Blender cloth preview."""
import json
from pathlib import Path

import bpy
from bpy.props import BoolProperty, IntProperty, FloatProperty
from .cloth_pins import OBJECT_OT_dbh_cloth_select_unpinned,OBJECT_OT_dbh_cloth_pin_unpinned
from mathutils import Matrix, Vector

from .cloth_native import GROUP, PIN_GROUP, PREVIEW, ClothIndex, apply_masks, baseline_for, envelope
from .metadata_text import write_metadata


def mask(obj):
    group = obj.vertex_groups.get(GROUP)
    if not group: return None
    return [next((g.weight for g in v.groups if g.group == group.index), 0.) for v in obj.data.vertices]


def native_mesh_unchanged(obj, package, metadata, record, mesh):
    from .native import decode_vertices, decode_faces
    from .blender_io import object_armature
    from .legacy import _from_blender
    arm = object_armature(obj)
    matrix = (arm.matrix_world.inverted() if arm else Matrix.Identity(4)) @ obj.matrix_world
    md = package.mesh(record); sub = md.flat()[mesh]
    source = decode_vertices(md, sub)
    if len(obj.data.vertices) != len(source):
        raise ValueError(f'{obj.name}: native cloth painting retains its source vertex count')
    for vertex, original in zip(obj.data.vertices, source):
        position = list(_from_blender(matrix @ vertex.co)); position[1] -= metadata.get('y_offset', 0.)
        if any(abs(a - b) > 2e-5 for a, b in zip(position, original['position'])):
            raise ValueError(f'{obj.name}: edited cloth positions need a rebuilt Havok simulation mapping')
    obj.data.calc_loop_triangles()
    faces = [tuple(reversed(t.vertices)) for t in obj.data.loop_triangles]
    if faces != decode_faces(md, sub):
        raise ValueError(f'{obj.name}: edited cloth triangles need a rebuilt Havok simulation mapping')
    from .scene_organization import preview_modifier
    if any(m.show_viewport and m.type != 'ARMATURE' and not preview_modifier(m) for m in obj.modifiers):
        raise ValueError(f'{obj.name}: apply geometry modifiers before creating a new Havok simulation')
    from .bone_attachment import active_attachment
    if active_attachment(obj, metadata): raise ValueError('A rigid bone attachment cannot also be native cloth')


def prepare(context):
    from .scene_export import manifest_for_object, slot_for_object, slot_key
    from .native import Package
    obj = context.active_object
    if not obj or obj.type != 'MESH': raise ValueError('Select a mesh to paint cloth simulation')
    if obj.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    found = manifest_for_object(obj)
    supported = False
    if found:
        text, metadata = found
        source = Path(metadata.get('source_segs') or obj.get('dbh_source_segs', ''))
        package = Package(source.read_bytes())
        record, mesh = slot_for_object(obj, metadata)
        try:
            binding = ClothIndex(package).binding(record, mesh)
            native_mesh_unchanged(obj, package, metadata, record, mesh)
            supported = True
        except ValueError as error:
            reason = str(error)
        entry = metadata.get('slot_objects', {}).get(slot_key(record, mesh))
        if entry is not None:
            entry['cloth_simulation'] = dict(native=supported, status=binding.name if supported else reason)
            if supported:
                saved = next((r for r in metadata.get('cloth_simulation_masks', []) if
                              (r['resource_id'], r['buffer']) == (binding.asset, binding.buffer)), {})
                entry['cloth_simulation']['weight_mode'] = saved.get('weight_mode', 'MULTIPLIER')
            write_metadata(text, metadata)
    else:
        reason='Assign this mesh to an imported Detroit slot before game export'
    group = obj.vertex_groups.get(GROUP)
    if not group and supported:
        status = import_mask(obj, package, metadata, record, mesh, ClothIndex(package))
        if entry is not None: entry['cloth_simulation'] = status
        write_metadata(text, metadata)
        group = obj.vertex_groups.get(GROUP)
    if not group:
        group = obj.vertex_groups.new(name=GROUP)
        if obj.data.vertices: group.add(list(range(len(obj.data.vertices))), 1. if supported else 0., 'REPLACE')
    obj.vertex_groups.active_index = group.index
    return supported


def preview(context, enable=True):
    obj = context.active_object
    if not obj or obj.type != 'MESH' or mask(obj) is None: raise ValueError('Create CLOTH_SIMULATION first')
    if obj.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    if not enable:
        modifier = obj.modifiers.get(PREVIEW)
        if modifier and modifier.type == 'CLOTH': obj.modifiers.remove(modifier)
        return
    modifiers = [m for m in obj.modifiers if m.type == 'CLOTH' and m.name != PREVIEW]
    if modifiers: raise ValueError('An existing Cloth modifier is present; use its settings or remove it first')
    weights = mask(obj)
    support = [1.] * len(weights)
    from .scene_export import manifest_for_object, slot_for_object
    found = manifest_for_object(obj)
    if found:
        metadata = found[1]
        from .native import Package
        package = Package(Path(metadata.get('source_segs') or obj['dbh_source_segs']).read_bytes())
        try:
            record, mesh = slot_for_object(obj, metadata)
            binding = ClothIndex(package).binding(record, mesh)
            native_mesh_unchanged(obj, package, metadata, record, mesh)
            # Use the pristine maximum native blend, not an already masked export.
            report = next((r for r in metadata.get('cloth_simulation_masks', [])
                           if (r['resource_id'], r['buffer']) == (binding.asset, binding.buffer)), None)
            if not report or report.get('weight_mode') != 'ABSOLUTE':
                support = envelope(binding, baseline_for(binding, report))
        except ValueError: pass
    if not any(1. - w * s > .99 for w, s in zip(weights, support)):
        raise ValueError('Paint at least one pinned area at weight 0 before starting the preview')
    pin = obj.vertex_groups.get(PIN_GROUP) or obj.vertex_groups.new(name=PIN_GROUP)
    for i, (weight, native) in enumerate(zip(weights, support)): pin.add([i], 1. - weight * native, 'REPLACE')
    modifier = obj.modifiers.get(PREVIEW)
    if modifier and modifier.type != 'CLOTH': raise ValueError('Detroit Cloth Preview name is used by another modifier')
    modifier = modifier or obj.modifiers.new(PREVIEW, 'CLOTH')
    modifier.settings.vertex_group_mass = PIN_GROUP
    modifier.settings.use_dynamic_mesh = True
    modifier.settings.quality = 8
    modifier.collision_settings.collision_quality = 4
    if found:
        from .cloth_settings import preview_settings
        controls=preview_settings(obj,metadata,package,sum(w*s>1e-6 for w,s in zip(weights,support)))
        modifier.settings.mass=controls['mass']
        modifier.settings.quality=max(1,min(80,controls['quality']))
        modifier.collision_settings.use_collision=controls['collisions']
        modifier.collision_settings.distance_min=max(.001,controls['radius'])
        # A per-modifier gravity multiplier avoids changing other scene objects.
        # Blender cannot reproduce arbitrary Havok gravity directions per mesh.
        world=context.scene.gravity.z
        if abs(world)>1e-6 and abs(controls['gravity'][0])+abs(controls['gravity'][2])<1e-6:
            modifier.settings.effector_weights.gravity=max(0.,min(200.,controls['gravity'][1]/world))
        from .scene_organization import organize
        text,metadata = found
        root,count = organize(context,text,metadata,package)
        modifier.collision_settings.collection = root
    modifier.point_cache.frame_start = context.scene.frame_start
    modifier.point_cache.frame_end = context.scene.frame_end
    obj.vertex_groups.active_index = obj.vertex_groups[GROUP].index
    context.scene.frame_set(context.scene.frame_start)


def import_mask(obj, package, metadata, record, mesh, index, source_weights=False):
    """Show the native envelope by default; migrate legacy multiplier sidecars."""
    reports = metadata.get('cloth_simulation_masks', [])
    try: binding = index.binding(record, mesh)
    except ValueError as error:
        return dict(native=False, status=str(error))
    report = next((r for r in reports if (r['resource_id'], r['buffer']) == (binding.asset, binding.buffer)), None)
    if source_weights:
        baseline = baseline_for(binding)
        weights = envelope(binding, baseline)
    else:
        baseline = baseline_for(binding, report)
        weights = envelope(binding, baseline)
        if report:
            weights = (list(report['weights']) if report.get('weight_mode') == 'ABSOLUTE' or report.get('new_topology')
                       else [w * b for w, b in zip(report['weights'], weights)])
    if len(weights) != len(obj.data.vertices): raise ValueError('Saved cloth mask has the wrong vertex count')
    group = obj.vertex_groups.get(GROUP) or obj.vertex_groups.new(name=GROUP)
    if obj.data.vertices: group.remove(list(range(len(obj.data.vertices))))
    buckets = {}
    for i, value in enumerate(weights):
        if value: buckets.setdefault(value, []).append(i)
    for value, indices in buckets.items(): group.add(indices, value, 'REPLACE')
    obj.vertex_groups.active_index = group.index
    restored = dict(report or {})
    restored.update(record=record, mesh=mesh, resource_id=binding.asset, buffer=binding.buffer,
                    resource_record=binding.resource, name=binding.name, weights=weights,
                    baseline=sorted(baseline.items()), weight_mode='ABSOLUTE')
    metadata['cloth_simulation_masks'] = [r for r in reports if
            (r['resource_id'], r['buffer']) != (binding.asset, binding.buffer)] + [restored]
    from .cloth_settings import import_profiles
    asset=import_profiles(package,metadata,record,mesh,index,binding)
    return dict(native=True, status=binding.name, weight_mode='ABSOLUTE',profile_asset=asset)


def load_source_weights(context):
    from .scene_export import manifest_for_object, slot_for_object, slot_key
    from .native import Package
    obj = context.active_object
    found = manifest_for_object(obj)
    if not found: raise ValueError('Select an imported Detroit mesh')
    if obj.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    text, metadata = found
    raw = Path(metadata.get('source_segs') or obj['dbh_source_segs']).read_bytes()
    from .blender_io import digest
    if metadata.get('package_sha256') and digest(raw) != metadata['package_sha256']:
        raise ValueError('Source package changed; reimport it before loading cloth weights')
    package = Package(raw)
    record, mesh = slot_for_object(obj, metadata)
    native_mesh_unchanged(obj, package, metadata, record, mesh)
    status = import_mask(obj, package, metadata, record, mesh, ClothIndex(package), source_weights=True)
    if not status['native']: raise ValueError(status['status'])
    metadata['slot_objects'][slot_key(record, mesh)]['cloth_simulation'] = status
    write_metadata(text, metadata)
    obj.vertex_groups.active_index = obj.vertex_groups[GROUP].index
    if obj.modifiers.get(PREVIEW): preview(context, enable=False)


def export_masks(context, package, objects, metadata):
    from .scene_export import slot_for_object
    index = ClothIndex(package)
    requests = [];new={};wind_requests=[];physics_requests=[];collider_requests=[];world_requests=[]
    for obj in objects:
        weights = mask(obj)
        if weights is None: continue
        if any(m.show_viewport and m.type=='CLOTH' and m.name!=PREVIEW for m in obj.modifiers):
            raise ValueError(f'{obj.name}: apply/remove another Cloth modifier, or use Detroit Cloth Preview')
        record, mesh = slot_for_object(obj, metadata)
        from .cloth_settings import mesh_settings
        desired=mesh_settings(context.scene,obj,metadata,package,index)
        try:
            binding=index.binding(record, mesh)
            native_mesh_unchanged(obj, package, metadata, record, mesh)
            previous=next((r for r in metadata.get('cloth_simulation_masks',[]) if
                           (r['resource_id'],r['buffer'])==(binding.asset,binding.buffer)),None)
            # A source-preserving default is not an edit to untouched vanilla.
            authored=index.profiles(binding)['cloth'].startswith('DBH_CLOTH_')
            changed_donor=not authored and desired['donor_asset']!=binding.asset
            if changed_donor or (previous and previous.get('new_topology') and weights!=previous['weights']):
                raise ValueError('Authored cloth pins/solver settings changed; rebuilding native topology')
        except ValueError as error:
            from .cloth_route import family
            if not context.scene.dbh_cloth_new_topology:
                raise ValueError(f'{obj.name}: {error}. Enable Experimental New Cloth Topology to rebuild the simulation.') from error
            from .bone_attachment import active_attachment
            if active_attachment(obj,metadata):raise ValueError('A rigid Child Of attachment cannot also be cloth; remove it and use bone weights')
            new[(record,mesh)]=family(package,record)
            continue
        requests.append((record, mesh, weights,None,desired['overrides'].get('movement',1.)))
        physics_requests.append((binding,desired['overrides']))
        wind_requests.append((binding,desired['wind']))
        collider_requests.append((binding,desired['colliders'],record))
        world_requests.append((binding,desired.get('world_collision')))
    from .cloth_world import apply as apply_world,plan as world_plan
    world_requests=list(world_plan(world_requests).values())
    result=apply_masks(index, requests, metadata.get('cloth_simulation_masks', []))
    apply_world(index,[(b,w) for b,w in world_requests if w is False],result[0],result[1],result[2])
    from .cloth_colliders import apply as apply_colliders
    apply_colliders(index,collider_requests,result[0],result[1],result[2])
    apply_world(index,[(b,w) for b,w in world_requests if w is True],result[0],result[1],result[2])
    from .cloth_scalars import apply_settings
    apply_settings(index,physics_requests,result[0],result[1])
    from .cloth_settings import apply_wind
    apply_wind(index,wind_requests,result[0],result[1])
    return (*result,new)


def settings(scene):
    return dict(mass=scene.dbh_cloth_mass,radius=scene.dbh_cloth_radius,stiffness=scene.dbh_cloth_stiffness,
                bend=scene.dbh_cloth_bend,max_distance=scene.dbh_cloth_distance,damping=scene.dbh_cloth_damping,
                substeps=scene.dbh_cloth_substeps,iterations=scene.dbh_cloth_iterations,collisions=scene.dbh_cloth_collisions)


class OBJECT_OT_dbh_cloth_prepare(bpy.types.Operator):
    bl_idname = 'object.dbh_cloth_prepare'
    bl_label = 'Create CLOTH_SIMULATION'
    bl_description = 'Create a weight group: 0 follows bones; 1 follows cloth. Edited topology can export a new experimental Havok simulation'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        try:
            native = prepare(context)
            self.report({'INFO'} if native else {'WARNING'}, 'Native cloth mask ready to paint and export' if native
                        else 'Paint pins at 0 and moving areas above 0; new native topology export is experimental')
            return {'FINISHED'}
        except Exception as error: self.report({'ERROR'}, str(error)); return {'CANCELLED'}


class OBJECT_OT_dbh_cloth_paint(bpy.types.Operator):
    bl_idname = 'object.dbh_cloth_paint'
    bl_label = 'Paint Cloth Simulation'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        try:
            obj = context.active_object
            if not obj or obj.type != 'MESH' or not obj.vertex_groups.get(GROUP): prepare(context)
            obj.vertex_groups.active_index = obj.vertex_groups[GROUP].index
            if obj.mode != 'OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.mode_set(mode='WEIGHT_PAINT')
            return {'FINISHED'}
        except Exception as error: self.report({'ERROR'}, str(error)); return {'CANCELLED'}


class OBJECT_OT_dbh_cloth_source(bpy.types.Operator):
    bl_idname = 'object.dbh_cloth_source'
    bl_label = 'Load Source Cloth Weights'
    bl_description = 'Replace the paint group with native blend weights from the imported SEGS (not collision or spring data)'
    bl_options = {'REGISTER', 'UNDO'}
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
    def execute(self, context):
        try:
            load_source_weights(context)
            self.report({'INFO'}, 'Source native blend weights loaded; paint edits were replaced')
            return {'FINISHED'}
        except Exception as error: self.report({'ERROR'}, str(error)); return {'CANCELLED'}


class OBJECT_OT_dbh_cloth_preview(bpy.types.Operator):
    bl_idname = 'object.dbh_cloth_preview'
    bl_label = 'Refresh / Start Cloth Preview'
    bl_options = {'REGISTER', 'UNDO'}
    enable: BoolProperty(default=True)
    def execute(self, context):
        try:
            preview(context, self.enable)
            self.report({'INFO'}, 'Preview refreshed; play the timeline. Export writes native cloth, not the Blender cache.' if self.enable else 'Cloth preview removed')
            return {'FINISHED'}
        except Exception as error: self.report({'ERROR'}, str(error)); return {'CANCELLED'}


class VIEW3D_PT_dbh_cloth(bpy.types.Panel):
    bl_label = 'Cloth Simulation'
    bl_idname = 'VIEW3D_PT_dbh_cloth'
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'Detroit'
    @classmethod
    def poll(cls, context): return context.active_object and context.active_object.type == 'MESH'
    def draw(self, context):
        layout = self.layout; obj = context.active_object
        layout.operator(OBJECT_OT_dbh_cloth_prepare.bl_idname)
        layout.operator(OBJECT_OT_dbh_cloth_source.bl_idname)
        from .cloth_settings import OBJECT_OT_dbh_cloth_settings,OBJECT_OT_dbh_cloth_reset_settings
        layout.operator(OBJECT_OT_dbh_cloth_settings.bl_idname)
        layout.operator(OBJECT_OT_dbh_cloth_reset_settings.bl_idname)
        if not obj.vertex_groups.get(GROUP): return
        layout.label(text='0 = bones; 1 = native cloth motion')
        layout.operator(OBJECT_OT_dbh_cloth_paint.bl_idname, icon='WPAINT_HLT')
        layout.operator(OBJECT_OT_dbh_cloth_select_unpinned.bl_idname)
        layout.operator(OBJECT_OT_dbh_cloth_pin_unpinned.bl_idname)
        from .scene_export import manifest_for_object, slot_entry
        found = manifest_for_object(obj)
        status = slot_entry(obj, found[1]).get('cloth_simulation', {}) if found else {}
        if status:
            layout.label(text='Native state-envelope weights' if status.get('weight_mode') == 'ABSOLUTE' else
                         'Existing native multiplier' if status.get('native') else 'New topology: experimental native export', icon='INFO')
        layout.label(text='Blend weights only, not collision/constraint data.')
        scene=context.scene
        layout.prop(scene,'dbh_cloth_new_topology')
        if scene.dbh_cloth_new_topology:
            box=layout.box();box.label(text='Native state profiles inherited by default')
            box.label(text='Physics & Wind: editable native defaults, per mesh.')
            box.label(text='Pin each disconnected piece at weight 0.')
        layout.operator(OBJECT_OT_dbh_cloth_preview.bl_idname)
        modifier = obj.modifiers.get(PREVIEW)
        if modifier and modifier.type == 'CLOTH':
            layout.label(text='Blender preview settings')
            layout.prop(modifier.settings, 'quality')
            layout.prop(modifier.settings, 'mass')
            layout.prop(modifier.settings, 'air_damping')
            layout.operator(OBJECT_OT_dbh_cloth_preview.bl_idname, text='Remove Preview').enable = False
        layout.label(text='Existing masks retain native collisions.')
        layout.label(text='New topology retains the donor\'s native colliders.')
        layout.label(text='Choose native collision meshes in Cloth Physics & In-Game Wind.')
        layout.label(text='Editing collider geometry for game export is not supported.')


CLASSES = (OBJECT_OT_dbh_cloth_prepare, OBJECT_OT_dbh_cloth_paint, OBJECT_OT_dbh_cloth_source,
           OBJECT_OT_dbh_cloth_select_unpinned,OBJECT_OT_dbh_cloth_pin_unpinned,
           OBJECT_OT_dbh_cloth_preview, VIEW3D_PT_dbh_cloth)
def register():
    bpy.types.Scene.dbh_cloth_new_topology=BoolProperty(name='Experimental New Cloth Topology',description='Build native game cloth for edited cloth or originally bone-skinned surfaces using this character\'s existing cloth family',default=True)
    bpy.types.Scene.dbh_cloth_collisions=BoolProperty(name='Use Model Collisions',default=True,
        description='Enable Blender collision helpers and retain the target donor colliders on new native cloth export; vanilla mask-only export preserves original collisions')
    bpy.types.Scene.dbh_cloth_mass=FloatProperty(name='Particle Mass (kg)',default=.04,min=.0001,max=10)
    bpy.types.Scene.dbh_cloth_radius=FloatProperty(name='Particle Radius (m)',default=.002,min=.00001,max=.1)
    bpy.types.Scene.dbh_cloth_stiffness=FloatProperty(name='Stretch Stiffness',default=.8,min=0,max=1)
    bpy.types.Scene.dbh_cloth_bend=FloatProperty(name='Bend Stiffness',default=.15,min=0,max=1)
    bpy.types.Scene.dbh_cloth_distance=FloatProperty(name='Maximum Movement (m)',default=.25,min=.0001,max=10)
    bpy.types.Scene.dbh_cloth_damping=FloatProperty(name='Damping / Second',default=.95,min=0,max=1)
    bpy.types.Scene.dbh_cloth_substeps=IntProperty(name='Substeps',default=2,min=1,max=16)
    bpy.types.Scene.dbh_cloth_iterations=IntProperty(name='Solver Iterations',default=4,min=1,max=16)
    from .cloth_settings import CLASSES as SETTINGS_CLASSES
    for cls in (*CLASSES,*SETTINGS_CLASSES): bpy.utils.register_class(cls)
def unregister():
    from .cloth_settings import CLASSES as SETTINGS_CLASSES,clear_donor_sessions
    for cls in reversed((*CLASSES,*SETTINGS_CLASSES)): bpy.utils.unregister_class(cls)
    clear_donor_sessions()
    for name in ('new_topology','collisions','mass','radius','stiffness','bend','distance','damping','substeps','iterations'):
        delattr(bpy.types.Scene,'dbh_cloth_'+name)
