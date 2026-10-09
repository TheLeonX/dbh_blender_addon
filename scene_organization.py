"""Per-import record collections and blue, solver-enabled collision helpers."""
import bpy
import re

COLLIDER = 'Detroit Cloth Collider'
BLUE = (0.02,0.22,1.0,1.0)


def armature_display(obj):
    if not obj or obj.type != 'ARMATURE': raise ValueError('Select a Detroit armature or a mesh driven by it')
    obj.data.display_type = 'STICK'
    obj.show_in_front = True
    obj.data.show_bone_colors = True
    for bone in obj.data.bones:
        twist = 'twist' in bone.name.casefold()
        unknown = bool(re.fullmatch(r'(?:0[xX])?[0-9a-fA-F]{8}(?:\.\d+)?',bone.name))
        normal = (.02,.22,1.) if unknown else (.04,.8,.08) if twist else (.85,.04,.04)
        select = (.2,.5,1.) if unknown else (.2,1.,.3) if twist else (1.,.2,.2)
        active = (.5,.75,1.) if unknown else (.5,1.,.6) if twist else (1.,.5,.5)
        for colored in (bone,obj.pose.bones.get(bone.name)):
            if colored is None: continue
            colored.color.palette = 'CUSTOM'
            colored.color.custom.normal = normal
            colored.color.custom.select = select
            colored.color.custom.active = active
    return obj


def preview_modifier(modifier):
    from .cloth_native import PREVIEW
    return ((modifier.type == 'COLLISION' and modifier.name == COLLIDER) or
            (modifier.type == 'CLOTH' and modifier.name == PREVIEW))


def create_collection(context,metadata):
    root = bpy.data.collections.new('DBH_'+str(metadata.get('code','Package')))
    context.collection.children.link(root)
    metadata['import_collection'] = root.name
    metadata['record_collections'] = {}
    return root


def record_collection(root,metadata,record):
    name = metadata.setdefault('record_collections',{}).get(str(record),'')
    collection = bpy.data.collections.get(name)
    if collection is None or collection.name not in root.children:
        collection = bpy.data.collections.new(f'DBH_{record}')
        root.children.link(collection)
        metadata['record_collections'][str(record)] = collection.name
    return collection


def collision_display(obj):
    obj.display_type = 'WIRE'
    obj.show_wire = True
    obj.color = BLUE
    obj.hide_render = True
    modifier = obj.modifiers.get(COLLIDER)
    if modifier and modifier.type != 'COLLISION': raise ValueError('Detroit Cloth Collider name is in use')
    if modifier is None: modifier = obj.modifiers.new(COLLIDER,'COLLISION')
    modifier.show_viewport = True
    modifier.show_render = True  # Needed for cloth in rendered previews too.
    obj.collision.use = True
    obj.collision.thickness_outer = .002
    obj.collision.thickness_inner = .001
    obj.collision.cloth_friction = 5.
    obj.collision.use_culling = False
    return modifier


def wire_colors():
    # This changes only wire colors, not normal materials/solid color mode.
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D': area.spaces.active.shading.wireframe_color_type = 'OBJECT'


def root_for(context,metadata):
    root = bpy.data.collections.get(metadata.get('import_collection',''))
    if root is None: root = create_collection(context,metadata)
    return root


def organize(context,text,metadata,package):
    """Refresh existing scenes without reimporting/replacing edited geometry."""
    from .auxiliary_mesh import collision_slots
    from .metadata_text import write_metadata
    colliders = collision_slots(package,metadata)
    objects = []
    scene_objects = set(context.scene.objects)
    for key,entry in metadata.get('slot_objects',{}).items():
        obj = bpy.data.objects.get(entry['name'])
        if obj is not None and obj in scene_objects:
            if obj.library: raise ValueError('Linked meshes cannot be reorganized')
            objects.append((key,entry,obj))
    def scene_collections(parent):
        yield parent
        for child in parent.children: yield from scene_collections(child)
    scope = set(scene_collections(context.scene.collection))
    root = root_for(context,metadata)
    if root not in scope: context.collection.children.link(root)
    armature = bpy.data.objects.get(metadata.get('armature_name',''))
    if armature is not None and armature in scene_objects: armature_display(armature)
    count = 0
    for key,entry,obj in objects:
        ri,mi = map(int,key.split(':'))
        group = record_collection(root,metadata,ri)
        if obj.name not in group.objects: group.objects.link(obj)
        for previous in list(obj.users_collection):
            if previous != group and previous in scope: previous.objects.unlink(obj)
        collision = key in colliders
        entry['collision_mesh'] = collision
        if collision:
            collision_display(obj); count += 1
    wire_colors()
    write_metadata(text,metadata)
    return root,count


class OBJECT_OT_dbh_organize(bpy.types.Operator):
    bl_idname = 'object.dbh_organize_package'
    bl_label = 'Refresh Collections & Colliders'
    bl_description = 'Group this package by record number; style its armature as colored sticks in front; show blue wire collision helpers; keep edited geometry'
    bl_options = {'REGISTER','UNDO'}
    def execute(self,context):
        try:
            from .scene_export import export_anchor,manifest_for_object
            from .native import Package
            from .blender_io import digest
            from pathlib import Path
            anchor = export_anchor(context)
            text,metadata = manifest_for_object(anchor)
            raw = Path(metadata['source_segs']).read_bytes()
            if digest(raw) != metadata['package_sha256']: raise ValueError('Source package changed; reimport first')
            root,count = organize(context,text,metadata,Package(raw))
            self.report({'INFO'},f'{count} blue collision helpers; package grouped in {root.name}')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'},str(error)); return {'CANCELLED'}


class OBJECT_OT_dbh_style_armature(bpy.types.Operator):
    bl_idname = 'object.dbh_style_armature'
    bl_label = 'Style Armature'
    bl_description = 'Display the selected/driving armature as sticks in front: named main bones red, twist bones green, unknown hex-ID bones blue'
    bl_options = {'REGISTER','UNDO'}
    def execute(self,context):
        try:
            from .blender_io import object_armature
            obj = context.active_object
            armature_display(obj if obj and obj.type == 'ARMATURE' else object_armature(obj) if obj else None)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'},str(error)); return {'CANCELLED'}


def register():
    for cls in (OBJECT_OT_dbh_organize,OBJECT_OT_dbh_style_armature): bpy.utils.register_class(cls)
def unregister():
    for cls in (OBJECT_OT_dbh_style_armature,OBJECT_OT_dbh_organize): bpy.utils.unregister_class(cls)
