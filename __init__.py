"""Detroit PC native geometry import and growing-buffer replacement export."""
import bpy
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper
from pathlib import Path
from . import blender_io, material_ui
from .legacy import DEFAULT_INDEX

bl_info = {
    'name': 'Detroit: Become Human DATA_CONTAINER / SEGS',
    'author': 'TheLeonX',
    'version': (1, 0, 1), 'blender': (4, 0, 0),
    'location': 'File > Import/Export; Material Properties > Detroit Shader Texture Slots',
    'description': 'Native mesh, animation and shader editing; experimental painted CLOTH_SIMULATION export',
    'category': 'Import-Export',
}

class DBHPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__
    game_index: StringProperty(name='Game BigFile_PC.idx', subtype='FILE_PATH', default=DEFAULT_INDEX)
    texture_folder: StringProperty(name='Converted texture folder (optional)', subtype='DIR_PATH')
    def draw(self, context):
        for name in ('game_index', 'texture_folder'):
            self.layout.prop(self, name)
        from .game_paths import resolve_index
        index = resolve_index(context, preferences=self)
        if index is None:
            self.layout.label(text='Select a Steam or Epic index above.', icon='INFO')
        else:
            self.layout.label(text=str(index), icon='FILE_TICK' if index.is_file() else 'ERROR')
            if not index.is_file():
                self.layout.label(text='Selected index missing; no default fallback.', icon='ERROR')
        self.layout.label(text='Texture names: hex resource ID or material ID, e.g. 3AB53.png.')

class IMPORT_SCENE_OT_dbh(bpy.types.Operator, ImportHelper):
    bl_idname = 'import_scene.dbh_container'
    bl_label = 'Import Detroit Native Package'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.segs'
    filter_glob: StringProperty(default='*.segs;*.data_container;*_container', options={'HIDDEN'})
    def execute(self, context):
        try:
            prefs = context.preferences.addons[__package__].preferences
            objects, warnings = blender_io.import_file(context, Path(self.filepath), prefs)
            self.report({'WARNING'} if warnings else {'INFO'},
                        f'Imported {len(objects)} smooth meshes. ' + '; '.join(warnings))
            return {'FINISHED'}
        except Exception as error:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, str(error)); return {'CANCELLED'}

class OBJECT_OT_dbh_replace(bpy.types.Operator):
    bl_idname = 'object.dbh_use_replacement'
    bl_label = 'Use Selected Mesh as Replacement'
    bl_description = 'Bind a custom mesh to the named game slot, transfer bone weights and use its original armature'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        try:
            blender_io.bind_replacement(context)
            self.report({'INFO'}, 'Replacement bound. Export a new SEGS package to save it.')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error)); return {'CANCELLED'}

class OBJECT_OT_dbh_skin_missing(bpy.types.Operator):
    bl_idname='object.dbh_skin_missing'
    bl_label='Skin Unweighted Vertices to Bones'
    bl_description='Give new/unweighted vertices nearest native bone weights; preserves existing weights and uses the imported game rig'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            count=blender_io.skin_missing_vertices(context)
            self.report({'INFO'},f'Skinned {count} unweighted vertices to the existing game bones')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'},str(error));return {'CANCELLED'}

class OBJECT_OT_dbh_attach_bone(bpy.types.Operator):
    bl_idname='object.dbh_attach_bone'
    bl_label='Attach Mesh to Game Bone'
    bl_description='Preview a rigid Child Of attachment and export it as full native bone weighting'
    bl_options={'REGISTER','UNDO'}
    bone: StringProperty(name='Bone')
    def invoke(self,context,event):
        from .scene_export import manifest_for_object
        from .bone_attachment import suggested_bone
        found=manifest_for_object(context.active_object)
        if not found:self.report({'ERROR'},'Select a named Detroit mesh slot');return {'CANCELLED'}
        self.bone=suggested_bone(context.active_object,found[1])
        return context.window_manager.invoke_props_dialog(self)
    def draw(self,context):
        from .blender_io import object_armature
        obj=context.active_object
        arm=object_armature(obj) if obj else None
        if arm:self.layout.prop_search(self,'bone',arm.data,'bones')
        else:self.layout.prop(self,'bone')
    def execute(self,context):
        try:
            from .bone_attachment import attach
            attach(context,self.bone)
            self.report({'INFO'},f'Attached to {self.bone}; export converts it to rigid game bone weights')
            return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}

class OBJECT_OT_dbh_detach_bone(bpy.types.Operator):
    bl_idname='object.dbh_detach_bone'
    bl_label='Remove Bone Attachment'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            from .bone_attachment import detach
            detach(context);return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}

class OBJECT_OT_dbh_simplify_names(bpy.types.Operator):
    bl_idname='object.dbh_simplify_names'
    bl_label='Simplify Imported Mesh Names'
    bl_description='Rename this imported package meshes to DBH_record_slot without changing their game bindings'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            count=blender_io.simplify_mesh_names(context)
            self.report({'INFO'},f'Renamed {count} mesh slots')
            return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}

class EXPORT_SCENE_OT_dbh(bpy.types.Operator, ExportHelper):
    bl_idname = 'export_scene.dbh_segs_patch'
    bl_label = 'Export Detroit Replacement Package'
    filename_ext = '.segs'
    filter_glob: StringProperty(default='*.segs', options={'HIDDEN'})
    bone_skin_cloth: BoolProperty(
        name='Bone-skin edited cloth', default=True,
        description='Use edited cloth surfaces as bone-skinned meshes across linked LODs; removes their cloth motion')
    experimental_textures: BoolProperty(name='Export Custom Textures',default=True,
        description='Export custom images, authored materials and Game Shader presets. New presets require game testing')
    embed_textures: BoolProperty(name='Include all textures in SEGS',default=True,
        description='Losslessly bundle original texture resources and all mip levels; requires the DBH loose-file loader')
    export_deletions: BoolProperty(name='Enable Deleting Meshes For Export',default=True,
        description='Remove render draws for deleted imported mesh objects, including linked LODs; retain cloth simulation inputs')
    export_bones: BoolProperty(name='Export Bone Edits', default=True,
        description='Save Edit Mode/rest bone head positions and matching inverse binds. Does not export pose, rotation, hierarchy or animation edits')
    single_lod: BoolProperty(name='Single Mesh LOD EXPERIMENTAL',default=False,
        description='Use only the highest-detail render LOD at all distances; retain native collision/cloth resources. Experimental: test in game')
    single_mip: BoolProperty(name='Single Texture Mip (only for custom textures) EXPERIMENTAL',default=False,
        description='Export only the full-resolution level for new/edited textures; unchanged vanilla textures keep their original mipmaps. May increase distant texture shimmer')
    def execute(self, context):
        try:
            count, triangles = blender_io.export_file(context, Path(self.filepath), self.bone_skin_cloth,self.experimental_textures,self.embed_textures,self.export_deletions,self.export_bones,self.single_lod,self.single_mip)
            textures=context.scene.get('dbh_export_texture_count',0)
            embedded=context.scene.get('dbh_export_embedded_count',0)
            deleted=context.scene.get('dbh_export_deleted_count',0)
            presets=context.scene.get('dbh_export_preset_count',0)
            bones=context.scene.get('dbh_export_bone_count',0)
            cloth=context.scene.get('dbh_export_cloth_count',0)
            self.report({'INFO'}, f'Exported {count} edited meshes, {cloth} cloth masks, {bones} bone positions, {deleted} deletions, {presets} preset materials, {textures} textures, {embedded} bundled resources.')
            if context.scene.get('dbh_export_bone_ik_warning'):
                self.report({'WARNING'}, 'Some edited bones are IK-controlled: native rest positions saved, but game animation/IK may override them.')
            return {'FINISHED'}
        except Exception as error:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, str(error)); return {'CANCELLED'}

class OBJECT_OT_dbh_fix_bone_sides(bpy.types.Operator):
    bl_idname = 'object.dbh_fix_bone_sides'
    bl_label = 'Fix Bone Left/Right Names'
    bl_description = 'Correct old Detroit side labels, weights and animation paths without moving bones or changing native IDs'
    bl_options = {'REGISTER', 'UNDO'}
    def execute(self, context):
        try:
            from .bone_names import repair
            count = repair(context)
            self.report({'INFO'}, f'Corrected {count} bone names' if count else 'Bone names are already corrected')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}

class VIEW3D_PT_dbh(bpy.types.Panel):
    bl_label = 'Detroit Mesh Replacement'
    bl_idname = 'VIEW3D_PT_dbh'
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'Detroit'
    def draw(self, context):
        layout = self.layout
        layout.operator('dbh.asset_editor',icon='PREFERENCES')
        layout.operator('object.dbh_organize_package',icon='OUTLINER_COLLECTION')
        layout.operator('object.dbh_style_armature')
        layout.operator(IMPORT_SCENE_OT_dbh.bl_idname)
        layout.label(text='Select custom mesh, then target LAST.')
        layout.operator(OBJECT_OT_dbh_replace.bl_idname)
        layout.operator(OBJECT_OT_dbh_skin_missing.bl_idname)
        layout.operator(OBJECT_OT_dbh_simplify_names.bl_idname)
        layout.operator(OBJECT_OT_dbh_fix_bone_sides.bl_idname)
        obj = context.active_object
        if obj and obj.type=='MESH':
            constraint=obj.constraints.get('Detroit Bone Attachment')
            if constraint and constraint.type=='CHILD_OF':
                layout.label(text=f'Bone attachment: {constraint.subtarget}',icon='CONSTRAINT_BONE')
                layout.operator(OBJECT_OT_dbh_detach_bone.bl_idname)
            else:layout.operator(OBJECT_OT_dbh_attach_bone.bl_idname)
        layout.operator(EXPORT_SCENE_OT_dbh.bl_idname)
        layout.label(text='Material presets: Material Properties.')
        layout.label(text='Bone physics: replacement inherits rig + weights.')
        layout.label(text='Bone positions: move heads in Edit Mode.')
        layout.label(text='Changed record: old morphs disabled.')
        from .scene_export import manifest_for_object,slot_for_object
        if obj and obj.type=='MESH':
            found=manifest_for_object(obj)
            if found:
                try:
                    slot_for_object(obj,found[1])
                    layout.label(text=f'Mesh slot: {obj.name}',icon='MESH_DATA')
                    layout.label(text='Number = container record index (decimal).')
                except ValueError:pass
        from .outline import draw
        draw(layout,obj)

def _import_menu(self, context):
    self.layout.operator(IMPORT_SCENE_OT_dbh.bl_idname, text='Detroit Native Package (.segs)')
    self.layout.operator('import_scene.dbh_anim_file', text='Detroit Animation (.anim_data)')

def _export_menu(self, context):
    self.layout.operator(EXPORT_SCENE_OT_dbh.bl_idname, text='Detroit Replacement Package (.segs)')
    self.layout.operator('export_scene.dbh_anim_data', text='Detroit Animation (.anim_data)')

CLASSES = (DBHPreferences, IMPORT_SCENE_OT_dbh, OBJECT_OT_dbh_replace, OBJECT_OT_dbh_skin_missing,
           OBJECT_OT_dbh_attach_bone,OBJECT_OT_dbh_detach_bone,
           OBJECT_OT_dbh_simplify_names,
           OBJECT_OT_dbh_fix_bone_sides,
           EXPORT_SCENE_OT_dbh, VIEW3D_PT_dbh)

def register():
    from . import asset_editor
    asset_editor.register()
    from . import scene_organization
    scene_organization.register()
    material_ui.register()
    from . import outline
    outline.register()
    for cls in CLASSES: bpy.utils.register_class(cls)
    from . import animation_blender
    animation_blender.register()
    from . import cloth_blender
    cloth_blender.register()
    bpy.types.TOPBAR_MT_file_import.append(_import_menu)
    bpy.types.TOPBAR_MT_file_export.append(_export_menu)

def unregister():
    from . import cloth_blender
    cloth_blender.unregister()
    from . import animation_blender
    animation_blender.unregister()
    from . import outline
    outline.unregister()
    bpy.types.TOPBAR_MT_file_export.remove(_export_menu)
    bpy.types.TOPBAR_MT_file_import.remove(_import_menu)
    for cls in reversed(CLASSES): bpy.utils.unregister_class(cls)
    material_ui.unregister()
    from . import asset_editor
    asset_editor.unregister()
    from . import scene_organization
    scene_organization.unregister()
