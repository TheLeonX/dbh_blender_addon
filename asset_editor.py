"""Friendly mesh/animation editor; internal IDs and signatures stay automatic."""
import base64
import json
import math
from pathlib import Path
import bpy
from bpy.props import StringProperty, BoolProperty, EnumProperty
from . import asset_metadata
from .scene_export import manifest_for_object, slot_for_object, slot_entry
from .metadata_text import write_metadata

_editor = None


def object_changed(self,context):
    obj = bpy.data.objects.get(self.mesh_object)
    if obj: self.mesh_name = obj.name


def animation_payload(action):
    values = asset_metadata.values(action)
    text = bpy.data.texts.get(values.get('dbh_native_source_text',''))
    return dict(schema=1,kind='ANIMATION',values=values,
                native_source=text.as_string() if text else '')


def paste_animation(action, payload):
    from .animation_codec import parse_animation
    from .animation_native_blender import _save_source
    if payload.get('schema') != 1 or payload.get('kind') != 'ANIMATION':
        raise ValueError('Clipboard does not contain Detroit animation settings')
    values = dict(payload['values'])
    if set(values)-set(asset_metadata.KEYS): raise ValueError('Unknown animation settings')
    for key,value in values.items():
        if key in ('dbh_native_frame_count','dbh_native_unmapped_tracks'):
            if type(value) is not int or value < 0 or value > 1000000: raise ValueError('Invalid animation count')
        elif key == 'dbh_native_timestep':
            if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 < value <= 100:
                raise ValueError('Invalid animation timestep')
        elif not isinstance(value,str): raise ValueError('Invalid animation text field')
    raw = None
    if payload.get('native_source'):
        raw = base64.b64decode(''.join(payload['native_source'].split()),validate=True)
        parsed = parse_animation(raw)
        values['dbh_native_frame_count'] = parsed.frame_count
        values['dbh_native_timestep'] = parsed.timestep
        values.pop('dbh_native_source_text',None)
    elif values.get('dbh_native_source_text') and bpy.data.texts.get(values['dbh_native_source_text']) is None:
        raise ValueError('Clipboard native source is missing; copy from an imported Action with its source present')
    if values.get('dbh_animation_space') not in ('native_bind_relative_v1','native_bind_relative_v2'):
        raise ValueError('Copy settings from a verified imported Detroit animation')
    asset = values.get('dbh_anim_asset_id','')
    if asset and not 0 <= int(asset,0) <= 0xFFFFFFFF: raise ValueError('Invalid animation ID')
    asset_metadata.update(action,values)
    if raw: _save_source(action,raw,'')


class DBH_OT_asset_clipboard(bpy.types.Operator):
    bl_idname = 'dbh.asset_clipboard'
    bl_label = 'Detroit Settings Clipboard'
    operation: EnumProperty(items=[('COPY','Copy',''),('PASTE','Paste','')])
    def execute(self,context):
        try:
            if _editor is None: raise ValueError('Open Detroit Data Editor first')
            if self.operation == 'COPY':
                context.window_manager.clipboard = json.dumps(_editor.copy_payload(),ensure_ascii=True)
                self.report({'INFO'},'Detroit settings copied (not geometry or keyframes)')
            else:
                text = context.window_manager.clipboard
                if len(text) > 16*1024*1024: raise ValueError('Clipboard settings exceed 16 MiB')
                _editor.load_payload(json.loads(text))
                self.report({'INFO'},'Settings pasted into the editor; press OK to apply')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'},str(error)); return {'CANCELLED'}


class DBH_OT_asset_editor(bpy.types.Operator):
    bl_idname = 'dbh.asset_editor'
    bl_label = 'Detroit Data Editor'
    bl_options = {'REGISTER','UNDO'}
    category: EnumProperty(items=[('MESH','Mesh',''),('ANIMATION','Animation','')])
    mesh_object: StringProperty(name='Object',update=object_changed)
    mesh_slot: StringProperty(name='Game mesh slot',description='Pick the existing imported mesh this object should replace')
    mesh_name: StringProperty(name='Mesh name')
    attach_enabled: BoolProperty(name='Attach to bone',default=False)
    bone: StringProperty(name='Bone')
    action_name: StringProperty(name='Action')
    source_action: StringProperty(name='Copy native settings from Action')
    source_file: StringProperty(name='Original ANIM_DATA',subtype='FILE_PATH',
        description='Optional original native file used as an encoding template, not a replacement for edited keys')
    source_id: StringProperty(name='Original animation ID',description='Load encoding metadata from the selected game index')
    export_id: StringProperty(name='Override animation ID',description='Target data_win32/animations/0xID folder suggested by export')
    confirm_basis: BoolProperty(name='Keys use this Detroit armature',
        description='Required when configuring an Action which was not imported by Detroit')
    activate_action: BoolProperty(name='Set as active Action',default=True)
    staged_animation: StringProperty(options={'HIDDEN'})

    def invoke(self,context,event):
        global _editor
        obj = context.active_object
        self.category = 'MESH' if obj and obj.type == 'MESH' else 'ANIMATION'
        if obj:
            self.mesh_object = obj.name
            self.mesh_name = obj.name
            found = manifest_for_object(obj)
            if obj.type == 'MESH' and found:
                self.mesh_slot = obj.name
                c = obj.constraints.get('Detroit Bone Attachment')
                self.attach_enabled = bool(c and not c.mute and c.influence > 0)
                self.bone = c.subtarget if c else ''
            arm = obj if obj.type == 'ARMATURE' else next((m.object for m in obj.modifiers if m.type == 'ARMATURE'),None)
            action = arm.animation_data.action if arm and arm.animation_data else None
            if action:
                self.action_name = action.name
                self.export_id = asset_metadata.get(action,'dbh_anim_asset_id','')
        _editor = self
        return context.window_manager.invoke_props_dialog(self,width=620)

    def copy_payload(self):
        if self.category == 'ANIMATION':
            action = bpy.data.actions.get(self.action_name)
            if not action: raise ValueError('Choose an Action')
            result = animation_payload(action)
            if self.export_id: result['values']['dbh_anim_asset_id'] = self.export_id
            return result
        obj = bpy.data.objects.get(self.mesh_slot or self.mesh_object)
        found = manifest_for_object(obj)
        if not found: raise ValueError('Choose an imported mesh slot')
        ri,mi = slot_for_object(obj,found[1])
        return dict(schema=1,kind='MESH',manifest=found[0].name,slot=f'{ri}:{mi}',
                    attached=self.attach_enabled,bone=self.bone)

    def load_payload(self,payload):
        if payload.get('schema') != 1: raise ValueError('Not Detroit settings')
        if payload.get('kind') == 'ANIMATION':
            if set(payload.get('values',{}))-set(asset_metadata.KEYS): raise ValueError('Unknown animation fields')
            self.category = 'ANIMATION'
            self.staged_animation = json.dumps(payload)
            self.export_id = payload.get('values',{}).get('dbh_anim_asset_id','')
        elif payload.get('kind') == 'MESH':
            text = bpy.data.texts.get(payload.get('manifest',''))
            if not text: raise ValueError('Import the source package before pasting its mesh slot')
            metadata = json.loads(text.as_string())
            entry = metadata['slot_objects'].get(payload.get('slot',''))
            if not entry: raise ValueError('Clipboard mesh slot is absent')
            self.category = 'MESH'; self.mesh_slot = entry['name']
            self.attach_enabled = bool(payload.get('attached')); self.bone = payload.get('bone','')
        else: raise ValueError('Unknown Detroit clipboard category')

    def draw(self,context):
        layout = self.layout
        layout.prop(self,'category',expand=True)
        if self.category == 'MESH':
            layout.prop_search(self,'mesh_object',bpy.data,'objects')
            layout.prop_search(self,'mesh_slot',bpy.data,'objects')
            layout.prop(self,'mesh_name')
            obj = bpy.data.objects.get(self.mesh_slot or self.mesh_object)
            found = manifest_for_object(obj) if obj else None
            if found and obj.type == 'MESH':
                metadata = found[1]; entry = slot_entry(obj,metadata)
                layout.label(text='Source: '+Path(metadata['source_segs']).name)
                layout.label(text='Game slot: '+entry.get('name',obj.name)+' (IDs and schema are automatic)')
                layout.prop(self,'attach_enabled')
                arm = bpy.data.objects.get(metadata.get('armature_name',''))
                row = layout.row(); row.enabled = self.attach_enabled
                if arm: row.prop_search(self,'bone',arm.data,'bones')
                else: row.prop(self,'bone')
                layout.label(text='Native NODE attachment' if entry.get('native_attachment') else 'Attachment exports as rigid bone skinning')
                if entry.get('native_attachment'):
                    layout.label(text='Off = keep the mesh under the model root, without a bone attachment.')
            else: layout.label(text='Choose an imported game mesh slot to bind a custom mesh.',icon='INFO')
        else:
            layout.prop_search(self,'action_name',bpy.data,'actions')
            layout.prop_search(self,'source_action',bpy.data,'actions')
            layout.prop(self,'source_file'); layout.prop(self,'source_id'); layout.prop(self,'export_id')
            layout.prop(self,'confirm_basis'); layout.prop(self,'activate_action')
            action = bpy.data.actions.get(self.action_name)
            if action:
                data = asset_metadata.values(action)
                layout.label(text='Frames: %s | Native timestep: %s' % (data.get('dbh_native_frame_count','unknown'),data.get('dbh_native_timestep','unknown')))
                if data.get('dbh_graph_labels'):
                    for line in str(data['dbh_graph_labels']).splitlines()[:4]: layout.label(text=line[:100])
            layout.label(text='Source metadata preserves native events and tracks. Edited keys stay in your Action.')
        row = layout.row(align=True)
        row.operator('dbh.asset_clipboard',text='Copy settings',icon='COPYDOWN').operation = 'COPY'
        row.operator('dbh.asset_clipboard',text='Paste settings',icon='PASTEDOWN').operation = 'PASTE'

    def execute(self,context):
        global _editor
        try:
            if self.category == 'MESH': self.apply_mesh(context)
            else: self.apply_animation(context)
            _editor = None
            self.report({'INFO'},'Detroit settings saved; export to write them into game files')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'},str(error)); return {'CANCELLED'}

    def cancel(self,context):
        global _editor
        _editor = None

    def apply_mesh(self,context):
        from .blender_io import bind_replacement
        from .bone_attachment import attach,detach
        obj = bpy.data.objects.get(self.mesh_object)
        target = bpy.data.objects.get(self.mesh_slot or self.mesh_object)
        if not obj or obj.type != 'MESH' or not target or target.type != 'MESH': raise ValueError('Choose mesh objects')
        found = manifest_for_object(target)
        if not found: raise ValueError('Choose an imported mesh as the game slot')
        metadata = found[1]
        if self.attach_enabled and self.bone not in {b['name'] for b in metadata['bones']}:
            raise ValueError('Choose a bone from the imported rig')
        if obj != target:
            if manifest_for_object(obj): raise ValueError('Another bound game mesh cannot overwrite this slot; duplicate it first')
            if slot_entry(target,metadata).get('native_attachment'):
                raise ValueError('Replace static auxiliary geometry by editing its imported object; ordinary custom meshes need a skinned slot')
            with context.temp_override(object=target,active_object=target,selected_objects=[target,obj],selected_editable_objects=[target,obj]):
                bind_replacement(context)
        found = manifest_for_object(obj)
        text,metadata = found
        key = '%d:%d' % slot_for_object(obj,metadata)
        if self.mesh_name:
            obj.name = self.mesh_name
            metadata['slot_objects'][key]['name'] = obj.name
            write_metadata(text,metadata)
        with context.temp_override(object=obj,active_object=obj,selected_objects=[obj],selected_editable_objects=[obj]):
            if self.attach_enabled: attach(context,self.bone)
            elif obj.constraints.get('Detroit Bone Attachment'): detach(context)

    def apply_animation(self,context):
        from .animation_native_blender import _save_source
        from .animation_assets import read_animation
        from .animation_codec import parse_animation
        from .animation_blender import _target
        action = bpy.data.actions.get(self.action_name)
        if not action: raise ValueError('Choose an Action')
        arm,metadata = _target(context)
        target = int(self.export_id,0) if self.export_id else None
        if target is not None and not 0 <= target <= 0xFFFFFFFF: raise ValueError('Invalid override animation ID')
        source = bpy.data.actions.get(self.source_action) if self.source_action else None
        payload = json.loads(self.staged_animation) if self.staged_animation else (animation_payload(source) if source else None)
        raw = None
        if self.source_file: raw = Path(bpy.path.abspath(self.source_file)).read_bytes()
        elif self.source_id:
            from .game_paths import resolve_index
            raw = read_animation(resolve_index(context, required=True),int(self.source_id,0))
        parsed = parse_animation(raw) if raw else None
        known = asset_metadata.get(action,'dbh_animation_space')
        if not known and not payload and not self.confirm_basis:
            raise ValueError('Confirm that the authored Action uses the selected Detroit rig')
        if not raw and not payload and not asset_metadata.get(action,'dbh_native_source_text'):
            raise ValueError('Choose an imported source Action, original ANIM_DATA file, or original ID')
        if payload: paste_animation(action,payload)
        if raw:
            _save_source(action,raw,'')
            asset_metadata.update(action,dict(dbh_native_frame_count=parsed.frame_count,dbh_native_timestep=parsed.timestep))
        fields = dict(dbh_armature_name=arm.name)
        if not known and not payload: fields['dbh_animation_space'] = 'native_bind_relative_v2'
        if target is not None: fields['dbh_anim_asset_id'] = f'0x{target:X}'
        asset_metadata.update(action,fields)
        if self.activate_action:
            if arm.animation_data is None: arm.animation_data_create()
            arm.animation_data.action = action
            if hasattr(action,'slots') and len(action.slots): arm.animation_data.action_slot = action.slots[0]


CLASSES = (DBH_OT_asset_clipboard,DBH_OT_asset_editor)


def register():
    asset_metadata.register()
    for cls in CLASSES: bpy.utils.register_class(cls)


def unregister():
    global _editor
    _editor = None
    for cls in reversed(CLASSES): bpy.utils.unregister_class(cls)
    asset_metadata.unregister()
