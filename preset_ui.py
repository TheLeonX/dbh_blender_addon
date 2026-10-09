"""Semantic texture slots and presets for every Blender material."""
import bpy
from pathlib import Path
from bpy.app.handlers import persistent
from bpy.props import EnumProperty
from .preset_shader import PRESETS,ROLES,LABELS,BINDINGS,required_roles,detect_preset,detect_uv_offset,role_bindings
from .cel_shader import CEL_PRESETS,detect_ambient
from .pbr_shader import PBR_PRESETS,DATA_ROLES


def _node(mat,typ,name):
    node=mat.node_tree.nodes.get(name) or mat.node_tree.nodes.new(typ);node.name=name
    return node


def _bindings(mat):
    return role_bindings(mat.dbh_shader_mode,opaque=mat.dbh_surface_mode=='OPAQUE',
                         dither=mat.dbh_surface_mode=='DITHERED')


def default_celshade_image():
    """Pack the bundled ramp so saved blends remain self-contained."""
    path=Path(__file__).with_name('celshade_0.png')
    try:
        image=bpy.data.images.load(str(path),check_existing=True)
        if image.packed_file is None:image.pack()
        return image
    except (OSError,RuntimeError) as error:
        print(f'Detroit default celshade texture unavailable: {error}')
        return None


def fill_default_celshade(mat):
    if mat.dbh_shader_mode not in CEL_PRESETS:return
    for slot in mat.dbh_preset_slots:
        if slot.role=='CEL' and slot.image is None:
            node=mat.node_tree.nodes.get(slot.node_name) if mat.node_tree else None
            slot.image=(node.image if node and node.type=='TEX_IMAGE' else None) or default_celshade_image()


def add_slot(mat,role,image=None):
    if any(s.role==role for s in mat.dbh_preset_slots):raise ValueError(f'{LABELS[role]} slot already exists')
    slot=mat.dbh_preset_slots.add();slot.role=role;slot.binding=_bindings(mat).get(role,-1)
    slot.name=LABELS[role];slot.asset='';slot.uv='UV1';slot.channel='RGB'
    node=_node(mat,'ShaderNodeTexImage','DBH Preset '+LABELS[role]);slot.node_name=node.name
    uv=_node(mat,'ShaderNodeUVMap','DBH Preset UV '+role);uv.uv_map='UV1'
    mat.node_tree.links.new(uv.outputs['UV'],node.inputs['Vector'])
    slot.image=image or node.image
    if role=='CEL' and slot.image is None and mat.dbh_shader_mode in CEL_PRESETS:
        slot.image=default_celshade_image()
    mat.dbh_preset_index=len(mat.dbh_preset_slots)-1
    return slot


def sync_preview(mat):
    if not mat.use_nodes or mat.dbh_shader_mode not in PRESETS:return
    nodes=mat.node_tree.nodes;links=mat.node_tree.links
    bsdf=nodes.get('Principled BSDF') or _node(mat,'ShaderNodeBsdfPrincipled','Principled BSDF')
    out=next((n for n in nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output),None)
    if out is None:out=_node(mat,'ShaderNodeOutputMaterial','Material Output')
    links.new(bsdf.outputs['BSDF'],out.inputs['Surface'])
    sockets={'COLOR':'Base Color','NORMAL':'Normal','EMISSION':'Emission Color'}
    for socket in ('Base Color','Normal','Emission Color','Roughness','Metallic','Alpha'):
        for link in list(bsdf.inputs[socket].links):links.remove(link)
    bsdf.inputs['Base Color'].default_value=(1,1,1,1)
    bsdf.inputs['Roughness'].default_value=.5;bsdf.inputs['Metallic'].default_value=0
    bsdf.inputs['Alpha'].default_value=1;bsdf.inputs['Emission Color'].default_value=(0,0,0,1)
    bsdf.inputs['Emission Strength'].default_value=1
    for slot in mat.dbh_preset_slots:
        if not slot.image or slot.role not in required_roles(mat.dbh_shader_mode):continue
        node=nodes.get(slot.node_name)
        if node is None:continue
        node.image=slot.image
        if slot.role in DATA_ROLES:slot.image.colorspace_settings.name='Non-Color'
        node.label=LABELS[slot.role];node.location=(-600,250-ROLES.index(slot.role)*280)
        uv=nodes.get('DBH Preset UV '+slot.role)
        if uv:uv.location=(-820,node.location.y);uv.uv_map='UV1'
        if slot.role in ('CEL','ORM','ALPHA','FABRIC','TANGENT'):continue
        output=node.outputs['Color']
        if slot.role=='NORMAL':
            normal=_node(mat,'ShaderNodeNormalMap','DBH Preset Normal Decode');normal.uv_map='UV1'
            links.new(output,normal.inputs['Color']);output=normal.outputs['Normal']
        links.new(output,bsdf.inputs[sockets[slot.role]])
    if mat.dbh_shader_mode in CEL_PRESETS:
        from .cel_preview import sync
        sync(mat,bsdf)
    if mat.dbh_shader_mode in PBR_PRESETS:
        from .pbr_preview import sync
        sync(mat,bsdf)
    if mat.dbh_shader_mode!='HAIR':
        from .lighting_control import preview
        preview(mat,bsdf,out)
    if mat.dbh_shader_mode in PBR_PRESETS or mat.dbh_shader_mode=='HAIR':
        from .pbr_preview import transparency
        transparency(mat,out,bsdf)
        if mat.dbh_shader_mode=='HAIR' and hasattr(mat,'surface_render_method'):
            # Direct Principled alpha, not an extra translucent PBR mix layer.
            mat.surface_render_method='BLENDED'
    elif hasattr(mat,'surface_render_method'):mat.surface_render_method='DITHERED'
    if mat.get('dbh_outline_generated'):
        from .outline import cull_preview
        cull_preview(mat)
    else:
        sync_face_preview(mat,out)


def sync_face_preview(mat,out):
    mode=mat.dbh_face_mode
    mat.use_backface_culling=mode=='FRONT'
    if mode=='BOTH':return
    links=mat.node_tree.links
    surface=out.inputs['Surface'].links[0].from_socket
    geometry=_node(mat,'ShaderNodeNewGeometry','DBH Face Direction')
    transparent=_node(mat,'ShaderNodeBsdfTransparent','DBH Culled Face')
    mix=_node(mat,'ShaderNodeMixShader','DBH Face Cull')
    links.new(geometry.outputs['Backfacing'],mix.inputs[0])
    front,back=(surface,transparent.outputs['BSDF']) if mode=='FRONT' else (transparent.outputs['BSDF'],surface)
    links.new(front,mix.inputs[1]);links.new(back,mix.inputs[2])
    links.new(mix.outputs[0],out.inputs['Surface'])


def emission_changed(mat,context):
    if mat.dbh_shader_mode in PRESETS:sync_preview(mat)


def surface_changed(mat,context):
    if mat.dbh_shader_mode in PBR_PRESETS:
        physical=_bindings(mat)
        for slot in mat.dbh_preset_slots:slot.binding=physical.get(slot.role,-1)
        sync_preview(mat)


@persistent
def refresh_presets_after_load(_=None):
    # Only generated preset previews; do not touch original/custom materials.
    for mat in bpy.data.materials:
        if mat.dbh_shader_mode=='EYE_EMISSION':
            from .eye_preview import sync
            try:sync(mat)
            except Exception as error:print(f'Detroit eye preview ({mat.name}): {error}')
        if mat.dbh_shader_mode in PRESETS:
            try:sync_preview(mat)
            except Exception as error:print(f'Detroit preview refresh ({mat.name}): {error}')


def uv_changed(mat,context):
    if mat.dbh_shader_mode in CEL_PRESETS:sync_preview(mat)


def mode_changed(mat,context):
    if mat.dbh_shader_mode=='EYE_EMISSION':
        from .eye_preview import sync
        sync(mat);return
    if mat.node_tree:
        from .eye_preview import remove
        remove(mat)
    if mat.dbh_shader_mode not in PRESETS:return
    mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get('Principled BSDF')
    diffuse=None
    if bsdf and bsdf.inputs['Base Color'].is_linked:
        node=bsdf.inputs['Base Color'].links[0].from_node
        if node.type=='TEX_IMAGE':diffuse=node.image
    needed=required_roles(mat.dbh_shader_mode)
    for i in reversed(range(len(mat.dbh_preset_slots))):
        if mat.dbh_preset_slots[i].role not in needed:mat.dbh_preset_slots.remove(i)
    for role in needed:
        if not any(s.role==role for s in mat.dbh_preset_slots):add_slot(mat,role,diffuse if role=='COLOR' else None)
    for slot in mat.dbh_preset_slots:slot.binding=_bindings(mat).get(slot.role,-1)
    fill_default_celshade(mat)
    mat.dbh_preset_index=0
    sync_preview(mat)


def populate(mat,record,package,loader):
    if not record.external:return False
    raw=package.members[package.record_members[record.index]].unpacked
    mode=detect_preset(record,raw)
    if not mode:return False
    from .materials import bindings
    table=bindings(record.payload)
    mat.dbh_shader_mode=mode
    from .lighting_control import detect_emission
    from .outline_shader import detect_outline
    if detect_outline(raw):mat['dbh_outline_generated']=True
    mat.dbh_diffuse_emission=detect_emission(raw)
    from .motion_control import detect_enabled
    mat.dbh_motion_blur=detect_enabled(raw)
    from .outline_shader import detect_face_mode
    mat.dbh_face_mode=detect_face_mode(raw)
    if mode=='HAIR':
        from .hair_shader import detect as detect_hair
        mat.dbh_alpha_channel=detect_hair(raw)
    if mode in PBR_PRESETS:
        from .pbr_shader import detect,detect_surface
        mat.dbh_surface_mode=detect_surface(raw)
        _,mat.dbh_fabric_scale,mat.dbh_alpha_channel=detect(raw)
        for slot in mat.dbh_preset_slots:
            slot.binding=_bindings(mat).get(slot.role,-1)
    if mode in CEL_PRESETS:
        mat.dbh_uv_offset2=detect_uv_offset(raw)
        mat.dbh_ambient_color=detect_ambient(raw)
    for slot in mat.dbh_preset_slots:
        binding=_bindings(mat).get(slot.role,-1)
        if binding<0:
            slot.asset='';slot.image=None;slot.original_image=None;continue
        slot.asset=f'{table[binding].texture:X}'
        slot.image=loader.load(int(slot.asset,16));slot.original_image=slot.image
    sync_preview(mat)
    return True


class MATERIAL_OT_dbh_new_material(bpy.types.Operator):
    bl_idname='material.dbh_new_material';bl_label='New Detroit Material';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        obj=context.object
        if not obj or obj.type!='MESH':self.report({'ERROR'},'Select a mesh');return {'CANCELLED'}
        mat=bpy.data.materials.new('Detroit Material');mat.use_nodes=True;mat.dbh_shader_mode=PRESETS[0]
        if obj.material_slots:obj.active_material=mat
        else:obj.data.materials.append(mat)
        return {'FINISHED'}


class MATERIAL_OT_dbh_add_slot(bpy.types.Operator):
    bl_idname='material.dbh_add_slot';bl_label='Add Texture Slot';bl_options={'REGISTER','UNDO'}
    role:EnumProperty(name='Texture role',items=[(r,LABELS[r],'') for r in ROLES])
    def invoke(self,context,event):return context.window_manager.invoke_props_dialog(self)
    def execute(self,context):
        mat=getattr(context,'material',None) or context.object.active_material
        try:
            if mat.dbh_shader_mode not in PRESETS:raise ValueError('Choose a standard Game Shader preset first')
            if any(s.role==self.role for s in mat.dbh_preset_slots):raise ValueError('That texture role already exists')
            roles={s.role for s in mat.dbh_preset_slots}|{self.role}
            candidates=[m for m in PRESETS if roles<=set(required_roles(m))]
            if not candidates:raise ValueError('No preset supports this combination of texture roles')
            if mat.dbh_shader_mode in candidates:candidates=[mat.dbh_shader_mode]
            mode=min(candidates,key=lambda m:len(required_roles(m)))
            if mode!=mat.dbh_shader_mode:mat.dbh_shader_mode=mode
            else:add_slot(mat,self.role)
            mat.dbh_preset_index=next(i for i,s in enumerate(mat.dbh_preset_slots) if s.role==self.role)
            sync_preview(mat);return {'FINISHED'}
        except Exception as e:self.report({'ERROR'},str(e));return {'CANCELLED'}


class MATERIAL_OT_dbh_remove_slot(bpy.types.Operator):
    bl_idname='material.dbh_remove_slot';bl_label='Delete Texture Slot';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        mat=getattr(context,'material',None) or context.object.active_material
        if not mat.dbh_preset_slots:return {'CANCELLED'}
        i=min(mat.dbh_preset_index,len(mat.dbh_preset_slots)-1)
        slot=mat.dbh_preset_slots[i]
        node=mat.node_tree.nodes.get(slot.node_name)
        if node:node.image=None
        mat.dbh_preset_slots.remove(i);mat.dbh_preset_index=max(0,i-1);sync_preview(mat)
        return {'FINISHED'}


def draw(layout,mat):
    pbr=mat.dbh_shader_mode in PBR_PRESETS
    hair=mat.dbh_shader_mode=='HAIR'
    layout.label(text=('Native hair' if hair else 'Skin/Cloth render path' if pbr else 'Opaque')+' · UV1')
    if pbr:
        layout.prop(mat,'dbh_surface_mode')
        layout.label(text='Auto: opaque if alpha is all white; otherwise blended.')
    if hair:
        layout.label(text='Native density prepass + smooth alpha; no added dither.')
        layout.label(text='Uses native hair motion handling, not the PBR blur switch.')
        layout.prop(mat,'dbh_alpha_channel')
        layout.label(text='Strand Direction: optional tangent map, not a normal map.')
        layout.label(text='Empty direction follows UV V. Preview is approximate.')
    else:
        layout.prop(mat,'dbh_motion_blur')
        if not mat.dbh_motion_blur:layout.label(text='Blur exclusion requested. Re-export; loader support required.')
    layout.prop(mat,'dbh_face_mode')
    layout.label(text='Front / Back / Both visible faces (custom presets).')
    if not hair:
        layout.prop(mat,'dbh_diffuse_emission',slider=True)
        layout.label(text='0: engine lighting · 1: no engine shading')
        if not pbr:layout.label(text='Cel shadows and separate emission stay visible.')
    layout.label(text='Game post-processing still applies. Re-export changes.')
    if pbr:
        layout.prop(mat,'dbh_alpha_channel')
        layout.label(text='Opacity: 0 transparent · 1 opaque. Empty = opaque.')
        layout.label(text='ORM: R ambient occlusion · G roughness · B metallic.')
        if mat.dbh_shader_mode=='CLOTH':
            layout.prop(mat,'dbh_fabric_scale')
            layout.label(text='Fabric normal tiles on UV1; larger scale = finer weave.')
        layout.label(text='Blended mode: native transparency, 50% shadow cutoff.')
        layout.label(text='Dithered: matched opacity coverage in color/depth/velocity.')
        layout.label(text='Use Dithered for hair cards; slight stippling is possible.')
        layout.label(text='Opaque mode: no partial transparency; sharper with camera blur.')
        layout.label(text='Skin: PBR surface, no specialized skin scattering.')
    if mat.dbh_shader_mode in CEL_PRESETS:
        layout.prop(mat,'dbh_ambient_color')
        layout.label(text='Ambient fills/tints cel shadows; black = no fill.')
        layout.prop(mat,'dbh_uv_offset2')
        layout.label(text='X: cel row (V) · Y: ramp bias (U)')
        layout.label(text='Preview: studio sun; game: Detroit sun direction.')
        layout.label(text='Changes save in material; re-export to update game.')
    layout.template_list('MATERIAL_UL_dbh_textures','preset',mat,'dbh_preset_slots',mat,'dbh_preset_index',rows=3)
    row=layout.row(align=True);row.operator('material.dbh_add_slot',icon='ADD');row.operator('material.dbh_remove_slot',icon='REMOVE')
    if mat.dbh_preset_slots:
        slot=mat.dbh_preset_slots[min(mat.dbh_preset_index,len(mat.dbh_preset_slots)-1)]
        layout.label(text=LABELS[slot.role]+' texture')
        layout.template_ID(slot,'image',open='material.dbh_load_texture')
        layout.operator('material.dbh_load_texture',icon='FILE_FOLDER')
        layout.operator('material.dbh_new_texture',icon='ADD')
        if slot.image:
            layout.template_ID_preview(slot,'image',rows=3,cols=6,hide_buttons=True)
            layout.prop(slot.image.colorspace_settings,'name')
    layout.label(text='Empty slots use neutral textures; maps are embedded in SEGS.')
    layout.label(text='Export: enable Experimental Textures.',icon='INFO')
    if not pbr and not hair:layout.label(text='These presets replace skin/hair/glass effects with opaque shading.')


CLASSES=(MATERIAL_OT_dbh_new_material,MATERIAL_OT_dbh_add_slot,MATERIAL_OT_dbh_remove_slot)
