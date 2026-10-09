"""Opt-in, non-destructive inverted hull, baked by the normal mesh exporter.

No extra unbound object is created. Join Geometry keeps original faces and
attributes; the expanded copy reverses winding and uses an ordinary opaque
unlit preset with an embedded one-color image. This is not a screen-space edge
detector. Export/reimport produces a separate native material draw for the hull.
"""
import bpy
from bpy.props import BoolProperty,FloatProperty,FloatVectorProperty,PointerProperty
from bpy.app.handlers import persistent


def _modifier(obj):
    return next((m for m in obj.modifiers if m.type=='NODES' and m.node_group and
                 m.node_group.get('dbh_outline_graph')),None)


def cull_preview(material):
    material.use_backface_culling=True
    tree=material.node_tree
    def node(typ,name):
        n=tree.nodes.get(name) or tree.nodes.new(typ);n.name=name;return n
    geometry=node('ShaderNodeNewGeometry','DBH Outline Facing')
    transparent=node('ShaderNodeBsdfTransparent','DBH Outline Backface')
    cull=node('ShaderNodeMixShader','DBH Outline Cull')
    tree.links.new(geometry.outputs['Backfacing'],cull.inputs[0])
    tree.links.new(tree.nodes['DBH Diffuse Lighting Blend'].outputs[0],cull.inputs[1])
    tree.links.new(transparent.outputs[0],cull.inputs[2])
    output=next(n for n in tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output)
    tree.links.new(cull.outputs[0],output.inputs['Surface'])


def sync(obj,context=None):
    if obj.type!='MESH':return
    modifier=_modifier(obj)
    if not obj.dbh_outline_enabled:
        if modifier:obj.modifiers.remove(modifier)
        return
    if obj.data.users>1:obj.data=obj.data.copy()
    material=obj.dbh_outline_material
    previous=material
    if material is None or material.get('dbh_outline_owner')!=obj.name:
        material=material.copy() if material else bpy.data.materials.new(obj.name+' Outline')
        material['dbh_outline_owner']=obj.name;material['dbh_outline_generated']=True
        obj.dbh_outline_material=material
        if previous:
            for i,m in enumerate(obj.data.materials):
                if m==previous:obj.data.materials[i]=material
    material.use_nodes=True
    material.dbh_shader_mode='STANDARD_DIFFUSE';material.dbh_diffuse_emission=1
    material.use_backface_culling=True
    slot=next(s for s in material.dbh_preset_slots if s.role=='COLOR')
    image=slot.image
    if image is None or image.get('dbh_outline_owner')!=obj.name:
        image=bpy.data.images.new(obj.name+' Outline Color',width=4,height=4,float_buffer=True)
        image['dbh_outline_owner']=obj.name;image.colorspace_settings.name='sRGB'
        slot.image=image
    image.pixels.foreach_set([*obj.dbh_outline_color,1.0]*16);image.update();image.pack()
    from .preset_ui import sync_preview
    sync_preview(material)
    if material.name not in obj.data.materials:obj.data.materials.append(material)
    if modifier is None:modifier=obj.modifiers.new('Detroit Outline','NODES')
    group=modifier.node_group
    if group is None:
        group=bpy.data.node_groups.new(obj.name+' Outline Hull','GeometryNodeTree')
        group.interface.new_socket(name='Geometry',in_out='INPUT',socket_type='NodeSocketGeometry')
        group.interface.new_socket(name='Geometry',in_out='OUTPUT',socket_type='NodeSocketGeometry')
        group['dbh_outline_graph']=True;modifier.node_group=group
    elif group.users>1:
        group=group.copy();modifier.node_group=group
    nodes,links=group.nodes,group.links
    def geo(typ,name):
        n=nodes.get(name) or nodes.new(typ);n.name=name;return n
    source=geo('NodeGroupInput','Original Mesh');target=geo('NodeGroupOutput','Outlined Mesh')
    normal=geo('GeometryNodeInputNormal','Outline Normal')
    scale=geo('ShaderNodeVectorMath','Outline Thickness');scale.operation='SCALE'
    scale.inputs['Scale'].default_value=obj.dbh_outline_width
    move=geo('GeometryNodeSetPosition','Expand Hull')
    flip=geo('GeometryNodeFlipFaces','Reverse Hull')
    assign=geo('GeometryNodeSetMaterial','Outline Material');assign.inputs['Material'].default_value=material
    join=geo('GeometryNodeJoinGeometry','Original And Hull')
    # Reconnect without accumulating duplicate links on a multi-input socket.
    links.clear()
    links.new(normal.outputs['Normal'],scale.inputs[0])
    links.new(source.outputs['Geometry'],move.inputs['Geometry'])
    links.new(scale.outputs['Vector'],move.inputs['Offset'])
    links.new(move.outputs['Geometry'],flip.inputs['Mesh'])
    links.new(flip.outputs['Mesh'],assign.inputs['Geometry'])
    links.new(assign.outputs['Geometry'],join.inputs['Geometry'])
    links.new(source.outputs['Geometry'],join.inputs['Geometry'])
    links.new(join.outputs['Geometry'],target.inputs['Geometry'])


@persistent
def refresh(_=None):
    for obj in bpy.data.objects:
        if obj.type=='MESH' and obj.dbh_outline_enabled:
            try:sync(obj)
            except Exception as error:print(f'Detroit outline refresh ({obj.name}): {error}')


def draw(layout,obj):
    if not obj or obj.type!='MESH':return
    box=layout.box();box.prop(obj,'dbh_outline_enabled')
    if obj.dbh_outline_enabled:
        box.prop(obj,'dbh_outline_color');box.prop(obj,'dbh_outline_width')
        if obj.dbh_outline_material:box.prop(obj.dbh_outline_material,'dbh_motion_blur',text='Outline Motion Blur')
        box.label(text='Inverted hull · thickness in local mesh units')
        box.label(text='Export: Experimental Textures; bone-skin edited cloth.')


class OBJECT_PT_dbh_outline(bpy.types.Panel):
    bl_label='Detroit Outline';bl_idname='OBJECT_PT_dbh_outline'
    bl_space_type='PROPERTIES';bl_region_type='WINDOW';bl_context='object'
    @classmethod
    def poll(cls,context):return context.object and context.object.type=='MESH'
    def draw(self,context):draw(self.layout,context.object)


def register():
    bpy.types.Object.dbh_outline_enabled=BoolProperty(name='Enable Outline',default=False,update=sync,
        description='Add an expanded inverted hull without changing the base mesh. Exports as an extra skinned draw; experimental in game')
    bpy.types.Object.dbh_outline_color=FloatVectorProperty(name='Outline Color',size=3,subtype='COLOR',default=(0,0,0),min=0,max=1,update=sync)
    bpy.types.Object.dbh_outline_width=FloatProperty(name='Outline Thickness',default=.002,min=.00001,soft_max=.05,precision=4,subtype='DISTANCE',update=sync)
    bpy.types.Object.dbh_outline_material=PointerProperty(type=bpy.types.Material)
    bpy.utils.register_class(OBJECT_PT_dbh_outline)
    bpy.app.handlers.load_post.append(refresh)
    if hasattr(bpy.data,'objects'):refresh()
    elif not bpy.app.timers.is_registered(refresh):bpy.app.timers.register(refresh,first_interval=.2)


def unregister():
    if refresh in bpy.app.handlers.load_post:bpy.app.handlers.load_post.remove(refresh)
    if bpy.app.timers.is_registered(refresh):bpy.app.timers.unregister(refresh)
    bpy.utils.unregister_class(OBJECT_PT_dbh_outline)
    for name in ('dbh_outline_enabled','dbh_outline_color','dbh_outline_width','dbh_outline_material'):delattr(bpy.types.Object,name)
