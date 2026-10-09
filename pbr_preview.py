"""Generated Blender PBR preview; engine illumination is not reproduced."""
from .pbr_shader import PBR_PRESETS


def sync(mat,bsdf):
    from .preset_ui import _node
    links=mat.node_tree.links
    def node(typ,name):return _node(mat,typ,'DBH PBR '+name)
    def math(name,op,a,b=None):
        n=node('ShaderNodeMath',name);n.operation=op
        for i,x in enumerate((a,b)):
            for link in list(n.inputs[i].links):links.remove(link)
            if x is None:continue
            if isinstance(x,(int,float)):n.inputs[i].default_value=x
            else:links.new(x,n.inputs[i])
        return n.outputs[0]
    images={s.role:mat.node_tree.nodes.get(s.node_name) for s in mat.dbh_preset_slots if s.image}
    orm=images.get('ORM')
    if orm:
        split=node('ShaderNodeSeparateColor','ORM Channels');links.new(orm.outputs['Color'],split.inputs['Color'])
        links.new(split.outputs['Green'],bsdf.inputs['Roughness']);links.new(split.outputs['Blue'],bsdf.inputs['Metallic'])
        # AO belongs to environment lighting in the native shader. Principled
        # has no AO input; expose its R output, don't darken direct light too.
        split.label='R: game ambient occlusion · G: roughness · B: metallic'
    def decoded(role):
        image=images.get(role)
        if not image:return (0,0,1)
        if role=='FABRIC':
            uv=mat.node_tree.nodes.get('DBH Preset UV FABRIC')
            scale=node('ShaderNodeVectorMath','Fabric UV Scale');scale.operation='SCALE'
            scale.inputs['Scale'].default_value=mat.dbh_fabric_scale
            links.new(uv.outputs['UV'],scale.inputs[0]);links.new(scale.outputs['Vector'],image.inputs['Vector'])
        sep=node('ShaderNodeSeparateColor',role+' XY');links.new(image.outputs['Color'],sep.inputs['Color'])
        x=math(role+' X','SUBTRACT',math(role+' X2','MULTIPLY',sep.outputs['Red'],2),1)
        y=math(role+' Y','SUBTRACT',math(role+' Y2','MULTIPLY',sep.outputs['Green'],2),1)
        squared=math(role+' XY Square','ADD',math(role+' XX','MULTIPLY',x,x),math(role+' YY','MULTIPLY',y,y))
        z=math(role+' Z','SQRT',math(role+' Z Positive','MAXIMUM',math(role+' Z Square','SUBTRACT',1,squared),0))
        return x,y,z
    xyz=decoded('NORMAL')
    if mat.dbh_shader_mode=='CLOTH':
        detail=decoded('FABRIC')
        xyz=tuple(math('Whiteout '+axis,op,a,b) for axis,op,a,b in zip('XYZ',('ADD','ADD','MULTIPLY'),xyz,detail))
    xyz=(*xyz[:2],math('Normal Z Safe','MAXIMUM',xyz[2],.00001))
    combine=node('ShaderNodeCombineXYZ','Tangent Normal')
    for i,x in enumerate(xyz):
        for link in list(combine.inputs[i].links):links.remove(link)
        if isinstance(x,(int,float)):combine.inputs[i].default_value=x
        else:links.new(x,combine.inputs[i])
    norm=node('ShaderNodeVectorMath','Normalize');norm.operation='NORMALIZE';links.new(combine.outputs[0],norm.inputs[0])
    half=node('ShaderNodeVectorMath','Encode Scale');half.operation='SCALE';half.inputs['Scale'].default_value=.5
    links.new(norm.outputs[0],half.inputs[0])
    offset=node('ShaderNodeVectorMath','Encode Bias');offset.operation='ADD';offset.inputs[1].default_value=(.5,.5,.5)
    links.new(half.outputs[0],offset.inputs[0])
    normal=node('ShaderNodeNormalMap','Normal Decode');normal.uv_map='UV1'
    links.new(offset.outputs[0],normal.inputs['Color']);links.new(normal.outputs['Normal'],bsdf.inputs['Normal'])


def transparency(mat,out,bsdf):
    from .preset_ui import _node
    links=mat.node_tree.links;images={s.role:mat.node_tree.nodes.get(s.node_name) for s in mat.dbh_preset_slots if s.image}
    image=images.get('ALPHA')
    # Blender's sorted BLENDED mode plus a Transparent/Principled Mix produces
    # an extra translucent layer and inverted-looking backfaces on hair.
    # Drive the Principled alpha directly and use stable dithered coverage.
    if hasattr(mat,'surface_render_method'):mat.surface_render_method='DITHERED'
    elif hasattr(mat,'blend_method'):mat.blend_method='HASHED'
    if not image:return
    opacity=image.outputs['Alpha']
    if mat.dbh_alpha_channel!='A':
        sep=_node(mat,'ShaderNodeSeparateColor','DBH PBR Opacity Channels')
        links.new(image.outputs['Color'],sep.inputs['Color']);opacity=sep.outputs[{'R':'Red','G':'Green','B':'Blue'}[mat.dbh_alpha_channel]]
    links.new(opacity,bsdf.inputs['Alpha'])
    # The diffuse-emission branch bypasses Principled. Give it the same
    # coverage so increasing emission never reveals the whole hair card.
    lighting=mat.node_tree.nodes.get('DBH Diffuse Lighting Blend')
    emission=mat.node_tree.nodes.get('DBH Diffuse Unlit')
    if lighting and emission:
        transparent=_node(mat,'ShaderNodeBsdfTransparent','DBH PBR Unlit Transparent')
        mix=_node(mat,'ShaderNodeMixShader','DBH PBR Unlit Opacity')
        links.new(opacity,mix.inputs[0])
        links.new(transparent.outputs[0],mix.inputs[1])
        links.new(emission.outputs[0],mix.inputs[2])
        links.new(mix.outputs[0],lighting.inputs[2])
