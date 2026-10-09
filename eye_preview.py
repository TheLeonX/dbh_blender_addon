"""Approximate additive preview; the game keeps native projected eye shading."""
def sync(mat):
    if not mat.use_nodes or not mat.node_tree:return
    nodes=mat.node_tree.nodes;links=mat.node_tree.links
    bsdf=nodes.get('Principled BSDF')
    if bsdf is None:return
    node=nodes.get('DBH Eye Emission') or nodes.new('ShaderNodeTexImage');node.name='DBH Eye Emission'
    node.label='Eye Emission (game uses projected eye UVs)';node.image=mat.dbh_eye_emission_image
    uv=nodes.get('DBH Eye Emission UV') or nodes.new('ShaderNodeUVMap');uv.name='DBH Eye Emission UV';uv.uv_map='UV1'
    links.new(uv.outputs['UV'],node.inputs['Vector'])
    if node.image:links.new(node.outputs['Color'],bsdf.inputs['Emission Color'])
    else:
        for link in list(bsdf.inputs['Emission Color'].links):links.remove(link)
        bsdf.inputs['Emission Color'].default_value=(0,0,0,1)
    bsdf.inputs['Emission Strength'].default_value=mat.dbh_eye_emission_strength
    node.location=(-480,-400);uv.location=(-680,-400)

def changed(mat,context):
    if mat.dbh_shader_mode=='EYE_EMISSION':sync(mat)

def remove(mat):
    if not mat.node_tree:return
    bsdf=mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        for link in list(bsdf.inputs['Emission Color'].links):
            if link.from_node.name=='DBH Eye Emission':mat.node_tree.links.remove(link)
        if not bsdf.inputs['Emission Color'].is_linked:bsdf.inputs['Emission Color'].default_value=(0,0,0,1)
