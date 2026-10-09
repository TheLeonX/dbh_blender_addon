"""Blender preview of the same F00A expression graph; studio sun approximation."""
from .cel_shader import graph,uv_values


def sync(mat,bsdf):
    import bpy
    tree=mat.node_tree
    group_node=tree.nodes.get('DBH Cel Preview') or tree.nodes.new('ShaderNodeGroup')
    group_node.name='DBH Cel Preview';group_node.label='F00A cel preview (studio sun)'
    group=group_node.node_tree
    if group is None:
        group=bpy.data.node_groups.new(mat.name+' F00A','ShaderNodeTree')
        for name,typ in [('Diffuse','NodeSocketColor'),('Normal','NodeSocketVector')]:
            group.interface.new_socket(name=name,in_out='INPUT',socket_type=typ)
        group.interface.new_socket(name='Color',in_out='OUTPUT',socket_type='NodeSocketColor')
        group_node.node_tree=group
    elif group.users>1:
        group=group.copy();group_node.node_tree=group
    # v044 always includes ambient nodes in the preview, including black.
    # Clear old generated internals once because positional names changed.
    if group.get('dbh_graph_revision')!=44:
        group.nodes.clear();group['dbh_graph_revision']=44
    nodes,links=group.nodes,group.links
    def node(typ,name):
        n=nodes.get(name) or nodes.new(typ);n.name=name;return n
    def connect(socket,value):
        if hasattr(value,'is_output'):links.new(value,socket)
        else:
            for l in list(socket.links):links.remove(l)
            socket.default_value=value
    def math(name,op,a,b=None):
        n=node('ShaderNodeMath',name);n.operation=op;connect(n.inputs[0],a)
        if b is not None:connect(n.inputs[1],b)
        return n.outputs[0]
    def vector(name,op,a,b=None):
        n=node('ShaderNodeVectorMath',name);n.operation=op;connect(n.inputs[0],a)
        if b is not None:connect(n.inputs[3] if op=='SCALE' else n.inputs[1],b)
        return n.outputs['Value' if op=='DOT_PRODUCT' else 'Vector']
    def combine(name,parts):
        n=node('ShaderNodeCombineXYZ',name)
        for i,x in enumerate(parts):connect(n.inputs[i],x)
        return n.outputs[0]
    def component(name,a,i):
        n=node('ShaderNodeSeparateXYZ',name);connect(n.inputs[0],a);return n.outputs[i]
    def choose(name,cond,a,b):
        n=node('ShaderNodeMixRGB',name);n.blend_type='MIX';n.use_clamp=False
        connect(n.inputs[0],cond);connect(n.inputs[1],b);connect(n.inputs[2],a);return n.outputs[0]
    source=node('NodeGroupInput','Inputs');target=node('NodeGroupOutput','Output')
    n=source.outputs['Normal']
    game_normal=combine('Game normal',(component('Normal X',n,0),component('Normal Z',n,2),
        math('Normal minusY','MULTIPLY',component('Normal Y',n,1),-1)))
    values={'normal':game_normal,'sun':combine('Studio sun',(0.4,-0.75,-0.5)),
            'diffuse':source.outputs['Diffuse']}
    slots={s.role:s for s in mat.dbh_preset_slots}
    image=slots.get('CEL').image if slots.get('CEL') else None
    expressions,result=graph(uv_values(mat.dbh_uv_offset2),mat.dbh_ambient_color,force_ambient=True)
    for name,typ,op,args in expressions:
        a=[values.get(x,x) if isinstance(x,str) else x for x in args]
        if op=='constant':
            n=node('ShaderNodeValue',name);n.outputs[0].default_value=a[0];value=n.outputs[0]
        elif op=='vector':value=combine(name,a)
        elif op=='extract':value=component(name,*a)
        elif op=='sample':
            n=node('ShaderNodeTexImage','Tone texture');n.image=image;n.interpolation='Linear';n.extension='REPEAT'
            # Exported images are top-down; Blender image coordinates bottom-up.
            u=component(name+' U',a[0],0);v=component(name+' V',a[0],1)
            coord=combine(name+' UV',(u,math(name+' Vflip','SUBTRACT',1,v),0))
            connect(n.inputs['Vector'],coord)
            if image:value=n.outputs['Color']
            else:value=combine(name+' White',(1,1,1))
        elif op=='choose':value=choose(name,*a)
        elif op=='mix':value=choose(name,a[2],a[1],a[0])
        elif op=='zero_power':
            parts=[]
            for i in range(3):
                light,exp,power=(component(name+str(i)+str(j),x,i) for j,x in enumerate(a))
                positive=math(name+str(i)+' light','GREATER_THAN',light,0)
                exponent=math(name+str(i)+' exp','GREATER_THAN',exp,0)
                zero=math(name+str(i)+' zero','SUBTRACT',1,exponent)
                parts.append(math(name+str(i)+' final','ADD',
                    math(name+str(i)+' lit','MULTIPLY',positive,power),
                    math(name+str(i)+' dark','MULTIPLY',math(name+str(i)+' inv','SUBTRACT',1,positive),zero)))
            value=combine(name,parts)
        elif op in ('round','pow') and typ=='v3':
            parts=[]
            for i in range(3):
                x=component(name+str(i)+' A',a[0],i)
                y=component(name+str(i)+' B',a[1],i) if len(a)>1 else None
                parts.append(math(name+str(i),'ROUND' if op=='round' else 'POWER',x,y))
            value=combine(name,parts)
        elif op=='clamp':
            fn=vector if typ=='v3' else math
            value=fn(name,'MINIMUM',fn(name+' max','MAXIMUM',a[0],a[1]),a[2])
        elif op=='negate':value=vector(name,'SCALE',a[0],-1)
        elif op=='dot':value=vector(name,'DOT_PRODUCT',*a)
        else:
            fn=vector if typ=='v3' else math
            operation={'add':'ADD','sub':'SUBTRACT','mul':'MULTIPLY','div':'DIVIDE','scale':'SCALE',
                       'min':'MINIMUM','max':'MAXIMUM','less':'LESS_THAN','greater':'GREATER_THAN',
                       'and':'MULTIPLY','inversesqrt':'INVERSE_SQRT'}[op]
            value=fn(name,operation,*a)
        values[name]=value
    links.new(values[result],target.inputs['Color'])
    # Original slot nodes remain editable/visible outside the generated group.
    diffuse=slots.get('COLOR');normal=slots.get('NORMAL')
    for link in list(group_node.inputs['Diffuse'].links):tree.links.remove(link)
    group_node.inputs['Diffuse'].default_value=(1,1,1,1)
    if diffuse and diffuse.image:tree.links.new(tree.nodes[diffuse.node_name].outputs['Color'],group_node.inputs['Diffuse'])
    normal_node=tree.nodes.get('DBH Preset Normal Decode')
    if normal and normal.image and normal_node:
        tree.links.new(normal_node.outputs['Normal'],group_node.inputs['Normal'])
    else:
        geometry=tree.nodes.get('DBH Cel Geometry') or tree.nodes.new('ShaderNodeNewGeometry')
        geometry.name='DBH Cel Geometry';tree.links.new(geometry.outputs['Normal'],group_node.inputs['Normal'])
    tree.links.new(group_node.outputs['Color'],bsdf.inputs['Base Color'])
    group_node.location=(-180,220)
