"""Fade engine lighting, retaining cel color and independent emission.

Zero is deliberately byte-identical to v0.4.2. No descriptors, QDIF reflection,
render targets, branches or resource layouts are added. Pass 8 is a material
debug pass, not the lit rendering pass, and is left unchanged.
"""
import math
import re
import struct


def emission_value(value):
    value=float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Diffuse emission must be a finite value between 0 and 1')
    return struct.unpack('<f',struct.pack('<f',value))[0]


def detect_emission(raw):
    marker=re.search(rb'// DBH_DIFFUSE_EMISSION_V1 ([^\s]+)',raw[:230000])
    return emission_value(float.fromhex(marker.group(1).decode())) if marker else 0.0


def glsl(chunk,program,value,cel=False):
    value=emission_value(value)
    if not value or program!=0:return chunk
    anchor=b'vec3  light = LightingResult.fLighting + MaterialAttributes.emissiveColor * fEmissiveIntensity;'
    if chunk.count(anchor)!=1:raise ValueError('Diffuse emission GLSL lighting anchor mismatch')
    t=format(value,'.9g');t=t if any(c in t for c in '.eE') else t+'.0'
    target='dbhCelColor' if cel else 'g_v640.rgb'
    if cel:
        declaration=b'S_MATERIAL_ATTRIBUTES ComputeHypershade('
        if chunk.count(declaration)!=1:raise ValueError('Cel color capture declaration mismatch')
        chunk=chunk.replace(declaration,b'vec3 dbhCelColor;\n'+declaration)
    emission='MaterialAttributes.emissiveColor * fEmissiveIntensity'
    if value==1:
        replacement=f'vec3  light = {target} + {emission};\n vGIandSSRSpecularFactor = vec3(0.0);'
    else:
        replacement=(f'vec3  light = mix(LightingResult.fLighting, {target}, {t}) + {emission};'
                     f'\n vGIandSSRSpecularFactor *= (1.0 - {t});')
    return chunk.replace(anchor,replacement.encode())


def spirv(chunk,program,value,cel=False):
    from .cel_shader import _instructions
    value=emission_value(value)
    if not value or program!=0:return chunk
    if chunk[:4]!=b'QDIF' or chunk[3944:3948]!=b'\x03\x02\x23\x07':
        raise ValueError('Unknown diffuse emission carrier')
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]))
    ins=list(_instructions(words));next_id=words[3];constants=[];patches={}
    def fresh():
        nonlocal next_id
        result=next_id;next_id+=1;return result
    def op(code,*args):return [((len(args)+1)<<16)|code,*args]
    def const(v):
        result=fresh();constants.extend(op(43,8,result,struct.unpack('<I',struct.pack('<f',v))[0]));return result
    # Verified IDs from the fingerprinted coat carrier. 22057 feeds out_color0;
    # 39979 feeds out_color2's later GI/SSR specular contribution.
    expected={22057:op(129,46,22057,39424,22056),
              39979:op(12,46,39979,1,46,1773,39447,39418)}
    for result,original in expected.items():
        matches=[i for i,w in enumerate(ins) if w==original]
        if len(matches)!=1:raise ValueError('Diffuse emission SSA lighting anchor mismatch')
        at=matches[0]
        if result==22057:
            # Cel base30081 is the pre-rain F00A result, not raw sample24453.
            # Preserve independent emission22056 at every slider value.
            target=30081 if cel else fresh()
            replacement=[] if cel else op(79,46,target,24453,24453,0,1,2)
            if value==1:replacement+=op(129,46,result,target,22056)
            else:
                t=const(value);weight=fresh();lit=fresh()
                replacement+=op(80,46,weight,t,t,t)
                replacement+=op(12,46,lit,1,46,39424,target,weight)
                replacement+=op(129,46,result,lit,22056)
            patches[at]=replacement
        elif value==1:
            patches[at]=op(83,46,result,1773)
        else:
            previous=fresh();replacement=original.copy();replacement[2]=previous
            replacement+=op(142,46,result,previous,const(1-value))
            patches[at]=replacement
    extra=len(constants)+sum(len(v)-len(ins[i]) for i,v in patches.items())
    removed=set();freed=0
    if extra>0:
        for i in reversed(range(len(ins))):
            if ins[i][0]&65535 in (5,6):
                removed.add(i);freed+=len(ins[i])
                if freed>=extra:break
    if freed<extra:raise ValueError('Insufficient debug space for diffuse emission')
    out=words[:5];out[3]=next_id;inserted=False;padded=False;entry=False
    for i,w in enumerate(ins):
        if i in removed:continue
        code=w[0]&65535
        if code==54 and not inserted:out+=constants;inserted=True
        if entry and not padded and code not in (59,8,317):
            out+=[1<<16]*(freed-extra);padded=True
        out+=patches.get(i,w)
        if code==248 and not entry:entry=True
    if len(out)!=len(words):raise ValueError('Diffuse emission changed wrapped size')
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)


def preview(mat,bsdf,out):
    """Same final blend as the game, not additive emission on a lit BSDF."""
    from .cel_shader import CEL_PRESETS
    tree=mat.node_tree
    def node(typ,name):
        n=tree.nodes.get(name) or tree.nodes.new(typ);n.name=name;return n
    emission=node('ShaderNodeEmission','DBH Diffuse Unlit')
    emission.inputs['Strength'].default_value=1
    color=emission.inputs['Color']
    for link in list(color.links):tree.links.remove(link)
    color.default_value=(1,1,1,1)
    slot=next((s for s in mat.dbh_preset_slots if s.role=='COLOR'),None)
    texture=tree.nodes.get(slot.node_name) if slot and slot.image else None
    if texture:tree.links.new(texture.outputs['Color'],color)
    if mat.dbh_shader_mode in CEL_PRESETS:
        tree.links.new(tree.nodes['DBH Cel Preview'].outputs['Color'],color)
    # Add the separate emission map to the self-lit branch too, so mixing does
    # not fade that map. The lit branch already includes it in the Principled.
    add=node('ShaderNodeMixRGB','DBH Cel Color Plus Emission');add.blend_type='ADD';add.use_clamp=False
    add.inputs[0].default_value=1
    base_link=color.links[0] if color.is_linked else None
    for socket in (add.inputs[1],add.inputs[2]):
        for link in list(socket.links):tree.links.remove(link)
    add.inputs[1].default_value=color.default_value
    add.inputs[2].default_value=(0,0,0,1)
    if base_link:tree.links.new(base_link.from_socket,add.inputs[1])
    slot=next((s for s in mat.dbh_preset_slots if s.role=='EMISSION'),None)
    texture=tree.nodes.get(slot.node_name) if slot and slot.image else None
    if texture:tree.links.new(texture.outputs['Color'],add.inputs[2])
    tree.links.new(add.outputs[0],color)
    mix=node('ShaderNodeMixShader','DBH Diffuse Lighting Blend')
    mix.inputs[0].default_value=emission_value(mat.dbh_diffuse_emission)
    tree.links.new(bsdf.outputs['BSDF'],mix.inputs[1])
    tree.links.new(emission.outputs[0],mix.inputs[2])
    tree.links.new(mix.outputs[0],out.inputs['Surface'])
    emission.location=(0,-260);mix.location=(320,180);out.location=(550,180)
