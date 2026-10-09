"""Skin/cloth PBR on the native alpha-blended 14B12 carrier.

No pipeline flag guessing: preserve this donor's reflection, render state,
vertex variants and four fragment interfaces. No skin-specific SSS is added.
"""
import hashlib
import math
import re
import struct
from .shader_profiles import fragment_variants
from .cel_shader import _instructions

PBR_PRESETS=('SKIN','CLOTH')
BINDINGS={'COLOR':2,'NORMAL':0,'ORM':3,'ALPHA':5,'FABRIC':1}
DEFAULTS={'COLOR':(1,1,1,1),'NORMAL':(.5,.5,1,1),'EMISSION':(0,0,0,1),
          'CEL':(1,1,1,1),'ORM':(1,.5,0,1),'ALPHA':(1,1,1,1),'FABRIC':(.5,.5,1,1),
          'TANGENT':(.5,1,.5,1),'HAIR_DAMAGE':(0,0,0,1),'HAIR_LOOKUP':(.5,.5,.5,1),'HAIR_SPEC':(.5,.5,.5,1)}
DATA_ROLES=('NORMAL','CEL','ORM','ALPHA','FABRIC','TANGENT','HAIR_DAMAGE','HAIR_LOOKUP','HAIR_SPEC')
PROFILES={
 (0,2):'eb9612aac2c3a4d9183343e2647703b5757ee4fcf6ed4fc902b5b96466a7b872',
 (0,3):'da4df6352ea75002c0cd874e5a65859ddbdd5aa609e0a0943825bfeda6c5a131',
 (2,2):'15ff9918a17b4b21949fa531e58a457bd74521c8092a75d1e1a678ed559f473d',
 (2,3):'cd928f4762fd8b6b08ea526999a4f8d8c6bb6a028c6d2206a193fc7ace3334c6',
 (4,2):'0216b600a1e11591564a4923ddd8e01361a44eed602ddd6ffb7ebbb02ef6d947',
 (4,3):'38fa206fe2da446a92bc3b34290eabaf303b2837a61db1f7505ab61b770ec8ec',
 (8,2):'608d5fc1c22c6134ba0ace979e51acb37a90ee14cd39a525e32abe680f695d15',
 (8,3):'91fdd36f0def18cfbf2cbf7b3741e19f3c4f724df4a9c687ebc42e21592d3e2c',
}


def settings(scale=10.0,channel='R'):
    scale=float(scale)
    if not math.isfinite(scale) or not .001<=scale<=10000:raise ValueError('Fabric Scale must be between 0.001 and 10000')
    if channel not in ('R','G','B','A'):raise ValueError('Choose R, G, B or A for opacity')
    return struct.unpack('<f',struct.pack('<f',scale))[0],channel


def detect(raw):
    marker=re.search(rb'// DBH_PBR_DITHER_V1 (SKIN|CLOTH) ([^\s]+) ([RGBA])',raw[:230000])
    if marker:return marker[1].decode(),*settings(float.fromhex(marker[2].decode()),marker[3].decode())
    marker=re.search(rb'// DBH_OPAQUE_PBR_V1 (SKIN|CLOTH) ([^\s]+) ([RGBA])',raw[:230000])
    if marker:return marker[1].decode(),*settings(float.fromhex(marker[2].decode()),marker[3].decode())
    marker=re.search(rb'// DBH_PBR_V1 (SKIN|CLOTH) ([^\s]+) ([RGBA])',raw[:230000])
    if not marker:return None
    return marker[1].decode(),*settings(float.fromhex(marker[2].decode()),marker[3].decode())


def detect_surface(raw):
    if b'// DBH_PBR_DITHER_V1 ' in raw[:230000]:return 'DITHERED'
    if b'// DBH_OPAQUE_PBR_V1 ' in raw[:230000]:return 'OPAQUE'
    if b'// DBH_PBR_V1 ' in raw[:230000]:return 'BLENDED'
    return 'AUTO'


def _op(code,*args):return [((len(args)+1)<<16)|code,*args]


def spirv(chunk,program,mode,scale,channel,emission,motion_blur=True):
    if chunk[:4]!=b'QDIF' or chunk[3944:3948]!=b'\x03\x02\x23\x07':raise ValueError('Unknown PBR QDIF')
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]));ins=list(_instructions(words))
    # f,vec2,vec3,vec4,bool,sampler pointer/type/array,material struct,uv0,
    # normal/base/rough/metal/alpha/cavity/alpha comparison.
    profiles={
      0:(8,9,51,43,36,3613,3607,3610,7100,10803,11339,11827,11526,11852,11665,11950,9118),
      8:(10,33,17,11,34,963,957,960,2538,5522,6058,6546,6245,6571,6384,6669,4583),
      2:(10,18,17,11,19,609,603,606,909,3091,0,0,0,0,3601,0,2955),
      4:(10,36,43,11,44,670,664,667,973,3527,0,0,0,0,4039,0,3018),
    }
    f,v2,v3,v4,boolean,ptr,st,arr,material,uv,normal,base,rough,metal,alpha,cavity,compare=profiles[program]
    next_id=words[3];constants=[];body=[];changes={}
    def new(typ,code,*args):
        nonlocal next_id
        n=next_id;next_id+=1;body.extend(_op(code,typ,n,*args));return n
    def const(value):
        nonlocal next_id
        n=next_id;next_id+=1;constants.extend(_op(43,f,n,struct.unpack('<I',struct.pack('<f',value))[0]));return n
    def ext(typ,code,*args):return new(typ,12,1,code,*args)
    zero,one,half,two=map(const,(0,1,.5,2))
    def splat(x):return new(v3,80,x,x,x)
    def sample(binding,coord):
        # S_MATERIAL members 0..8 are constants; s0 begins at member 9.
        idx=new(6,81,material,9+binding)
        sampler=new(st,61,new(ptr,65,arr,idx))
        return new(v4,87,sampler,coord)
    def extract(tex,c):return new(f,81,tex,c)
    def decode(tex):
        x=new(f,131,new(f,133,extract(tex,0),two),one)
        y=new(f,131,new(f,133,extract(tex,1),two),one)
        square=new(f,129,new(f,133,x,x),new(f,133,y,y))
        z=ext(f,31,new(f,131,one,ext(f,43,square,zero,one)))
        return x,y,z
    def replace(result,expected,new_words):changes[result]=(expected,new_words)
    opacity=ext(f,43,extract(sample(5,uv),'RGBA'.index(channel)),zero,one)
    if normal:
        color=sample(2,uv);orm=sample(3,uv)
        nx,ny,nz=decode(sample(0,uv))
        if mode=='CLOTH':
            tiled=new(v2,142,uv,const(scale));dx,dy,dz=decode(sample(1,tiled))
            nx=new(f,129,nx,dx);ny=new(f,129,ny,dy);nz=new(f,133,nz,dz)
        # Whiteout blend in tangent space; positive Z avoids zero-length vectors.
        nz=ext(f,40,nz,const(.00001))
        n=ext(v3,69,new(v3,80,nx,ny,nz))
        encoded=new(v3,129,new(v3,142,n,half),splat(half))
        body+=_op(83,v3,normal,encoded)
        replace(normal,12,body.copy());body.clear()
        replace(base,133,_op(79,v3,base,color,color,0,1,2))
        replace(rough,12,_op(81,f,rough,orm,1))
        replace(metal,12,_op(81,f,metal,orm,2))
        replace(cavity,12,_op(83,f,cavity,one))
        if program==0:
            ao=extract(orm,0)
            lit=new(v3,129,27237,new(v3,142,15804,ao))
            if emission:
                lit=ext(v3,46,lit,new(v3,79,color,color,0,1,2),splat(const(emission)))
            body+=_op(83,v3,17148,lit)
            replace(17148,129,body.copy());body.clear()
    else:
        # Depth/shadow and motion vectors retain the native alpha test, but
        # only high-coverage texels write depth. Color remains continuously blended.
        replace(alpha,81,body.copy()+_op(83,f,alpha,opacity));body.clear()
    if normal:replace(alpha,81,_op(83,f,alpha,opacity))
    # Color pass accepts every nonzero-opacity texel. When blur is excluded,
    # the velocity marker must cover the same alpha fringe; the native 0.5
    # cutout left semi-transparent hair edges to camera-motion reconstruction.
    velocity_fringe=program==4 and not motion_blur
    replace(compare,184,_op(188 if normal or velocity_fringe else 184,boolean,compare,alpha,
                            zero if normal or velocity_fringe else half))
    found=set();replacements={}
    for i,w in enumerate(ins):
        if len(w)>2 and w[0]&65535 in (12,81,129,133,184) and w[2] in changes:
            expected,value=changes[w[2]]
            if w[0]&65535!=expected:raise ValueError('PBR SSA anchor mismatch')
            replacements[i]=value;found.add(w[2])
    if found!=set(changes):raise ValueError('Incomplete PBR shader patch')
    extra=len(constants)+sum(len(v)-len(ins[i]) for i,v in replacements.items())
    removed=set();freed=0
    for i in reversed(range(len(ins))):
        if ins[i][0]&65535 in (5,6):
            removed.add(i);freed+=len(ins[i])
            if freed>=extra:break
    if freed<extra:raise ValueError('Insufficient PBR shader debug space')
    out=words[:5];out[3]=next_id;inserted=False;entry=False;padded=False
    for i,w in enumerate(ins):
        if i in removed:continue
        op=w[0]&65535
        if op==54 and not inserted:out+=constants;inserted=True
        if entry and not padded and op not in (59,8,317):out+=[1<<16]*(freed-extra);padded=True
        out+=replacements.get(i,w)
        if op==248 and not entry:entry=True
    if len(out)!=len(words):raise ValueError('PBR changed wrapped shader size')
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)


def glsl(chunk,program,mode,scale,channel,emission,motion_blur=True):
    def tex(binding,uv='Surface.fUv0'):
        return f'texture(sampler2D(g_rb2DTextures[Material.s{binding} + glslHack].rHandle.xy), {uv})'
    def assignment(name,value):
        nonlocal chunk
        chunk,n=re.subn((r'MaterialAttributes\.'+name+r' = g_v\d+;').encode(),
                        ('MaterialAttributes.'+name+' = '+value+';').encode(),chunk)
        if n!=1:raise ValueError('Unknown PBR GLSL '+name)
    opacity='clamp('+tex(5)+'.'+channel.lower()+',0.0,1.0)'
    assignment('transparency',opacity)
    velocity_fringe=program==4 and not motion_blur
    assignment('alphaTestValue','0.0' if program in (0,8) or velocity_fringe else '0.5')
    if program in (0,8):
        for name,value in {'baseColor':tex(2)+'.rgb','roughness':tex(3)+'.g','metallic':tex(3)+'.b',
                           'cavity':'1.0','emissiveColor':'vec3(0.0)'}.items():assignment(name,value)
        code='vec2 dbhNxy = '+tex(0)+'.xy * 2.0 - 1.0;\n'
        code+='vec3 dbhN = vec3(dbhNxy, sqrt(1.0-clamp(dot(dbhNxy,dbhNxy),0.0,1.0)));\n'
        if mode=='CLOTH':
            code+='vec2 dbhDxy = '+tex(1,'Surface.fUv0 * '+format(scale,'.9g')+'')+'.xy * 2.0 - 1.0;\n'
            code+='vec3 dbhD = vec3(dbhDxy, sqrt(1.0-clamp(dot(dbhDxy,dbhDxy),0.0,1.0)));\n'
            code+='dbhN = vec3(dbhN.xy + dbhD.xy, dbhN.z * dbhD.z);\n'
        code+='dbhN.z = max(dbhN.z,0.00001);\nMaterialAttributes.normalMap = normalize(dbhN)*0.5+0.5;'
        anchor=b'MaterialAttributes.normalMap = g_v0;'
        if chunk.count(anchor)!=1:raise ValueError('Unknown PBR normal expression')
        chunk=chunk.replace(anchor,code.encode())
        chunk=chunk.replace(b'MaterialAttributes.transparency < MaterialAttributes.alphaTestValue',
                            b'MaterialAttributes.transparency <= MaterialAttributes.alphaTestValue')
    elif velocity_fringe:
        anchor=b'MaterialAttributes.transparency < fAlphaTestValue'
        if chunk.count(anchor)!=1:raise ValueError('Unknown PBR velocity alpha test')
        chunk=chunk.replace(anchor,b'MaterialAttributes.transparency <= fAlphaTestValue')
        if program==0:
            anchor=b'AddLightContrib(colorAccum, colorAccumIBL);'
            if chunk.count(anchor)!=1:raise ValueError('Unknown PBR IBL expression')
            # ComputeClusterIllumination has no Material argument. Capture the
            # per-fragment map in ComputeHypershade before entering lighting.
            declaration=b'S_MATERIAL_ATTRIBUTES ComputeHypershade('
            chunk=chunk.replace(declaration,b'float dbhOcclusion;\n'+declaration,1)
            anchor_color=('MaterialAttributes.baseColor = '+tex(2)+'.rgb;').encode()
            chunk=chunk.replace(anchor_color,anchor_color+('\ndbhOcclusion = '+tex(3)+'.r;').encode())
            chunk=chunk.replace(anchor,b'colorAccumIBL.fLighting *= dbhOcclusion;\n'+anchor)
            if emission:
                anchor=b'vec3  vfColor = LightingResult.fLighting + (MaterialAttributes.emissiveColor * fEmissiveIntensity);'
                if chunk.count(anchor)!=1:raise ValueError('Unknown PBR lighting expression')
                chunk=chunk.replace(anchor,('vec3 vfColor = mix(LightingResult.fLighting, MaterialAttributes.baseColor, '+format(emission,'.9g')+');').encode())
    if chunk[-1:]!=b'\0':raise ValueError('Unterminated PBR GLSL')
    marker=f'\n// DBH_PBR_V1 {mode} {scale.hex()} {channel}\n'
    if emission:marker+='// DBH_DIFFUSE_EMISSION_V1 '+emission.hex()+'\n'
    return chunk[:-1]+marker.encode()+b'\0'


def build_shader(template,raw,mode,scale=10,channel='R',emission=0,motion_blur=True,face_mode='BOTH'):
    from .lighting_control import emission_value
    scale,channel=settings(scale,channel);emission=emission_value(emission)
    if mode not in PBR_PRESETS:raise ValueError('Unknown PBR preset')
    if template.asset!=0x14B12 or template.payload[44:52]!=struct.pack('<II',1,1):raise ValueError('Unknown PBR transparent carrier')
    data=bytearray(template.payload);out=bytearray();found=set()
    for program,typ,hash_at,at,n in fragment_variants(template,raw):
        chunk=raw[at:at+n];key=program,typ
        if hashlib.sha256(chunk).hexdigest()!=PROFILES.get(key):raise ValueError('Unknown PBR carrier revision')
        found.add(key)
        chunk=(spirv if typ==3 else glsl)(chunk,program,mode,scale,channel,emission,motion_blur)
        if face_mode!='BOTH':
            from .outline_shader import spirv as face_spirv, glsl as face_glsl
            chunk=face_spirv(chunk,face_mode) if typ==3 else face_glsl(chunk,face_mode,outline=False)
        if not motion_blur and program in (0,4):
            from .motion_control import patch
            chunk=patch(chunk,program,typ,pbr=True)
        struct.pack_into('<I',data,hash_at-8,len(chunk))
        data[hash_at:hash_at+16]=hashlib.sha256(b'DBH_PBR_V1'+mode.encode()+chunk).digest()[:16]
        from .shader_profiles import append_shader
        append_shader(out,chunk)
    if found!=set(PROFILES):raise ValueError('Incomplete transparent carrier')
    out+=b'\0'*(-len(out)%128)
    return bytes(data),bytes(out)


def upgrade_velocity_fringe(record,raw):
    """Repair older Motion Blur OFF PBR exports whose alpha fringe had no marker."""
    if detect_surface(raw) in ('OPAQUE','DITHERED'):return raw
    from .motion_control import detect_enabled
    mode=detect(raw)
    if mode is None or detect_enabled(raw):return raw
    variants=fragment_variants(record,raw)
    glsl=next((raw[at:at+n] for program,typ,_,at,n in variants if (program,typ)==(4,2)),None)
    spirv_chunk=next((raw[at:at+n] for program,typ,_,at,n in variants if (program,typ)==(4,3)),None)
    if glsl is None or spirv_chunk is None:raise ValueError('Incomplete PBR velocity variants')
    old_value=b'MaterialAttributes.alphaTestValue = 0.5;'
    new_value=b'MaterialAttributes.alphaTestValue = 0.0;'
    if new_value in glsl:
        if old_value in glsl:raise ValueError('Conflicting PBR velocity alpha thresholds')
        return raw
    if glsl.count(old_value)!=1 or glsl.count(b'MaterialAttributes.transparency < fAlphaTestValue')!=1:
        raise ValueError('Unknown legacy PBR velocity alpha coverage')
    if spirv_chunk[:4]!=b'QDIF' or spirv_chunk[3944:3948]!=b'\x03\x02\x23\x07':
        raise ValueError('Unknown legacy PBR velocity QDIF')
    words=list(struct.unpack('<'+'I'*((len(spirv_chunk)-3944)//4),spirv_chunk[3944:]))
    ins=list(_instructions(words))
    constants={w[2]:w for w in ins if w[0]&65535==43 and len(w)==4}
    if constants.get(4131,[None]*4)[3]!=0 or constants.get(4133,[None]*4)[3]!=0x3f000000:
        raise ValueError('Unknown legacy PBR alpha-test constants')
    matching=[i for i,w in enumerate(ins) if len(w)==5 and w[2:]==[3018,4039,4133] and w[0]&65535==184]
    if len(matching)!=1:raise ValueError('Unknown legacy PBR velocity alpha comparison')
    ins[matching[0]]=[5<<16|188,ins[matching[0]][1],3018,4039,4131]
    changed_words=words[:5]+[word for row in ins for word in row]
    changed_spirv=spirv_chunk[:3944]+struct.pack('<'+'I'*len(changed_words),*changed_words)
    if len(changed_spirv)!=len(spirv_chunk):raise ValueError('PBR velocity migration changed QDIF size')
    changed_glsl=glsl.replace(old_value,new_value).replace(
        b'MaterialAttributes.transparency < fAlphaTestValue',
        b'MaterialAttributes.transparency <= fAlphaTestValue')
    if changed_glsl[-1:]!=b'\0':raise ValueError('Unterminated legacy PBR GLSL')
    changed_glsl=changed_glsl[:-1]+b'\n// DBH_ALPHA_VELOCITY_V2\n\0'
    out=bytearray();payload=bytearray(record.payload)
    from .shader_profiles import append_shader
    for program,typ,hash_at,at,size in variants:
        chunk=raw[at:at+size]
        if (program,typ)==(4,2):chunk=changed_glsl
        elif (program,typ)==(4,3):chunk=changed_spirv
        if program==4:
            struct.pack_into('<I',payload,hash_at-8,len(chunk))
            payload[hash_at:hash_at+16]=hashlib.sha256(b'DBH_ALPHA_VELOCITY_V2'+chunk).digest()[:16]
        append_shader(out,chunk)
    out+=b'\0'*(-len(out)%128)
    record.payload=bytes(payload)
    return bytes(out)
