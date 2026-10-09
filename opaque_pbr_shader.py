"""Opaque Skin/Cloth carrier for surfaces whose opacity is fully white.

Uses the fingerprinted deferred coat interface; never changes render-state bits
on an incompatible transparent shader. The existing blended PBR path remains
available for genuinely translucent surfaces.
"""
import hashlib
import struct

from .cel_shader import _instructions
from .native import Record
from .shader_profiles import append_shader,fragment_variants

MARKER=b'// DBH_OPAQUE_PBR_V1 '
BINDINGS={'COLOR':21,'NORMAL':6,'ORM':5,'FABRIC':7}
PROFILES={
    0:dict(f=8,v2=9,v3=46,v4=43,zero=1062,one=1063,half=1170,
           two=1107,uv=22315,orm=22928,normal=22962,fabric=22975,
           rough=24105,metal=30101,result=23207),
    8:dict(f=10,v2=33,v3=17,v4=11,zero=385,one=386,half=493,
           two=430,uv=17747,orm=18360,normal=18394,fabric=18407,
           rough=19537,metal=25533,result=18639),
}


def _op(code,*args):return [((len(args)+1)<<16)|code,*args]


def spirv(chunk,program,mode,scale,emission):
    if chunk[:4]!=b'QDIF' or chunk[3944:3948]!=b'\x03\x02\x23\x07':
        raise ValueError('Unknown opaque PBR shader wrapper')
    ids=PROFILES[program]
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]))
    instructions=list(_instructions(words))
    next_id=words[3];constants=[];patches={}
    def fresh():
        nonlocal next_id
        result=next_id;next_id+=1;return result
    def emit(out,code,typ,*args):
        result=fresh();out+=_op(code,typ,result,*args);return result
    def const(value):
        result=fresh();constants.extend(_op(43,ids['f'],result,struct.unpack('<I',struct.pack('<f',float(value)))[0]));return result
    def ext(out,typ,code,*args):return emit(out,12,typ,1,code,*args)
    anchors={
        ids['rough']:(83,ids['half'],1),
        ids['metal']:(83,ids['zero'],2),
    }
    found=set()
    for i,w in enumerate(instructions):
        if len(w)<3:continue
        result=w[2]
        if result in anchors and w[0]&65535==83:
            opcode,source,channel=anchors[result]
            if w!=_op(opcode,ids['f'],result,source):raise ValueError('Opaque PBR ORM anchor mismatch')
            patches[i]=_op(81,ids['f'],result,ids['orm'],channel)
            found.add(result)
    if found!=set(anchors):raise ValueError('Incomplete opaque PBR ORM patch')
    if mode=='CLOTH':
        sample=next((i for i,w in enumerate(instructions) if len(w)>=3 and w[2]==ids['fabric']),None)
        target=next((i for i,w in enumerate(instructions) if len(w)>=3 and w[2]==ids['result']),None)
        if sample is None or target is None or sample>=target:
            raise ValueError('Opaque PBR fabric sample/normal anchor missing')
        w=instructions[sample]
        if w[0]&65535!=87 or w[-1]!=ids['uv']:
            raise ValueError('Opaque PBR fabric UV anchor mismatch')
        scale_id=const(scale)
        before=[]
        tiled=emit(before,142,ids['v2'],ids['uv'],scale_id)
        patches[sample]=before+_op(87,ids['v4'],ids['fabric'],w[3],tiled)
        if instructions[target]!=_op(83,ids['v3'],ids['result'],ids['normal']):
            raise ValueError('Opaque PBR normal result anchor mismatch')
        out=[]
        one2=emit(out,80,ids['v2'],ids['one'],ids['one'])
        bxy=emit(out,79,ids['v2'],ids['normal'],ids['normal'],0,1)
        dxy=emit(out,79,ids['v2'],ids['fabric'],ids['fabric'],0,1)
        bxy=emit(out,131,ids['v2'],emit(out,142,ids['v2'],bxy,ids['two']),one2)
        dxy=emit(out,131,ids['v2'],emit(out,142,ids['v2'],dxy,ids['two']),one2)
        xy=emit(out,129,ids['v2'],bxy,dxy)
        square=emit(out,148,ids['f'],dxy,dxy)
        dz=ext(out,ids['f'],31,emit(out,131,ids['f'],ids['one'],ext(out,ids['f'],43,square,ids['zero'],ids['one'])))
        bz=emit(out,131,ids['f'],emit(out,133,ids['f'],emit(out,81,ids['f'],ids['normal'],2),ids['two']),ids['one'])
        z=ext(out,ids['f'],40,emit(out,133,ids['f'],bz,dz),const(.00001))
        x=emit(out,81,ids['f'],xy,0);y=emit(out,81,ids['f'],xy,1)
        norm=ext(out,ids['v3'],69,emit(out,80,ids['v3'],x,y,z))
        bias=emit(out,80,ids['v3'],ids['half'],ids['half'],ids['half'])
        encoded=emit(out,129,ids['v3'],emit(out,142,ids['v3'],norm,ids['half']),bias)
        patches[target]=out+_op(83,ids['v3'],ids['result'],encoded)
    extra=len(constants)+sum(len(v)-len(instructions[i]) for i,v in patches.items())
    removed=set();freed=0
    if extra>0:
        for i in reversed(range(len(instructions))):
            if instructions[i][0]&65535 in (5,6):
                removed.add(i);freed+=len(instructions[i])
                if freed>=extra:break
    if freed<extra:raise ValueError('Insufficient opaque PBR shader debug space')
    out=words[:5];out[3]=next_id;global_done=False;entry=False;padded=False
    for i,w in enumerate(instructions):
        if i in removed:continue
        code=w[0]&65535
        if code==54 and not global_done:out+=constants;global_done=True
        if entry and not padded and code not in (59,8,317):
            out+=[1<<16]*(freed-extra);padded=True
        out+=patches.get(i,w)
        if code==248 and not entry:entry=True
    if len(out)!=len(words):raise ValueError('Opaque PBR changed wrapped shader size')
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)


def glsl(chunk,program,mode,scale,channel,emission):
    orm=b'texture(sampler2D(g_rb2DTextures[Material.s5 + glslHack].rHandle.xy), Surface.fUv0)'
    for old,new in ((b'\r\n\tMaterialAttributes.roughness = 0.5;',b'\r\n\tMaterialAttributes.roughness = '+orm+b'.g;'),
                    (b'\r\n\tMaterialAttributes.metallic = 0.0;',b'\r\n\tMaterialAttributes.metallic = '+orm+b'.b;')):
        if chunk.count(old)!=1:raise ValueError('Opaque PBR GLSL ORM anchor mismatch')
        chunk=chunk.replace(old,new)
    if mode=='CLOTH':
        old=b'MaterialAttributes.normalMap = g_v91.rgb;'
        if chunk.count(old)!=1:raise ValueError('Opaque PBR GLSL fabric normal anchor mismatch')
        fabric=('texture(sampler2D(g_rb2DTextures[Material.s7 + glslHack].rHandle.xy), '
                'Surface.fUv0 * '+format(scale,'.9g')+')').encode()
        code=(b'vec3 dbhBaseN = g_v91.rgb * 2.0 - 1.0;\n'
              b'vec2 dbhDetailXY = '+fabric+b'.xy * 2.0 - 1.0;\n'
              b'float dbhDetailZ = sqrt(1.0 - clamp(dot(dbhDetailXY,dbhDetailXY),0.0,1.0));\n'
              b'vec3 dbhCombinedN = normalize(vec3(dbhBaseN.xy + dbhDetailXY, '
              b'max(dbhBaseN.z * dbhDetailZ,0.00001)));\n'
              b'MaterialAttributes.normalMap = dbhCombinedN * 0.5 + 0.5;')
        chunk=chunk.replace(old,code)
    # The opaque coat carrier combines direct and indirect lighting before
    # this material's final color expression. Multiplying that combined value
    # by ORM.R blacks out even direct light when an artist supplies R=0 (as
    # James's packed map does). Until the separate indirect term is verified
    # in every backend, leave the donor's lighting untouched; ORM.G/B still
    # drive roughness and metallic without changing its render state.
    if chunk[-1:]!=b'\0':raise ValueError('Opaque PBR GLSL is not terminated')
    return chunk[:-1]+b'\n'+MARKER+mode.encode()+b' '+scale.hex().encode()+b' '+channel.encode()+b'\n\0'


def build_shader(template,raw,mode,scale=10,channel='R',emission=0,motion_blur=True,face_mode='BOTH'):
    from .pbr_shader import PBR_PRESETS,settings
    from .preset_shader import build_shader as build_standard
    if mode not in PBR_PRESETS:raise ValueError('Unknown opaque PBR mode')
    scale,channel=settings(scale,channel)
    if template.asset!=0x14B2A or template.payload[44:52]!=struct.pack('<II',1,0):
        raise ValueError('Unknown opaque PBR carrier')
    payload,original=build_standard(template,raw,'STANDARD_NORMAL',
                                   diffuse_emission=emission,motion_blur=motion_blur,face_mode=face_mode)
    base=Record(template.index,template.kind,template.asset,template.prefix,payload)
    data=bytearray(payload);out=bytearray();found=set()
    for program,typ,hash_at,offset,size in fragment_variants(base,original):
        chunk=original[offset:offset+size]
        if program in PROFILES:
            chunk=spirv(chunk,program,mode,scale,emission) if typ==3 else glsl(chunk,program,mode,scale,channel,emission)
            struct.pack_into('<I',data,hash_at-8,len(chunk))
            data[hash_at:hash_at+16]=hashlib.sha256(b'DBH_OPAQUE_PBR_V1'+mode.encode()+chunk).digest()[:16]
            found.add((program,typ))
        append_shader(out,chunk)
    if found!={(p,t) for p in PROFILES for t in (2,3)}:
        raise ValueError('Incomplete opaque PBR carrier')
    out+=b'\0'*(-len(out)%128)
    return bytes(data),bytes(out)
