"""Opaque standard and F00A-derived cel presets, generated from local game data.

The carrier keeps its verified GPU interface and 24 physical bindings; the user
edits semantic slots. Unused physical inputs receive neutral resources.
No shader/game bytes are bundled with the addon.
"""
import copy
import hashlib
import re
import struct
from .native import align
from .shader_profiles import coat_rgb,fragment_variants,PROFILES,append_shader
from .cel_shader import CEL_PRESETS,uv_values,ambient_values
from .pbr_shader import PBR_PRESETS

STANDARD_PRESETS=('STANDARD_DIFFUSE','STANDARD_NORMAL','STANDARD_EMISSION')
PRESETS=STANDARD_PRESETS+CEL_PRESETS+PBR_PRESETS+('HAIR',)
ROLES=('COLOR','NORMAL','EMISSION','CEL','ORM','ALPHA','FABRIC','TANGENT')
BINDINGS={'COLOR':21,'NORMAL':6,'EMISSION':5,'CEL':22}
LABELS={'COLOR':'Diffuse','NORMAL':'Normal','EMISSION':'Emission','CEL':'Celshade',
        'ORM':'ORM (Occlusion / Roughness / Metallic)','ALPHA':'Opacity','FABRIC':'Fabric Normal','TANGENT':'Hair Strand Direction'}


def role_bindings(mode,opaque=False,dither=False):
    if mode=='HAIR':
        from .hair_shader import BINDINGS as hair
        return hair
    if mode in PBR_PRESETS:
        if dither:
            from .dither_pbr_shader import BINDINGS as dither_bindings
            return dither_bindings
        if opaque:
            from .opaque_pbr_shader import BINDINGS as opaque_bindings
            return opaque_bindings
        from .pbr_shader import BINDINGS as pbr
        return pbr
    return BINDINGS


def required_roles(mode):
    if mode=='HAIR':return ('COLOR','ALPHA','TANGENT')
    if mode in PBR_PRESETS:return ('COLOR','ORM','NORMAL','ALPHA')+(('FABRIC',) if mode=='CLOTH' else ())
    if mode in STANDARD_PRESETS:return ROLES[:STANDARD_PRESETS.index(mode)+1]
    return {'CEL_DIFFUSE':('COLOR','CEL'),'CEL_NORMAL':('COLOR','NORMAL','CEL'),
            'CEL_NORMAL_EMISSION':ROLES[:4],'CEL_EMISSION':('COLOR','EMISSION','CEL')}[mode]


def detect_preset(record,raw):
    if record.kind!=2133 or not record.external:return None
    from .hair_shader import detect as detect_hair
    if detect_hair(raw):return 'HAIR'
    from .pbr_shader import detect
    pbr=detect(raw)
    if pbr:return pbr[0]
    marker=re.search(rb'// DBH_STANDARD_V1 (STANDARD_(?:DIFFUSE|NORMAL|EMISSION))\r?\n',raw[:230000])
    if marker:return marker.group(1).decode()
    marker=re.search(rb'// DBH_CEL_V1 (CEL_\w+) ',raw[:230000])
    mode=marker.group(1).decode() if marker else None
    return mode if mode in CEL_PRESETS else None


def detect_uv_offset(raw):
    marker=re.search(rb'// DBH_CEL_V1 CEL_\w+ ([^\s]+) ([^\s]+)',raw[:230000])
    return uv_values([float.fromhex(x.decode()) for x in marker.groups()]) if marker else (.5,0.0)


def _copy(typ,result,source):return [4<<16|83,typ,result,source]
def _shuffle(typ,result,source):return [8<<16|79,typ,result,source,source,0,1,2]


def _spirv(chunk,program,mode):
    # The QDIF reflection prefix is unchanged. Strict full-variant fingerprints
    # are checked by build_shader before these verified SSA IDs are used.
    start=3944
    if chunk[:4]!=b'QDIF' or chunk[start:start+4]!=b'\x03\x02\x23\x07':
        raise ValueError('Unknown QDIF shader wrapper')
    if program==0:
        vec,flt,flat,zero,half,one=46,8,3653,1062,1170,1063
        base,base_sample,normal,normal_sample,emission,emission_temp,emission_sample=30081,24453,23207,22962,30538,23064,22928
        rough,metal,intensity,exposure=24105,30101,30555,18001
        black=1773;alpha=None
    else:
        vec,flt,flat,zero,half,one=17,10,1423,385,493,386
        base,base_sample,normal,normal_sample,emission,emission_temp,emission_sample=25513,19885,18639,18394,25970,18496,18360
        rough,metal,intensity,exposure=19537,25533,25987,14177
        black=740;alpha=19582
    changes={
        base:(12,_shuffle(vec,base,base_sample)),
        normal:(131,_copy(vec,normal,normal_sample if 'NORMAL' in required_roles(mode) else flat)),
        rough:(81,_copy(flt,rough,half)),
        metal:(12,_copy(flt,metal,zero)),
        emission_temp:(12,_shuffle(vec,emission_temp,emission_sample)),
        emission:(131,_copy(vec,emission,emission_temp if 'EMISSION' in required_roles(mode) else black)),
        intensity:(81,_copy(flt,intensity,exposure)),
    }
    if alpha:changes[alpha]=(81,_copy(flt,alpha,one))
    out=bytearray(chunk);pos=start+20;found=set()
    while pos+4<=len(out):
        word=struct.unpack_from('<I',out,pos)[0];n,op=word>>16,word&65535
        if not n or pos+n*4>len(out):raise ValueError('Invalid carrier SPIR-V')
        if n>=3 and op in (12,81,131):
            result=struct.unpack_from('<I',out,pos+8)[0]
            if result in changes:
                expected,words=changes[result]
                if op!=expected or len(words)>n:raise ValueError('Carrier SSA mismatch')
                words=words+[1<<16]*(n-len(words)) # OpNop padding, identical byte length.
                struct.pack_into('<'+'I'*n,out,pos,*words);found.add(result)
        pos+=n*4
        if op==56:break
    if found!=set(changes):raise ValueError('Incomplete preset shader patch')
    return bytes(out)


def build_shader(template,raw,mode,uv_offset2=(.5,0),diffuse_emission=0.0,ambient_color=(0,0,0),outline=False,motion_blur=True,face_mode='BOTH'):
    from . import lighting_control
    diffuse_emission=lighting_control.emission_value(diffuse_emission)
    if mode not in PRESETS:raise ValueError('Unknown standard preset')
    if mode in PBR_PRESETS or mode=='HAIR':raise ValueError('Skin/cloth/hair require their dedicated shader carrier')
    uv_offset2=uv_values(uv_offset2)
    ambient_color=ambient_values(ambient_color)
    donor=copy.copy(template);donor.asset=0x14B2A
    payload,original=coat_rgb(donor,raw,False)
    donor.payload=payload
    data=bytearray(payload);out=bytearray()
    for program,typ,hash_at,offset,size in fragment_variants(donor,original):
        chunk=original[offset:offset+size]
        if (outline or face_mode!='BOTH' or (not motion_blur and program==4)) and (program,typ) not in PROFILES:
            from .outline_shader import EXTRA_PROFILES
            expected=EXTRA_PROFILES.get((program,typ))
            if expected is None or hashlib.sha256(chunk).hexdigest()!=expected:
                raise ValueError('Unknown outline depth/velocity shader revision')
        if (program,typ) in PROFILES:
            expected=PROFILES[program,typ][0]
            if hashlib.sha256(chunk).hexdigest()!=expected:raise ValueError('Unknown preset carrier revision')
            if typ==3:
                chunk=_spirv(chunk,program,mode)
                if mode in CEL_PRESETS:
                    from .cel_shader import spirv
                    chunk=spirv(chunk,program,uv_offset2,ambient_color)
                chunk=lighting_control.spirv(chunk,program,diffuse_emission,mode in CEL_PRESETS)
            else:
                expr={'normalMap':'g_v91.rgb' if 'NORMAL' in required_roles(mode) else 'vec3(0.5,0.5,1.0)',
                      'baseColor':'g_v640.rgb','roughness':'0.5','metallic':'0.0','transparency':'1.0',
                      'emissiveColor':'g_v89.rgb' if 'EMISSION' in required_roles(mode) else 'vec3(0.0)',
                      'emissiveIntensity':'Surface._fLocalExposureCompensationEV100'}
                for name,value in expr.items():
                    pattern=(r'MaterialAttributes\.'+name+r' = g_v\d+;').encode()
                    chunk,count=re.subn(pattern,('MaterialAttributes.'+name+' = '+value+';').encode(),chunk)
                    if count!=1:raise ValueError('Unknown preset GLSL expression')
                if chunk[-1:]!=b'\0':raise ValueError('GLSL string lacks terminator')
                if mode in CEL_PRESETS:
                    from .cel_shader import glsl
                    chunk=chunk.replace(b'MaterialAttributes.baseColor = g_v640.rgb;',
                        glsl(uv_offset2,ambient_color,capture=bool(diffuse_emission) and program==0).encode())
                    marker='// DBH_CEL_V1 '+mode+' '+' '.join(v.hex() for v in uv_offset2)
                    if any(ambient_color):marker+='\n// DBH_CEL_AMBIENT_V1 '+' '.join(v.hex() for v in ambient_color)
                else:marker='// DBH_STANDARD_V1 '+mode
                chunk=lighting_control.glsl(chunk,program,diffuse_emission,mode in CEL_PRESETS)
                if diffuse_emission:marker+='\n// DBH_DIFFUSE_EMISSION_V1 '+diffuse_emission.hex()
                chunk=chunk[:-1]+('\n'+marker+'\n').encode()+b'\0'
            # Variant size is serialized immediately before backend type/hash.
            struct.pack_into('<I',data,hash_at-8,len(chunk))
            data[hash_at:hash_at+16]=hashlib.sha256(b'DBH_STANDARD_V1'+mode.encode()+chunk).digest()[:16]
        if outline and face_mode=='BACK':raise ValueError('Outline hull requires front-facing faces')
        if outline or face_mode!='BOTH':
            from . import outline_shader
            visible='FRONT' if outline else face_mode
            chunk=outline_shader.spirv(chunk,visible) if typ==3 else outline_shader.glsl(chunk,visible,outline=outline)
            struct.pack_into('<I',data,hash_at-8,len(chunk))
            data[hash_at:hash_at+16]=hashlib.sha256(b'DBH_OUTLINE_V1'+chunk).digest()[:16]
        if not motion_blur and program in (0,4):
            from .motion_control import patch
            updated=patch(chunk,program,typ)
            if updated!=chunk:
                chunk=updated
                struct.pack_into('<I',data,hash_at-8,len(chunk))
                data[hash_at:hash_at+16]=hashlib.sha256(b'DBH_MOTION_BLUR_V2'+chunk).digest()[:16]
        append_shader(out,chunk)
    out+=b'\0'*(-len(out)%128)
    return bytes(data),bytes(out)
