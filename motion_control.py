"""Per-preset blur exclusion marker, preserving pass coverage and GPU layout.

Requires loose loader 0.2 for the final blur-pass exclusion. Exact zero triggers
Detroit's camera-motion reconstruction, so write +/-2^-14 instead: finite,
normal half floats, detected by the loader's five fingerprinted blur shaders.
No discard, interface or alpha-test changes. Call only after donor validation.
"""
import struct
from .cel_shader import _instructions

LEGACY_MARKER=b'// DBH_MOTION_BLUR_V1 OFF\n'
MARKER=b'// DBH_MOTION_BLUR_V2 OFF\n'
VELOCITY=(2**-14,-2**-14)
GLSL_OUTPUT=b'out_color0 = vec4(0.00006103515625, -0.00006103515625, 0.0, 1.0);'


def detect_enabled(raw):return not any(m in raw[:230000] for m in (MARKER,LEGACY_MARKER))


def patch(chunk,program,backend,pbr=False,legacy=False):
    if backend==2:
        if program==4:
            anchor=b'out_color0 = vec4(0.0, 0.0, 0.0, 1.0);' if legacy else b'out_color0 =  vec4 (fVelocity, 0.0, 1.0);'
            if chunk.count(anchor)!=1:raise ValueError('Unknown motion-vector GLSL output')
            chunk=chunk.replace(anchor,GLSL_OUTPUT)
        if program==0:
            if chunk[-1:]!=b'\0':raise ValueError('Unterminated motion-control marker carrier')
            if legacy:
                if chunk.count(LEGACY_MARKER)!=1:raise ValueError('Unknown legacy motion marker')
                chunk=chunk.replace(LEGACY_MARKER,MARKER)
            else:chunk=chunk[:-1]+b'\n'+MARKER+b'\0'
        return chunk
    if backend!=3:raise ValueError('Unknown motion-control backend')
    if program!=4:return chunk
    if chunk[:4]!=b'QDIF' or chunk[3944:3948]!=b'\x03\x02\x23\x07':raise ValueError('Unknown motion-control QDIF')
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]))
    # Verified final output construction (OpCompositeConstruct), then OpStore.
    typ,result,x,y,zero,one,output=(11,4099,4097,4098,263,267,865) if pbr else (7,3680,3678,3679,247,230,573)
    expected=[7<<16|80,typ,result,zero if legacy else x,zero if legacy else y,zero,one]
    ins=list(_instructions(words));matches=[i for i,w in enumerate(ins) if w==expected]
    if len(matches)!=1 or ins[matches[0]+1]!=[3<<16|62,output,result]:
        raise ValueError('Unknown motion-vector SPIR-V output')
    float_types={w[1] for w in ins if w[0]&65535==22 and w[2]==32}
    constants={w[2]:w for w in ins if w[0]&65535==43}
    if (zero not in constants or one not in constants or constants[zero][1] not in float_types
            or constants[zero][3]!=0 or constants[one][1]!=constants[zero][1] or constants[one][3]!=0x3f800000):
        raise ValueError('Motion-vector zero/one constant mismatch')
    bound=words[3];flt=constants[zero][1]
    additions=[4<<16|43,flt,bound,0x38800000,4<<16|43,flt,bound+1,0xb8800000]
    ins[matches[0]]=[7<<16|80,typ,result,bound,bound+1,zero,one]
    removed=set();space=0
    for i in reversed(range(len(ins))):
        if ins[i][0]&65535 in (5,6):
            removed.add(i);space+=len(ins[i])
            if space>=len(additions):break
    if space<len(additions):raise ValueError('Insufficient motion shader debug space')
    out=words[:5];out[3]=bound+2;inserted=False;entry=False;padded=False
    for i,w in enumerate(ins):
        if i in removed:continue
        op=w[0]&65535
        if op==54 and not inserted:out+=additions;inserted=True
        if entry and not padded and op not in (59,8,317):out+=[1<<16]*(space-len(additions));padded=True
        out+=w
        if op==248 and not entry:entry=True
    if len(out)!=len(words):raise ValueError('Motion control changed wrapped shader size')
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)


def upgrade_legacy(record,raw):
    """Migrate existing marked zero-velocity presets without rebuilding geometry."""
    if LEGACY_MARKER not in raw[:230000]:return raw
    from .shader_profiles import fragment_variants,append_shader
    from .preset_shader import detect_preset
    import hashlib
    mode=detect_preset(record,raw)
    if mode is None:raise ValueError('Unsupported legacy motion material')
    payload=bytearray(record.payload);out=bytearray()
    for program,backend,at,offset,size in fragment_variants(record,raw):
        chunk=raw[offset:offset+size]
        if program in (0,4):
            changed=patch(chunk,program,backend,pbr=mode in ('SKIN','CLOTH'),legacy=True)
            if chunk!=changed:
                chunk=changed;struct.pack_into('<I',payload,at-8,len(chunk))
                key=(b'DBH_PBR_V1'+mode.encode()) if mode in ('SKIN','CLOTH') else b'DBH_MOTION_BLUR_V2'
                payload[at:at+16]=hashlib.sha256(key+chunk).digest()[:16]
        append_shader(out,chunk)
    out+=b'\0'*(-len(out)%128)
    record.payload=bytes(payload)
    return bytes(out)
