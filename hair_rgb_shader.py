"""Direct diffuse RGB on verified native hair, preserving vertex/cloth paths.

Only the final fragment base-color expression in passes 0/8 is changed.
Alpha, density, depth, strand direction, lighting and inline vertex variants
remain native. Never replace a material's vertex carrier for a color edit.
"""
import hashlib
import struct
from .hair_shader import PROFILES, op
from .shader_profiles import fragment_variants, append_shader

MARKER=b'// DBH_HAIR_RGB_V1\n'
ALT_PROFILES={
    (0,3):'a9b9b4fb05863238fda535d40c759afd43fbe600d75ee52a275038051313f3ec',
    (2,3):'1b8184a613d10c73af28d928d885542d93b8276bebf0ef5fd86eb24defa1c3d6',
    (8,3):'a4443b17a907df485ad58dc26406dc0492e337f6e47b7831588f5022fdda6eed',
    (18,3):'f7c42d210b9eb225361883e5c87d0e06b7b0bf797aa46f523ee46edab6b8073e',
    (19,3):'d9434e5c7a6321180eac985984d5edab331b7cce44a9d5f9e7931643ea65f7f4',
}
EXPRESSION=b'MaterialAttributes.diffuseColor = texture(sampler2D(g_rb2DTextures[Material.s1 + glslHack].rHandle.xy), Surface.fUv0).rgb;'
NATIVE=b'MaterialAttributes.diffuseColor = g_v58;'

def detect(raw):
    return MARKER in raw[:230000]

def _spirv(chunk,program,restore=False):
    if chunk[:4]!=b'QDIF' or struct.unpack_from('<I',chunk,4)[0]!=3944:
        raise ValueError('Unknown native hair RGB wrapper')
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]))
    v3,result,color,a,b,last={0:(51,11500,11440,11463,11491,11229),8:(17,8276,8216,8239,8267,7999)}[program]
    original=op(12,v3,result,1,46,a,b,last)
    patched=op(79,v3,result,color,color,0,1,2)
    matches=[at for at in range(5,len(words)) if words[at:at+8]==(patched if restore else original)]
    if len(matches)!=1:raise ValueError('Native hair RGB SSA anchor mismatch')
    at=matches[0];words[at:at+8]=original if restore else patched
    return chunk[:3944]+struct.pack('<'+'I'*len(words),*words)

def build_shader(record,raw,enabled=True):
    if record.payload[44:52]!=struct.pack('<II',3,1):
        raise ValueError('Direct hair RGB needs the verified native class-3 hair shader')
    marked=detect(raw);payload=bytearray(record.payload);out=bytearray();found=set();profiles=None
    for program,typ,hash_at,at,size in fragment_variants(record,raw):
        key=program,typ;chunk=raw[at:at+size];found.add(key)
        original=chunk
        if marked and program in (0,8):
            if typ==2:
                if chunk.count(MARKER)!=1 or chunk.count(EXPRESSION)!=1:
                    raise ValueError('Invalid native hair RGB marker/expression')
                original=chunk.replace(EXPRESSION,NATIVE).replace(b'\n'+MARKER,b'')
            else:original=_spirv(chunk,program,True)
        digest=hashlib.sha256(original).hexdigest()
        candidates={name for name,table in (('connor',PROFILES),('int02',{**PROFILES,**ALT_PROFILES})) if table.get(key)==digest}
        profiles=candidates if profiles is None else profiles&candidates
        if not profiles:raise ValueError('Unknown native hair shader revision; no cloth or shader data changed')
        chunk=original
        if enabled and program in (0,8):
            if typ==2:
                if chunk.count(NATIVE)!=1 or chunk[-1:]!=b'\0':raise ValueError('Unknown hair diffuse expression')
                chunk=chunk.replace(NATIVE,EXPRESSION)[:-1]+b'\n'+MARKER+b'\0'
            else:chunk=_spirv(chunk,program)
        if chunk!=raw[at:at+size]:
            struct.pack_into('<I',payload,hash_at-8,len(chunk))
            payload[hash_at:hash_at+16]=hashlib.sha256(MARKER+chunk).digest()[:16]
        append_shader(out,chunk)
    if found!=set(PROFILES):raise ValueError('Incomplete native hair passes')
    out+=b'\0'*(-len(out)%128)
    if marked==bool(enabled):return record.payload,raw
    return bytes(payload),bytes(out)

def supported(record,raw):
    try:build_shader(record,raw);return True
    except ValueError:return False
