"""Authored textures on native Connor hair (14B2B), not dithered PBR.

Keep native class 3, soft-alpha output, density passes 18/19, adaptive depth
prepass and anisotropic lighting. Only texture expressions are redirected.
There is deliberately no invented velocity pass or render-state bit patch.
"""
import hashlib,re,struct
from .cel_shader import _instructions
from .shader_profiles import fragment_variants,append_shader

MARKER=b'// DBH_NATIVE_HAIR_V1 '
BINDINGS={'COLOR':1,'ALPHA':3,'TANGENT':0}
# Neutralize Connor-specific wound, environment-response and specular masks.
INTERNAL_BINDINGS={2:'HAIR_DAMAGE',4:'HAIR_LOOKUP',5:'HAIR_SPEC'}
PROFILES={
 (0,2):'5312109bbd42bec9ff2f70d4620ad1b46ae8d005707643b7c31f92217ce99fb7',
 (0,3):'dc8be5e13526f3bb0ec19d2ca6a8b5a094c87220206e6dacac4fcd34627018b4',
 (2,2):'93df8e207d17e9eeb5ce66622539ea5274fff4ef54dbc32055700846662f1449',
 (2,3):'a0bc283ccefae697e6a23d0fbe2b2e24ffdf04591e2a1fb1dba610bbf8908cfe',
 (8,2):'5cbfd71468ebca6d7e5c5462b1d32a7600463d22c585184c628e402f14008a1a',
 (8,3):'f9066abba85247ef17e1368f624b3f1990e669665f83b846532cac2ce57c6e5a',
 (18,2):'9dc038c8febf88047097785383a4ea19917921b2b6bd59d2f67aa0d2cfc9b95e',
 (18,3):'2353eeef630d83dcff3378f71e642f179a60782d34ff74dd0c085e7f5b5082e1',
 (19,2):'9dc038c8febf88047097785383a4ea19917921b2b6bd59d2f67aa0d2cfc9b95e',
 (19,3):'03e66bfa8115b1f8f7cd49af718ffb552690c192ad5e912f539a546cd9b62023',
}

def detect(raw):
    found=re.search(rb'// DBH_NATIVE_HAIR_V1 ([RGBA])',raw[:230000])
    return found[1].decode() if found else None

def op(code,*args):return [((len(args)+1)<<16)|code,*args]

def spirv(chunk,program,channel):
    if chunk[:4]!=b'QDIF' or struct.unpack_from('<I',chunk,4)[0]!=3944:raise ValueError('Unknown native hair wrapper')
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]))
    ins=list(_instructions(words));patches={}
    # Final transparency expression and the existing unmodified UV0 sample.
    f,alpha,sample={0:(8,11238,11156),8:(10,8008,7926),2:(10,3535,3453),
                    18:(10,3499,3417),19:(10,3499,3417)}[program]
    def replace(result,opcode,new):
        matches=[i for i,w in enumerate(ins) if len(w)>2 and w[2]==result and w[0]&65535==opcode]
        if len(matches)!=1:raise ValueError(f'Native hair SSA anchor {result}/{opcode} mismatch')
        i=matches[0]
        if len(new)>len(ins[i]):raise ValueError('Hair expression would grow shader')
        patches[i]=new+[1<<16]*(len(ins[i])-len(new))
    replace(alpha,81,op(81,f,alpha,sample,'RGBA'.index(channel)))
    if program in (0,8):
        v3,result,color=(51,11500,11440) if program==0 else (17,8276,8216)
        replace(result,12,op(79,v3,result,color,color,0,1,2))
    if program==0:
        # Tangent-map XY is authored data, not Connor's color-corrected map.
        replace(11260,133,op(83,43,11260,11255))
    out=words[:5]
    for i,w in enumerate(ins):out+=patches.get(i,w)
    assert len(out)==len(words)
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)

def glsl(chunk,program,channel):
    def tex(n):return f'texture(sampler2D(g_rb2DTextures[Material.s{n} + glslHack].rHandle.xy), Surface.fUv0)'
    chunk,n=re.subn(rb'MaterialAttributes.transparency = g_v\d+;',
                    ('MaterialAttributes.transparency = '+tex(3)+'.'+channel.lower()+';').encode(),chunk)
    if n!=1:raise ValueError('Unknown native hair opacity expression')
    if program in (0,8):
        chunk,n=re.subn(rb'MaterialAttributes.diffuseColor = g_v\d+;',
                        ('MaterialAttributes.diffuseColor = '+tex(1)+'.rgb;').encode(),chunk)
        if n!=1:raise ValueError('Unknown native hair diffuse expression')
    if program==0:
        chunk,n=re.subn(rb'g_v55 =  texture\([^\n]+;',('g_v55 = '+tex(0)+';').encode(),chunk)
        if n!=1:raise ValueError('Unknown native hair tangent expression')
    if chunk[-1:]!=b'\0':raise ValueError('Unterminated hair GLSL')
    return chunk[:-1]+b'\n'+MARKER+channel.encode()+b'\n\0'

def build_shader(template,raw,channel='R',face_mode='BOTH'):
    if channel not in 'RGBA' or len(channel)!=1:raise ValueError('Choose R/G/B/A opacity')
    if template.asset!=0x14B2B or template.payload[44:52]!=struct.pack('<II',3,1):raise ValueError('Unknown native hair carrier')
    payload=bytearray(template.payload);out=bytearray();found=set()
    for program,typ,hash_at,at,size in fragment_variants(template,raw):
        chunk=raw[at:at+size];key=program,typ
        if hashlib.sha256(chunk).hexdigest()!=PROFILES.get(key):raise ValueError('Unknown native hair shader revision')
        found.add(key)
        chunk=(spirv if typ==3 else glsl)(chunk,program,channel)
        if face_mode!='BOTH':
            from .outline_shader import spirv as face_spirv,glsl as face_glsl
            chunk=face_spirv(chunk,face_mode) if typ==3 else face_glsl(chunk,face_mode,outline=False)
        struct.pack_into('<I',payload,hash_at-8,len(chunk))
        payload[hash_at:hash_at+16]=hashlib.sha256(MARKER+chunk).digest()[:16]
        append_shader(out,chunk)
    if found!=set(PROFILES):raise ValueError('Incomplete native hair passes')
    out+=b'\0'*(-len(out)%128)
    return bytes(payload),bytes(out)
