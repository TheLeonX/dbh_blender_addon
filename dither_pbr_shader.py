"""PBR coverage on native masked 14B22; preserve every GPU/vertex interface.

The carrier samples opacity in depth and velocity already. Color, debug,
depth/shadow and velocity use the same screen-pixel coverage function and UV0.
No transparent render-state conversion or new descriptors are involved.
"""
import hashlib
import re
import struct
from .cel_shader import _instructions
from .shader_profiles import fragment_variants,append_shader
from .pbr_shader import settings,PBR_PRESETS
from .lighting_control import emission_value

MARKER=b'// DBH_PBR_DITHER_V1 '
BINDINGS={'COLOR':18,'NORMAL':6,'ORM':5,'ALPHA':14,'FABRIC':8}
PROFILES={
 (0,2):'3400dcd30eb81810001c5560d5d3c3d6df656db28cf939a29a210d5813b91fa0',
 (0,3):'cd0f333b9845c39f091e408e43dc0ee5b09857545d79b77ed8e01ec2ee272b1c',
 (2,2):'ac4199dc7a806408b94406245dfa5903cc436f27e362bdf2f31cbfc65f43547b',
 (2,3):'e7d7863b0638cff4ee651c380f16de92b5ceb216decacec66fa3d458ac720741',
 (4,2):'2bb4422a1b716355fbf8b522e929a953ef3b15d6cc94ec250b748ca5ca96b9cb',
 (4,3):'97775dffe9f020c19986edac099b8ed705d0fd476f4ced7255877b2b161d8735',
 (8,2):'4f93951ce2476e9f613ae0dcba5311f51bca77e5699aa3ac07d0d234ad219ab0',
 (8,3):'803dd6bf96dbaf5c6226a1c74acb9a1af6203ac32b6175975a129bd669846670',
}
IDS={
 0:dict(f=8,v2=9,v3=51,v4=43,b=36,uv=14188,frag=10185,alpha=15521,
        normal=15143,base=15988,rough=15512,metal=16007,cavity=16140,
        normal_rgb=14847,detail_rgb=14894,detail=14869,orm=14813,color=15701),
 8:dict(f=10,v2=33,v3=17,v4=11,b=34,uv=9839,frag=6583,alpha=11172,compare=8510,
        normal=10794,base=11639,rough=11163,metal=11658,cavity=11791,
        normal_rgb=10498,detail_rgb=10545,detail=10520,orm=10464,color=11352),
 2:dict(f=10,v2=18,v3=17,v4=11,b=19,uv_input=189,frag=886,alpha=3330,compare=2820),
 4:dict(f=10,v2=36,v3=43,v4=11,b=44,uv_input=250,frag=950,alpha=3768,compare=2883),
}

def op(code,*args):return [((len(args)+1)<<16)|code,*args]

def spirv(chunk,program,mode,scale,channel,emission,motion_blur):
    if chunk[:4]!=b'QDIF' or chunk[3944:3948]!=b'\x03\x02\x23\x07':
        raise ValueError('Unknown dithered PBR wrapper')
    p=IDS[program];words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]))
    ins=list(_instructions(words));next_id=words[3];constants=[];patches={};phi_split=None
    def fresh():
        nonlocal next_id
        result=next_id;next_id+=1;return result
    def emit(out,code,typ,*args):
        result=fresh();out+=op(code,typ,result,*args);return result
    def ext(out,typ,code,*args):return emit(out,12,typ,1,code,*args)
    def const(v):
        result=fresh();constants.extend(op(43,p['f'],result,struct.unpack('<I',struct.pack('<f',v))[0]));return result
    zero,one,half,two=map(const,(0.,1.,.5,2.))
    def at_result(result,opcode):
        found=[i for i,w in enumerate(ins) if len(w)>2 and w[2]==result and (w[0]&65535)==opcode]
        if len(found)!=1:raise ValueError(f'Dithered PBR SSA anchor {result}/{opcode} mismatch')
        return found[0]
    def replace(result,opcode,value):patches[at_result(result,opcode)]=value
    if program in (0,8):
        replace(p['rough'],81,op(81,p['f'],p['rough'],p['orm'],1))
        replace(p['metal'],12,op(81,p['f'],p['metal'],p['orm'],2))
        replace(p['base'],12,op(79,p['v3'],p['base'],p['color'],p['color'],0,1,2))
        replace(p['cavity'],12,op(83,p['f'],p['cavity'],one))
        if mode=='SKIN':
            replace(p['normal'],12,op(83,p['v3'],p['normal'],p['normal_rgb']))
        else:
            at=at_result(p['detail'],87);w=ins[at];out=[]
            if w[-1]!=p['uv']:raise ValueError('Unexpected fabric UV')
            tiled=emit(out,142,p['v2'],p['uv'],const(scale))
            patches[at]=out+op(87,p['v4'],p['detail'],w[3],tiled)
            out=[];ones=emit(out,80,p['v3'],one,one,one)
            base=emit(out,131,p['v3'],emit(out,142,p['v3'],p['normal_rgb'],two),ones)
            detail=emit(out,131,p['v3'],emit(out,142,p['v3'],p['detail_rgb'],two),ones)
            xy=emit(out,129,p['v2'],emit(out,79,p['v2'],base,base,0,1),emit(out,79,p['v2'],detail,detail,0,1))
            z=ext(out,p['f'],40,emit(out,133,p['f'],emit(out,81,p['f'],base,2),emit(out,81,p['f'],detail,2)),const(.00001))
            n=ext(out,p['v3'],69,emit(out,80,p['v3'],emit(out,81,p['f'],xy,0),emit(out,81,p['f'],xy,1),z))
            encoded=emit(out,129,p['v3'],emit(out,142,p['v3'],n,half),emit(out,80,p['v3'],half,half,half))
            replace(p['normal'],12,out+op(83,p['v3'],p['normal'],encoded))
    # Sample opacity through the native sampler in every pass, using the
    # already-declared UV0 input in depth/velocity (no new vertex varyings).
    at=at_result(p['alpha'],87);w=ins[at];out=[]
    uv=p.get('uv')
    if uv is None:
        loaded=emit(out,61,p['v4'],p['uv_input'])
        uv=emit(out,79,p['v2'],loaded,loaded,0,1)
    out+=op(87,p['v4'],p['alpha'],w[3],uv)
    alpha=ext(out,p['f'],43,emit(out,81,p['f'],p['alpha'],'RGBA'.index(channel)),zero,one)
    frag=emit(out,61,p['v4'],p['frag'])
    pixel=ext(out,p['v2'],8,emit(out,79,p['v2'],frag,frag,0,1))
    weights=emit(out,80,p['v2'],const(.06711056),const(.00583715))
    noise=ext(out,p['f'],10,emit(out,148,p['f'],pixel,weights))
    threshold=ext(out,p['f'],10,emit(out,133,p['f'],noise,const(52.9829189)))
    if program==0:
        condition=emit(out,188,p['b'],alpha,threshold)
        killed,merge=fresh(),fresh()
        label=next(row[1] for row in reversed(ins[:at]) if row[0]&65535==248)
        out+=op(247,merge,0)+op(250,condition,killed,merge)+op(248,killed)+op(252)+op(248,merge)
        phi_split=(at,label,merge)
    else:
        replace(p['compare'],184,op(188,p['b'],p['compare'],alpha,threshold))
    patches[at]=out
    if program==0 and emission:
        at=at_result(23943,79);w=ins[at]
        if w!=op(79,43,23943,23942,23546,4,5,6,3):raise ValueError('Unknown masked light output')
        out=[];rgb=emit(out,79,p['v3'],p['color'],p['color'],0,1,2)
        t=const(emission);weight=emit(out,80,p['v3'],t,t,t)
        mixed=ext(out,p['v3'],46,23546,rgb,weight)
        patches[at]=out+op(79,43,23943,23942,mixed,4,5,6,3)
        at=at_result(23936,12);w=ins[at];prior=fresh()
        patches[at]=[w[0],w[1],prior,*w[3:]]+op(142,p['v3'],23936,prior,const(1-emission))
    if program==4 and not motion_blur:
        at=at_result(3830,80)
        if ins[at]!=op(80,11,3830,3828,3829,263,267):raise ValueError('Unknown masked velocity output')
        patches[at]=op(80,11,3830,const(2**-14),const(-2**-14),263,267)
    extra=len(constants)+sum(len(v)-len(ins[i]) for i,v in patches.items())
    removed=set();freed=0
    for i in reversed(range(len(ins))):
        if ins[i][0]&65535 in (5,6):
            removed.add(i);freed+=len(ins[i])
            if freed>=extra:break
    if freed<extra:raise ValueError('Insufficient dithered PBR debug space')
    out=words[:5];out[3]=next_id;inserted=False;entry=False;padded=False
    for i,w in enumerate(ins):
        if i in removed:continue
        code=w[0]&65535
        if code==54 and not inserted:out+=constants;inserted=True
        if entry and not padded and code not in (59,8,317):out+=[1<<16]*(freed-extra);padded=True
        if phi_split and i>phi_split[0] and code==245:
            w=w.copy()
            for j in range(4,len(w),2):
                if w[j]==phi_split[1]:w[j]=phi_split[2]
        out+=patches.get(i,w)
        if code==248 and not entry:entry=True
    if len(out)!=len(words):raise ValueError('Dithered PBR changed wrapped shader size')
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)

def glsl(chunk,program,mode,scale,channel,emission,motion_blur):
    def tex(binding,uv='Surface.fUv0'):
        return f'texture(sampler2D(g_rb2DTextures[Material.s{binding} + glslHack].rHandle.xy), {uv})'
    def assign(name,value):
        nonlocal chunk
        chunk,n=re.subn((r'MaterialAttributes\.'+name+r' = g_v\d+(?:\.rgb)?;').encode(),
                        ('MaterialAttributes.'+name+' = '+value+';').encode(),chunk)
        if n!=1:raise ValueError('Unknown masked GLSL '+name)
    if program in (0,8):
        assign('baseColor',tex(18)+'.rgb');assign('roughness',tex(5)+'.g')
        assign('metallic',tex(5)+'.b');assign('cavity','1.0')
        code='vec2 dbhNxy = '+tex(6)+'.xy * 2.0 - 1.0;\n'
        code+='vec3 dbhN = vec3(dbhNxy, sqrt(1.0-clamp(dot(dbhNxy,dbhNxy),0.0,1.0)));\n'
        if mode=='CLOTH':
            code+='vec2 dbhDxy = '+tex(8,'Surface.fUv0 * '+format(scale,'.9g'))+'.xy * 2.0 - 1.0;\n'
            code+='vec3 dbhD = vec3(dbhDxy, sqrt(1.0-clamp(dot(dbhDxy,dbhDxy),0.0,1.0)));\n'
            code+='dbhN = vec3(dbhN.xy + dbhD.xy, dbhN.z * dbhD.z);\n'
        code+='dbhN.z = max(dbhN.z,0.00001);\nMaterialAttributes.normalMap = normalize(dbhN)*0.5+0.5;'
        anchor=b'MaterialAttributes.normalMap = g_v0;'
        if chunk.count(anchor)!=1:raise ValueError('Unknown masked GLSL normal')
        chunk=chunk.replace(anchor,code.encode())
    assign('transparency','clamp('+tex(14)+'.'+channel.lower()+',0.0,1.0)')
    # One coverage function in every pass. Its endpoints discard all alpha=0
    # pixels and retain all alpha=1 pixels, including depth and motion markers.
    test=('MaterialAttributes.transparency <= fract(52.9829189 * fract(dot(floor(gl_FragCoord.xy), '
          'vec2(0.06711056,0.00583715))))')
    if program==0:
        anchor=b'\tS_RAIN_SURFACE RainSurface;'
        if chunk.count(anchor)!=1:raise ValueError('Unknown masked GLSL main coverage anchor')
        chunk=chunk.replace(anchor,('if ('+test+') { discard; }\n').encode()+anchor)
    else:
        chunk,n=re.subn(rb'MaterialAttributes.transparency < (?:MaterialAttributes.alphaTestValue|fAlphaTestValue)',test.encode(),chunk)
        if n!=1:raise ValueError('Unknown masked GLSL coverage test')
    if program==0 and emission:
        anchor=b'vec3  light = LightingResult.fLighting + MaterialAttributes.emissiveColor * fEmissiveIntensity;'
        if chunk.count(anchor)!=1:raise ValueError('Unknown masked diffuse emission anchor')
        value=format(emission,'.9g')
        chunk=chunk.replace(anchor,('vec3 light = mix(LightingResult.fLighting, '+tex(18)+'.rgb, '+value+');\n'
                                  'vGIandSSRSpecularFactor *= (1.0 - '+value+');').encode())
    if program==4 and not motion_blur:
        from .motion_control import GLSL_OUTPUT
        anchor=b'out_color0 =  vec4 (fVelocity, 0.0, 1.0);'
        if chunk.count(anchor)!=1:raise ValueError('Unknown masked motion GLSL')
        chunk=chunk.replace(anchor,GLSL_OUTPUT)
    marker=MARKER+mode.encode()+b' '+scale.hex().encode()+b' '+channel.encode()+b'\n'
    if program==0:
        if emission:marker+=b'// DBH_DIFFUSE_EMISSION_V1 '+emission.hex().encode()+b'\n'
        if not motion_blur:
            from .motion_control import MARKER as MOTION
            marker+=MOTION
    if chunk[-1:]!=b'\0':raise ValueError('Unterminated masked GLSL')
    return chunk[:-1]+b'\n'+marker+b'\0'

def build_shader(template,raw,mode,scale=10,channel='R',emission=0,motion_blur=True,face_mode='BOTH'):
    scale,channel=settings(scale,channel);emission=emission_value(emission)
    if mode not in PBR_PRESETS:raise ValueError('Unknown dithered PBR mode')
    if template.asset!=0x14B22 or template.payload[44:52]!=struct.pack('<II',1,0):
        raise ValueError('Unknown native masked PBR carrier')
    payload=bytearray(template.payload);out=bytearray();found=set()
    for program,typ,hash_at,at,size in fragment_variants(template,raw):
        chunk=raw[at:at+size];key=program,typ
        if hashlib.sha256(chunk).hexdigest()!=PROFILES.get(key):raise ValueError('Unknown masked shader revision')
        found.add(key)
        chunk=(spirv if typ==3 else glsl)(chunk,program,mode,scale,channel,emission,motion_blur)
        if face_mode!='BOTH':
            from .outline_shader import spirv as faces_spirv,glsl as faces_glsl
            chunk=faces_spirv(chunk,face_mode) if typ==3 else faces_glsl(chunk,face_mode,outline=False)
        struct.pack_into('<I',payload,hash_at-8,len(chunk))
        payload[hash_at:hash_at+16]=hashlib.sha256(MARKER+chunk).digest()[:16]
        append_shader(out,chunk)
    if found!=set(PROFILES):raise ValueError('Incomplete masked carrier')
    out+=b'\0'*(-len(out)%128)
    return bytes(payload),bytes(out)
