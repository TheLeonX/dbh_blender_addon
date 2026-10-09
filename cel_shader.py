"""F00A half-Lambert/tone-color port into the verified Detroit carrier.

F00A's toneOffset/celShade bias and shade blend use neutral defaults;
ambient RGB is user-controlled (black default).
Detroit retains final lighting/fog. Its world sun direction drives the ramp.
No Naruto constant buffers, outline targets or copyrighted shader bytes ship.
"""
import math
import re
import struct

CEL_PRESETS=('CEL_DIFFUSE','CEL_NORMAL','CEL_NORMAL_EMISSION','CEL_EMISSION')


def uv_values(values):
    if len(values)!=2 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError('uvOffset2 must contain two finite values')
    try:return tuple(struct.unpack('<f',struct.pack('<f',float(v)))[0] for v in values)
    except OverflowError as error:raise ValueError('uvOffset2 exceeds the supported float range') from error


def ambient_values(values):
    if len(values)!=3 or not all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in values):
        raise ValueError('Ambient Color must contain three finite RGB values between 0 and 1')
    return tuple(struct.unpack('<f',struct.pack('<f',float(v)))[0] for v in values)


def detect_ambient(raw):
    marker=re.search(rb'// DBH_CEL_AMBIENT_V1 ([^\s]+) ([^\s]+) ([^\s]+)',raw[:230000])
    return ambient_values([float.fromhex(x.decode()) for x in marker.groups()]) if marker else (0.0,0.0,0.0)


def graph(uv,ambient=(0,0,0),force_ambient=False):
    """Typed shared expression graph for GPU code and the Blender preview.

    X retains F00A's ramp-row meaning; Y extends its unused second component
    with a horizontal half-Lambert bias. Tone sampling uses LOD0 for crisp ramps.
    """
    ambient=ambient_values(ambient)
    use_ambient=any(ambient) or force_ambient
    nodes=[]
    def add(typ,op,*args):
        name='cel'+str(len(nodes));nodes.append((name,typ,op,args));return name
    c=lambda x:add('f','constant',float(x))
    v=lambda *x:add('v3','vector',*(c(y) for y in x))
    ext=lambda a,i:add('f','extract',a,i)
    zero,one,half,eps=c(0),c(1),c(.5),c(1e-6)
    black,white=v(0,0,0),v(1,1,1)
    sun=add('v3','negate','sun')
    length=add('f','dot',sun,sun)
    safe=add('f','max',length,eps)
    inv=add('f','inversesqrt',safe)
    direction=add('v3','scale',sun,inv)
    valid=add('b','greater',length,eps)
    direction=add('v3','choose',valid,direction,v(0,1,0))
    ndotl=add('f','dot','normal',direction)
    u=add('f','add',add('f','mul',ndotl,half),half)
    u=add('f','clamp',add('f','add',u,c(uv[1])),c(.03125),c(.96875))
    row=add('f','clamp',c(uv[0]),zero,one)
    coord=add('v2','vector',u,row)
    tone=add('v3','sample',coord)
    tone=add('v3','scale',add('v3','round',add('v3','scale',tone,c(10000))),c(.0001))
    ambient_rgb=v(*ambient) if use_ambient else None
    light=add('v3','clamp',add('v3','add',tone,ambient_rgb) if use_ambient else tone,black,white)
    color=add('v3','clamp','diffuse',black,white)
    r,g,b=(ext(color,i) for i in range(3))
    high=add('f','max',r,add('f','max',g,b))
    low=add('f','min',r,add('f','min',g,b))
    saturation=add('f','div',add('f','sub',high,low),add('f','max',high,eps))
    if use_ambient:
        # F00A g_ambientColor: ambient fills the tone before saturation and
        # contributes luminance/saturation to the dark-tone blend weight.
        ar,ag,ab=(ext(ambient_rgb,i) for i in range(3))
        ah=add('f','max',ar,add('f','max',ag,ab))
        al=add('f','min',ar,add('f','min',ag,ab))
        sat=add('f','div',add('f','sub',ah,al),add('f','max',ah,eps))
        lum=add('f','dot',ambient_rgb,v(.299,.587,.114))
        saturation=add('f','max',saturation,add('f','max',sat,lum))
    multiplied=add('v3','mul',light,color)
    exponent=add('v3','scale',add('v3','sub',white,color),c(4))
    # Explicit zero-base handling avoids undefined pow(0,0), while preserving
    # the limiting 1.0 for a white diffuse channel (zero exponent).
    power=add('v3','pow',add('v3','max',light,v(1e-6,1e-6,1e-6)),exponent)
    power=add('v3','zero_power',light,exponent,power)
    dark=add('b','less',ext(tone,0),half)
    nonblack=add('b','greater',high,zero)
    selected=add('v3','choose',add('b','and',dark,nonblack),power,multiplied)
    result=add('v3','mix',selected,multiplied,saturation)
    return nodes,result


def glsl(uv,ambient=(0,0,0),capture=False):
    nodes,result=graph(uv,ambient);lines=[]
    names={'normal':'Surface.fWsNormal','sun':'Pass._vSunDir.xyz','diffuse':'g_v640.rgb'}
    types={'f':'float','v2':'vec2','v3':'vec3','b':'bool'}
    for name,typ,op,args in nodes:
        a=[names.get(x,x) if isinstance(x,str) else x for x in args]
        if op=='constant':expr=format(a[0],'.9g');expr=expr if any(x in expr for x in '.eE') else expr+'.0'
        elif op=='vector':expr=f'{types[typ]}('+','.join(a)+')'
        elif op=='extract':expr=f'{a[0]}[{a[1]}]'
        elif op in ('add','sub','mul','div','scale','less','greater','and'):
            sym={'add':'+','sub':'-','mul':'*','div':'/','scale':'*','less':'<','greater':'>','and':'&&'}[op]
            expr=f'({a[0]} {sym} {a[1]})'
        elif op=='negate':expr='-'+a[0]
        elif op=='choose':expr=f'({a[0]} ? {a[1]} : {a[2]})'
        elif op=='sample':expr=f'textureLod(sampler2D(g_rb2DTextures[Material.s22 + glslHack].rHandle.xy), {a[0]}, 0.0).rgb'
        elif op=='zero_power':
            expr='vec3('+','.join(f'({a[0]}[{i}] <= 0.0 ? ({a[1]}[{i}] <= 0.0 ? 1.0 : 0.0) : {a[2]}[{i}])' for i in range(3))+')'
        else:expr={'round':'roundEven'}.get(op,op)+'('+','.join(a)+')'
        lines.append(f'{types[typ]} {name} = {expr};')
    if capture:lines.append(f'dbhCelColor = {result};')
    return '\n'.join(lines)+f'\nMaterialAttributes.baseColor = {result};'


def _instructions(words):
    pos=5
    while pos<len(words):
        n=words[pos]>>16
        if not n or pos+n>len(words):raise ValueError('Invalid cel carrier SPIR-V')
        yield words[pos:pos+n];pos+=n


def spirv(chunk,program,uv,ambient=(0,0,0)):
    """Insert typed SSA math without changing QDIF or total module size.

    The small amount of extra code replaces non-executable debug names. Every
    original decoration, interface, executable instruction and result survives,
    except the intended final base-color definition. Spare words become Nops.
    """
    if chunk[:4]!=b'QDIF' or chunk[3944:3948]!=b'\x03\x02\x23\x07':raise ValueError('Unknown cel shader wrapper')
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]))
    if program==0:
        f,v2,v3,v4,boolean=8,9,46,43,36
        normal,sun_struct,diffuse,base=23265,18248,24453,30081
        sampler_ptr,sampler_type,sampler_array,sampler_index=3680,3674,3677,18085
    elif program==8:
        f,v2,v3,v4,boolean=10,33,17,11,34
        normal,sun_struct,diffuse,base=18697,14448,19885,25513
        sampler_ptr,sampler_type,sampler_array,sampler_index=1451,1445,1448,14261
    else:raise ValueError('Unsupported cel fragment pass')
    types={'f':f,'v2':v2,'v3':v3,'v4':v4,'b':boolean}
    next_id=words[3];body=[];constants=[]
    def emit(dst,op,*operands):dst.extend([((len(operands)+1)<<16)|op,*operands])
    def new(typ,op,*operands):
        nonlocal next_id
        result=next_id;next_id+=1;emit(body,op,typ,result,*operands);return result
    def constant(value):
        nonlocal next_id
        result=next_id;next_id+=1
        emit(constants,43,f,result,struct.unpack('<I',struct.pack('<f',value))[0]);return result
    def ext(typ,number,*args):return new(typ,12,1,number,*args)
    zero,one=constant(0),constant(1)
    sun4=new(v4,81,sun_struct,6)
    values={'normal':normal,'sun':new(v3,79,sun4,sun4,0,1,2),
            'diffuse':new(v3,79,diffuse,diffuse,0,1,2)}
    def choose(typ,cond,a,b):
        if typ==boolean or typ==f:return new(typ,169,cond,a,b)
        # SPIR-V 1.3 requires vector conditions when selecting vector values.
        parts=[new(f,169,cond,new(f,81,a,i),new(f,81,b,i)) for i in range(3)]
        return new(v3,80,*parts)
    nodes,result=graph(uv,ambient)
    for name,typ,op,args in nodes:
        a=[values.get(x,x) if isinstance(x,str) else x for x in args];t=types[typ]
        if op=='constant':value=constant(a[0])
        elif op=='vector':value=new(t,80,*a)
        elif op=='extract':value=new(f,81,*a)
        elif op in ('add','sub','mul','div','scale','less','greater','and','negate','dot'):
            value=new(t,{'add':129,'sub':131,'mul':133,'div':136,'scale':142,'less':184,'greater':186,'and':167,'negate':127,'dot':148}[op],*a)
        elif op=='choose':value=choose(t,*a)
        elif op=='sample':
            ptr=new(sampler_ptr,65,sampler_array,sampler_index)
            sampler=new(sampler_type,61,ptr)
            tex=new(v4,88,sampler,a[0],2,zero)
            value=new(v3,79,tex,tex,0,1,2)
        elif op=='mix':value=ext(t,46,a[0],a[1],new(v3,80,a[2],a[2],a[2]))
        elif op=='zero_power':
            parts=[]
            for i in range(3):
                light,exponent,power=(new(f,81,x,i) for x in a)
                at_zero=new(boolean,188,light,zero)
                zero_exp=new(boolean,188,exponent,zero)
                parts.append(choose(f,at_zero,choose(f,zero_exp,one,zero),power))
            value=new(v3,80,*parts)
        else:value=ext(t,{'min':37,'max':40,'clamp':43,'round':2,'inversesqrt':32,'pow':26}[op],*a)
        values[name]=value
    emit(body,83,v3,base,values[result])
    ins=list(_instructions(words))
    matches=[i for i,w in enumerate(ins) if w[0]&65535==79 and len(w)>2 and w[2]==base]
    if len(matches)!=1:raise ValueError('Cel shader base-color anchor mismatch')
    at=matches[0];extra=len(body)+len(constants)-len(ins[at])
    removed=set();freed=0
    for i in reversed(range(len(ins))):
        if ins[i][0]&65535 in (5,6):
            removed.add(i);freed+=len(ins[i])
            if freed>=extra:break
    if freed<extra:raise ValueError('Insufficient debug-name space for cel shader')
    out=words[:5];out[3]=next_id;inserted=False;padded=False;in_entry=False
    for i,w in enumerate(ins):
        if i in removed:continue
        op=w[0]&65535
        if op==54 and not inserted:out+=constants;inserted=True
        if in_entry and not padded and op not in (59,8,317):
            out += [1<<16]*(freed-extra);padded=True
        out+=body if i==at else w
        if op==248 and not in_entry:in_entry=True
    if len(out)!=len(words):raise ValueError('Cel shader changed wrapped size')
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)
