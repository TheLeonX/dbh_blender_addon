"""Final additive emission on verified native eye shaders, not a new carrier.

The native S_MATERIAL already reserves eight texture words; this eye material
uses five. Bind slot 5 within that existing layout. Only color program 0 is
patched; iris projection, cornea, reflections, vertex and debug passes remain.
The original Vulkan program is retained in a bounded compressed GLSL comment
for exact restoration and verification, including stripped debug names.
"""
import base64,hashlib,math,re,struct,zlib
from .shader_profiles import fragment_variants,append_shader
from .materials import bindings
from .cel_shader import _instructions

MARKER=b'// DBH_EYE_EMISSION_V1 '
PROFILES={
    (0,2):'2a4093685b29e98662d68129f84b28fa3ed3dd0fa6b0ebadfe2560bbdb257140',
    (0,3):'8d5b1987edabdf9141147ad7d476bcc34c33dd26eae18b8143c4056564ff1aeb',
    (8,2):'6d94cbce4e8af4d8ed47e009426765e96de11c69998b16873be6427fb06baa5a',
    (8,3):'79f6ae921416ccb7e63eb1e56f5be8f11a43b5e9f4ae7da0e15a9f7e51a00b31',
}
HOSTAGE_PROFILES={
    (0,2):'c7044a325fbe515b22ae1a3b0075235ee850cada65d8aa8d36f5a229116ef95f',
    (0,3):'fd49d566028feda4a4f3366a75443221991561ffbcbc1b914a6f9aa83d022353',
    (8,2):'6b8bef113ed91a2df25fcecb4571014a695c3309339ac67b6f22a345494b2cc3',
    (8,3):'ef64841393d92453daa45a503e5f4390771a3ddd5a2df8e632a323b36dc0eb00',
}
ANCHOR=b'vec3  fColor = diffuse + vEmissiveColor;'

def strength(value):
    value=float(value)
    if not math.isfinite(value) or not 0<=value<=100:raise ValueError('Eye emission strength must be finite, 0..100')
    return struct.unpack('<f',struct.pack('<f',value))[0]

def expression(value):
    number=format(strength(value),'.9g');number=number if any(c in number for c in '.eE') else number+'.0'
    return ANCHOR+b'\n fColor += texture(sampler2D(g_rb2DTextures[Material.s5 + glslHack].rHandle.xy), Surface.fUv0).rgb * '+number.encode()+b';'

def detect(raw):
    match=re.search(rb'// DBH_EYE_EMISSION_V1 ([^\s]+) ([A-Za-z0-9+/=]+)\n',raw)
    return strength(float.fromhex(match[1].decode())) if match else None

def op(code,*args):return [((len(args)+1)<<16)|code,*args]

def spirv(chunk,value):
    if chunk[:8]!=b'QDIF'+struct.pack('<I',3944):raise ValueError('Unknown eye shader wrapper')
    words=list(struct.unpack('<'+'I'*((len(chunk)-3944)//4),chunk[3944:]));ins=list(_instructions(words))
    # Resolve live typed IDs from the fingerprinted original, not IDs guessed
    # across compiler revisions. All expected patterns must be unique.
    names={w[1]:struct.pack('<'+'I'*(len(w)-2),*w[2:]).split(b'\0')[0].decode('utf8')
           for w in ins if w[0]&65535==5}
    def unique(values,label):
        if len(values)!=1:raise ValueError('Ambiguous native eye '+label)
        return values[0]
    output=unique([i for i,n in names.items() if n=='out_color0'],'output')
    array=unique([i for i,n in names.items() if n=='g_rb2DTextures'],'texture array')
    material_type=unique([i for i,n in names.items() if n=='S_MATERIAL'],'material type')
    material=unique([w[2] for w in ins if w[0]&65535==61 and w[1]==material_type],'material load')
    members={struct.pack('<'+'I'*(len(w)-3),*w[3:]).split(b'\0')[0].decode('utf8'):w[2]
             for w in ins if w[0]&65535==6 and w[1]==material_type}
    if 's2' not in members or 's5' not in members:raise ValueError('Native eye reserved texture slots unavailable')
    extract=unique([w for w in ins if w[0]&65535==81 and w[3:]==[material,members['s2']]],'diffuse slot')
    uint=extract[1];index2=extract[2]
    ptr=unique([w for w in ins if w[0]&65535==65 and w[3:]==[array,index2]],'diffuse pointer')
    sampler=unique([w for w in ins if w[0]&65535==61 and w[3:]==[ptr[2]]],'diffuse sampler')
    samples=[w for w in ins if w[0]&65535==87 and w[3]==sampler[2]]
    sample=unique(samples,'diffuse sample');uv=sample[4]
    stored=[w[2] for w in ins if w[0]&65535==62 and w[1]==output]
    if not stored:raise ValueError('Missing native eye color output')
    anchor=unique([w for w in ins if w[0]&65535==79 and w[2]==stored[-1]],'final RGB shuffle')
    if anchor[5:]!=[4,5,6,3]:raise ValueError('Unknown native eye RGB/alpha export')
    rgb=anchor[4];rgbdef=unique([w for w in ins if len(w)>2 and w[0]&65535 in (12,129) and w[2]==rgb],'native RGB')
    v3=rgbdef[1];v4=anchor[1]
    vectype=unique([w for w in ins if w[0]&65535==23 and w[1]==v3],'RGB vector type')
    if vectype[3]!=3:raise ValueError('Native eye RGB is not vec3')
    f=vectype[2]
    if not any(w==op(22,f,32) for w in ins):raise ValueError('Native eye float is not float32')
    if not any(w==op(61,v4,anchor[3],output) for w in ins):raise ValueError('Native eye output alpha not retained')
    layout=unique([w for w in ins if w[0]&65535==30 and w[1]==material_type],'material layout')
    if len(layout)<=2+members['s5'] or layout[2+members['s5']]!=uint:raise ValueError('Native eye has no reserved uint texture word')
    matches=[i for i,w in enumerate(ins) if w==anchor]
    if len(matches)!=1:raise ValueError('Native eye output anchor mismatch')
    bound=words[3];constant=bound;ids=iter(range(bound+1,bound+20))
    def new(typ,code,*args):
        result=next(ids);body.extend(op(code,typ,result,*args));return result
    body=[];index=new(uint,81,material,members['s5'])
    pointer=new(ptr[1],65,array,index);new_sampler=new(sampler[1],61,pointer)
    tex=new(v4,87,new_sampler,uv);emission=new(v3,79,tex,tex,0,1,2)
    scaled=new(v3,142,emission,constant);final=new(v3,129,rgb,scaled)
    body+=op(79,v4,anchor[2],anchor[3],final,4,5,6,3)
    constants=op(43,f,constant,struct.unpack('<I',struct.pack('<f',strength(value)))[0])
    extra=len(constants)+len(body)-len(anchor);removed=set();freed=0
    for i in reversed(range(len(ins))):
        if ins[i][0]&65535 in (5,6):
            removed.add(i);freed+=len(ins[i])
            if freed>=extra:break
    if freed<extra:raise ValueError('Insufficient native eye debug space')
    out=words[:5];out[3]=bound+8;inserted=False;entry=False;padded=False
    for i,w in enumerate(ins):
        if i in removed:continue
        code=w[0]&65535
        if code==54 and not inserted:out+=constants;inserted=True
        if entry and not padded and code not in (59,8,317):
            out+=[1<<16]*(freed-extra);padded=True
        out+=body if i==matches[0] else w
        if code==248 and not entry:entry=True
    if not inserted or not padded or len(out)!=len(words):raise ValueError('Eye patch changed native wrapped size')
    return chunk[:3944]+struct.pack('<'+'I'*len(out),*out)

def build_shader(record,raw,value=None):
    """None restores exact native programs. Texture bindings are kept separate."""
    if record.payload[44:52]!=struct.pack('<II',9,0) or not 5<=len(bindings(record.payload))<=8:
        raise ValueError('Eye emission needs a verified native eye material, not skin/hair/eyelids')
    variants=fragment_variants(record,raw);chunks={(p,t):raw[a:a+n] for p,t,_,a,n in variants}
    if set(chunks)!=set(PROFILES):raise ValueError('Unsupported native eye pass set')
    original=dict(chunks);marked=detect(raw)
    if marked is not None:
        match=re.search(rb'\n// DBH_EYE_EMISSION_V1 ([^\s]+) ([A-Za-z0-9+/=]+)\n',chunks[0,2])
        if match is None or chunks[0,2].count(MARKER)!=1:raise ValueError('Invalid eye emission metadata')
        encoded=match[2]
        if len(encoded)>180000:raise ValueError('Oversized eye shader backup')
        decoder=zlib.decompressobj();backup=decoder.decompress(base64.b64decode(encoded,validate=True),120001)
        if len(backup)>120000 or not decoder.eof or decoder.unused_data:raise ValueError('Invalid eye shader backup')
        original[0,3]=backup
        edited=expression(marked)
        if chunks[0,2].count(edited)!=1:raise ValueError('Eye emission expression mismatch')
        original[0,2]=chunks[0,2].replace(edited,ANCHOR).replace(match[0],b'')
        if spirv(backup,marked)!=chunks[0,3]:raise ValueError('Modified eye shader does not match its native backup')
    if not any(all(hashlib.sha256(c).hexdigest()==table[k] for k,c in original.items()) for table in (PROFILES,HOSTAGE_PROFILES)):
        raise ValueError('Unknown native eye shader revision; no shader data changed')
    updated=dict(original)
    if value is not None:
        value=strength(value)
        if len(bindings(record.payload))<6:raise ValueError('Export an emission texture binding first')
        c=original[0,2]
        if c.count(ANCHOR)!=1 or c[-1:]!=b'\0':raise ValueError('Native eye lighting expression mismatch')
        marker=b'\n'+MARKER+value.hex().encode()+b' '+base64.b64encode(zlib.compress(original[0,3],9))+b'\n'
        updated[0,2]=c.replace(ANCHOR,expression(value))[:-1]+marker+b'\0'
        updated[0,3]=spirv(original[0,3],value)
    if updated==chunks:return record.payload,raw
    payload=bytearray(record.payload);out=bytearray()
    for program,typ,at_hash,_,_ in variants:
        c=updated[program,typ]
        if c!=chunks[program,typ]:
            struct.pack_into('<I',payload,at_hash-8,len(c))
            payload[at_hash:at_hash+16]=hashlib.sha256(MARKER+c).digest()[:16]
        append_shader(out,c)
    out+=b'\0'*(-len(out)%128)
    return bytes(payload),bytes(out)

def supported(record,raw):
    try:build_shader(record,raw);return True
    except (ValueError,RuntimeError,zlib.error):return False

def export_texture(package,mat,record,experimental,single_mip=False):
    """Embed a separate sRGB emission map in reserved material texture word5."""
    import numpy as np
    from .texture_export import image_rgba,mip_chain,make_texture,validate_texture
    from .materials import texture_record
    from .native import Record
    raw=package.members[package.record_members[record.index]].unpacked
    build_shader(record,raw) # Verify before making a texture or editing a table.
    table=bindings(record.payload);image=mat.dbh_eye_emission_image
    if image is None and mat.dbh_eye_emission_strength>0:raise ValueError(f'{mat.name}: load the separate Eye Emission Texture first (or set strength to 0)')
    if len(table)>5 and image and image==next((s.original_image for s in mat.dbh_texture_slots if s.binding==5),None):
        from .material_ui import pixel_hash
        if pixel_hash(image)==image.get('dbh_pixels_hash'):return {},[]
    donor=texture_record(package,table[2].texture)
    if donor is None:raise ValueError('Native eye diffuse texture settings unavailable')
    if image:w,h,pixels=image_rgba(image,True)
    else:w=h=4;pixels=np.zeros((4,4,4),np.float32);pixels[:,:,3]=1
    levels=mip_chain(pixels,True,single_mip);settings=bytearray(donor.payload);settings[58]=1
    donor=Record(donor.index,donor.kind,donor.asset,donor.prefix,bytes(settings))
    digest=hashlib.sha256(b'DBH_EYE_EMISSION_TEXTURE_V1'+struct.pack('<II',w,h)+b''.join(levels)).digest()
    asset=0x60000000|(int.from_bytes(digest[:4],'little')&0xfffffff);lookup={r.asset:r for r in package.container.records}
    while asset in lookup:
        existing=lookup[asset];rec,data=make_texture(donor,existing.index,asset,w,h,levels)
        if existing.payload==rec.payload and existing.external and package.members[package.record_members[existing.index]].unpacked==data:break
        asset=0x60000000|((asset+1)&0xfffffff)
    else:rec,data=make_texture(donor,len(package.container.records),asset,w,h,levels)
    payload=bytearray(record.payload)
    if len(table)==5:
        struct.pack_into('<I',payload,79,6)
        payload[83+13*5:83+13*5]=struct.pack('<IIIB',0,2137,asset,0)
    else:struct.pack_into('<II',payload,table[5].offset,2137,asset);payload[table[5].offset+8]=0
    if bytes(payload)==record.payload:return {},[]
    if not experimental:raise ValueError('Eye shader/emission texture edits require Experimental Textures')
    validate_texture(rec,data);record.payload=bytes(payload);resources={}
    if asset not in lookup:package.container.records.append(rec);resources[rec.index]=data
    return resources,[dict(material=f'{record.asset:X}',binding=5,texture=f'{asset:X}',image=image.name if image else 'Black emission',role='EYE_EMISSION',width=w,height=h)]
