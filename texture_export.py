"""Experimental self-contained FILETEXT v24 BGRA8 resources.

Keeps shader code and binding count unchanged. New texture resources have their
own SEGS streams, not global BigFile overrides. Renderer path: 140258050,
140256310, 1402580B0. This needs in-game verification for each shader family.
"""
import hashlib
import struct
import numpy as np
from .native import Record
from .materials import bindings, texture_record, texture_info


def image_rgba(image,srgb):
    w,h=tuple(image.size)
    if not 1<=w<=4096 or not 1<=h<=4096:
        raise ValueError('Custom texture dimensions must be 1..4096 in each direction')
    if image.source not in ('FILE','GENERATED') or image.type not in ('IMAGE','UV_TEST'):
        raise ValueError('Bake movie, tiled/UDIM, viewer and render textures to a single image first')
    values=np.empty(w*h*4,np.float32);image.pixels.foreach_get(values)
    values=values.reshape(h,w,4)[::-1].copy() # Blender bottom-up -> game top-down.
    if not np.isfinite(values).all():raise ValueError('Texture contains non-finite pixels')
    # Blender's byte-image pixel buffer exposes normalized encoded samples;
    # floating images (e.g. EXR) expose linear samples. Do not gamma-encode PNG twice.
    if srgb and image.is_float and image.colorspace_settings.name not in ('Non-Color','Raw'):
        rgb=values[:,:,:3]
        values[:,:,:3]=np.where(rgb<=0.0031308,12.92*rgb,1.055*np.maximum(rgb,0)**(1/2.4)-0.055)
    return w,h,np.clip(values,0,1)


def mip_chain(pixels,srgb=False,single_level=False):
    """Full chain down to the smaller dimension = 1 (engine shifts both axes)."""
    levels=[];a=pixels.copy()
    while True:
        levels.append(np.rint(np.clip(a[:,:,(2,1,0,3)],0,1)*255).astype(np.uint8).tobytes())
        h,w=a.shape[:2]
        if single_level or min(w,h)==1:break
        nw,nh=w//2,h//2
        # Two-by-two box filter. Odd final rows/columns are not separately sampled.
        y=(np.arange(nh)*h/nh).astype(int);yy=np.minimum(y+1,h-1)
        x=(np.arange(nw)*w/nw).astype(int);xx=np.minimum(x+1,w-1)
        linear=a.copy()
        if srgb:
            c=linear[:,:,:3];linear[:,:,:3]=np.where(c<=.04045,c/12.92,((c+.055)/1.055)**2.4)
        a=(linear[y[:,None],x]+linear[yy[:,None],x]+linear[y[:,None],xx]+linear[yy[:,None],xx])*.25
        if srgb:
            c=a[:,:,:3];a[:,:,:3]=np.where(c<=.0031308,c*12.92,1.055*np.maximum(c,0)**(1/2.4)-.055)
    return levels


def make_texture(template,index,asset,w,h,levels):
    info=texture_info(template)
    if info['depth']!=1 or template.payload[56]!=1:
        raise ValueError('Custom image export supports 2D textures, not cube/volume textures')
    d=bytearray(template.payload[:128])
    if len(d)!=128:raise ValueError('Truncated texture settings')
    d[44]=0 # 14067B150: engine format 0 = Vulkan BGRA8 (44/50).
    struct.pack_into('<HHHB',d,45,w,h,1,len(levels))
    d[57]=0 # Not runtime/generated engine texture.
    struct.pack_into('<I',d,60,asset) # Unique shared-content key.
    # No texture-streaming LOD list; not QD entropy compressed; quality/type unused.
    d+=b'\0\0\0\0'
    d+=b''.join(struct.pack('<I',len(level)) for level in levels)
    d+=b'\0'*12 # null RAW reference + decoded RAW size.
    d+=b'FTEXCRAW'+struct.pack('<I',1)
    raw=b''.join(levels)
    prefix=bytearray(52)
    struct.pack_into('<IBIII',prefix,0,len(d),1,1,0,1)
    struct.pack_into('<I',prefix,18,len(raw))
    struct.pack_into('<I',prefix,23,0xffffffff)
    struct.pack_into('<I',prefix,28,0xffffffff)
    struct.pack_into('<I',prefix,32,1)
    struct.pack_into('<I',prefix,40,len(raw))
    return Record(index,2137,asset,bytes(prefix),bytes(d)),raw


def validate_texture(record,raw):
    info=texture_info(record);d=record.payload
    if info['format']!=0 or info['depth']!=1 or d[128:132]!=b'\0'*4:
        raise ValueError('Unsupported generated texture layout')
    count=info['mips'];sizes=struct.unpack_from('<'+'I'*count,d,132)
    expected=[max(1,info['width']>>i)*max(1,info['height']>>i)*4 for i in range(count)]
    if list(sizes)!=expected or sum(sizes)!=len(raw):raise ValueError('Texture mip sizes do not match pixels')
    if d[132+4*count:]!=b'\0'*12+b'FTEXCRAW'+struct.pack('<I',1):
        raise ValueError('Unexpected texture suffix')
    return info


def export_materials(package,objects,experimental=False,fingerprints=None,single_mip=False):
    from .material_ui import pixel_hash,pixel_hash_many
    from .preset_shader import PRESETS
    mats={}
    for obj in objects:
        for mat in obj.data.materials:
            if mat and mat.dbh_shader_mode not in PRESETS and 'dbh_material_id' in mat and hasattr(mat,'dbh_texture_slots'):
                mid=mat['dbh_material_id']
                if mid in mats and mats[mid]!=mat:
                    raise ValueError(f'Material {mid:X} has multiple Blender copies; use one shared material before export')
                mats[mid]=mat
    resources={};report=[];used={r.asset for r in package.container.records};dedup={}
    if fingerprints is None:
        fingerprints=pixel_hash_many(slot.image for mat in mats.values() for slot in mat.dbh_texture_slots
                                     if slot.image is not None and slot.image==slot.original_image)
    for mid,mat in mats.items():
        from .preset_shader import PRESETS
        if mat.dbh_shader_mode in PRESETS:continue
        if not mat.dbh_texture_slots:continue
        record=next((r for r in package.container.records if r.kind==2133 and r.asset==mid),None)
        if record is None:raise ValueError(f'Unknown game material {mid:X}')
        table=bindings(record.payload);payload=bytearray(record.payload)
        for slot in mat.dbh_texture_slots:
            if mat.dbh_shader_mode=='EYE_EMISSION' and slot.binding==5:continue
            if slot.binding>=len(table):raise ValueError('Cannot add shader inputs: assign images to existing slots')
            b=table[slot.binding]
            if int(slot.asset,16)!=b.texture:raise ValueError('Material source binding mismatch; reimport')
            image=slot.image
            color_override=mid==0x14B2A and mat.dbh_shader_mode=='COAT_RGB' and slot.binding==21
            original=texture_record(package,b.texture)
            force_color_space=color_override and original is not None and not original.payload[58]
            if image==slot.original_image and not force_color_space:
                if image is None:continue
                pointer=image.as_pointer()
                if pointer not in fingerprints:fingerprints[pointer]=pixel_hash(image)
                if fingerprints[pointer]==image.get('dbh_pixels_hash'):continue
            if image is None:raise ValueError(f'{mat.name} slot {slot.binding}: image was removed; reset or assign another image')
            original=texture_record(package,b.texture)
            if original is None:raise ValueError('Cannot infer settings for a missing source texture')
            if not experimental:
                raise ValueError('Texture edits detected. Enable Experimental Textures in export; game validation is still required.')
            if force_color_space:
                settings=bytearray(original.payload);settings[58]=1
                original=Record(original.index,original.kind,original.asset,original.prefix,bytes(settings))
            srgb=bool(original.payload[58])
            w,h,rgba=image_rgba(image,srgb)
            # Keep inherited detail metrics meaningful and avoid extreme resource
            # growth. Resolution changes are supported up to the stated limit.
            levels=mip_chain(rgba,srgb,single_mip)
            key=hashlib.sha256(original.payload[:128]+b''.join(levels)).digest()
            if key in dedup:asset=dedup[key]
            else:
                asset=0x60000000 | (int.from_bytes(key[:4],'little') & 0x0fffffff)
                while asset in used:asset=0x60000000 | ((asset+1)&0x0fffffff)
                used.add(asset);dedup[key]=asset
                rec,raw=make_texture(original,len(package.container.records),asset,w,h,levels)
                validate_texture(rec,raw)
                package.container.records.append(rec);resources[rec.index]=raw
            struct.pack_into('<I',payload,b.offset+4,asset)
            report.append(dict(material=f'{mid:X}',binding=b.index,original=f'{b.texture:X}',
                               texture=f'{asset:X}',image=image.name,width=w,height=h,mips=len(levels)))
        record.payload=bytes(payload)
        if mat.dbh_shader_mode=='EYE_EMISSION':
            from .eye_emission import export_texture
            eye_resources,eye_report=export_texture(package,mat,record,experimental,single_mip)
            resources.update(eye_resources);report.extend(eye_report)
    return resources,report
