"""Opt-in, fingerprinted shader edits for the verified PC Connor coat.

This is NOT a general shader compiler. Keep every interface, descriptor,
instruction result ID, byte count and resource offset unchanged. Only replace
the final base-color expression with the existing slot-21 RGB sample on UV0.
"""
import hashlib
import struct
from .native import Reader, align
from .materials import bindings

PROFILES = {
    (0,2): ('470ee2253b8ffd05131adb955c62705fe68e985953e812d4655b319791d9f86b',
            '346bae8bace0b12889f6d08993811877', None),
    (0,3): ('db3d1c19418875813ecdf219d568cd65aa89ab7207a3aa26648e72c079a8efd6',
            '49b3bcbf7df02ac809a3b8720d4a6eaa', (46,30081,27373,27378,24102,24453)),
    (8,2): ('1760686d5ebd7f4829b1ba70a12cfdc877e0496feb2c178bcafd67c5ede49f95',
            '03e5aeef9357704a33f58cff380618a6', None),
    (8,3): ('ee8cda54a1db8fdc516b037ba6651c8eab9a76491f3763f43ec6fc5717f6b7b8',
            '67e37de2b21c8eb4a4282b571c4046f2', (17,25513,22805,22810,19534,19885)),
}


def _fragment_variants(record,raw,legacy=False):
    # SHADCUST reader 140226EE0 / shader-code reader 14067F510.
    r=Reader(record.payload,83+13*len(bindings(record.payload)))
    nv,nf=r.uint(),r.uint()
    if nv>256 or nf>64:raise ValueError('Unsupported shader program count')
    offset=0;result=[]
    for j in range(nv+nf):
        external=j>=nv
        program=r.uint()
        if not external:r.uint() # vertex variant
        count=r.uint()
        if not count:continue
        if count>8:raise ValueError('Unsupported shader variant count')
        stage,flags=r.uint(),r.uint()
        for _ in range(count):
            size=r.uint()
            if not size:continue
            typ=r.uint();hash_at=r.pos;r.take(16)
            if external:
                if offset+size>len(raw):raise ValueError('Truncated shader stream')
                result.append((program,typ,hash_at,offset,size))
                # Native binder 14022A250 rounds strictly UP, including when
                # size is already aligned. Older addon exports used ceil().
                offset+=(size+63 if legacy else size+64)&~63
            else:r.take(size)
    return result


def _validate_chunks(variants,raw,zero_padding=False):
    cursor=0
    for program,typ,_,offset,size in variants:
        # Vanilla alignment gaps contain uninitialized compiler memory; only
        # our legacy exporter promises zero-filled padding.
        if zero_padding and any(raw[cursor:offset]):raise ValueError('Nonzero shader padding')
        chunk=raw[offset:offset+size]
        if typ==2:
            if not chunk.startswith(b'#version') or not chunk.endswith(b'\0'):
                raise ValueError(f'Invalid GLSL header/terminator in program {program}')
        elif typ==3:
            if len(chunk)<12 or chunk[:4]!=b'QDIF':
                raise ValueError(f'Invalid QDIF header in program {program}')
            prefix=struct.unpack_from('<I',chunk,4)[0]
            if prefix<12 or prefix+20>size or (size-prefix)%4 or chunk[prefix:prefix+4]!=b'\x03\x02\x23\x07':
                raise ValueError(f'Invalid SPIR-V header in program {program}')
        else:raise ValueError(f'Unsupported external shader backend {typ}')
        cursor=offset+size
    if zero_padding and any(raw[cursor:]):raise ValueError('Unknown bytes after shader stream')


def fragment_variants(record,raw,allow_legacy=True):
    try:
        variants=_fragment_variants(record,raw)
        _validate_chunks(variants,raw)
        return variants
    except ValueError as error:
        # Recover only our marked preset exports, not arbitrary damaged game
        # resources. Validate ALL chunks and gaps before accepting old offsets.
        marked=any(marker in raw[:230000] for marker in
                   (b'// DBH_STANDARD_V1 ',b'// DBH_CEL_V1 ',b'// DBH_PBR_V1 '))
        if not allow_legacy or not marked:raise
        try:
            variants=_fragment_variants(record,raw,legacy=True)
            _validate_chunks(variants,raw,zero_padding=True)
            return variants
        except ValueError:
            raise error


def append_shader(out,chunk):
    out+=chunk
    out+=b'\0'*(64-len(chunk)%64)


def normalize_stream(record,raw):
    """Keep native streams exact; repair only validated legacy preset layout."""
    variants=fragment_variants(record,raw)
    try:
        fragment_variants(record,raw,allow_legacy=False)
        return raw
    except ValueError:
        out=bytearray()
        for _,_,_,offset,size in variants:append_shader(out,raw[offset:offset+size])
        out+=b'\0'*(-len(out)%128)
        fragment_variants(record,out,allow_legacy=False)
        return bytes(out)


def validate_native_stream(record,raw):
    """Independent native Vulkan pointer walk; do not trust parser offsets."""
    from itertools import groupby
    variants=fragment_variants(record,raw,allow_legacy=False)
    cursor=0
    for program,group in groupby(variants,key=lambda item:item[0]):
        entries=list(group)
        # The supported PC presets carry exactly GLSL then Vulkan.
        if [e[1] for e in entries]!=[2,3]:
            raise ValueError(f'Unsupported shader backend grouping for {record.asset:X}/{program}')
        before,selected=entries[0][4],entries[1][4]
        cursor+=((before//64)+1)*64
        if raw[cursor:cursor+4]!=b'QDIF':
            raise ValueError(f'Native shader offset invalid for {record.asset:X}/{program}')
        cursor+=((selected//64)+1)*64



def _expressions(profile):
    if profile is None:
        old=b'\r\n\tMaterialAttributes.baseColor = g_v435;'
        new=b'\nMaterialAttributes.baseColor=g_v640.rgb;'
        new=new.ljust(len(old),b' ')
    else:
        typ,result,a,b,c,sample=profile
        old=struct.pack('<8I',8<<16|12,typ,result,1,46,a,b,c) # GLSL.std.450 FMix
        new=struct.pack('<8I',8<<16|79,typ,result,sample,sample,0,1,2) # RGB shuffle
    if len(old)!=len(new):raise ValueError('Shader profile changes serialized size')
    return old,new


def coat_rgb(record,raw,enabled):
    if record.asset!=0x14B2A:raise ValueError('Direct RGB profile only supports Connor coat 14B2A')
    out=bytearray(raw);payload=bytearray(record.payload);found=set()
    for program,typ,hash_at,offset,size in fragment_variants(record,raw):
        key=program,typ
        if key not in PROFILES:continue
        fingerprint,original_key,profile=PROFILES[key]
        old,new=_expressions(profile);chunk=raw[offset:offset+size]
        normalized=chunk.replace(new,old)
        if normalized.count(old)!=1 or hashlib.sha256(normalized).hexdigest()!=fingerprint:
            raise ValueError('Unknown coat shader revision; direct RGB was not applied')
        updated=normalized.replace(old,new) if enabled else normalized
        out[offset:offset+size]=updated
        # Serialized 128-bit shader identity, consumed by 14067F510. Use a
        # distinct deterministic identity so the original cached program is not reused.
        payload[hash_at:hash_at+16]=(hashlib.sha256(b'DBH_COAT_RGB_V1'+updated).digest()[:16]
                                    if enabled else bytes.fromhex(original_key))
        found.add(key)
    if found!=set(PROFILES):raise ValueError('Coat shader profile is incomplete')
    return bytes(payload),bytes(out)


def is_coat_rgb(record,raw):
    if record.asset!=0x14B2A:return False
    try:
        _,patched=coat_rgb(record,raw,True)
        return patched==raw
    except ValueError:return False


def export_shaders(package,objects,experimental):
    resources={};reports=[]
    from .eye_emission import build_shader as eye_shader,detect as is_eye
    eye_mats={m for o in objects for m in o.data.materials if m and m.get('dbh_material_id') is not None}
    for mat in eye_mats:
        record=next((r for r in package.container.records if r.kind==2133 and r.asset==mat['dbh_material_id']),None)
        if record is None or not record.external:continue
        raw=package.members[package.record_members[record.index]].unpacked
        enabled=mat.dbh_shader_mode=='EYE_EMISSION'
        if not enabled and is_eye(raw) is None:continue
        payload,updated=eye_shader(record,raw,mat.dbh_eye_emission_strength if enabled else None)
        if payload==record.payload and updated==raw:continue
        if not experimental:raise ValueError('Native eye shader edits require Experimental Textures')
        record.payload=payload;resources[record.index]=updated
        reports.append(dict(material=f'{record.asset:X}',mode=mat.dbh_shader_mode,emission_binding=5,
                            eye_emission_strength=mat.dbh_eye_emission_strength if enabled else 0,vertex_programs='unchanged'))
    from .hair_rgb_shader import build_shader as hair_rgb,detect as is_hair_rgb
    from .preset_shader import PRESETS
    hair_mats={m for o in objects for m in o.data.materials if m and m.dbh_shader_mode not in PRESETS and m.get('dbh_material_id') is not None}
    for mat in hair_mats:
        record=next((r for r in package.container.records if r.kind==2133 and r.asset==mat['dbh_material_id']),None)
        if record is None or not record.external:continue
        raw=package.members[package.record_members[record.index]].unpacked
        enabled=mat.dbh_shader_mode=='HAIR_RGB'
        if not enabled and not is_hair_rgb(raw):continue
        payload,updated=hair_rgb(record,raw,enabled)
        if payload==record.payload and updated==raw:continue
        if not experimental:raise ValueError('Game shader edits require Experimental Textures')
        record.payload=payload;resources[record.index]=updated
        reports.append(dict(material=f'{record.asset:X}',mode=mat.dbh_shader_mode,binding=1,uv='UV1',cloth_rebuilt=False))
    mats={m for o in objects for m in o.data.materials if m and m.get('dbh_material_id')==0x14B2A}
    for mat in mats:
        from .preset_shader import PRESETS
        if mat.dbh_shader_mode in PRESETS:continue
        record=next(r for r in package.container.records if r.kind==2133 and r.asset==0x14B2A)
        raw=package.members[package.record_members[record.index]].unpacked
        enabled=mat.dbh_shader_mode=='COAT_RGB'
        if not enabled and not is_coat_rgb(record,raw):continue
        payload,updated=coat_rgb(record,raw,enabled)
        if payload==record.payload and updated==raw:continue
        if not experimental:raise ValueError('Game shader edits require Experimental Textures')
        record.payload=payload;resources[record.index]=updated
        reports.append(dict(material='14B2A',mode=mat.dbh_shader_mode,binding=21,uv='UV1'))
    return resources,reports
