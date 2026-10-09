"""Bounded PC SHADCUST v91 bindings and read-only archive texture resolution.

Binding reader: 140227670 / 140227520. Never scan arbitrary shader bytes for IDs.
Texture FILETEXT v24: 140256310. Asset bytes remain separate from shader inputs.
"""
import struct
from dataclasses import dataclass
from pathlib import Path
from .native import Reader, u32
from .segs import scan_members


@dataclass(frozen=True)
class Binding:
    index: int
    offset: int
    texture: int
    semantic: int
    flags: int


def bindings(payload):
    if payload[:44] != (struct.pack('<I',3)+b'COM_CONT'+struct.pack('<II',6,0)
                        +b'LOADCONT'+struct.pack('<I',2)+b'SHADCUST'+struct.pack('<I',91)):
        raise ValueError('Unsupported material header; expected PC SHADCUST v91')
    r=Reader(payload,79)
    count=r.uint()
    if count>255: raise ValueError('Invalid shader texture count')
    result=[]
    for i in range(count):
        semantic=r.uint();offset=r.pos
        kind=r.uint();asset=r.uint()
        if kind not in (0,2137): raise ValueError(f'Unsupported shader texture reference {kind}')
        result.append(Binding(i,offset,asset,semantic,r.byte()))
    return result


def texture_record(package, asset):
    return next((r for r in package.container.records if r.kind==2137 and r.asset==asset),None)


def texture_info(record):
    d=record.payload
    if d[32:44] != b'FILETEXT'+struct.pack('<I',24):
        raise ValueError('Unsupported FILETEXT version')
    fmt,w,h,depth,mips=struct.unpack_from('<BHHHB',d,44)
    refs=[]
    if record.prefix[4]==1 and u32(record.prefix,9):
        count=u32(record.prefix,5)
        if len(record.prefix)!=22+12*count: raise ValueError('Invalid texture dependency prefix')
        refs=[struct.unpack_from('<III',record.prefix,22+12*i) for i in range(count)]
        if any(k not in (2229,2230) for k,a,s in refs): raise ValueError('Unknown texture dependency kind')
    return dict(format=fmt,width=w,height=h,depth=depth,mips=mips,dependencies=refs)


class ArchiveTextures:
    """Read only the referenced texture resources; never writes installed files."""
    def __init__(self,index):
        from .game_paths import normalize_index
        self.index=normalize_index(index);self.entries=None

    def raw(self,kind,asset):
        if self.entries is None:
            if self.index is None:
                raise ValueError('Choose your Steam or Epic BigFile_PC.idx in the Detroit add-on preferences; this resource is not bundled')
            if not self.index.is_file():
                raise FileNotFoundError(f'Selected BigFile_PC.idx not found: {self.index}. No default-path fallback was used.')
            d=self.index.read_bytes()
            if d[:20]!=b'QUANTICDREAMTABINDEX' or len(d)<105 or (len(d)-105)%28:
                raise ValueError('Invalid PC archive index')
            self.entries={(v[0],v[2]):v for v in struct.iter_unpack('>7I',d[105:])}
        e=self.entries.get((kind,asset))
        if e is None: raise ValueError(f'Archive has no texture {kind}/{asset:X}')
        path=self.index.with_suffix('.dat' if e[6]==0 else f'.d{e[6]:02d}')
        with path.open('rb') as f:
            f.seek(e[3]);data=f.read(e[4])
        if len(data)!=e[4]: raise ValueError('Truncated archive texture')
        return data,e[5]

    def read(self,kind,asset,package=None):
        attachment=package.texture_attachments.get((kind,asset)) if package is not None else None
        data=attachment.data if attachment is not None else self.raw(kind,asset)[0]
        if data[:4]==b'QZIP':
            if len(data)<32: raise ValueError('Truncated QZIP header')
            data=data[32:]
        if data[:4]==b'segs':
            members=scan_members(data)
            if len(members)!=1: raise ValueError('Unexpected texture SEGS count')
            data=members[0].unpacked
        return data

    def image(self,record,package=None):
        from .textures import decode_qd, decode_gpu, decode_qd_raw
        info=texture_info(record);deps=info['dependencies']
        if record.external and package is not None:
            data=package.members[package.record_members[record.index]].unpacked
            return decode_gpu(info['width'],info['height'],data,info['format'])
        if not deps: raise ValueError('Texture has no supported pixel dependency')
        # FILETEXT level zero is full resolution; base RAW is the streaming tail.
        kind,asset,size=deps[1 if len(deps)>1 else 0]
        data=self.read(kind,asset,package)
        if data[:2]==b'\x10\0':
            if int.from_bytes(data[2:4],'little')==255:
                return decode_qd_raw(data,info['format'])
            return decode_qd(data)
        return decode_gpu(info['width'],info['height'],data,info['format'])
