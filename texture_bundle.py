"""Lossless texture attachments for the DBH loose-file loader (DBHTEX01).

The native SEGS core is unchanged. Original QZIP resources, including all mip
levels, live after it; the loader maps their index entries into this same file.
This is a mod-loader extension, not an undocumented native game format.
"""
import struct
import zlib
from dataclasses import dataclass

MAGIC = b'DBHTEX01'
FOOTER = struct.Struct('<8sIIQQII')
ENTRY = struct.Struct('<IIQQII')


@dataclass(frozen=True)
class TextureAttachment:
    kind: int
    asset: int
    auxiliary: int
    data: bytes


def split_bundle(data):
    if len(data) < FOOTER.size or data[-FOOTER.size:-FOOTER.size+8] != MAGIC:
        return data, {}
    magic, version, count, core_size, directory, table_crc, core_crc = FOOTER.unpack_from(data, len(data)-FOOTER.size)
    if version != 1 or not 0 < count <= 10000 or not 16 <= core_size <= directory:
        raise ValueError('Invalid DBHTEX01 texture bundle header')
    if directory + count*ENTRY.size + FOOTER.size != len(data):
        raise ValueError('Truncated texture bundle directory')
    table = data[directory:-FOOTER.size]
    if zlib.crc32(table) != table_crc or zlib.crc32(data[:core_size]) != core_crc:
        raise ValueError('Texture bundle/core checksum mismatch')
    resources = {}; cursor = core_size
    for kind, asset, offset, size, auxiliary, checksum in ENTRY.iter_unpack(table):
        if kind not in (2229,2230) or not asset or (kind,asset) in resources:
            raise ValueError('Unsupported or duplicate texture attachment')
        if offset < cursor or not size or offset+size > directory or any(data[cursor:offset]):
            raise ValueError('Invalid texture attachment bounds/padding')
        raw = data[offset:offset+size]
        if zlib.crc32(raw) != checksum:
            raise ValueError(f'Texture attachment {kind}/{asset:X} checksum mismatch')
        resources[kind,asset] = TextureAttachment(kind,asset,auxiliary,raw)
        cursor = offset+size
    if any(data[cursor:directory]):raise ValueError('Unexpected data before texture directory')
    return data[:core_size], resources


def pack_bundle(core, resources):
    if not resources:return core
    if core[:4] != b'segs':raise ValueError('Texture bundle needs a native SEGS core')
    out = bytearray(core); table = bytearray()
    for key, item in sorted(resources.items()):
        if key != (item.kind,item.asset) or item.kind not in (2229,2230) or not item.data:
            raise ValueError('Invalid texture attachment')
        out += b'\0' * (-len(out) % 16)
        table += ENTRY.pack(item.kind,item.asset,len(out),len(item.data),item.auxiliary,zlib.crc32(item.data))
        out += item.data
    directory = len(out); out += table
    out += FOOTER.pack(MAGIC,1,len(resources),len(core),directory,zlib.crc32(table),zlib.crc32(core))
    return bytes(out)


def embed_dependencies(package, archive):
    """Include every FILETEXT dependency, not only currently visible materials."""
    from .materials import texture_info
    result = {}
    for record in package.container.records:
        if record.kind != 2137:continue
        for kind,asset,decoded_size in texture_info(record)['dependencies']:
            key = kind,asset
            if key in result:continue
            if key in package.texture_attachments:
                result[key] = package.texture_attachments[key]
            else:
                raw,auxiliary = archive.raw(kind,asset)
                result[key] = TextureAttachment(kind,asset,auxiliary,raw)
    return result
