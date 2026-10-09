"""Disposable, compressed-only PC archive for the legacy reference reader.

Never opens game archives for writing. All resources come from the selected
SEGS file; only framing and external offsets change in this private copy.
"""
import struct
import zlib
import re
from contextlib import ExitStack
from pathlib import Path
from .native import Package
from .segs import align, HEADER, ENTRY, scan_members,compress_chunks


def compressed_member(raw, attributes=1):
    # Leave room for zlib overhead inside the uint16 packed-size field.
    chunks=[raw[i:i+60000] for i in range(0,len(raw),60000)]
    if not chunks or len(chunks)>65535: raise ValueError('Reference member exceeds SEGS limits')
    base=align(16+8*len(chunks));out=bytearray(base)
    for i,encoded in enumerate(compress_chunks(chunks,4 if len(chunks)>=16 else 1)):
        chunk=chunks[i]
        if len(encoded)==len(chunk): encoded=zlib.compress(chunk,0)
        if len(encoded)>65535: raise ValueError('Reference compressed block too large')
        out+=b'\0'*(align(len(out))-len(out))
        ENTRY.pack_into(out,16+i*8,len(encoded),len(chunk),len(out)-base+1)
        out+=encoded
    HEADER.pack_into(out,0,b'segs',attributes,len(chunks),len(raw),len(out)-base)
    return bytes(out)


def compatible_package(data):
    package=Package(data);body=bytearray();locations={}
    for member in package.members[1:]:
        body+=b'\0'*(align(len(body))-len(body))
        packed=compressed_member(member.unpacked,member.attributes)
        locations[member.index]=(len(body),len(packed),len(member.unpacked))
        body+=packed
    for record in package.container.records:
        if record.index not in package.record_members: continue
        offset,packed_size,raw_size=locations[package.record_members[record.index]]
        prefix=bytearray(record.prefix)
        for at,value in ((18,raw_size),(36,packed_size),(40,raw_size),(44,offset)):
            struct.pack_into('<I',prefix,at,value)
        record.prefix=bytes(prefix)
    output=compressed_member(package.container.pack(),package.members[0].attributes)+body
    verified=Package(output)
    if len(verified.members)!=len(package.members): raise ValueError('Reference copy member count mismatch')
    if any(a.unpacked!=b.unpacked for a,b in zip(package.members[1:],verified.members[1:])):
        raise ValueError('Reference copy changed a resource')
    if any(a.payload!=b.payload for a,b in zip(package.container.records,verified.container.records)):
        raise ValueError('Reference copy changed metadata payloads')
    if any(b.packed_size==b.unpacked_size for m in verified.members for b in m.blocks):
        raise ValueError('Reference copy still contains a stored block')
    return output


def prepare_archive(source, game_index, directory):
    raw=Path(source).read_bytes();package=Package(raw);packed=compatible_package(raw)
    index=Path(game_index).read_bytes()
    if (index[:20]!=b'QUANTICDREAMTABINDEX' or len(index)<105 or (len(index)-105)%28
            or struct.unpack_from('>I',index,20)[0]!=18):
        raise ValueError('Unsupported PC BigFile index')
    entries=list(struct.iter_unpack('>7I',index[105:]))
    matches=[entry for entry in entries if entry[0]==29 and entry[4]==len(package.data)]
    entry=matches[0] if len(matches)==1 else (29,1,1,0,0,1,0)
    code=f'{entry[2]:X}'
    directory=Path(directory)
    destination=directory/'BigFile_PC.idx'
    items=[struct.pack('>7I',29,entry[1],entry[2],2048,len(packed),entry[5] or 1,0)]
    # Texture FILETEXT objects point to separate raw/LOD resources in the
    # installed index. Copy ONLY those dependencies, reading the game read-only.
    needed=set()
    for record in package.container.records:
        if record.kind!=2137: continue
        for match in re.finditer(b'[\xb5\xb6]\x08\x00\x00',record.payload):
            if match.start()+8<=len(record.payload):
                needed.add(struct.unpack_from('<2I',record.payload,match.start()))
    with ExitStack() as stack:
        output=stack.enter_context((directory/'BigFile_PC.dat').open('wb'))
        output.write(b'\0'*2048+packed)
        opened={}
        for resource in entries:
            kind,unknown,asset,offset,size,decoded,file_number=resource
            if (kind,asset) not in needed: continue
            if (kind,asset) in package.texture_attachments:
                attachment=package.texture_attachments[kind,asset]
                value=attachment.data;decoded=attachment.auxiliary;size=len(value)
            else:
                if file_number not in opened:
                    path=Path(game_index).with_suffix('.dat' if file_number==0 else f'.d{file_number:02d}')
                    opened[file_number]=stack.enter_context(path.open('rb'))
                handle=opened[file_number];handle.seek(offset);value=handle.read(size)
            if len(value)!=size: raise ValueError(f'Truncated texture resource {asset:X}')
            if value[:4]==b'segs':
                members=scan_members(value)
                if len(members)!=1 or any(value[members[0].end:]):
                    raise ValueError(f'Unsupported texture resource framing {asset:X}')
                value=compressed_member(members[0].unpacked,members[0].attributes)
            output.write(b'\0'*(align(output.tell())-output.tell()))
            items.append(struct.pack('>7I',kind,unknown,asset,output.tell(),len(value),decoded,0))
            output.write(value)
    header=bytearray(index[:105]);struct.pack_into('>I',header,101,len(items))
    destination.write_bytes(header+b''.join(items))
    return code,destination
