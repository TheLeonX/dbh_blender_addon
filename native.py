"""Native PC container, MESHDATA v41 and growing geometry serializer.

Field order verified against Detroit's v41 reader at 0x140296B40.
No Blender dependency: the same codec validates exported packages in tests.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

from .segs import align, scan_members, encode_member, read_member, relocate_member, validate_streaming


def u32(data, offset):
    return struct.unpack_from('<I', data, offset)[0]


class Reader:
    def __init__(self, data, offset=0):
        self.data, self.pos = data, offset

    def take(self, size):
        if size < 0 or self.pos + size > len(self.data):
            raise ValueError(f'Truncated data at 0x{self.pos:X}')
        value = self.data[self.pos:self.pos + size]
        self.pos += size
        return bytes(value)

    def uint(self):
        return struct.unpack('<I', self.take(4))[0]

    def byte(self):
        return self.take(1)[0]


@dataclass
class Record:
    index: int
    kind: int
    asset: int
    prefix: bytes
    payload: bytes

    @property
    def external(self):
        return self.prefix[4] == 1 and u32(self.prefix, 9) == 0


class Container:
    def __init__(self, data, allow_trailing=False):
        if data[:8] != b'DC_INFO ':
            raise ValueError('Expected a decompressed DATA_CONTAINER')
        count = u32(data, 20)
        r = Reader(data, 24 + 8 * count)
        if r.take(8) != b'DC_DATA ':
            raise ValueError('DC_DATA header missing')
        r.take(8)
        self.header = bytes(data[:r.pos])
        self.records = []
        for i in range(count):
            start = r.pos
            size, flag = r.uint(), r.byte()
            if flag == 0:
                r.take(8)
            elif flag == 1:
                n, key = r.uint(), r.uint()
                r.take(5)
                r.take(4 + n * 12 if key else 34)
            else:
                raise ValueError(f'Unsupported record flag {flag}')
            prefix = bytes(data[start:r.pos])
            kind, asset = struct.unpack_from('<II', data, 24 + 8 * i)
            self.records.append(Record(i, kind, asset, prefix, r.take(size)))
        self.end = r.pos
        if not allow_trailing and r.pos != len(data):
            raise ValueError('Unframed bytes after DATA_CONTAINER records')

    def pack(self, records=None):
        records = self.records if records is None else records
        out = bytearray(self.header)
        directory = b''.join(struct.pack('<II', rec.kind, rec.asset) for rec in records)
        if len(records)!=u32(self.header,20) or directory != self.header[24:-16]:
            out=bytearray(self.header[:24])
            struct.pack_into('<I',out,12,8+8*len(records))
            struct.pack_into('<I',out,20,len(records))
            out+=directory
            out+=self.header[-16:-4]+struct.pack('<I',sum(len(r.prefix)+len(r.payload) for r in records))
        for rec in records:
            out += struct.pack('<I', len(rec.payload)) + rec.prefix[4:] + rec.payload
        return bytes(out)


@dataclass
class VB:
    count: int
    flag: int
    layout: bytes
    strides: tuple
    streams: list = field(default_factory=list)
    inline_streams: dict = field(default_factory=dict)

    def attributes(self):
        return [tuple(self.layout[i:i + 4]) for i in range(0, len(self.layout), 4)]


@dataclass
class Submesh:
    descriptor: bytes
    suffix: bytes

    def value(self, offset):
        return u32(self.descriptor, offset)

    @property
    def vb(self): return self.value(6)
    @property
    def first_vertex(self): return self.value(10)
    @property
    def count(self): return self.value(14)
    @property
    def ib(self): return self.value(18)
    @property
    def first_index(self): return self.value(22)
    @property
    def index_count(self): return self.value(26)
    @property
    def material(self): return struct.unpack_from('<II', self.descriptor, 38 + self.value(34) * 4)
    @property
    def detail_offset(self): return 46 + self.value(34) * 4
    @property
    def detail_count(self): return self.value(self.detail_offset)
    @property
    def detail_table(self): return self.descriptor[self.detail_offset:]

    def replaced(self, first, count, first_index, index_count):
        # These 48-byte entries are texture-detail metrics, NOT morph ranges.
        # The runtime indexes them through the retained material references.
        # Keep the source metrics for replacement geometry (approximate LOD).
        desc = bytearray(self.descriptor)
        for offset, value in ((10, first), (14, count), (22, first_index), (26, index_count)):
            struct.pack_into('<I', desc, offset, value)
        return Submesh(bytes(desc), self.suffix)


@dataclass
class Group:
    header: bytes
    meshes: list
    suffix: bytes


class MeshData:
    def __init__(self, payload):
        r = Reader(payload)
        self.preamble = r.take(4)
        if r.take(8) != b'MESHDATA' or r.uint() != 41:
            raise ValueError('This mesh requires the PC MESHDATA v41 format')
        headers = [(r.uint(), r.uint(), r.byte()) for _ in range(r.uint())]
        self.index_counts = [r.uint() for _ in range(r.uint())]
        self.groups = []
        for _ in range(r.uint()):
            count, header = r.uint(), r.take(8)
            meshes = []
            for _ in range(count):
                start = r.pos
                if r.uint() != 3 or r.byte() != 0:
                    raise ValueError('Unsupported submesh descriptor')
                r.take(29)
                remaps = r.uint()
                r.take(remaps * 4 + 8)
                detail_count = r.uint()
                if detail_count > 255:
                    raise ValueError('Texture-detail count exceeds runtime uint8 capacity')
                r.take(detail_count * 48)
                descriptor = bytes(payload[start:r.pos])
                meshes.append(Submesh(descriptor, r.take(80)))
            self.groups.append(Group(header, meshes, r.take(78)))
        start = r.pos
        refs = r.uint()
        r.take((refs & 255) * 8 + (4 if refs & 256 else 0) + 4)
        self.references = bytes(payload[start:r.pos])
        self.vbs = []
        for n, count, flag in headers:
            layout = r.take(n * 4)
            strides, inline = [], {}
            for stream in range(4):
                stride = r.uint()
                strides.append(stride)
                # v41 reader 0x140321120: flag 1 embeds stream 1 after
                # that stream's stride, not after all four strides.
                if flag == 1 and stream == 1 and stride:
                    inline[stream] = r.take(count * stride)
            self.vbs.append(VB(count, flag, layout, tuple(strides), inline_streams=inline))
        start = r.pos
        r.take(r.uint())  # optional length-prefixed name/blob
        r.take(18)
        self.tail = bytes(payload[start:r.pos])
        if payload[r.pos:r.pos + 8] != b'CLUPSKME':
            raise ValueError('MESHDATA tail did not end at CLUPSKME')
        self.deformations = bytes(payload[r.pos:])
        self.indices = []

    def flat(self):
        return [sub for group in self.groups for sub in group.meshes]

    def load_geometry(self, raw):
        r = Reader(raw)
        self.indices = [bytearray(r.take(n * 2)) for n in self.index_counts]
        self.padding = []
        for vb in self.vbs:
            self.padding.append(r.take(align(r.pos) - r.pos))
            vb.streams = [bytearray(vb.inline_streams[i]) if i in vb.inline_streams
                          else bytearray(r.take(stride * vb.count))
                          for i, stride in enumerate(vb.strides)]
        external_end = r.pos
        self.end_padding = r.take(align(r.pos) - r.pos)
        self.inline_reservation = b''
        inline_size = sum(len(data) for vb in self.vbs for data in vb.inline_streams.values())
        reservation_sizes = {inline_size, align(external_end + inline_size) - r.pos}
        if inline_size and len(raw) - r.pos in reservation_sizes:
            # Flag-1 stream 1 is serialized inline in MESHDATA, but its
            # allocation remains reserved in the external member. CURTIS
            # contains nonzero opaque bytes here (including shader fragments),
            # not another vertex stream. Preserve them without interpreting
            # or zeroing them. Only accept the exact allocation/alignment sizes.
            self.inline_reservation = r.take(len(raw) - r.pos)
        self.inline_reservation_size = inline_size
        if r.pos != len(raw):
            raise ValueError(f'Unknown geometry suffix: {len(raw) - r.pos} bytes')
        self.vertex_origins = {}
        # Position-only collision groups use an implicit packed vertex cursor.
        # Require a full partition AND every serialized OBB to contain its
        # candidate vertices. Never infer this for overlapping render ranges.
        for vi, vb in enumerate(self.vbs):
            draws = [s for s in self.flat() if s.vb == vi]
            if (vb.attributes() != [(1, 0, 2, 0)] or len(draws) < 2 or
                    any(s.first_vertex for s in draws) or sum(s.count for s in draws) != vb.count):
                continue
            first, candidates, valid = 0, {}, True
            for s in draws:
                box = struct.unpack_from('<15f', s.suffix, 4)
                if not all(math.isfinite(x) for x in box) or any(x <= 0 for x in box[12:]):
                    valid = False; break
                for vertex in struct.iter_unpack('<3f', vb.streams[1][first*12:(first+s.count)*12]):
                    if any(abs(sum((vertex[j]-box[j])*box[3+3*k+j] for j in range(3))) >
                           box[12+k]+2e-5 for k in range(3)):
                        valid = False; break
                if not valid: break
                candidates[id(s)] = first
                first += s.count
            if valid: self.vertex_origins.update(candidates)

    def vertex_origin(self, sub):
        return getattr(self, 'vertex_origins', {}).get(id(sub), sub.first_vertex)

    def pack_geometry(self):
        out = bytearray().join(self.indices)
        self.index_counts = [len(buf) // 2 for buf in self.indices]
        for i, vb in enumerate(self.vbs):
            pad_size = align(len(out)) - len(out)
            out += self.padding[i] if len(self.padding[i]) == pad_size else b'\0' * pad_size
            for i_stream, (stride, stream) in enumerate(zip(vb.strides, vb.streams)):
                if len(stream) != stride * vb.count:
                    raise ValueError('Vertex buffer size mismatch')
                if i_stream in vb.inline_streams:
                    vb.inline_streams[i_stream] = bytes(stream)
                else:
                    out += stream
        pad_size = align(len(out)) - len(out)
        out += self.end_padding if len(self.end_padding) == pad_size else b'\0' * pad_size
        if getattr(self, 'inline_reservation', b''):
            size = sum(len(data) for vb in self.vbs for data in vb.inline_streams.values())
            if size != self.inline_reservation_size:
                raise ValueError('Resizing cloth output reservations requires a rebuilt cloth mapping')
            out += self.inline_reservation
        return bytes(out)

    def pack(self, clear_deformations=False):
        out = bytearray(self.preamble + b'MESHDATA' + struct.pack('<II', 41, len(self.vbs)))
        for vb in self.vbs:
            out += struct.pack('<IIB', len(vb.layout) // 4, vb.count, vb.flag)
        out += struct.pack('<I', len(self.index_counts))
        out += struct.pack('<' + 'I' * len(self.index_counts), *self.index_counts)
        out += struct.pack('<I', len(self.groups))
        for group in self.groups:
            out += struct.pack('<I', len(group.meshes)) + group.header
            for sub in group.meshes:
                if len(sub.detail_table) != 4 + 48 * sub.detail_count:
                    raise ValueError('Invalid texture-detail table length')
                out += sub.descriptor + sub.suffix
            out += group.suffix
        out += self.references
        for vb in self.vbs:
            out += vb.layout
            for i, stride in enumerate(vb.strides):
                out += struct.pack('<I', stride)
                if i in vb.inline_streams:
                    out += vb.inline_streams[i]
        out += self.tail
        if clear_deformations:
            out += b'CLUPSKME' + struct.pack('<IB', 2, 0)
            out += b'BLSHAPES' + struct.pack('<I', 3) + b'\0' * (4 * len(self.groups))
        else:
            out += self.deformations
        return bytes(out)


def unpack_attribute(fmt, raw, offset):
    if fmt == 2: return struct.unpack_from('<3f', raw, offset)
    if fmt in (4, 8):
        values = struct.unpack_from('<4B' if fmt == 4 else '<4b', raw, offset)
        return tuple(v / (255 if fmt == 4 else 127) for v in values)
    if fmt in (5, 6): return struct.unpack_from('<2e' if fmt == 5 else '<4e', raw, offset)
    if fmt == 13:
        word = u32(raw, offset)
        values = [(word >> (10 * i)) & 1023 for i in range(3)]
        return tuple(max(-1.0, (v - 1024 if v >= 512 else v) / 511.0) for v in values)
    raise ValueError(f'Unsupported vertex format {fmt}')


def pack_attribute(fmt, values):
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Mesh contains a non-finite vertex attribute')
    if fmt == 2: return struct.pack('<3f', *values[:3])
    if fmt in (4, 8):
        scale = 255 if fmt == 4 else 127
        lo = 0 if fmt == 4 else -127
        v = [min(scale, max(lo, round(x * scale))) for x in values[:4]]
        return struct.pack('<4B' if fmt == 4 else '<4b', *v)
    if fmt in (5, 6):
        return struct.pack('<2e' if fmt == 5 else '<4e', *values[:2 if fmt == 5 else 4])
    if fmt == 13:
        word = sum((max(-511, min(511, round(x * 511))) & 1023) << (10 * i)
                   for i, x in enumerate(values[:3]))
        return struct.pack('<I', word)
    raise ValueError(f'Unsupported vertex format {fmt}')


def decode_vertices(md, sub):
    vb = md.vbs[sub.vb]
    declaration = vb.attributes()
    if any(stream >= len(vb.streams) for stream, *_ in declaration):
        raise ValueError('Cloth virtual vertex streams cannot be decoded as ordinary bone weights')
    attributes = [(vb.streams[stream], vb.strides[stream], offset, fmt, semantic)
                  for stream, offset, fmt, semantic in declaration]
    first = md.vertex_origin(sub)
    result = []
    for local in range(sub.count):
        index = first + local
        attrs = {semantic: unpack_attribute(fmt, raw, index * stride + offset)
                 for raw, stride, offset, fmt, semantic in attributes}
        weights = attrs.get(6, ()) + attrs.get(8, ())
        bones = tuple(round(x / 3) + 1 for x in attrs.get(7, ()) + attrs.get(9, ()))
        uv_values = attrs.get(4, ()) + attrs.get(5, ())
        result.append(dict(position=attrs[0], normal=attrs.get(1, (0, 0, 1)),
                           color=attrs.get(2, (1, 1, 1, 1)),
                           uvs=tuple(zip(uv_values[::2], uv_values[1::2])),
                           bones=bones, weights=weights))
    return result


def decode_faces(md, sub):
    data = md.indices[sub.ib]
    return [(c, b, a) for a, b, c in struct.iter_unpack(
        '<3H', data[sub.first_index * 2:(sub.first_index + sub.index_count) * 2])]


def append_mesh(md, sub, vertices, faces):
    """Append independent full vertex records; return a replacement draw."""
    if len(vertices) > 65535:
        raise ValueError('Split mesh into 16-bit draw batches first')
    vb = md.vbs[sub.vb]
    if vb.inline_streams or any(stream >= len(vb.streams) for stream, *_ in vb.attributes()):
        raise ValueError('Cloth topology requires a rebuilt simulation mapping; ordinary mesh append is unsafe')
    first = vb.count
    attributes = vb.attributes()
    max_weights = 8 if any(a[3] == 8 for a in attributes) else 4
    for vertex in vertices:
        template = vertex.get('template_index', sub.first_vertex)
        records = [bytearray(stream[template * stride:(template + 1) * stride])
                   for stride, stream in zip(vb.strides, vb.streams)]
        influences = sorted(zip(vertex.get('bones', ()), vertex.get('weights', ())),
                            key=lambda item: item[1], reverse=True)
        influences = [(b,w) for b,w in influences if w > 0][:max_weights]
        if not influences: influences = [(1, 1.0)]
        total = sum(w for _,w in influences)
        bones = [(b - 1) * 3.0 for b,w in influences] + [0.0] * 8
        weights = [w / total for b,w in influences] + [0.0] * 8
        uv = list(vertex.get('uvs', ())) or [(0, 0)]
        uv += [uv[0]] * 4
        attrs = {0: vertex['position'], 1: vertex['normal'],
                 2: vertex.get('color', (1, 1, 1, 1)),
                 4: (*uv[0], *uv[1]), 5: (*uv[2], *uv[3]),
                 6: weights[:4], 8: weights[4:8], 7: bones[:4], 9: bones[4:8],
                 10: vertex.get('tangent', (1, 0, 0, 1))[:3]}
        for stream, offset, fmt, semantic in attributes:
            if semantic not in attrs:
                # Preserve engine-specific self-occlusion/auxiliary channels
                # from the nearest source vertex instead of inventing values.
                continue
            encoded = pack_attribute(fmt, attrs[semantic])
            if semantic == 10 and fmt == 13 and vertex.get('tangent', (0,0,0,1))[3] < 0:
                encoded = struct.pack('<I', u32(encoded, 0) | 0x80000000)
            if offset + len(encoded) > len(records[stream]):
                raise ValueError('Vertex declaration exceeds stream stride')
            records[stream][offset:offset + len(encoded)] = encoded
        for target, record in zip(vb.streams, records): target.extend(record)
    first_index = len(md.indices[sub.ib]) // 2
    for face in faces:
        if len(face) != 3 or any(i < 0 or i >= len(vertices) for i in face):
            raise ValueError('Triangle references a missing vertex')
        md.indices[sub.ib].extend(struct.pack('<3H', *reversed(face)))
    vb.count += len(vertices)
    result = sub.replaced(first, len(vertices), first_index, len(faces) * 3)
    radius = max((math.sqrt(sum(x*x for x in v['position'])) for v in vertices), default=0) + 1.0
    result.suffix = result.suffix[:-16] + struct.pack('<4f', 0, 0, 0, radius)
    return result


def split_mesh(vertices, faces, limit=65535):
    """Preserve corner attributes and partition a mesh into legal draw calls."""
    mapping, batch_vertices, batch_faces = {}, [], []
    for face in faces:
        if len(mapping) + sum(index not in mapping for index in set(face)) > limit:
            yield batch_vertices, batch_faces
            mapping, batch_vertices, batch_faces = {}, [], []
        output = []
        for index in face:
            if index not in mapping:
                mapping[index] = len(batch_vertices)
                batch_vertices.append(vertices[index])
            output.append(mapping[index])
        batch_faces.append(tuple(output))
    if batch_faces or not faces: yield batch_vertices, batch_faces


class Package:
    def __init__(self, data):
        from .texture_bundle import split_bundle
        self.source_data = data
        data, self.texture_attachments = split_bundle(data)
        if data[:8] == b'DC_INFO ':
            # Decompressed metadata followed by its original external SEGS
            # streams. External offsets stay relative to the metadata end;
            # replacing only the outer metadata framing preserves that base.
            container = Container(data, allow_trailing=True)
            if len(data) == container.end and any(r.external for r in container.records):
                raise ValueError('DATA_CONTAINER is metadata only; external mesh/shader streams are missing')
            data = encode_member(data[:container.end]) + data[container.end:]
        self.data = data
        self.members = scan_members(data)
        self.container = Container(self.members[0].unpacked)
        self.record_members = {}
        base = self.members[0].end
        by_offset = {m.start - base: m for m in self.members[1:]}
        for rec in self.container.records:
            if rec.external:
                if len(rec.prefix) != 52 or u32(rec.prefix, 32) != 1:
                    raise ValueError('Unsupported external resource directory')
                member = by_offset.get(u32(rec.prefix, 44))
                if member is None:
                    raise ValueError(f'Record {rec.index} has an unresolved external resource')
                if u32(rec.prefix,40) != len(member.unpacked):
                    raise ValueError(f'Record {rec.index} external decoded size mismatch')
                self.record_members[rec.index] = member.index

    def mesh(self, record_index):
        rec = self.container.records[record_index]
        md = MeshData(rec.payload)
        md.load_geometry(self.members[self.record_members[record_index]].unpacked)
        return md

    def rebuild(self, meshes, force_layout=False, resource_data=None):
        from .resource_order import serialization_order, validate_material_order
        resource_data=dict(resource_data or {})
        from .cloth_integrity import repair_authored_cloth
        from .cloth_skin_weights import repair_authored_skin_weights
        from .cloth_constraints import repair_authored_constraints
        for rec in self.container.records:
            if rec.kind != 2150: continue
            raw = resource_data.get(rec.index)
            if raw is None:
                raw = self.members[self.record_members[rec.index]].unpacked if rec.external else rec.payload
            repaired, _ = repair_authored_cloth(raw, self, rec)
            repaired, _ = repair_authored_skin_weights(repaired)
            repaired, _ = repair_authored_constraints(repaired)
            if repaired != raw:
                if rec.external: resource_data[rec.index] = repaired
                else:
                    rec.payload = repaired
                    force_layout = True
        from .shader_profiles import normalize_stream, validate_native_stream
        from .motion_control import upgrade_legacy
        from .pbr_shader import upgrade_velocity_fringe
        for rec in self.container.records:
            if rec.kind!=2133 or not rec.external:continue
            raw=resource_data.get(rec.index)
            if raw is None:raw=self.members[self.record_members[rec.index]].unpacked
            repaired=normalize_stream(rec,upgrade_velocity_fringe(rec,upgrade_legacy(rec,raw)))
            validate_native_stream(rec,repaired)
            if repaired!=raw:resource_data[rec.index]=repaired
        order = serialization_order(self.container.records)
        self.serialized_record_indices = {rec.index:i for i,rec in enumerate(order)}
        reordered = any(old != new for old,new in self.serialized_record_indices.items())
        if not meshes and not force_layout and not resource_data and not reordered:
            try:
                validate_streaming(self.members)
                from .texture_bundle import pack_bundle
                return pack_bundle(self.data, self.texture_attachments)
            except ValueError:
                pass  # Repair an older export even when geometry is unchanged.
        # Only zero-filled reservation/alignment gaps are currently understood.
        # Refuse to silently discard data in another package revision.
        cursor = 0
        for member in self.members:
            if any(self.data[cursor:member.start]):
                raise ValueError('Unknown nonzero data between SEGS members; cannot safely rebuild')
            cursor = member.end
        if any(self.data[cursor:]):
            raise ValueError('Unknown nonzero data after SEGS members; cannot safely rebuild')
        replacements = {}
        for ri,raw in resource_data.items():
            if ri in self.record_members:replacements[self.record_members[ri]]=raw
        for ri, md in meshes.items():
            replacements[self.record_members[ri]] = md.pack_geometry()
            self.container.records[ri].payload = md.pack(clear_deformations=not getattr(md,'preserve_deformations',False))
        # External directory offsets are relative to the END of member zero.
        # Build that body first, update all 91 references, then compress metadata.
        body = bytearray()
        locations = {}
        members_for_record=dict(self.record_members)
        for member in self.members[1:]:
            raw = replacements.get(member.index)
            source_member = read_member(encode_member(raw, member.attributes), 0, member.index) if raw is not None else member
            offset, packed = relocate_member(source_member, len(body))
            body += b'\0' * (offset - len(body))
            locations[member.index] = (offset, len(packed), len(raw) if raw is not None else member.unpacked_size)
            body += packed
        for ri,raw in resource_data.items():
            if ri in self.record_members:continue
            rec=self.container.records[ri]
            if rec.index!=ri or not rec.external:raise ValueError('New resource must have an external prefix')
            mi=len(self.members)+sum(1 for n in members_for_record.values() if n>=len(self.members))
            member=read_member(encode_member(raw),0,mi)
            offset,packed=relocate_member(member,len(body))
            body+=b'\0'*(offset-len(body))+packed
            members_for_record[ri]=mi
            locations[mi]=(offset,len(packed),len(raw))
        for rec in self.container.records:
            if rec.index in members_for_record:
                offset, packed_size, raw_size = locations[members_for_record[rec.index]]
                prefix = bytearray(rec.prefix)
                for at, value in ((18,raw_size),(36,packed_size),(40,raw_size),(44,offset)):
                    struct.pack_into('<I', prefix, at, value)
                rec.prefix = bytes(prefix)
        # Mesh payloads above may introduce new material references. Recompute
        # after packing them, without renumbering live/source record objects.
        order = serialization_order(self.container.records)
        validate_material_order(order)
        self.serialized_record_indices = {rec.index:i for i,rec in enumerate(order)}
        metadata = bytearray(self.container.pack(order))
        # DC_DATA span includes decoded metadata PLUS packed external region.
        directory_end=24+8*len(order)
        struct.pack_into('<I',metadata,directory_end+12,len(metadata)-directory_end-16+len(body))
        metadata=bytes(metadata)
        output = encode_member(metadata, self.members[0].attributes) + body
        validate_streaming(scan_members(output))
        from .texture_bundle import pack_bundle
        return pack_bundle(output, self.texture_attachments)
