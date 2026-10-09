"""Lossless patch support for Detroit PC MESHDATA v41 topology fields."""

from __future__ import annotations

import struct
from dataclasses import dataclass


def _u32(data: bytes, offset: int) -> int:
    if offset + 4 > len(data):
        raise ValueError(f"MESHDATA read past EOF at 0x{offset:X}")
    return struct.unpack_from("<I", data, offset)[0]


class Cursor:
    def __init__(self, data: bytes, offset: int):
        self.data = data
        self.offset = offset

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise ValueError(f"MESHDATA read past EOF at 0x{self.offset:X}")
        value = self.data[self.offset:end]
        self.offset = end
        return value

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]


@dataclass(frozen=True)
class VertexBuffer:
    vertex_count: int
    vertex_count_offset: int
    stream_sizes: tuple[int, int, int, int]


@dataclass(frozen=True)
class Submesh:
    vertex_buffer: int
    first_vertex: int
    vertex_count: int
    index_buffer: int
    first_index: int
    index_count: int
    vertex_count_offset: int
    first_index_offset: int
    index_count_offset: int


@dataclass(frozen=True)
class MeshData:
    vertex_buffers: tuple[VertexBuffer, ...]
    index_counts: tuple[int, ...]
    submeshes: tuple[Submesh, ...]


def find_meshdata_tag(data: bytes, record_index: int) -> int:
    """Locate MESHDATA inside one indexed DATA_CONTAINER record."""
    if data[:8] != b"DC_INFO ":
        raise ValueError("SEGS member 0 is not a DATA_CONTAINER")
    count = _u32(data, 20)
    if not 0 <= record_index < count:
        raise ValueError(f"DATA_CONTAINER has no record {record_index}")
    cursor = 24 + count * 8
    if data[cursor:cursor + 8] != b"DC_DATA ":
        raise ValueError("DATA_CONTAINER has no DC_DATA section")
    cursor += 16
    for index in range(record_index + 1):
        record_start = cursor
        payload_size = _u32(data, cursor)
        cursor += 4
        flag = data[cursor]
        cursor += 1
        if flag == 0:
            cursor += 8
        elif flag == 1:
            variable_count = _u32(data, cursor)
            value1 = _u32(data, cursor + 4)
            cursor += 13
            if value1:
                cursor += 4 + variable_count * 12
            else:
                cursor += 4 + 1 + 4 + 1 + 4 + 20
        else:
            raise ValueError(f"record {index} at 0x{record_start:X} has prefix flag {flag}")
        payload_end = cursor + payload_size
        if payload_end > len(data):
            raise ValueError(f"record {index} extends past DATA_CONTAINER EOF")
        if index == record_index:
            tag = data.find(b"MESHDATA", cursor, payload_end)
            if tag < 0:
                raise ValueError(f"record {record_index} has no MESHDATA chunk")
            return tag
        cursor = payload_end
    raise AssertionError("unreachable")


def read_v41(data: bytes, tag_offset: int) -> MeshData:
    if data[tag_offset:tag_offset + 8] != b"MESHDATA":
        raise ValueError(f"MESHDATA tag not found at 0x{tag_offset:X}")
    version = _u32(data, tag_offset + 8)
    if version != 41:
        raise ValueError(f"topology export requires MESHDATA v41, found v{version}")
    source = Cursor(data, tag_offset + 12)

    headers: list[tuple[int, int, int, int]] = []
    for _ in range(source.u32()):
        layout_count = source.u32()
        vertex_count_offset = source.offset
        vertex_count = source.u32()
        external_stream = source.u8()
        headers.append((layout_count, vertex_count, external_stream, vertex_count_offset))

    index_counts = tuple(source.u32() for _ in range(source.u32()))
    submeshes: list[Submesh] = []
    for _ in range(source.u32()):
        group_count = source.u32()
        source.take(8)
        for _ in range(group_count):
            descriptor_version = source.u32()
            descriptor_flag = source.u8()
            if descriptor_version != 3 or descriptor_flag != 0:
                raise ValueError("unsupported MESHDATA submesh descriptor")
            source.u8()  # flags
            vertex_buffer = source.u32()
            first_vertex = source.u32()
            vertex_count_offset = source.offset
            vertex_count = source.u32()
            index_buffer = source.u32()
            first_index_offset = source.offset
            first_index = source.u32()
            index_count_offset = source.offset
            index_count = source.u32()
            source.u32()  # layout slot
            remap_count = source.u32()
            source.take(remap_count * 4)
            source.take(8)  # material reference
            morph_count = source.u32()
            source.take(morph_count * 48)
            source.take(4 + 76)  # resource index, affine transform, bounds
            submeshes.append(Submesh(
                vertex_buffer, first_vertex, vertex_count,
                index_buffer, first_index, index_count,
                vertex_count_offset, first_index_offset, index_count_offset,
            ))
        source.take(2 + 76)  # two group flags, affine transform, bounds

    packed_refs = source.u32()
    source.take((packed_refs & 0xFF) * 8)
    if packed_refs & 0x100:
        source.take(4)
    source.take(4)

    vertex_buffers: list[VertexBuffer] = []
    for layout_count, vertex_count, external_stream, vertex_count_offset in headers:
        source.take(layout_count * 4)
        stream_sizes = []
        for stream_index in range(4):
            stride = source.u32()
            stream_sizes.append(stride)
            # The package's composite stream routes raw payload reads to a
            # sibling SEGS member; member 0 itself contains only these sizes.
        vertex_buffers.append(VertexBuffer(
            vertex_count, vertex_count_offset, tuple(stream_sizes)
        ))
    return MeshData(tuple(vertex_buffers), index_counts, tuple(submeshes))


def patch_submesh_topology(
    metadata: bytearray,
    meshdata: MeshData,
    vertex_counts: list[int],
    first_indices: list[int],
    index_counts: list[int],
) -> None:
    if not (len(vertex_counts) == len(first_indices) == len(index_counts) == len(meshdata.submeshes)):
        raise ValueError("MESHDATA patch list length mismatch")
    for submesh, vertex_count, first_index, index_count in zip(
        meshdata.submeshes, vertex_counts, first_indices, index_counts
    ):
        struct.pack_into("<I", metadata, submesh.vertex_count_offset, vertex_count)
        struct.pack_into("<I", metadata, submesh.first_index_offset, first_index)
        struct.pack_into("<I", metadata, submesh.index_count_offset, index_count)
