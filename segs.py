"""Detroit PC SEGS reader and fixed-slot writer used by the add-on."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass


HEADER = struct.Struct("<4sHHII")
ENTRY = struct.Struct("<HHI")
STREAM_WINDOW = 1 << 20


def align(value: int, alignment: int = 16) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


@dataclass(frozen=True)
class Block:
    packed_size: int
    unpacked_size: int
    source_offset: int
    output_offset: int
    compressed: bool = True


@dataclass(frozen=True)
class Member:
    index: int
    start: int
    end: int
    attributes: int
    declared_packed_size: int
    unpacked_size: int
    data_base: int
    blocks: tuple[Block, ...]
    unpacked: bytes
    raw: bytes


def _inflate(value: bytes, expected: int, compressed: bool = True) -> bytes:
    if not compressed:
        if len(value) != expected:
            raise ValueError('Stored SEGS block size mismatch')
        return value
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
        try:
            result = zlib.decompress(value, wbits)
        except zlib.error:
            continue
        if len(result) == expected:
            return result
    raise ValueError("invalid SEGS compressed block")


def read_member(package: bytes, start: int, index: int) -> Member:
    if start + HEADER.size > len(package):
        raise ValueError("truncated SEGS header")
    magic, attributes, count, output_size, packed_span = HEADER.unpack_from(package, start)
    if magic != b"segs" or not count:
        raise ValueError("not a PC SEGS member")
    data_base = start + align(HEADER.size + count * ENTRY.size)
    output = bytearray()
    blocks: list[Block] = []
    end = data_base
    for block_index in range(count):
        packed16, output16, one_based = ENTRY.unpack_from(
            package, start + HEADER.size + block_index * ENTRY.size
        )
        packed_size = packed16 or 0x10000
        unpacked_size = output16 or 0x10000
        # Low bit is compression, not a one-based address. Stored block zero
        # at the data base legitimately has an offset/flags value of zero.
        compressed = bool(one_based & 1)
        source = data_base + (one_based & ~1)
        source_end = source + packed_size
        if source_end > len(package):
            raise ValueError("SEGS block extends beyond file")
        output_offset = len(output)
        output.extend(_inflate(package[source:source_end], unpacked_size, compressed))
        blocks.append(Block(packed_size, unpacked_size, source, output_offset, compressed))
        end = max(end, source_end)
    if len(output) != output_size:
        raise ValueError("SEGS decoded size does not match header")
    return Member(
        index, start, end, attributes, packed_span, output_size, data_base,
        tuple(blocks), bytes(output), package[start:end]
    )


def scan_members(package: bytes) -> tuple[Member, ...]:
    result: list[Member] = []
    cursor = 0
    while True:
        start = package.find(b"segs", cursor)
        if start < 0:
            break
        try:
            member = read_member(package, start, len(result))
        except ValueError:
            cursor = start + 4
            continue
        result.append(member)
        cursor = member.end
    if not result:
        raise ValueError("no valid SEGS members found")
    return tuple(result)


def build_member(member: Member, output: bytes, level: int = 9) -> bytes:
    if len(output) != member.unpacked_size:
        raise ValueError(f"member {member.index} decoded size changed")
    data_base_relative = member.data_base - member.start
    capacity = member.end - member.start
    encoded: list[tuple[bytes, bytes]] = []
    for block in member.blocks:
        raw = output[block.output_offset:block.output_offset + block.unpacked_size]
        packed = zlib.compress(raw, level)
        if len(packed) >= len(raw):
            packed = raw
        encoded.append((raw, packed))

    fixed_layout_fits = True
    for block_index, (block, (_raw, packed)) in enumerate(zip(member.blocks, encoded)):
        destination = block.source_offset - member.start
        slot_end = (
            member.blocks[block_index + 1].source_offset - member.start
            if block_index + 1 < len(member.blocks) else capacity
        )
        if destination + len(packed) > slot_end:
            fixed_layout_fits = False
            break

    result = bytearray(member.raw)
    table: list[tuple[int, int, int]] = []
    if fixed_layout_fits:
        destinations = [block.source_offset - member.start for block in member.blocks]
    else:
        destinations = []
        cursor = data_base_relative
        for _raw, packed in encoded:
            cursor = align(cursor)
            destinations.append(cursor)
            cursor += len(packed)
        if cursor > capacity:
            raise ValueError(
                f"member {member.index} no longer fits its fixed member capacity"
            )
        result[data_base_relative:capacity] = b"\x00" * (capacity - data_base_relative)

    for (raw, packed), destination in zip(encoded, destinations):
        result[destination:destination + len(packed)] = packed
        table.append((
            0 if len(packed) == 0x10000 else len(packed),
            0 if len(raw) == 0x10000 else len(raw),
            (destination - data_base_relative) | int(packed != raw),
        ))
    HEADER.pack_into(
        result, 0, b"segs", member.attributes, len(member.blocks),
        member.unpacked_size, member.declared_packed_size
    )
    for index, values in enumerate(table):
        ENTRY.pack_into(result, HEADER.size + index * ENTRY.size, *values)
    return bytes(result)


def replace_fixed(package: bytes, members: tuple[Member, ...], replacements: dict[int, bytes]) -> bytes:
    result = bytearray(package)
    for index, output in replacements.items():
        member = members[index]
        rebuilt = build_member(member, output)
        result[member.start:member.end] = rebuilt
    validate_streaming(scan_members(result))
    return bytes(result)


def _place_blocks(blocks, attributes, region_offset=0):
    """Return (start, bytes), framed for the game's 1-MiB input reads.

    First frame blocks against the MEMBER's own origin. Then place the
    complete member in the external region: large members start on a 1-MiB
    boundary; small members fit entirely inside one window. This satisfies
    both independent-resource reads and combined external-region reads.
    """
    base = align(HEADER.size + len(blocks) * ENTRY.size)
    if base + len(blocks[0][0]) > STREAM_WINDOW:
        raise ValueError('SEGS header and first block exceed a streaming window')
    result = bytearray(base)
    for index, (packed, decoded_size, compressed) in enumerate(blocks):
        position = align(len(result))
        if position % STREAM_WINDOW + len(packed) > STREAM_WINDOW:
            position = align(position, STREAM_WINDOW)
        result += b'\0' * (position - len(result))
        ENTRY.pack_into(result, HEADER.size + index * ENTRY.size,
                        len(packed) & 65535, decoded_size & 65535,
                        (len(result) - base) | int(compressed))
        result += packed
    HEADER.pack_into(result, 0, b'segs', attributes, len(blocks),
                     sum(b[1] for b in blocks), len(result) - base)
    start = align(region_offset)
    if len(result) > STREAM_WINDOW or start % STREAM_WINDOW + len(result) > STREAM_WINDOW:
        start = align(start, STREAM_WINDOW)
    return start, bytes(result)


def relocate_member(member: Member, region_offset: int):
    """Preserve every compressed stream; change only framing and padding."""
    blocks = [(member.raw[b.source_offset-member.start:
                          b.source_offset-member.start+b.packed_size],
               b.unpacked_size, b.compressed) for b in member.blocks]
    return _place_blocks(blocks, member.attributes, region_offset)


def compress_chunks(chunks, workers=4):
    """Ordered, bounded zlib work on plain bytes; identical level-9 streams."""
    from itertools import islice
    if workers<=1:
        for raw in chunks:yield zlib.compress(raw,9)
        return
    from concurrent.futures import ThreadPoolExecutor
    iterator=iter(chunks)
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='DBH-SEGS') as pool:
        while True:
            batch=list(islice(iterator,32))
            if not batch:break
            yield from pool.map(lambda raw:zlib.compress(raw,9),batch)


def encode_member(output: bytes, attributes: int = 1, *, workers: int = 4) -> bytes:
    """Create a member with as many 64-KiB blocks as the new payload needs."""
    if not output:
        raise ValueError('Empty SEGS payload')
    count = (len(output) + 65535) // 65536
    if count > 65535 or len(output) > 0xFFFFFFFF:
        raise ValueError('Payload exceeds SEGS format limits')
    blocks = []
    chunks=(output[index*65536:(index+1)*65536] for index in range(count))
    for index,packed in enumerate(compress_chunks(chunks,workers if count>=16 else 1)):
        raw = output[index * 65536:(index + 1) * 65536]
        if len(packed) >= len(raw): packed = raw
        blocks.append((packed, len(raw), packed != raw))
    return _place_blocks(blocks, attributes)[1]


def validate_streaming(members):
    """Check combined-region AND individual-resource streaming boundaries."""
    for member in members:
        origin = 0 if member.index == 0 else members[0].end
        start, base = member.start - origin, member.data_base - origin
        if start % 16 or member.blocks[0].source_offset != member.data_base:
            raise ValueError(f'Member {member.index}: invalid header/first-block alignment')
        if start // STREAM_WINDOW != (base - 1) // STREAM_WINDOW:
            raise ValueError(f'Member {member.index}: header crosses streaming boundary')
        for i, block in enumerate(member.blocks):
            for label, read_origin in (('region', origin), ('resource', member.start)):
                pos = block.source_offset - read_origin
                if pos % 16 or pos // STREAM_WINDOW != (pos + block.packed_size - 1) // STREAM_WINDOW:
                    raise ValueError(f'Member {member.index} block {i}: crosses {label} streaming boundary or is misaligned')
        if member.declared_packed_size != member.end - member.data_base:
            raise ValueError(f'Member {member.index}: packed span mismatch')
