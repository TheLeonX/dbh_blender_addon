"""Infer position stream locations by correlating verified XPS output."""

from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .segs import Member
from .xps import Mesh


@dataclass(frozen=True)
class Stream:
    member: int
    base: int
    stride: int
    matches: int
    vertices: int


def _geometry_candidates(members: tuple[Member, ...]) -> list[Member]:
    result: list[Member] = []
    for member in members[1:]:
        sample = member.unpacked[:32]
        if sample.startswith(b"#version"):
            continue
        # Geometry streams in the tested PC packages precede shader-source
        # members.  Keep binary data and ignore obvious serialized containers.
        if sample.startswith((b"DC_INFO ", b"COM_CONT")):
            continue
        result.append(member)
    return result


def _scan_hits(
    members: list[Member], wanted: set[tuple[float, float]]
) -> dict[int, dict[tuple[float, float], list[tuple[int, tuple[float, float, float]]]]]:
    all_hits: dict[int, dict[tuple[float, float], list[tuple[int, tuple[float, float, float]]]]] = {}
    for member in members:
        buckets: dict[tuple[float, float], list[tuple[int, tuple[float, float, float]]]] = defaultdict(list)
        data = member.unpacked
        for offset in range(0, len(data) - 11, 4):
            value = struct.unpack_from("<fff", data, offset)
            if not all(math.isfinite(component) and abs(component) < 10000.0 for component in value):
                continue
            key = (round(value[0], 4), round(value[2], 4))
            if key in wanted:
                buckets[key].append((offset, value))
        if buckets:
            all_hits[member.index] = buckets
    return all_hits


def _infer_y_offset(meshes: tuple[Mesh, ...], hits: dict[int, dict]) -> float:
    positions = [vertex.position for mesh in meshes for vertex in mesh.vertices]
    step = max(1, len(positions) // 3000)
    votes: Counter[float] = Counter()
    for x, y, z in positions[::step]:
        key = (round(x, 4), round(z, 4))
        for buckets in hits.values():
            for _, raw in buckets.get(key, ()):
                if abs(raw[0] - x) > 0.0000011 or abs(raw[2] - z) > 0.0000011:
                    continue
                difference = y - raw[1]
                if abs(difference) < 10.0:
                    votes[round(difference, 6)] += 1
    if not votes:
        return 0.0
    return votes.most_common(1)[0][0]


def infer_streams(
    meshes: tuple[Mesh, ...], members: tuple[Member, ...], y_offset: float | None = None
) -> tuple[float, list[list[Stream]]]:
    wanted = {
        (round(vertex.position[0], 4), round(vertex.position[2], 4))
        for mesh in meshes for vertex in mesh.vertices
    }
    hits = _scan_hits(_geometry_candidates(members), wanted)
    if y_offset is None:
        y_offset = _infer_y_offset(meshes, hits)
    result: list[list[Stream]] = []
    tolerance = 0.0000012

    for mesh in meshes:
        streams: list[Stream] = []
        minimum = max(3, int(len(mesh.vertices) * 0.80))
        for member_index, buckets in hits.items():
            votes: Counter[tuple[int, int]] = Counter()
            for vertex_index, vertex in enumerate(mesh.vertices):
                x, y, z = vertex.position
                raw_position = (x, y - y_offset, z)
                key = (round(x, 4), round(z, 4))
                for offset, value in buckets.get(key, ()):
                    if not all(abs(a - b) <= tolerance for a, b in zip(value, raw_position)):
                        continue
                    for stride in range(12, 65, 4):
                        base = offset - vertex_index * stride
                        if base >= 0:
                            votes[(stride, base)] += 1
            if not votes:
                continue
            (stride, base), matches = votes.most_common(1)[0]
            if matches >= minimum:
                streams.append(Stream(member_index, base, stride, matches, len(mesh.vertices)))
        result.append(streams)
    return y_offset, result


def streams_to_jsonable(streams: list[Stream]) -> list[dict[str, int]]:
    return [asdict(stream) for stream in streams]
