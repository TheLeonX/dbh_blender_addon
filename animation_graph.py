"""Typed graph nodes mapped to their ANIM_DATA array indices, not clip fields."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import struct
import html


ANIM_DATA_KIND = 0x7F1
REFERENCE_STRIDE = 150


@dataclass(frozen=True)
class GraphLabel:
    name: str
    offset: int
    node_id: int
    node_type: int
    type_index: int


@dataclass(frozen=True)
class ClipReference:
    asset_id: int
    node_id: int
    offset: int
    label: str | None


@dataclass(frozen=True)
class MotionGraph:
    labels: tuple[GraphLabel, ...]
    references: tuple[ClipReference, ...]

    @property
    def asset_references(self):
        return tuple(reference.asset_id for reference in self.references)

    @property
    def clip_counts(self):
        return Counter(self.asset_references)

    @property
    def clip_labels(self):
        result = defaultdict(set)
        for reference in self.references:
            if reference.label:
                result[reference.asset_id].add(reference.label)
        return {asset_id: tuple(sorted(names, key=str.casefold))
                for asset_id, names in result.items()}

    @property
    def label_clips(self):
        result = defaultdict(set)
        for reference in self.references:
            if reference.label:
                result[(reference.node_id, reference.label)].add(reference.asset_id)
        return {key: tuple(sorted(ids)) for key, ids in result.items()}


def _read_name_rows(data: bytes, start: int, limit: int):
    rows = []
    cursor = start
    while cursor + 22 <= limit and len(rows) < 65536:
        node_type, node_id, _, _, size = struct.unpack_from('<IHIII', data, cursor)
        end = cursor + 18 + size
        if node_type > 255 or node_id != len(rows) or size > 512 or end + 4 > limit:
            break
        raw = data[cursor + 18:end]
        if any(byte < 32 or byte == 127 for byte in raw):
            break
        try:
            name = html.unescape(raw.decode('utf-8'))
        except UnicodeDecodeError:
            break
        type_index = struct.unpack_from('<I', data, end)[0]
        rows.append(GraphLabel(name, cursor + 18, node_id, node_type, type_index))
        cursor = end + 4
    return tuple(rows)


def _name_table(data: bytes, first_reference: int):
    # Rows begin with type/id metadata, then a length-prefixed name and the
    # index into that type's serialized array. IDs are zero-based.
    best = ()
    for start in range(61, first_reference - 30):
        kind, node_id = struct.unpack_from('<IH', data, start)
        if kind > 255 or node_id != 0:
            continue
        size = struct.unpack_from('<I', data, start + 14)[0]
        if size > 512 or start + size + 22 > first_reference:
            continue
        rows = _read_name_rows(data, start, first_reference)
        if len(rows) > len(best):
            best = rows
    if len(best) < 10:
        raise ValueError('Motion graph numbered node-name table not found')
    return best


def _reference_table(data: bytes):
    marker = struct.pack('<I', ANIM_DATA_KIND)
    starts = []
    pos = data.find(marker, 0x500)
    while pos >= 0 and pos + 8 <= len(data):
        starts.append(pos)
        pos = data.find(marker, pos + 4)
    if not starts:
        raise ValueError('Motion graph has no ANIM_DATA references')
    known = set(starts)
    best = ()
    for start in starts:
        if start - REFERENCE_STRIDE in known or start < 32:
            continue
        run = []
        cursor = start
        while cursor in known and cursor + 8 <= len(data):
            asset_id = struct.unpack_from('<I', data, cursor + 4)[0]
            if not asset_id:
                break
            run.append((cursor, asset_id))
            cursor += REFERENCE_STRIDE
        if len(run) > len(best):
            best = tuple(run)
    if not best:
        raise ValueError('Motion graph ANIM_DATA reference array not found')
    return best


def parse_motion_graph(data: bytes) -> MotionGraph:
    if len(data) < 0x500 or data[:8] != b'COM_CONT' or data[45:54] != b'MG_DATA_%':
        raise ValueError('Expected a decompressed COM_CONT / MG_DATA_% motion graph')
    declared = struct.unpack_from('<I', data, 57)[0]
    if declared != len(data) - 61:
        raise ValueError('Motion graph payload size mismatch')
    table = _reference_table(data)
    labels = _name_table(data, table[0][0])
    animation_nodes = [label for label in labels if label.node_type == 2]
    by_index = {label.type_index: label for label in animation_nodes}
    if len(by_index) != len(animation_nodes) or set(by_index) != set(range(len(table))):
        raise ValueError('Motion graph animation-node indices do not match the reference array; names were not guessed')
    references = tuple(ClipReference(asset_id, by_index[i].node_id, offset, by_index[i].name or None)
                       for i, (offset, asset_id) in enumerate(table))
    return MotionGraph(labels, references)


def read_motion_graph(path: Path) -> MotionGraph:
    return parse_motion_graph(Path(path).read_bytes())
