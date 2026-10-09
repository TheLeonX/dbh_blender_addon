"""Extract Detroit ANIM_DATA by exact archive ID and decode with daemon1's tool.

No game archive is modified. The proprietary keyframe codec remains in the
user-provided Detroit_anim.exe; this module validates its SMD output.
"""
from dataclasses import dataclass
from pathlib import Path
import math
import struct
import subprocess
import tempfile


ANIM_DATA_KIND = 0x7F1


def _animation_rows(index_path: Path):
    raw = Path(index_path).read_bytes()
    if raw[:20] != b'QUANTICDREAMTABINDEX' or struct.unpack_from('>I', raw, 20)[0] != 18:
        raise ValueError('Unsupported BigFile_PC.idx')
    pos = 105
    for _ in range(struct.unpack_from('>I', raw, 101)[0]):
        kind, count = struct.unpack_from('>II', raw, pos)
        pos += 8
        end = pos + count * 20
        if end > len(raw):
            raise ValueError('Truncated BigFile index')
        if kind == ANIM_DATA_KIND:
            for i in range(pos, end, 20):
                yield struct.unpack_from('>IIIII', raw, i)
        pos = end
    if pos != len(raw):
        raise ValueError('BigFile index length mismatch')


def available_animation_ids(index_path: Path):
    return {row[0] for row in _animation_rows(index_path)}


def archive_row(index_path: Path, asset_id: int):
    found = [row for row in _animation_rows(index_path) if row[0] == asset_id]
    if len(found) != 1:
        raise ValueError(f'Expected one ANIM_DATA 0x{asset_id:X}, found {len(found)}')
    return found[0]


def read_animation(index_path: Path, asset_id: int) -> bytes:
    index_path = Path(index_path)
    _, offset, size, _, archive_number = archive_row(index_path, asset_id)
    if not 64 <= size <= 256 * 1024 * 1024:
        raise ValueError('Unreasonable ANIM_DATA size')
    name = 'BigFile_PC.dat' if archive_number == 0 else f'BigFile_PC.d{archive_number:02d}'
    with (index_path.parent / name).open('rb') as handle:
        handle.seek(offset)
        raw = handle.read(size)
    if len(raw) != size or raw[:8] != b'COM_CONT' or raw[45:53] != b'ANIMDATA':
        raise ValueError(f'ANIM_DATA 0x{asset_id:X} has an unsupported archive payload')
    return raw


def convert_to_smd(animation: bytes, nodes: bytes, converter: Path, stem='dbh_clip') -> tuple[str, str]:
    if animation[:8] != b'COM_CONT' or animation[45:53] != b'ANIMDATA':
        raise ValueError('Expected a decompressed COM_CONT / ANIMDATA file')
    if nodes[:4] != b'\x02\0\0\0' or nodes[4:12] != b'NODE    ':
        raise ValueError('Expected native NODE skeleton bytes')
    converter = Path(converter)
    if not converter.is_file():
        raise ValueError(f'Detroit_anim.exe not found: {converter}')
    with tempfile.TemporaryDirectory(prefix='dbh_animation_') as temporary:
        folder = Path(temporary)
        input_path = folder / (stem + '.anim')
        nodes_path = folder / 'model.nodes'
        input_path.write_bytes(animation)
        nodes_path.write_bytes(nodes)
        result = subprocess.run(
            [str(converter), str(input_path), str(nodes_path)], cwd=folder,
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=180, check=False, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
        output_path = folder / (stem + '.smd')
        log = (result.stdout + '\n' + result.stderr).strip()
        if result.returncode != 0 or not output_path.is_file():
            raise ValueError(f'Detroit_anim.exe did not produce an SMD: {log[-700:]}')
        return output_path.read_text(encoding='utf-8', errors='replace'), log


@dataclass(frozen=True)
class SMDBone:
    index: int
    name: str
    parent: int


@dataclass(frozen=True)
class SMDClip:
    bones: tuple[SMDBone, ...]
    frames: tuple[tuple[int, dict[int, tuple[float, ...]]], ...]


def parse_smd(text: str) -> SMDClip:
    lines = iter(text.splitlines())
    if next(lines, '').strip() != 'version 1' or next(lines, '').strip() != 'nodes':
        raise ValueError('Converter output is not SMD version 1')
    bones = []
    for line in lines:
        if line == 'end':
            break
        parts = line.split('"')
        if len(parts) != 3:
            raise ValueError('Invalid SMD node')
        index = int(parts[0])
        parent = int(parts[2])
        if index != len(bones) or parent >= index or parent < -1:
            raise ValueError('SMD skeleton is not parent-first')
        bones.append(SMDBone(index, parts[1], parent))
    if not bones or next(lines, '').strip() != 'skeleton':
        raise ValueError('SMD skeleton section missing')
    frames = []
    current = None
    for line in lines:
        if line == 'end':
            break
        if line.startswith('time '):
            frame = int(line.split()[1])
            if frames and frame <= frames[-1][0]:
                raise ValueError('SMD frame order is invalid')
            current = {}
            frames.append((frame, current))
            if len(frames) > 10000:
                raise ValueError('SMD animation is too long')
            continue
        if current is None:
            raise ValueError('SMD pose before frame')
        fields = line.split()
        if len(fields) != 7:
            raise ValueError('Invalid SMD pose line')
        index = int(fields[0])
        transform = tuple(float(v) for v in fields[1:])
        if index < 0 or index >= len(bones) or index in current or not all(math.isfinite(v) for v in transform):
            raise ValueError('Invalid SMD bone pose')
        current[index] = transform
    if not frames or not frames[0][1]:
        raise ValueError('SMD has no animated poses')
    return SMDClip(tuple(bones), tuple(frames))
