"""Native import discovery; no subprocesses, XPS files or converter required."""
import hashlib
import math
import statistics
import struct
from functools import lru_cache
from pathlib import Path

from .runtime_mesh import catalogs
from .skeleton import Rig, decode_inverse, inverse, mv, mm, sub, norm, IDENTITY
from .animation_codec import quat_mul, quat_normalize


@lru_cache(maxsize=1)
def bone_dictionary():
    # Established hash labels bundled as data, not an external tool dependency.
    result = {}
    for line in Path(__file__).with_name('bone_names.txt').read_text(encoding='utf-8-sig').splitlines():
        if '=' not in line:
            continue
        key, name = line.split('=', 1)
        key, name = int(key.strip(), 16), name.strip()
        if key in result and result[key] != name:
            raise ValueError('Conflicting bundled bone names')
        result[key] = name
    return result


def mesh_alignment(package, rig):
    """Derive mesh-to-rig translation from explicitly linked inverse binds.

    The current geometry workflow supports a common Y translation only.
    Validate every joint and every linked catalog rather than guessing an
    offset from mesh extents or an arbitrary vertex.
    """
    shifts = []
    for rec in package.container.records:
        raw = rec.payload
        if rec.kind != 2013 or len(raw) < 52 or struct.unpack_from('<II', raw, 44) != (2138, rig.record.asset):
            continue
        at = raw.find(b'SKELETON')
        if at < 0 or raw.find(b'SKELETON', at + 8) >= 0 or at + 20 > len(raw):
            raise ValueError('Missing or ambiguous native SKELETON inverse binds')
        if struct.unpack_from('<I', raw, at + 8)[0] != 15:
            raise ValueError('Unsupported native SKELETON version')
        count = struct.unpack_from('<I', raw, at + 16)[0]
        if count != len(rig.joints) or at + 20 + count * 64 > len(raw):
            raise ValueError('Native inverse-bind count mismatch')
        for i, joint in enumerate(rig.joints):
            matrix = decode_inverse(raw, at + 20 + i * 64)
            linear = tuple(row[:3] for row in matrix[:3])
            product = mm(linear, joint.world_linear)
            if max(abs(product[r][c] - IDENTITY[r][c]) for r in range(3) for c in range(3)) > 2e-4:
                raise ValueError('Native inverse binds do not match skeleton axes')
            bind_position = mv(inverse(linear), tuple(-row[3] for row in matrix[:3]))
            shifts.append(sub(joint.world_position, bind_position))
    if not shifts:
        return 0.0  # Static/bone-attached package without skinned inverse binds.
    offset = statistics.median(s[1] for s in shifts)
    if not math.isfinite(offset) or any(norm(sub(s, (0., offset, 0.))) > 5e-5 for s in shifts):
        raise ValueError('Unsupported mesh/skeleton alignment; native binds are not a common Y translation')
    return offset


def metadata(path, package):
    rigs = [Rig(rec) for rec in package.container.records if rec.kind == 2138]
    if len(rigs) > 1:
        raise ValueError('This package contains multiple armatures; native multi-rig import is not supported yet')
    bones, offset = [], 0.0
    if rigs:
        rig = rigs[0]
        names = bone_dictionary()
        bones = [dict(name='Root', parent=-1, position=(0., 0., 0.), rotation=(0., 0., 0., 1.))]
        rotations = []
        for h, joint in zip(rig.hashes, rig.joints):
            w, x, y, z = struct.unpack_from('<4f', rig.record.payload, joint.offset + 4)
            rotation = quat_normalize((x, y, z, w))
            if joint.parent >= 0:
                rotation = quat_normalize(quat_mul(rotations[joint.parent], rotation))
            rotations.append(rotation)
            bones.append(dict(name=names.get(h, f'{h:08X}'), parent=joint.parent + 1,
                              position=joint.world_position, rotation=rotation))
        if len({b['name'] for b in bones}) != len(bones):
            raise ValueError('Duplicate native bone labels; cannot safely bind the armature')
        offset = mesh_alignment(package, rig)
    links = catalogs(package)
    # Import highest-detail render resources and static helper/eye catalogs.
    # Do not expose the lower LODs or cloth-input companion as duplicate meshes.
    lookup = {(r.kind, r.asset): r for r in package.container.records}
    def associated(c):
        if not rigs:
            return True
        raw = package.container.records[c.record].payload
        if raw[32:40] != b'ENTITY  ' or len(raw) < 52:
            return False
        key = struct.unpack_from('<II', raw, 44)
        if key == (2138, rigs[0].record.asset):
            return True
        node = lookup.get(key)
        return bool(node and node.kind == 2002 and len(node.payload) == 196 and
                    node.payload[4:12] == b'NODE    ' and
                    struct.unpack_from('<II', node.payload, 160) == (2138, rigs[0].record.asset))
    selected = {c.lods[0] for c in links if associated(c)}
    if not links:
        selected = {rec.index for rec in package.container.records if rec.payload[4:12] == b'MESHDATA'}
    selected.intersection_update(package.record_members)
    if not selected:
        raise ValueError('No supported external native render meshes found in this package')
    return dict(code=Path(path).stem, bones=bones, y_offset=offset,
                texture_folders=[], package_sha256=hashlib.sha256(package.source_data).hexdigest(),
                mesh_records=sorted(selected), import_metadata_reader='native_v1')
