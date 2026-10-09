"""Explicit MESH v29 -> ENTITY v7 -> NODE v5 bone links for static draws.

Keep collision/eye streams static. Native attachments are NOT skin weights.
Only identity-local NODEs with verified rig references are supported here;
other entity transforms are retained and not guessed.
"""
import struct
import math
from .native import u32
from .runtime_mesh import catalogs
from .skeleton import Rig


def bindings(package, metadata, include_transformed=False):
    rigs = [Rig(r) for r in package.container.records if r.kind == 2138]
    rigs = [r for r in rigs if r.matches(metadata.get('bones', []))]
    if len(rigs) != 1: return {}
    rig = rigs[0]
    lookup = {(r.kind, r.asset): r for r in package.container.records}
    result = {}
    mesh_cache = {}
    for c in catalogs(package):
        entity = package.container.records[c.record].payload
        if entity[32:40] != b'ENTITY  ' or u32(entity, 40) != 7: continue
        node = lookup.get(struct.unpack_from('<II', entity, 44))
        if not node or node.kind != 2002 or len(node.payload) != 196: continue
        raw = node.payload
        if raw[4:12] != b'NODE    ' or u32(raw, 12) != 5: continue
        if struct.unpack_from('<II', raw, 160) != (2138, rig.record.asset): continue
        bone = struct.unpack_from('<i', raw, 168)[0]
        if not -1 <= bone < len(rig.joints): continue
        # These fields are identity on this supported static local variant.
        # Nonidentity local nodes need a separate transform decoder.
        values = struct.unpack_from('<7f', raw, 20)
        if not include_transformed and max(abs(a-b) for a,b in zip(values, (0,0,0,1,0,0,0))) > 1e-6: continue
        for ri in c.lods:
            if ri not in mesh_cache: mesh_cache[ri] = package.mesh(ri)
            md = mesh_cache[ri]
            if c.group >= len(md.groups): continue
            offset = sum(len(g.meshes) for g in md.groups[:c.group])
            for si, sub in enumerate(md.groups[c.group].meshes):
                if any(a[3] in (6,7,8,9) for a in md.vbs[sub.vb].attributes()): continue
                key = f'{ri}:{offset+si}'
                value = dict(node=node.index, rig=rig.record.index, bone=bone+1)
                if key in result and result[key] != value:
                    raise ValueError('Static mesh has ambiguous native NODE attachments')
                result[key] = value
    return result


def collision_slots(package, metadata):
    """Non-rendering static NODE draws, including material-bearing capsules.

    A nonzero material is NOT evidence that a shape is a visible surface.
    The low render bits distinguish these helpers from the eye draws. The
    transformed variant is used for classification only, not bind decoding.
    """
    from .native import MeshData
    cache = {}
    result = set()
    for key in bindings(package,metadata,include_transformed=True):
        ri,mi = map(int,key.split(':'))
        if ri not in cache: cache[ri] = MeshData(package.container.records[ri].payload)
        sub = cache[ri].flat()[mi]
        if u32(sub.suffix,0) & 3 == 0: result.add(key)
    return result


def native_bind(package, binding, bone=None):
    from mathutils import Matrix
    from .animation_native_blender import AXIS
    rig = Rig(package.container.records[binding['rig']])
    bi = binding['bone'] if bone is None else bone
    if bi == 0: return Matrix.Identity(4)
    j = rig.joints[bi-1]
    matrix = Matrix(tuple((*row, pos) for row,pos in zip(j.world_linear,j.world_position)) + ((0,0,0,1),))
    return AXIS @ matrix @ AXIS.inverted()


def setup(obj, armature, metadata, package, binding, bone=None):
    """Target the display bone but preserve the native local joint axes."""
    from mathutils import Matrix
    bi = binding['bone'] if bone is None else bone
    name = metadata['bones'][bi]['name']
    constraint = obj.constraints.get('Detroit Bone Attachment')
    if constraint is None: constraint = obj.constraints.new('CHILD_OF')
    constraint.name = 'Detroit Bone Attachment'
    constraint.target = armature; constraint.subtarget = name
    constraint.influence = 1.; constraint.mute = bi == 0
    constraint.inverse_matrix = armature.data.bones[name].matrix_local.inverted() @ native_bind(package,binding,bi)
    obj.parent = armature
    obj.matrix_basis = Matrix.Identity(4)
    for mod in obj.modifiers:
        if mod.type == 'ARMATURE': mod.show_viewport = mod.show_render = False
    return constraint


def attachment_needs_bake(obj, metadata, package, binding):
    """A custom Child Of inverse is baked into local geometry, not discarded."""
    from .bone_attachment import active_attachment
    active = active_attachment(obj, metadata)
    if not active: return False
    c, bone = active
    if not all(getattr(c,'use_'+axis+'_'+coord) for axis in ('location','rotation','scale') for coord in ('x','y','z')):
        raise ValueError(f'{obj.name}: partial Child Of axes cannot be saved as a native attachment')
    if not all(math.isfinite(v) for row in c.inverse_matrix for v in row) or abs(c.inverse_matrix.determinant()) < 1e-12:
        raise ValueError(f'{obj.name}: Child Of inverse must be finite and invertible')
    expected = c.target.data.bones[c.subtarget].matrix_local.inverted() @ native_bind(package,binding,bone)
    return max(abs(c.inverse_matrix[i][j]-expected[i][j]) for i in range(4) for j in range(4)) > 2e-4


def export_binding(obj, metadata, package, binding):
    from .bone_attachment import active_attachment
    from mathutils import Matrix
    active = active_attachment(obj, metadata)
    bone = active[1] if active else 0
    attachment_needs_bake(obj, metadata, package, binding)
    raw = bytearray(package.container.records[binding['node']].payload)
    struct.pack_into('<i',raw,168,bone-1)
    return bytes(raw), bone
