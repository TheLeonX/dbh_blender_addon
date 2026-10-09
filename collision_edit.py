"""Position edits for native collision helpers; no inferred Havok authoring.

Keep the native vertex order, indices, packed cursor, declarations and opaque
channels. Collision helpers and Havok hclShape resources are separate data.
"""
import math
import struct


def replace_positions(md, sub, positions):
    """Patch one non-overlapping native range, including implicit VB cursors."""
    if len(positions) != sub.count or not positions:
        raise ValueError('Collision vertex edits must keep the original vertex count')
    if any(len(p) != 3 or not all(math.isfinite(v) for v in p) for p in positions):
        raise ValueError('Collision geometry contains a non-finite position')
    vb = md.vbs[sub.vb]
    attrs = [a for a in vb.attributes() if a[3] == 0]
    if len(attrs) != 1 or attrs[0][2] != 2:
        raise ValueError('Collision edits require one verified float3 position channel')
    stream, offset, _, _ = attrs[0]
    if stream >= len(vb.streams) or offset + 12 > vb.strides[stream]:
        raise ValueError('Invalid native collision position stream')
    first = md.vertex_origin(sub)
    if first < 0 or first + sub.count > vb.count:
        raise ValueError('Collision vertex range exceeds the native buffer')
    for other in md.flat():
        if other is sub or other.vb != sub.vb or not other.count: continue
        begin = md.vertex_origin(other)
        if max(first, begin) < min(first + sub.count, begin + other.count):
            raise ValueError('Collision vertices overlap another draw; separate-range editing is required')
    # Calculate and validate everything before touching the target buffer.
    encoded = [struct.pack('<3f', *p) for p in positions]
    positions = [struct.unpack('<3f', p) for p in encoded]
    box = struct.unpack_from('<15f', sub.suffix, 4)
    axes = [box[3 + i * 3:6 + i * 3] for i in range(3)]
    if (not all(math.isfinite(v) for a in axes for v in a) or
        any(abs(sum(v*v for v in a) - 1) > 1e-4 for a in axes) or
        any(abs(sum(x*y for x,y in zip(axes[i],axes[j]))) > 1e-4 for i in range(3) for j in range(i))):
        raise ValueError('Collision OBB axes are not a verified orthonormal frame')
    projected = [[sum(p[j] * a[j] for j in range(3)) for p in positions] for a in axes]
    lo, hi = [min(a) for a in projected], [max(a) for a in projected]
    center = tuple(sum((lo[k] + hi[k]) * .5 * axes[k][j] for k in range(3)) for j in range(3))
    extent = tuple(max((hi[i] - lo[i]) * .5 + 2e-6, 2e-6) for i in range(3))
    radius = max(math.sqrt(sum((p[j]-center[j])**2 for j in range(3))) for p in positions) + 2e-6
    suffix = bytearray(sub.suffix)
    struct.pack_into('<15f', suffix, 4, *center, *box[3:12], *extent)
    struct.pack_into('<4f', suffix, 64, *center, radius)
    for i, data in enumerate(encoded):
        start = (first + i) * vb.strides[stream] + offset
        vb.streams[stream][start:start+12] = data
    sub.suffix = bytes(suffix)
    # No topology was changed: retain CLUPSKME/BLSHAPES and all other channels.
    md.preserve_deformations = True


def extract_positions(context, obj, md, sub, metadata, package):
    """Evaluate object/attachment offsets in bind space, without corner splits."""
    from collections import Counter
    from mathutils import Matrix
    from .native import decode_faces
    from .legacy import _from_blender
    from .scene_export import slot_entry
    from .bone_attachment import active_attachment
    from .auxiliary_mesh import native_bind
    from .blender_io import object_armature
    from .scene_organization import preview_modifier
    binding = slot_entry(obj,metadata).get('native_attachment')
    attachment = active_attachment(obj,metadata)
    arm = attachment[0].target if attachment else object_armature(obj)
    if arm is None: arm = __import__('bpy').data.objects.get(metadata.get('armature_name',''))
    rest = arm.data.pose_position if arm else None
    muted = [(m,m.show_viewport) for m in obj.modifiers if m.type=='ARMATURE' or preview_modifier(m)]
    evaluated = None
    try:
        if arm: arm.data.pose_position = 'REST'
        for mod,_ in muted: mod.show_viewport = False
        context.view_layer.update()
        matrix = (arm.matrix_world.inverted() if arm else Matrix.Identity(4)) @ obj.matrix_world
        if binding:
            matrix = native_bind(package,binding,attachment[1] if attachment else 0).inverted() @ matrix
        if not all(math.isfinite(v) for row in matrix for v in row) or abs(matrix.determinant()) < 1e-12:
            raise ValueError(f'{obj.name}: collision transform is not finite/invertible')
        evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
        mesh = evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=context.evaluated_depsgraph_get())
        mesh.calc_loop_triangles()
        def canonical(face):
            # These helpers keep native indices and do not export face normals.
            # Flipping normals/winding in Blender doesn't change connectivity.
            return tuple(sorted(face))
        actual = Counter(canonical(tuple(t.vertices)) for t in mesh.loop_triangles)
        # mesh.validate() on import removes native repeated-index triangles and
        # duplicate polygons. Preserve those opaque native indices on export,
        # but compare against the visible, cleaned Blender connectivity.
        expected = Counter({canonical(f) for f in decode_faces(md,sub) if len(set(f)) == 3})
        if len(mesh.vertices) != sub.count or actual != expected:
            raise ValueError(f'{obj.name}: collision vertex edits support moving/scaling existing vertices; '
                             'keep the original vertex count and faces (new collider topology is not supported)')
        points = [list(_from_blender(matrix @ v.co)) for v in mesh.vertices]
        if not binding:
            for p in points: p[1] -= float(metadata.get('y_offset',0.))
        return points
    finally:
        if evaluated: evaluated.to_mesh_clear()
        for mod,enabled in muted: mod.show_viewport = enabled
        if arm: arm.data.pose_position = rest
        context.view_layer.update()
