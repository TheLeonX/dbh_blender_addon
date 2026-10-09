"""PC MESH v29 render-group routing, leaving Havok's group-zero inputs intact.

Experimental until exercised in gameplay. Do not delete cloth triangles in
group zero: 0x140286D60/0x140286FD0 expose those exact buffers to Havok.
"""
import copy
import struct
from dataclasses import dataclass

from .native import Reader, Submesh, VB, u32


@dataclass
class Catalog:
    record: int
    lods: tuple
    cloth: int | None
    group: int
    group_offset: int


def catalogs(package):
    """Resolve explicit references; never infer a family from adjacent records."""
    records = package.container.records
    lookup = {(r.kind, r.asset): r.index for r in records}
    result = []
    for rec in records:
        if rec.kind not in (2009, 2013):
            continue
        tag = rec.payload.find(b'MESH____')
        if tag < 0:
            continue
        if rec.payload.find(b'MESH____', tag + 1) >= 0:
            raise ValueError('Ambiguous MESH serializer in catalog')
        r = Reader(rec.payload, tag + 8)
        if r.uint() != 29:
            continue
        count = r.uint()
        if not 0 < count <= 255:
            raise ValueError('Unsupported MESH LOD count')
        lods = []
        for _ in range(count):
            key = struct.unpack('<II', r.take(8))
            r.take(4)  # LOD threshold
            if key not in lookup or key[0] != 2130:
                break
            lods.append(lookup[key])
        if len(lods) != count:
            continue  # External/not included catalog; not safe to rewrite.
        r.take(8)  # range
        cloth_key = struct.unpack('<II', r.take(8))
        cloth = lookup.get(cloth_key)
        if cloth_key != (0, 0) and (cloth is None or cloth_key[0] != 2130):
            continue
        r.take(12)  # physics reference and MESH flags
        at = r.pos
        result.append(Catalog(rec.index, tuple(lods), cloth, r.uint(), at))
    return result


def active_draw_indices(package, ri):
    groups = {c.group for c in catalogs(package) if ri in c.lods}
    md = package.mesh(ri)
    if not groups:
        return set(range(len(md.flat())))
    if max(groups) >= len(md.groups):
        raise ValueError('Catalog points to a missing mesh group')
    result, offset = set(), 0
    for gi, group in enumerate(md.groups):
        if gi in groups:
            result.update(range(offset, offset + len(group.meshes)))
        offset += len(group.meshes)
    return result


def keep_single_lod(package,edited_records=()):
    """Select the highest-detail render LOD, retaining all physics resources.

    Do not delete MESHDATA records: independent catalogs/Havok can share them.
    This truncates only verified MESH v29 LOD reference arrays. The surviving
    distance threshold becomes FLT_MAX, like the native last LOD.
    """
    links={c.record:c for c in catalogs(package)};planned=[]
    for rec in package.container.records:
        if rec.kind not in (2009,2013):continue
        tag=rec.payload.find(b'MESH____')
        if tag<0:continue
        if tag+16>len(rec.payload) or u32(rec.payload,tag+8)!=29:
            raise ValueError('Single Mesh LOD supports only verified MESH v29 catalogs')
        count=u32(rec.payload,tag+12)
        if count<=1:continue
        c=links.get(rec.index)
        if c is None:raise ValueError('Single Mesh LOD cannot safely rewrite external/unresolved LOD references')
        if set(edited_records).intersection(c.lods[1:]):
            raise ValueError('Single Mesh LOD requires editing the highest-detail LOD, not a lower-distance LOD')
        at=tag+16;data=bytearray(rec.payload[:at+12]+rec.payload[at+count*12:])
        struct.pack_into('<I',data,tag+12,1)
        struct.pack_into('<I',data,at+8,0x7f7fffff)
        report=dict(catalog=c.record,lods=[c.lods[0]],inactive_lods=list(c.lods[1:]),
                    physics_resources_preserved=True)
        planned.append((rec,bytes(data),report))
    for rec,payload,_ in planned:rec.payload=payload
    return [report for _,_,report in planned]


def _matches(group, material):
    return [(i, sub) for i, sub in enumerate(group.meshes) if sub.material == material]


def _selector(md, gi, i):
    """Use the active draw's selector, never the preserved simulation copy's.

    A routed group-one draw with selector zero is already independent of
    cloth. Group zero may retain its old conditional selector (and sometimes
    an authored material), but resurrecting that selector during re-export
    wrongly requires a second cloth-input match for the edited render draw.
    """
    sub = md.groups[gi].meshes[i]
    return sub.descriptor[5]


def _edit_selector(package, ri, gi, old):
    md = package.mesh(ri)
    slots = [i for i,s in enumerate(md.groups[gi].meshes) if s == old]
    if len(slots) != 1:
        raise ValueError('Edited draw does not identify a unique source slot')
    return _selector(md, gi, slots[0])


def _surface_matches(package, ri, gi, material, selector):
    md = package.mesh(ri)
    # A new render group is a copy of group zero.
    source_gi = gi if gi < len(md.groups) else 0
    return [(i,s) for i,s in _matches(md.groups[source_gi], material)
            if _selector(md, source_gi, i) == selector]


def _clone_draw(target, source, sub):
    vb = source.vbs[sub.vb]
    if vb.flag != 0 or vb.inline_streams:
        raise ValueError('Replacement must use an ordinary external vertex buffer')
    if len(target.vbs) >= 256:
        raise ValueError('Runtime vertex-buffer index exceeds uint8 capacity')
    streams = [bytearray(data[sub.first_vertex * stride:(sub.first_vertex + sub.count) * stride])
               for stride, data in zip(vb.strides, vb.streams)]
    new_vb = len(target.vbs)
    target.vbs.append(VB(sub.count, vb.flag, vb.layout, vb.strides, streams))
    target.padding.append(b'')
    ib = 0
    first_index = len(target.indices[ib]) // 2
    target.indices[ib].extend(source.indices[sub.ib][sub.first_index * 2:(sub.first_index + sub.index_count) * 2])
    desc = bytearray(sub.descriptor)
    for at, value in ((6, new_vb), (10, 0), (18, ib), (22, first_index)):
        struct.pack_into('<I', desc, at, value)
    desc[5] = 0  # ordinary draw, independent of the cloth-active bitmask
    return Submesh(bytes(desc), sub.suffix)


def _ordinary_surface_matches(package, source_ri, source_gi, old, target_ri, selector):
    """Match original draws before geometry/material replacement changes them.

    Material alone is insufficient for e.g. CURTIS's two 0x4D82 surfaces.
    Their retained 80-byte suffixes distinguish them across reordered LODs.
    Use this only as an exact, unique discriminator; never guess by ordering
    or nearest bounds when the serialized evidence is ambiguous.
    """
    source = _surface_matches(package, source_ri, source_gi, old.material, selector)
    slots = [(i, sub) for i, sub in source if sub == old]
    if len(slots) != 1:
        raise ValueError('Edited draw does not identify a unique source slot')
    if target_ri == source_ri:
        return slots
    targets = _surface_matches(package, target_ri, source_gi, old.material, selector)
    if not targets or (len(source) == 1 and len(targets) == 1):
        return targets
    source_matches = [(i, sub) for i, sub in source if sub.suffix == old.suffix]
    target_matches = [(i, sub) for i, sub in targets if sub.suffix == old.suffix]
    if len(source_matches) == len(target_matches) == 1:
        return target_matches
    raise ValueError(f'Cannot uniquely match material 0x{old.material[1]:X} '
                     f'from mesh {source_ri} to LOD {target_ri}; '
                     'duplicate draws have no unique retained descriptor')


def route_edited_cloth(package, meshes, edits, enabled=False):
    """edits: (record, original group, original draw, replacement draw list).

    Only explicit, unambiguous catalog/material links are supported. The
    original simulation VB/IB ranges and draw ordering are never changed.
    """
    links = catalogs(package)
    reports, prepared, routed = [], set(), set()
    for ri, gi, old, draws in edits:
        families = [c for c in links if ri in c.lods and c.group == gi and c.cloth is not None]
        for c in families:
            selector = _edit_selector(package, ri, gi, old)
            if selector == 0:
                continue  # Ordinary draws can share a cloth surface's material.
            cloth = meshes.get(c.cloth) or package.mesh(c.cloth)
            matches = [(i,s) for i,s in _matches(cloth.groups[0], old.material)
                       if (0x40 | s.descriptor[5]) == selector]
            if not matches:
                if selector & 0x3f:
                    raise ValueError('Edited conditional draw has no matching cloth input')
                continue
            if len(matches) != 1:
                raise ValueError('Ambiguous cloth material mapping; no package written')
            cloth_i, cloth_sub = matches[0]
            cloth_id = cloth_sub.descriptor[5]
            if not 1 <= cloth_id <= 32:
                raise ValueError('Unsupported cloth selector')
            if selector not in (0, 0x40 | cloth_id):
                raise ValueError('Cloth/ordinary draw selectors do not agree')
            if not enabled and draws:
                raise ValueError('Edited cloth needs "Bone-skin edited cloth" in export options. '
                                 'This experimental mode removes cloth motion from the edited surface.')
            key = (c.record, old.material, selector)
            if key in routed:
                raise ValueError('Multiple edited LODs/material slots target the same cloth surface')
            routed.add(key)
            if c.group not in (0, 1):
                raise ValueError('Only verified group-zero/group-one cloth routing is supported')
            all_records = (*c.lods, c.cloth)
            if c.record not in prepared:
                for target_ri in all_records:
                    if target_ri not in meshes:
                        meshes[target_ri] = package.mesh(target_ri)
                    target = meshes[target_ri]
                    if c.group == 0:
                        if len(target.groups) != 1:
                            raise ValueError('Cannot add a render group to an unknown group layout')
                        target.groups.append(copy.deepcopy(target.groups[0]))
                    elif len(target.groups) != 2:
                        raise ValueError('Previously routed package has an unexpected group layout')
                # Every simulation mapping still indexes group zero. Keep its
                # serialization intact apart from appending the new empty
                # per-group deformation count during Package.rebuild.
                empty = b'CLUPSKME' + struct.pack('<IB', 2, 0) + b'BLSHAPES' + struct.pack('<I', 3)
                if cloth.deformations != empty + bytes(4 * (c.group + 1)):
                    raise ValueError('Cloth companion has unhandled deformation data')
                rec = package.container.records[c.record]
                data = bytearray(rec.payload)
                struct.pack_into('<I', data, c.group_offset, 1)
                rec.payload = bytes(data)
                prepared.add(c.record)
            source = meshes[ri]
            for target_ri in c.lods:
                target = meshes[target_ri]
                group = target.groups[1]
                candidates = _surface_matches(package, target_ri, c.group, old.material, selector)
                if not candidates:
                    raise ValueError('A linked LOD lacks the edited cloth surface')
                if c.group == 0 and target_ri != ri and len(candidates) != 1:
                    raise ValueError('Multiple LOD draws share this material; mapping is ambiguous')
                # Extra 16-bit batches from an earlier conversion share the
                # material; retain all unrelated draw slots and order.
                if any(s.descriptor[5] not in (0, 0x40 | cloth_id) for _, s in candidates):
                    raise ValueError('Unexpected selector in a linked LOD')
                first = candidates[0][0]
                replacements = []
                for draw in draws:
                    if target_ri == ri:
                        desc = bytearray(draw.descriptor)
                        desc[5] = 0
                        replacements.append(Submesh(bytes(desc), draw.suffix))
                    else:
                        replacements.append(_clone_draw(target, source, draw))
                # Keep all unrelated draw indices stable. New 16-bit batches
                # go at the end, not between existing material slots.
                for n, (at, existing) in enumerate(candidates):
                    if n < len(replacements):
                        group.meshes[at] = replacements[n]
                    else:
                        desc = bytearray(existing.descriptor)
                        struct.pack_into('<I', desc, 26, 0)
                        group.meshes[at] = Submesh(bytes(desc), existing.suffix)
                if target_ri != ri or c.group != gi:
                    group.meshes.extend(replacements[len(candidates):])
                group.suffix = source.groups[gi].suffix
            render_cloth = meshes[c.cloth].groups[1]
            desc = bytearray(render_cloth.meshes[cloth_i].descriptor)
            struct.pack_into('<I', desc, 26, 0)
            render_cloth.meshes[cloth_i] = Submesh(bytes(desc), render_cloth.meshes[cloth_i].suffix)
            reports.append(dict(catalog=c.record, material=old.material[1], lods=list(c.lods),
                                cloth=c.cloth, cloth_draw=cloth_i, selector=selector, render_group=1,
                                simulation_inputs_preserved=True))
    return reports


def route_render_edits(package,meshes,edits,enabled=False):
    """Propagate ordinary edits/deletions to explicit LODs, preserving Havok."""
    original=catalogs(package)
    reports=route_edited_cloth(package,meshes,edits,enabled)
    cloth_keys={(r['catalog'],r['material'],r['selector']) for r in reports}
    current={c.record:c for c in catalogs(package)}
    done=set()
    for ri,gi,old,draws in edits:
        for before in original:
            if ri not in before.lods or gi!=before.group:continue
            selector=_edit_selector(package,ri,gi,old)
            key=before.record,old.material[1],selector
            if key in cloth_keys:continue
            c=current[before.record];source=meshes[ri]
            for target_ri in c.lods:
                target=meshes.setdefault(target_ri,package.mesh(target_ri)) if target_ri not in meshes else meshes[target_ri]
                group=target.groups[c.group]
                candidates=_ordinary_surface_matches(package,ri,gi,old,target_ri,selector)
                if not candidates:continue # This surface is absent at this LOD.
                at,existing=candidates[0]
                target_key=(before.record,target_ri,at)
                if target_key in done:
                    raise ValueError('Multiple edited draws/LODs target the same original surface')
                done.add(target_key)
                if not draws:
                    desc=bytearray(existing.descriptor);struct.pack_into('<I',desc,26,0)
                    group.meshes[at]=Submesh(bytes(desc),existing.suffix)
                elif target_ri!=ri or c.group!=gi:
                    replacements=[_clone_draw(target,source,draw) for draw in draws]
                    group.meshes[at]=replacements[0]
                    if target_ri!=ri:group.meshes.extend(replacements[1:])
                    group.suffix=source.groups[gi].suffix
            reports.append(dict(catalog=c.record,material=old.material[1],lods=list(c.lods),
                                deleted=not draws,ordinary_lod_route=True))
    return reports
