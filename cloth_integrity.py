"""Guard native cloth initialization and recover known legacy export metadata.

The native hclSimClothInstance constructor indexes collidablePinchingDatas
once for every perInstanceCollidable, even with pinchDetectionEnabled == 0.
Never synthesize these records or select a donor by an unrelated mesh name.
"""
import copy
import struct
from .havok_encode import Graph, Node, Writer


def validate_collider_metadata(cloth):
    for sim in cloth.value['simClothDatas']:
        shapes = sim.value['perInstanceCollidables']
        metadata = sim.value['collidablePinchingDatas']
        if len(shapes) != len(metadata):
            raise ValueError('Native cloth requires one collidablePinchingDatas record per collision shape '
                             f'({len(shapes)} shapes, {len(metadata)} records)')


def _named(value, tag):
    if isinstance(value, Node): return tag.types[value.type].name, _named(value.value, tag)
    if isinstance(value, dict): return {k: _named(v, tag) for k, v in value.items()}
    if isinstance(value, list): return [_named(v, tag) for v in value]
    return value


def _matching_metadata(package, record, target_graph, cloth, sim):
    from .cloth_native import resource_tag
    from .runtime_mesh import catalogs
    from .native import u32
    families = []
    for catalog in catalogs(package):
        if catalog.cloth is None: continue
        refs = package.mesh(catalog.cloth).references
        keys = {struct.unpack_from('<II', refs, 4 + 8 * i) for i in range(u32(refs, 0) & 255)}
        if (record.kind, record.asset) in keys: families.append(keys)
    if len(families) != 1:
        raise ValueError('Cannot repair cloth collider metadata without one verified character family')
    matches = []
    for donor in package.container.records:
        if donor.kind != 2150 or donor is record or (donor.kind, donor.asset) not in families[0]: continue
        payload = package.members[package.record_members[donor.index]].unpacked if donor.external else donor.payload
        tag = resource_tag(payload); graph = Graph(tag)
        candidate = graph.read(next(tag.objects('hclClothData')))
        if candidate.value['name'].startswith('DBH_CLOTH_') or not candidate.value['simClothDatas']: continue
        # v0.9.3-v0.9.10 always copied the donor's first simulation's shapes.
        # Require the exact ordered shapes, external map and transform schema.
        source = candidate.value['simClothDatas'][0]
        if (_named(source.value['perInstanceCollidables'], tag) != _named(sim.value['perInstanceCollidables'], target_graph.tag)
            or _named(source.value['collidableTransformMap'], tag) != _named(sim.value['collidableTransformMap'], target_graph.tag)
            or _named(candidate.value['transformSetDefinitions'], tag) != _named(cloth.value['transformSetDefinitions'], target_graph.tag)):
            continue
        validate_collider_metadata(candidate)
        values = target_graph.rebase(copy.deepcopy(source.value['collidablePinchingDatas']), graph)
        matches.append((candidate.value['name'], donor.asset, values))
    if len(matches) != 1:
        raise ValueError('Cannot uniquely repair missing cloth collider metadata; use the original model as the export source')
    return matches[0]


def repair_authored_cloth(payload, package, record):
    """Only migrate the known empty-metadata bug in marked authored resources."""
    from .cloth_native import resource_tag, repair_authored_array_items
    raw = repair_authored_array_items(payload)
    if not raw[20:].startswith(b'DBH_CLOTH_'): return raw, []
    tag = resource_tag(raw)
    root=next(tag.objects('hclClothData'))
    sizes=[]
    for ptr in tag.array(tag.field(root,'simClothDatas')):
        sim=tag.array(ptr)[0]
        sizes.append((len(tag.array(tag.field(sim,'perInstanceCollidables'))),
                      len(tag.array(tag.field(sim,'collidablePinchingDatas')))))
    if all(a==b for a,b in sizes):return raw,[]
    graph = Graph(tag)
    cloth = graph.read(next(tag.objects('hclClothData')))
    repairs = []
    for sim in cloth.value['simClothDatas']:
        shapes = sim.value['perInstanceCollidables']; values = sim.value['collidablePinchingDatas']
        if len(shapes) == len(values): continue
        if values: validate_collider_metadata(cloth)  # Do not overwrite partial custom metadata.
        name, asset, metadata = _matching_metadata(package, record, graph, cloth, sim)
        sim.value['collidablePinchingDatas'] = metadata
        repairs.append(dict(physics_donor=name, donor_id=asset, collision_metadata_count=len(metadata)))
    validate_collider_metadata(cloth)
    if repairs:
        encoded = Writer(graph).pack(cloth)
        raw = raw[:tag.start - 4] + struct.pack('<I', len(encoded)) + encoded + raw[tag.root[2]:]
        check = resource_tag(raw); validate_collider_metadata(Graph(check).read(next(check.objects('hclClothData'))))
    return raw, repairs
