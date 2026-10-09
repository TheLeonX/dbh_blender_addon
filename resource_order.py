"""PC mesh finalization requires local SHADCUST objects to be read first.

Only serialization order changes. In-memory record indexes remain source
indexes throughout export, so open Blender scenes keep their source bindings.
"""
import copy


def mesh_materials(record):
    if record.kind != 2130:
        return ()
    from .native import MeshData
    return tuple(dict.fromkeys(s.material for s in MeshData(record.payload).flat()
                               if s.material[0] == 2133))


def serialization_order(records):
    lookup = {(r.kind, r.asset): r for r in records}
    if len(lookup) != len(records):
        raise ValueError('Duplicate container resource IDs')
    result, emitted = [], set()

    def emit(rec):
        if rec.index not in emitted:
            result.append(rec); emitted.add(rec.index)

    for rec in records:
        if rec.kind == 2130:
            from .native import MeshData, u32
            import struct
            refs = MeshData(rec.payload).references
            for n in range(u32(refs,0)&255):
                key = struct.unpack_from('<II',refs,4+8*n)
                if key[0] == 2150 and key in lookup: emit(lookup[key])
        for key in mesh_materials(rec):
            material = lookup.get(key)
            if material is None or material.index in emitted:
                continue  # Out-of-package references retain engine resolution.
            # Move textures along with a late material. Do not globally reorder
            # existing materials/textures: vanilla supports lazy texture refs.
            from .materials import bindings
            for binding in bindings(material.payload):
                texture = lookup.get((2137, binding.texture))
                if texture is not None:
                    emit(texture)
            emit(material)
        emit(rec)
    return result


def validate_material_order(records):
    positions = {(r.kind, r.asset): i for i, r in enumerate(records)}
    for i, rec in enumerate(records):
        for key in mesh_materials(rec):
            if key in positions and positions[key] >= i:
                raise ValueError(f'Mesh {rec.asset:X} precedes required material {key[1]:X}')


def remap_metadata(metadata, index_map):
    """Only known record-index fields, never asset IDs or bone/draw indices."""
    result = copy.deepcopy(metadata)
    def mapped(value):
        if value is None:return None
        if value not in index_map:raise ValueError(f'Unknown metadata record index {value}')
        return index_map[value]
    if 'mesh_records' in result:
        result['mesh_records'] = [mapped(i) for i in result['mesh_records']]
    if 'imported_slots' in result:
        result['imported_slots'] = [[mapped(ri), mi] for ri, mi in result['imported_slots']]
    for item in result.get('deleted_meshes', []):item['record'] = mapped(item['record'])
    for item in result.get('bone_skinned_cloth', []):
        for key in ('catalog', 'cloth'):
            if key in item:item[key] = mapped(item[key])
        if 'lods' in item:item['lods'] = [mapped(i) for i in item['lods']]
    for item in result.get('preset_materials', []):
        if 'surface' in item:item['surface'][0] = mapped(item['surface'][0])
    for item in result.get('cloth_simulation_masks', []):
        item['record'] = mapped(item['record'])
        item['resource_record'] = mapped(item['resource_record'])
        for key in ('catalog','cloth_record'):
            if key in item:item[key]=mapped(item[key])
        if 'lods' in item:item['lods']=[mapped(i) for i in item['lods']]
    for item in result.get('single_lod_catalogs',[]):
        item['catalog']=mapped(item['catalog'])
        for key in ('lods','inactive_lods'):item[key]=[mapped(i) for i in item[key]]
    return result
