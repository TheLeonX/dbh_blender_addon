"""Native high-influence cloth weights and narrowly scoped legacy migration.

Epic captured deformer RVA 0x1638D50, switch cases 5..8: uint16 weights,
AND 0xff00, then float multiply by 1/65280. Cases 0..2 use uint8 /255.
Do not confuse the reflected uint16 storage with a full-range UNORM16.
"""
import struct

HIGH_BLOCKS = {'Five': 5, 'Six': 6, 'Seven': 7, 'Eight': 8}


def repair_authored_skin_weights(payload):
    """Patch only exact legacy low-byte blocks; retain every other byte."""
    from .cloth_native import resource_tag
    if not payload[20:].startswith(b'DBH_CLOTH_'):
        return payload, []
    tag = resource_tag(payload)
    patches = []; report = []
    for label, count in HIGH_BLOCKS.items():
        kind = 'hclObjectSpaceDeformer::' + label + 'BlendEntryBlock'
        for block in tag.objects(kind):
            field = tag.field(block, 'boneWeights')
            values = tag.value(field)
            if len(values) != count * 16:
                raise ValueError('Unexpected native cloth blend block capacity')
            rows = [values[n:n+count] for n in range(0, len(values), count)]
            correct = all(all(v & 255 == 0 for v in row) and sum(row) == 65280 for row in rows)
            if correct:
                continue
            legacy = all(all(0 <= v <= 255 for v in row) and sum(row) == 255 for row in rows)
            if not legacy:
                raise ValueError('Invalid or ambiguous high-influence authored cloth weights; re-export from Blender')
            at = field[1]
            if bytes(tag.raw[at:at+len(values)*2]) != struct.pack('<'+'H'*len(values), *values):
                raise ValueError('Unexpected high-influence cloth weight storage')
            patches.append((at, struct.pack('<'+'H'*len(values), *(v << 8 for v in values))))
            report.append(dict(skin_weight_block=label, skin_weight_offset=at, skin_weight_values=len(values)))
    if not patches:
        return payload, []
    for at, data in patches:
        tag.raw[at:at+len(data)] = data
    result = bytes(tag.raw)
    if len(result) != len(payload):
        raise ValueError('Cloth weight migration changed resource size')
    if repair_authored_skin_weights(result)[1]:
        raise ValueError('Cloth weight migration is not idempotent')
    return result, report
