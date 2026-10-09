"""Native per-render-vertex Havok cloth masks, without rebuilding topology.

Game user buffers are group-zero draws selected by CLOTHDAT reference order.
Each reflected blend operator retains its own state-dependent weights. The
painted mask either scales those weights (legacy scenes) or edits their native
envelope (vanilla inspection). Particle/collision/operator graphs stay intact.
"""
from dataclasses import dataclass
import math
import struct

from .havok_tag import TagFile
from .native import u32
from .runtime_mesh import catalogs

GROUP = 'CLOTH_SIMULATION'
PIN_GROUP = '_DBH_CLOTH_PIN'
PREVIEW = 'Detroit Cloth Preview'


@dataclass
class Binding:
    resource: int
    asset: int
    buffer: int
    count: int
    cloth_record: int
    cloth_draw: int
    name: str
    tag: TagFile
    entries: list
    support: list


def resource_tag(payload):
    if payload[:12] != b'\x01\0\0\0CLOTHDAT' or u32(payload, 12) != 13:
        raise ValueError('Native cloth painting requires CLOTHDAT v13')
    name_length = u32(payload, 16)
    start = 24 + name_length
    if start > len(payload): raise ValueError('Truncated CLOTHDAT name')
    size = u32(payload, start - 4)
    tag = TagFile(payload, start)
    if tag.root[2] - start != size or len(payload) - tag.root[2] != 66:
        raise ValueError('Unsupported CLOTHDAT v13 framing')
    return tag


def repair_authored_array_items(payload):
    """Repair only add-on-authored Havok ITEM array flags from older exports."""
    if payload[:12] != b'\x01\0\0\0CLOTHDAT' or u32(payload, 12) != 13:
        return payload
    name_length = u32(payload, 16)
    if payload[20:20 + name_length].startswith(b'DBH_CLOTH_'):
        from .havok_encode import array_item_flags
        raw, _ = array_item_flags(resource_tag(payload), repair=True)
        return raw
    return payload


class ClothIndex:
    def __init__(self, package):
        self.package = package
        self.links = catalogs(package)
        self.tags = {}
        self.bindings = {}
        self.profile_cache = {}

    def profiles(self, binding):
        if binding.resource not in self.profile_cache:
            from .cloth_profiles import read_profiles
            self.profile_cache[binding.resource] = read_profiles(bytes(binding.tag.raw))
        return self.profile_cache[binding.resource]

    def binding(self, record, mesh):
        key = record, mesh
        if key in self.bindings: return self.bindings[key]
        families = [c for c in self.links if record in c.lods and c.cloth is not None]
        if len(families) != 1: raise ValueError('This mesh has no unique native cloth family')
        family = families[0]
        md = self.package.mesh(record)
        draws = md.flat()
        if mesh >= len(draws): raise ValueError('Native cloth mesh slot is missing')
        sub = draws[mesh]
        selector = sub.descriptor[5]
        cloth_id = selector & 63
        if not selector & 64 or not 1 <= cloth_id <= 32:
            raise ValueError('This surface is bone-skinned; native cloth needs an original conditional cloth draw')
        cloth = self.package.mesh(family.cloth)
        matching = [(i, s) for i, s in enumerate(cloth.groups[0].meshes)
                    if s.descriptor[5] == cloth_id and s.material == sub.material]
        if len(matching) != 1: raise ValueError('No unique native cloth surface for this material/selector')
        cloth_draw, source = matching[0]
        if source.count != sub.count:
            raise ValueError('Paint native cloth on the matching full-resolution mesh; this LOD has different topology')
        refs = cloth.references
        count = u32(refs, 0) & 255
        if cloth_id > count: raise ValueError('Cloth selector exceeds its resource references')
        kind, asset = struct.unpack_from('<II', refs, 4 + (cloth_id - 1) * 8)
        if kind != 2150: raise ValueError('Cloth reference is not CLOTHDAT')
        records = [r for r in self.package.container.records if (r.kind, r.asset) == (kind, asset)]
        if len(records) != 1: raise ValueError('Referenced CLOTHDAT is absent or ambiguous')
        rec = records[0]
        if rec.index not in self.tags:
            raw = (self.package.members[self.package.record_members[rec.index]].unpacked
                   if rec.external else rec.payload)
            self.tags[rec.index] = resource_tag(raw)
        tag = self.tags[rec.index]
        roots = list(tag.objects('hclClothData'))
        if len(roots) != 1: raise ValueError('Expected one Havok cloth root')
        buffers = []
        for bi, ptr in enumerate(tag.array(tag.field(roots[0], 'bufferDefinitions'))):
            obj = tag.array(ptr)
            if len(obj) != 1: raise ValueError('Invalid Havok buffer definition')
            obj = obj[0]
            if tag.value(tag.field(obj, 'type')) == 4 and tag.value(tag.field(obj, 'numVertices')) == source.count:
                buffers.append((bi, tag.value(tag.field(obj, 'name'))))
        if len(buffers) != 1:
            raise ValueError('Native cloth buffer is ambiguous; cannot map by vertex count safely')
        bi, name = buffers[0]
        entries, support = [], [0.] * source.count
        for op in tag.objects('hclBlendSomeVerticesOperator'):
            if tag.value(tag.field(op, 'bufferIdx_C')) != bi: continue
            if tag.value(tag.field(op, 'bufferIdx_B')) != bi or tag.value(tag.field(op, 'dynamicBlend')):
                raise ValueError('Unsupported dynamic/native cloth blending')
            constant = tag.field(op, 'blendVertices')
            if tag.array(tag.field(constant, 'vertexIndices')):
                raise ValueError('Constant blend vertex lists require another native cloth encoder')
            for entry in tag.array(tag.field(op, 'blendEntries')):
                vertex = tag.value(tag.field(entry, 'vertexIndex'))
                field = tag.field(entry, 'blendWeight')
                weight = tag.value(field)
                if not 0 <= vertex < source.count or not math.isfinite(weight) or not 0 <= weight <= 1:
                    raise ValueError('Invalid native cloth blend entry')
                entries.append((vertex, field))
                support[vertex] = max(support[vertex], weight)
        if not entries: raise ValueError('This native cloth buffer has no supported per-vertex blending')
        binding = Binding(rec.index, rec.asset, bi, source.count, family.cloth, cloth_draw,
                          name, tag, entries, support)
        self.bindings[key] = binding
        return binding


def baseline_for(binding, saved=None):
    baseline = (dict(saved['baseline']) if saved else
                {field[1]: binding.tag.value(field) for _, field in binding.entries})
    baseline = {int(k): v for k, v in baseline.items()}
    if set(baseline) != {field[1] for _, field in binding.entries} or any(
            not math.isfinite(v) or not 0 <= v <= 1 for v in baseline.values()):
        raise ValueError('Saved cloth baseline does not match the native blend table; reimport a matching package')
    return baseline


def envelope(binding, baseline):
    """Maximum native blend over operators/states, in render-vertex order."""
    weights = [0.] * binding.count
    for vertex, field in binding.entries:
        weights[vertex] = max(weights[vertex], baseline[field[1]])
    return weights


def apply_masks(index, requests, previous=()):
    """Plan and apply masks once per native buffer, retaining pristine baselines.

    The sidecar baseline prevents repeated exports/imports multiplying a mask
    again. A record's physical fields keep the same offsets and ITEM indexes.
    """
    previous = {(r['resource_id'], r['buffer']): r for r in previous}
    planned = {}
    for request in requests:
        record, mesh, weights = request[:3]
        binding = index.binding(record, mesh)
        weights = list(weights)
        if len(weights) != binding.count or any(not math.isfinite(w) or not 0 <= w <= 1 for w in weights):
            raise ValueError('CLOTH_SIMULATION needs one finite 0..1 weight per native vertex')
        key = binding.asset, binding.buffer
        saved = previous.get(key)
        mode = request[3] if len(request) > 3 and request[3] else (saved or {}).get('weight_mode', 'MULTIPLIER')
        movement = request[4] if len(request)>4 else (saved or {}).get('movement',1.)
        if not math.isfinite(movement) or not 0<=movement<=1: raise ValueError('Cloth movement must be 0..1')
        if mode not in ('ABSOLUTE', 'MULTIPLIER'): raise ValueError('Unknown native cloth weight mode')
        if key in planned:
            old = planned[key]
            if weights == old[1] and mode == old[5] and movement == old[6]: continue
            # Default imported LOD views may share one buffer. An untouched
            # duplicate is not a competing edit; two different edits still are.
            initial = ((saved or {}).get('weights') if (saved or {}).get('weight_mode') == 'ABSOLUTE'
                       else envelope(binding, old[2]))
            if movement != old[6]:
                if mode==old[5]=='ABSOLUTE' and weights==initial and movement==1.:continue
                if not (mode==old[5]=='ABSOLUTE' and old[1]==initial and old[6]==1.):
                    raise ValueError('Conflicting movement amounts on one native cloth buffer')
            if mode == old[5] == 'ABSOLUTE':
                if weights == initial and movement==old[6]: continue
                if old[1] == initial:
                    planned.pop(key)
                else: raise ValueError('Two edited mesh/LOD masks conflict on the same native cloth buffer')
            else: raise ValueError('Two mesh/LOD masks conflict on the same native cloth buffer')
        baseline = baseline_for(binding, saved)
        if mode == 'ABSOLUTE':
            covered = {v for v, _ in binding.entries}
            if any(w > 0 and i not in covered for i, w in enumerate(weights)):
                raise ValueError('This vertex has no native cloth blend entry; extending vanilla physics needs a topology rebuild')
        planned[key] = binding, weights, baseline, record, mesh, mode, movement
    reports = []
    for binding, weights, baseline, record, mesh, mode, movement in planned.values():
        changed = 0
        native = envelope(binding, baseline)
        for vertex, field in binding.entries:
            original = baseline[field[1]]
            value = original * weights[vertex]
            if mode == 'ABSOLUTE':
                # Preserve original bytes at the native envelope; otherwise
                # preserve relative state-specific blends rather than square them.
                value = (original if weights[vertex] == native[vertex] else
                         original * (weights[vertex] / native[vertex]) if native[vertex] else weights[vertex])
            value *= movement
            if struct.pack('<f', value) != struct.pack('<f', binding.tag.value(field)):
                binding.tag.write_float(field, value); changed += 1
        report = dict(previous.get((binding.asset, binding.buffer), {}))
        report.update(record=record, mesh=mesh, resource_id=binding.asset, buffer=binding.buffer,
                      resource_record=binding.resource, name=binding.name, weights=weights,
                      baseline=sorted(baseline.items()), changed_entries=changed, weight_mode=mode,movement=movement)
        reports.append(report)
    payloads, resources = {}, {}
    for ri, tag in index.tags.items():
        rec = index.package.container.records[ri]
        original = (index.package.members[index.package.record_members[ri]].unpacked if rec.external else rec.payload)
        raw = bytes(tag.raw)
        if raw == original: continue
        resource_tag(raw)  # read back every edited table before serialization
        (resources if rec.external else payloads)[ri] = raw
    return payloads, resources, reports
