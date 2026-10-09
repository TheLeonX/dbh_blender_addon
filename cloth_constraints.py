"""Mass-normalized native StandardLink coefficients and guarded migration.

Captured Epic solver RVA 0x1631290 multiplies distance error by the stored
coefficient, then by each particle's inverse mass. Vanilla coefficients times
the sum of inverse masses are 1.0. UI stiffness is the normalized fraction,
not the stored coefficient. Equal-length name markers prevent double repair.
"""
import math
import struct

LINK_NAMES = {'DBH Stretch Links': 'DBH Stretch Mass2',
              'DBH Bend Links': 'DBH Bend Mass2'}


def link_coefficient(stiffness, inverse_a, inverse_b):
    if not all(math.isfinite(v) for v in (stiffness, inverse_a, inverse_b)):
        raise ValueError('Non-finite cloth constraint parameter')
    if not 0 <= stiffness <= 1 or min(inverse_a, inverse_b) < 0:
        raise ValueError('Invalid cloth stiffness or inverse mass')
    total = inverse_a + inverse_b
    result = stiffness / total if total else 0.
    try:
        return struct.unpack('<f', struct.pack('<f', result))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError('Cloth link coefficient exceeds native float range') from exc


def repair_authored_constraints(payload):
    """Only migrate recognized authored links; preserve framing/all other bytes."""
    from .cloth_native import resource_tag
    from .havok_encode import Graph, Node
    if not payload[20:].startswith(b'DBH_CLOTH_'):
        return payload, []
    tag = resource_tag(payload); graph = Graph(tag)
    root = next(tag.objects('hclClothData'))
    sims=[]
    for ptr in tag.array(tag.field(root,'simClothDatas')):
        sim=tag.array(ptr)[0]
        sets=[]
        for reference in tag.array(tag.field(sim,'staticConstraintSets')):
            cs=tag.array(reference)[0]
            if tag.types[cs[0]].name=='hclStandardLinkConstraintSet':sets.append(graph.read(cs))
        sims.append(Node(sim[0],dict(particleDatas=graph.read(tag.field(sim,'particleDatas')),staticConstraintSets=sets)))
    cloth=Node(root[0],{'simClothDatas':sims})
    locations = {id(node): obj for obj, node in graph.cache.items()}
    patches = {}; report = []

    def patch(at, data):
        if at in patches and patches[at] != data:
            raise ValueError('Shared cloth constraint has inconsistent particle masses')
        patches[at] = data

    for simnode in cloth.value['simClothDatas']:
        sim = simnode.value; particles = sim['particleDatas']
        sets = [s for s in sim['staticConstraintSets']
                if tag.types[s.type].name == 'hclStandardLinkConstraintSet']
        expected = set(LINK_NAMES) | set(LINK_NAMES.values())
        names = [s.value['name'] for s in sets]
        if len(sets) != 2 or len(set(names)) != 2 or any(n not in expected for n in names):
            raise ValueError('Unknown authored cloth link layout; re-export from Blender')
        families = [next(old for old, new in LINK_NAMES.items() if n in (old, new)) for n in names]
        if set(families) != set(LINK_NAMES):
            raise ValueError('Missing authored cloth stretch/bend constraints')
        for cs in sets:
            name = cs.value['name']; legacy = name in LINK_NAMES
            values = [link.value['stiffness'] for link in cs.value['links']]
            if legacy and (not all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                           or len(set(values)) > 1):
                raise ValueError('Ambiguous legacy cloth stiffness; re-export from Blender')
            for linknode in cs.value['links']:
                link = linknode.value; a = link['particleA']; b = link['particleB']
                if not 0 <= a < len(particles) or not 0 <= b < len(particles) or a == b:
                    raise ValueError('Invalid cloth link particle index')
                inverse_a = particles[a].value['invMass']; inverse_b = particles[b].value['invMass']
                if not all(math.isfinite(v) and v >= 0 for v in (inverse_a, inverse_b)):
                    raise ValueError('Invalid cloth particle inverse mass')
                coefficient = link['stiffness']; total = inverse_a + inverse_b
                if not math.isfinite(coefficient) or coefficient < 0:
                    raise ValueError('Invalid cloth link coefficient')
                if legacy:
                    field = tag.field(locations[id(linknode)], 'stiffness')
                    # Check reflected scalar type before making the byte patch.
                    if tag.types[tag.base(field[0])].subtype & 255 != 5:
                        raise ValueError('Unexpected native cloth coefficient storage')
                    patch(field[1], struct.pack('<f', link_coefficient(coefficient, inverse_a, inverse_b)))
                elif coefficient * total > 1.00001 or (total == 0 and coefficient != 0):
                    raise ValueError('Unstable normalized cloth link; re-export from Blender')
            if legacy:
                chars = tag.array(tag.field(locations[id(cs)], 'name'))
                old = name.encode('ascii') + b'\0'; new = LINK_NAMES[name].encode('ascii') + b'\0'
                if len(old) != len(new) or len(chars) != len(old):
                    raise ValueError('Unexpected native cloth constraint name storage')
                at = chars[0][1]
                if any(obj[1] != at+i for i, obj in enumerate(chars)) or bytes(tag.raw[at:at+len(old)]) != old:
                    raise ValueError('Noncontiguous authored cloth name storage')
                patch(at, new)
                report.append(dict(constraint=name, marker=LINK_NAMES[name], links=len(values)))
    if not patches:
        return payload, []
    for at, data in patches.items():
        tag.raw[at:at+len(data)] = data
    result = bytes(tag.raw)
    if len(result) != len(payload) or repair_authored_constraints(result)[1]:
        raise ValueError('Cloth constraint migration failed readback/idempotence')
    return result, report
