"""Reflection-driven Havok 20160200 graph writer using a donor's native TYPE.

The TYPE table is retained verbatim. DATA allocations, ITEM indexes and PTCH
fixups are constructed anew from typed values, including subclass pointers.
"""
from dataclasses import dataclass
import struct


@dataclass
class Node:
    type: int
    value: object


class Graph:
    def __init__(self, tag):
        self.tag = tag
        self.cache = {}
        self._indexes = {}
        for i, typ in enumerate(tag.types):
            self._indexes.setdefault(typ.name, []).append(i)

    def index(self, name):
        found = self._indexes.get(name, [])
        if len(found) != 1: raise ValueError(f'Havok type {name} is unavailable/ambiguous')
        return found[0]

    def new(self, type_name, **fields):
        typ = self.index(type_name)
        value = self.zero(typ)
        if not isinstance(value, dict): raise ValueError('New Havok object must be a class')
        if set(fields) - set(value): raise ValueError(f'Unknown fields in {type_name}: {set(fields) - set(value)}')
        value.update(fields)
        return Node(typ, value)

    def rebase(self,value,source):
        """Move typed values to another native TYPE table, with schema guards.

        Class IDs are local to each tagfile. Reusing their integers would
        silently change classes; map named classes and check inherited field
        offsets/storage/pointer targets instead. No TYPE definitions invented.
        """
        memo={};checked=set()
        def schema(old,new):
            if old==0 or new==0:
                if old!=new:raise ValueError('Incompatible native Havok void pointer schema')
                return
            key=old,new
            if key in checked:return
            checked.add(key)
            a=source.tag.types[source.tag.base(old)];b=self.tag.types[self.tag.base(new)]
            if (a.name,a.subtype,a.size,a.alignment)!=(b.name,b.subtype,b.size,b.alignment):
                raise ValueError(f'Incompatible native Havok schema for {a.name}')
            kind=a.subtype&255
            if kind==7:
                am=source.tag.members(old);bm=self.tag.members(new)
                if [(n,o,f) for n,o,t,f in am]!=[(n,o,f) for n,o,t,f in bm]:
                    raise ValueError(f'Incompatible native Havok fields for {a.name}')
                for (_,_,at,_),(_,_,bt,_) in zip(am,bm):schema(at,bt)
            elif kind in (6,8,40):
                # Polymorphic pointers can target different derived classes,
                # but their declared base and storage must match exactly.
                schema(a.pointer,b.pointer)
        def move(v):
            if isinstance(v,Node):
                if id(v) in memo:return memo[id(v)]
                typ=self.index(source.tag.types[v.type].name);schema(v.type,typ)
                node=Node(typ,None);memo[id(v)]=node;node.value=move(v.value);return node
            if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
            if isinstance(v,list):return [move(x) for x in v]
            return v
        return move(value)

    def zero(self, typ):
        base = self.tag.types[self.tag.base(typ)]
        kind = base.subtype & 255
        if kind in (2, 4, 5): return 0
        if kind == 3: return ''
        if kind == 6: return None
        if kind == 8: return []
        if kind == 7: return {n: self.zero(t) for n, at, t, flags in self.tag.members(typ)}
        if kind == 40: return [self.zero(base.pointer) for _ in range(base.subtype >> 8)]
        raise ValueError(f'Unsupported Havok value type {base.name}')

    def read(self, obj):
        typ, at = obj
        base = self.tag.types[self.tag.base(typ)]
        kind = base.subtype & 255
        if kind in (2, 3, 4, 5): return self.tag.value(obj)
        if kind == 7:
            key = typ, at
            if key not in self.cache:
                node = Node(typ, {})
                self.cache[key] = node
                node.value.update({n: self.read((t, at + off)) for n, off, t, flags in self.tag.members(typ)})
            return self.cache[key]
        if kind in (6, 8):
            values = [self.read(o) for o in self.tag.array(obj)]
            if kind == 6: return values[0] if values else None
            return values
        if kind == 40:
            size = self.tag.types[self.tag.base(base.pointer)].size
            return [self.read((base.pointer, at + i * size)) for i in range(base.subtype >> 8)]
        raise ValueError(f'Unsupported Havok value type {base.name}')


def section(name, data, leaf=True):
    data += b'\0' * (-len(data) % 4)
    return struct.pack('>I', (0x40000000 if leaf else 0) | (8 + len(data))) + name + data


def array_item_flags(tag, repair=False):
    """Audit native array allocation markers independently of Graph.read.

    The reflection reader follows ITEM indexes regardless of allocation flags;
    consequently a Python roundtrip cannot catch the old pointer-array bug.
    Only fix ITEM markers for reflected, nonempty arrays/strings. DATA, TYPE,
    pointers, PTCH, root objects, and all game-specific bytes remain untouched.
    """
    index = next(s for s in tag.children(tag.root) if s[0] == b'INDX')
    items = next(s for s in tag.children(index) if s[0] == b'ITEM')
    arrays = set()
    objects = set()

    def visit(typ, at):
        base = tag.types[tag.base(typ)]
        kind = base.subtype & 255
        if kind in (3, 6, 8):
            target = struct.unpack_from('<I', tag.raw, at)[0]
            if not target: return
            if target >= len(tag.items): raise ValueError('Invalid Havok allocation index')
            (objects if kind == 6 else arrays).add(target)
        elif kind == 7:
            for _, offset, field_type, _ in tag.members(typ): visit(field_type, at + offset)
        elif kind == 40:
            size = tag.types[tag.base(base.pointer)].size
            for i in range(base.subtype >> 8): visit(base.pointer, at + i * size)

    for item in tag.items[1:]:
        size = tag.types[tag.base(item.type)].size
        for i in range(item.count): visit(item.type, item.offset + i * size)
    if arrays & objects: raise ValueError('Havok allocation is both array and object')
    raw = bytearray(tag.raw)
    changes = []
    for target in sorted(arrays):
        at = items[1] + 12 * target
        flags = struct.unpack_from('<I', raw, at)[0]
        if flags & 0xF0000000 == 0x20000000: continue
        if flags & 0xF0000000 != 0x10000000:
            raise ValueError('Unsupported native Havok allocation flags')
        if not repair: raise ValueError('Havok array incorrectly marked as an object')
        struct.pack_into('<I', raw, at, (flags & 0x0FFFFFFF) | 0x20000000)
        changes.append(at)
    return bytes(raw), changes


class Writer:
    def __init__(self, graph):
        self.graph, self.tag = graph, graph.tag
        self.data = bytearray()
        self.items = [None]
        self.identity = {}
        self.patches = {}

    def allocate(self, typ, values, pointer=False, identity=None):
        if not values: return 0
        if identity is not None and identity in self.identity: return self.identity[identity]
        index = len(self.items)
        self.items.append([typ, values, pointer, 0])
        if identity is not None: self.identity[identity] = index
        return index

    def write(self, typ, value, at):
        base = self.tag.types[self.tag.base(typ)]
        kind = base.subtype & 255
        if kind == 7 and isinstance(value, Node): value = value.value
        if kind in (2, 4):
            size = 1 if kind == 2 else next(n // 8 for flag, n in ((0x2000, 8), (0x4000, 16), (0x8000, 32), (0x10000, 64)) if base.subtype & flag)
            self.data[at:at + size] = int(value).to_bytes(size, 'little', signed=bool(base.subtype & 512))
        elif kind == 5: struct.pack_into('<f', self.data, at, value)
        elif kind in (3, 6, 8):
            if kind == 3:
                index = self.allocate(self.graph.index('char'), list(value.encode('utf8') + b'\0')) if value else 0
            elif kind == 6:
                if value is not None and not isinstance(value, Node): raise ValueError('Expected typed Havok pointer')
                index = self.allocate(value.type, [value], True, id(value)) if value else 0
            else:
                # ITEM's flag describes the allocation, not its element type.
                # hkArray<hkRefPtr<T>> is still an array (0x20000000), never
                # an object allocation (0x10000000). Native Havok otherwise
                # leaves its runtime size zero despite valid ITEM references.
                index = self.allocate(base.pointer, value, False)
            struct.pack_into('<I', self.data, at, index)
            if index: self.patches.setdefault(self.tag.base(typ), []).append(at)
        elif kind == 7:
            for name, offset, field_type, flags in self.tag.members(typ): self.write(field_type, value[name], at + offset)
        elif kind == 40:
            if len(value) != base.subtype >> 8: raise ValueError('Havok tuple length mismatch')
            size = self.tag.types[self.tag.base(base.pointer)].size
            for i, member in enumerate(value): self.write(base.pointer, member, at + size * i)
        else: raise ValueError(f'Unsupported Havok encoding kind {kind}')

    def pack(self, root):
        self.allocate(root.type, [root], True, id(root))
        i = 1
        while i < len(self.items):
            typ, values, pointer, _ = self.items[i]
            base = self.tag.types[self.tag.base(typ)]
            alignment = max(16, base.alignment)
            self.data += b'\0' * (-len(self.data) % alignment)
            at = len(self.data); self.items[i][3] = at
            self.data += bytes(base.size * len(values))
            for j, value in enumerate(values): self.write(typ, value, at + j * base.size)
            i += 1
        self.data += b'\0' * (-len(self.data) % 16)
        items = bytes(12) + b''.join(struct.pack('<III', typ | (0x10000000 if pointer else 0x20000000), at, len(values))
                                     for typ, values, pointer, at in self.items[1:])
        patches = b''.join(struct.pack('<II', typ, len(offsets)) + struct.pack('<' + str(len(offsets)) + 'I', *offsets)
                           for typ, offsets in sorted(self.patches.items()))
        type_section = next(s for s in self.tag.children(self.tag.root) if s[0] == b'TYPE')
        native_types = bytes(self.tag.raw[type_section[1] - 8:type_section[2]])
        return section(b'TAG0', section(b'SDKV', b'20160200') + section(b'DATA', bytes(self.data)) + native_types +
                       section(b'INDX', section(b'ITEM', items) + section(b'PTCH', patches), False), False)
