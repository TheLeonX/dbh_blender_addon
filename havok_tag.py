"""Bounded reflection reader for the native 2016 binary Havok tagfiles.

Offsets are obtained from each file's TYPE/ITEM tables. Writes retain every
section and relocation entry; callers edit only validated scalar fields.
"""
from dataclasses import dataclass, field
import struct


@dataclass
class Type:
    name: str = ''
    parent: int = 0
    flags: int = 0
    subtype: int = 0
    pointer: int = 0
    size: int = 0
    alignment: int = 0
    members: list = field(default_factory=list)


@dataclass(frozen=True)
class Item:
    type: int
    offset: int
    count: int


class Cursor:
    def __init__(self, raw): self.raw, self.pos = raw, 0
    def packed(self):
        if self.pos >= len(self.raw): raise ValueError('Truncated Havok type table')
        first = self.raw[self.pos]; self.pos += 1
        length = 1 if first < 128 else 2 if first < 192 else 3 if first < 224 else 4
        mask = (0x7f, 0x3f, 0x1f, 0x07)[length - 1]
        if self.pos + length - 1 > len(self.raw): raise ValueError('Truncated packed integer')
        value = first & mask
        for _ in range(length - 1): value = value * 256 + self.raw[self.pos]; self.pos += 1
        return value


class TagFile:
    def __init__(self, raw, offset=0):
        self.raw = bytearray(raw)
        self.start = offset
        self.root = self.section(offset, len(raw))
        if self.root[0] != b'TAG0': raise ValueError('Expected Havok TAG0')
        sections = {s[0]: s for s in self.children(self.root)}
        if self.bytes(sections[b'SDKV']) != b'20160200':
            raise ValueError('Cloth requires Havok 20160200')
        self.data_start, self.data_end = sections[b'DATA'][1:]
        if b'TYPE' not in sections: raise ValueError('External Havok type compendium unsupported')
        subs = {s[0]: s for s in self.children(sections[b'TYPE'])}
        names = self.bytes(subs[b'TSTR']).decode('utf8').split('\0')
        fields = self.bytes(subs[b'FSTR']).decode('utf8').split('\0')
        cur = Cursor(self.bytes(subs.get(b'TNAM', subs.get(b'TNA1'))))
        count = cur.packed()
        if not 1 < count < 65536: raise ValueError('Invalid Havok type count')
        self.types = [Type() for _ in range(count)]
        for typ in self.types[1:]:
            typ.name = names[cur.packed()]
            for _ in range(cur.packed()): cur.packed(); cur.packed()
        cur = Cursor(self.bytes(subs.get(b'TBOD', subs.get(b'TBDY'))))
        while cur.pos < len(cur.raw):
            index = cur.packed()
            if index == 0: continue
            typ = self.types[index]
            typ.parent, typ.flags = cur.packed(), cur.packed()
            if typ.flags & 1: typ.subtype = cur.packed()
            if typ.flags & 2: typ.pointer = cur.packed()
            if typ.flags & 4: cur.packed()
            if typ.flags & 8: typ.size, typ.alignment = cur.packed(), cur.packed()
            if typ.flags & 16: cur.packed()
            if typ.flags & 32:
                for _ in range(cur.packed()):
                    name = fields[cur.packed()]; flags = cur.packed()
                    typ.members.append((name, cur.packed(), cur.packed(), flags))
            if typ.flags & 64:
                for _ in range(cur.packed()): cur.packed(); cur.packed()
            if typ.flags & 128: raise ValueError('Unsupported Havok type flags')
        index = {s[0]: s for s in self.children(sections[b'INDX'])}
        raw_items = self.bytes(index[b'ITEM'])
        if len(raw_items) % 12: raise ValueError('Invalid Havok ITEM span')
        self.items = [Item(t & 0xffffff, self.data_start + at, n)
                      for t, at, n in struct.iter_unpack('<III', raw_items)]
        # TYPE is immutable after parsing; DATA scalar edits do not invalidate
        # these schema caches. Keep bounds/inheritance checks on cache misses.
        self._bases = {}; self._members = {}; self._fields = {}
        for item in self.items[1:]:
            size = self.types[self.base(item.type)].size
            if not size or item.offset < self.data_start or item.offset + size * item.count > self.data_end:
                raise ValueError('Havok item extends outside DATA')

    def section(self, offset, end):
        if offset < 0 or offset + 8 > end: raise ValueError('Truncated Havok section')
        length = struct.unpack_from('>I', self.raw, offset)[0] & 0x3fffffff
        if length < 8 or offset + length > end: raise ValueError('Invalid Havok section span')
        return bytes(self.raw[offset + 4:offset + 8]), offset + 8, offset + length

    def children(self, section):
        pos = section[1]
        while pos < section[2]:
            child = self.section(pos, section[2]); yield child; pos = child[2]

    def bytes(self, section): return bytes(self.raw[section[1]:section[2]])

    def base(self, index):
        if index in self._bases: return self._bases[index]
        original = index
        visited = set()
        while not self.types[index].flags & 1:
            if index in visited or index == 0: raise ValueError('Invalid Havok type inheritance')
            visited.add(index); index = self.types[index].parent
        for typ in visited: self._bases[typ] = index
        self._bases[original] = index
        return index

    def members(self, index):
        if index in self._members: return self._members[index]
        typ = self.types[index]
        result = (self.members(typ.parent) if typ.parent else []) + typ.members
        self._members[index] = result
        return result

    def objects(self, name):
        for item in self.items[1:]:
            if self.types[item.type].name == name:
                size = self.types[self.base(item.type)].size
                for n in range(item.count): yield item.type, item.offset + n * size

    def field(self, obj, name):
        key = obj[0], name
        if key not in self._fields:
            self._fields[key] = [(typ, at) for field, at, typ, flags in self.members(obj[0]) if field == name]
        matches = self._fields[key]
        if len(matches) != 1: raise ValueError(f'Havok {self.types[obj[0]].name}.{name} is missing/ambiguous')
        typ, at = matches[0]
        return typ, obj[1] + at

    def array(self, obj):
        typ, at = obj
        base = self.types[self.base(typ)]
        if base.subtype & 255 not in (3, 6, 8): raise ValueError('Expected Havok pointer/array')
        index = struct.unpack_from('<I', self.raw, at)[0]
        if not index: return []
        if index >= len(self.items): raise ValueError('Invalid Havok item reference')
        item = self.items[index]
        size = self.types[self.base(item.type)].size
        return [(item.type, item.offset + i * size) for i in range(item.count)]

    def value(self, obj):
        typ, at = obj
        base = self.types[self.base(typ)]
        kind = base.subtype & 255
        if kind in (2, 4):
            bits = next((n for flag, n in ((0x2000, 8), (0x4000, 16), (0x8000, 32), (0x10000, 64)) if base.subtype & flag), None)
            if not bits: raise ValueError('Unsupported Havok integer')
            return int.from_bytes(self.raw[at:at + bits // 8], 'little', signed=bool(base.subtype & 512))
        if kind == 5: return struct.unpack_from('<f', self.raw, at)[0]
        if kind == 3: return bytes(self.value(o) for o in self.array(obj)).rstrip(b'\0').decode('utf8')
        if kind in (6, 8): return [self.value(o) for o in self.array(obj)]
        if kind == 7: return {name: self.value((t, at + off)) for name, off, t, flags in self.members(typ)}
        if kind == 40:
            size = self.types[self.base(base.pointer)].size
            return [self.value((base.pointer, at + i * size)) for i in range(base.subtype >> 8)]
        return None

    def write_float(self, obj, value):
        if self.types[self.base(obj[0])].subtype & 255 != 5: raise ValueError('Expected Havok float field')
        struct.pack_into('<f', self.raw, obj[1], value)

