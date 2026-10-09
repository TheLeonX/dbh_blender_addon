"""Independent PC ANIMDATA v13/v14 reader and native sample representation.

Rotations are XYZW quaternions. Position values retain their bind-position
coefficient; relative rotations retain track flag 0x40. No external converter
or Blender module is required by this file.
"""
from dataclasses import dataclass
import math
import struct


class Reader:
    def __init__(self, data, at=0):
        self.data, self.at = data, at

    def take(self, count):
        if count < 0 or self.at + count > len(self.data):
            raise ValueError(f'Truncated ANIM_DATA at 0x{self.at:X}')
        result = self.data[self.at:self.at + count]
        self.at += count
        return result

    def get(self, fmt):
        result = struct.unpack('<' + fmt, self.take(struct.calcsize('<' + fmt)))
        return result[0] if len(result) == 1 else result


@dataclass(frozen=True)
class Pose:
    rotation: tuple
    translation: tuple
    scale: tuple


@dataclass(frozen=True)
class Track:
    bone_hash: int
    order: int
    vector_index: int
    flags: int
    translation_bind: float
    samples: tuple


@dataclass(frozen=True)
class ExtraTrack:
    property_hash: int
    order: int
    vector_index: int
    flags: int
    samples: tuple


@dataclass(frozen=True)
class Animation:
    raw: bytes
    version: int
    flags: int
    frame_count: int
    timestep: float
    tracks: tuple
    extra_tracks: tuple
    block_size: int
    vector_offset: int
    stream_offset: int
    stream_end: int
    header_offset: int
    track_table_offset: int
    track_flags_offset: int
    vector_count: int
    pre_tracks: bytes
    between_tracks_flags: bytes
    stream_word: int


def quat_mul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z)


def quat_normalize(q):
    norm = math.sqrt(sum(v*v for v in q))
    if norm < 1e-10 or not math.isfinite(norm):
        raise ValueError('Invalid animation quaternion')
    return tuple(v/norm for v in q)


def absolute_pose(track, frame, bind_rotation, bind_position, bind_scale=(1., 1., 1.)):
    sample = track.samples[frame]
    rotation = quat_mul(bind_rotation, sample.rotation) if track.flags & 0x40 else sample.rotation
    translation = tuple(a + track.translation_bind*b for a, b in zip(sample.translation, bind_position))
    return Pose(quat_normalize(rotation), translation, sample.scale)


def parse_animation(data):
    data = bytes(data)
    if len(data) < 80 or data[:8] != b'COM_CONT':
        raise ValueError('Expected native COM_CONT / ANIMDATA')
    owner_size = struct.unpack_from('<I', data, 12)[0]
    tag = 32 + owner_size
    if owner_size > 4096 or data[tag:tag + 8] != b'ANIMDATA':
        raise ValueError('ANIMDATA chunk was not found')
    r = Reader(data, tag + 8)
    version, payload_size = r.get('II')
    if version not in (13, 14) or payload_size != len(data) - r.at:
        raise ValueError('Unsupported ANIMDATA version or chunk length')
    if r.get('I') != 2:
        raise ValueError('Unsupported ANIMDATA payload type')
    if version == 14:
        r.get('II')
    header_offset = r.at
    flags, frames, count, timestep = r.get('IIIf')
    if flags & 0x242 != 0x242:
        raise ValueError('Unsupported ANIMDATA track-table representation')
    if not 1 <= frames <= 100000 or not 1 <= count <= 4096 or frames*count > 8000000:
        raise ValueError('Unreasonable animation frame/track count')
    if not math.isfinite(timestep) or not 1/1000 <= timestep <= 1:
        raise ValueError('Invalid animation sample interval')
    extra_count = r.get('I') if flags & 0x400 else 0
    if extra_count > 4096 or frames*(count+extra_count)>8000000:
        raise ValueError('Invalid animation property-track count')
    if flags & 0x4000 or version == 13:
        r.take(r.get('I') * 12)
    table_offset = r.at
    rows = [r.get('IHH') for _ in range(count)]
    extra_rows = [r.get('IHH') for _ in range(extra_count)]
    if len({row[0] for row in rows}) != count:
        raise ValueError('Duplicate animation bone hash')
    between_start = r.at
    if flags & 0x100:
        r.take(r.get('I') * 12)
    if flags & 0x9000:
        r.take(12)
        r.take(r.get('I'))
    if flags & 0x2000:
        r.take(5)
    flags_offset = r.at
    track_flags = list(r.take(count))
    extra_flags = list(r.take(extra_count))
    if any(f & 0x80 for f in track_flags):
        raise ValueError('Unsupported animation bone-track flags')
    if any(f not in (16, 27) for f in extra_flags):
        raise ValueError('Unsupported animation property-track flags')
    vector_count = r.get('I')
    if not 1 <= vector_count <= 65535:
        raise ValueError('Invalid animation vector-table count')
    vector_offset = r.at
    vector_data = r.take(vector_count * 16)
    block_size, block_tag = r.get('IB')
    if not 1 <= block_size <= 4096 or block_tag != 0:
        raise ValueError('Unsupported animation block mode')
    bounds_low = r.get('256f') if block_size != 1 else None
    bounds_high = r.get('256f') if block_size != 1 else None
    stream_word = r.get('I')
    if stream_word != 0:
        raise ValueError('Unsupported animation stream tag')
    stream_offset = r.at
    constants = []
    weights = []
    for (_, _, index), f in zip(rows, track_flags):
        v = Reader(vector_data, index * 16)
        rotation, position, scale, weight = (0., 0., 0., 1.), (0., 0., 0.), (1., 1., 1.), 1.
        if f & 16:
            if not f & 2:
                rotation = v.get('4f')
            if not f & 1:
                p = v.get('4f')
                position, weight = p[:3], p[3]
            if flags & 0x800 and not f & 8:
                scale = v.get('4f')[:3]
        constants.append(Pose(rotation, position, scale))
        weights.append(weight)
    extra_constants = []
    for row, f in zip(extra_rows, extra_flags):
        v = Reader(vector_data, row[2] * 16)
        extra_constants.append(v.get('f') if f == 16 else 0.)
    samples = [[] for _ in rows]
    extra_samples = [[] for _ in extra_rows]
    left = frames
    while left:
        block_frames = min(left, block_size)
        coefficients = []
        extra_coefficients = []
        if block_size != 1:
            for f in track_flags:
                n = (1 if f & 2 else 0) + (3 if f & 1 else 0) + (3 if f & 8 else 0)
                coefficients.append([(bounds_low[i], bounds_high[i]) for i in r.take(n)])
            for f in extra_flags:
                i = r.get('B') if f == 27 else 0
                extra_coefficients.append((bounds_low[i], bounds_high[i]))
            r.take((-(r.at - stream_offset)) % 4)
        anchors = []
        for j, f in enumerate(track_flags):
            base = constants[j]
            q = tuple((v - 32767.) / 32767. for v in r.get('4H')) if f & 2 else base.rotation
            t = r.get('3f') if f & 1 else base.translation
            s = r.get('3f') if f & 8 else base.scale
            pose = Pose(q, t, s)
            anchors.append(pose)
            samples[j].append(pose)
        extra_anchors = []
        for j, f in enumerate(extra_flags):
            value = r.get('f') if f == 27 else extra_constants[j]
            extra_anchors.append(value)
            extra_samples[j].append(value)
        for _ in range(1, block_frames):
            for j, f in enumerate(track_flags):
                anchor = anchors[j]
                coefficients_iter = iter(coefficients[j])
                def delta(n, per_component):
                    pairs = [next(coefficients_iter) for _ in range(n if per_component else 1)]
                    if not per_component:
                        pairs *= n
                    return tuple(lo + v * ((hi - lo)/255.) for v, (lo, hi) in zip(r.take(n), pairs))
                q = tuple(a+b for a,b in zip(anchor.rotation, delta(4, False))) if f & 2 else anchor.rotation
                t = tuple(a+b for a,b in zip(anchor.translation, delta(3, True))) if f & 1 else anchor.translation
                s = tuple(a+b for a,b in zip(anchor.scale, delta(3, True))) if f & 8 else anchor.scale
                samples[j].append(Pose(q, t, s))
            for j, f in enumerate(extra_flags):
                value = extra_anchors[j]
                if f == 27:
                    lo, hi = extra_coefficients[j]
                    value += lo + r.get('B') * ((hi-lo)/255.)
                extra_samples[j].append(value)
        left -= block_frames
    for poses in samples:
        if any(not all(math.isfinite(v) for v in (*p.rotation, *p.translation, *p.scale)) for p in poses):
            raise ValueError('Non-finite animation sample')
    if any(not math.isfinite(v) for values in extra_samples for v in values):
        raise ValueError('Non-finite animation property sample')
    padded_stream = (r.at - stream_offset + 127) & ~127
    if stream_offset + padded_stream != len(data):
        raise ValueError('ANIMDATA frame stream size differs from native 128-byte padding')
    return Animation(data, version, flags, frames, timestep,
                     tuple(Track(*row, f, weight, tuple(poses)) for row, f, weight, poses in zip(rows, track_flags, weights, samples)),
                     tuple(ExtraTrack(*row, f, tuple(values)) for row, f, values in zip(extra_rows, extra_flags, extra_samples)),
                     block_size, vector_offset, stream_offset, r.at, header_offset,
                     table_offset, flags_offset, vector_count,
                     data[:table_offset], data[between_start:flags_offset], stream_word)


def encode_animation(source, tracks=None, extra_tracks=None, frame_count=None, timestep=None):
    """Write native keyframe blocks of one frame, with the game's offset table.

    The original COM_CONT ownership, events, attachments and property tracks
    are retained. Bone samples may be replaced or additional hash tracks added.
    """
    tracks = tuple(source.tracks if tracks is None else tracks)
    extra_tracks = tuple(source.extra_tracks if extra_tracks is None else extra_tracks)
    frame_count = source.frame_count if frame_count is None else frame_count
    timestep = source.timestep if timestep is None else timestep
    if not 1 <= frame_count <= 100000 or not tracks or len(tracks) > 4096:
        raise ValueError('Invalid export frame or bone-track count')
    if frame_count*len(tracks) > 8000000:
        raise ValueError('Export animation is too large')
    if len(extra_tracks) != len(source.extra_tracks):
        raise ValueError('Changing the native property-track set is not supported')
    if len({t.bone_hash for t in tracks}) != len(tracks):
        raise ValueError('Duplicate exported animation bone hash')
    if frame_count != source.frame_count and source.flags & 0x9100:
        raise ValueError('This animation has linked/attached animation timing; retain its original duration')
    vectors = bytearray()
    rows, extra_rows = [], []
    anchor_stride = delta_stride = coefficient_stride = 0
    for track in tracks:
        f = track.flags
        if not f & 16 or f & 0x80 or len(track.samples) != frame_count:
            raise ValueError('Unsupported exported bone-track layout')
        if f & 0xb:
            vectors += struct.pack('<4I', anchor_stride, delta_stride, coefficient_stride, 0)
        index = len(vectors)//16
        rows.append((track.bone_hash, track.order, index))
        anchor_stride += (8 if f&2 else 0) + (12 if f&1 else 0) + (12 if f&8 else 0)
        delta_stride += (4 if f&2 else 0) + (3 if f&1 else 0) + (3 if f&8 else 0)
        coefficient_stride += (1 if f&2 else 0) + (3 if f&1 else 0) + (3 if f&8 else 0)
        first = track.samples[0]
        for bit, attr in ((2,'rotation'), (1,'translation'), (8,'scale')):
            if not f & bit and any(getattr(p,attr) != getattr(first,attr) for p in track.samples):
                raise ValueError(f'Changing samples in constant {attr} channel')
        if not f & 2:
            vectors += struct.pack('<4f', *first.rotation)
        if not f & 1:
            vectors += struct.pack('<4f', *first.translation, track.translation_bind)
        elif abs(track.translation_bind-1.) > 1e-7:
            raise ValueError('Animated positions must be expressed relative to the bind position')
        if source.flags & 0x800 and not f & 8:
            vectors += struct.pack('<4f', *first.scale, 0.0001)
    for track in extra_tracks:
        if len(track.samples) != frame_count or track.flags not in (16,27):
            raise ValueError('Invalid exported animation property samples')
        if any(not math.isfinite(v) for v in track.samples):
            raise ValueError('Non-finite exported animation property sample')
        if track.flags == 27:
            vectors += struct.pack('<4I', anchor_stride, delta_stride, coefficient_stride, 0)
            anchor_stride += 4
            delta_stride += 1
            coefficient_stride += 1
        extra_rows.append((track.property_hash, track.order, len(vectors)//16))
        if track.flags == 16:
            if any(v != track.samples[0] for v in track.samples):
                raise ValueError('Changing samples in a constant property track')
            vectors += struct.pack('<4f', track.samples[0], 0., 0., 0.)
    if len(vectors)//16 > 65535:
        raise ValueError('Animation vector table exceeds native 16-bit addressing')
    prefix = bytearray(source.pre_tracks)
    struct.pack_into('<IIIf', prefix, source.header_offset, source.flags, frame_count, len(tracks), timestep)
    if frame_count != source.frame_count and (source.flags & 0x4000 or source.version==13):
        at = source.header_offset+16+(4 if source.flags&0x400 else 0)
        events = struct.unpack_from('<I', prefix, at)[0]
        for i in range(events):
            at_event = at+4+12*i
            start,end = struct.unpack_from('<HH', prefix, at_event)
            ratio = (frame_count-1)/max(1,source.frame_count-1)
            struct.pack_into('<HH', prefix, at_event, min(65535,round(start*ratio)), min(65535,round(end*ratio)))
    output = prefix
    for row in rows+extra_rows:
        output += struct.pack('<IHH', *row)
    output += source.between_tracks_flags
    output += bytes(t.flags for t in tracks) + bytes(t.flags for t in extra_tracks)
    output += struct.pack('<I', len(vectors)//16) + vectors
    output += struct.pack('<IBI', 1, 0, 0)
    stream_offset = len(output)
    for frame in range(frame_count):
        for track in tracks:
            sample = track.samples[frame]
            if not all(math.isfinite(v) for v in (*sample.rotation,*sample.translation,*sample.scale)):
                raise ValueError('Non-finite exported bone sample')
            if track.flags & 2:
                q = quat_normalize(sample.rotation)
                packed = tuple(max(0,min(65534,round((v+1.)*32767.))) for v in q)
                output += struct.pack('<4H', *packed)
            if track.flags & 1:
                output += struct.pack('<3f', *sample.translation)
            if track.flags & 8:
                output += struct.pack('<3f', *sample.scale)
        for track in extra_tracks:
            if track.flags == 27:
                value = track.samples[frame]
                if not math.isfinite(value):
                    raise ValueError('Non-finite exported property sample')
                output += struct.pack('<f', value)
    expected_size = frame_count*anchor_stride
    if len(output)-stream_offset != expected_size:
        raise ValueError('Native animation frame-stride mismatch')
    output += bytes((-expected_size)%128)
    chunk_size_at = 44 + struct.unpack_from('<I',output,12)[0]
    struct.pack_into('<I',output,chunk_size_at,len(output)-chunk_size_at-4)
    result = bytes(output)
    checked = parse_animation(result)
    if checked.frame_count != frame_count or len(checked.tracks) != len(tracks):
        raise ValueError('ANIMDATA export verification failed')
    return result
