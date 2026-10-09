"""Native CLOTHDAT v13 wind parameters, not Blender force fields.

Verified reader RVA 0x61F670: the 66-byte trailer maps to CClothSettings.
Copy RVA 0x61F1C0 supplies CCloth's settings at +336. Update RVA 0x6255E0
uses +388 for world wind, +356 for movement wind, and +360 for deadzone;
0x626DB0 registers the embedded native hclSimpleWindAction on the instance.
No synthetic Havok type, runtime hook, or gravity-as-wind substitution.
"""
import math
import struct
from .cloth_native import resource_tag

# serialized offset -> settings offset -> live CCloth offset
FIELDS = {'global_scale': 25,  # +52 -> +388
          'local_scale': 8,   # +20 -> +356
          'local_deadzone': 12,  # +24 -> +360, km/h
          'local_frequency': 37, # +28 -> +364
          'minimum_speed': 29,   # +56 -> +392
          'maximum_speed': 61}  # +60 -> +396


def read_wind(payload):
    tag = resource_tag(payload); at = tag.root[2]
    values = {name: struct.unpack_from('<f', payload, at+off)[0] for name, off in FIELDS.items()}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Non-finite native cloth wind parameters')
    return values


def write_wind(payload, overrides):
    if not overrides:return payload
    if set(overrides)-set(FIELDS):raise ValueError('Unknown native cloth wind parameter')
    tag=resource_tag(payload);at=tag.root[2];raw=bytearray(payload)
    for name,value in overrides.items():
        if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1000:
            raise ValueError('Cloth wind parameters must be finite and within 0..1000')
        struct.pack_into('<f',raw,at+FIELDS[name],value)
    result=bytes(raw);read_wind(result)
    return result
