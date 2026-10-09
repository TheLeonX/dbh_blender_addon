"""Conservative NODEAR v4 rest translations and SKELETON v15 inverse binds.

No new joints, hierarchy changes, animation tracks, or native rotations are
authored. Blender's imported bone axes are display axes, not native joint axes.
All untouched bytes (including Magik IK data) remain unchanged.
"""
import math
import struct
from dataclasses import dataclass


def mv(matrix, vector):
    return tuple(sum(a*b for a,b in zip(row,vector)) for row in matrix)


def mm(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def add(a, b): return tuple(x+y for x,y in zip(a,b))
def sub(a, b): return tuple(x-y for x,y in zip(a,b))
def norm(a): return math.sqrt(sum(x*x for x in a))


IDENTITY = ((1.,0.,0.), (0.,1.,0.), (0.,0.,1.))


def inverse(m):
    a,b,c=m[0];d,e,f=m[1];g,h,i=m[2]
    det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g)
    if not math.isfinite(det) or abs(det)<1e-10:
        raise ValueError('Singular native skeleton transform')
    return tuple(tuple(x/det for x in row) for row in
                 ((e*i-f*h,c*h-b*i,b*f-c*e), (f*g-d*i,a*i-c*g,c*d-a*f), (d*h-e*g,b*g-a*h,a*e-b*d)))


def basis(q, scale):
    length=norm(q)
    if not 0.99<length<1.01: raise ValueError('Invalid native bone quaternion')
    w,x,y,z=(v/length for v in q)
    r=((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
       (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
       (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    return tuple(tuple(row[j]*scale[j] for j in range(3)) for row in r)


@dataclass
class Joint:
    offset: int
    parent: int
    position: tuple
    linear: tuple
    world_position: tuple
    world_linear: tuple
    children: tuple


class Rig:
    def __init__(self, record):
        self.record=record
        raw=record.payload
        if record.kind!=0x85a or raw[4:12]!=b'NODE    ' or struct.unpack_from('<I',raw,12)[0]!=5:
            raise ValueError('Unsupported native skeleton NODE')
        if raw[196:204]!=b'NODEAR  ' or struct.unpack_from('<I',raw,204)[0]!=4:
            raise ValueError('Unsupported NODEAR skeleton version')
        if struct.unpack_from('<I',raw,208)[0]!=14:
            raise ValueError('Unsupported inner skeleton version (expected 14)')
        count=struct.unpack_from('<I',raw,212)[0]
        if not 0<count<=4096: raise ValueError('Invalid native bone count')
        at=220
        self.joints=[]
        for i in range(count):
            if at+60>len(raw): raise ValueError('Truncated native bone')
            q=struct.unpack_from('<4f',raw,at+4)
            pos=struct.unpack_from('<3f',raw,at+20)
            scale=struct.unpack_from('<3f',raw,at+32)
            index,parent,n=struct.unpack_from('<IiI',raw,at+44)
            if index!=i or parent < -1 or parent>=i or n>count or at+60+4*n>len(raw):
                raise ValueError('Invalid native bone hierarchy')
            children=struct.unpack_from('<'+str(n)+'I',raw,at+56)
            if any(c<=i or c>=count for c in children) or len(set(children))!=n:
                raise ValueError('Invalid native bone child table')
            if not all(math.isfinite(v) for v in (*q,*pos,*scale)):
                raise ValueError('Non-finite native bone transform')
            local=basis(q,scale)
            parent_linear=self.joints[parent].world_linear if parent>=0 else IDENTITY
            parent_pos=self.joints[parent].world_position if parent>=0 else (0.,0.,0.)
            self.joints.append(Joint(at,parent,pos,local,add(mv(parent_linear,pos),parent_pos),
                                     mm(parent_linear,local),children))
            at+=60+4*n
        for i,j in enumerate(self.joints):
            if j.parent>=0 and i not in self.joints[j.parent].children:
                raise ValueError('Native parent/child mismatch')
            if any(self.joints[c].parent!=i for c in j.children):
                raise ValueError('Native child/parent mismatch')
        # Per-joint name/property table follows the variable-length transforms.
        if at+count*12>len(raw): raise ValueError('Truncated native bone name table')
        self.hashes=struct.unpack_from('<'+str(count*3)+'I',raw,at)[::3]
        self.ik_map=()
        tag=raw.find(b'MAGKFMAP',at+count*12)
        if tag>=0:
            size=struct.unpack_from('<I',raw,tag+8)[0]
            if size!=1024 or tag+16+size>len(raw): raise ValueError('Unsupported IK mapping')
            self.ik_map=struct.unpack_from('<256i',raw,tag+16)

    def matches(self, bones):
        return (len(bones)==len(self.joints)+1 and bones[0]['parent']==-1 and
                all(b['parent']==j.parent+1 and norm(sub(b['position'],j.world_position))<5e-5
                    for b,j in zip(bones[1:],self.joints)))


def decode_inverse(raw, offset):
    """Native serializer stores a 4x4 matrix in reversed column-major order."""
    values=struct.unpack_from('<16f',raw,offset)
    if not all(math.isfinite(v) for v in values): raise ValueError('Non-finite inverse bind')
    matrix=tuple(tuple(values[15-(c*4+r)] for c in range(4)) for r in range(4))
    if any(abs(x-y)>1e-5 for x,y in zip(matrix[3],(0,0,0,1))):
        raise ValueError('Unsupported non-affine inverse bind')
    return matrix


def plan_positions(package, bones, positions, y_offset=0.):
    """Return inline record replacements and audit; never mutate on failure.

    Positions are armature-space rest heads in game XYZ, synthetic Root first.
    Use deltas from imported positions to retain reference-exporter precision.
    """
    if len(positions)!=len(bones) or not all(len(p)==3 and all(math.isfinite(v) for v in p) for p in positions):
        raise ValueError('Invalid exported bone positions')
    deltas=[sub(p,b['position']) for p,b in zip(positions,bones)]
    deltas=[d if norm(d)>1e-6 else (0.,0.,0.) for d in deltas]
    if not any(norm(d) for d in deltas):return {},[]
    if norm(deltas[0]):raise ValueError('The synthetic Root cannot be moved; move native bones in Edit Mode')
    rigs=[]
    for r in package.container.records:
        if r.kind!=0x85a:continue
        rig=Rig(r)
        if rig.matches(bones):rigs.append(rig)
    if len(rigs)!=1:raise ValueError('Cannot uniquely match edited armature to a native skeleton; reimport this package')
    rig=rigs[0];deltas=deltas[1:]
    raw=bytearray(rig.record.payload)
    report=[]
    for i,(joint,delta) in enumerate(zip(rig.joints,deltas)):
        parent_delta=deltas[joint.parent] if joint.parent>=0 else (0.,0.,0.)
        parent_basis=rig.joints[joint.parent].world_linear if joint.parent>=0 else IDENTITY
        local_delta=mv(inverse(parent_basis),sub(delta,parent_delta))
        if norm(local_delta)>1e-8:
            struct.pack_into('<3f',raw,joint.offset+20,*add(joint.position,local_delta))
        if norm(delta):
            report.append(dict(name=bones[i+1]['name'],bone=i+1,native_bone=i,
                               rig_id=f'0x{rig.record.asset:X}',position=list(positions[i+1]),delta=list(delta),
                               ik_controlled=i<len(rig.ik_map) and rig.ik_map[i]>=0))
    replacements={rig.record.index:bytes(raw)}
    entities=[]
    for rec in package.container.records:
        if rec.kind!=0x7dd or len(rec.payload)<52 or struct.unpack_from('<II',rec.payload,44)!=(0x85a,rig.record.asset):continue
        raw=bytearray(rec.payload)
        start=raw.find(b'SKELETON')
        if start<0 or raw.find(b'SKELETON',start+8)>=0:raise ValueError('Missing or ambiguous SKELETON inverse-bind block')
        if struct.unpack_from('<I',raw,start+8)[0]!=15:raise ValueError('Unsupported SKELETON version')
        count=struct.unpack_from('<I',raw,start+16)[0]
        if count!=len(rig.joints) or start+20+64*count>len(raw):raise ValueError('Native inverse-bind count mismatch')
        for i,(joint,delta) in enumerate(zip(rig.joints,deltas)):
            offset=start+20+64*i
            matrix=decode_inverse(raw,offset)
            linear=tuple(row[:3] for row in matrix[:3])
            # Validate both the layout and the rig association before any write.
            product=mm(linear,joint.world_linear)
            # The importer adds y_offset to native vertices to align them with
            # the reference skeleton; inverse binds use the native mesh space.
            bind_position=add(joint.world_position,(0.,-y_offset,0.))
            error=add(mv(linear,bind_position),tuple(row[3] for row in matrix[:3]))
            if max(abs(product[r][c]-IDENTITY[r][c]) for r in range(3) for c in range(3))>1e-4 or norm(error)>1e-4:
                raise ValueError(f'Unrecognized inverse bind for {bones[i+1]["name"]}; no package written')
            if norm(delta):
                translation=sub(tuple(row[3] for row in matrix[:3]),mv(linear,delta))
                for row,value in enumerate(translation):struct.pack_into('<f',raw,offset+4*(3-row),value)
        replacements[rec.index]=bytes(raw)
        entities.append(rec.asset)
    if not entities:raise ValueError('Edited skeleton has no verified SKELETON inverse-bind block')
    return replacements,report
