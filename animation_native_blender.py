"""Blender integration for the independent native ANIM_DATA codec."""
import base64
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import tempfile

import bpy
from mathutils import Matrix, Quaternion, Vector

from .animation_assets import read_animation
from . import asset_metadata as action_data
from .animation_codec import (Pose, Track, absolute_pose, encode_animation,
                              parse_animation, quat_mul, quat_normalize)


AXIS = Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))


def _bindings(rig):
    result = [(0, -1, Pose((0.,0.,0.,1.),(0.,0.,0.),(1.,1.,1.)))]
    for bone_hash, joint in zip(rig.hashes, rig.joints):
        w,x,y,z=struct.unpack_from('<4f',rig.record.payload,joint.offset+4)
        scale=struct.unpack_from('<3f',rig.record.payload,joint.offset+32)
        result.append((bone_hash, joint.parent+1, Pose((x,y,z,w),joint.position,scale)))
    return result


def _matrix(pose):
    x,y,z,w=quat_normalize(pose.rotation)
    return Matrix.LocRotScale(Vector(pose.translation),Quaternion((w,x,y,z)),Vector(pose.scale))


def _pose(matrix):
    position,rotation,scale=matrix.decompose()
    return Pose((rotation.x,rotation.y,rotation.z,rotation.w),tuple(position),tuple(scale))


def _save_source(action, raw, label):
    digest=hashlib.sha256(raw).hexdigest()[:16]
    name=f'.DBH_ANIM_DATA_{digest}'
    text=bpy.data.texts.get(name)
    if text is None:
        text=bpy.data.texts.new(name)
        encoded=base64.b64encode(raw).decode('ascii')
        # Huge single lines are expensive for Blender's editable Text storage.
        text.write('\n'.join(encoded[i:i+120] for i in range(0,len(encoded),120)))
    text.use_fake_user=True
    action_data.update(action,{'dbh_native_source_text':name})
    match=re.search(r'0x([0-9a-f]+)',label,re.I)
    if match:
        action_data.update(action,{'dbh_anim_asset_id':f'0x{int(match.group(1),16):X}'})


def source_for_action(context, action):
    text=bpy.data.texts.get(action_data.get(action,'dbh_native_source_text',''))
    if text is not None:
        return parse_animation(base64.b64decode(''.join(text.as_string().split()),validate=True))
    asset=action_data.get(action,'dbh_anim_asset_id','')
    if not asset:
        match=re.search(r'DBH_0x([0-9a-f]+)',action.name,re.I)
        if match:
            asset='0x'+match.group(1)
    if not asset:
        raise ValueError('Import the original ANIM_DATA once, then edit or replace its Action keys; its native metadata is required')
    from .game_paths import resolve_index
    raw=read_animation(resolve_index(context, required=True),int(asset,0))
    _save_source(action,raw,asset)
    return parse_animation(raw)


def import_native_animation(context, animation, label):
    from .animation_blender import _target, _native_rig_for_animation, _native_bind_matrices
    armature,metadata=_target(context)
    source=parse_animation(animation)
    rig=_native_rig_for_animation(metadata)
    bindings=_bindings(rig)
    if len(bindings)!=len(metadata['bones']):
        raise ValueError('Native animation rig metadata differs from the selected rig')
    bones=[armature.pose.bones.get(item['name']) for item in metadata['bones']]
    if any(b is None for b in bones):
        raise ValueError('The rig is missing imported bones')
    if armature.animation_data is None:
        armature.animation_data_create()
    if armature.animation_data.action:
        armature.animation_data.action.use_fake_user=True
    action=bpy.data.actions.new(label)
    action.use_fake_user=True
    action_data.update(action,dict(dbh_animation_space='native_bind_relative_v2',
        dbh_armature_name=armature.name,dbh_native_frame_count=source.frame_count,
        dbh_native_timestep=source.timestep))
    _save_source(action,animation,label)
    armature.data.pose_position='POSE'
    for bone in bones:
        bone.rotation_mode='QUATERNION'
    by_hash={track.bone_hash:track for track in source.tracks}
    rest=[bone.bone.matrix_local.copy() for bone in bones]
    native_bind=_native_bind_matrices(rig)
    display=[native.inverted() @ current for native,current in zip(native_bind,rest)]
    relative_inverse=[(rest[parent].inverted() @ rest[i]).inverted() if parent>=0 else rest[i].inverted()
                      for i,(_,parent,_) in enumerate(bindings)]
    mapped={i for i,(h,_,_) in enumerate(bindings) if h in by_hash}
    coordinates=[[[] for _ in range(10)] for _ in bones]
    previous=[None]*len(bones)
    axis_inverse=AXIS.inverted()
    for frame in range(source.frame_count):
        world=[]
        for bone_hash,parent,bind in bindings:
            track=by_hash.get(bone_hash)
            local=_matrix(absolute_pose(track,frame,bind.rotation,bind.translation,bind.scale) if track else bind)
            world.append(world[parent] @ local if parent>=0 else local)
        target=[AXIS @ pose @ axis_inverse @ display[i] for i,pose in enumerate(world)]
        for i,bone in enumerate(bones):
            parent=bindings[i][1]
            matrix=relative_inverse[i] @ (target[parent].inverted() @ target[i] if parent>=0 else target[i])
            location,rotation,scale=matrix.decompose()
            if previous[i] is not None and previous[i].dot(rotation)<0:
                rotation.negate()
            previous[i]=rotation.copy()
            if i in mapped or frame in (0,source.frame_count-1):
                values=(*location,*rotation,*scale)
                for channel,value in zip(coordinates[i],values):
                    channel.extend((frame+1,value))
    for i,bone in enumerate(bones):
        channel=0
        for prop,count in (('location',3),('rotation_quaternion',4),('scale',3)):
            for index in range(count):
                curve=action.fcurves.new(bone.path_from_id(prop),index=index,action_group=bone.name)
                values=coordinates[i][channel]
                curve.keyframe_points.add(len(values)//2)
                curve.keyframe_points.foreach_set('co',values)
                linear=bpy.types.Keyframe.bl_rna.properties['interpolation'].enum_items['LINEAR'].value
                curve.keyframe_points.foreach_set('interpolation',[linear]*(len(values)//2))
                curve.update()
                channel+=1
    context.scene.render.fps=round(1/source.timestep)
    context.scene.render.fps_base=context.scene.render.fps*source.timestep
    context.scene.frame_end=max(context.scene.frame_end,source.frame_count)
    # Build off-rig to avoid dependency-graph updates for each inserted curve.
    armature.animation_data.action=action
    # Blender 4.4+ legacy fcurves create a slot which must be bound explicitly.
    if hasattr(action,'slots') and len(action.slots):
        armature.animation_data.action_slot=action.slots[0]
    action.update_tag()
    armature.update_tag(refresh={'OBJECT','DATA','TIME'})
    context.scene.frame_set(1)
    context.view_layer.update()
    missing=len(set(by_hash)-{row[0] for row in bindings})
    action_data.update(action,{'dbh_native_unmapped_tracks':missing})
    return action,source,missing


def _resample(values, count, quaternion=False):
    if len(values)==count:
        return tuple(values)
    result=[]
    for i in range(count):
        at=i*(len(values)-1)/max(1,count-1)
        low=int(at)
        alpha=at-low
        a,b=values[low],values[min(low+1,len(values)-1)]
        if isinstance(a,Pose):
            qa,qb=quat_normalize(a.rotation),quat_normalize(b.rotation)
            if sum(x*y for x,y in zip(qa,qb))<0:
                qb=tuple(-v for v in qb)
            result.append(Pose(quat_normalize(tuple(x*(1-alpha)+y*alpha for x,y in zip(qa,qb))),
                               tuple(x*(1-alpha)+y*alpha for x,y in zip(a.translation,b.translation)),
                               tuple(x*(1-alpha)+y*alpha for x,y in zip(a.scale,b.scale))))
        else:
            result.append(a*(1-alpha)+b*alpha)
    return tuple(result)


def export_native_animation(context, destination, use_scene_range=False):
    from .animation_blender import _target, _native_rig_for_animation, _native_bind_matrices
    armature,metadata=_target(context)
    if not armature.animation_data or not armature.animation_data.action:
        raise ValueError('Select a Detroit rig with an active Action')
    action=armature.animation_data.action
    if action_data.get(action,'dbh_animation_space') not in ('native_bind_relative_v1','native_bind_relative_v2') and not action.name.endswith('_retargeted'):
        raise ValueError('This Action has no verified Detroit bone space; reimport the native animation before editing/exporting its keys')
    source=source_for_action(context,action)
    rig=_native_rig_for_animation(metadata)
    bindings=_bindings(rig)
    bones=[armature.pose.bones.get(item['name']) for item in metadata['bones']]
    if any(bone is None for bone in bones):
        raise ValueError('Missing bone in the export rig')
    if use_scene_range:
        first,last=context.scene.frame_start,context.scene.frame_end
    else:
        first,last=map(float,action.frame_range)
        if action_data.get(action,'dbh_animation_space')=='native_bind_relative_v1' or action.name.endswith('_retargeted'):
            if round(last-first+1)==source.frame_count+2:
                first+=2  # Old SMD importer inserted bind/static setup frames.
    if last<first or last-first>100000:
        raise ValueError('Invalid animation export range')
    scene_fps=context.scene.render.fps/context.scene.render.fps_base
    step=scene_fps*source.timestep
    count=round((last-first)/step)+1
    if count*len(bones)>8000000:
        raise ValueError('Animation export is too large')
    rest=[bone.bone.matrix_local.copy() for bone in bones]
    bind=_native_bind_matrices(rig)
    transform=[rest[i].inverted() @ bind[i] @ AXIS for i in range(len(bones))]
    axis_inverse=AXIS.inverted()
    samples=[[] for _ in bones]
    original_frame=context.scene.frame_current
    original_subframe=context.scene.frame_subframe
    try:
        for i in range(count):
            at=first+i*step
            context.scene.frame_set(math.floor(at),subframe=at-math.floor(at))
            world=[axis_inverse @ bone.matrix @ transform[j] for j,bone in enumerate(bones)]
            for j,(_,parent,_) in enumerate(bindings):
                local=world[parent].inverted() @ world[j] if parent>=0 else world[j]
                samples[j].append(_pose(local))
    finally:
        context.scene.frame_set(original_frame,subframe=original_subframe)
    original={track.bone_hash:track for track in source.tracks}
    output={h:replace(t,samples=_resample(t.samples,count)) for h,t in original.items()}
    next_order=max(t.order for t in source.tracks)+1
    written=0
    for i,(bone_hash,_,bind_pose) in enumerate(bindings):
        poses=samples[i]
        old=original.get(bone_hash)
        if old is None:
            difference=max(max(abs(a-b) for a,b in zip(p.translation,bind_pose.translation)) for p in poses)
            difference=max(difference,max(1-abs(sum(a*b for a,b in zip(quat_normalize(p.rotation),quat_normalize(bind_pose.rotation)))) for p in poses))
            difference=max(difference,max(max(abs(a-b) for a,b in zip(p.scale,bind_pose.scale)) for p in poses))
            if difference<1e-5:
                continue
        flags=(old.flags if old else 0x50) | 0x13
        flags &= ~8
        varied_scale=any(max(abs(a-b) for a,b in zip(p.scale,poses[0].scale))>1e-6 for p in poses[1:])
        if varied_scale or max(abs(v-1.) for v in poses[0].scale)>1e-5:
            source=replace(source,flags=source.flags|0x800)
        if varied_scale:
            flags |= 8
        inv_bind=(-bind_pose.rotation[0],-bind_pose.rotation[1],-bind_pose.rotation[2],bind_pose.rotation[3])
        previous=None
        converted=[]
        for pose in poses:
            rotation=quat_mul(inv_bind,pose.rotation) if flags&0x40 else pose.rotation
            rotation=quat_normalize(rotation)
            if previous is not None and sum(a*b for a,b in zip(previous,rotation))<0:
                rotation=tuple(-v for v in rotation)
            previous=rotation
            position=tuple(a-b for a,b in zip(pose.translation,bind_pose.translation))
            scale=pose.scale if varied_scale else poses[0].scale
            converted.append(Pose(rotation,position,scale))
        order=old.order if old else next_order
        if old is None:
            next_order+=1
        output[bone_hash]=Track(bone_hash,order,0,flags,1.,tuple(converted))
        written+=1
    extras=tuple(replace(t,samples=_resample(t.samples,count)) for t in source.extra_tracks)
    raw=encode_animation(source,tuple(output[h] for h in sorted(output)),extras,count)
    verified=parse_animation(raw)
    by_hash={t.bone_hash:t for t in verified.tracks}
    max_position=max_rotation=max_scale=0.
    for i,(bone_hash,_,bind_pose) in enumerate(bindings):
        track=by_hash.get(bone_hash)
        if track is None:
            continue
        for frame,expected in enumerate(samples[i]):
            actual=absolute_pose(track,frame,bind_pose.rotation,bind_pose.translation,bind_pose.scale)
            max_position=max(max_position,max(abs(a-b) for a,b in zip(actual.translation,expected.translation)))
            max_scale=max(max_scale,max(abs(a-b) for a,b in zip(actual.scale,expected.scale)))
            dot=min(1.,abs(sum(a*b for a,b in zip(actual.rotation,quat_normalize(expected.rotation)))))
            max_rotation=max(max_rotation,2*math.acos(dot))
    if max_position>2e-4 or max_rotation>8e-5 or max_scale>2e-4:
        raise ValueError(f'Native export readback failed: position {max_position:.6g}, angle {max_rotation:.6g}, scale {max_scale:.6g}')
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix='.dbh_anim_',suffix='.pending',dir=destination.parent)
    try:
        with os.fdopen(descriptor,'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary,destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {'frames':count,'tracks':len(verified.tracks),'edited_rig_tracks':written,
            'max_position_error':max_position,'max_rotation_error':max_rotation,
            'max_scale_error':max_scale,'bytes':len(raw)}
