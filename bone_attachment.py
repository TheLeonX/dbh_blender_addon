"""Opt-in rigid attachment to an existing native skeleton bone."""
import bpy


def active_attachment(obj,metadata):
    names={bone['name']:i for i,bone in enumerate(metadata.get('bones',[]))}
    matches=[c for c in obj.constraints if c.type=='CHILD_OF' and not c.mute and c.influence>0]
    if not matches:return None
    if len(matches)!=1:raise ValueError(f'{obj.name}: only one active Child Of constraint can export')
    constraint=matches[0]
    if constraint.influence<.999999:
        raise ValueError(f'{obj.name}: Child Of influence must be 1.0 for game export')
    if not constraint.target or constraint.target.type!='ARMATURE' or constraint.subtarget not in names:
        raise ValueError(f'{obj.name}: Child Of must target an imported game bone')
    expected=metadata.get('armature_name')
    if expected and constraint.target.name!=expected:
        raise ValueError(f'{obj.name}: Child Of targets a different package armature')
    return constraint,names[constraint.subtarget]


def suggested_bone(obj,metadata):
    names={bone['name'] for bone in metadata.get('bones',[])}
    groups={g.index:g.name for g in obj.vertex_groups if g.name in names}
    totals={name:0.0 for name in groups.values()}
    for vertex in obj.data.vertices:
        for assignment in vertex.groups:
            if assignment.group in groups:totals[groups[assignment.group]]+=assignment.weight
    if totals:return max(totals,key=totals.get)
    return next((bone['name'] for bone in metadata.get('bones',[]) if bone['name']!='Root'),'')


def attach(context,bone):
    from .scene_export import manifest_for_object,slot_for_object
    obj=context.active_object
    found=manifest_for_object(obj)
    if not obj or obj.type!='MESH' or not found:raise ValueError('Select a named Detroit mesh slot')
    _,metadata=found
    slot_for_object(obj,metadata)
    if bone not in {b['name'] for b in metadata.get('bones',[])}:
        raise ValueError('Choose one of the imported game bones')
    from .blender_io import object_armature
    arm=object_armature(obj) or bpy.data.objects.get(metadata.get('armature_name',''))
    if not arm or arm.type!='ARMATURE' or bone not in arm.pose.bones:
        raise ValueError('Imported armature/bone is missing')
    if context.object.mode!='OBJECT':bpy.ops.object.mode_set(mode='OBJECT')
    from .scene_export import slot_entry
    binding = slot_entry(obj,metadata).get('native_attachment')
    if binding:
        from .native import Package
        from .auxiliary_mesh import setup
        from pathlib import Path
        package = Package(Path(metadata['source_segs']).read_bytes())
        bi = next(i for i,b in enumerate(metadata['bones']) if b['name'] == bone)
        result = setup(obj,arm,metadata,package,binding,bi)
        context.view_layer.update()
        return result
    constraint=obj.constraints.get('Detroit Bone Attachment')
    if constraint and constraint.type!='CHILD_OF':raise ValueError('Constraint name is already in use')
    if constraint is None:constraint=obj.constraints.new('CHILD_OF');constraint.name='Detroit Bone Attachment'
    constraint.target=arm;constraint.subtarget=bone;constraint.influence=1.0
    context.view_layer.update()
    with context.temp_override(object=obj,active_object=obj,selected_objects=[obj],selected_editable_objects=[obj]):
        result=bpy.ops.constraint.childof_set_inverse(constraint=constraint.name,owner='OBJECT')
    if result!={'FINISHED'}:raise ValueError('Blender could not set Child Of inverse')
    for modifier in obj.modifiers:
        if modifier.type=='ARMATURE' and modifier.object==arm:
            modifier.show_viewport=False;modifier.show_render=False
    context.view_layer.update()
    return constraint


def detach(context):
    obj=context.active_object
    if not obj or obj.type!='MESH':raise ValueError('Select a mesh')
    constraint=obj.constraints.get('Detroit Bone Attachment')
    if not constraint or constraint.type!='CHILD_OF':raise ValueError('No Detroit bone attachment on this mesh')
    obj.constraints.remove(constraint)
    from .scene_export import manifest_for_object,slot_entry
    found = manifest_for_object(obj)
    if found and slot_entry(obj,found[1]).get('native_attachment'):
        context.view_layer.update()
        return
    for modifier in obj.modifiers:
        if modifier.type=='ARMATURE':
            modifier.show_viewport=True;modifier.show_render=True
    context.view_layer.update()
