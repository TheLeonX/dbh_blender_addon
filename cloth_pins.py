"""Explicit diagnostics/repair for unanchored islands; never change paint on export."""
from pathlib import Path
import bpy
from mathutils.kdtree import KDTree
from .cloth_native import GROUP,PREVIEW
from .cloth_build import geometry,connected_pieces


def pin_plan(vertices,faces):
    weights,edges,seams=geometry(vertices,faces,require_pins=False)
    pieces=connected_pieces(len(vertices),edges,seams)
    missing=[c for c in pieces if not any(weights[i]<=1e-6 for i in c)]
    pinned=[i for i,w in enumerate(weights) if w<=1e-6]
    tree=KDTree(len(pinned)) if pinned else None
    if tree:
        for i in pinned: tree.insert(vertices[i]['position'],i)
        tree.balance()
    by_piece={i:n for n,c in enumerate(missing) for i in c}
    candidates=[[] for c in missing]
    for face in faces:
        n=by_piece.get(face[0])
        if n is not None and all(by_piece.get(i)==n for i in face): candidates[n].append(face)
    anchors=set()
    for c,triangles in zip(missing,candidates):
        if not triangles: raise ValueError('Unpinned cloth piece has no usable triangle; remove loose vertices')
        def score(face):
            center=tuple(sum(vertices[i]['position'][a] for i in face)/3 for a in range(3))
            distance=tree.find(center)[2] if tree else 0.
            return distance,sum(weights[i] for i in face),tuple(sorted(face))
        anchors.update(min(triangles,key=score))
    # A split render vertex and its seam twins must receive the same paint.
    while True:
        before=len(anchors)
        for a,b in seams:
            if a in anchors or b in anchors: anchors.update((a,b))
        if len(anchors)==before: break
    return missing,anchors


def analyze(context):
    from .scene_export import manifest_for_object,slot_for_object
    from .blender_io import extract_mesh,digest
    from .native import Package
    obj=context.active_object
    if not obj or obj.type!='MESH' or not obj.vertex_groups.get(GROUP):
        raise ValueError('Select a mesh with CLOTH_SIMULATION')
    if obj.mode!='OBJECT': bpy.ops.object.mode_set(mode='OBJECT')
    # Source vertex indexes cannot safely paint generated/reordered vertices.
    from .scene_organization import preview_modifier
    if any(m.show_viewport and m.type!='ARMATURE' and not preview_modifier(m) for m in obj.modifiers):
        raise ValueError('Apply/remove geometry-changing modifiers before checking cloth pins')
    found=manifest_for_object(obj)
    if not found: raise ValueError('Assign the mesh to a Detroit replacement slot first')
    metadata=found[1];raw=Path(metadata['source_segs']).read_bytes()
    if digest(raw)!=metadata['package_sha256']: raise ValueError('Source package changed; reimport first')
    package=Package(raw);ri,mi=slot_for_object(obj,metadata);md=package.mesh(ri)
    vertices,faces=extract_mesh(context,obj,md,md.flat()[mi],metadata,package=package,cloth_sources=True)
    if not vertices: raise ValueError('No cloth triangles to check')
    if not any(v['cloth_weight']>0 for v in vertices): return obj,vertices,[],set()
    missing,anchors=pin_plan(vertices,faces)
    return obj,vertices,missing,anchors


def repair(context):
    obj,vertices,missing,anchors=analyze(context)
    source={vertices[i]['cloth_source_vertex'] for i in anchors}
    if source:
        obj.vertex_groups[GROUP].add(sorted(source),0.,'REPLACE')
        mod=obj.modifiers.get(PREVIEW)
        if mod and mod.type=='CLOTH': obj.modifiers.remove(mod)
    obj.vertex_groups.active_index=obj.vertex_groups[GROUP].index
    return len(missing),len(source)


class OBJECT_OT_dbh_cloth_select_unpinned(bpy.types.Operator):
    bl_idname='object.dbh_cloth_select_unpinned'
    bl_label='Select Unpinned Pieces'
    bl_description='Highlight only islands without zero-weight anchors, using the same seam rules as game export'
    bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            obj,vertices,missing,anchors=analyze(context)
            source={vertices[i]['cloth_source_vertex'] for c in missing for i in c}
            for selected in context.selected_objects: selected.select_set(False)
            obj.select_set(True)
            for v in obj.data.vertices: v.select=v.index in source
            for e in obj.data.edges: e.select=False
            for p in obj.data.polygons: p.select=False
            context.tool_settings.mesh_select_mode=(True,False,False)
            bpy.ops.object.mode_set(mode='EDIT')
            self.report({'WARNING'} if missing else {'INFO'},f'{len(missing)} unpinned piece(s); {len(source)} source vertices selected')
            return {'FINISHED'}
        except Exception as error: self.report({'ERROR'},str(error));return {'CANCELLED'}


class OBJECT_OT_dbh_cloth_pin_unpinned(bpy.types.Operator):
    bl_idname='object.dbh_cloth_pin_unpinned'
    bl_label='Pin Unpinned Pieces'
    bl_description='Paint a triangle-sized anchor at 0 on each unpinned island, nearest existing pinned geometry; keep other weights. Stops preview; Undo supported'
    bl_options={'REGISTER','UNDO'}
    def invoke(self,context,event): return context.window_manager.invoke_confirm(self,event)
    def execute(self,context):
        try:
            count,vertices=repair(context)
            self.report({'INFO'},f'{count} piece(s) anchored; {vertices} source vertices painted 0. Restart cloth preview.')
            return {'FINISHED'}
        except Exception as error: self.report({'ERROR'},str(error));return {'CANCELLED'}
