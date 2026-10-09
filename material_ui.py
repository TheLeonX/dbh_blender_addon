"""Material Properties texture bindings and explicit Blender preview controls."""
import hashlib
import json
import tempfile
from pathlib import Path
import bpy
import numpy as np
from bpy.props import StringProperty, IntProperty, PointerProperty, CollectionProperty, EnumProperty, FloatVectorProperty, FloatProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper
from .materials import bindings, texture_record, texture_info, ArchiveTextures
from .textures import decode_qd, decode_dds, png_bytes
from .native import Package


def sha(data): return hashlib.sha256(data).hexdigest()


def pixel_hash(image):
    pixels=np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(pixels)
    # Identical digest without allocating two additional full image buffers.
    import hashlib
    digest=hashlib.sha256(str(tuple(image.size)).encode());digest.update(memoryview(pixels))
    return digest.hexdigest()


def pixel_hash_many(images):
    """Bounded parallel SHA; Blender pixel access stays on the main thread.

    Hash every pixel, including saved/packed/painted images. Do not trust
    is_dirty as proof of unchanged content. Digests match pixel_hash exactly.
    """
    from concurrent.futures import ThreadPoolExecutor
    unique={image.as_pointer():image for image in images if image is not None}
    if len(unique)<4:return {ptr:pixel_hash(image) for ptr,image in unique.items()}
    def fingerprint(size,pixels):
        digest=hashlib.sha256(str(size).encode());digest.update(memoryview(pixels))
        return digest.hexdigest()
    result={};pending=[];memory=0
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix='DBH-image-hash') as pool:
        for ptr,image in unique.items():
            count=len(image.pixels)
            while pending and (len(pending)>=4 or memory+count*4>256*1024*1024):
                key,future,n=pending.pop(0);result[key]=future.result();memory-=n
            pixels=np.empty(count,np.float32);image.pixels.foreach_get(pixels)
            pending.append((ptr,pool.submit(fingerprint,tuple(image.size),pixels),pixels.nbytes));memory+=pixels.nbytes
        for ptr,future,n in pending:result[ptr]=future.result()
    return result


class PixelHashQueue:
    """Overlap import hashing with main-thread Blender work, bounded to 256 MiB.

    Only plain NumPy buffers enter workers; all RNA access and hash-property
    writes remain on the importing thread. No skipped pixels or lazy hashes.
    """
    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        self.pool=ThreadPoolExecutor(max_workers=4,thread_name_prefix='DBH-import-hash')
        self.pending=[];self.memory=0

    @staticmethod
    def fingerprint(size,pixels):
        digest=hashlib.sha256(str(size).encode());digest.update(memoryview(pixels))
        return digest.hexdigest()

    def flush_one(self):
        image,future,size=self.pending.pop(0)
        image['dbh_pixels_hash']=future.result();self.memory-=size

    def add(self,image):
        count=len(image.pixels)
        while self.pending and (len(self.pending)>=4 or self.memory+count*4>256*1024*1024):self.flush_one()
        pixels=np.empty(count,np.float32);image.pixels.foreach_get(pixels)
        self.pending.append((image,self.pool.submit(self.fingerprint,tuple(image.size),pixels),pixels.nbytes))
        self.memory+=pixels.nbytes

    def finish(self):
        while self.pending:self.flush_one()

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True);self.pending.clear();self.memory=0


class TextureLoader:
    def __init__(self,package,folders,index,override_folder='',defer_hashes=False):
        self.package=package;self.folders=folders;self.archive=ArchiveTextures(index)
        self.override_folder=override_folder
        self.images={};self.errors={}
        self.hash_queue=PixelHashQueue() if defer_hashes else None
        self.cache_keys={}

    def embedded(self,record):
        if record is None:return []
        if record.external:return [self.package.members[self.package.record_members[record.index]].unpacked]
        return [self.package.texture_attachments[k,a].data for k,a,s in texture_info(record)['dependencies']
                if (k,a) in self.package.texture_attachments]

    @staticmethod
    def cache_key(payload,embedded,stamp,bgra):
        digest=hashlib.sha256(payload)
        if embedded:
            for chunk in embedded:digest.update(chunk)
        else:digest.update(stamp.encode())
        key=digest.hexdigest()[:20]
        return sha((key+':bgra1').encode())[:20] if bgra else key

    def prefetch(self,assets):
        # Embedded bytes belong to this immutable Package; no file or Blender
        # access occurs on workers. Preserve the exact existing PNG-cache key.
        if not self.hash_queue:return
        for asset in assets:
            if not asset or asset in self.cache_keys:continue
            record=texture_record(self.package,asset)
            data=self.embedded(record)
            if not data:continue  # Archive keys keep their live index stamp.
            self.cache_keys[asset]=self.hash_queue.pool.submit(self.cache_key,record.payload,tuple(data),'',record.payload[44]==0)

    def finish_hashes(self):
        if self.hash_queue:self.hash_queue.finish()

    def close(self):
        if self.hash_queue:self.hash_queue.close()

    def load(self,asset):
        if asset in self.images:return self.images[asset]
        if asset in self.errors:return None
        record=texture_record(self.package,asset)
        failures=[];image=None
        # Cache is keyed by FILETEXT metadata AND the archive index identity.
        # A replaced game texture cannot silently reuse a stale decoded image.
        cache=Path(tempfile.gettempdir())/'dbh_preview_v036';cache.mkdir(exist_ok=True)
        stamp='missing'
        try:
            st=self.archive.index.stat();stamp=f'{self.archive.index}:{st.st_size}:{st.st_mtime_ns}'
        except OSError:pass
        embedded=self.embedded(record)
        if asset in self.cache_keys:key=self.cache_keys[asset].result()
        else:key=self.cache_key(record.payload if record else b'',embedded,stamp,bool(record and record.payload[44]==0))
        generated=cache/f'{asset:X}_{key}.png'
        candidates=[];overrides=[]
        for folder in self.folders:
            for ext in ('.png','.tga','.tif','.tiff','.dds','.jpg','.jpeg','.exr',''):
                path=Path(folder)/(f'{asset:X}'+ext)
                if path.is_file():
                    (overrides if self.override_folder and Path(folder)==Path(self.override_folder) else candidates).append(path)
        # Embedded SEGS bytes are authoritative, including over explicit folders.
        if embedded:overrides=[];candidates=[]
        # Prefer current archive data to stale/partial legacy extractor caches.
        if record and not generated.is_file() and not overrides:
            try:generated.write_bytes(png_bytes(*self.archive.image(record,self.package)))
            except (ValueError,RuntimeError,OSError,OverflowError) as error:failures.append(str(error))
        if generated.is_file():candidates.insert(0,generated)
        candidates=overrides+candidates
        for path in candidates:
            try:
                target=path
                if path.suffix.lower() in ('','.dds'):
                    raw=path.read_bytes();target=cache/(path.stem+'_'+sha(raw)[:20]+'.png')
                    if not target.is_file():target.write_bytes(png_bytes(*(decode_dds(raw) if raw[:4]==b'DDS ' else decode_qd(raw))))
                image=bpy.data.images.load(str(target),check_existing=False)
                if not image.size[0]:raise ValueError('Empty image')
                break
            except (ValueError,RuntimeError,OSError,OverflowError) as error:failures.append(str(error))
        if image is None and record:
            try:
                generated.write_bytes(png_bytes(*self.archive.image(record,self.package)))
                image=bpy.data.images.load(str(generated),check_existing=False)
            except (ValueError,RuntimeError,OSError,OverflowError) as error:failures.append(str(error))
        if image:
            image.name=f'DBH_tex_{asset:X}'
            if record:image.colorspace_settings.name='sRGB' if record.payload[58] else 'Non-Color'
            image['dbh_texture_id']=f'{asset:X}'
            if self.hash_queue:self.hash_queue.add(image)
            else:image['dbh_pixels_hash']=pixel_hash(image)
            image.pack()
            self.images[asset]=image
            return image
        self.errors[asset]='; '.join(dict.fromkeys(failures)) or 'No texture record or file'
        return None


def update_slot(self,context):
    mat=self.id_data
    if not isinstance(mat,bpy.types.Material) or not mat.node_tree:return
    from .preset_shader import PRESETS
    if mat.dbh_shader_mode in PRESETS:
        from .preset_ui import sync_preview
        sync_preview(mat);return
    node=mat.node_tree.nodes.get(self.node_name)
    if node:node.image=self.image
    if mat.get('dbh_diffuse_source')==self.asset:
        diffuse=mat.node_tree.nodes.get('DBH Diffuse')
        if diffuse:diffuse.image=self.image


class DBHTextureSlot(bpy.types.PropertyGroup):
    name:StringProperty()
    binding:IntProperty(default=-1)
    asset:StringProperty(name='Original resource')
    game_flags:IntProperty()
    image:PointerProperty(name='Image',type=bpy.types.Image,update=update_slot)
    original_image:PointerProperty(type=bpy.types.Image)
    node_name:StringProperty()
    status:StringProperty()
    uv:StringProperty(name='Preview UV map',default='UV1')
    role:EnumProperty(name='Blender preview',items=[('COLOR','Base Color',''),('NORMAL','Normal',''),
        ('ROUGHNESS','Roughness',''),('METALLIC','Metallic',''),('ALPHA','Alpha',''),('EMISSION','Emission',''),('CEL','Celshade','Tone lookup texture'),
        ('ORM','ORM','R occlusion, G roughness, B metallic'),('FABRIC','Fabric Normal','Tiled fabric normal map'),
        ('TANGENT','Hair Strand Direction','Native hair tangent XY map; not a surface normal map')])
    channel:EnumProperty(name='Channel',items=[(x,x,'') for x in ('RGB','R','G','B','A')])


def populate(material,package,loader):
    if material.dbh_texture_slots:return
    record=next((r for r in package.container.records if r.kind==2133 and r.asset==material.get('dbh_material_id')),None)
    if record is None:return
    from .preset_ui import populate as populate_preset
    if populate_preset(material,record,package,loader):return
    try:slots=bindings(record.payload)
    except ValueError as error:
        material['dbh_binding_error']=str(error);return
    material['dbh_shader_version']=91
    if record.external:
        from .hair_rgb_shader import detect
        if detect(package.members[package.record_members[record.index]].unpacked):material.dbh_shader_mode='HAIR_RGB'
        from .eye_emission import detect as detect_eye
        value=detect_eye(package.members[package.record_members[record.index]].unpacked)
        if value is not None:
            material.dbh_shader_mode='EYE_EMISSION';material.dbh_eye_emission_strength=value
    if record.asset==0x14B2A and record.external:
        from .shader_profiles import is_coat_rgb
        raw=package.members[package.record_members[record.index]].unpacked
        material.dbh_shader_mode='COAT_RGB' if is_coat_rgb(record,raw) else 'ORIGINAL'
    nodes=material.node_tree.nodes
    for b in slots:
        slot=material.dbh_texture_slots.add();slot.binding=b.index;slot.asset=f'{b.texture:X}'
        slot.name=f'Slot {b.index:02d} — {b.texture:X}';slot.game_flags=b.flags
        node=nodes.new('ShaderNodeTexImage');node.name=f'DBH Slot {b.index:02d}'
        node.label=slot.name;node.location=(-900-(b.index//10)*340,600-(b.index%10)*280)
        slot.node_name=node.name;slot.image=loader.load(b.texture) if b.texture else None
        slot.original_image=slot.image
        if slot.image:
            slot.status=f'{slot.image.size[0]} × {slot.image.size[1]}'
        else:slot.status=loader.errors.get(b.texture,'Empty game reference')
        uv=nodes.new('ShaderNodeUVMap');uv.uv_map='UV1';uv.name=f'DBH UV {b.index:02d}'
        uv.location=(node.location.x-180,node.location.y)
        material.node_tree.links.new(uv.outputs['UV'],node.inputs['Vector'])
    material['dbh_texture_slot_count']=len(slots)
    if material.dbh_shader_mode=='EYE_EMISSION':
        material.dbh_eye_emission_image=next((s.image for s in material.dbh_texture_slots if s.binding==5),None)
        from .eye_preview import sync
        sync(material)
    # Improve base colour when the old extractor omitted an RGB texture.
    if material.get('dbh_diffuse_status')!='LOADED':
        candidates=[]
        for slot in material.dbh_texture_slots:
            rec=texture_record(package,int(slot.asset,16))
            if slot.image and rec:
                info=texture_info(rec)
                if info['format'] in (5,7):candidates.append((slot.image.size[0]*slot.image.size[1],slot.binding))
        preferred=next((s for s in material.dbh_texture_slots if s.image and s.asset==material.get('dbh_preferred_diffuse')),None)
        if candidates or preferred:
            slot=preferred if preferred is not None else material.dbh_texture_slots[max(candidates)[1]]
            preview(material,slot)
            material['dbh_diffuse_status']='LOADED'
            material['dbh_diffuse_source']=slot.asset
    # Select the diffuse slot if known, so the first displayed map is useful.
    chosen=material.get('dbh_diffuse_source','')
    if material.dbh_shader_mode=='HAIR_RGB':chosen=f'{slots[1].texture:X}'
    if material.dbh_shader_mode=='EYE_EMISSION':chosen=f'{slots[2].texture:X}'
    for slot in material.dbh_texture_slots:
        if slot.asset==chosen:
            material.dbh_texture_index=slot.binding
            diffuse=nodes.get('DBH Diffuse')
            if diffuse and slot.image:diffuse.image=slot.image
            if 'dbh_preview_bindings' not in material:
                material['dbh_preview_bindings']=json.dumps({'COLOR':{'binding':slot.binding,'channel':'RGB','uv':'UV1'}})
            if material.dbh_shader_mode in ('HAIR_RGB','EYE_EMISSION') and slot.image:preview(material,slot)
            break


def preview(mat,slot):
    from .preset_shader import PRESETS
    if mat.dbh_shader_mode in PRESETS:
        from .preset_ui import sync_preview
        sync_preview(mat);return
    if not slot.image:raise ValueError('Load an image into this slot first')
    nodes=mat.node_tree.nodes;links=mat.node_tree.links
    node=nodes.get(slot.node_name)
    if node is None:raise ValueError('Texture node is missing; refresh the material')
    node.image=slot.image
    nodes.active=node;node.select=True
    uv=nodes.get(f'DBH UV {slot.binding:02d}')
    if uv:uv.uv_map=slot.uv
    output=node.outputs['Color']
    if slot.channel=='A':output=node.outputs['Alpha']
    elif slot.channel!='RGB':
        name=f'DBH Channels {slot.binding:02d}'
        split=nodes.get(name) or nodes.new('ShaderNodeSeparateColor')
        split.name=name;links.new(output,split.inputs['Color'])
        output=split.outputs[{'R':'Red','G':'Green','B':'Blue'}[slot.channel]]
    target=nodes.get('Principled BSDF')
    if target is None:raise ValueError('Principled BSDF node is missing')
    socket={'COLOR':'Base Color','NORMAL':'Normal','ROUGHNESS':'Roughness',
            'METALLIC':'Metallic','ALPHA':'Alpha','EMISSION':'Emission Color'}[slot.role]
    if slot.role=='NORMAL':
        name=f'DBH Normal {slot.binding:02d}'
        normal=nodes.get(name) or nodes.new('ShaderNodeNormalMap')
        normal.name=name;normal.uv_map=slot.uv
        links.new(output,normal.inputs['Color']);output=normal.outputs['Normal']
    links.new(output,target.inputs[socket])
    if slot.role=='EMISSION':target.inputs['Emission Strength'].default_value=1
    saved=json.loads(mat.get('dbh_preview_bindings','{}'))
    saved[slot.role]=dict(binding=slot.binding,channel=slot.channel,uv=slot.uv)
    mat['dbh_preview_bindings']=json.dumps(saved)


def active_slot(context):
    mat=getattr(context,'material',None) or (context.object.active_material if context.object else None)
    from .preset_shader import PRESETS
    if mat and mat.dbh_shader_mode in PRESETS:
        if not mat.dbh_preset_slots:raise ValueError('Add a texture slot first')
        return mat,mat.dbh_preset_slots[min(mat.dbh_preset_index,len(mat.dbh_preset_slots)-1)]
    if not mat or not mat.dbh_texture_slots:raise ValueError('Select an imported Detroit material')
    return mat,mat.dbh_texture_slots[min(mat.dbh_texture_index,len(mat.dbh_texture_slots)-1)]


class MATERIAL_OT_dbh_load_texture(bpy.types.Operator,ImportHelper):
    bl_idname='material.dbh_load_texture';bl_label='Load / Add Texture';bl_options={'REGISTER','UNDO'}
    filter_glob:StringProperty(default='*.png;*.jpg;*.jpeg;*.tga;*.tif;*.tiff;*.bmp;*.dds;*.exr;*.hdr',options={'HIDDEN'})
    def execute(self,context):
        try:
            mat,slot=active_slot(context)
            image=bpy.data.images.load(self.filepath,check_existing=False)
            if not image.size[0]:raise ValueError('Blender could not decode the image')
            from .preset_shader import PRESETS
            from .pbr_shader import DATA_ROLES
            if mat.dbh_shader_mode in PRESETS:image.colorspace_settings.name='Non-Color' if slot.role in DATA_ROLES else 'sRGB'
            image.pack();slot.image=image;slot.status='Custom image — assigned to shader slot'
            preview(mat,slot)
            return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}


class MATERIAL_OT_dbh_preview(bpy.types.Operator):
    bl_idname='material.dbh_preview_texture';bl_label='Preview Selected Map';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:preview(*active_slot(context));return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}


class MATERIAL_OT_dbh_new_texture(bpy.types.Operator):
    bl_idname='material.dbh_new_texture';bl_label='New Blank Texture';bl_options={'REGISTER','UNDO'}
    width:IntProperty(name='Width',default=1024,min=1,max=4096)
    height:IntProperty(name='Height',default=1024,min=1,max=4096)
    def invoke(self,context,event):return context.window_manager.invoke_props_dialog(self)
    def execute(self,context):
        try:
            mat,slot=active_slot(context)
            image=bpy.data.images.new(f'{mat.name}_slot_{slot.binding:02d}',width=self.width,height=self.height,alpha=True)
            from .preset_shader import PRESETS
            color=(1,1,1,1)
            if mat.dbh_shader_mode in PRESETS:
                from .pbr_shader import DEFAULTS,DATA_ROLES
                color=DEFAULTS[slot.role]
                image.colorspace_settings.name='Non-Color' if slot.role in DATA_ROLES else 'sRGB'
            image.generated_color=color;image.pack();slot.image=image
            slot.status='New image — assigned to existing shader slot';preview(mat,slot)
            return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}


class MATERIAL_OT_dbh_reset_texture(bpy.types.Operator):
    bl_idname='material.dbh_reset_texture';bl_label='Reset to Imported Texture';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        mat,slot=active_slot(context);slot.image=slot.original_image
        if slot.image:preview(mat,slot)
        return {'FINISHED'}


class MATERIAL_OT_dbh_refresh(bpy.types.Operator):
    bl_idname='material.dbh_refresh_textures';bl_label='Load / Refresh Model Textures';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            obj=context.object
            from .scene_export import manifest_for_object
            found=manifest_for_object(obj)
            if not obj or not found:raise ValueError('Select an imported Detroit mesh')
            text,meta=found
            source=meta.get('source_segs') or obj['dbh_source_segs']
            package=Package(Path(source).read_bytes())
            if sha(package.source_data)!=meta.get('package_sha256',obj.get('dbh_source_sha256')):raise ValueError('Source changed; reimport the package')
            prefs=context.preferences.addons[__package__].preferences
            folders=list(meta.get('texture_folders',[]))
            override=bpy.path.abspath(prefs.texture_folder) if prefs.texture_folder else ''
            if override and override not in folders:folders.insert(0,override)
            from .game_paths import resolve_index
            index=resolve_index(context,meta,preferences=prefs)
            meta['game_index']=str(index) if index is not None else ''
            loader=TextureLoader(package,folders,index,override)
            names={entry['name'] for entry in meta.get('slot_objects',{}).values()}
            mats={mat for other in context.scene.objects if other.type=='MESH' and
                  (other.name in names if names else other.get('dbh_source_segs')==source)
                  for mat in other.data.materials if mat}
            for mat in mats:
                if 'dbh_material_id' not in mat:continue
                populate(mat,package,loader)
                for slot in mat.dbh_texture_slots:
                    if slot.image is None and slot.original_image is None:
                        slot.image=loader.load(int(slot.asset,16));slot.original_image=slot.image
            missing=sum(not slot.image for mat in mats for slot in mat.dbh_texture_slots)
            from .metadata_text import write_metadata
            write_metadata(text,meta)
            self.report({'WARNING'} if missing else {'INFO'},f'{len(mats)} materials; {missing} missing texture slots')
            return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}


class MATERIAL_UL_dbh_textures(bpy.types.UIList):
    def draw_item(self,context,layout,data,item,icon,active_data,active_propname,index):
        row=layout.row();row.label(text=item.name,icon='IMAGE_DATA' if item.image else 'ERROR')
        if item.image and item.image!=item.original_image:row.label(text='Changed',icon='FILE_REFRESH')


class MATERIAL_PT_dbh_textures(bpy.types.Panel):
    bl_label='Detroit Shader Texture Slots';bl_idname='MATERIAL_PT_dbh_textures'
    bl_space_type='PROPERTIES';bl_region_type='WINDOW';bl_context='material'
    @classmethod
    def poll(cls,context):return getattr(context,'material',None) is not None or (context.object and context.object.type=='MESH')
    def draw(self,context):
        mat=context.material;layout=self.layout
        from .preset_ui import draw as draw_preset
        from .preset_shader import PRESETS
        layout.operator('material.dbh_new_material',icon='ADD')
        if mat is None:return
        layout.prop(mat,'dbh_shader_mode')
        if mat.dbh_shader_mode in PRESETS:
            draw_preset(layout,mat);return
        if 'dbh_material_id' not in mat:
            layout.label(text='Choose a standard Game Shader preset to export this material.')
            return
        layout.label(text=f"Game material {mat['dbh_material_id']:X} — SHADCUST")
        if mat.dbh_shader_mode=='HAIR_RGB':
            layout.label(text='Diffuse: Slot 1 RGB, UV1; native cloth and hair passes retained',icon='INFO')
        if mat.dbh_shader_mode=='EYE_EMISSION':
            layout.label(text='Native iris/cornea retained; emission uses projected eye UVs',icon='INFO')
            layout.template_ID(mat,'dbh_eye_emission_image',open='image.open')
            layout.prop(mat,'dbh_eye_emission_strength')
            layout.label(text='Select an imported native eye material; not eyelids or eye sockets.')
        if mat['dbh_material_id']==0x14B2A:
            if mat.dbh_shader_mode=='COAT_RGB':
                layout.label(text='Game base color: Slot 21 RGB, UV1',icon='INFO')
            else:
                layout.label(text='Slot 21 is grayscale masks, not diffuse.',icon='INFO')
            layout.label(text='Slot 11: noise; Slot 19: tiled fabric detail.')
        if not mat.dbh_texture_slots:
            layout.operator(MATERIAL_OT_dbh_refresh.bl_idname)
            layout.label(text=mat.get('dbh_binding_error','No loaded texture slots'))
            return
        layout.template_list('MATERIAL_UL_dbh_textures','',mat,'dbh_texture_slots',mat,'dbh_texture_index',rows=6)
        slot=mat.dbh_texture_slots[min(mat.dbh_texture_index,len(mat.dbh_texture_slots)-1)]
        layout.label(text=f'Binding {slot.binding} / original texture {slot.asset}')
        layout.template_ID(slot,'image',open=MATERIAL_OT_dbh_load_texture.bl_idname)
        layout.template_ID_preview(slot,'image',rows=3,cols=6,hide_buttons=True)
        layout.operator(MATERIAL_OT_dbh_load_texture.bl_idname,icon='FILE_FOLDER')
        layout.operator(MATERIAL_OT_dbh_new_texture.bl_idname,icon='ADD')
        if slot.image:
            layout.label(text=f'{slot.image.size[0]} × {slot.image.size[1]} · {slot.image.name}')
            layout.prop(slot.image.colorspace_settings,'name',text='Image Color Space')
        else:layout.label(text=slot.status,icon='ERROR')
        layout.prop(slot,'role');layout.prop(slot,'channel');layout.prop(slot,'uv')
        layout.operator(MATERIAL_OT_dbh_preview.bl_idname,icon='SHADING_TEXTURE')
        layout.operator(MATERIAL_OT_dbh_reset_texture.bl_idname)
        layout.label(text='Preview controls do not rewrite the game shader.',icon='INFO')
        layout.label(text='New images use existing game bindings.')
        layout.label(text='Choose a Standard preset to add/delete role slots.')
        layout.label(text='Custom image export: enable Experimental Textures.')
        layout.operator(MATERIAL_OT_dbh_refresh.bl_idname,icon='FILE_REFRESH')


CLASSES=(DBHTextureSlot,MATERIAL_OT_dbh_load_texture,MATERIAL_OT_dbh_preview,MATERIAL_OT_dbh_new_texture,
         MATERIAL_OT_dbh_reset_texture,MATERIAL_OT_dbh_refresh,MATERIAL_UL_dbh_textures,MATERIAL_PT_dbh_textures)

def register():
    for c in CLASSES:bpy.utils.register_class(c)
    from .preset_ui import CLASSES as PRESET_CLASSES,mode_changed,uv_changed,emission_changed,surface_changed
    for c in PRESET_CLASSES:bpy.utils.register_class(c)
    bpy.types.Material.dbh_texture_slots=CollectionProperty(type=DBHTextureSlot)
    bpy.types.Material.dbh_texture_index=IntProperty(default=0,min=0)
    bpy.types.Material.dbh_preset_slots=CollectionProperty(type=DBHTextureSlot)
    bpy.types.Material.dbh_preset_index=IntProperty(default=0,min=0)
    bpy.types.Material.dbh_motion_blur=BoolProperty(name='Motion Blur',default=True,
        description='Game export only: disable to exclude covered pixels from camera/object motion blur. Requires DBH Loose Loader 0.2 and re-export. Global blur and temporal AA remain enabled')
    bpy.types.Material.dbh_surface_mode=EnumProperty(name='Skin/Cloth Render',items=[
        ('AUTO','Auto','Use opaque deferred shading for a fully white opacity map; otherwise blended transparency'),
        ('OPAQUE','Opaque','Require a fully white opacity map; use the sharp opaque deferred path'),
        ('BLENDED','Blended','Keep partial transparency; camera-motion blur may affect the surface'),
        ('DITHERED','Dithered','Screen-door opacity with matching color, depth and motion coverage; recommended for hair cards; may show stippling')],
        default='AUTO',update=surface_changed)
    bpy.types.Material.dbh_face_mode=EnumProperty(name='Visible Faces',items=[
        ('FRONT','Front','Render front-facing triangles only'),
        ('BACK','Back','Render back-facing triangles only'),
        ('BOTH','Both','Render both sides of triangles')],default='BOTH',update=emission_changed,
        description='Game shader fragment culling and Blender preview; applies to authored presets')
    bpy.types.Material.dbh_fabric_scale=FloatProperty(name='Fabric Scale',default=10,min=.001,max=10000,
        soft_max=100,update=emission_changed,description='UV1 tiling of the fabric normal map. Larger = finer fabric. Saved in blend and SEGS')
    bpy.types.Material.dbh_alpha_channel=EnumProperty(name='Opacity Channel',items=[(c,c,'') for c in ('R','G','B','A')],
        default='R',update=emission_changed,description='Read this channel of the opacity texture. R for grayscale maps; A for embedded alpha')
    bpy.types.Material.dbh_diffuse_emission=FloatProperty(name='Diffuse Emission',default=0,min=0,max=1,
        precision=3,update=emission_changed,
        description='0: engine lighting. 1: remove engine shading while retaining cel ramp shadows, ambient tint and separate emission. Saved in blend and SEGS; re-export to update game')
    from .eye_preview import changed as eye_changed
    bpy.types.Material.dbh_eye_emission_image=PointerProperty(name='Eye Emission Texture',type=bpy.types.Image,update=eye_changed,
        description='Separate RGB emission texture; black is no glow. Game uses the native projected iris/sclera UVs')
    bpy.types.Material.dbh_eye_emission_strength=FloatProperty(name='Eye Emission Strength',default=1,min=0,max=100,soft_max=10,
        update=eye_changed,description='Add this map after native eye lighting, before game exposure/tone mapping. 0 disables glow; preserved in blend and SEGS')
    bpy.types.Material.dbh_ambient_color=FloatVectorProperty(name='Ambient Color',subtype='COLOR',size=3,
        default=(0,0,0),min=0,max=1,update=uv_changed,
        description='F00A ambient RGB fill for cel shadows, before the tone/color blend. Black is no fill. Saved in blend and SEGS; re-export to update game')
    bpy.types.Material.dbh_uv_offset2=FloatVectorProperty(name='uvOffset2',size=2,default=(.5,0),
        soft_min=-1,soft_max=1,precision=4,update=uv_changed,
        description='X: F00A cel texture row (V). Y: extra horizontal ramp bias. Saved in blend and exported SEGS; re-export to update game')
    bpy.types.Material.dbh_shader_mode=EnumProperty(name='Game shader',items=[
        ('ORIGINAL','Original layered shader','Keep the game material behavior'),
        ('COAT_RGB','Coat: Direct RGB (experimental)','Use slot 21 RGB on UV1 as game base color; verified shader revision only'),
        ('HAIR_RGB','Original Hair: Direct Diffuse RGB','Native hair slot 1 RGB without replacing vertex programs or cloth; verified revisions only'),
        ('STANDARD_DIFFUSE','Standard: Diffuse','Opaque lit color on UV1; flat normal'),
        ('STANDARD_NORMAL','Standard: Diffuse + Normal','Opaque lit color with tangent-space normal on UV1'),
        ('STANDARD_EMISSION','Standard: Diffuse + Normal + Emission','Opaque lit color, tangent-space normal and emission on UV1'),
        ('CEL_DIFFUSE','Celshade: Diffuse + Celshade','F00A tone ramp with flat normal'),
        ('CEL_NORMAL','Celshade: Diffuse + Normal + Celshade','F00A tone ramp with tangent-space normal map'),
        ('CEL_NORMAL_EMISSION','Celshade: Diffuse + Normal + Emission + Celshade','F00A tone ramp, normal map and emission'),
        ('CEL_EMISSION','Celshade: Diffuse + Emission + Celshade','F00A tone ramp and emission; flat normal'),
        ('SKIN','Skin: Diffuse + ORM + Normal + Alpha','Auto opaque/blended PBR skin surface; no specialized SSS'),
        ('CLOTH','Cloth: Diffuse + ORM + Normal + Alpha + Fabric','Auto opaque/blended PBR cloth with a tiled fabric normal map'),
        ('HAIR','Native Hair: Diffuse + Alpha + Strand Direction','Native density/depth hair passes and smooth alpha edges; no added screen-door noise'),
        # Append, never insert: existing .blend files store numeric enum values.
        ('EYE_EMISSION','Original Eyes + Emission','Retain verified native iris/cornea/vertex shaders; add a separate emission texture at the end of color shading')],update=mode_changed)
    from .preset_ui import refresh_presets_after_load
    if refresh_presets_after_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(refresh_presets_after_load)
    if hasattr(bpy.data,'materials'):refresh_presets_after_load()
    elif not bpy.app.timers.is_registered(refresh_presets_after_load):
        bpy.app.timers.register(refresh_presets_after_load,first_interval=.1)

def unregister():
    from .preset_ui import refresh_presets_after_load
    if refresh_presets_after_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(refresh_presets_after_load)
    if bpy.app.timers.is_registered(refresh_presets_after_load):
        bpy.app.timers.unregister(refresh_presets_after_load)
    del bpy.types.Material.dbh_shader_mode
    del bpy.types.Material.dbh_motion_blur
    del bpy.types.Material.dbh_surface_mode
    del bpy.types.Material.dbh_face_mode
    del bpy.types.Material.dbh_fabric_scale
    del bpy.types.Material.dbh_alpha_channel
    del bpy.types.Material.dbh_diffuse_emission
    del bpy.types.Material.dbh_eye_emission_image
    del bpy.types.Material.dbh_eye_emission_strength
    del bpy.types.Material.dbh_ambient_color
    del bpy.types.Material.dbh_uv_offset2
    del bpy.types.Material.dbh_preset_index
    del bpy.types.Material.dbh_preset_slots
    del bpy.types.Material.dbh_texture_index
    del bpy.types.Material.dbh_texture_slots
    for c in reversed(CLASSES):bpy.utils.unregister_class(c)
    from .preset_ui import CLASSES as PRESET_CLASSES
    for c in reversed(PRESET_CLASSES):bpy.utils.unregister_class(c)
