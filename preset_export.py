"""Export authored materials, texture-role slots and canonical draw layouts."""
import copy
import hashlib
import struct
from pathlib import Path
import numpy as np
from .native import Package,Container,Record,VB,Submesh,MeshData,append_mesh
from .materials import ArchiveTextures,bindings,texture_record
from .texture_export import make_texture,image_rgba,mip_chain,validate_texture
from .preset_shader import PRESETS,ROLES,BINDINGS,required_roles,build_shader,detect_preset,role_bindings
from .cel_shader import CEL_PRESETS
from .pbr_shader import PBR_PRESETS,DEFAULTS,DATA_ROLES


class PresetExport:
    def __init__(self,package,metadata,experimental,fingerprints=None,single_mip=False):
        self.package=package;self.metadata=metadata;self.experimental=experimental
        self.resources={};self.textures={};self.materials={};self.reports=[];self.texture_reports=[]
        # Export-local only: a later export must see every texture paint/edit.
        self.image_textures={};self.opacity_checks={}
        self.fingerprints=fingerprints
        self.single_mip=single_mip
        self.donor=None;self.canonical={};self.pbr_donor=None;self.dither_donor=None;self.hair_donor=None

    def template(self,pbr=False,dither=False,hair=False):
        cached=self.hair_donor if hair else self.dither_donor if dither else self.pbr_donor if pbr else self.donor
        if cached:return cached
        asset=0x14B2B if hair else 0x14B22 if dither else 0x14B12 if pbr else 0x14B2A
        p=self.package
        r=next((r for r in p.container.records if r.kind==2133 and r.asset==asset),None)
        if r is None:
            archive=ArchiveTextures(self.metadata.get('game_index'))
            raw,_=archive.raw(29,0x1B40)
            if raw.startswith(b'QZIP\0DC_INFO '):
                # The installed PC archive keeps its directory uncompressed,
                # then appends the ordinary SEGS external members. Wrap only
                # that directory in a temporary SEGS member; no game file is
                # modified and all resource offsets remain relative to it.
                body=raw[5:]
                end=Container(body,allow_trailing=True).end
                if body[end:end+4]!=b'segs':
                    raise ValueError('Archive donor has no SEGS resources after DATA_CONTAINER')
                from .reference_archive import compressed_member
                raw=compressed_member(body[:end])+body[end:]
            p=Package(raw)
            r=next((r for r in p.container.records if r.kind==2133 and r.asset==asset),None)
        if r is None or not r.external:raise ValueError(f'Game presets need native shader carrier {asset:X} in package 0x1B40')
        raw=p.members[p.record_members[r.index]].unpacked
        # Verify donor before exporting any mutations.
        if hair:
            from .hair_shader import build_shader as build_hair
            build_hair(r,raw)
        elif dither:
            from .dither_pbr_shader import build_shader as build_dither
            build_dither(r,raw,'SKIN')
        elif pbr:
            from .pbr_shader import build_shader as build_pbr
            build_pbr(r,raw,'SKIN')
        else:build_shader(r,raw,'STANDARD_DIFFUSE')
        # Native FILETEXT donor remains the verified coat image format.
        tex=self.template()[2] if pbr or dither or hair else texture_record(p,bindings(r.payload)[21].texture)
        found=None
        for rec in p.container.records:
            if rec.kind!=2130:continue
            try:header=MeshData(rec.payload)
            except ValueError:continue
            if not any(s.material[1]==asset for s in header.flat()):continue
            md=p.mesh(rec.index)
            for sub in md.flat():
                if sub.material[1]!=asset:continue
                vb=md.vbs[sub.vb]
                if vb.flag==0 and vb.strides==((20,0,0,52) if dither else (16,0,0,52) if pbr or hair else (28,0,0,52)):found=(sub,vb);break
            if found:break
        if not found:raise ValueError('Preset carrier vertex schema is unavailable')
        value=(r,raw,tex,*found)
        if hair:self.hair_donor=value
        elif dither:self.dither_donor=value
        elif pbr:self.pbr_donor=value
        else:self.donor=value
        return value

    def _texture(self,image,role,channel=None):
        input_key=(image.as_pointer() if image is not None else None,role,channel)
        if input_key in self.image_textures:return self.image_textures[input_key]
        _,_,template,_,_=self.template()
        srgb=role not in DATA_ROLES
        if image:w,h,pixels=image_rgba(image,srgb)
        else:
            color=DEFAULTS[role]
            w=h=4;pixels=np.tile(np.array(color,np.float32),(h,w,1))
        if channel is not None:
            pixels=np.repeat(pixels[:,:,'RGBA'.index(channel):'RGBA'.index(channel)+1],4,axis=2)
        levels=mip_chain(pixels,srgb,self.single_mip)
        key=hashlib.sha256(b'DBH_PRESET_TEXTURE_V1'+bytes([srgb])+struct.pack('<II',w,h)+b''.join(levels)).digest()
        if key in self.textures:
            self.image_textures[input_key]=self.textures[key];return self.textures[key]
        asset=0x60000000|(int.from_bytes(key[:4],'little')&0xfffffff)
        settings=bytearray(template.payload);settings[58]=int(srgb);settings[52]=4;settings[53:56]=bytes((0,0,1))
        template=Record(0,2137,0,template.prefix,bytes(settings))
        existing={r.asset:r for r in self.package.container.records}
        while True:
            rec,raw=make_texture(template,len(self.package.container.records),asset,w,h,levels)
            old=existing.get(asset)
            if old is None:break
            if old.kind==2137 and old.payload==rec.payload:
                original=self.resources.get(old.index)
                if original is None and old.index in self.package.record_members:
                    original=self.package.members[self.package.record_members[old.index]].unpacked
                if original==raw:
                    self.textures[key]=asset;self.image_textures[input_key]=asset;return asset
            asset=0x60000000|((asset+1)&0xfffffff)
        validate_texture(rec,raw)
        self.package.container.records.append(rec);self.resources[rec.index]=raw
        self.textures[key]=asset
        self.image_textures[input_key]=asset
        self.texture_reports.append(dict(texture=f'{asset:X}',role=role,image=image.name if image else 'Neutral default',width=w,height=h))
        return asset

    @staticmethod
    def opaque_opacity(image,channel):
        if image is None:return True
        _,_,pixels=image_rgba(image,False)
        index='RGBA'.index(channel)
        # The opaque path cannot represent partial transparency. Require an
        # effectively white channel before routing a material to deferred.
        return bool((pixels[:,:,index]>=1-1/1024).all())

    def material(self,mat,surface):
        if mat is None:raise ValueError('A face uses an empty material slot; assign a Detroit preset material')
        mode=mat.dbh_shader_mode
        if mode=='HAIR':
            # Native hair color edits keep their own vertex schema and cloth.
            original=next((r for r in self.package.container.records if r.kind==2133 and r.asset==mat.get('dbh_material_id')),None)
            if original is not None and original.external:
                raw=self.resources.get(original.index,self.package.members[self.package.record_members[original.index]].unpacked)
                from .hair_rgb_shader import supported
                if supported(original,raw):return self.native_hair(mat,original,raw)
        if mode not in PRESETS:
            asset=mat.get('dbh_material_id')
            if asset is None:raise ValueError(f'{mat.name}: choose a Game Shader preset before export')
            if asset!=0 and not any(r.kind==2133 and r.asset==asset for r in self.package.container.records):
                raise ValueError(f'{mat.name}: material belongs to another package; choose a standard preset')
            return asset,None
        cache=mat.as_pointer(),surface
        if cache in self.materials:return self.materials[cache]
        slots={}
        for s in mat.dbh_preset_slots:
            if s.role not in ROLES or s.role in slots:raise ValueError(f'{mat.name}: duplicate/unsupported texture role')
            if s.role not in required_roles(mode):raise ValueError(f'{mat.name}: {s.role} requires a higher Game Shader preset')
            slots[s.role]=s.image
        pbr=mode in PBR_PRESETS
        opaque=False;dither=pbr and mat.dbh_surface_mode=='DITHERED'
        if pbr:
            image=slots.get('ALPHA');opacity_key=(image.as_pointer() if image is not None else None,mat.dbh_alpha_channel)
            if opacity_key not in self.opacity_checks:
                self.opacity_checks[opacity_key]=self.opaque_opacity(image,mat.dbh_alpha_channel)
            white=self.opacity_checks[opacity_key]
            if mat.dbh_surface_mode=='OPAQUE' and not white:
                raise ValueError(f'{mat.name}: Opaque Skin/Cloth requires an all-white opacity channel')
            opaque=white and mat.dbh_surface_mode not in ('BLENDED','DITHERED')
        template,raw,_,sub,vb=self.template(pbr and not opaque,dither,mode=='HAIR')
        if mode=='HAIR':
            from .hair_shader import build_shader as build_hair
            payload,shader=build_hair(template,raw,mat.dbh_alpha_channel,mat.dbh_face_mode)
        elif pbr:
            if dither:from .dither_pbr_shader import build_shader as build_pbr
            elif opaque:from .opaque_pbr_shader import build_shader as build_pbr
            else:from .pbr_shader import build_shader as build_pbr
            payload,shader=build_pbr(template,raw,mode,mat.dbh_fabric_scale,mat.dbh_alpha_channel,mat.dbh_diffuse_emission,
                                     motion_blur=mat.dbh_motion_blur,face_mode=mat.dbh_face_mode)
        else:
            payload,shader=build_shader(template,raw,mode,mat.dbh_uv_offset2,mat.dbh_diffuse_emission,mat.dbh_ambient_color,
                                        outline=bool(mat.get('dbh_outline_generated')),motion_blur=mat.dbh_motion_blur,
                                        face_mode=mat.dbh_face_mode)
        payload=bytearray(payload)
        physical_roles=tuple(role for role in required_roles(mode) if role!='ALPHA') if opaque else (
            required_roles(mode) if pbr or mode=='HAIR' else ROLES[:4] if mode in CEL_PRESETS else ROLES[:3])
        role_ids={role:self._texture(slots.get(role),role) for role in physical_roles}
        internal={}
        if mode=='HAIR':
            from .hair_shader import INTERNAL_BINDINGS
            internal={index:self._texture(None,role) for index,role in INTERNAL_BINDINGS.items()}
        table=bindings(payload)
        for b in table:
            role=next((role for role,i in role_bindings(mode,opaque=opaque,dither=dither).items() if i==b.index and role in physical_roles),None)
            # All inherited, unused carrier inputs are safe neutral resources.
            asset=internal.get(b.index,role_ids[role or 'COLOR'])
            struct.pack_into('<II',payload,b.offset,2137,asset)
            payload[b.offset+8]=0 # all authored channels use UV0 (Blender UV1)
        payload=bytes(payload)
        previous=mat.get('dbh_material_id')
        existing={r.asset:r for r in self.package.container.records}
        old=existing.get(previous)
        if old and old.kind==2133 and old.payload==payload:
            oldraw=self.resources.get(old.index)
            if oldraw is None and old.index in self.package.record_members:
                oldraw=self.package.members[self.package.record_members[old.index]].unpacked
            # A legacy packing repair alone is not a material edit and must
            # not allocate a new material or require Experimental Textures.
            from .shader_profiles import normalize_stream
            if oldraw is not None and normalize_stream(old,oldraw)==shader:
                value=previous,sub;self.materials[cache]=value;return value
        if not self.experimental:raise ValueError('New materials and Game Shader presets require Experimental Textures')
        key=hashlib.sha256(b'DBH_PRESET_MATERIAL_V1'+repr(surface).encode()+payload+shader).digest()
        asset=0x50000000|(int.from_bytes(key[:4],'little')&0xfffffff)
        while asset in existing:asset=0x50000000|((asset+1)&0xfffffff)
        rec=Record(len(self.package.container.records),2133,asset,template.prefix,payload)
        self.package.container.records.append(rec);self.resources[rec.index]=shader
        self.reports.append(dict(material=f'{asset:X}',name=mat.name,preset=mode,
                                 textures={role:f'{a:X}' for role,a in role_ids.items()},surface=list(surface)))
        if mode in CEL_PRESETS:self.reports[-1]['uvOffset2']=list(mat.dbh_uv_offset2)
        self.reports[-1]['diffuseEmission']=mat.dbh_diffuse_emission
        self.reports[-1]['motionBlur']=mat.dbh_motion_blur
        self.reports[-1]['visibleFaces']=mat.dbh_face_mode
        if mode=='HAIR':self.reports[-1].update(alphaChannel=mat.dbh_alpha_channel,alphaMode='NATIVE_HAIR',
            diffuseEmission=0,motionBlur='native hair pipeline',internalTextures={str(k):f'{v:X}' for k,v in internal.items()})
        if mode in CEL_PRESETS:self.reports[-1]['ambientColor']=list(mat.dbh_ambient_color)
        if pbr:
            self.reports[-1].update(alphaChannel=mat.dbh_alpha_channel,alphaMode='DITHERED' if dither else 'OPAQUE' if opaque else 'BLENDED',
                                    fabricScale=mat.dbh_fabric_scale)
        if mat.get('dbh_outline_generated'):self.reports[-1]['outline']=True
        value=asset,sub;self.materials[cache]=value;return value

    def native_hair(self,mat,record,raw):
        """Keep this exact material ID, vertex programs and native draw links."""
        cache=mat.as_pointer(),'native_hair'
        if cache in self.materials:return self.materials[cache]
        if mat.dbh_face_mode!='BOTH':
            raise ValueError('Source-preserving native hair keeps its native face mode; choose Both')
        from .hair_rgb_shader import build_shader
        from .material_ui import pixel_hash
        payload,shader=build_shader(record,raw);payload=bytearray(payload)
        table=bindings(payload);slots={s.role:s for s in mat.dbh_preset_slots}
        if len(slots)!=len(mat.dbh_preset_slots) or any(role not in required_roles('HAIR') for role in slots):
            raise ValueError('Native hair has duplicate/unsupported texture roles')
        native_slots={s.binding:s for s in mat.dbh_texture_slots};textures={}
        for role,index in role_bindings('HAIR').items():
            slot=slots.get(role);image=slot.image if slot else None
            if image is None:continue
            imported=native_slots.get(index)
            if imported and image==imported.original_image:
                fingerprint=self.fingerprints.get(image) if self.fingerprints is not None else pixel_hash(image)
                if fingerprint==image.get('dbh_pixels_hash'):continue
            asset=self._texture(image,role,mat.dbh_alpha_channel if role=='ALPHA' else None)
            struct.pack_into('<II',payload,table[index].offset,2137,asset)
            payload[table[index].offset+8]=0
            textures[role]=f'{asset:X}'
        payload=bytes(payload)
        if payload!=record.payload or shader!=raw:
            if not self.experimental:raise ValueError('Native hair RGB/texture edits require Experimental Textures')
            record.payload=payload;self.resources[record.index]=shader
            self.reports.append(dict(material=f'{record.asset:X}',name=mat.name,preset='HAIR_RGB',
                textures=textures,source_preserving=True,cloth_rebuilt=False,vertex_programs='unchanged'))
        value=record.asset,None;self.materials[cache]=value;return value

    def append(self,md,sub,vertices,faces,asset,carrier):
        if carrier is None:
            # Existing original shader reassignment is allowed only when its
            # source interface/detail metrics are known, not guessed.
            if asset!=sub.material[1]:raise ValueError('Choose a standard preset when assigning another original game shader')
            return append_mesh(md,sub,vertices,faces)
        pbr=carrier.material[1]==0x14B12
        _,_,_,donor_sub,donor_vb=self.template(pbr,carrier.material[1]==0x14B22,carrier.material[1]==0x14B2B)
        key=id(md),carrier.material[1]
        if key not in self.canonical:
            if len(md.vbs)>=256:raise ValueError('Too many vertex buffers for preset geometry')
            streams=[bytearray(s[donor_sub.first_vertex*n:(donor_sub.first_vertex+1)*n]) for s,n in zip(donor_vb.streams,donor_vb.strides)]
            self.canonical[key]=len(md.vbs)
            md.vbs.append(VB(1,0,donor_vb.layout,donor_vb.strides,streams));md.padding.append(b'')
        desc=bytearray(sub.descriptor[:sub.detail_offset])
        struct.pack_into('<I',desc,6,self.canonical[key]);struct.pack_into('<II',desc,10,0,1)
        struct.pack_into('<I',desc,30,8) # carrier's eight-influence declaration
        struct.pack_into('<II',desc,sub.detail_offset-8,2133,asset)
        desc+=donor_sub.detail_table
        target=Submesh(bytes(desc),sub.suffix)
        verts=[dict(v,template_index=0,color=(1,1,1,1)) for v in vertices]
        return append_mesh(md,target,verts,faces)
