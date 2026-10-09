"""Per-mesh physics/wind editor stored in the package manifest, not objects."""
import copy
import math
import struct
import uuid
from pathlib import Path
import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty, EnumProperty, StringProperty,CollectionProperty
from .cloth_native import ClothIndex,resource_tag
from .cloth_profiles import read_profiles
from .scene_export import slot_key,slot_for_object,manifest_for_object
from .metadata_text import write_metadata
from .cloth_colliders import donors


def default_donor(package,metadata,record,mesh,index=None):
    index=index or ClothIndex(package)
    try:binding=index.binding(record,mesh)
    except ValueError:binding=None
    rows=donors(package,record)
    if binding:
        current=next(row for row in rows if row[0]==binding.asset)
        if not current[1].startswith('DBH_CLOTH_'):return current
        saved=next((r for r in metadata.get('cloth_simulation_masks',[]) if r.get('resource_id')==binding.asset),{})
        named=[r for r in rows if r[1]==saved.get('physics_donor') and not r[1].startswith('DBH_CLOTH_')]
        if len(named)==1:return named[0]
        # Retained vanilla group-zero inputs are the safe same-material fallback.
        from .cloth_route import family
        md=package.mesh(family(package,record).cloth);mat=package.mesh(record).flat()[mesh].material
        assets=set()
        for sub in md.groups[0].meshes:
            selector=sub.descriptor[5]-1
            if sub.material==mat and md.vbs[sub.vb].flag==1 and 0<=selector<(struct.unpack_from('<I',md.references)[0]&255):
                assets.add(struct.unpack_from('<II',md.references,4+8*selector)[1])
        matches=[r for r in rows if r[0] in assets and not r[1].startswith('DBH_CLOTH_')]
        if len(matches)==1:return matches[0]
        # Do not guess another garment if its original donor was removed.
        return current
    from .cloth_route import donor,family
    _,_,raw=donor(package,family(package,record),package.mesh(record).flat()[mesh])
    matches=[r for r in rows if r[2]==raw]
    if len(matches)!=1:raise ValueError('New cloth needs a unique registered physics donor')
    return matches[0]


def import_profiles(package,metadata,record,mesh,index,binding):
    cache_profiles(package,metadata,record,mesh,index)
    report=next((r for r in metadata.get('cloth_simulation_masks',[]) if (r.get('resource_id'),r.get('buffer'))==(binding.asset,binding.buffer)),{})
    if report.get('collider_selection') is not None:
        metadata['slot_objects'].setdefault(slot_key(record,mesh),{}).setdefault('cloth_physics',{})['colliders']=report['collider_selection']
    if report.get('world_collision') is not None:
        metadata['slot_objects'].setdefault(slot_key(record,mesh),{}).setdefault('cloth_physics',{})['world_collision']=report['world_collision']
    # Defaults are implicit; do not create override values on import.
    return binding.asset


def cache_profiles(package,metadata,record,mesh,index=None):
    if metadata.get('cloth_profile_sha256')!=metadata.get('package_sha256'):
        metadata['cloth_native_profiles']={};metadata['cloth_donor_catalog']={};metadata['cloth_default_donors']={};metadata['cloth_collider_catalog']={}
        metadata['cloth_profile_sha256']=metadata.get('package_sha256')
    cache=metadata.setdefault('cloth_native_profiles',{})
    rows=donors(package,record)
    metadata.setdefault('cloth_donor_catalog',{})[str(record)]=[(asset,name) for asset,name,raw in rows]
    for asset,name,raw in rows:
        existing=cache.get(str(asset))
        if not existing or any(not p.get('controls') or 'world_collision' not in p or not set(PHYSICS_FIELDS)<=p['controls'].keys() for p in existing['profiles']):cache[str(asset)]=read_profiles(raw)
    metadata.setdefault('cloth_default_donors',{})[slot_key(record,mesh)]=default_donor(package,metadata,record,mesh,index)[0]


def editor_data(metadata,record,mesh):
    """Use imported scalar snapshots; decode older scenes only on first use."""
    from .native import Package
    from .blender_io import digest
    raw=Path(metadata['source_segs']).read_bytes()
    if metadata.get('package_sha256') and digest(raw)!=metadata['package_sha256']:
        raise ValueError('Source package changed; reimport before editing its cloth settings')
    rows=metadata.get('cloth_donor_catalog',{}).get(str(record))
    profiles=metadata.get('cloth_native_profiles',{})
    valid=(metadata.get('cloth_profile_sha256')==metadata.get('package_sha256') and rows and
           slot_key(record,mesh) in metadata.get('cloth_default_donors',{}) and
           all(str(asset) in profiles and all(p.get('controls') and 'world_collision' in p and set(PHYSICS_FIELDS)<=p['controls'].keys() for p in profiles[str(asset)]['profiles']) for asset,name in rows))
    if not valid:cache_profiles(Package(raw),metadata,record,mesh)
    rows=metadata['cloth_donor_catalog'][str(record)]
    return rows,{str(asset):metadata['cloth_native_profiles'][str(asset)] for asset,name in rows},not valid


def collider_editor_data(metadata,record):
    cache=metadata.setdefault('cloth_collider_catalog',{})
    if str(record) not in cache:
        from .native import Package
        from .blender_io import digest
        from .cloth_colliders import rows,mesh_links,default_names
        raw=Path(metadata['source_segs']).read_bytes()
        if digest(raw)!=metadata['package_sha256']:raise ValueError('Source package changed; reimport before selecting colliders')
        package=Package(raw);aliases=mesh_links(package,metadata,record)
        cache[str(record)]=dict(rows=[dict(r,slot=aliases.get(r['name'],'')) for r in rows(package,record)],
            defaults={str(asset):default_names(raw) for asset,name,raw in donors(package,record)})
    return cache[str(record)]


def mesh_settings(scene,obj,metadata,package,index=None):
    record,mesh=slot_for_object(obj,metadata)
    preferences=metadata.get('slot_objects',{}).get(slot_key(record,mesh),{}).get('cloth_physics',{})
    asset=preferences.get('donor_asset')
    if asset is None:asset=default_donor(package,metadata,record,mesh,index)[0]
    result=dict(inherit_source=True,donor_asset=asset,overrides=copy.deepcopy(preferences.get('overrides',{})),wind={},colliders=copy.deepcopy(preferences.get('colliders')))
    result['world_collision']=preferences.get('world_collision')
    if preferences.get('override_wind'):
        wind=preferences.get('wind',{})
        result['wind']=dict(global_scale=wind['global_scale'] if preferences.get('wind_enabled',True) else 0.,
                            local_scale=wind['local_scale'] if preferences.get('wind_enabled',True) else 0.,local_deadzone=wind['local_deadzone'])
    return result


def preview_settings(obj,metadata,package,moving):
    """Approximate native default profile in Blender's different cloth solver."""
    desired=mesh_settings(None,obj,metadata,package)
    raw=next(raw for asset,name,raw in donors(package,slot_for_object(obj,metadata)[0]) if asset==desired['donor_asset'])
    snapshot=read_profiles(raw);profile=snapshot['profiles'][snapshot['default_index']]
    overrides=desired['overrides']
    return dict(mass=overrides.get('mass',profile['total_mass']/max(1,moving)),
                radius=overrides.get('radius',profile['radius_range'][1]),
                quality=overrides.get('substeps',profile['configs'][0]['substeps'])*overrides.get('iterations',profile['configs'][0]['iterations']),
                collisions=overrides.get('collisions',bool(profile['colliders'])),
                gravity=profile['gravity'])


def apply_wind(index,requests,payloads,resources):
    from .cloth_wind import write_wind
    planned={}
    for binding,values in requests:
        if not values:continue
        old=planned.get(binding.resource)
        if old and old[1]!=values:raise ValueError('Conflicting wind settings on meshes sharing one native cloth resource')
        planned[binding.resource]=binding,values
    for ri,(binding,values) in planned.items():
        rec=index.package.container.records[ri];target=resources if rec.external else payloads
        raw=target.get(ri,bytes(binding.tag.raw));fixed=write_wind(raw,values)
        if fixed!=raw:target[ri]=fixed


NO_DONORS=(('NONE','No native donor',''),)
_DONOR_SESSIONS={}


def donor_items(self,context):
    # Blender calls this with OperatorProperties, not the Python Operator.
    # RNA owns the token; Python-only instance attributes are not visible here.
    # Retain enum strings for the complete RNA/UI lifetime (Blender requirement).
    session=_DONOR_SESSIONS.get(getattr(self,'donor_session',''))
    return session['items'] if session else NO_DONORS


def release_donor_session(self):
    session=_DONOR_SESSIONS.get(self.donor_session)
    if session:
        # Keep only enum strings after closing; do not retain full cloth payloads.
        for field in ('rows','snapshot','profiles','baseline','colliders'):session.pop(field,None)


def clear_donor_sessions():
    _DONOR_SESSIONS.clear()


def changed_donor(self,context):
    session=_DONOR_SESSIONS.get(self.donor_session)
    if not session or session.get('initializing') or not session.get('rows'):return
    snapshot=session['profiles'].get(self.donor_asset)
    if snapshot is None:return
    session['snapshot']=snapshot
    controls=snapshot['profiles'][snapshot['default_index']]['controls']
    session['baseline']={field:controls[field] for field in PHYSICS_FIELDS}
    for field in PHYSICS_FIELDS:setattr(self,field,controls[field])
    for field in WIND_FIELDS:setattr(self,field,snapshot['wind'][field])
    self.wind_enabled=True
    self.inherit_world_collisions=True
    self.world_collisions=snapshot['profiles'][snapshot['default_index']]['world_collision']
    if hasattr(self,'collider_items'):
        defaults=session.get('colliders',{}).get('defaults',{}).get(self.donor_asset,[])
        for item in self.collider_items:item.enabled=item.native_name in defaults


PHYSICS_FIELDS=('mass','radius','stiffness','bend','elasticity','bounciness','movement','max_distance','damping','substeps','iterations','collisions')
WIND_FIELDS=('global_scale','local_scale','local_deadzone')


def different(value,source):
    if isinstance(source,bool) or isinstance(source,int):return value!=source
    return not math.isclose(value,source,rel_tol=1e-6,abs_tol=1e-9)


def changed_bounciness(self,context):
    if abs(self.damping-(1-self.bounciness))>1e-6:self.damping=1-self.bounciness


def changed_damping(self,context):
    if abs(self.bounciness-(1-self.damping))>1e-6:self.bounciness=1-self.damping


class DBHClothColliderItem(bpy.types.PropertyGroup):
    native_name:StringProperty()
    mesh_name:StringProperty()
    shape:StringProperty()
    enabled:BoolProperty(default=False)


class DBH_UL_cloth_colliders(bpy.types.UIList):
    def draw_item(self,context,layout,data,item,icon,active_data,active_propname,index):
        row=layout.row(align=True);row.prop(item,'enabled',text='')
        row.label(text=item.mesh_name or item.native_name.replace('_CHA_:',''),icon='MESH_DATA')
        if item.mesh_name:row.label(text=item.native_name.replace('_CHA_:',''))
    def filter_items(self,context,data,propname):
        query=self.filter_name.casefold()
        return [self.bitflag_filter_item if query in (i.mesh_name+' '+i.native_name).casefold() else 0 for i in getattr(data,propname)],[]


class OBJECT_OT_dbh_cloth_settings(bpy.types.Operator):
    bl_idname='object.dbh_cloth_settings'
    bl_label='Cloth Physics & In-Game Wind'
    bl_description='Read native state profiles and save explicit per-mesh physics/wind overrides in the package manifest'
    bl_options={'UNDO'}
    donor_session:StringProperty(options={'HIDDEN','SKIP_SAVE'})
    donor_asset:EnumProperty(name='Physics Donor',items=donor_items,update=changed_donor)
    inherit_colliders:BoolProperty(name='Inherit Native Per-State Colliders',default=True,description='Uncheck to use the selected same-character collision shapes in every simulation state')
    inherit_world_collisions:BoolProperty(name='Inherit Native World Collision Flags',default=True,description='Retain each simulation state\'s original landscape selection')
    world_collisions:BoolProperty(name='World Collisions (Experimental)',default=False,description='Enable native landscape collision for free particles. Requires runtime world geometry; not yet verified to collide with scenery in Detroit')
    collider_items:CollectionProperty(type=DBHClothColliderItem)
    collider_index:IntProperty(default=0)
    use_source:BoolProperty(name='Inherit Native State Profiles',default=True,options={'HIDDEN'})
    mass:FloatProperty(name='Particle Mass (kg)',default=.04,min=.000001,max=10,precision=6)
    radius:FloatProperty(name='Collision Radius (m)',default=.002,min=0,max=.1,precision=6)
    stiffness:FloatProperty(name='Link Stiffness',default=.8,min=0,max=1)
    bend:FloatProperty(name='Bending Stiffness',default=1,min=0,max=1)
    elasticity:FloatProperty(name='Elasticity (link compliance approximation)',description='Softens standard stretch links without changing their rest lengths; not a material Young modulus',default=0,min=0,max=1)
    bounciness:FloatProperty(name='Bounciness (damping-based approximation)',description='1 minus native damping; not collision restitution',default=.05,min=0,max=1,update=changed_bounciness)
    movement:FloatProperty(name='Movement Amount',description='0 follows bones; 1 uses the full painted native cloth motion',default=1,min=0,max=1)
    max_distance:FloatProperty(name='Native Range Limit (m; 0 = disable)',description='Changes existing native distance constraints; does not create new constraints on unchanged topology',default=0,min=0,max=20,precision=6)
    damping:FloatProperty(name='Damping / Second',default=.95,min=0,max=1,precision=4,update=changed_damping)
    substeps:IntProperty(name='Substeps',default=2,min=1,max=16)
    iterations:IntProperty(name='Solver Iterations',default=4,min=1,max=16)
    collisions:BoolProperty(name='Use Native Model Collisions',default=True)
    override_wind:BoolProperty(name='Override Native Wind',default=False,options={'HIDDEN'})
    wind_enabled:BoolProperty(name='Receive Wind',default=True)
    global_scale:FloatProperty(name='Environmental Wind Scale',default=1,min=0,max=1000)
    local_scale:FloatProperty(name='Movement Wind Scale',default=1,min=0,max=1000)
    local_deadzone:FloatProperty(name='Movement Dead Zone (km/h)',default=5,min=0,max=1000)

    def invoke(self,context,event):
        try:
            found=manifest_for_object(context.active_object)
            if not found:raise ValueError('Select an imported Detroit mesh')
            self._text,self._metadata=found;self._record,self._mesh=slot_for_object(context.active_object,self._metadata)
            rows,profiles,updated=editor_data(self._metadata,self._record,self._mesh)
            if not rows:raise ValueError('This character has no registered native cloth donors')
            collider_data=collider_editor_data(self._metadata,self._record)
            write_metadata(self._text,self._metadata)
            self.donor_session=uuid.uuid4().hex
            session=dict(items=tuple((str(asset),name,f'Registered native cloth {name}',i) for i,(asset,name) in enumerate(rows)),
                         rows=rows,profiles=profiles,colliders=collider_data,initializing=True)
            _DONOR_SESSIONS[self.donor_session]=session
            pref=self._metadata['slot_objects'][slot_key(self._record,self._mesh)].get('cloth_physics',{})
            chosen=pref.get('donor_asset')
            if chosen is None:chosen=self._metadata['cloth_default_donors'][slot_key(self._record,self._mesh)]
            if chosen not in {asset for asset,name in rows}:
                raise ValueError('Saved physics donor is not registered in this source package; reset native settings or reimport')
            self.donor_asset=str(chosen);snapshot=profiles[str(chosen)]
            self.inherit_world_collisions=pref.get('world_collision') is None
            self.world_collisions=snapshot['profiles'][snapshot['default_index']]['world_collision'] if self.inherit_world_collisions else pref['world_collision']
            selected=pref.get('colliders');self.inherit_colliders=selected is None
            defaults=collider_data['defaults'].get(str(chosen),[])
            self.collider_items.clear()
            for row in collider_data['rows']:
                item=self.collider_items.add();item.native_name=row['name'];item.shape=row['shape']
                item.mesh_name=self._metadata.get('slot_objects',{}).get(row['slot'],{}).get('name','')
                item.enabled=row['name'] in (defaults if selected is None else selected)
            session['snapshot']=snapshot
            native=snapshot['profiles'][snapshot['default_index']]['controls']
            session['baseline']={field:native[field] for field in PHYSICS_FIELDS}
            controls=native|pref.get('overrides',{})
            for field in PHYSICS_FIELDS:setattr(self,field,controls[field])
            self.use_source=not bool(pref.get('overrides'))
            self.override_wind=pref.get('override_wind',False);self.wind_enabled=pref.get('wind_enabled',True)
            for field in WIND_FIELDS:setattr(self,field,pref.get('wind',{}).get(field,snapshot['wind'][field]) if self.override_wind else snapshot['wind'][field])
            session['initializing']=False
            return context.window_manager.invoke_props_dialog(self,width=620)
        except Exception as error:
            release_donor_session(self);self.report({'ERROR'},str(error));return {'CANCELLED'}

    def draw(self,context):
        layout=self.layout;layout.prop(self,'donor_asset')
        snapshot=_DONOR_SESSIONS[self.donor_session]['snapshot']
        profile=snapshot['profiles'][snapshot['default_index']]
        layout.label(text='Vanilla defaults: '+(', '.join(profile['states']) or profile['name']))
        layout.label(text='Edit any value; unchanged fields retain native state profiles.')
        custom=layout.box();custom.label(text='Cloth Physics — editable vanilla defaults')
        for field in PHYSICS_FIELDS:custom.prop(self,field)
        custom.label(text='Mass/limits are representative where native particles vary.')
        custom.label(text='Elasticity/bounciness are approximations, not restitution.')
        custom.label(text='Settings-only edits retain native anchors and state graphs.')
        collision=layout.box();collision.label(text='Native Collision Mesh Selector')
        collision.prop(self,'inherit_colliders')
        selection=collision.column();selection.enabled=not self.inherit_colliders
        selection.template_list('DBH_UL_cloth_colliders','',self,'collider_items',self,'collider_index',rows=8)
        collision.label(text='Custom list applies to all simulation states (maximum 32).')
        collision.label(text='Meshes sharing one native cloth resource share this list.')
        collision.label(text='Named shapes without a matched mesh are selectable too.')
        collision.label(text='Uses native shapes; editing/adding arbitrary collision meshes is not supported.')
        world=layout.box();world.label(text='In-Game World Collision — Experimental')
        world.prop(self,'inherit_world_collisions')
        flag=world.column();flag.enabled=not self.inherit_world_collisions;flag.prop(self,'world_collisions')
        world.label(text='Exports the native flag and free-particle selection only.',icon='INFO')
        world.label(text='Scenery contacts require runtime world geometry; not yet verified.')
        world.label(text='World enabled: at most 31 model collision shapes.')
        wind=layout.box();wind.label(text='In-Game Wind Only');wind.prop(self,'wind_enabled')
        for field in WIND_FIELDS:wind.prop(self,field)
        wind.label(text='Environmental wind requires an active game wind source.')
        wind.label(text='Shared vanilla cloth-resource meshes share wind settings.')

    def execute(self,context):
        try:
            if not hasattr(self,'_metadata'):raise ValueError('Open the Cloth Physics & In-Game Wind window first')
            # Read current metadata so other operator changes are not overwritten.
            import json
            metadata=json.loads(self._text.as_string());key=slot_key(self._record,self._mesh)
            session=_DONOR_SESSIONS[self.donor_session]
            overrides={k:getattr(self,k) for k in PHYSICS_FIELDS if different(getattr(self,k),session['baseline'][k])}
            wind={k:getattr(self,k) for k in WIND_FIELDS}
            wind_changed=not self.wind_enabled or any(different(wind[k],session['snapshot']['wind'][k]) for k in WIND_FIELDS)
            selection=None if self.inherit_colliders else [i.native_name for i in self.collider_items if i.enabled]
            if selection is not None and len(selection)>32:raise ValueError('Select at most 32 native collision meshes')
            if not self.inherit_world_collisions and self.world_collisions and selection is not None and len(selection)>31:
                raise ValueError('World collision reserves mask bit 31; select at most 31 model colliders')
            metadata['slot_objects'][key]['cloth_physics']=dict(donor_asset=int(self.donor_asset),overrides=overrides,
                override_wind=wind_changed,wind_enabled=self.wind_enabled,wind=wind,colliders=selection,
                world_collision=None if self.inherit_world_collisions else self.world_collisions)
            write_metadata(self._text,metadata)
            self.report({'INFO'},'Per-mesh native profiles/wind saved. Export SEGS to apply in game.')
            return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}
        finally:release_donor_session(self)

    def cancel(self,context):
        release_donor_session(self)


class OBJECT_OT_dbh_cloth_reset_settings(bpy.types.Operator):
    bl_idname='object.dbh_cloth_reset_settings';bl_label='Reset to Native Settings';bl_options={'REGISTER','UNDO'}
    def execute(self,context):
        try:
            text,metadata=manifest_for_object(context.active_object);record,mesh=slot_for_object(context.active_object,metadata)
            metadata['slot_objects'][slot_key(record,mesh)].pop('cloth_physics',None)
            write_metadata(text,metadata);self.report({'INFO'},'Native profiles restored; cloth paint unchanged');return {'FINISHED'}
        except Exception as error:self.report({'ERROR'},str(error));return {'CANCELLED'}


CLASSES=(DBHClothColliderItem,DBH_UL_cloth_colliders,OBJECT_OT_dbh_cloth_settings,OBJECT_OT_dbh_cloth_reset_settings)
