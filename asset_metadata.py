"""Scene-owned animation records: datablock pointers survive Action renames.

New Actions have no Detroit ID properties. Old saved scenes are readable and
their known Detroit fields are migrated when edited; unrelated fields stay.
"""
import json
import bpy
from bpy.props import PointerProperty, CollectionProperty, StringProperty

KEYS = ('dbh_animation_space','dbh_armature_name','dbh_native_frame_count',
        'dbh_native_timestep','dbh_native_source_text','dbh_anim_asset_id',
        'dbh_native_unmapped_tracks','dbh_graph_labels','dbh_graph_source')


class DBHActionRecord(bpy.types.PropertyGroup):
    action: PointerProperty(type=bpy.types.Action)
    payload: StringProperty(options={'HIDDEN'})


def record(action, create=False):
    current = bpy.context.scene
    scenes = [current] + [s for s in bpy.data.scenes if s != current]
    for scene in scenes:
        for item in getattr(scene,'dbh_action_records',()):
            if item.action == action: return item
    if create:
        item = current.dbh_action_records.add(); item.action = action
        return item


def values(action):
    item = record(action)
    result = {k:action[k] for k in KEYS if k in action}
    if item and item.payload: result.update(json.loads(item.payload))
    return result


def get(action, key, default=None):
    return values(action).get(key, default)


def update(action, data):
    if set(data)-set(KEYS): raise ValueError('Unknown Detroit animation fields')
    result = values(action); result.update(data)
    encoded = json.dumps(result, ensure_ascii=True)
    item = record(action, True); item.payload = encoded
    for key in KEYS:
        if key in action: del action[key]


def register():
    bpy.utils.register_class(DBHActionRecord)
    bpy.types.Scene.dbh_action_records = CollectionProperty(type=DBHActionRecord)


def unregister():
    del bpy.types.Scene.dbh_action_records
    bpy.utils.unregister_class(DBHActionRecord)
