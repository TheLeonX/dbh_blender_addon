"""Reference-tool skeleton bridge; geometry is decoded by native.py."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector

from .xps import Bone
from .reference_archive import prepare_archive


DEFAULT_EXPORTER = r"C:\Users\ARDOR\Documents\ChatGPT\Cosplay\reference_tools\detroit\Detroit.exe"
DEFAULT_INDEX = ''  # Select an edition explicitly, or discover one installed game.



def _to_blender(value) -> tuple[float, float, float]:
    x, y, z = value
    return x, -z, y


def _from_blender(value) -> tuple[float, float, float]:
    return float(value.x), float(value.z), -float(value.y)


def _resolve_segs(path: Path) -> Path:
    """Accept either complete framing; only legacy metadata stubs need a sibling."""
    if path.suffix.lower() == ".segs":
        return path
    if path.suffix.lower() == '.data_container' or path.name.lower() == '_container':
        with path.open('rb') as stream:
            head = stream.read(8)
        if head == b'DC_INFO ':
            from .native import Container, Package
            raw = path.read_bytes()
            container = Container(raw, allow_trailing=True)
            if len(raw) == container.end and any(r.external for r in container.records):
                sibling = path.with_suffix('.segs')
                if sibling.is_file():
                    package = Package(sibling.read_bytes())
                    if package.members[0].unpacked != raw:
                        raise ValueError('Metadata-only DATA_CONTAINER does not match its sibling SEGS; no file imported')
                    return sibling
                raise ValueError('This DATA_CONTAINER contains metadata only, not mesh/shader streams. '
                                 'Select a complete DATA_CONTAINER or place its matching SEGS beside it.')
        return path
    raise ValueError('Select a .segs, .data_container or _container file')


def _run_reference_exporter(exe: Path, game_index: Path, segs: Path) -> tuple[str, Path, list[Path], Path]:
    if not exe.is_file():
        raise ValueError(f"Detroit.exe not found: {exe}")
    if not game_index.is_file():
        raise ValueError(f"BigFile_PC.idx not found: {game_index}")

    temporary = Path(tempfile.mkdtemp(prefix="dbh_blender_"))
    try:
        # The old executable cannot handle stored SEGS blocks. Give it a
        # disposable compressed-only archive built from the SELECTED package.
        # Installed game files and the selected source are never rewritten.
        code, reference_index = prepare_archive(segs, game_index, temporary)
        (temporary / "detroit.ini").write_text(str(reference_index) + "\n", encoding="utf-8")
        bone_names = exe.with_name("bonenames.txt")
        if bone_names.is_file():
            shutil.copy2(bone_names, temporary / bone_names.name)
        extraction = subprocess.run(
            [str(exe), code], cwd=temporary, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600, check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        models = sorted(
            (path for path in temporary.glob(f"{code}_*.ascii")
             if "_tan" not in path.name and "_skel" not in path.name),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        skeletons = list(temporary.glob(f"{code}_*_skel.ascii"))
        if extraction.returncode != 0 or not models:
            message = (extraction.stdout + "\n" + extraction.stderr).strip()
            raise ValueError(f"Detroit.exe could not export temporary package {code}: {message[-500:]}")
        return code, temporary, models, skeletons[0] if skeletons else Path()
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _create_armature(context, name: str, bones: tuple[Bone, ...]):
    if not bones:
        return None, []
    data = bpy.data.armatures.new(name + "_Armature")
    obj = bpy.data.objects.new(name + "_Armature", data)
    context.collection.objects.link(obj)
    context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")

    child_positions: dict[int, list[Vector]] = {}
    positions = [Vector(_to_blender(bone.position)) for bone in bones]
    for index, bone in enumerate(bones):
        if 0 <= bone.parent < len(bones):
            child_positions.setdefault(bone.parent, []).append(positions[index])
    edit_bones = []
    names = []
    for index, bone in enumerate(bones):
        edit = data.edit_bones.new(bone.name or f"Bone_{index}")
        edit.head = positions[index]
        children = child_positions.get(index, ())
        if children:
            direction = children[0] - edit.head
            edit.tail = children[0] if direction.length > 0.001 else edit.head + Vector((0, 0, 0.03))
        else:
            edit.tail = edit.head + Vector((0, 0, 0.03))
        edit_bones.append(edit)
        names.append(edit.name)
    for index, bone in enumerate(bones):
        if 0 <= bone.parent < len(edit_bones):
            edit_bones[index].parent = edit_bones[bone.parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    from .scene_organization import armature_display
    armature_display(obj)
    obj.select_set(False)
    return obj, names
