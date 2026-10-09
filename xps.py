"""Reader for daemon1's XNALara/XPS ASCII interchange files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Bone:
    name: str
    parent: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class Vertex:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    color: tuple[int, int, int, int]
    uvs: tuple[tuple[float, float], ...]
    bone_indices: tuple[int, ...]
    bone_weights: tuple[float, ...]


@dataclass(frozen=True)
class Mesh:
    name: str
    textures: tuple[tuple[str, int], ...]
    vertices: tuple[Vertex, ...]
    faces: tuple[tuple[int, int, int], ...]
    uv_layers: int


class Lines:
    def __init__(self, path: Path):
        self.values = path.read_text(encoding="utf-8-sig").splitlines()
        self.index = 0

    def line(self) -> str:
        if self.index >= len(self.values):
            raise ValueError("unexpected EOF in XPS ASCII file")
        value = self.values[self.index].strip()
        self.index += 1
        return value

    def peek(self) -> str:
        if self.index >= len(self.values):
            raise ValueError("unexpected EOF in XPS ASCII file")
        return self.values[self.index].strip()

    def integer(self) -> int:
        return int(self.line().split()[0])


def _read_bones(source: Lines, count: int) -> tuple[Bone, ...]:
    bones: list[Bone] = []
    for _ in range(count):
        name = source.line()
        parent = source.integer()
        values = tuple(map(float, source.line().split()))
        if len(values) < 3:
            raise ValueError(f"bone {name!r} has no position")
        rotation = values[3:7] if len(values) >= 7 else (0.0, 0.0, 0.0, 1.0)
        bones.append(Bone(name, parent, values[:3], rotation))
    return tuple(bones)


def read_skeleton(path: Path) -> tuple[Bone, ...]:
    source = Lines(path)
    bones = _read_bones(source, source.integer())
    if source.index != len(source.values):
        raise ValueError("unexpected trailing lines in XPS skeleton")
    return bones


def read_model(path: Path) -> tuple[Mesh, ...]:
    source = Lines(path)
    embedded_bones = source.integer()
    _read_bones(source, embedded_bones)
    mesh_count = source.integer()
    meshes: list[Mesh] = []
    for _ in range(mesh_count):
        name = source.line()
        uv_layers = source.integer()
        textures = tuple((source.line(), source.integer()) for _ in range(source.integer()))
        vertices: list[Vertex] = []
        for _ in range(source.integer()):
            position = tuple(map(float, source.line().split()))
            normal = tuple(map(float, source.line().split()))
            color = tuple(map(int, source.line().split()))
            uvs = tuple(tuple(map(float, source.line().split())) for _ in range(uv_layers))
            indices: tuple[int, ...] = ()
            weights: tuple[float, ...] = ()
            # Detroit.exe leaves the model bone count at zero because the
            # skeleton is a separate file, but still writes 4/8 skin values.
            if len(source.peek().split()) > 3:
                indices = tuple(map(int, source.line().split()))
                weights = tuple(map(float, source.line().split()))
            vertices.append(Vertex(position, normal, color, uvs, indices, weights))
        faces = tuple(tuple(map(int, source.line().split())) for _ in range(source.integer()))
        meshes.append(Mesh(name, textures, tuple(vertices), faces, uv_layers))
    if source.index != len(source.values):
        raise ValueError(f"{len(source.values) - source.index} unconsumed lines in {path.name}")
    return tuple(meshes)
