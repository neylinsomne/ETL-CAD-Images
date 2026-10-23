"""Lectura de anotaciones de CubiCasa5K y CVC-FP a un formato común.

Ambos datasets marcan el muro exterior a mano:
- CubiCasa5K: <g id="Wall" class="Wall External">; las ventanas/puertas cuelgan del muro.
- CVC-FP: la relación <relation type="outerP" objects="..."/> lista el perímetro exterior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.dom import minidom

from shapely.geometry import Polygon

from filtro_planos.geometry import largest_valid_polygon

_NUMBER_PAIR = re.compile(r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_FLOOR = re.compile(r"Floor-(\d+)")


@dataclass
class Element:
    element_id: str
    cls: str  # wall | railing | window | door | room | separation
    subtype: str  # clase original, p. ej. "Wall External", "Door Swing Beside", "Outdoor Terrace"
    polygon: Polygon
    floor: int = 1
    is_exterior: bool | None = None  # solo muros y vanos
    parent_id: str | None = None  # muro que contiene la ventana/puerta (CubiCasa)


@dataclass
class Relation:
    kind: str
    objects: list[str]


@dataclass
class Annotation:
    image_id: str
    dataset: str
    subset: str
    image_path: Path
    elements: list[Element]
    relations: list[Relation] = field(default_factory=list)


def parse_points(points: str) -> Polygon | None:
    return largest_valid_polygon([(float(x), float(y)) for x, y in _NUMBER_PAIR.findall(points)])


def _direct_polygon(node) -> Polygon | None:
    pol = next((c for c in node.childNodes if c.nodeName == "polygon"), None)
    return parse_points(pol.getAttribute("points")) if pol is not None else None


def _ancestor(node, predicate):
    node = node.parentNode
    while node is not None and node.nodeType == node.ELEMENT_NODE:
        if predicate(node):
            return node
        node = node.parentNode
    return None


def parse_cubicasa_svg(svg_path: Path, image_id: str) -> list[Element]:
    dom = minidom.parse(str(svg_path))
    elements: list[Element] = []
    wall_ids: dict[int, Element] = {}

    for g in dom.getElementsByTagName("g"):
        gid, gclass = g.getAttribute("id"), g.getAttribute("class")
        if gid in ("Wall", "Railing"):
            cls = gid.lower()
        elif gid in ("Window", "Door"):
            cls = gid.lower()
        elif gclass.startswith("Space "):
            cls = "room"
        else:
            continue
        polygon = _direct_polygon(g)
        if polygon is None:
            continue
        floor_node = _ancestor(g, lambda n: "Floorplan" in n.getAttribute("class"))
        floor_match = _FLOOR.search(floor_node.getAttribute("class")) if floor_node else None
        element = Element(
            element_id=f"{image_id}:{len(elements)}",
            cls=cls,
            subtype=gclass.removeprefix("Space ") if cls == "room" else gclass,
            polygon=polygon,
            floor=int(floor_match.group(1)) if floor_match else 1,
        )
        if cls == "wall":
            element.is_exterior = "External" in gclass.split()
            wall_ids[id(g)] = element
        elif cls in ("window", "door"):
            parent = _ancestor(g, lambda n: n.getAttribute("id") in ("Wall", "Railing"))
            parent_el = wall_ids.get(id(parent)) if parent is not None else None
            if parent_el is not None:
                element.parent_id = parent_el.element_id
                element.is_exterior = bool(parent_el.is_exterior)
        elements.append(element)
    return elements


def cubicasa_index(root: Path) -> list[tuple[Path, str]]:
    """(carpeta del plano, split) según train/val/test.txt de cubicasa5k/."""
    items = []
    for split in ("train", "val", "test"):
        for line in (root / f"{split}.txt").read_text().split():
            folder = root / line.strip("/")
            if (folder / "model.svg").exists():
                items.append((folder, split))
    return items


def cubicasa_image_id(folder: Path) -> str:
    return f"cubicasa5k/{folder.parent.name}/{folder.name}"


def load_cubicasa(folder: Path) -> Annotation:
    image_id = cubicasa_image_id(folder)
    return Annotation(
        image_id=image_id,
        dataset="cubicasa5k",
        subset=folder.parent.name,
        image_path=folder / "F1_scaled.png",  # las coordenadas del SVG están en este espacio
        elements=parse_cubicasa_svg(folder / "model.svg", image_id),
    )


_CVC_CLASSES = {
    "Wall": "wall",
    "Window": "window",
    "Door": "door",
    "Parking": "door",
    "Room": "room",
    "Separation": "separation",
}


def parse_cvc_svg(svg_path: Path, image_id: str) -> tuple[list[Element], list[Relation]]:
    dom = minidom.parse(str(svg_path))
    relations = [
        Relation(r.getAttribute("type"), [f"{image_id}:{o}" for o in r.getAttribute("objects").split(",") if o])
        for r in dom.getElementsByTagName("relation")
    ]
    exterior = {o for r in relations if r.kind == "outerP" for o in r.objects}
    elements: list[Element] = []
    for pol in dom.getElementsByTagName("polygon"):
        raw = pol.getAttribute("class")
        cls = _CVC_CLASSES.get(raw)
        polygon = parse_points(pol.getAttribute("points"))
        if cls is None or polygon is None:
            continue
        element_id = f"{image_id}:{pol.getAttribute('id')}"
        elements.append(
            Element(
                element_id=element_id,
                cls=cls,
                subtype=raw,
                polygon=polygon,
                is_exterior=(element_id in exterior) if cls in ("wall", "window", "door") else None,
            )
        )
    return elements, relations


def cvc_index(root: Path) -> list[tuple[Path, str]]:
    """(svg, split). CVC-FP no trae splits: todo va a "test" (evaluación externa)."""
    return [(svg, "test") for svg in sorted(root.glob("*_gt_*.svg"))]


def cvc_image_id(svg: Path) -> str:
    return f"cvc_fp/{svg.name.split('_gt_')[0]}"


def load_cvc(svg: Path) -> Annotation:
    stem = svg.name.split("_gt_")[0]
    image = next(p for p in svg.parent.glob(f"{stem}.*") if p.suffix.lower() in (".png", ".jpg"))
    image_id = cvc_image_id(svg)
    elements, relations = parse_cvc_svg(svg, image_id)
    return Annotation(image_id, "cvc_fp", "cvc_fp", image, elements, relations)
