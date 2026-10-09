"""Ideogram 4 caption schema and canonical serialization.

Based on https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md
We require all three root fields for rich image captions; the official format
allows high_level_description and style_description to be omitted. No Ideogram
runtime or external service is needed to validate these captions.
"""

import json
from typing import Any

from jsonschema import Draft202012Validator

TEXT = {"type": "string", "minLength": 1}
BBOX = {
    "type": "array",
    "items": {"type": "integer", "minimum": 0, "maximum": 1000},
    "minItems": 4,
    "maxItems": 4,
}


def palette(max_colors: int) -> dict[str, Any]:
    """Uppercase #RRGGBB colors, as required by the official verifier."""
    return {
        "type": "array",
        "maxItems": max_colors,
        "items": {"type": "string", "pattern": "^#[0-9A-F]{6}$"},
    }


def object_schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Construct a closed object schema while preserving property order."""
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


PHOTO_STYLE = object_schema(
    {
        "aesthetics": TEXT,
        "lighting": TEXT,
        "photo": TEXT,
        "medium": {"const": "photograph"},
        "color_palette": palette(16),
    },
    ["aesthetics", "lighting", "photo", "medium"],
)
ART_STYLE = object_schema(
    {
        "aesthetics": TEXT,
        "lighting": TEXT,
        "medium": TEXT,
        "art_style": TEXT,
        "color_palette": palette(16),
    },
    ["aesthetics", "lighting", "medium", "art_style"],
)
OBJECT_ELEMENT = object_schema(
    {
        "type": {"const": "obj"},
        "bbox": BBOX,
        "desc": TEXT,
        "color_palette": palette(5),
    },
    ["type", "desc"],
)
TEXT_ELEMENT = object_schema(
    {
        "type": {"const": "text"},
        "bbox": BBOX,
        "text": TEXT,
        "desc": TEXT,
        "color_palette": palette(5),
    },
    ["type", "text", "desc"],
)

IDEOGRAM_CAPTION_SCHEMA = object_schema(
    {
        "high_level_description": TEXT,
        "style_description": {"anyOf": [PHOTO_STYLE, ART_STYLE]},
        "compositional_deconstruction": object_schema(
            {
                "background": TEXT,
                "elements": {
                    "type": "array",
                    "items": {"anyOf": [OBJECT_ELEMENT, TEXT_ELEMENT]},
                },
            },
            ["background", "elements"],
        ),
    },
    ["high_level_description", "style_description", "compositional_deconstruction"],
)

VALIDATOR = Draft202012Validator(IDEOGRAM_CAPTION_SCHEMA)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject ambiguous JSON instead of silently overwriting repeated fields."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def serialize_ideogram_caption(raw: str) -> str:
    """Validate structure/geometry and serialize in the model's trained key order.

    This validates format, not visual accuracy. Do not invent or repair invalid
    geometry; let the batch report the image as failed instead.
    """
    try:
        caption = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except ValueError as exc:
        raise ValueError("Ideogram caption is not valid, unambiguous JSON.") from exc
    error = next(VALIDATOR.iter_errors(caption), None)
    if error is not None:
        path = ".".join(str(part) for part in error.absolute_path) or "root"
        raise ValueError(f"Invalid Ideogram caption at {path}: {error.message}")
    style = caption["style_description"]
    if "art_style" in style and style["medium"] == "photograph":
        raise ValueError("Photographs must use photo, not art_style.")
    style_order = (
        tuple(PHOTO_STYLE["properties"])
        if "photo" in style
        else tuple(ART_STYLE["properties"])
    )
    ordered_style = {key: style[key] for key in style_order if key in style}
    composition = caption["compositional_deconstruction"]
    ordered_elements = []
    for element in composition["elements"]:
        if "bbox" in element:
            ymin, xmin, ymax, xmax = element["bbox"]
            if ymin > ymax or xmin > xmax:
                raise ValueError(
                    "Ideogram bbox must be [y_min, x_min, y_max, x_max] with ordered bounds."
                )
        order = (
            tuple(TEXT_ELEMENT["properties"])
            if element["type"] == "text"
            else tuple(OBJECT_ELEMENT["properties"])
        )
        ordered_elements.append({key: element[key] for key in order if key in element})
    ordered = {
        "high_level_description": caption["high_level_description"],
        "style_description": ordered_style,
        "compositional_deconstruction": {
            "background": composition["background"],
            "elements": ordered_elements,
        },
    }
    return json.dumps(
        ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
