"""MOSS recaption JSON, distinct from its experimental layout prompt schema."""

import json
from typing import Any

from jsonschema import Draft202012Validator

MOSS_CAPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["caption", "watermark", "borders"],
    "properties": {
        "caption": {
            "type": "object",
            "additionalProperties": False,
            "required": ["long", "medium", "short"],
            "properties": {
                key: {"type": "string", "minLength": 1}
                for key in ("long", "medium", "short")
            },
        },
        "watermark": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["box_2d", "label"],
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                    },
                    "label": {"type": "string", "minLength": 1},
                },
            },
        },
        "borders": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"enum": ["none", "left", "right", "up", "down"]},
        },
    },
}
_VALIDATOR = Draft202012Validator(MOSS_CAPTION_SCHEMA)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate MOSS JSON key: {key}")
        result[key] = value
    return result


def serialize_moss_caption(text: str) -> str:
    """Validate fields/geometry without normalizing or translating visible text."""
    try:
        data = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as exc:
        raise ValueError("MOSS returned invalid recaption JSON.") from exc
    if next(_VALIDATOR.iter_errors(data), None) is not None:
        raise ValueError("MOSS recaption JSON does not match the required schema.")
    if any(not value.strip() for value in data["caption"].values()):
        raise ValueError("All MOSS caption lengths must be nonempty.")
    if "none" in data["borders"] and len(data["borders"]) != 1:
        raise ValueError("MOSS borders cannot combine none with directions.")
    for watermark in data["watermark"]:
        y_min, x_min, y_max, x_max = watermark["box_2d"]
        if y_min > y_max or x_min > x_max or not watermark["label"].strip():
            raise ValueError("Invalid MOSS watermark box or label.")
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
