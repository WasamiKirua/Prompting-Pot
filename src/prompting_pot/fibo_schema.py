"""Native Fibo schema and token processing, vendored from the supplied analyzer.

The mask rasterization helper is omitted to avoid an unused NumPy dependency.
"""

from collections.abc import Iterator
from typing import Literal

from pydantic import BaseModel, Field

#: Attributes that can hold nested scene objects.
_NESTED_LIST_FIELDS = ("children", "clothing", "footwear", "accessories")
_NESTED_SINGLE_FIELDS = ("eyes", "hair", "facial_hair")


def _iter_nested(object, seen=None):
    """Yield `object` and every scene object nested below it, parent first."""
    if object is None:
        return
    if seen is None:
        seen = set()
    if id(object) in seen:
        return
    seen.add(id(object))
    yield object
    for field in _NESTED_LIST_FIELDS:
        for item in getattr(object, field, None) or []:
            yield from _iter_nested(item, seen)
    for field in _NESTED_SINGLE_FIELDS:
        yield from _iter_nested(getattr(object, field, None), seen)


def _format_coord_token(value: float) -> str:
    return f"<coord_{round(float(value), 1):.1f}>"


def _parse_coord_token(value: float | str) -> float:
    if isinstance(value, str) and value.startswith("<coord_") and value.endswith(">"):
        return float(value[len("<coord_") : -1])
    return float(value)


def _format_color_token(value: int) -> int | str:
    if value < 0 or value > 255:
        raise ValueError("Color channels must be between 0 and 255")
    return f"<color_{value}>"


def _parse_color_token(value: int | str) -> int:
    if isinstance(value, str) and value.startswith("<color_") and value.endswith(">"):
        return int(value[len("<color_") : -1])
    return int(value)


class Color(BaseModel):
    name: str | None = Field(None)
    red: int | str = Field(...)
    green: int | str = Field(...)
    blue: int | str = Field(...)

    def post_process(self):
        if isinstance(self.red, str):
            self.red = _parse_color_token(self.red)
        if isinstance(self.green, str):
            self.green = _parse_color_token(self.green)
        if isinstance(self.blue, str):
            self.blue = _parse_color_token(self.blue)


class Point(BaseModel):
    x: float | str = Field(...)
    y: float | str = Field(...)

    def post_process(self):
        if isinstance(self.x, str):
            self.x = _parse_coord_token(self.x)
        if isinstance(self.y, str):
            self.y = _parse_coord_token(self.y)


class BoundingBox(BaseModel):
    top_left: Point = Field(...)
    bottom_right: Point = Field(...)

    def post_process(self):
        self.top_left.post_process()
        self.bottom_right.post_process()


class BodyKeypoint(BaseModel):
    name: str = Field(..., description="The name of the keypoint")
    point: Point = Field(...)

    def post_process(self):
        self.point.post_process()


class ObjectMask(BaseModel):
    polygons: list[list[float | str]] | None = Field(None)

    def post_process(self):
        if self.polygons is not None:
            self.polygons = [
                [_parse_coord_token(value) for value in ring] for ring in self.polygons
            ]


class Relationship(BaseModel):
    to: str = Field(...)
    relationship: str = Field(...)


class BaseObject(BaseModel):
    name: str = Field(
        "",
        description="Short description of the object",
    )
    kind: Literal["base_object", "object", "person", "animal", "text"] = "base_object"
    description: str | None = Field(
        None,
        description="A more detailed description of the object",
    )
    location: str | None = Field(None)
    position: Literal["foreground", "midground", "background"] | None = Field(None)
    relative_size: Literal["small", "medium", "large"] | None = Field(None)
    bounding_boxes: list[BoundingBox] | None = Field(None)
    num_objects: int | None = Field(None)
    relative_distance: float | None = Field(
        None,
        ge=0,
        le=100,
        description="The relative distance of the object from the camera, 0 is at the camera, 100 is at the far edge of the image, None if not applicable",
    )
    colors: list[Color] | None = Field(None)
    blur: str | None = Field(
        None,
        description="Description of the blurring of the object, if any",
    )
    details: list[str] | None = Field(
        None,
        description="Additional details about the object",
    )
    relationships: list[Relationship] | None = Field(None)
    mask: ObjectMask | None = Field(None)
    artifacts: list[str] | None = Field(
        None,
        description="Visual issues with the object",
    )

    def make_full_mask(self):
        self.location = None
        self.position = None
        self.relative_size = None

    def make_full(self):
        self.make_full_mask()
        self.mask = None

    def make_long(self):
        self.make_full()
        self.add_location()
        self.add_position()
        self.add_relative_size()
        self.bounding_boxes = None
        self.relative_distance = None
        self.mask = None

    def make_short(self):
        self.make_long()
        self.relationships = None
        self.details = None
        self.colors = None
        self.blur = None
        self.artifacts = None

    def drop_masks(self):
        """Clear `mask` on this object and every object nested below it."""
        for object in _iter_nested(self):
            object.mask = None

    def _keep_only(self, fields_to_keep: set[str]) -> None:
        for field_name in type(self).model_fields:
            if field_name not in fields_to_keep:
                setattr(self, field_name, None)

    def make_for_detect(self):
        self._keep_only({"name", "kind", "bounding_boxes"})

    def make_for_mask(self):
        self._keep_only({"name", "kind", "bounding_boxes", "mask"})

    def make_for_keypoints(self):
        self._keep_only({"name", "kind", "keypoints", "bounding_boxes"})

    def _box_extents(self) -> list[tuple[float, float, float, float]]:
        """(left, top, right, bottom) of each bounding box, in 0-100 coordinates."""
        extents = []
        for bounding_box in self.bounding_boxes or []:
            left, right = sorted(
                (
                    _parse_coord_token(bounding_box.top_left.x),
                    _parse_coord_token(bounding_box.bottom_right.x),
                )
            )
            top, bottom = sorted(
                (
                    _parse_coord_token(bounding_box.top_left.y),
                    _parse_coord_token(bounding_box.bottom_right.y),
                )
            )
            extents.append((left, top, right, bottom))
        return extents

    def add_location(self):
        """Set `location` from the joint extent of the bounding boxes, on a grid of thirds."""
        extents = self._box_extents()
        if not extents:
            return self
        left = min(extent[0] for extent in extents)
        top = min(extent[1] for extent in extents)
        right = max(extent[2] for extent in extents)
        bottom = max(extent[3] for extent in extents)
        if left <= 1 and top <= 1 and right >= 99 and bottom >= 99:
            self.location = "full frame"
            return self

        third, two_thirds, edge = 100 / 3, 200 / 3, 1.0

        def classify(low: float, high: float) -> str:
            # "full" is an object that spans the axis, such as sky or ground.
            if low < third and high > two_thirds and high - low > 50:
                return "full"
            center = (low + high) / 2
            if center < third:
                return "low"
            if center > two_thirds:
                return "high"
            return "mid"

        horizontal, vertical = classify(left, right), classify(top, bottom)
        h_name = {"low": "left", "mid": "center", "high": "right"}.get(horizontal)
        v_name = {"low": "top", "mid": "middle", "high": "bottom"}.get(vertical)
        if horizontal == "full" and vertical == "full":
            location = "center"
        elif horizontal == "full":
            location = {"low": "top", "high": "bottom", "mid": "center"}[vertical]
        elif vertical == "full":
            location = {"low": "left", "high": "right", "mid": "center"}[horizontal]
        elif horizontal == "low" and vertical == "low" and left <= edge and top <= edge:
            location = "top-left corner"
        elif (
            horizontal == "high"
            and vertical == "low"
            and right >= 100 - edge
            and top <= edge
        ):
            location = "top-right corner"
        elif (
            horizontal == "low"
            and vertical == "high"
            and left <= edge
            and bottom >= 100 - edge
        ):
            location = "bottom-left corner"
        elif (
            horizontal == "high"
            and vertical == "high"
            and right >= 100 - edge
            and bottom >= 100 - edge
        ):
            location = "bottom-right corner"
        elif horizontal == "mid" and vertical == "mid":
            location = "center"
        elif vertical == "mid":
            location = f"{h_name} of center"
        elif horizontal == "mid":
            location = f"{v_name} center"
        else:
            location = f"{v_name} {h_name}"
        self.location = location
        return self

    def add_position(self):
        """Set `position` from `relative_distance`, in thirds of the depth."""
        if self.relative_distance is None:
            return self
        if self.relative_distance < 100 / 3:
            self.position = "foreground"
        elif self.relative_distance < 200 / 3:
            self.position = "midground"
        else:
            self.position = "background"
        return self

    def add_relative_size(self):
        """Set `relative_size` from the summed area of the bounding boxes."""
        extents = self._box_extents()
        if not extents:
            return self
        ratio = sum(
            (right - left) * (bottom - top) for left, top, right, bottom in extents
        ) / (100.0 * 100.0)
        if ratio < 0.05:
            self.relative_size = "small"
        elif ratio < 0.25:
            self.relative_size = "medium"
        else:
            self.relative_size = "large"
        return self

    def post_process(self):
        if self.bounding_boxes is not None:
            for bbox in self.bounding_boxes:
                bbox.post_process()
        if self.colors is not None:
            for color in self.colors:
                color.post_process()
        if self.mask is not None:
            self.mask.post_process()


class Object(BaseObject):
    kind: Literal["object"] = "object"
    children: list["SceneObject"] | None = Field(None)
    material: str | None = Field(None)
    texture: str | None = Field(None)
    transparency: float | None = Field(
        None,
        ge=0,
        le=100,
        description="The transparency of the object, 0 is fully transparent, 100 is fully opaque",
    )

    def make_long(self):
        super().make_long()
        self.children = None

    def make_short(self):
        super().make_short()
        self.material = None
        self.texture = None
        self.transparency = None

    def post_process(self):
        super().post_process()
        if self.children is not None:
            for child in self.children:
                child.post_process()


class TextLine(BaseObject):
    text: str = Field(...)


class Text(BaseObject):
    kind: Literal["text"] = "text"
    language: str | None = Field(None, pattern=r"^[A-Za-z]{2,3}(-[A-Za-z]{2})?$")
    lines: list[TextLine] = Field(...)
    font_family: str | None = Field(None)
    font_description: str | None = Field(None)
    font_weight: (
        Literal["100", "200", "300", "400", "500", "600", "700", "800", "900"] | None
    ) = Field(None)
    font_style: Literal["normal", "italic", "oblique"] | None = Field(None)
    text_case: (
        Literal["lowercase", "uppercase", "title", "sentence", "mixed"] | None
    ) = Field(None)
    text_orientation: Literal["horizontal", "vertical"] | None = Field(None)
    text_alignment: Literal["left", "center", "right", "justify"] | None = Field(None)
    text_decoration: (
        Literal["none", "underline", "line-through", "overline", "shadow"] | None
    ) = Field(None)
    rotation: float | None = Field(None, ge=-180, le=180)

    def make_long(self):
        super().make_long()
        for line in self.lines:
            line.make_long()

    def make_short(self):
        super().make_short()
        self.language = None
        self.font_family = None
        self.font_description = None
        self.font_weight = None
        self.font_style = None
        self.text_case = None
        self.text_orientation = None
        self.text_alignment = None
        self.text_decoration = None
        self.rotation = None
        for line in self.lines:
            line.make_short()

    def post_process(self):
        super().post_process()
        for line in self.lines:
            line.post_process()


class Person(BaseObject):
    kind: Literal["person"] = "person"
    age: str | None = Field(None)
    gender: str | None = Field(None)
    ethnicity: str | None = Field(None)
    skin_color: Color | None = Field(None)
    keypoints: list[BodyKeypoint] | None = Field(None)
    activity: str | None = Field(None)
    face_bbox: BoundingBox | None = Field(None)
    expression: str | None = Field(None)
    eyes: Object | None = Field(None, description="Description of the eyes")
    mouth: str | None = Field(None, description="Description of the mouth")
    nose: str | None = Field(None, description="Description of the nose")
    hair: Object | None = Field(None, description="Description of the hair")
    ears: str | None = Field(None, description="Description of the ears")
    facial_hair: Object | None = Field(
        None,
        description="Description of the facial hair",
    )
    body: str | None = Field(None)
    left_hand: str | None = Field(None)
    right_hand: str | None = Field(None)
    left_leg: str | None = Field(None)
    right_leg: str | None = Field(None)
    clothing: list[Object] | None = Field(None)
    footwear: list[Object] | None = Field(None)
    accessories: list[Object] | None = Field(None)
    children: list[Object | Text] | None = Field(None)

    def make_long(self):
        super().make_long()
        self.face_bbox = None
        self.keypoints = None
        self.children = None

    def make_short(self):
        super().make_short()
        self.skin_color = None
        self.activity = None
        self.expression = None
        self.eyes = None
        self.mouth = None
        self.nose = None
        self.hair = None
        self.ears = None
        self.facial_hair = None
        self.body = None
        self.left_hand = None
        self.right_hand = None
        self.left_leg = None
        self.right_leg = None
        self.clothing = None
        self.footwear = None
        self.accessories = None
        self.face_bbox = None
        self.children = None

    def post_process(self):
        super().post_process()
        if self.keypoints is not None:
            for keypoint in self.keypoints:
                keypoint.post_process()
        if self.face_bbox is not None:
            self.face_bbox.post_process()
        if self.skin_color is not None:
            self.skin_color.post_process()
        if self.facial_hair is not None:
            self.facial_hair.post_process()
        if self.hair is not None:
            self.hair.post_process()
        if self.eyes is not None:
            self.eyes.post_process()
        if self.clothing is not None:
            for clothing in self.clothing:
                clothing.post_process()
        if self.footwear is not None:
            for footwear in self.footwear:
                footwear.post_process()
        if self.accessories is not None:
            for accessory in self.accessories:
                accessory.post_process()
        if self.children is not None:
            for child in self.children:
                child.post_process()


class Animal(BaseObject):
    kind: Literal["animal"] = "animal"
    children: list[Object | Text] | None = Field(None)
    head: str | None = Field(None)
    body: str | None = Field(None)
    legs: str | None = Field(None)
    hands: str | None = Field(None)
    tail: str | None = Field(None)
    wings: str | None = Field(None)

    def make_long(self):
        super().make_long()

    def make_short(self):
        super().make_short()
        self.head = None
        self.body = None
        self.legs = None
        self.hands = None
        self.tail = None
        self.wings = None

    def post_process(self):
        super().post_process()
        if self.children is not None:
            for child in self.children:
                child.post_process()


SceneObject = Object | Person | Animal | Text | BaseObject

Object.model_rebuild()


class SceneObjectList(BaseModel):
    objects: list[SceneObject] = Field(...)

    def post_process(self):
        if self.objects is not None:
            for object in self.objects:
                object.post_process()


class GeneralDescription(BaseModel):
    user_prompt: str | None = Field(None, description="The user's prompt for the image")
    image_type: str | None = Field(None, description="The type of the image")
    style: str | None = Field(None, description="The style of the image")
    medium: str | None = Field(None)
    color_palette: list[Color] | None = Field(None)
    aesthetic_score: float | None = Field(
        None, ge=0, le=10, description="The aesthetic score of the image"
    )
    atmosphere: str | None = Field(
        None, description="The overall feeling or emotion the image evokes"
    )
    setting: str | None = Field(
        None, description="The environment or context where the image takes place"
    )
    lighting: str | None = Field(None)
    composition: str | None = Field(None)
    depth_of_field: str | None = Field(None)
    viewpoint: str | None = Field(None)
    intent: str | None = Field(None, description="The purpose of the image")
    description: str | None = Field(
        None, description="A short and comprehensive description of the image"
    )
    artifacts: list[str] | None = Field(
        None, description="Visual discrepancies or anomalies in the image"
    )
    camera_model: str | None = Field(None)
    lens_model: str | None = Field(None)
    focal_length: float | None = Field(None)
    aperture: float | None = Field(None)
    shutter_speed: float | None = Field(None)
    iso: int | None = Field(None)
    f_number: float | None = Field(None)
    exposure_time_ms: float | None = Field(None)

    def post_process(self):
        if self.color_palette is not None:
            for color in self.color_palette:
                color.post_process()
        return self


class SceneDescription(BaseModel):
    general: GeneralDescription = Field(...)
    objects: list[SceneObject] | None = Field(None)

    def iter_objects(self) -> Iterator[SceneObject]:
        """Yield every object in the `children` tree, parent first. Wearables stay behind."""

        def walk(objects, seen):
            for object in objects or []:
                if id(object) in seen:
                    continue
                seen.add(id(object))
                yield object
                yield from walk(getattr(object, "children", None), seen)

        yield from walk(self.objects, set())

    def iter_all_objects(self) -> Iterator[SceneObject]:
        """Yield every object anywhere in the tree, wearables and face parts included."""
        seen: set[int] = set()
        for object in self.objects or []:
            yield from _iter_nested(object, seen)

    def _hoisted_wearables(self) -> list[SceneObject]:
        """Wearables with a box, renamed "<item> on <person>", as standalone targets."""
        hoisted = []
        for object in self.iter_objects():
            if object.kind != "person":
                continue
            for field in ("clothing", "footwear", "accessories"):
                for item in getattr(object, field, None) or []:
                    if not item.name or not item.bounding_boxes:
                        continue
                    item.name = f"{item.name} on {object.name}"
                    hoisted.append(item)
        return hoisted

    def make_for_detect(self):
        all_objects = list(self.iter_objects()) + self._hoisted_wearables()
        for object in all_objects:
            object.make_for_detect()
        self.objects = all_objects

    def make_for_mask(self):
        all_objects = list(self.iter_objects()) + self._hoisted_wearables()
        for object in all_objects:
            object.make_for_mask()
        self.objects = all_objects

    def make_for_keypoints(self):
        people = [
            object
            for object in self.iter_objects()
            if object.kind == "person" and object.keypoints is not None
        ]
        for object in people:
            object.make_for_keypoints()
        self.objects = people

    def make_general(self):
        self.objects = None

    def make_long(self):
        every_object = list(self.iter_all_objects())
        flattened = list(self.iter_objects())
        for object in every_object:
            object.make_long()
        self.objects = flattened

    def make_short(self):
        new_objects = []
        for object in self.iter_objects():
            new_objects.append(object)
        for object in new_objects:
            object.make_short()
        self.objects = new_objects

    def make_full_mask(self):
        for object in self.iter_all_objects():
            object.make_full_mask()

    def make_full(self):
        for object in self.iter_all_objects():
            object.make_full()

    def post_process(self):
        if self.general is not None:
            self.general.post_process()
        for object in self.objects or []:
            object.post_process()


for _model in (
    BaseObject,
    Object,
    Person,
    Animal,
    TextLine,
    Text,
    SceneObjectList,
    GeneralDescription,
    SceneDescription,
):
    _model.model_rebuild(force=True)
