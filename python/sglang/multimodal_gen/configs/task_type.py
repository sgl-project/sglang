# SPDX-License-Identifier: Apache-2.0
"""Shared task and output types for pipeline capabilities and requests."""

from enum import Enum, auto


class DataType(Enum):
    IMAGE = auto()
    VIDEO = auto()
    MESH = auto()
    ACTION = auto()

    def get_default_extension(self) -> str:
        if self == DataType.IMAGE:
            return "png"
        if self == DataType.VIDEO:
            return "mp4"
        if self == DataType.ACTION:
            return "json"
        return "glb"


class ModelTaskType(Enum):
    # TODO: check if I2V/TI2V models can work w/wo text

    I2V = auto()  # Image to Video
    T2V = auto()  # Text to Video
    TI2V = auto()  # Text and Image to Video

    T2I = auto()  # Text to Image
    I2I = auto()  # Image to Image
    TI2I = auto()  # Image to Image or Text-Image to Image
    I2M = auto()  # Image to Mesh
    VLA_ACTION = auto()  # Vision-language-action policy output

    V2V = auto()  # Video to Video
    F2V = auto()  # Video-prefix-conditioned Video

    @classmethod
    def parse(cls, value: "ModelTaskType | str") -> "ModelTaskType":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls[value.strip().upper()]
            except KeyError:
                pass
        raise ValueError(
            f"Unknown task_type {value!r}; choose from {[task.name for task in cls]}"
        )

    def requires_video_input(self) -> bool:
        return self in (ModelTaskType.V2V, ModelTaskType.F2V)

    def accepts_video_input(self) -> bool:
        return self.requires_video_input()

    def is_image_gen(self) -> bool:
        return (
            self == ModelTaskType.T2I
            or self == ModelTaskType.I2I
            or self == ModelTaskType.TI2I
        )

    def is_action_gen(self) -> bool:
        return self == ModelTaskType.VLA_ACTION

    def is_mesh_gen(self) -> bool:
        return self == ModelTaskType.I2M

    def is_video_gen(self) -> bool:
        return (
            self == ModelTaskType.I2V
            or self == ModelTaskType.T2V
            or self == ModelTaskType.TI2V
            or self == ModelTaskType.V2V
            or self == ModelTaskType.F2V
        )

    def is_visual_gen(self) -> bool:
        return self.is_image_gen() or self.is_video_gen()

    def requires_image_input(self) -> bool:
        return (
            self == ModelTaskType.I2V
            or self == ModelTaskType.I2I
            or self == ModelTaskType.I2M
        )

    def accepts_image_input(self) -> bool:
        return (
            self == ModelTaskType.I2V
            or self == ModelTaskType.I2I
            or self == ModelTaskType.TI2I
            or self == ModelTaskType.TI2V
            or self == ModelTaskType.I2M
            or self == ModelTaskType.VLA_ACTION
        )

    def data_type(self) -> DataType:
        if self.is_action_gen():
            return DataType.ACTION
        if self.is_mesh_gen():
            return DataType.MESH
        if self.is_image_gen():
            return DataType.IMAGE
        return DataType.VIDEO


def get_request_task_type(batch, pipeline_config) -> ModelTaskType:
    """Read the admitted task, falling back for legacy direct stage callers."""
    return ModelTaskType.parse(
        getattr(batch, "task_type", None) or pipeline_config.task_type
    )
