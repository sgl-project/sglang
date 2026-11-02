# Copyright 2026 The SGLang team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Config classes for MiniCPM-OCR 1.0.

The checkpoint reuses the MiniCPM-V 4.6 stack with ``downsample_mode`` ``"4x"``
and keeps every vision layer. ``model_type`` strings are the checkpoint's own,
unversioned values; the class and module names carry the release number.
"""

from typing import Any

from sglang.srt.configs.minicpmv4_6 import MiniCPMV4_6Config, MiniCPMV4_6VisionConfig


class MiniCPMOCR1_0VisionConfig(MiniCPMV4_6VisionConfig):
    model_type = "minicpm_ocr_vision"


class MiniCPMOCR1_0Config(MiniCPMV4_6Config):
    model_type = "minicpm_ocr"
    sub_configs = {
        "vision_config": MiniCPMOCR1_0VisionConfig,
        "text_config": MiniCPMV4_6Config.sub_configs["text_config"],
    }

    def __init__(self, downsample_mode: str = "4x", **kwargs: Any) -> None:
        super().__init__(downsample_mode=downsample_mode, **kwargs)
        self.drop_vision_last_layer = False


__all__ = ["MiniCPMOCR1_0Config", "MiniCPMOCR1_0VisionConfig"]
