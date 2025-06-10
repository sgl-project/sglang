import concurrent
import concurrent.futures
import dataclasses
import multiprocessing as mp
import os
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessorFast

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.utils import encode_video, load_audio, load_image


class MultimodalInputFormat(Enum):
    """Enum for different multimodal input formats."""

    RAW_IMAGES = "raw_images"
    PRECOMPUTED_FEATURES = "precomputed_features"
    PIXEL_VALUES = "pixel_values"


@dataclasses.dataclass
class BaseMultiModalProcessorOutput:
    # input_text, with each frame of video/image represented with a image_token
    input_text: str

    # frames loaded from image and video, in given order
    images: Optional[list[Union[Image.Image, dict]]] = None

    # audios
    audios: Optional[list[Union[np.ndarray, dict]]] = None

    def normalize(self):
        for field_name in ["images", "audios"]:
            field = getattr(self, field_name, None)
            if field is not None and isinstance(field, list) and len(field) == 0:
                setattr(self, field_name, None)


@dataclasses.dataclass
class MultimodalSpecialTokens:
    image_token: Optional[Union[int, str, List[str]]] = None
    video_token: Optional[Union[int, str, List[str]]] = None
    audio_token: Optional[Union[int, str, List[str]]] = None

    def convert_to_str(self, token: Union[str, int], processor) -> str:
        if token is None:
            return token
        if isinstance(token, str):
            return token
        return processor.tokenizer.convert_ids_to_tokens([token])[0]

    def convert_to_strs(self, processor):
        self.image_token = self.convert_to_str(self.image_token, processor)
        self.video_token = self.convert_to_str(self.video_token, processor)
        self.audio_token = self.convert_to_str(self.audio_token, processor)

    image_token_regex: Optional[re.Pattern] = None
    video_token_regex: Optional[re.Pattern] = None
    audio_token_regex: Optional[re.Pattern] = None

    def __post_init__(self):
        if self.image_token_regex is None and self.image_token is not None:
            self.image_token_regex = re.compile(re.escape(self.image_token))
        if self.video_token_regex is None and self.video_token is not None:
            self.video_token_regex = re.compile(re.escape(self.video_token))
        if self.audio_token_regex is None and self.audio_token is not None:
            self.audio_token_regex = re.compile(re.escape(self.audio_token))

    def collect(self) -> re.Pattern:
        tokens = [
            self.image_token_regex,
            self.video_token_regex,
            self.audio_token_regex,
        ]
        patterns = []
        flags = 0
        for t in tokens:
            if t is not None:
                patterns.append(t.pattern)
                flags |= t.flags
        combined = "(" + "|".join(f"(?:{p})" for p in patterns) + ")"
        return re.compile(combined, flags)
