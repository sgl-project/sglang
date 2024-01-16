import os
from typing import List, Optional, Union

import numpy as np
from sglang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams

try:
    import vertexai
    from vertexai.preview.generative_models import (
        GenerationConfig,
        GenerativeModel,
        Image,
    )
except ImportError as e:
    GenerativeModel = e

GEMINI_MODEL_NAMES = [
    "gemini-pro",
    "gemini-pro-vision",
]


class Gemini(BaseBackend):
    def __init__(self, model_name):
        super().__init__()

        if isinstance(GenerativeModel, Exception):
            raise GenerativeModel

        project_id = os.environ["GCP_PROJECT_ID"]
        location = os.environ["GCP_LOCATION"]
        vertexai.init(project=project_id, location=location)

        self.name = "gemini"
        self.model_name = model_name
        self.chat_template = get_chat_template("default")

    def get_chat_template(self):
        return self.chat_template

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if s.messages_:
            prompt = s.messages_
        else:
            # single-turn
            prompt = (
                self.to_gemini_input(s.text_, s.cur_images) if s.cur_images else s.text_
            )
        ret = GenerativeModel(self.model_name).generate_content(
            prompt,
            generation_config=GenerationConfig(**sampling_params.to_gemini_kwargs()),
        )

        comp = ret.text

        return comp, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        if s.messages_:
            prompt = s.messages_
        else:
            # single-turn
            prompt = (
                self.to_gemini_input(s.text_, s.cur_images) if s.cur_images else s.text_
            )
        generator = GenerativeModel(self.model_name).generate_content(
            prompt,
            stream=True,
            generation_config=GenerationConfig(**sampling_params.to_gemini_kwargs()),
        )
        for ret in generator:
            yield ret.text, {}

    def to_gemini_input(self, text, images):
        input = []
        # split with image token
        text_segs = text.split(self.chat_template.image_token)
        for image_path, image_base64_data in images:
            text_seg = text_segs.pop(0)
            if text_seg != "":
                input.append(text_seg)
            input.append(Image.from_bytes(image_base64_data))
        text_seg = text_segs.pop(0)
        if text_seg != "":
            input.append(text_seg)
        return input
