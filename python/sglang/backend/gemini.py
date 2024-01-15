import os
from typing import List, Optional, Union

import numpy as np
from sglang.backend.base_backend import BaseBackend
from sglang.lang.chat_template import get_chat_template
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SamplingParams

try:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
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

        # project_id = os.environ["GCP_PROJECT_ID"]
        # location = os.environ["GCP_LOCATION"]
        project_id = "alpa-373720"
        location = "us-central1"
        vertexai.init(project=project_id, location=location)

        self.model_name = model_name
        self.chat_template = get_chat_template("gemini")
        # self.chat = GenerativeModel(model_name).start_chat()

    def get_chat_template(self):
        return self.chat_template

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SamplingParams,
    ):
        if self.model_name in ["gemini-pro"]:
            prompt = s.text_
        elif self.model_name in ["gemini-pro-vision"]:
            # TODO: add image support
            prompt = s.text_
        ret = GenerativeModel(self.model_name).generate_content(
            prompt, generation_config=GenerationConfig(**sampling_params.to_gemini_kwargs())
        )

        comp = ret.text

        return comp, {}

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SamplingParams,
    ):
        if self.model_name in ["gemini-pro"]:
            prompt = s.text_
        elif self.model_name in ["gemini-pro-vision"]:
            # TODO: add image support
            prompt = s.text_
        generator = GenerativeModel(self.model_name).generate_content(
            prompt,
            stream=True,
            generation_config=GenerationConfig(**sampling_params.to_gemini_kwargs()),
        )
        for ret in generator:
            yield ret.text, {}
