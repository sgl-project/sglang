import os
import warnings
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
            prompt = self.messages_to_gemini_input(s.messages_)
        else:
            # single-turn
            prompt = (
                self.text_to_gemini_input(s.text_, s.cur_images)
                if s.cur_images
                else s.text_
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
            prompt = self.messages_to_gemini_input(s.messages_)
        else:
            # single-turn
            prompt = (
                self.text_to_gemini_input(s.text_, s.cur_images)
                if s.cur_images
                else s.text_
            )
        generator = GenerativeModel(self.model_name).generate_content(
            prompt,
            stream=True,
            generation_config=GenerationConfig(**sampling_params.to_gemini_kwargs()),
        )
        for ret in generator:
            yield ret.text, {}

    def text_to_gemini_input(self, text, images):
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

    def messages_to_gemini_input(self, messages):
        gemini_message = []
        # from openai message format to gemini message format
        for msg in messages:
            if isinstance(msg["content"], str):
                text = msg["content"]
            else:
                text = msg["content"][0]["text"]

            if msg["role"] == "system":
                warnings.warn("Warning: system prompt is not supported in Gemini.")
                gemini_message.append({
                    "role": "user",
                    "parts": [{"text": "System prompt: " + text}],
                })
                gemini_message.append({
                    "role": "model",
                    "parts": [{"text": "Understood."}],
                })
                continue
            if msg["role"] == "user":
                gemini_msg = {
                    "role": "user",
                    "parts": [{"text": text}],
                }
            elif msg["role"] == "assistant":
                gemini_msg = {
                    "role": "model",
                    "parts": [{"text": text}],
                }

            # images
            if isinstance(msg["content"], list) and len(msg["content"]) > 1:
                for image in msg["content"][1:]:
                    assert image["type"] == "image_url"
                    gemini_msg["parts"].append(
                        {
                            "inline_data": {
                                "data": image["image_url"]["url"].split(",")[1],
                                "mime_type": "image/jpeg",
                            }
                        }
                    )

            gemini_message.append(gemini_msg)
        return gemini_message
