"""
Usage:
python3 -m unittest test_bnb.TestVisionModel.test_vlm
python3 -m unittest test_bnb.TestLanguageModel.test_mmlu
"""

import base64
import io
import json
import multiprocessing as mp
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import openai
import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

VISION_MODELS = [
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
]
LANGUAGE_MODELS = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-7B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
]

# image
IMAGE_MAN_IRONING_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
IMAGE_SGL_LOGO_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png"

# video
VIDEO_JOBS_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/videos/jobs_presenting_ipod.mp4"

# audio
AUDIO_TRUMP_SPEECH_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3"
AUDIO_BIRD_SONG_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/bird_song.mp3"


def popen_launch_server_wrapper(base_url, model, other_args):
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


class TestVisionModel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.base_url += "/v1"
        cls.api_key = "sk-123456"

    def _run_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a very short sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        # `driver` is for gemma-3-it
        assert "man" in text or "person" or "driver" in text, text
        assert "cab" in text or "taxi" in text or "SUV" in text, text
        # MiniCPMO fails to recognize `iron`, but `hanging`
        assert "iron" in text or "hang" in text, text
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def _run_multi_turn_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a very short sentence.",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "There is a man at the back of a yellow cab ironing his clothes.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Repeat your previous answer."}
                    ],
                },
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert "man" in text or "cab" in text, text
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def _run_multi_images_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
                            "modalities": "multi-images",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_SGL_LOGO_URL},
                            "modalities": "multi-images",
                        },
                        {
                            "type": "text",
                            "text": "I have two very different images. They are not related at all. "
                            "Please describe the first image in one sentence, and then describe the second image in another sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        print("-" * 30)
        print(f"Multi images response:\n{text}")
        print("-" * 30)
        assert "man" in text or "cab" in text or "SUV" in text or "taxi" in text, text
        assert "logo" in text or '"S"' in text or "SG" in text, text
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def run_decode_with_image(self, image_id):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        content = []
        if image_id == 0:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_MAN_IRONING_URL},
                }
            )
        elif image_id == 1:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_SGL_LOGO_URL},
                }
            )
        else:
            pass

        content.append(
            {
                "type": "text",
                "text": "Describe this image in a very short sentence.",
            }
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": content},
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)

    def _run_test_mixed_batch(self):
        image_ids = [0, 1, 2] * 4
        with ThreadPoolExecutor(4) as executor:
            list(executor.map(self.run_decode_with_image, image_ids))

    def test_vlm(self):
        models_to_test = VISION_MODELS

        if is_in_ci():
            models_to_test = [random.choice(VISION_MODELS)]

        for model in models_to_test:
            with self.subTest(model=model):
                other_args = [
                    "--mem-fraction-static",
                    "0.6",
                    "--load-format",
                    "bitsandbytes",
                ]
                try:
                    process = popen_launch_server_wrapper(
                        DEFAULT_URL_FOR_TEST, model, other_args
                    )
                    self._run_test_mixed_batch()
                    self._run_multi_images_chat_completion()
                    self._run_multi_turn_chat_completion()
                    self._run_single_image_chat_completion()
                finally:
                    kill_process_tree(process.pid)


class TestLanguageModel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        # cls.base_url += "/v1"
        cls.api_key = "sk-123456"

    def test_mmlu(self):
        models_to_test = LANGUAGE_MODELS

        if is_in_ci():
            models_to_test = [random.choice(LANGUAGE_MODELS)]

        for model in models_to_test:
            with self.subTest(model=model):
                other_args = [
                    "--mem-fraction-static",
                    "0.6",
                    "--load-format",
                    "bitsandbytes",
                ]
                try:
                    process = popen_launch_server_wrapper(
                        DEFAULT_URL_FOR_TEST, model, other_args
                    )
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mmlu",
                        num_examples=32,
                        num_threads=16,
                    )

                    metrics = run_eval(args)
                    print(f"{metrics=}")
                    self.assertGreater(metrics["score"], 0.3)
                finally:
                    kill_process_tree(process.pid)
