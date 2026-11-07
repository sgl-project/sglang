"""E2E test: /v1/chat/completions with ``input_ids`` keeps the images.

A pre-tokenized client (the miles TITO session server) sends the prompt as
``input_ids`` with one image placeholder per image, plus the messages that
carry the images. Before this change the chat endpoint dropped the images
whenever ``input_ids`` was set (``image_data=None``), so the model never saw
the screenshot. Now the images reach the multimodal processor, which expands
each placeholder in place and keeps every other id verbatim
(SGLANG_MM_AVOID_RETOKENIZE, test_token_id_retokenize_e2e.py).

The check: for one non-canonical prompt ("Describe" split into "D" +
"escribe") plus one image, the chat endpoint's ``usage.prompt_tokens`` equals
the ``/generate`` path's ``prompt_tokens`` for the same ids and image, and
both equal ``len(input_ids) - 1 + N`` where N is the processor's token count
for the image. A text-only prompt through the chat endpoint stays at
``len(input_ids)``.
"""

import base64
import io
import unittest

import requests
from PIL import Image
from transformers import AutoProcessor

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-large")

IMAGE_SIZE = (64, 64)


def _data_uri():
    img = Image.new("RGB", IMAGE_SIZE, (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_prompt(model, image_token):
    """A non-canonical prompt ("D" + "escribe" re-encodes to one token) with
    one image placeholder, and the processor's expanded token count for the
    image."""
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    tok = processor.tokenizer

    def enc(text):
        return tok.encode(text, add_special_tokens=False)

    input_ids = enc("D") + enc("escribe") + enc(" the picture: ") + enc(image_token)
    image_tokens = int(
        processor._get_num_multimodal_tokens(
            image_sizes=[(IMAGE_SIZE[1], IMAGE_SIZE[0])]
        ).num_image_tokens[0]
    )
    return input_ids, image_tokens


def _generate_prompt_tokens(base_url, input_ids, image):
    resp = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "image_data": [image],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["meta_info"]["prompt_tokens"]


def _chat_prompt_tokens(base_url, input_ids, image):
    content = [{"type": "text", "text": "Describe the picture: "}]
    if image is not None:
        content.append({"type": "image_url", "image_url": {"url": image}})
    resp = requests.post(
        base_url + "/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": content}],
            "input_ids": input_ids,
            "max_tokens": 1,
            "temperature": 0.0,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["usage"]["prompt_tokens"]


class TestChatInputIdsKeepImages(CustomTestCase):
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    other_args = ["--trust-remote-code", "--mem-fraction-static", "0.7"]

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env={"SGLANG_MM_AVOID_RETOKENIZE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_chat_with_input_ids_sees_the_image(self):
        input_ids, image_tokens = _build_prompt(self.model, self.image_token)
        image = _data_uri()
        expected = len(input_ids) - 1 + image_tokens

        via_generate = _generate_prompt_tokens(DEFAULT_URL_FOR_TEST, input_ids, image)
        via_chat = _chat_prompt_tokens(DEFAULT_URL_FOR_TEST, input_ids, image)
        self.assertEqual(via_generate, expected, "the /generate path is the reference")
        self.assertEqual(
            via_chat, expected, f"chat dropped or re-tokenized the image: {via_chat}"
        )

    def test_chat_with_input_ids_and_no_image_is_unchanged(self):
        input_ids, _ = _build_prompt(self.model, "")
        self.assertEqual(
            _chat_prompt_tokens(DEFAULT_URL_FOR_TEST, input_ids, None), len(input_ids)
        )


if __name__ == "__main__":
    unittest.main()
