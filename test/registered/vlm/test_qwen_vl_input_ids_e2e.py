"""E2E test: a Qwen VL request sent as ``input_ids`` runs the client's ids.

sglang-miles routes Qwen VL requests through ``legacy_load_mm_data``
(sglang#18781). That call did not pass the request's ``input_ids``, so the
exact-token block of ``process_and_combine_mm_data`` (SGLANG_MM_AVOID_RETOKENIZE,
sglang#26555) was skipped without a warning: the server decoded the client's
ids to text and re-tokenized them. A non-canonical prompt ("Describe" split
into "D" + "escribe") came back one token shorter, so the ids the model
sampled from were not the ids an RL trainer scores.

With the ids passed through, the prompt keeps its length: ``prompt_tokens``
equals ``len(input_ids) - 1 + N`` where N is the processor's token count for
the one image (the one placeholder expanded in place), and the ids the engine
ran, read back through ``return_logprob`` with ``logprob_start_len=0``, equal
the sent ids with the placeholder expanded. Modeled on
``test_token_id_retokenize_e2e.py``; the model is overridable with
``SGLANG_TEST_QWEN_VL_MODEL`` (a local Qwen3-VL / Qwen3.5+ checkpoint works).
"""

import base64
import io
import os
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

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

IMAGE_SIZE = (64, 64)
MODEL = os.environ.get("SGLANG_TEST_QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
IMAGE_TOKENS = "<|vision_start|><|image_pad|><|vision_end|>"


def _data_uri():
    img = Image.new("RGB", IMAGE_SIZE, (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_prompt(model):
    """A non-canonical prompt ("D" + "escribe" re-encodes to one token) with
    one image placeholder; the processor's token count for the image; the
    placeholder's id; and how many ids re-tokenization would lose."""
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    tok = processor.tokenizer

    def enc(text):
        return tok.encode(text, add_special_tokens=False)

    input_ids = enc("D") + enc("escribe") + enc(" the picture: ") + enc(IMAGE_TOKENS)
    canonical = enc(tok.decode(input_ids))
    drift = len(input_ids) - len(canonical)
    image_tokens = int(
        processor._get_num_multimodal_tokens(
            image_sizes=[(IMAGE_SIZE[1], IMAGE_SIZE[0])]
        ).num_image_tokens[0]
    )
    pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    return input_ids, image_tokens, pad_id, drift


def _generate(base_url, input_ids, image):
    resp = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "image_data": [image],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=300,
    )
    resp.raise_for_status()
    meta = resp.json()["meta_info"]
    ran = [item[1] for item in meta["input_token_logprobs"]]
    return meta["prompt_tokens"], ran


class TestQwenVLInputIds(CustomTestCase):
    other_args = ["--trust-remote-code", "--mem-fraction-static", "0.7"]

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env={"SGLANG_MM_AVOID_RETOKENIZE": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_the_engine_runs_the_client_ids_with_the_placeholder_expanded(self):
        input_ids, image_tokens, pad_id, drift = _build_prompt(MODEL)
        self.assertGreater(drift, 0, "the prompt must be non-canonical for the test to mean anything")
        prompt_tokens, ran = _generate(DEFAULT_URL_FOR_TEST, input_ids, _data_uri())

        at = input_ids.index(pad_id)
        expected = input_ids[:at] + [pad_id] * image_tokens + input_ids[at + 1 :]
        self.assertEqual(prompt_tokens, len(input_ids) - 1 + image_tokens)
        # input_token_logprobs may omit the first position, and it reports the
        # image positions with the multimodal item's pad value (0) instead of
        # the placeholder id: compare every text position id for id, and the
        # image positions by count and place.
        self.assertGreaterEqual(len(ran), len(expected) - 1)
        tail = expected[len(expected) - len(ran) :]
        text = [(i, t) for i, t in enumerate(tail) if t != pad_id]
        self.assertEqual([(i, ran[i]) for i, _ in text], text)
        image_positions = [i for i, t in enumerate(tail) if t == pad_id]
        self.assertEqual(len(image_positions), image_tokens)
        self.assertEqual(image_positions, list(range(image_positions[0], image_positions[0] + image_tokens)))

    def test_a_text_only_prompt_keeps_its_length(self):
        input_ids, _, _, _ = _build_prompt(MODEL)
        text_ids = input_ids[:-3]  # drop the vision triple
        resp = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={"input_ids": text_ids, "sampling_params": {"temperature": 0.0, "max_new_tokens": 1}},
            timeout=300,
        )
        resp.raise_for_status()
        self.assertEqual(resp.json()["meta_info"]["prompt_tokens"], len(text_ids))


if __name__ == "__main__":
    unittest.main()
