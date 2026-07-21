"""E2E test for SGLANG_MM_AVOID_RETOKENIZE on the pre-tokenized VLM path.

A client may send a multimodal request as input_ids (list[int]) instead of text.
On that path the server decodes the ids back to text and the HF processor
re-tokenizes them. If the original ids were non-canonical (decode -> re-encode is
not identity), that re-tokenization drifts: the reported prompt_tokens changes.

With SGLANG_MM_AVOID_RETOKENIZE ON (default), the server keeps the user's
original tokens verbatim and only expands the image placeholder, so prompt_tokens
stays faithful to what the client sent.

For each model we launch a real server twice with the same predefined,
non-canonical prompt ("Describe" split into "D"+"escribe") plus one image:

  * flag OFF -> the prompt re-tokenizes (drift): prompt_tokens shrinks by the
    drift delta.
  * flag ON  -> no drift: prompt_tokens equals the original length (with the
    image placeholder expanded).
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

register_cuda_ci(est_time=83, stage="base-b", runner_config="1-gpu-large")


def _data_uri():
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_drift_prompt(model, image_token):
    """Return (input_ids, drift_delta).

    input_ids is a predefined non-canonical prompt: "Describe" is split into
    "D"+"escribe" (decodes to the same text but re-encodes to the single merged
    token), followed by one image placeholder. drift_delta is how many extra
    tokens the non-canonical form carries vs. the canonical re-tokenization.
    """
    tok = AutoProcessor.from_pretrained(
        model, trust_remote_code=True, use_fast=True
    ).tokenizer

    def enc(text):
        return tok.encode(text, add_special_tokens=False)

    input_ids = enc("D") + enc("escribe") + enc(" the picture: ") + enc(image_token)
    canonical = enc(tok.decode(input_ids))
    drift_delta = len(input_ids) - len(canonical)
    return input_ids, drift_delta


def _prompt_tokens(base_url, input_ids, image):
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


class TestQwenVLTokenIdRetokenize(CustomTestCase):
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    other_args = ["--trust-remote-code", "--mem-fraction-static", "0.7"]

    def test_flag_off_drifts_flag_on_does_not(self):
        input_ids, drift_delta = _build_drift_prompt(self.model, self.image_token)
        self.assertGreater(drift_delta, 0, "prompt is canonical; no drift to exercise")
        image = _data_uri()

        prompt_tokens = {}
        for flag in ("0", "1"):
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=self.other_args,
                env={"SGLANG_MM_AVOID_RETOKENIZE": flag},
            )
            try:
                prompt_tokens[flag] = _prompt_tokens(
                    DEFAULT_URL_FOR_TEST, input_ids, image
                )
            finally:
                kill_process_tree(process.pid)

        # ON keeps the user's original tokens; OFF loses the drift_delta tokens.
        pt_off, pt_on = prompt_tokens["0"], prompt_tokens["1"]
        self.assertEqual(pt_on - pt_off, drift_delta, f"on={pt_on}, off={pt_off}")


if __name__ == "__main__":
    unittest.main()
