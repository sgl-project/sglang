"""E2E test for SGLANG_MM_AVOID_RETOKENIZE on Kimi-K2.5 (pre-tokenized VLM path).

Same check as test_token_id_retokenize_e2e.py (Qwen), but for the Kimi-K2.5
processor (python/sglang/srt/multimodal/processors/kimi_k25.py). Kimi-K2.5 is a
large MoE model, so this is an 8-GPU nightly test.

A predefined non-canonical prompt ("Describe" -> "D"+"escribe") plus one image is
sent as input_ids. With SGLANG_MM_AVOID_RETOKENIZE OFF the prompt re-tokenizes
(prompt_tokens shrinks by the drift delta); with it ON the original tokens are
kept verbatim and only the image placeholder is expanded.
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
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=3600, suite="nightly-8-gpu-h200", nightly=True)

KIMI_MODEL = "moonshotai/Kimi-K2.5"
KIMI_IMAGE_TOKEN = "<|media_pad|>"
SERVER_LAUNCH_TIMEOUT = 3600


def _data_uri():
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_drift_prompt(model, image_token):
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


class TestKimiK25TokenIdRetokenize(CustomTestCase):
    model = KIMI_MODEL
    other_args = [
        "--tp",
        "8",
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--chunked-prefill-size",
        "131072",
    ]

    def test_flag_off_drifts_flag_on_does_not(self):
        input_ids, drift_delta = _build_drift_prompt(self.model, KIMI_IMAGE_TOKEN)
        self.assertGreater(
            drift_delta, 0, "prompt is canonical; cannot exercise retokenize drift"
        )
        image = _data_uri()

        prompt_tokens = {}
        for flag in ("0", "1"):
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=self.other_args,
                env={"SGLANG_MM_AVOID_RETOKENIZE": flag},
            )
            try:
                prompt_tokens[flag] = _prompt_tokens(
                    DEFAULT_URL_FOR_TEST, input_ids, image
                )
            finally:
                kill_process_tree(process.pid)

        pt_off, pt_on = prompt_tokens["0"], prompt_tokens["1"]
        self.assertNotEqual(pt_on, pt_off, "flag had no effect on prompt_tokens")
        self.assertEqual(
            pt_on - pt_off,
            drift_delta,
            f"expected ON to keep {drift_delta} extra tokens vs OFF "
            f"(on={pt_on}, off={pt_off})",
        )


if __name__ == "__main__":
    unittest.main()
