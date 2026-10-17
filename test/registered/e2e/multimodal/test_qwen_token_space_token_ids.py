"""Qwen token-ID preservation through a real /generate server.

    [D][escribe] + [image] -> startup-enabled token-space server -> prompt_tokens
                                 |                    |
                 SGLANG_MM_AVOID_RETOKENIZE=0 or 1      |
                                 +--------------------+
                                   both equal len(ids) - 1 + image_tokens

The client confirms decode/encode would merge [D][escribe] into [Describe].
Both servers enable token-space preprocessing, preserve the original IDs and
only expand the image token.
This test loads model weights; CPU component coverage lives in
unit/multimodal/test_qwen_tokenized_media.py.
"""

import base64
import io
import unittest

import requests
from PIL import Image
from transformers import AutoProcessor

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=92, stage="base-b", runner_config="1-gpu-large")
register_cpu_ci(est_time=123, suite="stage-b-test-cpu-intel")


def _data_uri():
    img = Image.new("RGB", (64, 64), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _build_drift_prompt(tokenizer, image_token):
    """Return original IDs and the token count lost by decode/encode."""

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = enc("D") + enc("escribe") + enc(" the picture: ") + enc(image_token)
    canonical = enc(tokenizer.decode(input_ids))
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


class TestQwenVLTokenSpaceTokenIds(CustomTestCase):
    model = "Qwen/Qwen2.5-VL-3B-Instruct"
    image_token = "<|vision_start|><|image_pad|><|vision_end|>"
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.7",
        "--enable-token-space-processor",
    ]

    def test_input_ids_are_preserved_under_both_legacy_flag_values(self):
        processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        input_ids, drift_delta = _build_drift_prompt(
            processor.tokenizer, self.image_token
        )
        self.assertGreater(drift_delta, 0, "prompt is canonical; no drift to exercise")
        image = _data_uri()
        counts = processor._get_num_multimodal_tokens(image_sizes=[(64, 64)])
        expected = len(input_ids) - 1 + int(counts.num_image_tokens[0])

        for flag in ("0", "1"):
            process = popen_launch_server(
                self.model,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=self.other_args,
                env={"SGLANG_MM_AVOID_RETOKENIZE": flag},
            )
            try:
                prompt_tokens = _prompt_tokens(DEFAULT_URL_FOR_TEST, input_ids, image)
            finally:
                kill_process_tree(process.pid)
            self.assertEqual(prompt_tokens, expected, f"legacy flag={flag}")


if __name__ == "__main__":
    unittest.main()
