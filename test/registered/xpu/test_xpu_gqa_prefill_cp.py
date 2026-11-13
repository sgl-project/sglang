"""Zigzag prefill context parallelism for dense GQA attention on Intel XPU.

With --enable-prefill-cp each CP rank holds two zigzag segments of a prompt's
tokens; intel_xpu writes the all-gathered chunk K/V to its pool and runs each
segment against the full cache with that segment's own lengths. Pass keys at the
start, middle and end of a long prompt sit in different segments, so a wrong
segment length or a missed gather returns the wrong key.
"""

import random
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_xpu_ci(est_time=600, suite="nightly-kernel-main-test-2-gpu-xpu", nightly=True)

_XPU_DEVICE_COUNT = torch.xpu.device_count() if torch.xpu.is_available() else 0
_SKIP_REASON = f"Prefill CP needs 2 XPUs, found {_XPU_DEVICE_COUNT}"

FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. " * 5
# Paragraph index of each key in a 36-paragraph prompt (~3000 tokens).
KEY_POSITIONS = {"first": 1, "middle": 18, "last": 34}


def _prompt(keys: dict[str, int], asked: str) -> str:
    paragraphs = [FILLER] * 36
    for name, index in KEY_POSITIONS.items():
        paragraphs[index] = (
            f"The pass key for the {name} door is {keys[name]}. Remember it. " + FILLER
        )
    return (
        "There is important info hidden inside a lot of irrelevant text. "
        "Find it and memorize it.\n"
        + "\n".join(paragraphs)
        + f"\nWhat is the pass key for the {asked} door? "
        f"The pass key for the {asked} door is"
    )


@unittest.skipUnless(_XPU_DEVICE_COUNT >= 2 or is_in_ci(), _SKIP_REASON)
class TestXpuGqaPrefillCp(CustomTestCase):
    model = "Qwen/Qwen3-0.6B"

    @classmethod
    def setUpClass(cls):
        if _XPU_DEVICE_COUNT < 2:
            raise RuntimeError(_SKIP_REASON)
        cls.process = None
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--device",
                "xpu",
                "--attention-backend",
                "intel_xpu",
                "--tp-size",
                "2",
                "--attn-cp-size",
                "2",
                "--enable-prefill-cp",
                "--cp-strategy",
                "zigzag",
                "--disable-radix-cache",
                # ~3K-token prompts span 3 chunks: later chunks attend over a
                # prefix written by earlier CP chunks.
                "--chunked-prefill-size",
                "1024",
                "--mem-fraction-static",
                "0.60",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_keys_in_every_zigzag_segment_are_retrieved(self):
        rng = random.Random(0)
        for asked in KEY_POSITIONS:
            with self.subTest(door=asked):
                keys = {name: rng.randint(10000, 99999) for name in KEY_POSITIONS}
                response = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": _prompt(keys, asked),
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    },
                    timeout=300,
                )
                self.assertEqual(response.status_code, 200, response.text)
                out = response.json()
                self.assertGreater(out["meta_info"]["prompt_tokens"], 2000)
                self.assertIn(str(keys[asked]), out["text"])


if __name__ == "__main__":
    unittest.main()
