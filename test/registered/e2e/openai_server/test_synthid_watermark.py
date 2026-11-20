import json

import requests
import torch

from sglang.srt.sampling.watermarking.synthid.algorithm import (
    SynthIDConfig,
    compute_g_values_torch,
)
from sglang.srt.sampling.watermarking.synthid.state import SynthIDStatePool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.synthid_utils import SYNTHID_TEST_KEYS
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-small")

_CONFIG = SynthIDConfig(keys=SYNTHID_TEST_KEYS)
# Each prompt is over 64 tokens, so a 64-token chunked prefill splits it.
_PROMPTS = [
    f"{prompt} Write for a general audience that enjoys detail. Use concrete "
    "names, places, sounds, and colors; avoid lists and headings; vary the "
    "length of your sentences; and keep a calm, curious voice from start to "
    "finish, as if talking to a friend on a long evening walk through a town "
    "you both know well."
    for prompt in [
        "Write a long, varied field guide to urban trees.",
        "Tell a long story about a lighthouse keeper and a storm.",
        "Explain in detail how bread is made, from wheat to loaf.",
        "Describe a walking tour of an old harbor city.",
        "Imagine a week aboard a cargo ship crossing the Pacific.",
        "Recount the history of a small mountain village over a century.",
        "Narrate a day in the life of a beekeeper in late summer.",
        "Sketch the plans of two friends restoring an old sailboat.",
    ]
]


def _g_values(output_ids):
    """The g-values of one output at its watermarked positions, each the mean
    over the keys, replayed from a fresh slot as the server seeds a request."""
    pool = SynthIDStatePool(_CONFIG, 1, "cpu")
    slots = torch.zeros(1, dtype=torch.int64)
    keys = torch.tensor(_CONFIG.keys)
    pool.seed(slots, [[]])
    g_values = []
    for token_id in output_ids:
        context_hash, watermarked = pool.current(slots)
        if watermarked.item():
            g = compute_g_values_torch(context_hash, torch.tensor([[token_id]]), keys)
            g_values.append(g.float().mean().item())
        pool.commit(slots, torch.tensor([token_id]))
    return g_values


def _z_score(g_values):
    """Without the watermark each of a position's per-key g-values is a fair
    coin, so the mean over n positions has standard deviation
    0.5 / sqrt(n * len(keys)); a z-score of 4 is a 3e-5 false positive."""
    mean = sum(g_values) / len(g_values)
    return (mean - 0.5) * 2 * (len(g_values) * len(_CONFIG.keys)) ** 0.5


class SynthIDServerTest(CustomTestCase):
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    extra_args = []
    process = None

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-watermark",
                "--watermark-algorithm",
                "synthid",
                "--watermark-config",
                json.dumps({"keys": list(_CONFIG.keys)}),
                *cls.extra_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)

    def generate(self, enabled, *, top_p=0.95, max_new_tokens=256):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": [_PROMPTS[i % len(_PROMPTS)] for i in range(len(enabled))],
                "sampling_params": [
                    {
                        "temperature": 1.0,
                        "top_p": top_p,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                        "watermark": {"enabled": on},
                    }
                    for on in enabled
                ],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def assert_watermarked(self, outputs, enabled):
        for on in set(enabled):
            g_values = [
                g
                for output, row_on in zip(outputs, enabled, strict=True)
                if row_on == on
                for g in _g_values(output["output_ids"])
            ]
            z_score = _z_score(g_values)
            print(f"watermark {on}: z-score {z_score:.1f}")
            if on:
                self.assertGreater(z_score, 4)
            else:
                self.assertLess(abs(z_score), 4)


class TestSynthIDWatermark(SynthIDServerTest):
    # Without the radix cache, a prompt sent twice is chunked both times.
    extra_args = ["--chunked-prefill-size", "64", "--disable-radix-cache"]

    def test_mixed_batch_with_chunked_prefill(self):
        # Every prompt is chunked, and the watermarked rows use top-p.
        enabled = [True, False] * 4
        outputs = self.generate(enabled)
        self.assert_watermarked(outputs, enabled)
        for output, on in zip(outputs, enabled, strict=True):
            if on:
                z_score = _z_score(_g_values(output["output_ids"]))
                self.assertGreater(z_score, 4)

    def test_all_watermarked_batch(self):
        # Every row watermarked, with no request filter.
        enabled = [True] * 8
        self.assert_watermarked(self.generate(enabled, top_p=1.0), enabled)


class TestSynthIDWatermarkEagle3(SynthIDServerTest):
    # This draft's vocabulary (32000) is smaller than the target's, so its
    # proposal is scattered onto the target vocabulary before verification.
    model = DEFAULT_TARGET_MODEL_EAGLE3
    extra_args = [
        "--dtype",
        "float16",
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        DEFAULT_DRAFT_MODEL_EAGLE3,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-use-rejection-sampling",
        "--mem-fraction-static",
        "0.7",
    ]

    def test_verify_watermarks_enabled_requests_and_accepts_drafts(self):
        # This target's text has low entropy, so it takes more tokens to score.
        enabled = [True, False] * 8
        outputs = self.generate(enabled, max_new_tokens=512)
        self.assert_watermarked(outputs, enabled)
        for output, on in zip(outputs, enabled, strict=True):
            meta_info = output["meta_info"]
            accept_length = meta_info["completion_tokens"] / meta_info["spec_verify_ct"]
            print(f"watermark {on}: accept length {accept_length:.2f}")
            self.assertGreater(accept_length, 1.0)


if __name__ == "__main__":
    import unittest

    unittest.main()
