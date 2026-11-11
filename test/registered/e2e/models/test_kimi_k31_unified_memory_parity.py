"""Kimi-K3.1 on the unified memory pool matches the static pools.

Sparse attention reads FP8 latent rows and indexer keys that the unified pool
lays out per page of its full sub-pool, beside the KDA state. Kimi-K3.1 cut to
its first eight layers (real weights) runs the same requests on both pools: a
chunked prefill, decoding, and a second request that reuses the first one's
prefix, so the indexer scores cached keys. Prompt and output logprobs must
agree.
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=1800, stage="nightly", runner_config="8-gpu-b300")

MODEL_PATH = "moonshotai/Kimi-K3.1-Pro"
# The checkpoint's first eight layers, numbered from 1 as its config numbers
# them: two MLA+DSA layers among KDA layers. Everything else is its own config.
FULL_LAYERS = [4, 8]
# Loading reads every checkpoint shard, though only eight layers are kept.
SERVER_LAUNCH_TIMEOUT = 3600
KDA_LAYERS = [1, 2, 3, 5, 6, 7]
BASE_ARGS = [
    "--trust-remote-code",
    "--skip-tokenizer-init",
    "--tp-size",
    "2",
    "--random-seed",
    "1",
    "--chunked-prefill-size",
    "256",
    "--max-running-requests",
    "4",
    "--mem-fraction-static",
    "0.6",
]
PREFIX = [1] + [100 + (i * 37) % 5000 for i in range(700)]


def _model_override(model_path):
    """The checkpoint's config with only the kept layers."""
    from sglang.srt.hf_transformers.config import get_config

    text = get_config(model_path, trust_remote_code=True).text_config
    linear = dict(text.linear_attn_config)
    linear.update(full_attn_layers=FULL_LAYERS, kda_layers=KDA_LAYERS)
    dsa = dict(text.dsa_config)
    dsa.update(dsa_layers=FULL_LAYERS)
    return {
        "text_config": {
            "num_hidden_layers": len(FULL_LAYERS) + len(KDA_LAYERS),
            "linear_attn_config": linear,
            "dsa_config": dsa,
        }
    }


class TestKimiK31UnifiedMemoryParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.base_args = BASE_ARGS + [
            "--json-model-override-args",
            json.dumps(_model_override(MODEL_PATH)),
        ]

    def _generate(self, input_ids, logprob_start_len=0):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "logprob_start_len": logprob_start_len,
            },
            timeout=300,
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        return (
            [lp for lp, _, _ in meta["input_token_logprobs"] if lp is not None],
            [(lp, tid) for lp, tid, _ in meta["output_token_logprobs"]],
            meta.get("cached_tokens", 0),
        )

    def _run(self, extra_args):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=self.base_args + extra_args,
        )
        try:
            first = self._generate(PREFIX)
            # The same prefix again, then new tokens: a cache hit whose new
            # tokens score the cached indexer keys.
            # Prompt logprobs only past the prefix: asking for them from the
            # start would recompute the prefix instead of reusing it.
            second = self._generate(
                PREFIX + [200 + i for i in range(64)], logprob_start_len=len(PREFIX)
            )
            return first, second
        finally:
            kill_process_tree(process.pid, wait_timeout=60)

    def _assert_close(self, unified, static, what):
        unified_input, unified_output, _ = unified
        static_input, static_output, _ = static
        self.assertEqual(len(unified_input), len(static_input), what)
        for i, (u, s) in enumerate(zip(unified_input, static_input)):
            self.assertAlmostEqual(u, s, delta=0.05, msg=f"{what}: prompt token {i}")
        self.assertEqual(
            [tid for _, tid in unified_output],
            [tid for _, tid in static_output],
            f"{what}: greedy tokens diverged",
        )
        for (u, _), (s, _) in zip(unified_output, static_output):
            self.assertAlmostEqual(u, s, delta=0.05, msg=f"{what}: output")

    def test_unified_matches_static(self):
        static = self._run([])
        for name, extra in (
            ("cuda graph", []),
            ("eager", ["--disable-cuda-graph"]),
        ):
            with self.subTest(mode=name):
                unified = self._run(["--enable-unified-memory", *extra])
                self.assertGreater(unified[1][2], 0, "the second request hit no prefix")
                self._assert_close(unified[0], static[0], f"{name}, first request")
                self._assert_close(unified[1], static[1], f"{name}, prefix hit")


if __name__ == "__main__":
    unittest.main()
