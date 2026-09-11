"""Deterministic inference on the unified memory pool (virtual-id KV).

`--enable-deterministic-inference` routes Triton extend through
`TritonAttnBackend._forward_extend_unified`, a 1-stage kernel that reads BOTH
the prefix and the extend half of the KV out of the pool -- unlike the default
2-stage path, which takes the extend half from its `k`/`v` arguments. On the
unified memory pool `forward_batch.out_cache_loc` holds VIRTUAL ids, so feeding
it to that kernel untranslated read the prefix at physical ids and the extend
tokens at virtual ones. The mismatch produced garbage logits, surfacing as
`NaN detected! sampler: next_token_logits` once CI arms
`SGLANG_ENABLE_ASYNC_ASSERT` and as silent corruption without it.

Needs all three of unified memory + Triton attention + deterministic inference;
any two are clean, which is why no existing test covered it.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=71, stage="base-b", runner_config="1-gpu-large")

KIMI_LINEAR_MODEL = "yujiepan/kimi-linear-tiny-random"
SERVER_ENV = {"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0"}
# Triton is pinned on purpose here: the defect lives in the Triton deterministic
# extend kernel, so resolving to another backend would not exercise it.
BASE_ARGS = [
    "--trust-remote-code",
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--enable-deterministic-inference",
    "--attention-backend",
    "triton",
    "--linear-attn-backend",
    "triton",
    "--mamba-backend",
    "triton",
    "--max-mamba-cache-size",
    "32",
    "--max-total-tokens",
    "4096",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]


class TestUnifiedMemoryDeterministicParity(CustomTestCase):
    """Unified pool must match the static pool token-for-token. A multi-token
    prompt is required: the defect only shows once an extend batch writes tokens
    that the same forward then reads back through `out_cache_loc`."""

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_LINEAR_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def _generate(cls):
        response = requests.post(
            cls.base_url + "/generate",
            json={
                "input_ids": [1] + [100 + i % 1000 for i in range(256)],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["meta_info"]["output_token_logprobs"]

    def _run(self, extra_args):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=BASE_ARGS + extra_args,
            env=SERVER_ENV,
        )
        try:
            return self._generate()
        finally:
            kill_process_tree(process.pid, wait_timeout=60)

    def test_matches_static_pool(self):
        unified = self._run(["--enable-unified-memory"])
        static = self._run([])

        self.assertEqual(len(unified), 4)
        for logprob, _token_id, _text in unified:
            self.assertEqual(logprob, logprob, "logprob is NaN")

        self.assertEqual(
            [token_id for _lp, token_id, _t in unified],
            [token_id for _lp, token_id, _t in static],
            "unified pool diverged from the static pool",
        )
        for (unified_lp, _uid, _ut), (static_lp, _sid, _st) in zip(unified, static):
            self.assertAlmostEqual(unified_lp, static_lp, delta=0.05)


if __name__ == "__main__":
    unittest.main()
