# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression test for the fa_skip_kv_cache embedding fast path under piecewise
CUDA graph.

PR #21971 added an embedding fast path (`fa_skip_kv_cache`) that serves attention
with `flash_attn_varlen_func` on raw K/V. Under a piecewise CUDA graph the model
forward runs at a padded token-bucket size, so `q` has more rows than
`cu_seqlens_q` covers. `flash_attn_varlen_func` requires
`q.shape[0] == cu_seqlens_q[-1]`; when that is violated the boundary query block
corrupts the **last real token's** output. Because embedding models use LAST-token
pooling, that corrupted row IS the returned embedding -> ~40% of *short* inputs
came back fully NaN (long inputs, which fill the bucket, were unaffected).

This test feeds a spread of short inputs through `fa3 + piecewise + fa_skip_kv_cache`
and asserts no embedding contains NaN, and that the embeddings match the
non-piecewise path.
"""

import os
import unittest

import torch

from sglang import Engine
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

# This regression covers the fa3 + piecewise CUDA graph embedding path (the prod
# config), which only exists on Ampere/Ada/Hopper (SM 80-90). FA3 is unavailable on
# Blackwell (sm100 B200 / sm120 consumer e.g. RTX 5090), so gate to SM 80-90.
_DEVICE_SM = get_device_sm()

# Overridable so the test can run against a locally-mounted model in dev.
MODEL_PATH = os.environ.get("SGLANG_TEST_EMB_MODEL", "Qwen/Qwen3-Embedding-0.6B")

_WORDS = [
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "embedding",
    "vector",
    "token",
    "sample",
]


def _short_prompts():
    """A spread of short inputs (~1..150 tokens).

    The bug only triggers when a prefill is PADDED up to a piecewise bucket, i.e.
    for token counts that are not exactly a capture size. Using many lengths
    guarantees several land just below a bucket boundary (80/96/112/128 ...).
    """
    return [" ".join(_WORDS[i % len(_WORDS)] for i in range(n)) for n in range(1, 150)]


def _embed(prompts, **engine_kwargs):
    # fa_skip_kv_cache is enabled by: is_embedding + chunked_prefill_size == -1
    # + disable_radix_cache (+ a non-MLA model + the FA3 backend).
    engine = Engine(
        model_path=MODEL_PATH,
        is_embedding=True,
        attention_backend="fa3",
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        **engine_kwargs,
    )
    try:
        # Encode one request per forward (batch size 1). The bug corrupts the
        # last real token, which sits exactly at the real/pad boundary; when many
        # requests are batched into one forward only the tail request hits the
        # boundary, which hides the per-request failure rate.
        embs = []
        for prompt in prompts:
            out = engine.encode(prompt)
            emb = out["embedding"] if isinstance(out, dict) else out[0]["embedding"]
            embs.append(torch.tensor(emb, dtype=torch.float32))
        return embs
    finally:
        engine.shutdown()


# Enables the piecewise CUDA graph the way production does (ENABLE_PIECEWISE_CUDA_GRAPH).
# NOTE: `enforce_piecewise_cuda_graph=True` does NOT reproduce the bug for this path.
_PIECEWISE_KWARGS = dict(
    piecewise_cuda_graph_max_tokens=32768,
    piecewise_cuda_graph_compiler="inductor",
)


@unittest.skipUnless(
    80 <= _DEVICE_SM <= 90,
    f"fa3 + piecewise embedding repro requires CUDA SM 80-90 (Ampere/Ada/Hopper); got SM {_DEVICE_SM}",
)
class TestFaSkipKvCachePiecewiseNoNaN(CustomTestCase):
    def test_no_nan_with_piecewise(self):
        prompts = _short_prompts()
        embs = _embed(prompts, **_PIECEWISE_KWARGS)
        nan_idx = [i for i, e in enumerate(embs) if torch.isnan(e).any()]
        self.assertEqual(
            nan_idx,
            [],
            f"{len(nan_idx)}/{len(embs)} short-input embeddings contain NaN under "
            f"fa_skip_kv_cache + piecewise CUDA graph (e.g. prompt indices {nan_idx[:10]})",
        )

    def test_matches_non_piecewise(self):
        prompts = _short_prompts()
        with_pcg = _embed(prompts, **_PIECEWISE_KWARGS)
        without_pcg = _embed(prompts, disable_piecewise_cuda_graph=True)
        for i, (a, b) in enumerate(zip(with_pcg, without_pcg)):
            self.assertFalse(
                torch.isnan(a).any(),
                f"prompt {i}: NaN embedding with piecewise CUDA graph",
            )
            cos = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
            self.assertGreater(
                cos, 0.99, f"prompt {i}: cosine {cos:.4f} < 0.99 vs non-piecewise"
            )


if __name__ == "__main__":
    unittest.main()
