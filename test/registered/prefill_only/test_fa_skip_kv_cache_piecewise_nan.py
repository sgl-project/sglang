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
and asserts no embedding contains NaN, and that the embeddings match the HuggingFace
reference (cosine).

The fa_skip_kv_cache path is opt-in via `--prefill-only-disable-kv-cache`, so this
test sets that flag to keep exercising the fast path; it also checks the default
paged path (flag off, what deployments run unless they opt in) over the same
no-NaN + HF-parity assertions under the piecewise graph.
"""

import multiprocessing as mp
import os
import unittest

import torch

from sglang import Engine
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import HFRunner
from sglang.test.test_utils import CustomTestCase, get_similarities

_HF_COSINE_FLOOR = 0.999

# Route to the H100 runner (1-gpu-large, SM90) -- NOT 1-gpu-small, which is an
# RTX 5090 (SM120/Blackwell) where FA3 does not exist. FA3 + the piecewise embedding
# path this regression covers only runs on Ampere/Ada/Hopper (SM 80-90), so the test
# must land on the H100 pool to actually execute (on 1-gpu-small it would skip 100%).
register_cuda_ci(est_time=600, stage="base-b", runner_config="1-gpu-large")

# Lowest/highest CUDA SM that supports the FA3 + piecewise embedding path. FA3 is
# unavailable on Blackwell (sm100 B200 / sm120 consumer e.g. RTX 5090); the gate is
# applied at RUNTIME (see setUp) so the SM is read after CUDA is initialized on the
# actual runner, never at import/collection time.
_FA3_SM_MIN, _FA3_SM_MAX = 80, 90

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


def _embed(prompts, prefill_only_disable_kv_cache=True, **engine_kwargs):
    # The raw-K/V fa_skip_kv_cache fast path is opt-in via
    # --prefill-only-disable-kv-cache. is_embedding + chunked_prefill_size == -1 +
    # disable_radix_cache (+ a non-MLA model + the FA3 backend) are the structural
    # preconditions, but the flag is what routes onto the fast path; without it the
    # request runs the default paged flash_attn_with_kvcache path. This regression
    # exercises the fast path, so it defaults the flag on.
    engine = Engine(
        model_path=MODEL_PATH,
        is_embedding=True,
        attention_backend="fa3",
        chunked_prefill_size=-1,
        disable_radix_cache=True,
        prefill_only_disable_kv_cache=prefill_only_disable_kv_cache,
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


# Enables the piecewise CUDA graph for prefill the way production does. After the
# cuda-graph refactor (#23906) the piecewise config lives in cuda_graph_config; the
# convenience kwargs below fold into cuda_graph_config[prefill]:
#   - cuda_graph_backend_prefill="tc_piecewise" -> prefill.backend (also the default)
#   - cuda_graph_max_bs_prefill=32768           -> prefill.max_bs (for tc_piecewise
#     prefill, max_bs/bs carries the captured TOKEN count -- the old
#     piecewise_cuda_graph_max_tokens)
#   - cuda_graph_tc_compiler="inductor"         -> prefill.tc_compiler
_PIECEWISE_KWARGS = dict(
    cuda_graph_backend_prefill="tc_piecewise",
    cuda_graph_max_bs_prefill=32768,
    cuda_graph_tc_compiler="inductor",
)


class TestFaSkipKvCachePiecewiseNoNaN(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # HFRunner spawns a CUDA subprocess; forking after CUDA is initialized in
        # the parent fails, so force spawn (matches the other embedding tests).
        mp.set_start_method("spawn", force=True)

    def setUp(self):
        # Gate at runtime: read the SM after CUDA is initialized on the runner. If
        # the hardware can't run FA3 (e.g. SM120 RTX 5090 / SM100 B200), skip --
        # a skip is NOT a CI failure, it just records the test as inapplicable here.
        sm = get_device_sm()
        if not (_FA3_SM_MIN <= sm <= _FA3_SM_MAX):
            self.skipTest(
                f"fa3 + piecewise embedding repro requires CUDA SM "
                f"{_FA3_SM_MIN}-{_FA3_SM_MAX} (Ampere/Ada/Hopper); got SM {sm}"
            )

    def test_embeddings_no_nan_and_match_hf(self):
        prompts = _short_prompts()
        # HF reference is path-independent: compute once and reuse for both paths.
        with HFRunner(
            MODEL_PATH, torch_dtype=torch.float16, model_type="embedding"
        ) as hf_runner:
            hf_embs = [
                torch.tensor(e, dtype=torch.float32)
                for e in hf_runner.forward(prompts).embed_logits
            ]
        # Both attention paths must stay NaN-free under the piecewise CUDA graph and
        # match the HF reference:
        #   flag on  -> the opt-in raw-K/V fa_skip_kv_cache fast path (where the bug
        #               surfaced).
        #   flag off -> the default paged flash_attn_with_kvcache path that embedding
        #               deployments run unless they opt in.
        for prefill_only_disable_kv_cache in (True, False):
            label = f"prefill_only_disable_kv_cache={prefill_only_disable_kv_cache}"
            with self.subTest(label):
                srt_embs = _embed(
                    prompts,
                    prefill_only_disable_kv_cache=prefill_only_disable_kv_cache,
                    **_PIECEWISE_KWARGS,
                )
                nan_idx = [i for i, e in enumerate(srt_embs) if torch.isnan(e).any()]
                self.assertEqual(
                    nan_idx,
                    [],
                    f"{len(nan_idx)}/{len(srt_embs)} short-input embeddings contain NaN "
                    f"under {label} + piecewise CUDA graph "
                    f"(e.g. prompt indices {nan_idx[:10]})",
                )
                for i, (srt, hf) in enumerate(zip(srt_embs, hf_embs)):
                    cos = float(get_similarities(hf, srt))
                    self.assertGreater(
                        cos,
                        _HF_COSINE_FLOOR,
                        f"{label}, prompt {i}: SRT-vs-HF cosine "
                        f"{cos:.4f} < {_HF_COSINE_FLOOR}",
                    )


if __name__ == "__main__":
    unittest.main()
