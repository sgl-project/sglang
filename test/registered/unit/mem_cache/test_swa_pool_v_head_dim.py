# Copyright 2023-2026 SGLang Team
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
"""`SWAKVPool.get_v_head_dim()` — the pool method a mambaish + hybrid-SWA
model reaches on boot.

`TritonAttnBackend.__init__` picks its `v_head_dim` from one of three
branches, and the middle one asks the POOL:

    if sliding_window_size is not None and swa_v_head_dim != v_head_dim:
        ... from model_config ...                     # asymmetric hybrid SWA
    elif mambaish_config(model_config) is not None:
        v_head_dim = token_to_kv_pool.get_v_head_dim()   # <-- this one
    else:
        ... from get_value_buffer(start_layer) ...

A model that is BOTH mambaish AND hybrid-SWA with MATCHING full/SWA value
head dims (Inkling-class) skips the first branch and lands in the second —
where its pool is an SWA-shaped pool, which had no `get_v_head_dim`. The
server died at backend construction with

    AttributeError: 'SWAKVPool' object has no attribute 'get_v_head_dim'

on the STATIC pool and, identically, on `UnifiedSWAKVPool`. Neither the
mamba-hybrid pools (`HybridLinearKVPool` has the method) nor pure hybrid-SWA
models (not mambaish, so the branch is never taken) can reach it, which is
why it went unnoticed.

    python -m pytest test/registered/unit/mem_cache/test_swa_pool_v_head_dim.py -v
"""

import inspect
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_DEV = "cpu"
_FULL_V_HEAD_DIM = 8
_SWA_V_HEAD_DIM = 8  # MATCHING — this is what routes Inkling into the branch


def _swa_pool():
    """A static SWAKVPool with the Inkling-class layer split: full and SWA
    layers interleaved, layer 0 NOT a full-attention layer (which is exactly
    why the backend asks the pool instead of indexing layer 0)."""
    return SWAKVPool(
        size=32,
        size_swa=16,
        page_size=1,
        dtype=torch.float16,
        head_num=2,
        head_dim=_FULL_V_HEAD_DIM,
        swa_attention_layer_ids=[0, 2],
        full_attention_layer_ids=[1, 3],
        device=_DEV,
        enable_memory_saver=False,
    )


class TestSWAPoolVHeadDim(unittest.TestCase):
    def test_static_pool_reports_the_full_side_value_head_dim(self):
        """Red before the fix with AttributeError; the value must be the FULL
        side's, since that is the geometry the caller means."""
        pool = _swa_pool()
        self.assertEqual(pool.get_v_head_dim(), _FULL_V_HEAD_DIM)

    def test_answer_matches_the_full_pool_buffer_not_layer_zero(self):
        """Layer 0 is an SWA layer here, so a naive `get_value_buffer(0)`
        would read the SWA side. Pin that the method routes through the FULL
        sub-pool at its own start_layer — the property that makes it correct
        under pipeline parallelism too."""
        pool = _swa_pool()
        want = pool.full_kv_pool.get_value_buffer(pool.full_kv_pool.start_layer).shape[
            -1
        ]
        self.assertEqual(pool.get_v_head_dim(), want)
        # And layer 0 really is the SWA side in this fixture.
        _, is_swa = pool.layers_mapping[0]
        self.assertTrue(is_swa, "fixture must keep layer 0 on the SWA side")

    def test_unified_swa_pool_inherits_it(self):
        """`UnifiedSWAKVPool` subclasses `SWAKVPool`, so the unified tri-pool
        path (mambaish + hybrid SWA in one buffer) is covered by the same
        method — no second implementation to drift."""
        self.assertTrue(issubclass(UnifiedSWAKVPool, SWAKVPool))
        self.assertIs(
            UnifiedSWAKVPool.get_v_head_dim,
            SWAKVPool.get_v_head_dim,
            "the unified pool must inherit the method, not shadow it",
        )

    def test_signature_matches_the_hybrid_linear_precedent(self):
        """The backend calls this method on whichever pool it holds, so every
        pool reachable from the mambaish branch must expose the SAME
        zero-argument shape. `HybridLinearKVPool` is the precedent this one
        mirrors; a future pool added to that branch has to match too."""
        for cls in (SWAKVPool, HybridLinearKVPool):
            sig = inspect.signature(cls.get_v_head_dim)
            self.assertEqual(
                [p for p in sig.parameters if p != "self"],
                [],
                f"{cls.__name__}.get_v_head_dim must take no arguments",
            )


if __name__ == "__main__":
    unittest.main()
