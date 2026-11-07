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
"""MLA decode on gfx115x must stay correct on ragged batches where 3+ sequences hit the
64-split maximum; stage 1 returned NaN or faulted there (#48967).

    python -m pytest test/registered/unit/layers/attention/test_mla_decode_gfx115.py -v
"""

import unittest

import torch

from sglang.kernels.ops.attention import decode_attention as da
from sglang.srt.utils import is_gfx115_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# No gfx115x runner yet; skips everywhere else.
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

LK, LV, HEADS, MAX_KV_SPLITS = 576, 512, 16, 64

# Split counts are what the scheduler's heuristic writes for these lengths on gfx1151.
CASES = (
    ("short_mid_long_x2", [3, 700, 9000] * 2, [1, 5, 64] * 2),
    ("three_long_one_short", [9000] * 3 + [3], [64] * 3 + [1]),
)
RUNS = 5
# bf16 rounding of the output alone measures ~1e-2 against fp32.
ATOL = 2e-2


@unittest.skipUnless(is_gfx115_supported(), "gfx115x only")
class TestMlaDecodeGfx115(CustomTestCase):
    def test_ragged_batches_match_reference(self):
        for seed, (name, seq_lens, splits) in enumerate(CASES):
            with self.subTest(case=name):
                gen = torch.Generator(device="cuda").manual_seed(seed)
                batch, total = len(seq_lens), sum(seq_lens)
                pool = torch.randn(
                    total, 1, LK, dtype=torch.bfloat16, device="cuda", generator=gen
                )
                q = torch.randn(
                    batch, HEADS, LK, dtype=torch.bfloat16, device="cuda", generator=gen
                )
                kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
                kv_indptr[1:] = torch.cumsum(
                    torch.tensor(seq_lens, dtype=torch.int32, device="cuda"), dim=0
                )
                kv_indices = torch.arange(total, dtype=torch.int64, device="cuda")
                num_kv_splits = torch.tensor(splits, dtype=torch.int32, device="cuda")
                sm_scale = 1.0 / LK**0.5

                ref = torch.stack(
                    [
                        torch.softmax(q[b].float() @ k.T * sm_scale, -1) @ k[:, :LV]
                        for b in range(batch)
                        for k in [pool[kv_indptr[b] : kv_indptr[b + 1], 0].float()]
                    ]
                )
                # The bad partials varied run to run, so one clean run proves little.
                for _ in range(RUNS):
                    o = torch.zeros(
                        batch, HEADS, LV, dtype=torch.bfloat16, device="cuda"
                    )
                    logits = torch.empty(
                        batch,
                        HEADS,
                        MAX_KV_SPLITS,
                        LV,
                        dtype=torch.float32,
                        device="cuda",
                    )
                    lse = torch.full(
                        (batch, HEADS, MAX_KV_SPLITS),
                        -float("inf"),
                        dtype=torch.float32,
                        device="cuda",
                    )
                    da.decode_attention_fwd_grouped(
                        q,
                        pool,
                        pool[..., :LV],
                        o,
                        kv_indptr,
                        kv_indices,
                        logits,
                        lse,
                        num_kv_splits,
                        MAX_KV_SPLITS,
                        sm_scale,
                        1.0,
                        has_mla=True,
                    )
                    torch.cuda.synchronize()
                    self.assertFalse(torch.isnan(o).any())
                    torch.testing.assert_close(o.float(), ref, rtol=0, atol=ATOL)


if __name__ == "__main__":
    unittest.main()
