"""Geometric gate for vision_topk's ids argument.

Regression test for https://github.com/sgl-project/sglang/issues/52215.

The MoE transport CP fast path (a2a none) unconditionally all-gathers
hidden_states to the full CP-interleaved sequence (rows *= attn_cp_size).
The ids side, however, is attached under a predicate whose liveness can
flip during the batch's lifetime (is_cp_active -> attn_cp_metadata).
When the two views diverge, moe_fused_gate asserts
`input_ids.shape == (scores.size(0),)` and every CP rank dies in warm-up.

The consumer-side gate `_gate_ids_for_vision_topk` degrades any mismatch
to `input_ids=None` (plain MoE routing, HAS_TOKEN_BIAS=False), letting
text traffic pass while keeping real ids for geometrically consistent
batches. Verified on 8xH20 + DeepSeek-V4.1-Flash + CP8 + a2a none:
v19 with this gate completed the warm-up that v16 (is_cp_active elif)
and v17 (geometric elif at the model entry) both failed on, with no
throughput regression (BM A32/B48 within -3% / -12% walls).
"""

import unittest

import torch

from sglang.srt.models.deepseek_v2 import _gate_ids_for_vision_topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGateIdsForVisionTopk(CustomTestCase):
    def test_none_propagates(self):
        hs = torch.zeros(16, 4)
        self.assertIsNone(_gate_ids_for_vision_topk(None, hs))

    def test_matching_ids_pass_through(self):
        # Consistent batch: real ids reach the fused gate.
        hs = torch.zeros(16, 4)
        ids = torch.arange(16)
        self.assertIs(_gate_ids_for_vision_topk(ids, hs), ids)

    def test_mismatch_degrades_to_none(self):
        # The exact regression: hidden gathered to 128 rows (16 * cp8),
        # ids still rank-local 16. moe_fused_gate would otherwise assert.
        hs = torch.zeros(128, 4)
        ids = torch.arange(16)
        self.assertIsNone(_gate_ids_for_vision_topk(ids, hs))

    def test_long_prefill_aligned(self):
        # Realistic CP8 warm-up: rank sees 9600 ids, 76800 hs rows.
        hs = torch.zeros(76800, 4)
        ids = torch.arange(76800)
        self.assertIs(_gate_ids_for_vision_topk(ids, hs), ids)

    def test_long_prefill_mismatch(self):
        # The divergent case from issue probe: 16-token warm-up, CP8.
        # ids row count never matches the gathered rows.
        hs = torch.zeros(128, 4)
        ids = torch.arange(16)
        self.assertIsNone(_gate_ids_for_vision_topk(ids, hs))

    def test_oversized_ids_also_gated(self):
        # Defensive: if ids somehow grows past hidden (e.g. a future
        # upstream attach that pads wrong), also gate to None rather
        # than let a different shape assert fire inside the kernel.
        hs = torch.zeros(16, 4)
        ids = torch.arange(128)
        self.assertIsNone(_gate_ids_for_vision_topk(ids, hs))


if __name__ == "__main__":
    unittest.main()
