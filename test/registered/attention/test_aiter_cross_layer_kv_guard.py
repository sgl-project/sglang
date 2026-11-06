"""PR2 guard: aiter target-verify rejects cross-layer KV sharing (k/v is None).

Cross-layer KV sharing (Gemma4) passes ``k=v=None``. The legacy ragged
``extend_attention_fwd`` target-verify path has no pool-reading fallback, so it
must fail loudly with a ``ValueError`` naming the unified verify path rather
than an opaque ``AttributeError`` on ``.contiguous``. The guard is inert when
real K/V is passed.
"""

import unittest

import torch

from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestAiterCrossLayerKVGuard(CustomTestCase):
    def test_raises_when_k_is_none(self):
        with self.assertRaises(ValueError) as ctx:
            AiterAttnBackend._reject_target_verify_cross_layer_kv(None, torch.empty(0))
        self.assertIn("cross-layer KV", str(ctx.exception))
        self.assertIn("unified verify path", str(ctx.exception))

    def test_raises_when_v_is_none(self):
        with self.assertRaises(ValueError):
            AiterAttnBackend._reject_target_verify_cross_layer_kv(torch.empty(0), None)

    def test_raises_when_both_none(self):
        with self.assertRaises(ValueError):
            AiterAttnBackend._reject_target_verify_cross_layer_kv(None, None)

    def test_inert_when_kv_present(self):
        # Real K/V present -> no raise (the guard must not fire for standard
        # models that pass ragged K/V into the legacy target-verify path).
        AiterAttnBackend._reject_target_verify_cross_layer_kv(
            torch.empty(0), torch.empty(0)
        )


if __name__ == "__main__":
    unittest.main()
