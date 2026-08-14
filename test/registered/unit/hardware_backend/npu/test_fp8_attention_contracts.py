"""Derived-property tests for Ascend packed FP8 attention contracts."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.attention.fp8_contracts import (
    get_dsa_fp8_packed_cache_dim,
    normalize_required_fp8_scale,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAscendFP8AttentionContracts(CustomTestCase):
    def test_glm52_packed_cache_width_accounts_for_each_storage_dtype(self):
        """A layout rewrite must not drop RoPE bytes or per-tile FP32 scales."""

        self.assertEqual(
            get_dsa_fp8_packed_cache_dim(
                kv_lora_rank=512,
                qk_rope_head_dim=64,
            ),
            656,
        )

    def test_packed_cache_rejects_partial_quantization_tiles(self):
        """A partial final tile has no representable scale slot in the ABI."""

        with self.assertRaisesRegex(ValueError, "must be divisible"):
            get_dsa_fp8_packed_cache_dim(
                kv_lora_rank=513,
                qk_rope_head_dim=64,
            )

    def test_runtime_scale_is_required_and_normalized(self):
        """Missing checkpoint scales must fail instead of changing logits silently."""

        with self.assertRaisesRegex(RuntimeError, "required"):
            normalize_required_fp8_scale(
                None,
                name="fak_descale_float",
                device=torch.device("cpu"),
            )

        normalized = normalize_required_fp8_scale(
            torch.tensor([[0.5]], dtype=torch.float64),
            name="fak_descale_float",
            device=torch.device("cpu"),
        )
        self.assertEqual(normalized.shape, (1,))
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertEqual(normalized.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
