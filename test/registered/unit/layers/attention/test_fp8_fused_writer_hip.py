import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsa import fp8_fused_writer_hip
from sglang.kernels.ops.attention.dsa.fp8_fused_writer_hip import (
    aiter_fused_fp8_qk_write,
    aiter_fused_fp8_writer_available,
    fused_fp8_writer_geometry_supported,
    prepare_aiter_rope_caches,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFusedFp8WriterHip(CustomTestCase):
    def test_geometry_gate(self):
        self.assertTrue(fused_fp8_writer_geometry_supported(128, 64, 128))
        self.assertFalse(fused_fp8_writer_geometry_supported(64, 64, 128))
        self.assertFalse(fused_fp8_writer_geometry_supported(128, 32, 128))
        self.assertFalse(fused_fp8_writer_geometry_supported(128, 64, 64))

    def _available_on(self, gcn_arch):
        """Gate result on a faked arch, with the AITER symbol importable."""
        cache = ModuleType("aiter.ops.cache")
        cache.indexer_qk_rope_quant_and_cache = object()
        aiter_modules = {
            "aiter": ModuleType("aiter"),
            "aiter.ops": ModuleType("aiter.ops"),
            "aiter.ops.cache": cache,
        }
        hip = None if gcn_arch is None else SimpleNamespace(gcn_arch=gcn_arch)
        with (
            patch.object(
                fp8_fused_writer_hip,
                "current_arch",
                return_value=SimpleNamespace(hip=hip),
            ),
            patch.dict(sys.modules, aiter_modules),
        ):
            return aiter_fused_fp8_writer_available()

    def test_available_only_on_gfx950(self):
        self.assertTrue(self._available_on("gfx950"))
        # gfx942 (MI300) can import the symbol; None is a non-HIP device.
        for other in ("gfx942", "gfx951", None):
            with self.subTest(gcn_arch=other):
                self.assertFalse(self._available_on(other))

    def test_prepare_aiter_rope_caches_squeezes_cpu_tables(self):
        cos = torch.randn(128, 1, 1, 32, dtype=torch.float32)
        sin = torch.randn(128, 1, 1, 32, dtype=torch.float32)
        got_cos, got_sin = prepare_aiter_rope_caches(
            cos, sin, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertEqual(tuple(got_cos.shape), (128, 32))
        self.assertEqual(got_cos.dtype, torch.float32)
        self.assertTrue(torch.equal(got_cos, cos.squeeze()))
        self.assertTrue(torch.equal(got_sin, sin.squeeze()))

    def _inputs(self):
        tokens, heads, dim, page = 2, 32, 128, 64
        return dict(
            q=torch.empty(tokens, heads, dim, dtype=torch.bfloat16),
            q_out=torch.empty(tokens, heads, dim, dtype=torch.uint8),
            weights=torch.empty(tokens, heads, dtype=torch.bfloat16),
            weights_out=torch.empty(tokens, heads, dtype=torch.float32),
            k=torch.empty(tokens, dim, dtype=torch.bfloat16),
            kv_cache=torch.empty(1, page, dim + 4, dtype=torch.uint8),
            slot_mapping=torch.arange(tokens, dtype=torch.int64),
            norm_weight=torch.ones(dim, dtype=torch.float32),
            norm_bias=torch.zeros(dim, dtype=torch.float32),
            positions=torch.arange(tokens, dtype=torch.int64),
            cos_cache=torch.empty(1024, 32, dtype=torch.bfloat16),
            sin_cache=torch.empty(1024, 32, dtype=torch.bfloat16),
        )

    def test_refuses_non_fp32_layernorm(self):
        args = self._inputs()
        args["norm_weight"] = args["norm_weight"].bfloat16()
        with self.assertRaisesRegex(TypeError, "LayerNorm parameters as fp32"):
            aiter_fused_fp8_qk_write(
                **args,
                epsilon=1e-6,
                quant_block_size=128,
                scale_fmt="ue8m0",
                weights_scale=1.0,
                preshuffle=True,
                is_neox=False,
                compute_all_q_rope=True,
            )

    def test_refuses_wrong_geometry(self):
        args = self._inputs()
        args["cos_cache"] = torch.empty(1024, 16, dtype=torch.bfloat16)
        args["sin_cache"] = torch.empty(1024, 16, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "rope_dim=32"):
            aiter_fused_fp8_qk_write(
                **args,
                epsilon=1e-6,
                quant_block_size=128,
                scale_fmt="ue8m0",
                weights_scale=1.0,
                preshuffle=True,
                is_neox=False,
                compute_all_q_rope=True,
            )


if __name__ == "__main__":
    unittest.main()
