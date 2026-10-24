"""CPU unit tests for --enable-nvfp4-ckpt-fp8-dense.

The flag quantizes the linear layers a ModelOpt NVFP4 checkpoint leaves in BF16 to
block-FP8 while loading. These tests pin what decides its behavior without a GPU:

- dispatch in `ModelOptQuantConfig._unquantized_linear_method`: off by default, NVFP4
  configs only, never `lm_head`, and modules a model excludes (DeepSeek-V2 / GLM:
  kv_b_proj and the DSA indexer's wk / weights_proj; the NextN draft model: all) stay
  unquantized;
- the load-time quantization (`quant_weight_block_fp8_online`): E4M3 with one FP32
  scale per 128x128 block (amax / 448), power-of-two scales when DeepGEMM will
  requantize them to UE8M0, and that requantization then being exact.

The DeepGEMM GEMM and the weight-loading path need the GPU kernels and a model, and are
covered on-device.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

# isort: off
import sglang.srt.layers.quantization.modelopt_quant as mq
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.layers.quantization.fp8 import quant_weight_block_fp8_online
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    per_block_cast_to_fp8,
)

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

# isort: on
from sglang.test.test_utils import CustomTestCase

BLOCK = [128, 128]
# GLM-5.x per-rank dense shapes (out, in): fused q_a+kv_a (ragged rows), q_b, o_proj, shared experts.
SHAPES = [(2624, 6144), (16384, 2048), (6144, 16384), (4096, 6144), (6144, 2048)]


def _weight(shape, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g) * 0.02
    w[:7, :9] *= 40  # one outlier block
    return w.to(torch.bfloat16)


def _linear():
    layer = ReplicatedLinear.__new__(ReplicatedLinear)  # isinstance checks only
    torch.nn.Module.__init__(layer)
    return layer


def _lm_head():
    layer = ParallelLMHead.__new__(ParallelLMHead)
    torch.nn.Module.__init__(layer)
    return layer


class _StubOnlineMethod:
    def __init__(self, quant_config):
        self.quant_config = quant_config


class TestFp8DenseDispatch(CustomTestCase):
    def _config(self, excluded=()):
        qc = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "kv_cache_quant_algo": "FP8",
                    "exclude_modules": ["lm_head", "model.layers.*.self_attn*"],
                }
            }
        )
        qc.update_online_fp8_exclude_modules(excluded)
        return qc

    def _method(self, qc, layer, prefix, enabled=True):
        with (
            patch.object(mq, "_online_fp8_dense_enabled", return_value=enabled),
            patch(
                "sglang.srt.layers.quantization.fp8.Fp8OnlineBlockLinearMethod",
                _StubOnlineMethod,
            ),
        ):
            return qc._unquantized_linear_method(layer, prefix)

    def test_off_by_default(self):
        qc = self._config()
        m = self._method(qc, _linear(), "model.layers.3.self_attn.o_proj", False)
        self.assertIsInstance(m, UnquantizedLinearMethod)

    def test_unquantized_linear_goes_fp8(self):
        qc = self._config(DeepseekV2ForCausalLM.online_fp8_exclude_modules)
        for prefix in (
            "model.layers.3.self_attn.fused_qkv_a_proj_with_mqa",
            "model.layers.3.self_attn.q_b_proj",
            "model.layers.3.self_attn.o_proj",
            "model.layers.3.self_attn.indexer.wq_b",
            "model.layers.3.mlp.shared_experts.gate_up_proj",
            "model.layers.0.mlp.down_proj",
        ):
            m = self._method(qc, _linear(), prefix)
            self.assertIsInstance(m, _StubOnlineMethod, prefix)
            cfg = m.quant_config
            self.assertEqual(cfg.weight_block_size, [128, 128])
            self.assertTrue(cfg.is_checkpoint_fp8_serialized)

    def test_model_exclusions_and_lm_head(self):
        qc = self._config(DeepseekV2ForCausalLM.online_fp8_exclude_modules)
        for prefix in (
            "model.layers.10.self_attn.kv_b_proj",
            "model.layers.3.self_attn.indexer.wk_weights_proj",
            "model.layers.3.self_attn.indexer.wk",
            "model.layers.3.self_attn.indexer.weights_proj",
        ):
            m = self._method(qc, _linear(), prefix)
            self.assertIsInstance(m, UnquantizedLinearMethod, prefix)
        self.assertIsInstance(
            self._method(qc, _lm_head(), "lm_head"), UnquantizedLinearMethod
        )

    def test_nextn_excludes_all(self):
        qc = self._config(("*",))
        m = self._method(qc, _linear(), "model.decoder.self_attn.o_proj")
        self.assertIsInstance(m, UnquantizedLinearMethod)

    def test_non_nvfp4_config_does_not_opt_in(self):
        self.assertFalse(mq.ModelOptFp8Config.supports_online_fp8_dense)
        self.assertTrue(mq.ModelOptMixedPrecisionConfig.supports_online_fp8_dense)


class TestFp8DenseOnlineQuant(CustomTestCase):
    def test_layout_and_error(self):
        for shape in SHAPES:
            w = _weight(shape, seed=shape[0])
            for pow2 in (False, True):
                q, s = quant_weight_block_fp8_online(w, BLOCK, pow2)
                self.assertEqual(q.dtype, torch.float8_e4m3fn)
                self.assertEqual(tuple(q.shape), shape)
                self.assertEqual(s.dtype, torch.float32)
                self.assertEqual(
                    tuple(s.shape), (-(-shape[0] // 128), -(-shape[1] // 128))
                )
                self.assertTrue(bool(torch.isfinite(s).all() and (s > 0).all()))
                d = block_quant_dequant(q, s, BLOCK, torch.float32)
                rel = ((d - w.float()).norm() / w.float().norm()).item()
                self.assertLess(rel, 0.035, f"{shape} pow2={pow2}: rel err {rel}")

    def test_fp32_scale_is_amax_over_448(self):
        w = _weight((512, 1024), seed=1)
        _, s = quant_weight_block_fp8_online(w, BLOCK, pow2_scale=False)
        amax = w.float().view(4, 128, 8, 128).abs().amax(dim=(1, 3))
        torch.testing.assert_close(s, amax / 448.0)

    def test_pow2_scales_requant_exactly(self):
        # DeepGEMM on Blackwell requantizes block scales to UE8M0 (dequant -> BF16 ->
        # per_block_cast_to_fp8). With power-of-two scales that round trip must be exact,
        # so the weight is rounded once, from BF16.
        for shape in ((4096, 6144), (16384, 2048), (6144, 16384)):
            w = _weight(shape, seed=7)
            q, s = quant_weight_block_fp8_online(w, BLOCK, pow2_scale=True)
            torch.testing.assert_close(s, torch.exp2(torch.round(torch.log2(s))))
            d = block_quant_dequant(q, s, BLOCK, torch.float32)
            q2, s2 = per_block_cast_to_fp8(
                block_quant_dequant(q, s, BLOCK, torch.bfloat16)
            )
            torch.testing.assert_close(
                block_quant_dequant(q2, s2, BLOCK, torch.float32), d, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
