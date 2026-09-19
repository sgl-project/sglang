"""Regression tests for GLM MLA absorb-weight selection in the weight loader."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.deepseek_common import deepseek_weight_loader
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_GLM_ARCHITECTURE = "GlmMoeDsaForCausalLM"
_LOCAL_HEADS = 8
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 192
_V_HEAD_DIM = 256


class _LoaderHarness(DeepseekV2WeightLoaderMixin):
    pass


def _quark_config() -> SimpleNamespace:
    return SimpleNamespace(get_name=lambda: "quark")


def _make_loader(
    *,
    architecture: str,
    quant_config: SimpleNamespace | None,
    device: torch.device,
) -> tuple[_LoaderHarness, SimpleNamespace, torch.Tensor]:
    torch.manual_seed(0)
    weight = (
        torch.randn(
            _LOCAL_HEADS * (_QK_NOPE_HEAD_DIM + _V_HEAD_DIM),
            _KV_LORA_RANK,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    self_attn = SimpleNamespace(
        kv_b_proj=SimpleNamespace(weight=weight),
        num_local_heads=_LOCAL_HEADS,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        w_kc=None,
        w_vc=None,
        w_scale=1.0,
        w_kc_decode=None,
        w_vc_decode=None,
        w_scale_decode=None,
        use_glm_bf16_prefill_fp8_decode=False,
        w_scale_k=None,
        w_scale_v=None,
        use_deep_gemm_bmm=False,
        o_proj=SimpleNamespace(
            weight=torch.empty(1, device=device, dtype=torch.bfloat16)
        ),
    )

    loader = _LoaderHarness()
    loader.config = SimpleNamespace(
        architectures=[architecture],
        num_hidden_layers=1,
    )
    loader.quant_config = quant_config
    loader.model = SimpleNamespace(
        start_layer=0,
        end_layer=1,
        layers=[SimpleNamespace(self_attn=self_attn)],
    )
    return loader, self_attn, weight


class TestDeepseekWeightLoaderAbsorbDtype(CustomTestCase):
    def test_glm_quark_gfx950_default_keeps_existing_fp8_absorb(self):
        loader, self_attn, weight = _make_loader(
            architecture=_GLM_ARCHITECTURE,
            quant_config=_quark_config(),
            device=torch.device("cpu"),
        )

        with (
            patch.object(deepseek_weight_loader, "_use_aiter_gfx95", True),
            patch.object(
                deepseek_weight_loader.envs.SGLANG_GLM_BF16_PREFILL_FP8_DECODE,
                "get",
                return_value=False,
            ),
            patch.object(
                deepseek_weight_loader,
                "quark_post_load_weights",
                side_effect=AssertionError("GLM must not enter DeepSeek Quark fixup"),
                create=True,
            ),
        ):
            loader.post_load_weights()

        expected_weight, expected_scale = deepseek_weight_loader.input_to_float8(
            weight,
            dtype=torch.float8_e4m3fn,
        )
        weight_by_head = expected_weight.unflatten(
            0, (_LOCAL_HEADS, _QK_NOPE_HEAD_DIM + _V_HEAD_DIM)
        )
        expected_w_kc, expected_w_vc = weight_by_head.split(
            [_QK_NOPE_HEAD_DIM, _V_HEAD_DIM],
            dim=1,
        )

        self.assertEqual(self_attn.w_kc.dtype, torch.float8_e4m3fn)
        self.assertEqual(self_attn.w_vc.dtype, torch.float8_e4m3fn)
        torch.testing.assert_close(self_attn.w_scale, expected_scale)
        self.assertEqual(
            self_attn.w_kc.shape,
            (_LOCAL_HEADS, _QK_NOPE_HEAD_DIM, _KV_LORA_RANK),
        )
        self.assertEqual(
            self_attn.w_vc.shape,
            (_LOCAL_HEADS, _KV_LORA_RANK, _V_HEAD_DIM),
        )
        self.assertEqual(
            self_attn.w_kc.stride(),
            (_QK_NOPE_HEAD_DIM * _KV_LORA_RANK, 1, _QK_NOPE_HEAD_DIM),
        )
        self.assertEqual(
            self_attn.w_vc.stride(),
            (_KV_LORA_RANK * _V_HEAD_DIM, 1, _KV_LORA_RANK),
        )
        self.assertTrue(torch.equal(self_attn.w_kc, expected_w_kc))
        self.assertTrue(torch.equal(self_attn.w_vc, expected_w_vc.transpose(1, 2)))
        self.assertFalse(self_attn.use_glm_bf16_prefill_fp8_decode)
        self.assertIsNone(self_attn.w_kc_decode)
        self.assertIsNone(self_attn.w_vc_decode)
        self.assertIsNone(self_attn.w_scale_decode)

    def test_glm_hybrid_keeps_bf16_and_builds_fp8_decode_weights(self):
        loader, self_attn, weight = _make_loader(
            architecture=_GLM_ARCHITECTURE,
            quant_config=_quark_config(),
            device=torch.device("cpu"),
        )

        with (
            patch.object(deepseek_weight_loader, "_use_aiter_gfx95", True),
            patch.object(
                deepseek_weight_loader.envs.SGLANG_GLM_BF16_PREFILL_FP8_DECODE,
                "get",
                return_value=True,
            ),
            patch.object(
                deepseek_weight_loader,
                "quark_post_load_weights",
                side_effect=AssertionError("GLM must not enter DeepSeek Quark fixup"),
                create=True,
            ),
        ):
            loader.post_load_weights()

        expected_decode_weight, expected_decode_scale = (
            deepseek_weight_loader.input_to_float8(
                weight,
                dtype=torch.float8_e4m3fn,
            )
        )
        expected_decode_by_head = expected_decode_weight.unflatten(
            0,
            (_LOCAL_HEADS, _QK_NOPE_HEAD_DIM + _V_HEAD_DIM),
        )
        expected_decode_w_kc, expected_decode_w_vc = expected_decode_by_head.split(
            [_QK_NOPE_HEAD_DIM, _V_HEAD_DIM],
            dim=1,
        )

        self.assertTrue(self_attn.use_glm_bf16_prefill_fp8_decode)
        self.assertEqual(self_attn.w_kc.dtype, torch.bfloat16)
        self.assertEqual(self_attn.w_vc.dtype, torch.bfloat16)
        self.assertEqual(self_attn.w_kc_decode.dtype, torch.float8_e4m3fn)
        self.assertEqual(self_attn.w_vc_decode.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(self_attn.w_kc_decode, expected_decode_w_kc))
        self.assertTrue(
            torch.equal(
                self_attn.w_vc_decode,
                expected_decode_w_vc.transpose(1, 2),
            )
        )
        torch.testing.assert_close(
            self_attn.w_scale_decode,
            expected_decode_scale,
        )

        expected_extra_bytes = (
            self_attn.w_kc_decode.numel() * self_attn.w_kc_decode.element_size()
            + self_attn.w_vc_decode.numel() * self_attn.w_vc_decode.element_size()
            + self_attn.w_scale_decode.numel() * self_attn.w_scale_decode.element_size()
        )
        self.assertEqual(expected_extra_bytes, 1_835_012)

    def test_unquantized_unrelated_architecture_keeps_bf16(self):
        loader, self_attn, _ = _make_loader(
            architecture="DeepseekV2ForCausalLM",
            quant_config=None,
            device=torch.device("cpu"),
        )

        with (
            patch.object(deepseek_weight_loader, "_use_aiter_gfx95", True),
            patch.object(
                deepseek_weight_loader.envs.SGLANG_GLM_BF16_PREFILL_FP8_DECODE,
                "get",
                return_value=True,
            ),
        ):
            loader.post_load_weights()

        self.assertEqual(self_attn.w_kc.dtype, torch.bfloat16)
        self.assertEqual(self_attn.w_vc.dtype, torch.bfloat16)
        self.assertEqual(self_attn.w_scale, 1.0)
        self.assertFalse(self_attn.use_glm_bf16_prefill_fp8_decode)
        self.assertIsNone(self_attn.w_kc_decode)
        self.assertIsNone(self_attn.w_vc_decode)

    def test_glm_hybrid_flag_is_ignored_off_gfx950(self):
        loader, self_attn, _ = _make_loader(
            architecture=_GLM_ARCHITECTURE,
            quant_config=_quark_config(),
            device=torch.device("cpu"),
        )

        with (
            patch.object(deepseek_weight_loader, "_use_aiter_gfx95", False),
            patch.object(
                deepseek_weight_loader.envs.SGLANG_GLM_BF16_PREFILL_FP8_DECODE,
                "get",
                return_value=True,
            ),
        ):
            loader.post_load_weights()

        self.assertEqual(self_attn.w_kc.dtype, torch.bfloat16)
        self.assertEqual(self_attn.w_vc.dtype, torch.bfloat16)
        self.assertFalse(self_attn.use_glm_bf16_prefill_fp8_decode)
        self.assertIsNone(self_attn.w_kc_decode)
        self.assertIsNone(self_attn.w_vc_decode)

    def test_deepseek_quark_gfx950_still_uses_quark_fixup(self):
        loader, self_attn, _ = _make_loader(
            architecture="DeepseekV3ForCausalLM",
            quant_config=_quark_config(),
            device=torch.device("cpu"),
        )
        expected_scale_k = torch.ones((_LOCAL_HEADS, 1, 1))
        expected_scale_v = torch.ones((_LOCAL_HEADS, 1, 1))

        def fake_quark_post_load_weights(attn, weight, quant_format):
            weight_by_head = weight.unflatten(
                0, (_LOCAL_HEADS, _QK_NOPE_HEAD_DIM + _V_HEAD_DIM)
            )
            w_kc, w_vc = weight_by_head.split(
                [_QK_NOPE_HEAD_DIM, _V_HEAD_DIM],
                dim=1,
            )
            self.assertIs(attn, self_attn)
            self.assertEqual(quant_format, "mxfp4")
            return (
                w_kc.to(torch.uint8),
                expected_scale_k,
                w_vc.to(torch.uint8),
                expected_scale_v,
            )

        with (
            patch.object(deepseek_weight_loader, "_use_aiter_gfx95", True),
            patch.object(
                deepseek_weight_loader.envs.SGLANG_GLM_BF16_PREFILL_FP8_DECODE,
                "get",
                return_value=True,
            ),
            patch.object(
                deepseek_weight_loader,
                "quark_post_load_weights",
                side_effect=fake_quark_post_load_weights,
                create=True,
            ) as quark_fixup,
        ):
            loader.post_load_weights()

        quark_fixup.assert_called_once()
        self.assertEqual(self_attn.w_kc.dtype, torch.uint8)
        self.assertEqual(self_attn.w_vc.dtype, torch.uint8)
        self.assertIs(self_attn.w_scale_k, expected_scale_k)
        self.assertIs(self_attn.w_scale_v, expected_scale_v)


if __name__ == "__main__":
    unittest.main()
