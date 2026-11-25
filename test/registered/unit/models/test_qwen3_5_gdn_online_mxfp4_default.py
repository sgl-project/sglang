"""Unit tests verifying Qwen3.5 MI35x GDN online MXFP4 is opt-in (off by default)."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import initialize_dp_attention_flags
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    AiterMxfp4OnlineLinearMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models import qwen3_5
from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _build_qwen3_5_gdn(
    *,
    model_type: str = "qwen3_5_moe_text",
    hidden_size: int = 256,
    num_heads: int = 2,
    head_dim: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> qwen3_5.Qwen3_5GatedDeltaNet:
    config = SimpleNamespace(
        model_type=model_type,
        hidden_size=hidden_size,
        linear_num_value_heads=num_heads,
        linear_num_key_heads=num_heads,
        linear_num_value_heads_cpu=num_heads,
        linear_num_key_heads_cpu=num_heads,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=3,
        hidden_act="silu",
        output_gate_type=None,
        rms_norm_eps=1e-6,
        torch_dtype=dtype,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with patch(
            "torch.get_device_module",
            return_value=SimpleNamespace(current_device=lambda: "cpu"),
        ):
            return qwen3_5.Qwen3_5GatedDeltaNet(config, layer_id=0, quant_config=None)
    finally:
        torch.set_default_dtype(prev_dtype)


class TestQwen35GdnOnlineMxfp4Default(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        server = ServerArgs(model_path="dummy", device="cpu", tp_size=1)
        publish(server, role="test", ranks=SpawnRanks(world_rank=0))
        initialize_dp_attention_flags(server)

    def test_env_var_defaults_to_false(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_ROCM_GDN_ONLINE_MXFP4", None)
            self.assertIs(envs.SGLANG_ROCM_GDN_ONLINE_MXFP4.get(), False)

            os.environ["SGLANG_ROCM_GDN_ONLINE_MXFP4"] = "1"
            self.assertIs(envs.SGLANG_ROCM_GDN_ONLINE_MXFP4.get(), True)

            os.environ["SGLANG_ROCM_GDN_ONLINE_MXFP4"] = "0"
            self.assertIs(envs.SGLANG_ROCM_GDN_ONLINE_MXFP4.get(), False)

    def test_bf16_gdn_stays_unquantized_by_default_on_gfx95_aiter(self):
        with (
            patch.dict(os.environ),
            patch.object(qwen3_5, "_use_aiter", True),
            patch.object(qwen3_5, "_is_gfx95", True),
        ):
            os.environ.pop("SGLANG_ROCM_GDN_ONLINE_MXFP4", None)
            for model_type in ("qwen3_5_text", "qwen3_5_moe_text"):
                with self.subTest(model_type=model_type):
                    gdn = _build_qwen3_5_gdn(model_type=model_type)
                    self.assertIsInstance(
                        gdn.in_proj_qkvz.quant_method, UnquantizedLinearMethod
                    )
                    self.assertIsInstance(
                        gdn.out_proj.quant_method, UnquantizedLinearMethod
                    )
                    self.assertEqual(gdn.in_proj_qkvz.weight.dtype, torch.bfloat16)
                    self.assertEqual(gdn.out_proj.weight.dtype, torch.bfloat16)

    def test_bf16_gdn_enables_online_mxfp4_when_opted_in(self):
        with (
            patch.dict(os.environ, {"SGLANG_ROCM_GDN_ONLINE_MXFP4": "1"}),
            patch.object(qwen3_5, "_use_aiter", True),
            patch.object(qwen3_5, "_is_gfx95", True),
        ):
            for model_type in ("qwen3_5_text", "qwen3_5_moe_text"):
                with self.subTest(model_type=model_type):
                    gdn = _build_qwen3_5_gdn(model_type=model_type)
                    self.assertIsInstance(
                        gdn.in_proj_qkvz.quant_method, AiterMxfp4OnlineLinearMethod
                    )
                    self.assertIsInstance(
                        gdn.out_proj.quant_method, AiterMxfp4OnlineLinearMethod
                    )
                    self.assertIsInstance(
                        gdn.in_proj_ba.quant_method, UnquantizedLinearMethod
                    )

    def test_online_mxfp4_opt_in_skips_unaligned_and_non_bf16_and_other_models(self):
        with (
            patch.dict(os.environ, {"SGLANG_ROCM_GDN_ONLINE_MXFP4": "1"}),
            patch.object(qwen3_5, "_use_aiter", True),
            patch.object(qwen3_5, "_is_gfx95", True),
        ):
            unaligned = _build_qwen3_5_gdn(hidden_size=128)
            self.assertIsInstance(
                unaligned.in_proj_qkvz.quant_method, UnquantizedLinearMethod
            )
            self.assertIsInstance(
                unaligned.out_proj.quant_method, UnquantizedLinearMethod
            )

            fp16_gdn = _build_qwen3_5_gdn(dtype=torch.float16)
            self.assertIsInstance(
                fp16_gdn.in_proj_qkvz.quant_method, UnquantizedLinearMethod
            )
            self.assertIsInstance(
                fp16_gdn.out_proj.quant_method, UnquantizedLinearMethod
            )

            other_model = _build_qwen3_5_gdn(model_type="qwen4_exp_text")
            self.assertIsInstance(
                other_model.in_proj_qkvz.quant_method, UnquantizedLinearMethod
            )
            self.assertIsInstance(
                other_model.out_proj.quant_method, UnquantizedLinearMethod
            )

    def test_online_mxfp4_requires_gfx95_and_aiter(self):
        with patch.dict(os.environ, {"SGLANG_ROCM_GDN_ONLINE_MXFP4": "1"}):
            for use_aiter, is_gfx95 in ((False, True), (True, False), (False, False)):
                with (
                    self.subTest(use_aiter=use_aiter, is_gfx95=is_gfx95),
                    patch.object(qwen3_5, "_use_aiter", use_aiter),
                    patch.object(qwen3_5, "_is_gfx95", is_gfx95),
                ):
                    self.assertEqual(qwen3_5._get_gdn_online_mxfp4(), ())
                    gdn = _build_qwen3_5_gdn()
                    self.assertIsInstance(
                        gdn.in_proj_qkvz.quant_method, UnquantizedLinearMethod
                    )
                    self.assertIsInstance(
                        gdn.out_proj.quant_method, UnquantizedLinearMethod
                    )


if __name__ == "__main__":
    unittest.main()
