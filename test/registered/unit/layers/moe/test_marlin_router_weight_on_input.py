"""CPU unit tests for Marlin MoE layers that apply the router weight to the
expert input: the gate that admits them, the refusals with a w1 bias, the
callers that pass the flag to fused_marlin_moe, and the float32 top-k weights
the Marlin GEMM reads."""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.moe import moe_wna16_marlin
from sglang.srt.layers.moe.fused_moe_triton import fused_marlin_moe
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import (
    MarlinMoeQuantInfo,
    fused_experts_none_to_marlin,
)
from sglang.srt.layers.quantization import marlin_utils_fp4, mxfp4
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    CompressedTensorsWNA16MoE,
)
from sglang.srt.layers.quantization.marlin_utils import (
    check_moe_marlin_supports_layer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _runner_config() -> MoeRunnerConfig:
    return MoeRunnerConfig(
        num_experts=4,
        num_local_experts=4,
        hidden_size=256,
        intermediate_size_per_partition=128,
        activation="silu",
        is_gated=True,
        apply_router_weight_on_input=True,
    )


def _dispatch_output(num_tokens: int = 2):
    return SimpleNamespace(
        hidden_states=torch.zeros(num_tokens, 256, dtype=torch.bfloat16),
        topk_output=SimpleNamespace(
            topk_weights=torch.ones(num_tokens, 1),
            topk_ids=torch.zeros(num_tokens, 1, dtype=torch.int32),
            router_logits=torch.zeros(num_tokens, 4),
        ),
    )


def _fake_fused_marlin_moe(seen: dict):
    def fake(*args, **kwargs):
        seen.update(kwargs)
        return kwargs.get("hidden_states", args[0] if args else None)

    return fake


class TestMarlinRouterWeightOnInput(CustomTestCase):
    def test_gate_admits_router_weight_on_input_without_w13_bias(self):
        """The Marlin runner weights GEMM1's rows after adding the w13 bias, so
        only a layer without one can take its router weight on the input."""
        layer = SimpleNamespace(
            hidden_size=256,
            intermediate_size_per_partition=128,
            moe_runner_config=_runner_config(),
        )
        self.assertTrue(check_moe_marlin_supports_layer(layer, 128))
        self.assertFalse(check_moe_marlin_supports_layer(layer, 128, has_w13_bias=True))
        layer.moe_runner_config.apply_router_weight_on_input = False
        self.assertTrue(check_moe_marlin_supports_layer(layer, 128, has_w13_bias=True))

    def test_mxfp4_marlin_refuses_router_weight_on_input(self):
        """Mxfp4MoEMethod's Marlin path always carries a w13 bias, so a layer
        with its router weight on the input stops at weight prep."""
        method = mxfp4.Mxfp4MoEMethod.__new__(mxfp4.Mxfp4MoEMethod)
        method.use_marlin, method.use_mega_moe = True, False
        method.moe_runner_config = _runner_config()
        layer = SimpleNamespace(
            hidden_size=256,
            intermediate_size_per_partition=128,
            moe_runner_config=method.moe_runner_config,
        )
        platform = SimpleNamespace(is_sm90=True, is_sm100=False, is_sm120=False)
        with (
            mock.patch.object(
                mxfp4,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_mori=lambda: False),
            ),
            mock.patch.object(mxfp4, "get_platform", return_value=platform),
            mock.patch.object(marlin_utils_fp4, "prepare_moe_mxfp4_layer_for_marlin"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not supported by Marlin"):
                method.process_weights_after_loading(layer)

    def test_fused_marlin_moe_refuses_router_weight_on_input_with_w1_bias(self):
        """Marlin adds the w1 bias before it weights GEMM1's rows, so the call
        stops before any kernel runs."""
        # The unwrapped function: the registered op has no CPU kernel.
        fused = inspect.unwrap(fused_marlin_moe.fused_marlin_moe)
        with self.assertRaisesRegex(AssertionError, "w1 bias"):
            fused(
                hidden_states=torch.zeros(1, 64, dtype=torch.bfloat16),
                w1=torch.zeros(1, 4, 8, dtype=torch.int32),
                w2=torch.zeros(1, 2, 128, dtype=torch.int32),
                w1_scale=torch.zeros(1),
                w2_scale=torch.zeros(1),
                gating_output=torch.zeros(1, 1),
                topk_weights=torch.ones(1, 1),
                topk_ids=torch.zeros(1, 1, dtype=torch.int32),
                w1_bias=torch.zeros(1, 32, dtype=torch.bfloat16),
                num_bits=4,
                apply_router_weight_on_input=True,
            )

    def test_marlin_runner_passes_the_flag(self):
        seen = {}
        quant_info = MarlinMoeQuantInfo(
            w13_qweight=None,
            w2_qweight=None,
            w13_scales=torch.zeros(1),
            w2_scales=torch.zeros(1),
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
        )
        with (
            mock.patch.object(
                fused_marlin_moe, "fused_marlin_moe", _fake_fused_marlin_moe(seen)
            ),
            mock.patch(
                "sglang.srt.layers.quantization.marlin_utils.marlin_make_workspace"
            ),
        ):
            fused_experts_none_to_marlin(
                _dispatch_output(), quant_info, _runner_config()
            )
        self.assertIs(seen["apply_router_weight_on_input"], True)

    def test_compressed_tensors_wna16_passes_the_flag(self):
        """This scheme calls fused_marlin_moe directly, not through the runner."""
        method = CompressedTensorsWNA16MoE.__new__(CompressedTensorsWNA16MoE)
        method.moe_runner_config = _runner_config()
        method.num_bits, method.sym, method.is_k_full = 4, True, True
        layer = SimpleNamespace(
            **{
                name: None
                for name in (
                    "w13_weight_packed",
                    "w2_weight_packed",
                    "w13_weight_scale",
                    "w2_weight_scale",
                    "w13_weight_g_idx",
                    "w2_weight_g_idx",
                    "w13_g_idx_sort_indices",
                    "w2_g_idx_sort_indices",
                    "workspace",
                )
            }
        )
        dispatch_output = _dispatch_output()
        topk = dispatch_output.topk_output
        dispatch_output.topk_output = (
            topk.topk_weights,
            topk.topk_ids,
            topk.router_logits,
        )
        seen = {}
        with mock.patch.object(
            fused_marlin_moe, "fused_marlin_moe", _fake_fused_marlin_moe(seen)
        ):
            method.apply_weights(layer, dispatch_output)
        self.assertIs(seen["apply_router_weight_on_input"], True)

    def test_marlin_gemm_reads_float32_top_k_weights(self):
        """The kernel reads the top-k weights as a flat float32 array whatever
        dtype the router returned, and only when it multiplies by them."""
        weights = torch.rand(3, 4, dtype=torch.bfloat16).t()
        for mul_topk_weights in (True, False):
            with self.subTest(mul_topk_weights=mul_topk_weights):
                module = mock.MagicMock()
                with mock.patch.object(
                    moe_wna16_marlin,
                    "_jit_moe_wna16_marlin_module",
                    return_value=module,
                ):
                    moe_wna16_marlin.moe_wna16_marlin_gemm(
                        torch.zeros(4, 64, dtype=torch.bfloat16),
                        torch.zeros(12, 64, dtype=torch.bfloat16),
                        torch.zeros(1),
                        None,
                        torch.zeros(1, 1),
                        None,
                        None,
                        None,
                        None,
                        torch.zeros(1),
                        torch.zeros(16, dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        torch.zeros(1, dtype=torch.int32),
                        weights,
                        moe_block_size=16,
                        top_k=3,
                        mul_topk_weights=mul_topk_weights,
                        is_ep=False,
                        b_q_type=SimpleNamespace(id=0),
                        size_m=4,
                        size_n=64,
                        size_k=64,
                    )
                passed = module.moe_wna16_marlin_gemm.call_args.args[13]
                if mul_topk_weights:
                    self.assertEqual(passed.dtype, torch.float32)
                    self.assertTrue(passed.is_contiguous())
                    self.assertTrue(torch.equal(passed, weights.float()))
                else:
                    self.assertIs(passed, weights)


if __name__ == "__main__":
    unittest.main()
