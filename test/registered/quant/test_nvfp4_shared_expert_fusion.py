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
"""Unit tests for FP4/NVFP4 shared-expert fusion in the FlashInfer TRTLLM path.

No GPU compute is needed: every FlashInfer kernel call is replaced by a
unittest.mock stub.  CUDA must be importable (flashinfer loads GPU extensions at
import time), so the tests run on a 1-GPU runner, but they allocate no device
memory and launch no kernels.
"""

import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

# Break the circular import: sglang.srt.layers.quantization imports from
# flashinfer_trtllm at module level; ensuring it is fully initialised first
# prevents the "partially initialised module" ImportError.
import sglang.srt.layers.quantization  # noqa: F401

from sglang.srt.arg_groups.overrides import (
    _fp4_trtllm_supports_fused_shared,
    _moe_runner_fusion_disable,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as _ft_mod
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    FlashInferTrtllmFp4MoeQuantInfo,
    fused_experts_none_to_flashinfer_trtllm_fp4,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NUM_EXPERTS = 4
_HIDDEN = 16
_INTERMEDIATE = 32
_NUM_TOKENS = 3
_TOP_K = 2


def _make_quant_info(
    *,
    local_num_experts: int = _NUM_EXPERTS,
    global_num_experts: int = _NUM_EXPERTS,
) -> FlashInferTrtllmFp4MoeQuantInfo:
    """Minimal CPU dummy for FlashInferTrtllmFp4MoeQuantInfo."""
    N = global_num_experts
    H = _HIDDEN
    I = _INTERMEDIATE
    return FlashInferTrtllmFp4MoeQuantInfo(
        # FP4-packed weight: each byte encodes 2 FP4 values along the K-dim.
        w13_weight=torch.zeros(N, 2 * I // 2, H // 2, dtype=torch.uint8),
        w2_weight=torch.zeros(N, H, I // 2, dtype=torch.uint8),
        # Block scales: uint8 layout that .view(float8_e4m3fn) is valid.
        w13_weight_scale=torch.zeros(N, 2 * I, H // 16, dtype=torch.uint8),
        w2_weight_scale=torch.zeros(N, H, I // 16, dtype=torch.uint8),
        g1_scale_c=torch.ones(N, dtype=torch.float32),
        g1_alphas=torch.ones(N, dtype=torch.float32),
        g2_alphas=torch.ones(N, dtype=torch.float32),
        w13_input_scale_quant=torch.ones(1, dtype=torch.float32),
        global_num_experts=global_num_experts,
        local_expert_offset=0,
        local_num_experts=local_num_experts,
        intermediate_size_per_partition=I,
        routing_method_type=0,
    )


def _make_runner_config(*, num_fused_shared_experts: int = 0) -> MoeRunnerConfig:
    return MoeRunnerConfig(
        activation="silu",
        is_gated=True,
        no_combine=False,
        num_fused_shared_experts=num_fused_shared_experts,
    )


def _make_dispatch_output_routed(*, num_tokens: int = _NUM_TOKENS, top_k: int = _TOP_K):
    """Dispatch output shaped for the pre-quantised (NVFP4 A2A) routed path.

    Setting hidden_states_scale to a non-None float8 tensor causes the
    function to skip runtime quantisation (uses pre-quantised inputs).
    The topk_output carries a packed_topk_ids tensor so
    _get_packed_topk_ids_for_flashinfer_routed returns it directly.
    """
    T, H = num_tokens, _HIDDEN
    hs_fp4 = torch.zeros(T, H // 2, dtype=torch.uint8)
    hs_scale = torch.zeros(T, H // 16, dtype=torch.float8_e4m3fn)
    topk_output = SimpleNamespace(
        packed_topk_ids=torch.zeros(T, top_k, dtype=torch.int16)
    )
    return SimpleNamespace(
        hidden_states=hs_fp4,
        hidden_states_scale=hs_scale,
        topk_output=topk_output,
    )


def _make_dispatch_output_bypassed(*, num_tokens: int = _NUM_TOKENS):
    """Dispatch output shaped for the bypassed (router-logits) path.

    Uses a real BypassedTopKOutput so TopKOutputChecker.format_is_bypassed
    returns True and format_is_standard returns False.
    """
    from sglang.srt.layers.moe.topk import BypassedTopKOutput, TopKConfig

    T, H, N = num_tokens, _HIDDEN, _NUM_EXPERTS
    # Pre-quantised path: non-None hidden_states_scale → skip fp4_quantize.
    hs_fp4 = torch.zeros(T, H // 2, dtype=torch.uint8)
    hs_scale = torch.zeros(T, H // 16, dtype=torch.float8_e4m3fn)
    topk_cfg = TopKConfig(
        top_k=_TOP_K,
        topk_group=0,
        num_expert_group=0,
        correction_bias=None,
    )
    topk_output = BypassedTopKOutput(
        hidden_states=torch.zeros(T, H),
        router_logits=torch.zeros(T, N),
        topk_config=topk_cfg,
    )
    return SimpleNamespace(
        hidden_states=hs_fp4,
        hidden_states_scale=hs_scale,
        topk_output=topk_output,
    )


# ---------------------------------------------------------------------------
# Tests 1–2: _fp4_trtllm_supports_fused_shared
# ---------------------------------------------------------------------------


class TestFp4TrtllmSupportsFusedShared(CustomTestCase):
    """Guard _fp4_trtllm_supports_fused_shared against signature changes in
    the installed FlashInfer.  A regression here would silently prevent
    shared-expert fusion from being enabled even on a new-enough FlashInfer."""

    def _stub_without_param(self, a):
        """Stand-in for a FlashInfer FP4 function that lacks the new kwarg."""

    def _stub_with_param(self, a, num_fused_shared_experts=0):
        """Stand-in for a FlashInfer FP4 function that exposes the new kwarg."""

    def test_returns_false_when_param_absent(self):
        """False when neither FP4 FlashInfer function has num_fused_shared_experts."""
        with patch(
            "flashinfer.fused_moe.trtllm_fp4_block_scale_moe",
            self._stub_without_param,
        ), patch(
            "flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe",
            self._stub_without_param,
        ):
            self.assertFalse(_fp4_trtllm_supports_fused_shared())

    def test_returns_true_when_both_fp4_functions_have_param(self):
        """True when both FP4 FlashInfer functions expose num_fused_shared_experts."""
        with patch(
            "flashinfer.fused_moe.trtllm_fp4_block_scale_moe",
            self._stub_with_param,
        ), patch(
            "flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe",
            self._stub_with_param,
        ):
            self.assertTrue(_fp4_trtllm_supports_fused_shared())


# ---------------------------------------------------------------------------
# Tests 3–5: _moe_runner_fusion_disable
# ---------------------------------------------------------------------------


class TestMoeRunnerFusionDisable(CustomTestCase):
    """Guard the logic that decides when to auto-set disable_shared_experts_fusion.

    Regressions here would silently enable fused shared experts on a FlashInfer
    build that does not support them, or silently disable them on one that does.
    """

    def _view(self, *, backend: str, quantization: str) -> SimpleNamespace:
        return SimpleNamespace(moe_runner_backend=backend, quantization=quantization)

    def test_auto_disables_for_non_fp4_quantization(self):
        """Non-FP4 quant (e.g. fp8_w8a8) must always set disable_shared_experts_fusion."""
        view = self._view(backend="flashinfer_trtllm", quantization="fp8_w8a8")
        result = _moe_runner_fusion_disable(view)
        self.assertTrue(result.get("disable_shared_experts_fusion"))

    def test_does_not_disable_for_fp4_when_flashinfer_supports(self):
        """modelopt_fp4 + FlashInfer support → no auto-disable (empty dict)."""
        with patch(
            "sglang.srt.arg_groups.overrides._fp4_trtllm_supports_fused_shared",
            return_value=True,
        ):
            view = self._view(backend="flashinfer_trtllm", quantization="modelopt_fp4")
            result = _moe_runner_fusion_disable(view)
            self.assertEqual(result, {})

    def test_still_disables_for_fp4_when_flashinfer_lacks_support(self):
        """modelopt_fp4 but FlashInfer lacks support → must still auto-disable."""
        with patch(
            "sglang.srt.arg_groups.overrides._fp4_trtllm_supports_fused_shared",
            return_value=False,
        ):
            view = self._view(backend="flashinfer_trtllm", quantization="modelopt_fp4")
            result = _moe_runner_fusion_disable(view)
            self.assertTrue(result.get("disable_shared_experts_fusion"))

    def test_does_not_disable_for_fp4_routed_when_flashinfer_supports(self):
        """flashinfer_trtllm_routed + modelopt_fp4 + support → no auto-disable."""
        with patch(
            "sglang.srt.arg_groups.overrides._fp4_trtllm_supports_fused_shared",
            return_value=True,
        ):
            view = self._view(
                backend="flashinfer_trtllm_routed", quantization="modelopt_fp4"
            )
            result = _moe_runner_fusion_disable(view)
            self.assertEqual(result, {})


# ---------------------------------------------------------------------------
# Tests 6–8: fused_experts_none_to_flashinfer_trtllm_fp4 runtime guards
# ---------------------------------------------------------------------------


class TestFusedExpertsFp4RuntimeGuards(CustomTestCase):
    """Guard the early-exit checks inside fused_experts_none_to_flashinfer_trtllm_fp4.

    These checks protect against calling a FlashInfer build that lacks the
    num_fused_shared_experts parameter (RuntimeError) or calling the fused path
    with expert-parallelism active (NotImplementedError).
    """

    def test_raises_runtime_error_when_no_flashinfer_support(self):
        """RuntimeError when num_fused_shared_experts>0 but FlashInfer lacks the param."""
        runner_config = _make_runner_config(num_fused_shared_experts=1)
        quant_info = MagicMock()
        quant_info.local_num_experts = _NUM_EXPERTS
        quant_info.global_num_experts = _NUM_EXPERTS
        dispatch_output = MagicMock()

        with patch.object(_ft_mod, "_FP4_TRTLLM_HAS_FUSED_SHARED", False):
            with self.assertRaises(RuntimeError) as ctx:
                fused_experts_none_to_flashinfer_trtllm_fp4(
                    dispatch_output, quant_info, runner_config
                )
        self.assertIn("FlashInfer", str(ctx.exception))
        self.assertIn("num_fused_shared_experts", str(ctx.exception))

    def test_raises_not_implemented_for_ep_with_fused_shared(self):
        """NotImplementedError when fused shared experts requested under EP."""
        runner_config = _make_runner_config(num_fused_shared_experts=1)
        quant_info = MagicMock()
        quant_info.local_num_experts = 32
        quant_info.global_num_experts = 256  # EP: local < global
        dispatch_output = MagicMock()

        with patch.object(_ft_mod, "_FP4_TRTLLM_HAS_FUSED_SHARED", True):
            with self.assertRaises(NotImplementedError) as ctx:
                fused_experts_none_to_flashinfer_trtllm_fp4(
                    dispatch_output, quant_info, runner_config
                )
        # The message must mention expert parallelism so operators understand why.
        self.assertIn("expert", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Tests 8a–8b: num_fused_shared_experts reaches FlashInfer kwargs
# ---------------------------------------------------------------------------


class TestFusedExpertsFp4KwargsPassthrough(CustomTestCase):
    """Guard that num_fused_shared_experts is injected into the FlashInfer call.

    A regression here would silently drop the field and the kernel would not
    fuse the shared expert, producing wrong numerics at inference time.
    """

    # Module-level patch target for the module-global bool.
    _MOD_BOOL = "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm._FP4_TRTLLM_HAS_FUSED_SHARED"
    _MOD_ROUTED_BOOL = "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm._FP4_TRTLLM_ROUTED_HAS_FUSED_SHARED"
    # Module-level patch targets for distributed helpers imported by the module.
    _GET_TP = "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm.get_tp_group"

    def _common_patches(self):
        """Context manager that patches the boolean flag and distributed helpers."""
        return [
            patch(self._MOD_BOOL, True),
            patch(self._GET_TP, return_value=MagicMock(world_size=1)),
        ]

    def test_num_fused_shared_experts_kwarg_routed_path(self):
        """num_fused_shared_experts=1 is forwarded to trtllm_fp4_block_scale_routed_moe
        when _FP4_TRTLLM_ROUTED_HAS_FUSED_SHARED is True (future FlashInfer)."""
        runner_config = _make_runner_config(num_fused_shared_experts=1)
        quant_info = _make_quant_info()
        dispatch_output = _make_dispatch_output_routed()

        mock_routed = MagicMock(return_value=[torch.zeros(_NUM_TOKENS, _HIDDEN)])

        with patch(self._MOD_BOOL, True), patch(
            self._MOD_ROUTED_BOOL, True
        ), patch(
            self._GET_TP, return_value=MagicMock(world_size=1)
        ), patch(
            "flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe", mock_routed
        ):
            fused_experts_none_to_flashinfer_trtllm_fp4(
                dispatch_output, quant_info, runner_config, use_routed_topk=True
            )

        self.assertTrue(mock_routed.called, "FlashInfer routed kernel was not called")
        called_kwargs = mock_routed.call_args.kwargs
        self.assertIn(
            "num_fused_shared_experts",
            called_kwargs,
            "num_fused_shared_experts missing from routed FlashInfer call",
        )
        self.assertEqual(called_kwargs["num_fused_shared_experts"], 1)

    def test_num_fused_shared_experts_kwarg_routed_path_absent_when_unsupported(self):
        """num_fused_shared_experts must NOT be passed to routed kernel when
        _FP4_TRTLLM_ROUTED_HAS_FUSED_SHARED is False (current FlashInfer nightly)."""
        runner_config = _make_runner_config(num_fused_shared_experts=1)
        quant_info = _make_quant_info()
        dispatch_output = _make_dispatch_output_routed()

        mock_routed = MagicMock(return_value=[torch.zeros(_NUM_TOKENS, _HIDDEN)])

        with patch(self._MOD_BOOL, True), patch(
            self._MOD_ROUTED_BOOL, False
        ), patch(
            self._GET_TP, return_value=MagicMock(world_size=1)
        ), patch(
            "flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe", mock_routed
        ):
            fused_experts_none_to_flashinfer_trtllm_fp4(
                dispatch_output, quant_info, runner_config, use_routed_topk=True
            )

        called_kwargs = mock_routed.call_args.kwargs
        self.assertNotIn(
            "num_fused_shared_experts",
            called_kwargs,
            "num_fused_shared_experts must not be passed when routed kernel lacks support",
        )

    def test_num_fused_shared_experts_kwarg_bypassed_path(self):
        """num_fused_shared_experts=1 is forwarded to trtllm_fp4_block_scale_moe (bypassed)."""
        runner_config = _make_runner_config(num_fused_shared_experts=1)
        quant_info = _make_quant_info()
        dispatch_output = _make_dispatch_output_bypassed()

        mock_moe = MagicMock(return_value=[torch.zeros(_NUM_TOKENS, _HIDDEN)])

        with patch(self._MOD_BOOL, True), patch(
            self._GET_TP, return_value=MagicMock(world_size=1)
        ), patch("flashinfer.fused_moe.trtllm_fp4_block_scale_moe", mock_moe):
            fused_experts_none_to_flashinfer_trtllm_fp4(
                dispatch_output, quant_info, runner_config, use_routed_topk=False
            )

        self.assertTrue(mock_moe.called, "FlashInfer bypassed kernel was not called")
        called_kwargs = mock_moe.call_args.kwargs
        self.assertIn(
            "num_fused_shared_experts",
            called_kwargs,
            "num_fused_shared_experts missing from bypassed FlashInfer call",
        )
        self.assertEqual(called_kwargs["num_fused_shared_experts"], 1)

    def test_kwarg_absent_when_num_fused_shared_is_zero(self):
        """num_fused_shared_experts must NOT appear when fusion is inactive (0)."""
        runner_config = _make_runner_config(num_fused_shared_experts=0)
        quant_info = _make_quant_info()
        dispatch_output = _make_dispatch_output_routed()

        mock_routed = MagicMock(return_value=[torch.zeros(_NUM_TOKENS, _HIDDEN)])

        with patch(self._MOD_BOOL, True), patch(
            self._GET_TP, return_value=MagicMock(world_size=1)
        ), patch(
            "flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe", mock_routed
        ):
            fused_experts_none_to_flashinfer_trtllm_fp4(
                dispatch_output, quant_info, runner_config, use_routed_topk=True
            )

        called_kwargs = mock_routed.call_args.kwargs
        self.assertNotIn(
            "num_fused_shared_experts",
            called_kwargs,
            "num_fused_shared_experts should not appear when fusion is inactive",
        )


if __name__ == "__main__":
    unittest.main()
