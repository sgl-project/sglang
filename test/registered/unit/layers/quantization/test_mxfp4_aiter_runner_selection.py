"""Unit tests for srt/layers/quantization/mxfp4.py AITER runner selection.

`Mxfp4MoEMethod` pads and shuffles weights for AITER and dispatches to
`_apply_aiter` separately from `create_moe_runner`, so all of them must make
the same AITER decision: an explicit `--moe-runner-backend` wins over
`SGLANG_USE_AITER`, which only applies in auto mode.
"""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import DispatchOutputChecker
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization import mxfp4 as mxfp4_module
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_EXPERTS = 2
_HIDDEN = 64
_INTERMEDIATE = 64
_AITER_ALIGN = 256

# (runner_backend, SGLANG_USE_AITER, a2a_backend, runner is AITER)
_CASES = {
    "triton-aiter_env": ("triton", True, "none", False),
    "triton": ("triton", False, "none", False),
    "aiter": ("aiter", False, "none", True),
    "aiter-aiter_env": ("aiter", True, "none", True),
    "auto-aiter_env": ("auto", True, "none", True),
    "auto": ("auto", False, "none", False),
    "auto-aiter_env-unsupported_a2a": ("auto", True, "flashinfer", False),
}


class TestMxfp4AiterRunnerSelection(CustomTestCase):
    def _build(self, stack: ExitStack, case: str):
        runner_backend, use_aiter, a2a_backend, expect_aiter = _CASES[case]
        for name, value in (
            ("_use_aiter", use_aiter),
            ("_aiter_k3_opt", False),
            ("_is_cpu", False),
            ("_is_xpu", False),
            ("has_triton_kernels", False),
        ):
            stack.enter_context(patch.object(mxfp4_module, name, value))
        stack.enter_context(
            patch.object(
                mxfp4_module,
                "MoeRunner",
                lambda backend, config: SimpleNamespace(
                    runner_backend=backend, run=Mock()
                ),
            )
        )
        stack.enter_context(get_context().override_server_args(model_path="dummy"))
        stack.enter_context(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend(runner_backend),
                a2a_backend=MoeA2ABackend(a2a_backend),
            )
        )
        method = Mxfp4MoEMethod(prefix="model.layers.0.mlp.experts")
        layer = torch.nn.Module()
        layer.hidden_size = _HIDDEN
        layer.intermediate_size_per_partition = _INTERMEDIATE
        layer.num_local_experts = _EXPERTS
        method.create_weights(
            layer,
            num_experts=_EXPERTS,
            hidden_size=_HIDDEN,
            intermediate_size_per_partition=_INTERMEDIATE,
            params_dtype=torch.bfloat16,
        )
        method.create_moe_runner(layer, MoeRunnerConfig(activation="silu"))
        self.assertEqual(method.runner.runner_backend.is_aiter(), expect_aiter)
        return method, layer, expect_aiter

    def test_weight_layout_matches_runner(self):
        for case in _CASES:
            with self.subTest(case=case), ExitStack() as stack:
                _, layer, expect_aiter = self._build(stack, case)
                inter = _AITER_ALIGN if expect_aiter else _INTERMEDIATE
                hidden = _AITER_ALIGN if expect_aiter else _HIDDEN
                expected = (
                    (_EXPERTS, 2 * inter, hidden // 2),
                    (_EXPERTS, hidden, inter // 2),
                )
                shapes = (tuple(layer.w13_weight.shape), tuple(layer.w2_weight.shape))
                self.assertEqual(shapes, expected)

    def test_apply_dispatch_matches_runner(self):
        for case in _CASES:
            for dispatch in ("standard", "deepep"):
                with self.subTest(case=case, dispatch=dispatch), ExitStack() as stack:
                    method, layer, expect_aiter = self._build(stack, case)
                    apply_aiter = stack.enter_context(
                        patch.object(method, "_apply_aiter", return_value="aiter")
                    )
                    stack.enter_context(
                        patch.object(
                            DispatchOutputChecker,
                            "format_is_deepep",
                            return_value=dispatch == "deepep",
                        )
                    )
                    dispatch_output = SimpleNamespace(
                        hidden_states=torch.zeros(1, _HIDDEN), topk_output=None
                    )
                    method.apply(layer, dispatch_output)
                    called = (apply_aiter.called, method.runner.run.called)
                    self.assertEqual(called, (expect_aiter, not expect_aiter))


if __name__ == "__main__":
    unittest.main()
