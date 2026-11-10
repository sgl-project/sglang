"""-1 drop-sentinel handling at the sgl-kernel-xpu fused_experts call sites."""

import sys
import types
import unittest
import unittest.mock
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig  # noqa: E402
from sglang.srt.layers.moe.moe_runner.triton_utils import (  # noqa: E402
    fused_moe as fused_moe_module,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput  # noqa: E402
from sglang.srt.layers.moe.utils import route_dropped_experts_to_local  # noqa: E402
from sglang.srt.layers.quantization import fp8 as fp8_module  # noqa: E402
from sglang.srt.layers.quantization import mxfp4 as mxfp4_module  # noqa: E402
from sglang.srt.layers.quantization import unquant as unquant_module  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_TOKENS = 8, 16, 12, 4


class TestRouteDroppedExpertsToLocal(CustomTestCase):
    def test_dropped_slots_become_local_expert_zero_at_weight_zero(self) -> None:
        for id_dtype in (torch.int32, torch.int64):
            for weight_dtype in (torch.float32, torch.bfloat16):
                with self.subTest(id_dtype=id_dtype, weight_dtype=weight_dtype):
                    topk_ids = torch.tensor([[3, -1, 5], [-1, -1, 2]], dtype=id_dtype)
                    topk_weights = torch.tensor(
                        [[0.5, 0.3, 0.2], [0.6, 0.3, 0.1]], dtype=weight_dtype
                    )
                    ids_before, weights_before = topk_ids.clone(), topk_weights.clone()
                    new_weights, new_ids = route_dropped_experts_to_local(
                        topk_weights=topk_weights, topk_ids=topk_ids
                    )
                    dropped = ids_before < 0
                    self.assertEqual(new_ids.dtype, id_dtype)
                    self.assertEqual(new_weights.dtype, weight_dtype)
                    self.assertTrue(bool((new_ids[dropped] == 0).all()))
                    self.assertTrue(bool((new_weights[dropped] == 0).all()))
                    self.assertTrue(
                        torch.equal(new_ids[~dropped], ids_before[~dropped])
                    )
                    self.assertTrue(
                        torch.equal(new_weights[~dropped], weights_before[~dropped])
                    )
                    # the dispatcher's tensors stay as they were
                    self.assertTrue(torch.equal(topk_ids, ids_before))
                    self.assertTrue(torch.equal(topk_weights, weights_before))


class TestXpuCallSitesNeverPassTheSentinel(CustomTestCase):
    """Each XPU call site, with the XPU branch forced on and fused_experts mocked."""

    def setUp(self) -> None:
        self.kernel_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def fused_experts(
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            **kernel_options: object,
        ) -> torch.Tensor:
            self.kernel_calls.append((hidden_states, topk_weights, topk_ids))
            return torch.zeros_like(hidden_states)

        self.fake_sgl_kernel = types.ModuleType("sgl_kernel")
        self.fake_sgl_kernel.fused_experts = fused_experts
        self.hidden_states = torch.randn(NUM_TOKENS, HIDDEN_SIZE)
        self.layer = SimpleNamespace(
            w13_weight=torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
            w2_weight=torch.zeros(NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            w13_weight_scale=None,
            w2_weight_scale=None,
        )

    def routing(self, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        topk_weights = torch.full((NUM_TOKENS, top_k), 1.0 / top_k)
        topk_ids = torch.arange(NUM_TOKENS * top_k).reshape(NUM_TOKENS, top_k)
        topk_ids = topk_ids % NUM_EXPERTS
        topk_ids[::2, 0] = -1
        return topk_weights, topk_ids

    def topk_output(
        self, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> StandardTopKOutput:
        return StandardTopKOutput(
            topk_weights, topk_ids, torch.zeros(NUM_TOKENS, NUM_EXPERTS)
        )

    def dispatch_output(
        self, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> SimpleNamespace:
        return SimpleNamespace(
            hidden_states=self.hidden_states,
            topk_output=self.topk_output(topk_weights, topk_ids),
        )

    def xpu_branch(self, module: ModuleType, **module_flags: bool) -> ExitStack:
        patches = ExitStack()
        patches.enter_context(
            unittest.mock.patch.dict(sys.modules, {"sgl_kernel": self.fake_sgl_kernel})
        )
        non_triton_backend = SimpleNamespace(is_triton=lambda: False)
        patches.enter_context(
            unittest.mock.patch.object(
                module, "get_moe_runner_backend", lambda: non_triton_backend
            )
        )
        for flag_name, flag_value in module_flags.items():
            patches.enter_context(
                unittest.mock.patch.object(module, flag_name, flag_value)
            )
        return patches

    def assert_kernel_saw_no_sentinel(self, topk_ids: torch.Tensor) -> None:
        self.assertEqual(len(self.kernel_calls), 1)
        _, kernel_weights, kernel_ids = self.kernel_calls[0]
        dropped = topk_ids < 0
        self.assertFalse(bool((kernel_ids < 0).any()), "a -1 reached fused_experts")
        self.assertTrue(bool((kernel_weights[dropped] == 0).all()))
        self.assertTrue(torch.equal(kernel_ids[~dropped], topk_ids[~dropped]))

    def test_fused_moe(self) -> None:
        topk_weights, topk_ids = self.routing(top_k=2)
        with self.xpu_branch(fused_moe_module, _is_xpu=True):
            fused_moe_module.fused_moe(
                self.hidden_states,
                self.layer.w13_weight,
                self.layer.w2_weight,
                self.topk_output(topk_weights, topk_ids),
            )
        self.assert_kernel_saw_no_sentinel(topk_ids)

    def test_unquant_forward_xpu(self) -> None:
        for apply_on_input, top_k in ((False, 2), (True, 1)):
            with self.subTest(apply_router_weight_on_input=apply_on_input):
                self.kernel_calls.clear()
                topk_weights, topk_ids = self.routing(top_k=top_k)
                moe_method = SimpleNamespace(
                    moe_runner_config=MoeRunnerConfig(
                        activation="silu", apply_router_weight_on_input=apply_on_input
                    ),
                    runner=SimpleNamespace(runner_backend=None),
                )
                # unquant has no _is_xpu flag; forward_xpu is the XPU path
                with self.xpu_branch(unquant_module):
                    unquant_module.UnquantizedFusedMoEMethod.forward_xpu(
                        moe_method,
                        self.layer,
                        self.dispatch_output(topk_weights, topk_ids),
                    )
                self.assert_kernel_saw_no_sentinel(topk_ids)
                if apply_on_input:
                    kernel_hidden_states, kernel_weights, _ = self.kernel_calls[0]
                    torch.testing.assert_close(
                        kernel_hidden_states, self.hidden_states * topk_weights
                    )
                    self.assertTrue(
                        torch.equal(
                            kernel_weights, (topk_ids >= 0).to(topk_weights.dtype)
                        )
                    )

    def test_fp8_apply(self) -> None:
        topk_weights, topk_ids = self.routing(top_k=2)
        moe_method = SimpleNamespace(
            moe_runner_config=MoeRunnerConfig(activation="silu"),
            is_fp4_expert=False,
            block_quant=False,
        )
        with self.xpu_branch(fp8_module, _is_xpu=True, _is_npu=False, _is_hip=False):
            with unittest.mock.patch.object(
                fp8_module, "use_intel_amx_backend", lambda layer: False
            ):
                fp8_module.Fp8MoEMethod.apply(
                    moe_method, self.layer, self.dispatch_output(topk_weights, topk_ids)
                )
        self.assert_kernel_saw_no_sentinel(topk_ids)

    def test_mxfp4_apply(self) -> None:
        topk_weights, topk_ids = self.routing(top_k=2)
        moe_method = SimpleNamespace(
            moe_runner_config=MoeRunnerConfig(activation="silu"),
            use_deep_gemm=False,
        )
        with self.xpu_branch(
            mxfp4_module, _is_xpu=True, _is_cpu=False, _use_aiter=False
        ):
            mxfp4_module.Mxfp4MoEMethod.apply(
                moe_method, self.layer, self.dispatch_output(topk_weights, topk_ids)
            )
        self.assert_kernel_saw_no_sentinel(topk_ids)


if __name__ == "__main__":
    unittest.main()
