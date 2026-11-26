"""The GLM-5.2 router's sort reaches AITER only through the experts call it was
written for: the AITER runner takes it at the top of its call and registers it
right before its fused_moe."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import zero_copy_context
from sglang.srt.layers.moe import glm_aiter_presort
from sglang.srt.layers.moe.moe_runner import aiter as aiter_runner
from sglang.srt.layers.moe.moe_runner.aiter import (
    AiterMoeQuantInfo,
    AiterQuantType,
    AiterRunnerCore,
    AiterRunnerInput,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOPK = 9
MODEL_DIM = 8


def _runner_input(m=2):
    return AiterRunnerInput(
        hidden_states=torch.zeros((m, MODEL_DIM), dtype=torch.bfloat16),
        topk_ids=torch.zeros((m, TOPK), dtype=torch.int32),
        topk_weights=torch.zeros((m, TOPK), dtype=torch.float32),
        quant_type=AiterQuantType.PER_1X32,
    )


def _quant_info():
    return AiterMoeQuantInfo(
        w13_weight=torch.empty((2, 8, 2)),
        w2_weight=torch.empty((2, MODEL_DIM, 2)),
        quant_type=AiterQuantType.PER_1X32,
    )


def _staged_for(runner_input):
    # What route_and_stage keeps, with the sort tensors left out.
    moe_buf = torch.zeros(
        (runner_input.hidden_states.shape[0], MODEL_DIM), dtype=torch.bfloat16
    )
    return {
        "layout": "plain",
        "topk_ids": runner_input.topk_ids,
        "topk_weights": runner_input.topk_weights,
        "moe_buf": moe_buf,
        "moe_buf_zeroed": True,
        "moe_buf_ptr": moe_buf.data_ptr(),
    }


class _FakeAiter:
    """aiter.fused_moe with a fused_moe that records what was registered when it
    ran and the output it was given."""

    def __init__(self):
        self.registrations = []
        self.calls = []

    def fused_moe(self, hidden_states, output=None, **kwargs):
        self.calls.append((list(self.registrations), output))
        return hidden_states if output is None else output

    def install(self, test):
        fake_aiter = ModuleType("aiter")
        fake_aiter.__path__ = []
        fake_aiter.ActivationType = SimpleNamespace(Silu="Silu")
        fake_aiter.QuantType = SimpleNamespace(per_1x32="per_1x32")
        fake_fused_moe = ModuleType("aiter.fused_moe")
        fake_fused_moe.fused_moe = self.fused_moe
        for patcher in (
            patch.dict(
                sys.modules, {"aiter": fake_aiter, "aiter.fused_moe": fake_fused_moe}
            ),
            patch.object(
                glm_aiter_presort, "_registrar", lambda: self.registrations.append
            ),
            patch.object(
                aiter_runner, "_aiter_fused_moe_supports_output", lambda: True
            ),
        ):
            patcher.start()
            test.addCleanup(patcher.stop)


class TestStagedSortHandoff(CustomTestCase):
    def setUp(self):
        self.addCleanup(setattr, glm_aiter_presort, "_staged", None)

    def test_take_returns_the_sort_once_for_its_own_tensors(self):
        runner_input = _runner_input()
        staged = _staged_for(runner_input)
        glm_aiter_presort._staged = staged
        self.assertIs(
            glm_aiter_presort.take_staged(
                runner_input.topk_ids, runner_input.topk_weights
            ),
            staged,
        )
        self.assertIsNone(
            glm_aiter_presort.take_staged(
                runner_input.topk_ids, runner_input.topk_weights
            )
        )

    def test_take_drops_a_sort_staged_for_other_tensors(self):
        runner_input = _runner_input()
        glm_aiter_presort._staged = _staged_for(_runner_input())
        self.assertIsNone(
            glm_aiter_presort.take_staged(
                runner_input.topk_ids, runner_input.topk_weights
            )
        )
        self.assertIsNone(glm_aiter_presort._staged)

    def test_take_accepts_a_view_of_the_same_tensors(self):
        """The runner may see .to() or view results of the router's tensors."""
        runner_input = _runner_input()
        staged = _staged_for(runner_input)
        glm_aiter_presort._staged = staged
        self.assertIs(
            glm_aiter_presort.take_staged(
                runner_input.topk_ids.view(runner_input.topk_ids.shape),
                runner_input.topk_weights.to(torch.float32),
            ),
            staged,
        )


class TestAiterRunnerRegistersTheStagedSort(CustomTestCase):
    def setUp(self):
        self.aiter = _FakeAiter()
        self.aiter.install(self)
        self.runner = AiterRunnerCore(MoeRunnerConfig(activation="silu"))
        self.addCleanup(setattr, glm_aiter_presort, "_staged", None)

    def test_registers_right_before_fused_moe_and_passes_the_zeroed_output(self):
        runner_input = _runner_input()
        staged = _staged_for(runner_input)
        glm_aiter_presort._staged = staged
        self.runner.run(runner_input, _quant_info(), running_state={})
        ((registered_then, output),) = self.aiter.calls
        self.assertEqual(len(registered_then), 1)
        self.assertIs(registered_then[0], staged)
        self.assertIs(output, staged["moe_buf"])
        self.assertIsNone(glm_aiter_presort._staged)

    def test_a_sort_for_other_tensors_is_never_registered(self):
        glm_aiter_presort._staged = _staged_for(_runner_input())
        self.runner.run(_runner_input(), _quant_info(), running_state={})
        ((registered_then, output),) = self.aiter.calls
        self.assertEqual(registered_then, [])
        self.assertIsNone(output)

    def test_a_call_that_skips_fused_moe_drops_the_sort(self):
        """Nothing is registered, so no later fused_moe can pick the sort up."""
        runner_input = _runner_input(m=0)
        glm_aiter_presort._staged = _staged_for(runner_input)
        self.runner.run(runner_input, _quant_info(), running_state={})
        self.assertEqual(self.aiter.calls, [])
        self.assertEqual(self.aiter.registrations, [])
        self.assertIsNone(glm_aiter_presort._staged)

    def test_a_published_output_wins_over_the_zeroed_one(self):
        """AITER then zeroes the published buffer itself: the registered
        moe_buf_ptr names the router's buffer, not this one."""
        runner_input = _runner_input()
        staged = _staged_for(runner_input)
        glm_aiter_presort._staged = staged
        published = torch.ones((2, MODEL_DIM), dtype=torch.bfloat16)
        with zero_copy_context.set_moe_output(published):
            self.runner.run(runner_input, _quant_info(), running_state={})
        ((registered_then, output),) = self.aiter.calls
        self.assertIs(registered_then[0], staged)
        self.assertIs(output, published)


class TestKernelReady(CustomTestCase):
    def setUp(self):
        for target, value in (
            ("covered", lambda scores, bias: True),
            ("available", lambda: True),
            ("ready", lambda device, m: True),
        ):
            patcher = patch.object(glm_aiter_presort.glm_route_sort, target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(glm_aiter_presort, "_registrar", lambda: print)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _ready(self, a2a=MoeA2ABackend.NONE, compiling=False):
        with (
            glm_aiter_presort.envs.SGLANG_OPT_HIP_GLM_ROUTE_SORT.override(True),
            patch.object(glm_aiter_presort, "get_moe_a2a_backend", lambda: a2a),
            patch.object(torch.compiler, "is_compiling", lambda: compiling),
        ):
            return glm_aiter_presort.kernel_ready(
                torch.zeros((4, 256)), torch.zeros(256)
            )

    def test_ready_on_the_standard_dispatcher(self):
        self.assertTrue(self._ready())

    def test_off_under_an_all_to_all_dispatcher(self):
        """The dispatcher hands the runner topk tensors of its own."""
        self.assertFalse(self._ready(a2a=MoeA2ABackend.DEEPEP))

    def test_off_while_compiling(self):
        self.assertFalse(self._ready(compiling=True))

    def test_off_without_the_env_switch(self):
        with glm_aiter_presort.envs.SGLANG_OPT_HIP_GLM_ROUTE_SORT.override(False):
            self.assertFalse(
                glm_aiter_presort.kernel_ready(torch.zeros((4, 256)), torch.zeros(256))
            )


if __name__ == "__main__":
    unittest.main()
