import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.token_dispatcher.moonep import (
    MoonEPBuffer,
    MoonEPCombineInput,
    MoonEPDispatcher,
    MoonEPDispatchOutput,
    MoonEPExpertWeightLayout,
    get_moonep_expert_weight_layout,
    run_moonep_bf16_expert,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeMoonEPBuffer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.destroy_calls = 0
        self.__class__.instances.append(self)

    def destroy(self):
        self.destroy_calls += 1


class _FakeEvent:
    def __init__(self, name, calls, fail=False):
        self.name = name
        self.calls = calls
        self.fail = fail

    def wait(self, stream):
        self.calls.append((f"{self.name}.wait", stream))
        if self.fail:
            raise RuntimeError(f"{self.name} wait failed")


class _FakeAsyncMoonEPBuffer(_FakeMoonEPBuffer):
    fail_wait_for = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def _event(self, name):
        return _FakeEvent(
            name,
            self.calls,
            fail=self.fail_wait_for == name,
        )

    def dispatch(
        self,
        hidden_states,
        topk_weights,
        _topk_ids,
        _tokens_per_expert,
        *,
        async_finish,
        zero_copy,
    ):
        self.calls.append(("dispatch", async_finish, zero_copy))
        plan = SimpleNamespace(experts_to_copy=torch.tensor([0, 1], dtype=torch.int32))
        result = (
            hidden_states,
            topk_weights.reshape(-1),
            torch.tensor([1, 2, 2, 2], dtype=torch.int32),
            plan,
        )
        if async_finish:
            return (*result, self._event("dispatch"))
        return result

    def prefetch_weight(self, *, plan, async_finish, **_weights):
        self.calls.append(("prefetch", plan, async_finish))
        if async_finish:
            return self._event("prefetch")
        return None

    def combine(
        self,
        *,
        plan,
        hidden_nvsh,
        route_weights_nvs,
        async_finish,
        zero_copy,
    ):
        self.calls.append(
            (
                "combine",
                plan,
                route_weights_nvs,
                async_finish,
                zero_copy,
            )
        )
        event = self._event("combine") if async_finish else None
        return hidden_nvsh, None, event


def _fake_moonep_module(buffer_cls=_FakeMoonEPBuffer):
    module = types.ModuleType("moonep")
    module.Buffer = buffer_cls
    return module


class TestMoonEPBuffer(unittest.TestCase):
    def setUp(self):
        reset_context()
        _FakeMoonEPBuffer.instances.clear()

    def tearDown(self):
        try:
            MoonEPBuffer.destroy_all_buffers()
        finally:
            reset_context()
            _FakeMoonEPBuffer.instances.clear()

    def test_lazily_constructs_and_reuses_buffer_for_static_key(self):
        group = object()

        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=4,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )
            same_buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )
            larger_buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=512,
                num_prefetch_slots=16,
                token_padding=64,
                num_sms=20,
            )

        self.assertIs(buffer, same_buffer)
        self.assertIsNot(buffer, larger_buffer)
        self.assertEqual(len(_FakeMoonEPBuffer.instances), 2)
        self.assertEqual(
            buffer.kwargs,
            {
                "S": 256,
                "H": 1024,
                "K": 8,
                "E": 64,
                "num_ep_ranks": 4,
                "num_sms": 20,
                "token_padding": 64,
                "B": 16,
                "group": group,
            },
        )
        self.assertIs(MoonEPBuffer.get_existing_buffer(), larger_buffer)

    def test_resolves_env_defaults_and_training_safe_prefetch_slots(self):
        group = object()

        with (
            envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(384),
            envs.SGLANG_MOONEP_NUM_PREFETCH_SLOTS.override(-1),
            envs.SGLANG_MOONEP_TOKEN_PADDING.override(32),
            envs.SGLANG_MOONEP_NUM_SMS.override(18),
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=8,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=2048,
                router_topk=6,
                num_experts=128,
            )

        self.assertEqual(buffer.kwargs["S"], 384)
        self.assertEqual(buffer.kwargs["token_padding"], 32)
        self.assertEqual(buffer.kwargs["num_sms"], 18)
        self.assertEqual(buffer.kwargs["B"], 16)

    def test_rejects_non_divisible_experts_before_allocating(self):
        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=6,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "divisible"):
                MoonEPBuffer.get_moonep_buffer(
                    group=object(),
                    hidden_size=1024,
                    router_topk=8,
                    num_experts=64,
                )

        self.assertEqual(_FakeMoonEPBuffer.instances, [])

    def test_destroy_all_buffers_releases_cached_buffers(self):
        group = object()

        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=4,
            ),
        ):
            buffer = MoonEPBuffer.get_moonep_buffer(
                group=group,
                hidden_size=1024,
                router_topk=8,
                num_experts=64,
                num_max_dispatch_tokens_per_rank=256,
            )

        MoonEPBuffer.destroy_all_buffers()

        self.assertEqual(buffer.destroy_calls, 1)
        self.assertIsNone(MoonEPBuffer.get_existing_buffer())


class TestMoonEPExpertWeightLayout(unittest.TestCase):
    def _fake_layer(self):
        num_experts, hidden_size, intermediate_size = 3, 4, 5
        w13_weight = torch.arange(
            num_experts * 2 * intermediate_size * hidden_size,
            dtype=torch.bfloat16,
        ).reshape(num_experts, 2 * intermediate_size, hidden_size)
        w2_weight = torch.arange(
            num_experts * hidden_size * intermediate_size,
            dtype=torch.bfloat16,
        ).reshape(num_experts, hidden_size, intermediate_size)

        return SimpleNamespace(
            quant_config=None,
            moe_runner_config=SimpleNamespace(
                num_fused_shared_experts=0,
                is_gated=True,
            ),
            use_triton_kernels=False,
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            num_experts=num_experts,
            intermediate_size_per_partition=intermediate_size,
            hidden_size=hidden_size,
        )

    def test_layout_splits_gate_up_down_and_adds_prefetch_slots(self):
        layer = self._fake_layer()

        layout = get_moonep_expert_weight_layout(layer, num_prefetch_slots=2)

        self.assertEqual(tuple(layout.full_gate_weight.shape), (5, 5, 4))
        self.assertEqual(tuple(layout.full_up_weight.shape), (5, 5, 4))
        self.assertEqual(tuple(layout.full_down_weight.shape), (5, 4, 5))
        torch.testing.assert_close(
            layout.full_gate_weight[:3],
            layer.w13_weight[:, :5, :],
        )
        torch.testing.assert_close(
            layout.full_up_weight[:3],
            layer.w13_weight[:, 5:10, :],
        )
        torch.testing.assert_close(layout.full_down_weight[:3], layer.w2_weight)
        self.assertTrue(torch.all(layout.full_gate_weight[3:] == 0))
        self.assertTrue(torch.all(layout.full_up_weight[3:] == 0))
        self.assertTrue(torch.all(layout.full_down_weight[3:] == 0))

    def test_layout_is_cached_for_same_weight_storage(self):
        layer = self._fake_layer()

        first = get_moonep_expert_weight_layout(layer, num_prefetch_slots=2)
        second = get_moonep_expert_weight_layout(layer, num_prefetch_slots=2)

        self.assertIs(first, second)

    def test_layout_rejects_local_expert_storage(self):
        layer = self._fake_layer()
        layer.w13_weight = layer.w13_weight[:2].contiguous()

        with self.assertRaisesRegex(ValueError, "global w13_weight"):
            get_moonep_expert_weight_layout(layer, num_prefetch_slots=2)


class TestMoonEPBf16ExpertRunner(unittest.TestCase):
    def test_segment_runner_applies_expert_weights_and_route_weights(self):
        hidden_states = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=torch.float32,
        )
        route_weights = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
        cu_seqlens = torch.tensor([2, 3], dtype=torch.int32)
        expert_ids = torch.tensor([0, 1], dtype=torch.int32)
        gate = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 0.5]],
            ]
        )
        up = torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[1.5, 0.0], [0.0, 1.5]],
            ]
        )
        down = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[2.0, 0.0], [0.0, 2.0]],
            ]
        )
        layout = MoonEPExpertWeightLayout(
            full_gate_weight=gate,
            full_up_weight=up,
            full_down_weight=down,
            num_prefetch_slots=0,
        )
        dispatch_output = MoonEPDispatchOutput(
            hidden_states=hidden_states,
            route_weights_nvs=route_weights,
            cu_seqlens=cu_seqlens,
            plan=object(),
            expert_ids=expert_ids,
            num_tokens=2,
        )

        combine_input = run_moonep_bf16_expert(dispatch_output, layout)

        expected = torch.empty_like(hidden_states)
        for start, end, expert in [(0, 2, 0), (2, 3, 1)]:
            x = hidden_states[start:end]
            y = torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(x, gate[expert]))
                * torch.nn.functional.linear(x, up[expert]),
                down[expert],
            )
            expected[start:end] = y * route_weights[start:end, None]

        torch.testing.assert_close(combine_input.hidden_states, expected)
        self.assertIs(combine_input.plan, dispatch_output.plan)
        self.assertEqual(combine_input.num_tokens, 2)


class TestMoonEPDispatcherAsync(unittest.TestCase):
    def setUp(self):
        reset_context()
        _FakeAsyncMoonEPBuffer.instances.clear()
        _FakeAsyncMoonEPBuffer.fail_wait_for = None
        self.group = object()
        self.stream = object()
        self.hidden_states = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.bfloat16,
        )
        self.topk_output = StandardTopKOutput(
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            topk_ids=torch.tensor([[0], [1]], dtype=torch.int32),
            router_logits=None,
        )
        self.weight_layout = MoonEPExpertWeightLayout(
            full_gate_weight=torch.empty((4, 1, 1), dtype=torch.bfloat16),
            full_up_weight=torch.empty((4, 1, 1), dtype=torch.bfloat16),
            full_down_weight=torch.empty((4, 1, 1), dtype=torch.bfloat16),
            num_prefetch_slots=2,
        )
        self._patches = [
            envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(2),
            envs.SGLANG_MOONEP_NUM_PREFETCH_SLOTS.override(2),
            patch.dict(
                sys.modules,
                {"moonep": _fake_moonep_module(_FakeAsyncMoonEPBuffer)},
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.torch.cuda.current_stream",
                return_value=self.stream,
            ),
        ]
        for context in self._patches:
            context.__enter__()

    def tearDown(self):
        for context in reversed(self._patches):
            context.__exit__(None, None, None)
        reset_context()
        _FakeAsyncMoonEPBuffer.instances.clear()
        _FakeAsyncMoonEPBuffer.fail_wait_for = None

    def _dispatcher(self, *, async_finish=True):
        return MoonEPDispatcher(
            group=self.group,
            router_topk=1,
            num_experts=2,
            hidden_size=2,
            async_finish=async_finish,
        )

    @staticmethod
    def _combine_input(dispatch_output):
        return MoonEPCombineInput(
            hidden_states=dispatch_output.hidden_states,
            route_weights_nvs=dispatch_output.route_weights_nvs,
            plan=dispatch_output.plan,
            num_tokens=dispatch_output.num_tokens,
        )

    def test_async_events_wait_at_each_public_consumer_boundary(self):
        dispatcher = self._dispatcher()
        calls = []
        derive_expert_ids = dispatcher._expert_ids_from_plan

        def derive_after_dispatch_wait(cu_seqlens, plan):
            calls.extend(_FakeAsyncMoonEPBuffer.instances[0].calls)
            calls.append(("derive_expert_ids", self.stream))
            return derive_expert_ids(cu_seqlens, plan)

        with patch.object(
            dispatcher,
            "_expert_ids_from_plan",
            side_effect=derive_after_dispatch_wait,
        ):
            dispatch_output = dispatcher.dispatch(
                self.hidden_states,
                self.topk_output,
            )

        buffer = _FakeAsyncMoonEPBuffer.instances[0]
        self.assertEqual(
            calls,
            [
                ("dispatch", True, False),
                ("dispatch.wait", self.stream),
                ("derive_expert_ids", self.stream),
            ],
        )

        dispatcher.prefetch_weight(dispatch_output.plan, self.weight_layout)
        self.assertEqual(buffer.calls[-1], ("prefetch.wait", self.stream))

        output = dispatcher.combine(self._combine_input(dispatch_output))
        self.assertEqual(buffer.calls[-1], ("combine.wait", self.stream))
        torch.testing.assert_close(output, self.hidden_states)

    def test_synchronous_mode_preserves_calls_without_event_waits(self):
        dispatcher = self._dispatcher(async_finish=False)

        dispatch_output = dispatcher.dispatch(self.hidden_states, self.topk_output)
        dispatcher.prefetch_weight(dispatch_output.plan, self.weight_layout)
        dispatcher.combine(self._combine_input(dispatch_output))

        calls = _FakeAsyncMoonEPBuffer.instances[0].calls
        self.assertEqual(calls[0], ("dispatch", False, False))
        self.assertEqual(calls[1][0], "prefetch")
        self.assertFalse(calls[1][2])
        self.assertEqual(calls[2][0], "combine")
        self.assertFalse(calls[2][3])
        self.assertFalse(any(call[0].endswith(".wait") for call in calls))

    def test_valid_sequence_returns_to_idle(self):
        dispatcher = self._dispatcher()

        first = dispatcher.dispatch(self.hidden_states, self.topk_output)
        dispatcher.prefetch_weight(first.plan, self.weight_layout)
        dispatcher.combine(self._combine_input(first))
        second = dispatcher.dispatch(self.hidden_states, self.topk_output)

        self.assertIsNot(first.plan, second.plan)

    def test_shared_buffer_rejects_a_second_in_flight_dispatch(self):
        first_dispatcher = self._dispatcher()
        second_dispatcher = self._dispatcher()
        first_dispatcher.dispatch(self.hidden_states, self.topk_output)

        with self.assertRaisesRegex(RuntimeError, "requires IDLE"):
            second_dispatcher.dispatch(self.hidden_states, self.topk_output)

    def test_prequeue_errors_preserve_the_active_flight(self):
        dispatcher = self._dispatcher()
        dispatch_output = dispatcher.dispatch(self.hidden_states, self.topk_output)

        with self.assertRaisesRegex(RuntimeError, "requires PREFETCHED"):
            dispatcher.combine(self._combine_input(dispatch_output))
        with self.assertRaisesRegex(RuntimeError, "same MoonEP plan"):
            dispatcher.prefetch_weight(object(), self.weight_layout)

        dispatcher.prefetch_weight(dispatch_output.plan, self.weight_layout)
        dispatcher.combine(self._combine_input(dispatch_output))

    def test_queued_failure_poisons_shared_buffer(self):
        dispatcher = self._dispatcher()
        _FakeAsyncMoonEPBuffer.fail_wait_for = "dispatch"

        with self.assertRaisesRegex(RuntimeError, "dispatch wait failed"):
            dispatcher.dispatch(self.hidden_states, self.topk_output)

        _FakeAsyncMoonEPBuffer.fail_wait_for = None
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            self._dispatcher().dispatch(self.hidden_states, self.topk_output)


if __name__ == "__main__":
    unittest.main()
