import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe import DeepEPMode, MoeA2ABackend, MoeRunnerConfig
from sglang.srt.layers.moe.fused_moe_triton.layer import create_moe_dispatcher
from sglang.srt.layers.moe.token_dispatcher.moonep import (
    MoonEPBuffer,
    MoonEPDispatcher,
    MoonEPDispatchOutput,
    MoonEPExpertWeightLayout,
    get_moonep_expert_weight_layout,
    run_moonep_bf16_expert,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_flags, get_forward, reset_context
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

    def dispatch(
        self,
        hidden_states,
        topk_weights,
        topk_ids,
        tokens_per_expert,
        async_finish,
    ):
        return (
            hidden_states,
            topk_weights.reshape(-1),
            torch.tensor([hidden_states.shape[0]], dtype=torch.int32),
            SimpleNamespace(experts_to_copy=torch.empty(0, dtype=torch.int32)),
        )


def _fake_moonep_module():
    module = types.ModuleType("moonep")
    module.Buffer = _FakeMoonEPBuffer
    return module


class _MoonEPBufferTestCase(unittest.TestCase):
    def setUp(self):
        reset_context()
        _FakeMoonEPBuffer.instances.clear()

    def tearDown(self):
        try:
            MoonEPBuffer.destroy_all_buffers()
        finally:
            reset_context()
            _FakeMoonEPBuffer.instances.clear()


class TestMoonEPBuffer(_MoonEPBufferTestCase):
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


class TestMoonEPCapacityLogging(_MoonEPBufferTestCase):
    LOGGER_NAME = "sglang.srt.layers.moe.token_dispatcher.moonep"
    LOG_PREFIX = f"DEBUG:{LOGGER_NAME}:"

    @staticmethod
    def _make_dispatch_inputs(num_tokens):
        hidden_states = torch.ones((num_tokens, 3), dtype=torch.bfloat16)
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones((num_tokens, 2), dtype=torch.float32),
            topk_ids=torch.arange(num_tokens * 2, dtype=torch.int64).reshape(
                num_tokens, 2
            )
            % 4,
            router_logits=torch.empty((num_tokens, 4), dtype=torch.float32),
        )
        return hidden_states, topk_output

    @staticmethod
    def _make_dispatcher(capacity, layer_id):
        with envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(capacity):
            return MoonEPDispatcher(
                group=object(),
                router_topk=2,
                num_experts=4,
                hidden_size=3,
                layer_id=layer_id,
            )

    def _dispatch_and_capture(
        self,
        dispatcher,
        hidden_states,
        topk_output,
        *,
        rank,
        is_extend_in_batch,
    ):
        with (
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_world_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.moe.token_dispatcher.moonep.dist.get_rank",
                return_value=rank,
            ),
            get_forward().scoped(is_extend_in_batch=is_extend_in_batch),
            self.assertLogs(self.LOGGER_NAME, level="DEBUG") as captured,
        ):
            dispatcher.dispatch(
                hidden_states=hidden_states,
                topk_output=topk_output,
            )
        return captured.output

    def test_dispatcher_requires_layer_id_for_attributable_logs(self):
        with self.assertRaisesRegex(ValueError, "requires layer_id"):
            MoonEPDispatcher(
                group=object(),
                router_topk=2,
                num_experts=4,
                hidden_size=3,
            )

    def test_partial_extend_dispatch_logs_host_capacity_utilization(self):
        dispatcher = self._make_dispatcher(capacity=4, layer_id=7)
        hidden_states, topk_output = self._make_dispatch_inputs(num_tokens=2)

        captured = self._dispatch_and_capture(
            dispatcher,
            hidden_states,
            topk_output,
            rank=3,
            is_extend_in_batch=True,
        )

        self.assertEqual(
            captured,
            [
                self.LOG_PREFIX
                + "phase=extend rank=3 layer_id=7 actual_tokens=2 capacity=4 "
                "padding_tokens=2 capacity_utilization=0.5 "
                "static_padding_ratio=0.5"
            ],
        )

    def test_factory_threads_layer_id_into_decode_capacity_log(self):
        group = object()
        config = MoeRunnerConfig(
            num_experts=4,
            num_local_experts=4,
            hidden_size=3,
            layer_id=11,
            top_k=2,
            params_dtype=torch.bfloat16,
        )
        hidden_states, topk_output = self._make_dispatch_inputs(num_tokens=2)

        with (
            envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(4),
            get_flags().moe.override(
                a2a_backend=MoeA2ABackend.MOONEP,
                deepep_mode=DeepEPMode.AUTO,
                tbo_enabled=False,
            ),
            patch(
                "sglang.srt.layers.moe.fused_moe_triton.layer._get_deepep_comm_group",
                return_value=group,
            ),
        ):
            dispatcher = create_moe_dispatcher(config)

        captured = self._dispatch_and_capture(
            dispatcher,
            hidden_states,
            topk_output,
            rank=0,
            is_extend_in_batch=False,
        )

        self.assertEqual(
            captured,
            [
                self.LOG_PREFIX
                + "phase=decode rank=0 layer_id=11 actual_tokens=2 capacity=4 "
                "padding_tokens=2 capacity_utilization=0.5 "
                "static_padding_ratio=0.5"
            ],
        )

    def test_full_capacity_dispatch_logs_zero_padding(self):
        dispatcher = self._make_dispatcher(capacity=2, layer_id=5)
        hidden_states, topk_output = self._make_dispatch_inputs(num_tokens=2)

        captured = self._dispatch_and_capture(
            dispatcher,
            hidden_states,
            topk_output,
            rank=1,
            is_extend_in_batch=False,
        )

        self.assertEqual(
            captured,
            [
                self.LOG_PREFIX
                + "phase=decode rank=1 layer_id=5 actual_tokens=2 capacity=2 "
                "padding_tokens=0 capacity_utilization=1.0 "
                "static_padding_ratio=0.0"
            ],
        )

    def test_over_capacity_dispatch_raises_without_capacity_log(self):
        dispatcher = self._make_dispatcher(capacity=4, layer_id=5)
        hidden_states, topk_output = self._make_dispatch_inputs(num_tokens=5)

        with (
            get_forward().scoped(is_extend_in_batch=True),
            self.assertNoLogs(self.LOGGER_NAME, level="DEBUG"),
            self.assertRaisesRegex(ValueError, "more tokens than its static buffer"),
        ):
            dispatcher.dispatch(hidden_states, topk_output)


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


if __name__ == "__main__":
    unittest.main()
