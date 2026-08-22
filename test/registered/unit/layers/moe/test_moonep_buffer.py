import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerOutput,
    _resolve_down_output,
    post_permute_deep_gemm_to_moonep,
    pre_permute_moonep_to_deep_gemm,
)
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
        self.dispatch_calls = []
        self.combine_calls = []
        self.plan = SimpleNamespace(
            experts_to_copy=torch.zeros(
                kwargs["num_ep_ranks"], kwargs["B"], dtype=torch.int32
            )
        )
        self.hidden_nvsh_buffer_view = torch.zeros(
            kwargs["S"], kwargs["H"], dtype=torch.bfloat16
        )
        self.__class__.instances.append(self)

    def destroy(self):
        self.destroy_calls += 1

    def dispatch(
        self,
        hidden_states,
        route_weights,
        topk_ids,
        tokens_per_expert,
        **kwargs,
    ):
        self.dispatch_calls.append(kwargs)
        self.hidden_nvsh_buffer_view.copy_(hidden_states)
        hidden_nvsh = (
            self.hidden_nvsh_buffer_view
            if kwargs["zero_copy"]
            else self.hidden_nvsh_buffer_view.clone()
        )
        route_weights_nvs = route_weights.reshape(-1).clone()
        cu_seqlens = torch.arange(
            1,
            self.kwargs["E"] + self.kwargs["B"] + 1,
            dtype=torch.int32,
        ).clamp_max(hidden_nvsh.shape[0])
        return hidden_nvsh, route_weights_nvs, cu_seqlens, self.plan

    def combine(self, **kwargs):
        self.combine_calls.append(kwargs)
        return kwargs["hidden_nvsh"].clone(), None, None


def _fake_moonep_module():
    module = types.ModuleType("moonep")
    module.Buffer = _FakeMoonEPBuffer
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
                torch.nn.functional.silu(
                    torch.nn.functional.linear(x, gate[expert])
                )
                * torch.nn.functional.linear(x, up[expert]),
                down[expert],
            )
            expected[start:end] = y * route_weights[start:end, None]

        torch.testing.assert_close(combine_input.hidden_states, expected)
        self.assertIs(combine_input.plan, dispatch_output.plan)
        self.assertEqual(combine_input.num_tokens, 2)

    def test_zero_copy_segment_runner_reuses_dispatch_view_and_zeros_padding(self):
        hidden_states = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
            dtype=torch.float32,
        )
        original = hidden_states.clone()
        layout = MoonEPExpertWeightLayout(
            full_gate_weight=torch.eye(2).unsqueeze(0),
            full_up_weight=(2 * torch.eye(2)).unsqueeze(0),
            full_down_weight=torch.eye(2).unsqueeze(0),
            num_prefetch_slots=0,
        )
        dispatch_output = MoonEPDispatchOutput(
            hidden_states=hidden_states,
            route_weights_nvs=torch.tensor([1.0, 0.5, 0.0]),
            cu_seqlens=torch.tensor([2], dtype=torch.int32),
            plan=object(),
            expert_ids=torch.tensor([0], dtype=torch.int32),
            num_tokens=2,
            hidden_states_zero_copy=True,
            buffer_key=object(),
            owner_id=7,
        )

        with (
            patch.object(MoonEPBuffer, "begin_zero_copy_output_write") as begin_write,
            patch.object(
                MoonEPBuffer, "mark_zero_copy_output_written"
            ) as mark_output_written,
        ):
            combine_input = run_moonep_bf16_expert(
                dispatch_output, layout, zero_copy=True
            )

        expected = torch.nn.functional.silu(original[:2]) * (2 * original[:2])
        expected[1].mul_(0.5)
        self.assertEqual(
            combine_input.hidden_states.data_ptr(), dispatch_output.hidden_states.data_ptr()
        )
        torch.testing.assert_close(combine_input.hidden_states[:2], expected)
        self.assertTrue(torch.count_nonzero(combine_input.hidden_states[2:]) == 0)
        self.assertTrue(combine_input.hidden_states_zero_copy)
        begin_write.assert_called_once()
        mark_output_written.assert_called_once()

    def test_bf16_runner_rejects_zero_copy_mode_mismatch(self):
        hidden_states = torch.ones(1, 2)
        dispatch_output = MoonEPDispatchOutput(
            hidden_states=hidden_states,
            route_weights_nvs=None,
            cu_seqlens=torch.tensor([1], dtype=torch.int32),
            plan=object(),
            expert_ids=torch.tensor([0], dtype=torch.int32),
            num_tokens=1,
            hidden_states_zero_copy=True,
            buffer_key=object(),
            owner_id=7,
        )
        layout = MoonEPExpertWeightLayout(
            full_gate_weight=torch.eye(2).unsqueeze(0),
            full_up_weight=torch.eye(2).unsqueeze(0),
            full_down_weight=torch.eye(2).unsqueeze(0),
            num_prefetch_slots=0,
        )

        with self.assertRaisesRegex(RuntimeError, "mode must match"):
            run_moonep_bf16_expert(dispatch_output, layout, zero_copy=False)


class TestMoonEPZeroCopyDispatcher(unittest.TestCase):
    def setUp(self):
        reset_context()
        _FakeMoonEPBuffer.instances.clear()

    def tearDown(self):
        try:
            MoonEPBuffer.destroy_all_buffers()
        finally:
            reset_context()
            _FakeMoonEPBuffer.instances.clear()

    def _dispatch(self, *, zero_copy: bool, zero_copy_supported: bool = True):
        group = SimpleNamespace(size=lambda: 1)
        with envs.SGLANG_MOONEP_DECODE_MAX_DISPATCH_TOKENS_PER_RANK.override(2):
            dispatcher = MoonEPDispatcher(
                group=group,
                router_topk=1,
                num_experts=1,
                num_local_experts=1,
                hidden_size=2,
                zero_copy_supported=zero_copy_supported,
            )
        topk_output = StandardTopKOutput(
            topk_weights=torch.ones(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int64),
            router_logits=torch.empty(2, 1),
        )
        contexts = (
            envs.SGLANG_MOONEP_ZERO_COPY.override(zero_copy),
            envs.SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.override(2),
            envs.SGLANG_MOONEP_NUM_PREFETCH_SLOTS.override(1),
            envs.SGLANG_MOONEP_TOKEN_PADDING.override(1),
            patch.dict(sys.modules, {"moonep": _fake_moonep_module()}),
        )
        return dispatcher, topk_output, contexts

    def test_feature_gate_preserves_copied_dispatch_and_combine(self):
        dispatcher, topk_output, contexts = self._dispatch(zero_copy=False)
        with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4]:
            output = dispatcher.dispatch(
                torch.ones(2, 2, dtype=torch.bfloat16), topk_output
            )
            combine_input = MoonEPCombineInput(
                hidden_states=output.hidden_states,
                route_weights_nvs=output.route_weights_nvs,
                plan=output.plan,
                num_tokens=output.num_tokens,
            )
            dispatcher.combine(combine_input)

        buffer = _FakeMoonEPBuffer.instances[0]
        self.assertFalse(output.hidden_states_zero_copy)
        self.assertEqual(
            buffer.dispatch_calls[-1],
            {
                "async_finish": False,
                "zero_copy": False,
                "router_weights_zero_copy": False,
            },
        )
        self.assertFalse(buffer.combine_calls[-1]["zero_copy"])
        self.assertFalse(buffer.combine_calls[-1]["router_weights_zero_copy"])

    def test_zero_copy_rejects_unsupported_runner_before_dispatch(self):
        dispatcher, topk_output, contexts = self._dispatch(
            zero_copy=True, zero_copy_supported=False
        )
        with (
            contexts[0],
            contexts[1],
            contexts[2],
            contexts[3],
            contexts[4],
            torch.no_grad(),
        ):
            with self.assertRaisesRegex(RuntimeError, "explicitly selected DeepGEMM"):
                dispatcher.dispatch(
                    torch.ones(2, 2, dtype=torch.bfloat16), topk_output
                )
        self.assertEqual(_FakeMoonEPBuffer.instances, [])

    def test_combine_rejects_flight_before_output_is_written(self):
        dispatcher, topk_output, contexts = self._dispatch(zero_copy=True)
        with (
            contexts[0],
            contexts[1],
            contexts[2],
            contexts[3],
            contexts[4],
            torch.no_grad(),
        ):
            output = dispatcher.dispatch(
                torch.ones(2, 2, dtype=torch.bfloat16), topk_output
            )
            combine_input = MoonEPCombineInput(
                hidden_states=output.hidden_states,
                route_weights_nvs=output.route_weights_nvs,
                plan=output.plan,
                num_tokens=output.num_tokens,
                hidden_states_zero_copy=True,
                buffer_key=output.buffer_key,
                owner_id=output.owner_id,
            )
            with self.assertRaisesRegex(RuntimeError, "state mismatch"):
                dispatcher.combine(combine_input)

    def test_zero_copy_flight_preserves_pointer_and_rejects_overlap(self):
        dispatcher, topk_output, contexts = self._dispatch(zero_copy=True)
        with (
            contexts[0],
            contexts[1],
            contexts[2],
            contexts[3],
            contexts[4],
            torch.no_grad(),
        ):
            output = dispatcher.dispatch(
                torch.ones(2, 2, dtype=torch.bfloat16), topk_output
            )
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                dispatcher.dispatch(
                    torch.ones(2, 2, dtype=torch.bfloat16), topk_output
                )

            combine_input = MoonEPCombineInput(
                hidden_states=output.hidden_states,
                route_weights_nvs=output.route_weights_nvs,
                plan=output.plan,
                num_tokens=output.num_tokens,
                hidden_states_zero_copy=True,
                buffer_key=output.buffer_key,
                owner_id=output.owner_id,
            )
            MoonEPBuffer.begin_zero_copy_output_write(
                key=output.buffer_key,
                plan=output.plan,
                hidden_states=output.hidden_states,
                owner_id=output.owner_id,
            )
            MoonEPBuffer.mark_zero_copy_output_written(
                key=output.buffer_key,
                plan=output.plan,
                hidden_states=output.hidden_states,
                owner_id=output.owner_id,
            )
            dispatcher.combine(combine_input)
            dispatcher.dispatch(torch.ones(2, 2, dtype=torch.bfloat16), topk_output)

        buffer = _FakeMoonEPBuffer.instances[0]
        self.assertEqual(
            output.hidden_states.data_ptr(), buffer.hidden_nvsh_buffer_view.data_ptr()
        )
        self.assertTrue(output.hidden_states_zero_copy)
        self.assertTrue(buffer.dispatch_calls[0]["zero_copy"])
        self.assertFalse(buffer.dispatch_calls[0]["router_weights_zero_copy"])
        self.assertTrue(buffer.combine_calls[0]["zero_copy"])
        self.assertFalse(buffer.combine_calls[0]["router_weights_zero_copy"])

    def test_pointer_replacement_poisons_zero_copy_flight(self):
        dispatcher, topk_output, contexts = self._dispatch(zero_copy=True)
        with (
            contexts[0],
            contexts[1],
            contexts[2],
            contexts[3],
            contexts[4],
            torch.no_grad(),
        ):
            output = dispatcher.dispatch(
                torch.ones(2, 2, dtype=torch.bfloat16), topk_output
            )
            combine_input = MoonEPCombineInput(
                hidden_states=output.hidden_states.clone(),
                route_weights_nvs=output.route_weights_nvs,
                plan=output.plan,
                num_tokens=output.num_tokens,
                hidden_states_zero_copy=True,
                buffer_key=output.buffer_key,
                owner_id=output.owner_id,
            )
            MoonEPBuffer.begin_zero_copy_output_write(
                key=output.buffer_key,
                plan=output.plan,
                hidden_states=output.hidden_states,
                owner_id=output.owner_id,
            )
            MoonEPBuffer.mark_zero_copy_output_written(
                key=output.buffer_key,
                plan=output.plan,
                hidden_states=output.hidden_states,
                owner_id=output.owner_id,
            )
            with self.assertRaisesRegex(RuntimeError, "pointer"):
                dispatcher.combine(combine_input)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                dispatcher.dispatch(
                    torch.ones(2, 2, dtype=torch.bfloat16), topk_output
                )

    def test_plan_mismatch_poisons_zero_copy_flight(self):
        dispatcher, topk_output, contexts = self._dispatch(zero_copy=True)
        with (
            contexts[0],
            contexts[1],
            contexts[2],
            contexts[3],
            contexts[4],
            torch.no_grad(),
        ):
            output = dispatcher.dispatch(
                torch.ones(2, 2, dtype=torch.bfloat16), topk_output
            )
            MoonEPBuffer.begin_zero_copy_output_write(
                key=output.buffer_key,
                plan=output.plan,
                hidden_states=output.hidden_states,
                owner_id=output.owner_id,
            )
            MoonEPBuffer.mark_zero_copy_output_written(
                key=output.buffer_key,
                plan=output.plan,
                hidden_states=output.hidden_states,
                owner_id=output.owner_id,
            )
            combine_input = MoonEPCombineInput(
                hidden_states=output.hidden_states,
                route_weights_nvs=output.route_weights_nvs,
                plan=object(),
                num_tokens=output.num_tokens,
                hidden_states_zero_copy=True,
                buffer_key=output.buffer_key,
                owner_id=output.owner_id,
            )
            with self.assertRaisesRegex(RuntimeError, "plan mismatch"):
                dispatcher.combine(combine_input)
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                dispatcher.dispatch(
                    torch.ones(2, 2, dtype=torch.bfloat16), topk_output
                )


class TestMoonEPDeepGemmZeroCopy(unittest.TestCase):
    def _dispatch_output(self, hidden_states):
        return MoonEPDispatchOutput(
            hidden_states=hidden_states,
            route_weights_nvs=torch.tensor([1.0, 0.5, 0.0]),
            cu_seqlens=torch.tensor([2], dtype=torch.int32),
            plan=object(),
            expert_ids=torch.tensor([0], dtype=torch.int32),
            num_tokens=2,
            hidden_states_zero_copy=True,
            buffer_key=object(),
            owner_id=7,
        )

    def test_adapters_reserve_and_preserve_moonep_output_buffer(self):
        hidden_states = torch.ones(3, 2, dtype=torch.bfloat16)
        dispatch_output = self._dispatch_output(hidden_states)
        quant_info = SimpleNamespace(
            w13_weight=torch.empty(1, 4, 2, dtype=torch.bfloat16)
        )
        running_state = {}

        with (
            patch.object(MoonEPBuffer, "begin_zero_copy_output_write") as begin_write,
            patch.object(
                MoonEPBuffer, "mark_zero_copy_output_written"
            ) as mark_output_written,
        ):
            runner_input = pre_permute_moonep_to_deep_gemm(
                dispatch_output,
                quant_info,
                SimpleNamespace(),
                running_state,
            )
            combine_input = post_permute_deep_gemm_to_moonep(
                DeepGemmRunnerOutput(hidden_states=hidden_states),
                quant_info,
                SimpleNamespace(),
                running_state,
            )

        self.assertIs(runner_input.output_buffer, hidden_states)
        self.assertEqual(combine_input.hidden_states.data_ptr(), hidden_states.data_ptr())
        self.assertTrue(combine_input.hidden_states_zero_copy)
        begin_write.assert_called_once()
        mark_output_written.assert_called_once()

    def test_post_adapter_rejects_replaced_output_buffer(self):
        hidden_states = torch.ones(3, 2, dtype=torch.bfloat16)
        running_state = {}
        dispatch_output = self._dispatch_output(hidden_states)
        quant_info = SimpleNamespace(
            w13_weight=torch.empty(1, 4, 2, dtype=torch.bfloat16)
        )
        with patch.object(MoonEPBuffer, "begin_zero_copy_output_write"):
            pre_permute_moonep_to_deep_gemm(
                dispatch_output,
                quant_info,
                SimpleNamespace(),
                running_state,
            )

        with self.assertRaisesRegex(RuntimeError, "pointer"):
            post_permute_deep_gemm_to_moonep(
                DeepGemmRunnerOutput(hidden_states=hidden_states.clone()),
                quant_info,
                SimpleNamespace(),
                running_state,
            )

    def test_bf16_runner_writes_down_projection_into_provided_buffer(self):
        hidden_states = torch.ones(2, 2, dtype=torch.bfloat16)
        runner_config = MoeRunnerConfig(
            num_experts=1,
            num_local_experts=1,
            hidden_size=2,
            intermediate_size_per_partition=2,
            top_k=1,
        )
        quant_info = DeepGemmMoeQuantInfo(
            w13_weight=torch.ones(1, 4, 2, dtype=torch.bfloat16),
            w2_weight=torch.ones(1, 2, 2, dtype=torch.bfloat16),
            use_fp8=False,
        )
        running_state = {
            "all_tokens": 2,
            "hidden_states_device": hidden_states.device,
            "hidden_states_shape": hidden_states.shape,
        }
        dispatch_output = self._dispatch_output(hidden_states)
        with patch.object(MoonEPBuffer, "begin_zero_copy_output_write"):
            runner_input = pre_permute_moonep_to_deep_gemm(
                dispatch_output, quant_info, runner_config, running_state
            )

        def fake_gemm(lhs, rhs, output, m_indices):
            output.fill_(2)

        def fake_activation(gateup, down_input):
            down_input.fill_(3)

        with (
            patch(
                "sglang.srt.layers.moe.moe_runner.deep_gemm.deep_gemm_wrapper."
                "grouped_gemm_nt_bf16_contig",
                side_effect=fake_gemm,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.deep_gemm._legacy_silu_and_mul",
                side_effect=fake_activation,
            ),
            patch(
                "sglang.srt.layers.moe.moe_runner.deep_gemm.dispose_tensor"
            ) as dispose,
        ):
            output = DeepGemmRunnerCore(runner_config).run(
                runner_input, quant_info, running_state
            )

        self.assertEqual(output.hidden_states.data_ptr(), hidden_states.data_ptr())
        self.assertTrue(
            all(call.args[0] is not hidden_states for call in dispose.call_args_list)
        )

    def test_output_buffer_validation_rejects_wrong_tensor_contract(self):
        expected_shape = (2, 2)
        good = torch.empty(expected_shape, dtype=torch.bfloat16)
        self.assertIs(
            _resolve_down_output(expected_shape, good.dtype, good.device, good), good
        )

        invalid_buffers = (
            torch.empty(3, 2, dtype=torch.bfloat16),
            torch.empty(expected_shape, dtype=torch.float32),
            torch.empty(2, 4, dtype=torch.bfloat16)[:, ::2],
        )
        for invalid in invalid_buffers:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaisesRegex(
                    ValueError, "shape|dtype|contiguous"
                ):
                    _resolve_down_output(
                        expected_shape, torch.bfloat16, torch.device("cpu"), invalid
                    )


if __name__ == "__main__":
    unittest.main()
