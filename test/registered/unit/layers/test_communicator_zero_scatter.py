import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.layers.communicator as communicator
import sglang.srt.layers.dp_attention as dp_attention
from sglang.srt.layers.communicator import (
    CommunicateSimpleFn,
    CommunicateWithAllReduceAndLayerNormFn,
    ScatterMode,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCommunicatorZeroScatter(CustomTestCase):
    def test_empty_dp_gather_uses_zero_fast_path(self):
        local_tokens = Mock()
        local_tokens.numel.return_value = 0
        local_tokens.shape = (0, 4)
        local_tokens.is_contiguous.return_value = True
        global_tokens = Mock()
        global_tokens.is_contiguous.return_value = True

        with (
            patch.object(dp_attention, "get_dp_local_info", return_value=(0, 0)),
            patch.object(dp_attention, "world_dp_gather_enabled", return_value=True),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
            patch.object(dp_attention, "memcpy_triton") as memcpy,
            patch.object(dp_attention, "zero_triton") as zero,
        ):
            dp_attention._dp_gather_via_all_reduce(
                global_tokens,
                local_tokens,
                Mock(),
                is_partial=False,
                force_standard_all_reduce=False,
            )

        zero.assert_called_once_with(global_tokens)
        global_tokens.fill_.assert_not_called()
        memcpy.assert_not_called()
        all_reduce.assert_called_once()

    def test_nonempty_dp_gather_keeps_fill_and_copy(self):
        local_tokens = Mock()
        local_tokens.numel.return_value = 4
        local_tokens.shape = (1, 4)
        local_tokens.is_contiguous.return_value = True
        local_tokens.untyped_storage.return_value = object()
        global_tokens = Mock()
        global_tokens.is_contiguous.return_value = True
        global_tokens.untyped_storage.return_value = object()

        with (
            patch.object(dp_attention, "get_dp_local_info", return_value=(0, 1)),
            patch.object(
                dp_attention,
                "get_attn_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch.object(dp_attention, "world_dp_gather_enabled", return_value=True),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
            patch.object(dp_attention, "memcpy_triton") as memcpy,
            patch.object(dp_attention, "zero_triton") as zero,
        ):
            dp_attention._dp_gather_via_all_reduce(
                global_tokens,
                local_tokens,
                Mock(),
                is_partial=False,
                force_standard_all_reduce=False,
            )

        global_tokens.fill_.assert_called_once_with(0)
        zero.assert_not_called()
        memcpy.assert_called_once_with(global_tokens, local_tokens, 0, 0, 1, False)
        all_reduce.assert_called_once()

    def test_idle_dp_gather_can_force_standard_all_reduce(self):
        local_tokens = Mock()
        local_tokens.numel.return_value = 0
        local_tokens.shape = (0, 4)
        local_tokens.is_contiguous.return_value = True
        global_tokens = Mock()
        global_tokens.is_contiguous.return_value = True
        device_group = object()
        tp_group = SimpleNamespace(device_group=device_group)

        with (
            patch.object(dp_attention, "get_dp_local_info", return_value=(0, 0)),
            patch.object(dp_attention, "world_dp_gather_enabled", return_value=False),
            patch.object(dp_attention, "get_tp_group", return_value=tp_group),
            patch.object(torch.distributed, "all_reduce") as all_reduce,
            patch.object(dp_attention, "zero_triton"),
        ):
            dp_attention._dp_gather_via_all_reduce(
                global_tokens,
                local_tokens,
                Mock(),
                is_partial=False,
                force_standard_all_reduce=True,
            )

        all_reduce.assert_called_once_with(global_tokens, group=device_group)

    def test_empty_dp_scatter_skips_device_work(self):
        local_tokens = Mock()
        local_tokens.numel.return_value = 0
        global_tokens = Mock()

        with (
            patch.object(dp_attention, "get_dp_local_info") as get_local_info,
            patch.object(dp_attention, "memcpy_triton") as memcpy,
        ):
            dp_attention.dp_scatter(local_tokens, global_tokens, Mock())

        get_local_info.assert_not_called()
        local_tokens.fill_.assert_not_called()
        memcpy.assert_not_called()

    def test_nonempty_dp_scatter_keeps_fill_and_copy(self):
        local_tokens = torch.ones((2, 4), dtype=torch.float32)
        global_tokens = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        with (
            patch.object(
                dp_attention, "get_dp_local_info", return_value=(2, 2)
            ) as get_local_info,
            patch.object(dp_attention, "memcpy_triton") as memcpy,
        ):
            dp_attention.dp_scatter(local_tokens, global_tokens, Mock())

        get_local_info.assert_called_once()
        memcpy.assert_called_once_with(local_tokens, global_tokens, 0, 2, 2, True)
        torch.testing.assert_close(local_tokens, torch.zeros_like(local_tokens))

    def test_empty_tensor_skips_all_gather(self):
        hidden_states = torch.empty((0, 4))
        context = SimpleNamespace(attn_tp_size=2)

        with (
            patch.object(communicator, "get_local_dp_buffer") as get_buffer,
            patch.object(communicator, "attn_tp_all_gather_into_tensor") as gather,
        ):
            output = CommunicateSimpleFn._scattered_to_tp_attn_full(
                hidden_states, None, context
            )

        get_buffer.assert_not_called()
        gather.assert_not_called()
        self.assertIs(output, hidden_states)

    def test_empty_tuple_skips_all_gather(self):
        hidden_states = (torch.empty((0, 4)), torch.empty((0, 2)))
        context = SimpleNamespace(attn_tp_size=2)

        with patch.object(communicator, "attn_tp_all_gather_into_tensor") as gather:
            output = CommunicateSimpleFn._scattered_to_tp_attn_full(
                hidden_states, None, context
            )

        gather.assert_not_called()
        self.assertIs(output[0], hidden_states[0])
        self.assertIs(output[1], hidden_states[1])

    def test_nonempty_tensor_keeps_all_gather(self):
        local_hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        gathered_hidden_states = torch.empty((4, 4))
        context = SimpleNamespace(attn_tp_size=2)
        parallel = SimpleNamespace(attn_tp_group=object())

        def all_gather(output, input_):
            output.copy_(input_.repeat(2, 1))

        with (
            patch.object(communicator, "get_parallel", return_value=parallel),
            patch.object(
                communicator,
                "get_local_dp_buffer",
                return_value=gathered_hidden_states,
            ) as get_buffer,
            patch.object(
                communicator,
                "attn_tp_all_gather_into_tensor",
                side_effect=all_gather,
            ) as gather,
        ):
            output = CommunicateSimpleFn._scattered_to_tp_attn_full(
                local_hidden_states, None, context
            )

        get_buffer.assert_called_once_with(parallel.attn_tp_group)
        gather.assert_called_once_with(gathered_hidden_states, local_hidden_states)
        torch.testing.assert_close(output, local_hidden_states.repeat(2, 1))

    def test_empty_input_skips_collective_and_scatters_residual(self):
        hidden_states = torch.empty((0, 4))
        residual = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        context = SimpleNamespace(attn_tp_size=2, attn_tp_rank=1)
        layernorm = Mock()

        with patch.object(communicator, "attn_tp_reduce_scatter_tensor") as scatter:
            output, output_residual = (
                CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual(
                    hidden_states,
                    residual,
                    None,
                    layernorm,
                    context,
                    residual_input_mode=ScatterMode.TP_ATTN_FULL,
                )
            )

        scatter.assert_not_called()
        layernorm.assert_not_called()
        self.assertEqual(tuple(output.shape), (0, 4))
        torch.testing.assert_close(output_residual, residual.tensor_split(2)[1])

    def test_nonempty_input_keeps_collective(self):
        hidden_states = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        residual = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        context = SimpleNamespace(attn_tp_size=2, attn_tp_rank=1)
        layernorm = Mock(side_effect=lambda hidden, res: (hidden + 1, res + 1))

        def reduce_scatter(output, input_):
            output.copy_(input_.tensor_split(2)[1])

        with patch.object(
            communicator,
            "attn_tp_reduce_scatter_tensor",
            side_effect=reduce_scatter,
        ) as scatter:
            output, output_residual = (
                CommunicateWithAllReduceAndLayerNormFn._scatter_hidden_states_and_residual(
                    hidden_states,
                    residual,
                    None,
                    layernorm,
                    context,
                    residual_input_mode=ScatterMode.TP_ATTN_FULL,
                )
            )

        scatter.assert_called_once()
        layernorm.assert_called_once()
        torch.testing.assert_close(output, hidden_states.tensor_split(2)[1] + 1)
        torch.testing.assert_close(output_residual, residual.tensor_split(2)[1] + 1)


if __name__ == "__main__":
    unittest.main()
