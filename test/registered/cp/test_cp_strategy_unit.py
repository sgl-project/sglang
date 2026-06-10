import functools
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.communicator import (
    CommunicateContext,
    CommunicateSummableTensorPairFn,
    CommunicateWithAllReduceAndLayerNormFn,
    ScatterMode,
)
from sglang.srt.layers.cp.base import ContextParallelStrategyKind
from sglang.srt.layers.cp.interleave import (
    _interleave_split_q_seqs_cpu,
    _strided_take,
)
from sglang.srt.layers.cp.utils import prepare_context_parallel_metadata
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _build_zigzag_metadata(extend_lens, cp_rank, cp_size):
    with (
        patch(
            "sglang.srt.layers.attention.dsa.utils.is_dsa_prefill_cp_round_robin_split",
            return_value=False,
        ),
        patch(
            "sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp",
            return_value=False,
        ),
    ):
        return prepare_context_parallel_metadata(
            kv_len=sum(extend_lens),
            cp_rank=cp_rank,
            cp_size=cp_size,
            seqs_len=extend_lens,
            extend_seqs_len=extend_lens,
            device="cpu",
        )


def _expected_interleave_lengths(extend_lens, cp_size, cp_rank):
    offset = 0
    q_lens = []
    bs_idx = []
    for i, cur_len in enumerate(extend_lens):
        cur_q_len = sum(
            1 for pos in range(offset, offset + cur_len) if pos % cp_size == cp_rank
        )
        if cur_q_len > 0:
            q_lens.append(cur_q_len)
            bs_idx.append(i)
        offset += cur_len
    return q_lens, bs_idx


def _fake_communicate_context():
    process_group_sizes = {
        ScatterMode.SCATTERED: 1,
        ScatterMode.TP_ATTN_FULL: 1,
        ScatterMode.FULL: 1,
        ScatterMode.MOE_FULL: 1,
    }
    return CommunicateContext(
        process_group_sizes=process_group_sizes,
        attn_tp_rank=0,
        attn_tp_size=1,
        attn_dp_size=1,
        attn_cp_rank=0,
        attn_cp_size=4,
        tp_size=4,
        tp_rank=0,
    )


class TestCPStrategyUnit(CustomTestCase):
    def test_strategy_kind_enum_values(self):
        self.assertEqual(ContextParallelStrategyKind.NONE.value, 0)
        self.assertEqual(
            ContextParallelStrategyKind.from_string("zigzag"),
            ContextParallelStrategyKind.ZIGZAG,
        )
        self.assertEqual(
            ContextParallelStrategyKind.from_string("interleave"),
            ContextParallelStrategyKind.INTERLEAVE,
        )

    def test_zigzag_metadata_index_invariants(self):
        cp_size = 4
        for extend_lens in ([1], [8], [17], [3, 16, 1025]):
            cp_segment_num = cp_size * 2
            for cp_rank in range(cp_size):
                meta = _build_zigzag_metadata(list(extend_lens), cp_rank, cp_size)

                self.assertEqual(
                    len(meta.split_list), len(extend_lens) * cp_segment_num
                )
                self.assertEqual(sum(meta.split_list), sum(extend_lens))
                self.assertEqual(len(meta.zigzag_index), len(extend_lens) * 2)
                self.assertEqual(
                    sorted(meta.cp_reverse_index),
                    list(range(len(extend_lens) * cp_segment_num)),
                )
                self.assertEqual(sum(meta.per_rank_actual_token), sum(extend_lens))

                local_tokens = sum(meta.split_list[i] for i in meta.zigzag_index)
                self.assertEqual(local_tokens, meta.per_rank_actual_token[cp_rank])
                self.assertEqual(
                    local_tokens,
                    meta.total_q_prev_tokens + meta.total_q_next_tokens,
                )
                self.assertEqual(
                    meta.cu_seqlens_q_prev_tensor.cpu().tolist()[-1],
                    meta.total_q_prev_tokens,
                )
                self.assertEqual(
                    meta.cu_seqlens_q_next_tensor.cpu().tolist()[-1],
                    meta.total_q_next_tokens,
                )

    def test_interleave_strided_take_tensor_list_tuple(self):
        tensor = torch.arange(11)
        values = list(range(11))
        cp_size = 4
        for cp_rank in range(cp_size):
            expected = list(range(cp_rank, 11, cp_size))
            self.assertEqual(_strided_take(tensor, cp_rank, cp_size).tolist(), expected)
            self.assertEqual(_strided_take(values, cp_rank, cp_size), expected)
            self.assertEqual(
                _strided_take(tuple(values), cp_rank, cp_size),
                tuple(expected),
            )

        self.assertEqual(_strided_take(torch.arange(2), 3, cp_size).numel(), 0)
        self.assertEqual(_strided_take([0, 1], 3, cp_size), [])

    def test_interleave_ragged_batch_lengths(self):
        extend_lens = [1, 2, 9, 0, 17]
        cp_size = 4
        for cp_rank in range(cp_size):
            self.assertEqual(
                _interleave_split_q_seqs_cpu(extend_lens, cp_size, cp_rank),
                _expected_interleave_lengths(extend_lens, cp_size, cp_rank),
            )

    def test_attn_cp_communicator_operator_selection_uses_layout_not_size(self):
        context = _fake_communicate_context()

        with patch(
            "sglang.srt.layers.communicator._is_attn_cp_layer_comm_enabled",
            return_value=True,
        ):
            gather_fn = CommunicateWithAllReduceAndLayerNormFn.get_fn(
                hidden_states_input_mode=ScatterMode.SCATTERED,
                residual_input_mode=ScatterMode.SCATTERED,
                hidden_states_output_mode=ScatterMode.FULL,
                residual_output_mode=ScatterMode.SCATTERED,
                context=context,
            )
            scatter_fn = CommunicateSummableTensorPairFn.get_fn(
                hidden_states_input_mode=ScatterMode.FULL,
                residual_input_mode=ScatterMode.SCATTERED,
                output_mode=ScatterMode.SCATTERED,
                context=context,
            )

        self.assertIsInstance(gather_fn, functools.partial)
        self.assertIs(
            gather_fn.func,
            CommunicateWithAllReduceAndLayerNormFn._cp_gather_hidden_states_and_residual,
        )
        self.assertIs(
            scatter_fn,
            CommunicateSummableTensorPairFn._cp_scatter_hidden_states,
        )

    def test_deprecated_mode_flags_enable_unified_cp(self):
        server_args = object.__new__(ServerArgs)
        server_args.enable_prefill_context_parallel = False
        server_args.enable_dsa_prefill_context_parallel = False
        server_args.enable_prefill_cp = False
        server_args.cp_strategy = "zigzag"
        server_args.dsa_prefill_cp_mode = "round-robin-split"
        server_args.prefill_cp_mode = "in-seq-split"
        server_args._legacy_dsa_prefill_cp_mode = "round-robin-split"
        server_args._legacy_prefill_cp_mode = None

        server_args._handle_context_parallelism(validate_topology=False)

        self.assertTrue(server_args.enable_prefill_cp)
        self.assertEqual(server_args.cp_strategy, "interleave")
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")


if __name__ == "__main__":
    unittest.main()
