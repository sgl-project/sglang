"""CPU unit tests for publishing DP-attention buffer sizes from a ForwardBatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    _DpGatheredBufferWrapper,
    get_dp_global_num_tokens,
    get_global_dp_buffer_len,
    get_local_dp_buffer_len,
    is_dp_max_padding,
    set_dp_buffer_len,
    set_dp_buffer_len_from_batch,
)
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(**overrides):
    fields = dict(
        global_dp_buffer_len=8,
        global_num_tokens_cpu=[3, 1],
        global_num_tokens_padded_cpu=[4, 4],
        global_num_tokens_gpu=torch.tensor([3, 1]),
        dp_padding_mode=DpPaddingMode.MAX_LEN,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestSetDpBufferLenFromBatch(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(
            device="cpu",
            tp_size=4,
            attn_dp_size=2,
            cuda_graph_config=CudaGraphConfig(prefill=PhaseConfig(bs=[])),
        )
        override.install()
        self.addCleanup(override.restore)
        self.addCleanup(set_dp_buffer_len, 0, 0, False)

    def test_publishes_the_padded_list_and_this_ranks_entry(self):
        batch = _batch()
        with get_parallel().override(attn_dp_rank=1):
            set_dp_buffer_len_from_batch(batch)
        self.assertEqual(get_global_dp_buffer_len(), 8)
        self.assertEqual(get_local_dp_buffer_len(), 4)
        self.assertTrue(is_dp_max_padding())
        self.assertEqual(get_dp_global_num_tokens(), [4, 4])
        self.assertIs(
            _DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu(),
            batch.global_num_tokens_gpu,
        )

    def test_falls_back_to_the_raw_list_when_no_padded_list_is_carried(self):
        batch = _batch(
            global_dp_buffer_len=4,
            global_num_tokens_padded_cpu=None,
            dp_padding_mode=DpPaddingMode.SUM_LEN,
        )
        with get_parallel().override(attn_dp_rank=0):
            set_dp_buffer_len_from_batch(batch)
        self.assertEqual(get_local_dp_buffer_len(), 3)
        self.assertFalse(is_dp_max_padding())
        self.assertEqual(get_dp_global_num_tokens(), [3, 1])

    def test_a_single_entry_list_ignores_the_rank(self):
        batch = _batch(
            global_dp_buffer_len=3,
            global_num_tokens_cpu=[3],
            global_num_tokens_padded_cpu=None,
            global_num_tokens_gpu=torch.tensor([3]),
            dp_padding_mode=DpPaddingMode.SUM_LEN,
        )
        with get_parallel().override(attn_dp_rank=1):
            set_dp_buffer_len_from_batch(batch)
        self.assertEqual(get_local_dp_buffer_len(), 3)

    def test_real_counts_are_independent_of_gather_counts(self):
        real_counts = torch.tensor([3, 0])
        batch = _batch(
            global_num_tokens_gpu=torch.tensor([4, 4]),
            global_num_tokens_unpadded_gpu=real_counts,
        )
        with get_parallel().override(attn_dp_rank=0):
            set_dp_buffer_len_from_batch(batch)
        self.assertEqual(get_dp_global_num_tokens(), [4, 4])
        self.assertIs(
            _DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu(), real_counts
        )

    def test_real_counts_survive_multiple_prefill_slices(self):
        real_counts = torch.tensor([3, 0])
        batch = SimpleNamespace(
            global_num_tokens_cpu=[3, 0],
            global_num_tokens_gpu=real_counts,
            global_num_tokens_unpadded_cpu=None,
            global_num_tokens_unpadded_gpu=None,
            global_num_tokens_for_logprob_cpu=[1, 0],
            batch_size=1,
            is_extend_in_batch=True,
            forward_mode=ForwardMode.SPLIT_PREFILL,
            can_run_dp_prefill_cuda_graph=False,
            dp_spec_prefill_coordination_applied=False,
            tbo_children=None,
            _pad_inputs_to_size=lambda *args: None,
        )
        runner = SimpleNamespace(
            is_draft_worker=False, attn_tp_sequence_sharded=lambda tokens: False
        )
        prefix = "sglang.srt.model_executor.forward_batch_info"
        with (
            get_parallel().override(attn_dp_rank=0, attn_tp_size=2),
            patch(f"{prefix}._is_cpu", True),
            patch(f"{prefix}._mega_moe_materializes_idle_rank", return_value=False),
            patch(
                f"{prefix}._elastic_should_preserve_local_token_counts",
                return_value=False,
            ),
            patch.object(
                DpPaddingMode, "get_dp_padding_mode", return_value=DpPaddingMode.MAX_LEN
            ),
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
            ),
        ):
            for _ in range(3):
                ForwardBatch.prepare_mlp_sync_batch(batch, runner)
                self.assertEqual(batch.global_num_tokens_cpu, [4, 4])
                self.assertEqual(batch.global_num_tokens_unpadded_cpu, [3, 0])
                torch.testing.assert_close(real_counts, torch.tensor([3, 0]))
                torch.testing.assert_close(
                    batch.global_num_tokens_gpu, torch.tensor([4, 4])
                )
                self.assertIsNot(batch.global_num_tokens_gpu, real_counts)

            with patch.object(
                DpPaddingMode, "get_dp_padding_mode", return_value=DpPaddingMode.SUM_LEN
            ):
                ForwardBatch.prepare_mlp_sync_batch(batch, runner)
            self.assertEqual(batch.global_num_tokens_padded_cpu, [4, 0])
            self.assertEqual(batch.global_dp_buffer_len, 4)


if __name__ == "__main__":
    unittest.main()
