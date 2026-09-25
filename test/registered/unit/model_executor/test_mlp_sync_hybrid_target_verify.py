import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor import forward_batch_info as fbi
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NUM_TOKENS_PER_REQ = 4


def _verify_spec_info():
    return SimpleNamespace(
        num_tokens_per_req=NUM_TOKENS_PER_REQ, is_draft_input=lambda: False
    )


def _sync(forward_mode, num_reqs, global_num_tokens):
    """Run prepare_mlp_sync_batch for one rank of a hybrid-SSM model in a DP
    step padded to MAX_LEN with an extend-mode batch somewhere in the group."""
    num_tokens = num_reqs * NUM_TOKENS_PER_REQ
    batch = ForwardBatch(
        forward_mode=forward_mode,
        batch_size=num_reqs,
        input_ids=torch.zeros(num_tokens, dtype=torch.long),
        req_pool_indices=torch.arange(num_reqs),
        seq_lens=torch.full((num_reqs,), 9),
        out_cache_loc=torch.arange(num_tokens),
        seq_lens_sum=9 * num_reqs,
        positions=torch.arange(num_tokens),
        spec_info=_verify_spec_info(),
        is_extend_in_batch=True,
        global_num_tokens_cpu=list(global_num_tokens),
        global_num_tokens_for_logprob_cpu=list(global_num_tokens),
        global_num_tokens_gpu=torch.zeros(len(global_num_tokens), dtype=torch.int64),
    )
    model_runner = MagicMock(is_draft_worker=False)
    model_runner.attn_tp_sequence_sharded.return_value = False
    no_prefill_graph = SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(bs=[]))
        )
    )
    with (
        patch.multiple(
            fbi,
            get_parallel=lambda: SimpleNamespace(attn_tp_size=1),
            get_exec=lambda: no_prefill_graph,
            _elastic_should_preserve_local_token_counts=lambda **kwargs: False,
            dp_gather_slot=lambda: 0,
            set_dp_buffer_len_from_batch=lambda *args: None,
            set_is_extend_in_batch=lambda *args: None,
            _is_cpu=True,
        ),
        patch.object(
            fbi, "mambaish_config", lambda model_config: object(), create=True
        ),
        patch.object(
            fbi.DpPaddingMode,
            "get_dp_padding_mode",
            return_value=fbi.DpPaddingMode.MAX_LEN,
        ),
        patch.object(ForwardBatch, "_pad_inputs_to_size"),
        patch(
            "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
        ),
    ):
        batch.prepare_mlp_sync_batch(model_runner)
    return batch


class TestMlpSyncHybridTargetVerify(CustomTestCase):
    def test_verify_batch_keeps_its_verify_layout(self):
        """In a DP step padded to MAX_LEN while another rank prefills, a hybrid-SSM
        verify batch keeps its verify layout: one request per draft width."""
        batch = _sync(ForwardMode.TARGET_VERIFY, num_reqs=2, global_num_tokens=[8, 0])
        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(batch.batch_size, 2)
        self.assertIsNone(batch.extend_seq_lens)

    def test_idle_rank_joins_as_verify_batch(self):
        """An idle rank in the same verify step runs as a verify batch of the
        padded width, not as a fabricated prefill."""
        batch = _sync(ForwardMode.IDLE, num_reqs=0, global_num_tokens=[0, 8])
        self.assertEqual(batch.forward_mode, ForwardMode.TARGET_VERIFY)
        self.assertEqual(batch._original_forward_mode, ForwardMode.IDLE)
        self.assertEqual(batch.batch_size, 8 // NUM_TOKENS_PER_REQ)


if __name__ == "__main__":
    unittest.main()
