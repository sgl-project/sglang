import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

import sglang.srt.speculative.ngram_worker as ngram_worker_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker

DRAFT_TOKEN_NUM = 3


def _make_worker(watermark_state, target_result):
    worker = object.__new__(NGRAMWorker)
    worker.device = "cpu"
    worker.draft_token_num = DRAFT_TOKEN_NUM
    worker.speculative_num_draft_tokens = DRAFT_TOKEN_NUM
    worker.model_runner = SimpleNamespace(watermark_state=watermark_state)
    worker._target_worker = SimpleNamespace(
        forward_batch_generation=Mock(return_value=target_result)
    )
    worker.token_to_kv_pool_allocator = None
    worker.ngram_corpus = Mock()
    worker._prev_decode_rids = set()
    worker._prepare_for_speculative_decoding = Mock()
    worker._update_ngram_corpus = Mock()
    return worker


def _make_batch(forward_mode, num_reqs):
    return SimpleNamespace(
        reqs=[SimpleNamespace(rid=f"r{forward_mode.name}{i}") for i in range(num_reqs)],
        forward_mode=forward_mode,
        spec_info=NgramVerifyInput(
            draft_token_num=DRAFT_TOKEN_NUM,
            new_seq_lens=torch.zeros(num_reqs, dtype=torch.int64),
        ),
        has_grammar=False,
        return_logprob=False,
        seq_lens=torch.full((num_reqs,), 8, dtype=torch.int64),
        req_pool_indices=torch.arange(num_reqs, dtype=torch.int64),
        sampling_info=SimpleNamespace(has_watermark_candidates=True),
    )


class TestNgramAcceptTokensLayout(CustomTestCase):
    def test_decode_and_prefill_draft_inputs_merge(self):
        """A verify batch's accept_tokens must stay flat so it merges with a prefill batch."""
        decode_reqs, prefill_reqs = 2, 1
        predict = torch.arange(decode_reqs * DRAFT_TOKEN_NUM, dtype=torch.int32)
        accept_index = torch.tensor([[0, 1, -1], [3, -1, -1]], dtype=torch.int64)
        accept_lens = torch.tensor([2, 1], dtype=torch.int32)
        watermark_state = Mock()
        verify_result = SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=torch.zeros(1)),
            can_run_cuda_graph=False,
        )
        prefill_result = SimpleNamespace(
            logits_output=None,
            next_token_ids=torch.tensor([7], dtype=torch.int32),
            can_run_cuda_graph=False,
        )

        with (
            patch.object(ngram_worker_module, "record_stream_for_v2_verify"),
            patch.object(ngram_worker_module, "set_time_batch"),
            patch.object(ngram_worker_module, "maybe_detect_nan"),
            patch.object(ngram_worker_module, "maybe_detect_inf"),
            patch.object(ngram_worker_module, "commit_mamba_states_after_verify"),
            patch.object(ngram_worker_module, "move_accept_tokens_to_target_kvcache"),
            patch.object(
                ngram_worker_module,
                "eagle_sample",
                return_value=(predict, accept_lens, accept_index),
            ),
        ):
            decode = _make_worker(
                watermark_state, verify_result
            ).forward_batch_generation(
                _make_batch(ForwardMode.TARGET_VERIFY, decode_reqs)
            )
            prefill = _make_worker(None, prefill_result).forward_batch_generation(
                _make_batch(ForwardMode.EXTEND, prefill_reqs)
            )

        appended = watermark_state.append_speculative.call_args.args[1]
        self.assertEqual(tuple(appended.shape), (decode_reqs, DRAFT_TOKEN_NUM))

        merged = decode.next_draft_input
        merged.merge_batch(prefill.next_draft_input)
        self.assertEqual(
            tuple(merged.accept_tokens.shape),
            ((decode_reqs + prefill_reqs) * DRAFT_TOKEN_NUM,),
        )
        self.assertEqual(merged.accept_lens.tolist(), [2, 1, 1])


if __name__ == "__main__":
    unittest.main()
