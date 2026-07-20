import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.arg_groups.speculative_hook import _resolve_dflash_verify_budget
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import resolve_num_tokens_per_req
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashVerifyBudget(unittest.TestCase):
    def _server_args(
        self,
        verify_budget,
        attention_backend="triton",
        prefill_attention_backend=None,
        decode_attention_backend=None,
        speculative_attention_mode="prefill",
    ):
        return SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=16,
            speculative_dflash_verify_budget=verify_budget,
            attention_backend=attention_backend,
            prefill_attention_backend=prefill_attention_backend,
            decode_attention_backend=decode_attention_backend,
            speculative_attention_mode=speculative_attention_mode,
        )

    def test_budget_validation(self):
        with self.assertRaisesRegex(ValueError, "to be in"):
            _resolve_dflash_verify_budget(self._server_args(0), 16)
        with self.assertRaisesRegex(ValueError, "to be in"):
            _resolve_dflash_verify_budget(self._server_args(17), 16)

    def test_reduced_budget_requires_triton_target_attention(self):
        with self.assertRaisesRegex(ValueError, "require Triton attention"):
            _resolve_dflash_verify_budget(
                self._server_args(4, attention_backend="flashinfer"), 16
            )

    def test_reduced_budget_checks_selected_hybrid_backend(self):
        prefill_verify = self._server_args(
            4,
            prefill_attention_backend="triton",
            decode_attention_backend="flashinfer",
        )
        self.assertEqual(_resolve_dflash_verify_budget(prefill_verify, 16), 4)

        decode_verify = self._server_args(
            4,
            prefill_attention_backend="flashinfer",
            decode_attention_backend="triton",
            speculative_attention_mode="decode",
        )
        self.assertEqual(_resolve_dflash_verify_budget(decode_verify, 16), 4)

    def test_full_budget_preserves_other_backends(self):
        self.assertEqual(
            _resolve_dflash_verify_budget(
                self._server_args(16, attention_backend="flashinfer"), 16
            ),
            16,
        )

    def test_target_verify_width_uses_independent_budget(self):
        server_args = self._server_args(4)

        target_width = resolve_num_tokens_per_req(
            phase="target_verify",
            server_args=server_args,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            is_draft_worker=False,
        )
        draft_width = resolve_num_tokens_per_req(
            phase="target_verify",
            server_args=server_args,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            is_draft_worker=True,
        )

        self.assertEqual(target_width, 4)
        self.assertEqual(draft_width, 16)

    def test_unset_budget_preserves_full_block(self):
        server_args = self._server_args(None)
        width = resolve_num_tokens_per_req(
            phase="target_verify",
            server_args=server_args,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            is_draft_worker=False,
        )
        self.assertEqual(width, 16)

    def test_triton_verify_metadata_uses_runtime_width(self):
        backend = object.__new__(TritonAttnBackend)
        backend.num_draft_tokens = 16
        backend.device = torch.device("cpu")
        backend.qo_indptr = torch.empty(3, dtype=torch.int32)
        backend.cuda_graph_kv_indices = torch.empty(1, dtype=torch.int64)
        backend.cuda_graph_custom_mask = torch.empty(1, dtype=torch.bool)
        backend.mask_indptr = torch.empty(3, dtype=torch.int64)
        backend.sliding_window_size = None
        backend.window_kv_indptr = None
        backend._fill_kv_indptr_and_indices = MagicMock(
            return_value=torch.tensor([0, 10, 30], dtype=torch.int32)
        )

        metadata = backend._update_target_verify_buffers(
            bs=2,
            seq_lens=torch.tensor([10, 20], dtype=torch.int32),
            req_pool_indices=torch.tensor([1, 2], dtype=torch.int32),
            spec_info=SimpleNamespace(draft_token_num=4, custom_mask=None),
        )

        qo_indptr, _, _, mask_indptr = metadata[:4]
        self.assertEqual(qo_indptr.tolist(), [0, 4, 8])
        self.assertEqual(mask_indptr.tolist(), [0, 56, 152])

    def test_scheduler_result_keeps_native_block_layout(self):
        worker = object.__new__(DFlashWorkerV2)
        worker.block_size = 16
        worker.verify_budget = 4
        worker.device = torch.device("cpu")
        worker._accept_bonus_buffer_cap = 0
        worker._result_tokens_bufs = []
        worker._result_tokens_buffer_slot = 0

        output = worker._pad_result_tokens(torch.arange(8).view(2, 4))

        self.assertEqual(output.shape, (2, 16))
        torch.testing.assert_close(output[:, :4], torch.arange(8).view(2, 4))
        torch.testing.assert_close(output[:, 4:], torch.zeros(2, 12, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
