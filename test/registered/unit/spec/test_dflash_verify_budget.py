import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import resolve_num_tokens_per_req
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDFlashVerifyBudget(unittest.TestCase):
    def _server_args(self, verify_budget):
        return SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=16,
            speculative_dflash_verify_budget=verify_budget,
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


if __name__ == "__main__":
    unittest.main()
