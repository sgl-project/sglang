"""AITER unified attention: CUDA-graph decode refreshes the block page table."""

import unittest

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")

_RUNNABLE = is_hip()
if _RUNNABLE:
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend


@unittest.skipUnless(_RUNNABLE, "requires HIP with AITER")
class TestAiterUnifiedGraphPageTable(CustomTestCase):
    MAX_BS = 4
    MAX_CONTEXT_LEN = 64

    def _make_backend(self, page_size):
        backend = object.__new__(AiterAttnBackend)
        backend.use_mla = False
        backend.use_triton_unified_attention = True
        backend.use_sliding_window_kv_pool = False
        backend.page_size = page_size
        backend.max_context_len = self.MAX_CONTEXT_LEN
        backend.device = "cuda"
        num_reqs = 6
        backend.req_to_token = (
            torch.randperm(num_reqs * self.MAX_CONTEXT_LEN, device="cuda")
            .to(torch.int32)
            .view(num_reqs, self.MAX_CONTEXT_LEN)
        )
        backend.cuda_graph_page_table = torch.zeros(
            (self.MAX_BS, -(-self.MAX_CONTEXT_LEN // page_size)),
            dtype=torch.int32,
            device="cuda",
        )
        backend.qo_indptr_unified_decode = torch.arange(
            self.MAX_BS + 1, dtype=torch.int32, device="cuda"
        )
        return backend

    def test_decode_replay_writes_page_table(self):
        req_pool_indices = torch.tensor([4, 1, 3], device="cuda")
        seq_lens = torch.tensor([5, 17, 33], device="cuda")
        bs = len(seq_lens)
        for page_size in (1, 16):
            with self.subTest(page_size=page_size):
                backend = self._make_backend(page_size)
                backend._apply_cuda_graph_metadata(
                    bs,
                    req_pool_indices,
                    seq_lens,
                    int(seq_lens.sum()),
                    ForwardMode.DECODE,
                    None,
                    seq_lens.cpu(),
                    None,
                )
                expected = backend.req_to_token[req_pool_indices, : int(seq_lens.max())]
                if page_size > 1:
                    expected = expected[:, ::page_size] // page_size
                table = backend.forward_metadata.kv_indices
                self.assertIs(table, backend.cuda_graph_page_table)
                self.assertTrue(torch.equal(table[:bs, : expected.shape[1]], expected))


if __name__ == "__main__":
    unittest.main()
