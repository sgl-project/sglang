import itertools
import unittest

import numpy as np
import torch

from sglang.srt.layers.attention.utils import (
    create_flashinfer_kv_indices_triton,
    create_flashmla_kv_indices_triton,
)
from sglang.test.test_utils import CustomTestCase


class TestCreateKvIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_test(self, batch, max_batch, max_context_len, page_size):
        np.random.seed(9)
        PAGE_SIZE = page_size
        req_to_token = torch.arange(
            max_batch * max_context_len, dtype=torch.int32, device="cuda"
        ).reshape((max_batch, max_context_len))

        # the block table
        req_pool_indices = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_batch), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device="cuda",
        )
        seq_lens = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_context_len), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device="cuda",
        )
        num_pages_per_req = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(num_pages_per_req, dim=0)

        # ref
        kv_indices_ref = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        seq_lens_cpu = seq_lens.cpu().numpy()
        for i in range(batch):
            kv_indptr_req = kv_indptr[i]
            num_toks_seq = seq_lens_cpu[i]
            curr_req_pool = req_pool_indices_cpu[i]
            curr_num_pages = num_pages_per_req[i]
            curr_token_ids = req_to_token[curr_req_pool]
            curr_pages = (curr_token_ids[:num_toks_seq] // PAGE_SIZE).unique()
            assert (
                len(curr_pages) == curr_num_pages
            ), f"req {i} has #{curr_num_pages} pages, but got {len(curr_pages)} pages"
            kv_indices_ref[kv_indptr_req : kv_indptr_req + curr_num_pages] = curr_pages

        # triton
        kv_indices_triton = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
        create_flashinfer_kv_indices_triton[(batch,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices_triton,
            req_to_token.size(1),
            PAGE_SIZE,
        )
        max_pages = max_context_len // PAGE_SIZE
        kv_indices_flashmla = torch.empty(
            batch, max_pages, dtype=torch.int32, device="cuda"
        )

        create_flashmla_kv_indices_triton[(batch,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            None,
            kv_indices_flashmla,
            req_to_token.size(1),
            max_pages,
            PAGE_SIZE,
        )
        # Check
        self.assertTrue(torch.equal(kv_indices_ref, kv_indices_triton))

    def test_create_kvindices(self):
        BATCH = [4, 37, 512, 1786]
        MAX_BATCH = 4096
        MAX_CONTEXT_LEN = 4096
        PAGE_SIZE = [1, 2, 16, 64]
        # for debug
        # BATCH = [4]
        # MAX_BATCH = 4
        # MAX_CONTEXT_LEN = 10
        # Test for small batch size
        for page_size in PAGE_SIZE[:1]:
            print(f"Running test for page size: {page_size} and batch size: {BATCH[0]}")
            self._run_test(BATCH[0], MAX_BATCH, MAX_CONTEXT_LEN, page_size)

        # Test for larger batch size
        for batch in BATCH[1:]:
            for page_size in PAGE_SIZE:
                print(
                    f"Running test for batch size: {batch} and page size: {page_size}"
                )
                self._run_test(batch, MAX_BATCH, MAX_CONTEXT_LEN, page_size)


if __name__ == "__main__":
    unittest.main()
