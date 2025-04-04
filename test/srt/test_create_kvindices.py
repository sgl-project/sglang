import itertools
import unittest

import numpy as np
import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.test.test_utils import CustomTestCase


class TestCreateKvIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_test(self, batch, max_batch, max_context_len):
        req_to_token = torch.arange(
            max_batch * max_context_len, dtype=torch.int32, device="cuda"
        ).reshape((max_batch, max_context_len))
        req_pool_indices = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_batch), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device="cuda",
        )
        paged_kernel_lens = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_context_len), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device="cuda",
        )

        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # ref
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
        kv_indices_ref = torch.cat(
            [
                req_to_token[req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]]
                for i in range(batch)
            ],
            dim=0,
        ).contiguous()

        # triton
        kv_indices_triton = torch.empty(kv_indptr[-1], dtype=torch.int32, device="cuda")
        create_flashinfer_kv_indices_triton[(batch,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            kv_indptr,
            None,
            kv_indices_triton,
            req_to_token.size(1),
        )

        # Check
        self.assertTrue(torch.equal(kv_indices_ref, kv_indices_triton))

    def test_create_kvindices(self):
        BATCH = [1, 37, 1786]
        MAX_BATCH = 4096
        MAX_CONTEXT_LEN = 4096
        for batch in BATCH:
            self._run_test(batch, MAX_BATCH, MAX_CONTEXT_LEN)


if __name__ == "__main__":
    unittest.main()
