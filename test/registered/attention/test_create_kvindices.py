import unittest

import numpy as np
import torch

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Triton kernel unit test for KV indices creation
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestCreateKvIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device(get_device())

    def _run_test(self, batch, max_batch, max_context_len):
        req_to_token = torch.arange(
            max_batch * max_context_len, dtype=torch.int32, device=get_device()
        ).reshape((max_batch, max_context_len))
        req_pool_indices = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_batch), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device=get_device(),
        )
        paged_kernel_lens = torch.tensor(
            torch.from_numpy(
                np.random.choice(range(max_context_len), size=batch, replace=False)
            ),
            dtype=torch.int32,
            device=get_device(),
        )

        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32, device=get_device())
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
        kv_indices_triton = torch.empty(
            kv_indptr[-1], dtype=torch.int32, device=get_device()
        )
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

    def _run_page_table_test(self, batch, ps, with_window_start):
        """ENTRY_PAGE_SIZE > 1: the source is a PAGE-granular table (the unified
        pool's read table); the kernel must reconstruct token ids by the affine
        rule token = entry * ps + pos % ps -- including the kv_start_idx
        (sliding-window) offset path, whose pos is an absolute token position."""
        max_batch, max_pages = 64, 128
        page_table = torch.randint(
            0, 1 << 20, (max_batch, max_pages), dtype=torch.int32
        )
        req_pool_indices = torch.tensor(
            np.random.choice(range(max_batch), size=batch, replace=False),
            dtype=torch.int32,
        )
        lens = torch.tensor(
            np.random.randint(1, max_pages * ps, size=batch), dtype=torch.int32
        )
        if with_window_start:
            start = torch.clamp(
                lens - torch.randint(1, ps * 3, (batch,), dtype=torch.int32), min=0
            )
            gather_lens = lens - start
        else:
            start, gather_lens = None, lens
        kv_indptr = torch.zeros((batch + 1,), dtype=torch.int32)
        kv_indptr[1:] = torch.cumsum(gather_lens, dim=0)

        # ref: absolute positions [start, start+len) through the affine rule
        refs = []
        for i in range(batch):
            s = int(start[i]) if start is not None else 0
            pos = torch.arange(s, s + int(gather_lens[i]), dtype=torch.int64)
            entry = page_table[int(req_pool_indices[i])][pos // ps].to(torch.int64)
            refs.append(entry * ps + pos % ps)
        ref = torch.cat(refs).contiguous()

        out = torch.empty(int(kv_indptr[-1]), dtype=torch.int64)
        create_flashinfer_kv_indices_triton[(batch,)](
            page_table,
            req_pool_indices,
            gather_lens,
            kv_indptr,
            start,
            out,
            page_table.size(1),
            ENTRY_PAGE_SIZE=ps,
        )
        self.assertTrue(torch.equal(ref, out))

    def test_page_table_source_reconstruction(self):
        for batch in (1, 37):
            for ps in (4, 64, 256):
                self._run_page_table_test(batch, ps, with_window_start=False)
                self._run_page_table_test(batch, ps, with_window_start=True)


if __name__ == "__main__":
    unittest.main()
