import unittest

import torch

from sglang.srt.layers.attention.utils import (
    create_triton_kv_indices_for_dcp_triton,
    get_dcp_lens,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-c-test-large-8-gpu-amd-mi35x")


class TestTritonDCPUtils(CustomTestCase):
    def test_get_dcp_lens_without_start(self):
        lens = torch.tensor([0, 1, 2, 3, 4, 5, 6], dtype=torch.int32)

        torch.testing.assert_close(
            get_dcp_lens(lens, dcp_size=2, dcp_rank=0),
            torch.tensor([0, 1, 1, 2, 2, 3, 3], dtype=torch.int32),
        )
        torch.testing.assert_close(
            get_dcp_lens(lens, dcp_size=2, dcp_rank=1),
            torch.tensor([0, 0, 1, 1, 2, 2, 3], dtype=torch.int32),
        )

    def test_get_dcp_lens_with_start(self):
        lens = torch.tensor([0, 1, 2, 5, 6, 7], dtype=torch.int32)
        start = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int32)

        for dcp_rank in range(2):
            expected = []
            for s, n in zip(start.tolist(), lens.tolist()):
                expected.append(
                    sum(1 for pos in range(s, s + n) if pos % 2 == dcp_rank)
                )
            torch.testing.assert_close(
                get_dcp_lens(lens, dcp_size=2, dcp_rank=dcp_rank, start=start),
                torch.tensor(expected, dtype=torch.int32),
            )

    def test_create_triton_kv_indices_for_dcp_triton(self):
        if not torch.cuda.is_available():
            self.skipTest("Triton DCP KV-index test requires CUDA")

        dcp_size = 2
        req_to_token = torch.arange(2 * 16, device="cuda", dtype=torch.int64).view(
            2, 16
        )
        req_pool_indices = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
        lens = torch.tensor([7, 6], device="cuda", dtype=torch.int32)
        start = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

        for dcp_rank in range(dcp_size):
            dcp_lens = get_dcp_lens(lens, dcp_size, dcp_rank, start).to(torch.int32)
            kv_indptr = torch.empty(3, device="cuda", dtype=torch.int64)
            kv_indptr[0] = 0
            kv_indptr[1:] = torch.cumsum(dcp_lens.to(torch.int64), dim=0)
            kv_indices = torch.empty(
                int(dcp_lens.sum().item()), device="cuda", dtype=torch.int64
            )

            create_triton_kv_indices_for_dcp_triton[(len(req_pool_indices),)](
                req_to_token,
                req_pool_indices,
                dcp_lens,
                kv_indptr,
                start,
                kv_indices,
                req_to_token.stride(0),
                dcp_size,
                dcp_rank,
            )
            torch.cuda.synchronize()

            expected = []
            for req_idx, req_start, local_len in zip(
                req_pool_indices.tolist(), start.tolist(), dcp_lens.cpu().tolist()
            ):
                first = req_start + ((dcp_rank - req_start) % dcp_size)
                for offset in range(local_len):
                    abs_pos = first + offset * dcp_size
                    expected.append(req_to_token[req_idx, abs_pos].item() // dcp_size)

            torch.testing.assert_close(
                kv_indices.cpu(), torch.tensor(expected, dtype=torch.int64)
            )


if __name__ == "__main__":
    unittest.main()
