"""CUDA unit tests for the rank-major packed-DSA DCP index layout."""

import unittest

import torch

from sglang.kernels.ops.attention.dcp_kernels import (
    create_packed_dsa_dcp_kv_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestPackedDsaDcpIndices(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for Triton kernel tests")

    def _run_case(self, dcp_size, prefix_lens, extend_lens):
        prefix_lens_cpu = torch.tensor(prefix_lens, dtype=torch.int32)
        extend_lens_cpu = torch.tensor(extend_lens, dtype=torch.int32)
        seq_lens_cpu = prefix_lens_cpu + extend_lens_cpu

        prefix_cu_cpu = torch.zeros(len(prefix_lens), dtype=torch.int32)
        extend_cu_cpu = torch.zeros(len(extend_lens), dtype=torch.int32)
        if len(prefix_lens) > 1:
            prefix_cu_cpu[1:] = prefix_lens_cpu.cumsum(0)[:-1]
            extend_cu_cpu[1:] = extend_lens_cpu.cumsum(0)[:-1]
        kv_indptr_cpu = torch.zeros(len(prefix_lens) + 1, dtype=torch.int32)
        kv_indptr_cpu[1:] = seq_lens_cpu.cumsum(0)

        tensors = [
            tensor.cuda()
            for tensor in (
                kv_indptr_cpu,
                extend_lens_cpu,
                extend_cu_cpu,
                prefix_lens_cpu,
                prefix_cu_cpu,
            )
        ]
        got = torch.empty(int(seq_lens_cpu.sum()), dtype=torch.int32, device="cuda")
        prefix_storage = int(prefix_lens_cpu.sum())
        total_local_prefix = prefix_storage // dcp_size
        create_packed_dsa_dcp_kv_indices[(len(prefix_lens),)](
            *tensors,
            got,
            total_local_prefix,
            prefix_storage,
            dcp_size,
        )

        expected = []
        for req, (prefix_len, extend_len) in enumerate(zip(prefix_lens, extend_lens)):
            local_req_start = int(prefix_cu_cpu[req]) // dcp_size
            expected.extend(
                (offset % dcp_size) * total_local_prefix
                + local_req_start
                + offset // dcp_size
                for offset in range(prefix_len)
            )
            expected.extend(
                prefix_storage + int(extend_cu_cpu[req]) + offset
                for offset in range(extend_len)
            )

        torch.testing.assert_close(got.cpu(), torch.tensor(expected, dtype=torch.int32))
        self.assertEqual(len(set(expected)), len(expected))
        physical_rows = int(seq_lens_cpu.sum())
        self.assertTrue(all(0 <= index < physical_rows for index in expected))

        # Independent storage reconstruction check: populate the raw collective
        # layout rank by rank, then prove the produced indices recover every
        # request's logical prefix+extend sequence.
        logical_prefixes = [
            [req * 1_000_000 + offset for offset in range(prefix_len)]
            for req, prefix_len in enumerate(prefix_lens)
        ]
        logical_extends = [
            [500_000_000 + req * 1_000_000 + offset for offset in range(extend_len)]
            for req, extend_len in enumerate(extend_lens)
        ]
        physical = []
        for rank in range(dcp_size):
            for request_prefix in logical_prefixes:
                physical.extend(request_prefix[rank::dcp_size])
        for request_extend in logical_extends:
            physical.extend(request_extend)
        logical = []
        for request_prefix, request_extend in zip(logical_prefixes, logical_extends):
            logical.extend(request_prefix)
            logical.extend(request_extend)
        restored = torch.tensor(physical, dtype=torch.int64)[got.cpu().long()]
        torch.testing.assert_close(restored, torch.tensor(logical, dtype=torch.int64))

    def test_page_boundaries_and_mixed_batch(self):
        for dcp_size in (2, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                self._run_case(
                    dcp_size,
                    prefix_lens=[0, 64, 128, 192],
                    extend_lens=[1, 5, 2, 9],
                )

    def test_long_prefixes(self):
        for dcp_size in (2, 4, 8):
            with self.subTest(dcp_size=dcp_size):
                self._run_case(
                    dcp_size,
                    prefix_lens=[64 * 1024, 64 * 17],
                    extend_lens=[64, 1],
                )


if __name__ == "__main__":
    unittest.main()
