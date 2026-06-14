"""Unit tests for common.py — GPU-required tests"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.common import (
    get_last_loc_torch,
    get_last_loc_triton,
    write_cache_indices,
)
from sglang.test.test_utils import CustomTestCase

CUDA = torch.cuda.is_available()


# ---------------------------------------------------------------------------
# get_last_loc_triton (GPU)
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA, "CUDA required")
class TestGetLastLocTriton(CustomTestCase):
    D = "cuda"

    def _pool(self, rows, cols):
        return torch.arange(rows * cols, dtype=torch.int64, device=self.D).reshape(
            rows, cols
        )

    def _t(self, data):
        return torch.tensor(data, dtype=torch.int64, device=self.D)

    def test_matches_torch_impl_single(self):
        pool = self._pool(4, 8)
        req, pre = self._t([0]), self._t([3])
        self.assertTrue(
            torch.equal(
                get_last_loc_torch(pool, req, pre),
                get_last_loc_triton(pool, req, pre),
            )
        )

    def test_matches_torch_impl_batch(self):
        pool = self._pool(8, 16)
        req = self._t([0, 1, 2, 3])
        pre = self._t([0, 4, 2, 8])
        self.assertTrue(
            torch.equal(
                get_last_loc_torch(pool, req, pre),
                get_last_loc_triton(pool, req, pre),
            )
        )

    def test_zero_prefix_returns_minus_one(self):
        pool = self._pool(4, 8)
        out = get_last_loc_triton(pool, self._t([1]), self._t([0]))
        self.assertEqual(out[0].item(), -1)

    def test_all_zero_prefix(self):
        pool = self._pool(4, 8)
        out = get_last_loc_triton(pool, self._t([0, 1, 2, 3]), self._t([0, 0, 0, 0]))
        self.assertTrue((out == -1).all())

    def test_output_shape(self):
        pool = self._pool(8, 16)
        pre = self._t([1, 2, 3])
        out = get_last_loc_triton(pool, self._t([0, 1, 2]), pre)
        self.assertEqual(out.shape, pre.shape)

    def test_large_batch(self):
        n = 256
        pool = self._pool(n, 32)
        req = torch.arange(n, dtype=torch.int64, device=self.D)
        pre = torch.randint(1, 32, (n,), dtype=torch.int64, device=self.D)
        self.assertTrue(
            torch.equal(
                get_last_loc_torch(pool, req, pre),
                get_last_loc_triton(pool, req, pre),
            )
        )


# ---------------------------------------------------------------------------
# get_last_loc dispatcher
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA, "CUDA required")
class TestGetLastLocDispatcher(CustomTestCase):
    def _pool(self, rows, cols, device="cuda"):
        return torch.arange(rows * cols, dtype=torch.int64, device=device).reshape(
            rows, cols
        )

    def test_routes_to_triton_for_default_backend(self):
        from sglang.srt.mem_cache.common import get_last_loc

        pool = self._pool(4, 8)
        req = torch.tensor([0], dtype=torch.int64, device="cuda")
        pre = torch.tensor([3], dtype=torch.int64, device="cuda")
        with patch("sglang.srt.mem_cache.common.get_global_server_args") as mock_args:
            mock_args.return_value.attention_backend = "flashinfer"
            out = get_last_loc(pool, req, pre)
        self.assertEqual(out[0].item(), pool[0, 2].item())

    def test_routes_to_torch_for_torch_native_backend(self):
        from sglang.srt.mem_cache.common import get_last_loc

        pool = self._pool(4, 8)
        req = torch.tensor([0], dtype=torch.int64, device="cuda")
        pre = torch.tensor([3], dtype=torch.int64, device="cuda")
        with patch("sglang.srt.mem_cache.common.get_global_server_args") as mock_args:
            mock_args.return_value.attention_backend = "torch_native"
            out = get_last_loc(pool, req, pre)
        self.assertEqual(out[0].item(), pool[0, 2].item())

    def test_routes_to_torch_for_ascend_backend(self):
        from sglang.srt.mem_cache.common import get_last_loc

        pool = self._pool(4, 8)
        req = torch.tensor([0], dtype=torch.int64, device="cuda")
        pre = torch.tensor([0], dtype=torch.int64, device="cuda")
        with patch("sglang.srt.mem_cache.common.get_global_server_args") as mock_args:
            mock_args.return_value.attention_backend = "ascend"
            out = get_last_loc(pool, req, pre)
        self.assertEqual(out[0].item(), -1)

    def test_triton_and_torch_agree(self):
        from sglang.srt.mem_cache.common import get_last_loc

        pool = self._pool(8, 16)
        req = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")
        pre = torch.tensor([0, 4, 2, 8], dtype=torch.int64, device="cuda")

        with patch("sglang.srt.mem_cache.common.get_global_server_args") as mock_args:
            mock_args.return_value.attention_backend = "flashinfer"
            triton_out = get_last_loc(pool, req, pre)

        with patch("sglang.srt.mem_cache.common.get_global_server_args") as mock_args:
            mock_args.return_value.attention_backend = "torch_native"
            torch_out = get_last_loc(pool, req, pre)

        self.assertTrue(torch.equal(triton_out, torch_out))


# ---------------------------------------------------------------------------
# write_cache_indices — Triton path
# ---------------------------------------------------------------------------


@unittest.skipUnless(CUDA, "CUDA required")
class TestWriteCacheIndicesTritonPath(CustomTestCase):
    def test_triton_path_executes_without_error(self):
        n_reqs, prefix_len, seq_len = 2, 2, 4
        extend_len = seq_len - prefix_len
        max_ctx = 16

        req_to_token = torch.zeros(n_reqs, max_ctx, dtype=torch.int32, device="cuda")
        pool = MagicMock()
        pool.device = "cuda"
        pool.req_to_token = req_to_token

        out_cache_loc = torch.arange(
            n_reqs * extend_len, dtype=torch.int32, device="cuda"
        )
        req_idx_dev = torch.arange(n_reqs, dtype=torch.int64, device="cuda")
        req_idx_cpu = req_idx_dev.cpu()
        pre_dev = torch.full((n_reqs,), prefix_len, dtype=torch.int64, device="cuda")
        pre_cpu = pre_dev.cpu()
        seq_dev = torch.full((n_reqs,), seq_len, dtype=torch.int64, device="cuda")
        seq_cpu = seq_dev.cpu()
        ext_dev = torch.full((n_reqs,), extend_len, dtype=torch.int64, device="cuda")
        ext_cpu = ext_dev.cpu()
        prefix_tensors = [
            torch.zeros(prefix_len, dtype=torch.int64, device="cuda")
            for _ in range(n_reqs)
        ]

        with patch(
            "sglang.srt.mem_cache.common.support_triton", return_value=True
        ), patch("sglang.srt.mem_cache.common.get_global_server_args") as mock_args:
            mock_args.return_value.attention_backend = "flashinfer"
            write_cache_indices(
                out_cache_loc,
                req_idx_dev,
                req_idx_cpu,
                pre_dev,
                pre_cpu,
                seq_dev,
                seq_cpu,
                ext_dev,
                ext_cpu,
                prefix_tensors,
                pool,
            )


if __name__ == "__main__":
    unittest.main()
