"""Host-side extend bookkeeping for backends without Triton.

The non-Triton paths of ``compute_position`` and ``write_cache_indices`` build
their ragged index vectors on the host and touch the device once. These tests
pin them to the previous per-request torch implementations.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocation import (
    write_cache_indices,
    write_cache_indices_torch,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    compute_position,
    compute_position_host,
    compute_position_torch,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

if torch.cuda.is_available():
    _ACCEL_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _ACCEL_DEVICE = "mps"
else:
    _ACCEL_DEVICE = None

# (extend_prefix_lens, extend_seq_lens)
POSITION_CASES = [
    ([0], [1]),
    ([0], [7]),
    ([5], [3]),
    ([0, 0, 0], [4, 1, 6]),
    ([0, 12, 3, 0], [2, 5, 1, 9]),
    ([128, 0], [64, 1314]),
    ([7, 0, 2], [0, 3, 4]),
]


def _old_write_cache_indices(
    out_cache_loc,
    req_pool_indices_cpu,
    prefix_lens_cpu,
    seq_lens_cpu,
    extend_lens_cpu,
    prefix_tensors,
    req_to_token_pool,
):
    """The per-request torch fallback this change replaces."""
    pt = 0
    for i in range(req_pool_indices_cpu.shape[0]):
        req_idx = req_pool_indices_cpu[i].item()
        prefix_len = prefix_lens_cpu[i].item()
        seq_len = seq_lens_cpu[i].item()
        extend_len = extend_lens_cpu[i].item()
        req_to_token_pool.write((req_idx, slice(0, prefix_len)), prefix_tensors[i])
        req_to_token_pool.write(
            (req_idx, slice(prefix_len, seq_len)),
            out_cache_loc[pt : pt + extend_len],
        )
        pt += extend_len


class TestComputePositionHost(unittest.TestCase):
    def _check(self, prefix, seq, dtype=torch.int32):
        prefix_t = torch.tensor(prefix, dtype=dtype)
        seq_t = torch.tensor(seq, dtype=dtype)
        ref_pos, ref_start = compute_position_torch(prefix_t, seq_t)
        pos, start = compute_position_host(
            prefix, seq, device="cpu", start_loc_dtype=seq_t.dtype
        )
        self.assertEqual(pos.dtype, ref_pos.dtype)
        self.assertEqual(start.dtype, ref_start.dtype)
        self.assertTrue(torch.equal(pos, ref_pos), (prefix, seq, pos, ref_pos))
        self.assertTrue(torch.equal(start, ref_start), (prefix, seq))

    def test_matches_torch_reference(self):
        for prefix, seq in POSITION_CASES:
            with self.subTest(prefix=prefix, seq=seq):
                self._check(prefix, seq)

    def test_int64_lengths(self):
        self._check([3, 0], [2, 4], dtype=torch.int64)

    def test_dispatch_uses_host_mirrors(self):
        prefix, seq = [0, 12, 3], [2, 5, 1]
        prefix_t = torch.tensor(prefix, dtype=torch.int32)
        seq_t = torch.tensor(seq, dtype=torch.int32)
        ref_pos, ref_start = compute_position_torch(prefix_t, seq_t)
        with mock.patch(
            "sglang.srt.model_executor.forward_batch_info.compute_position_torch",
            side_effect=AssertionError("host mirrors should be used"),
        ):
            pos, start = compute_position(
                "torch_native", prefix_t, seq_t, sum(seq), prefix, seq
            )
        self.assertTrue(torch.equal(pos, ref_pos))
        self.assertTrue(torch.equal(start, ref_start))

    def test_dispatch_without_mirrors_falls_back(self):
        prefix_t = torch.tensor([4, 0], dtype=torch.int32)
        seq_t = torch.tensor([3, 2], dtype=torch.int32)
        ref_pos, ref_start = compute_position_torch(prefix_t, seq_t)
        pos, start = compute_position("torch_native", prefix_t, seq_t, 5)
        self.assertTrue(torch.equal(pos, ref_pos))
        self.assertTrue(torch.equal(start, ref_start))


class TestWriteCacheIndicesTorch(unittest.TestCase):
    SIZE = 8
    WIDTH = 64

    def _pool(self):
        pool = ReqToTokenPool(
            size=self.SIZE,
            max_context_len=self.WIDTH,
            device="cpu",
            enable_memory_saver=False,
        )
        # Non-zero background so stray writes are visible.
        pool.req_to_token.fill_(-1)
        return pool

    def _inputs(self, rows, prefix_lens, extend_lens, seed=0):
        g = torch.Generator().manual_seed(seed)
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        prefix_tensors = [
            (
                torch.randint(0, 10_000, (p,), generator=g, dtype=torch.int64)
                if p > 0
                else allocation._EMPTY_PREFIX
            )
            for p in prefix_lens
        ]
        out_cache_loc = torch.randint(
            0, 10_000, (sum(extend_lens),), generator=g, dtype=torch.int64
        )
        as_t = lambda x: torch.tensor(x, dtype=torch.int64)
        return dict(
            out_cache_loc=out_cache_loc,
            req_pool_indices_cpu=as_t(rows),
            prefix_lens_cpu=as_t(prefix_lens),
            seq_lens_cpu=as_t(seq_lens),
            extend_lens_cpu=as_t(extend_lens),
            prefix_tensors=prefix_tensors,
        )

    def _check(self, rows, prefix_lens, extend_lens):
        inp = self._inputs(rows, prefix_lens, extend_lens)
        ref_pool, new_pool, api_pool = self._pool(), self._pool(), self._pool()
        _old_write_cache_indices(**inp, req_to_token_pool=ref_pool)

        write_cache_indices_torch(
            inp["out_cache_loc"],
            inp["req_pool_indices_cpu"],
            inp["prefix_lens_cpu"],
            inp["extend_lens_cpu"],
            inp["prefix_tensors"],
            new_pool.req_to_token,
        )
        self.assertTrue(torch.equal(new_pool.req_to_token, ref_pool.req_to_token))

        # Through the public entry point with a non-Triton prefill backend.
        with mock.patch.object(
            allocation, "attention_backends", return_value=("torch_native", None)
        ):
            write_cache_indices(
                inp["out_cache_loc"],
                inp["req_pool_indices_cpu"],
                inp["req_pool_indices_cpu"],
                inp["prefix_lens_cpu"],
                inp["prefix_lens_cpu"],
                inp["seq_lens_cpu"],
                inp["seq_lens_cpu"],
                inp["extend_lens_cpu"],
                inp["extend_lens_cpu"],
                inp["prefix_tensors"],
                api_pool,
            )
        self.assertTrue(torch.equal(api_pool.req_to_token, ref_pool.req_to_token))

    def test_single_request_no_prefix(self):
        self._check(rows=[1], prefix_lens=[0], extend_lens=[13])

    def test_single_request_with_prefix(self):
        self._check(rows=[3], prefix_lens=[20], extend_lens=[7])

    def test_multiple_requests_mixed_prefix(self):
        self._check(
            rows=[5, 1, 8, 2],
            prefix_lens=[0, 16, 3, 0],
            extend_lens=[9, 4, 1, 30],
        )

    def test_full_row(self):
        self._check(rows=[2, 4], prefix_lens=[40, 0], extend_lens=[24, 64])

    def test_zero_extend_request(self):
        self._check(rows=[1, 2], prefix_lens=[5, 0], extend_lens=[0, 3])

    @unittest.skipUnless(_ACCEL_DEVICE is not None, "needs a non-CPU device")
    def test_host_table_with_device_locs(self):
        # A platform may keep req_to_token on the host while the allocator
        # hands out device-resident slots.
        rows, prefix_lens, extend_lens = [5, 1], [16, 0], [4, 9]
        inp = self._inputs(rows, prefix_lens, extend_lens)
        ref_pool, new_pool = self._pool(), self._pool()
        _old_write_cache_indices(**inp, req_to_token_pool=ref_pool)

        write_cache_indices_torch(
            inp["out_cache_loc"].to(_ACCEL_DEVICE),
            inp["req_pool_indices_cpu"],
            inp["prefix_lens_cpu"],
            inp["extend_lens_cpu"],
            [t.to(_ACCEL_DEVICE) for t in inp["prefix_tensors"]],
            new_pool.req_to_token,
        )
        self.assertTrue(torch.equal(new_pool.req_to_token, ref_pool.req_to_token))


if __name__ == "__main__":
    unittest.main()
