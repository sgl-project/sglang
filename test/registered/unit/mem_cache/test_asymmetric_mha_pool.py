"""Device-side tests for MHATokenToKVPool with asymmetric KV (head_dim != v_head_dim).

Covers the wiring the kernel-level tests cannot see: that the pool derives
``v_row_dim`` from ``v_head_dim`` and threads it into the fused store_cache kernel.
A mis-wired ``v_row_dim`` still writes the right bytes into the right K slots, so
the untouched-slot assertions are what pin the V width and stride down.

Skipped on CPU -- the fused path is CUDA/HIP only.

    python -m pytest test/registered/unit/mem_cache/test_asymmetric_mha_pool.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.kvcache.kvcache import can_use_store_cache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

DTYPE = torch.bfloat16
HEAD_NUM = 2
POOL_SIZE = 63  # buffers get POOL_SIZE + page_size rows
NUM_WRITES = 16

# (head_dim, v_head_dim). Both orderings, since nothing may assume K is wider.
# The last pair is wide enough for the split heuristic to pick num_split=2.
ASYM_DIM_PAIRS = [(192, 128), (128, 192), (512, 256)]


def _build_pool(head_dim: int, v_head_dim: int) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=POOL_SIZE,
        page_size=1,
        dtype=DTYPE,
        head_num=HEAD_NUM,
        head_dim=head_dim,
        v_head_dim=v_head_dim,
        layer_num=1,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


@unittest.skipUnless(_HAS_CUDA, "fused store_cache path requires CUDA")
class TestAsymmetricMHAPoolRowDims(unittest.TestCase):
    def test_v_row_dim_tracks_v_head_dim(self):
        for head_dim, v_head_dim in ASYM_DIM_PAIRS:
            with self.subTest(head_dim=head_dim, v_head_dim=v_head_dim):
                pool = _build_pool(head_dim, v_head_dim)
                self.assertEqual(pool.row_dim, HEAD_NUM * head_dim)
                self.assertEqual(pool.v_row_dim, HEAD_NUM * v_head_dim)

    def test_v_row_dim_defaults_to_row_dim_when_symmetric(self):
        pool = _build_pool(128, 128)
        self.assertEqual(pool.v_row_dim, pool.row_dim)

    def test_swa_dims_override_row_dims(self):
        # A hybrid sliding-window model builds a second pool through the swa_*
        # parameters, which override head_num/head_dim/v_head_dim wholesale. Both
        # of MiMoV2's pools are asymmetric, so v_row_dim has to follow
        # swa_v_head_dim rather than the full pool's v_head_dim.
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=1,
            dtype=DTYPE,
            head_num=HEAD_NUM,
            head_dim=512,
            v_head_dim=256,
            swa_head_num=1,
            swa_head_dim=192,
            swa_v_head_dim=128,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        self.assertEqual(pool.row_dim, 1 * 192)
        self.assertEqual(pool.v_row_dim, 1 * 128)

    def test_swa_v_head_dim_falls_back_to_v_head_dim(self):
        # swa_v_head_dim omitted: head_dim comes from swa_head_dim but v_head_dim
        # does not, so the two are read from different sources. Pinned because a
        # pool built this way is asymmetric in a way neither config states.
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=1,
            dtype=DTYPE,
            head_num=HEAD_NUM,
            head_dim=512,
            v_head_dim=256,
            swa_head_num=1,
            swa_head_dim=192,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        self.assertEqual(pool.row_dim, 1 * 192)
        self.assertEqual(pool.v_row_dim, 1 * 256)


@unittest.skipUnless(_HAS_CUDA, "fused store_cache path requires CUDA")
class TestAsymmetricMHAPoolSetKVBuffer(unittest.TestCase):
    """set_kv_buffer round-trip through the fused kernel, per dim pair."""

    def _run_roundtrip(self, head_dim: int, v_head_dim: int):
        pool = _build_pool(head_dim, v_head_dim)
        k_buf, v_buf = pool.k_buffer[0], pool.v_buffer[0]
        self.assertEqual(tuple(k_buf.shape[1:]), (HEAD_NUM, head_dim))
        self.assertEqual(tuple(v_buf.shape[1:]), (HEAD_NUM, v_head_dim))

        itemsize = pool.store_dtype.itemsize
        self.assertTrue(
            can_use_store_cache(pool.row_dim * itemsize, pool.v_row_dim * itemsize),
            "fused store_cache unavailable; the naive fallback is also correct, so "
            "this test would pass without covering anything",
        )

        # Seed every slot so an over-wide V write shows up on a slot never targeted.
        k_buf.copy_(torch.randn_like(k_buf))
        v_buf.copy_(torch.randn_like(v_buf))
        k_before, v_before = k_buf.clone(), v_buf.clone()

        # Slot 0 is the reserved padding slot store_cache skips; target [1, num_slots).
        num_slots = k_buf.shape[0]
        loc = torch.randperm(num_slots - 1, device="cuda")[:NUM_WRITES] + 1
        cache_k = torch.randn(
            (NUM_WRITES, HEAD_NUM, head_dim), dtype=DTYPE, device="cuda"
        )
        cache_v = torch.randn(
            (NUM_WRITES, HEAD_NUM, v_head_dim), dtype=DTYPE, device="cuda"
        )

        pool.set_kv_buffer(SimpleNamespace(layer_id=0), loc, cache_k, cache_v)

        self.assertTrue(torch.equal(k_buf[loc], cache_k), "K target slots")
        self.assertTrue(torch.equal(v_buf[loc], cache_v), "V target slots")

        untouched = torch.ones(num_slots, dtype=torch.bool, device="cuda")
        untouched[loc] = False
        self.assertTrue(
            torch.equal(k_buf[untouched], k_before[untouched]),
            "K bled outside its target slots",
        )
        self.assertTrue(
            torch.equal(v_buf[untouched], v_before[untouched]),
            "V bled outside its target slots (wrong row width or stride)",
        )

    def test_asymmetric_roundtrip(self):
        for head_dim, v_head_dim in ASYM_DIM_PAIRS:
            with self.subTest(head_dim=head_dim, v_head_dim=v_head_dim):
                self._run_roundtrip(head_dim, v_head_dim)

    def test_symmetric_roundtrip_unchanged(self):
        self._run_roundtrip(128, 128)


@unittest.skipUnless(_HAS_CUDA, "prefix-valid tiled kernel requires CUDA")
class TestAsymmetricPrefixValidGuard(unittest.TestCase):
    """set_kv_buffer_prefix_valid's tiled kernel takes one row width for both
    tensors, so it must refuse asymmetric KV rather than truncate V."""

    def _call_prefix_valid(self, pool, head_dim, v_head_dim):
        rows = 2
        loc_2d = torch.tensor([[1, 2]], dtype=torch.int64, device="cuda")
        commit_lens = torch.tensor([rows], dtype=torch.int32, device="cuda")
        cache_k = torch.randn((rows, HEAD_NUM, head_dim), dtype=DTYPE, device="cuda")
        cache_v = torch.randn((rows, HEAD_NUM, v_head_dim), dtype=DTYPE, device="cuda")
        pool.set_kv_buffer_prefix_valid(
            SimpleNamespace(layer_id=0, k_scale=None, v_scale=None),
            loc_2d,
            commit_lens,
            cache_k,
            cache_v,
        )

    def test_rejects_asymmetric(self):
        for head_dim, v_head_dim in ASYM_DIM_PAIRS:
            with self.subTest(head_dim=head_dim, v_head_dim=v_head_dim):
                pool = _build_pool(head_dim, v_head_dim)
                with self.assertRaises(NotImplementedError):
                    self._call_prefix_valid(pool, head_dim, v_head_dim)

    def test_accepts_symmetric(self):
        # The guard must not tighten the equal-width path it already served.
        pool = _build_pool(128, 128)
        self._call_prefix_valid(pool, 128, 128)
        expected = torch.arange(1, 3, device="cuda")
        self.assertTrue(torch.any(pool.k_buffer[0][expected] != 0))
        self.assertTrue(torch.any(pool.v_buffer[0][expected] != 0))


if __name__ == "__main__":
    unittest.main()
