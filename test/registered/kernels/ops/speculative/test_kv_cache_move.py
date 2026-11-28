"""`MHATokenToKVPool.move_kv_cache`: the move must be exact at every tile width,
and the chunked path must not let a later chunk read a slot an earlier one wrote.

A wrong move here is a silently wrong answer rather than a crash, and end-to-end
byte-identity cannot catch it: below the cap the loop never runs, and above it the
batch composition is not reproducible run to run.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# kernel/, not unit/: this needs a real CUDA device, and unit suites are CPU-only.
# The runner size is what the kernel stage uses, not what the test needs -- it wants
# a GPU and nothing else, no H100-class memory and no Hopper kernels. The largest
# pool here is 8192 slots x 8 heads x 512 head_dim x bf16 x (K+V) x 4 layers, about
# 0.5 GiB.
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

LAYERS = 4
HEAD_NUM = 8
DEVICE = "cuda"


def _build_pool(size: int, head_dim: int, dtype=torch.bfloat16) -> MHATokenToKVPool:
    return MHATokenToKVPool(
        size=size,
        page_size=1,
        dtype=dtype,
        head_num=HEAD_NUM,
        head_dim=head_dim,
        layer_num=LAYERS,
        device=DEVICE,
        enable_memory_saver=False,
        enable_alt_stream=False,
        enable_kv_cache_copy=True,
    )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestKVCacheMove(CustomTestCase):
    # head_dim picks the tile width: row bytes = head_num * head_dim * itemsize,
    # and the configuration branches at 4096 and 8192 bytes. At head_num 8 and
    # bfloat16 these give 2048 / 4096 / 8192 B rows, i.e. all three branches.
    HEAD_DIMS = (128, 256, 512)

    def _fill(self, pool, n_slots):
        """Deterministic distinct contents per (layer, slot)."""
        for layer_id in range(LAYERS):
            k = pool.get_key_buffer(layer_id)
            v = pool.get_value_buffer(layer_id)
            base = torch.arange(n_slots, device=DEVICE, dtype=torch.float32)
            k[:n_slots] = (base[:, None, None] + layer_id * 1000).to(k.dtype)
            v[:n_slots] = (base[:, None, None] + layer_id * 1000 + 500).to(v.dtype)

    def _snapshot(self, pool, slots):
        idx = torch.tensor(slots, device=DEVICE, dtype=torch.int64)
        return [
            (
                pool.get_key_buffer(layer_id)[idx].clone(),
                pool.get_value_buffer(layer_id)[idx].clone(),
            )
            for layer_id in range(LAYERS)
        ]

    def _assert_moved(self, pool, tgt, src, expected):
        """Every target slot holds what its source slot held before the move."""
        idx = torch.tensor(tgt, device=DEVICE, dtype=torch.int64)
        for layer_id, (k_want, v_want) in enumerate(expected):
            torch.testing.assert_close(
                pool.get_key_buffer(layer_id)[idx], k_want, atol=0, rtol=0
            )
            torch.testing.assert_close(
                pool.get_value_buffer(layer_id)[idx], v_want, atol=0, rtol=0
            )

    def test_move_is_exact_for_every_tile_width(self):
        """Single launch, every tile width: the move is exact and no chunking runs.

        The cap is derived, not fixed, so the bound is asserted rather than trusted.
        """
        for head_dim in self.HEAD_DIMS:
            with self.subTest(head_dim=head_dim):
                n = 20
                pool = _build_pool(4096, head_dim)
                self.assertLessEqual(
                    n,
                    int(pool._kv_copy_config["num_locs_upper"]),
                    "this case is meant to stay on the single-launch path",
                )
                self._fill(pool, 4096)
                src = list(range(1000, 1000 + n))
                tgt = list(range(10, 10 + n))
                expected = self._snapshot(pool, src)
                pool.move_kv_cache(
                    torch.tensor(tgt, device=DEVICE, dtype=torch.int64),
                    torch.tensor(src, device=DEVICE, dtype=torch.int64),
                )
                self._assert_moved(pool, tgt, src, expected)

    def test_move_is_exact_when_it_has_to_chunk(self):
        """Large move: forces the chunk loop, whatever the configured cap is."""
        for head_dim in self.HEAD_DIMS:
            with self.subTest(head_dim=head_dim):
                pool = _build_pool(8192, head_dim)
                cap = int(pool._kv_copy_config["num_locs_upper"])
                n = cap * 3 + 7  # ragged last chunk on purpose
                self._fill(pool, 8192)
                src = list(range(4000, 4000 + n))
                tgt = list(range(10, 10 + n))
                expected = self._snapshot(pool, src)
                pool.move_kv_cache(
                    torch.tensor(tgt, device=DEVICE, dtype=torch.int64),
                    torch.tensor(src, device=DEVICE, dtype=torch.int64),
                )
                self._assert_moved(pool, tgt, src, expected)

    def test_backward_overlapping_move_across_chunks(self):
        """Heavy aliasing in the benign direction must not corrupt the move.

        Every alias here has the reader at the earlier index; the direction that can
        break a chunk split is covered by the two debug-check cases below.
        """
        for head_dim in self.HEAD_DIMS:
            with self.subTest(head_dim=head_dim):
                pool = _build_pool(8192, head_dim)
                cap = int(pool._kv_copy_config["num_locs_upper"])
                n = cap * 2 + 5
                self._fill(pool, 8192)
                # tgt[i] = 100 + i, src[i] = 108 + i: an 8-slot backward slide, so
                # most sources are also targets of a LATER entry.
                tgt = [100 + i for i in range(n)]
                src = [108 + i for i in range(n)]
                expected = self._snapshot(pool, src)
                pool.move_kv_cache(
                    torch.tensor(tgt, device=DEVICE, dtype=torch.int64),
                    torch.tensor(src, device=DEVICE, dtype=torch.int64),
                )
                self._assert_moved(pool, tgt, src, expected)

    def _clobber_pattern(self, cap, straddle):
        """One entry writes a slot another later reads; ``straddle`` decides whether
        they land in different chunks. Same aliasing either way, only the split differs.
        """
        n = cap * 2
        tgt = [5000 + i for i in range(n)]
        src = [6000 + i for i in range(n)]
        reader = cap + 3 if straddle else 3
        tgt[0] = 7777  # entry 0 writes slot 7777 ...
        src[reader] = 7777  # ... and a later entry reads it
        return tgt, src

    def test_debug_check_fires_on_forward_clobber_across_chunks(self):
        """The order check must reject a cross-chunk read-after-write.

        Every other case here is a pattern it should accept, so without this one it
        could degrade to always-pass unnoticed.
        """
        with envs.SGLANG_DEBUG_MEMORY_POOL.override(True):
            # Built inside the override: the pool snapshots the flag in __init__.
            pool = _build_pool(8192, 128)
            cap = int(pool._kv_copy_config["num_locs_upper"])
            self.assertTrue(pool._check_chunked_move_order, "flag did not take")
            tgt, src = self._clobber_pattern(cap, straddle=True)
            # Match the message: a bare assertRaises would also be satisfied by
            # any unrelated assertion inside the move, and pass for the wrong
            # reason.
            with self.assertRaisesRegex(
                AssertionError, r"read a slot an earlier chunk overwrote"
            ):
                pool.move_kv_cache(
                    torch.tensor(tgt, device=DEVICE, dtype=torch.int64),
                    torch.tensor(src, device=DEVICE, dtype=torch.int64),
                )

    def test_debug_check_accepts_the_same_clobber_inside_one_chunk(self):
        """The same aliasing must NOT be rejected when it cannot cross a chunk.

        Production aliases heavily and legitimately; a check that fired regardless of
        chunk boundaries would make the flag unusable.
        """
        with envs.SGLANG_DEBUG_MEMORY_POOL.override(True):
            pool = _build_pool(8192, 128)
            cap = int(pool._kv_copy_config["num_locs_upper"])
            tgt, src = self._clobber_pattern(cap, straddle=False)
            pool.move_kv_cache(
                torch.tensor(tgt, device=DEVICE, dtype=torch.int64),
                torch.tensor(src, device=DEVICE, dtype=torch.int64),
            )

    def test_empty_move_is_a_noop(self):
        pool = _build_pool(256, 128)
        self._fill(pool, 256)
        before = self._snapshot(pool, list(range(256)))
        empty = torch.zeros(0, device=DEVICE, dtype=torch.int64)
        pool.move_kv_cache(empty, empty)
        self._assert_moved(pool, list(range(256)), None, before)


if __name__ == "__main__":
    unittest.main()
