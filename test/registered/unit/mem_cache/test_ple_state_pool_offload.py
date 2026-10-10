"""Unit tests for ple_state_pool.py clearing while KV-cache storage is offloaded."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MambaPool,
)
from sglang.srt.mem_cache.ple_state_pool import NGramPool, ShortConvPool
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 2
NUM_SLOTS = 4
STATE_SHAPE = (3, 2)
CONTEXT_LEN = 2
EOS_TOKEN_ID = 17


def _short_conv_pool() -> ShortConvPool:
    return ShortConvPool(
        size=NUM_SLOTS,
        state_shape=STATE_SHAPE,
        layer_ids=list(range(NUM_LAYERS)),
        dtype=torch.float32,
        device="cpu",
    )


def _ngram_pool() -> NGramPool:
    return NGramPool(
        size=NUM_SLOTS,
        context_len=CONTEXT_LEN,
        eos_token_id=EOS_TOKEN_ID,
        device="cpu",
    )


class _OffloadedStorage:
    """Stands in for a tensor whose backing memory has been unmapped.

    torch_memory_saver's pause() calls cuMemUnmap and cuMemRelease, so the
    tensor object outlives its storage: attribute lookup still succeeds on a
    real tensor and the kernel then faults on the device. A CPU test cannot
    reproduce the fault, so it asserts the access itself never happens.
    """

    def __getattr__(self, name):
        raise AssertionError(f"touched offloaded storage: .{name}")

    def __getitem__(self, key):
        raise AssertionError("read from offloaded storage")

    def __setitem__(self, key, value):
        raise AssertionError("wrote to offloaded storage")


def _hybrid_pool(short_conv: ShortConvPool, ngram: NGramPool):
    """The parts of HybridReqToTokenPool.clear() that a CPU test can drive."""

    class _Allocator:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    pool = object.__new__(HybridReqToTokenPool)
    pool._alloc_size = NUM_SLOTS
    pool.free_slots = []
    pool.req_generation = torch.ones(NUM_SLOTS, dtype=torch.int64)
    pool._aux_cache = None
    pool.mamba_allocator = _Allocator()
    pool.short_conv_pool = short_conv
    pool.ngram_pool = ngram
    pool.mamba_ckpt_pool = None
    pool.req_index_to_mamba_index_mapping = torch.ones(NUM_SLOTS, dtype=torch.int32)
    pool.enable_mamba_extra_buffer = False
    return pool


def _mamba_pool(siblings) -> MambaPool:
    """A MambaPool stub carrying only what the slot hooks read."""
    pool = object.__new__(MambaPool)
    pool._slot_siblings = list(siblings)
    pool._conv_fuse_ok = False
    pool.replayssm_write_pos = None
    pool.mamba_cache = MambaPool.State(
        conv=[torch.zeros(NUM_LAYERS, NUM_SLOTS + 1, 2)],
        temporal=torch.zeros(NUM_LAYERS, NUM_SLOTS + 1, 2),
    )
    return pool


class TestPleClearIsOffloadSafe(CustomTestCase):
    """`release_memory_occupation` pauses the KV-cache tag and then calls
    `flush_cache()`, which reaches these pools. Their storage is allocated
    under that same tag, so clearing must not touch it."""

    def test_short_conv_clear_does_not_touch_storage(self):
        pool = _short_conv_pool()
        pool.conv_state = _OffloadedStorage()

        pool.clear()

    def test_ngram_clear_does_not_touch_storage(self):
        pool = _ngram_pool()
        pool.context = _OffloadedStorage()

        pool.clear()

    def test_hybrid_clear_still_resets_bookkeeping(self):
        """The owning pool keeps clearing its own bookkeeping; only the PLE
        storage writes go away, so a flush during an offload still leaves the
        allocator and the request mapping reset."""
        short_conv, ngram = _short_conv_pool(), _ngram_pool()
        short_conv.conv_state = _OffloadedStorage()
        ngram.context = _OffloadedStorage()
        pool = _hybrid_pool(short_conv, ngram)

        pool.clear()

        self.assertEqual(pool.free_slots, list(range(1, NUM_SLOTS)))
        self.assertTrue(pool.mamba_allocator.cleared)
        self.assertTrue(bool((pool.req_generation == 0).all()))
        self.assertTrue(bool((pool.req_index_to_mamba_index_mapping == 0).all()))

    def test_disabled_pools_clear_without_storage(self):
        """A model without PLE layers builds both pools disabled; clearing one
        must stay a no-op rather than start asserting."""
        ShortConvPool(
            size=NUM_SLOTS,
            state_shape=None,
            layer_ids=[],
            dtype=torch.float32,
            device="cpu",
        ).clear()
        NGramPool(
            size=NUM_SLOTS,
            context_len=0,
            eos_token_id=EOS_TOKEN_ID,
            device="cpu",
        ).clear()


class TestPleSlotLifecycleStillInitializes(CustomTestCase):
    """What makes dropping the bulk writes safe: a slot handed to a request is
    initialized through the Mamba slot hooks, which is where the deferred
    per-request clear and the cached-prefix copy both land. The backing storage
    is poisoned first so a pass cannot come from constructor values."""

    def test_clear_slots_initializes_only_the_named_slots(self):
        # EOS is load-bearing for n-gram history, so a reset that zeroed the rows
        # would be wrong; keep the fixture's EOS distinguishable from zero.
        self.assertNotEqual(EOS_TOKEN_ID, 0)
        short_conv, ngram = _short_conv_pool(), _ngram_pool()
        short_conv.conv_state.fill_(9.0)
        ngram.context.fill_(123)
        mamba = _mamba_pool([short_conv, ngram])
        fresh = torch.tensor([2])

        mamba.clear_slots(fresh)

        self.assertTrue(bool((short_conv.conv_state[:, 2] == 0).all()))
        self.assertTrue(bool((ngram.context[2] == EOS_TOKEN_ID).all()))
        # an untouched neighbour keeps the poison, so the reset is per-slot
        self.assertTrue(bool((short_conv.conv_state[:, 3] == 9.0).all()))
        self.assertTrue(bool((ngram.context[3] == 123).all()))

    def test_copy_slots_inherits_state_for_a_cached_prefix(self):
        """A prefix hit copies state instead of clearing it; the destination
        must inherit the source rather than be reset."""
        short_conv, ngram = _short_conv_pool(), _ngram_pool()
        short_conv.conv_state[:, 1] = 5.0
        ngram.context[1] = 42
        mamba = _mamba_pool([short_conv, ngram])

        mamba.copy_from(torch.tensor([1]), torch.tensor([3]))

        self.assertTrue(bool((short_conv.conv_state[:, 3] == 5.0).all()))
        self.assertTrue(bool((ngram.context[3] == 42).all()))
        self.assertTrue(bool((short_conv.conv_state[:, 1] == 5.0).all()))


if __name__ == "__main__":
    unittest.main()
