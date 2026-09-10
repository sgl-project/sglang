import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 2
NUM_SLOTS = 3


def _pool(temporal: torch.Tensor, num_conv: int = 2) -> MambaPool:
    """A MambaPool stub carrying only what the transfer accessors read."""
    pool = object.__new__(MambaPool)
    pool.num_mamba_layers = NUM_LAYERS
    pool.conv_slice_axis = 0
    pool.mamba_cache = MambaPool.State(
        conv=[torch.zeros(NUM_LAYERS, NUM_SLOTS, 4, 5) for _ in range(num_conv)],
        temporal=temporal,
    )
    pool.debug_memory_pool = False
    pool.replayssm_write_pos = None
    return pool


class TestMambaStateTransferBuffers(unittest.TestCase):
    def test_slot_lifecycle_accepts_int32_indices(self):
        pool = _pool(torch.ones(NUM_LAYERS, NUM_SLOTS, 6, 7, 8), num_conv=1)
        pool.mamba_cache.conv[0].fill_(1)
        indices = torch.tensor([0, 2], dtype=torch.int32)

        pool.clear_slots(indices)
        torch.testing.assert_close(
            pool.mamba_cache.conv[0][:, indices.long()],
            torch.zeros(NUM_LAYERS, 2, 4, 5),
        )
        torch.testing.assert_close(
            pool.mamba_cache.temporal[:, indices.long()],
            torch.zeros(NUM_LAYERS, 2, 6, 7, 8),
        )

        pool.copy_from(
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            pool.mamba_cache.conv[0][:, 0], pool.mamba_cache.conv[0][:, 1]
        )
        torch.testing.assert_close(
            pool.mamba_cache.temporal[:, 0], pool.mamba_cache.temporal[:, 1]
        )

    def test_load_cpu_copy_skips_empty_temporal_state(self):
        pool = _pool(torch.empty(NUM_LAYERS, NUM_SLOTS, 0, 0, 0), num_conv=1)
        indices = torch.tensor([0, 2], dtype=torch.int32)
        conv_cpu = [torch.ones(NUM_LAYERS, 2, 4, 5)]
        temporal_cpu = torch.empty(NUM_LAYERS, 2, 0, 0, 0)

        pool.load_cpu_copy((conv_cpu, temporal_cpu), indices)

        torch.testing.assert_close(
            pool.mamba_cache.conv[0][:, indices.long()], conv_cpu[0]
        )

    def test_copy_rejects_duplicate_destinations_in_debug_mode(self):
        pool = _pool(torch.zeros(NUM_LAYERS, NUM_SLOTS, 6, 7, 8), num_conv=1)
        with (
            envs.SGLANG_DEBUG_MEMORY_POOL.override(True),
            self.assertRaisesRegex(AssertionError, "unique destination slots"),
        ):
            pool.copy_from(torch.tensor([0, 1]), torch.tensor([2, 2]))

    def test_conv_only_state_advertises_no_empty_buffer(self):
        """A ShortConv layer declares a degenerate temporal shape, so the pool
        allocates an empty tensor for it. The RDMA engine rejects a zero-length
        region and fails the batch registration that carries the real buffers,
        so an empty buffer must never be advertised."""
        pool = _pool(torch.zeros(NUM_LAYERS, NUM_SLOTS, 0, 0, 0))

        _, lens, item_lens = pool.get_contiguous_buf_infos()

        self.assertNotIn(0, lens)
        self.assertNotIn(0, item_lens)
        self.assertEqual(len(lens), 2 * NUM_LAYERS)

    def test_temporal_state_is_still_advertised(self):
        pool = _pool(torch.zeros(NUM_LAYERS, NUM_SLOTS, 6, 7, 8))

        _, lens, _ = pool.get_contiguous_buf_infos()

        self.assertNotIn(0, lens)
        self.assertEqual(len(lens), 3 * NUM_LAYERS)

    def test_dims_stay_aligned_with_buffers(self):
        """The per-tensor lists are parallel-indexed, so dropping a buffer has to
        drop its dim too."""
        pool = _pool(torch.zeros(NUM_LAYERS, NUM_SLOTS, 0, 0, 0))

        _, lens, _ = pool.get_contiguous_buf_infos()

        self.assertEqual(len(pool.get_state_dim_per_tensor()), len(lens))


if __name__ == "__main__":
    unittest.main()
