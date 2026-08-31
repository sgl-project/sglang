import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

LAYERS = 2
PAGE_SIZE = 128


def _pool(interleaved: bool) -> MHATokenToKVPoolMXFP8:
    """A pool stub carrying only what the scale accessor reads."""
    pool = object.__new__(MHATokenToKVPoolMXFP8)
    pool.page_size = PAGE_SIZE
    pool.mxfp8_sf_interleaved = interleaved
    # Interleaved keeps pages on the leading axis; flat keeps slots.
    shape = (4, 3, 32, 4, 2) if interleaved else (4 * PAGE_SIZE, 3, 2)
    pool.k_scale_buffer = [
        torch.zeros(shape, dtype=torch.float8_e8m0fnu) for _ in range(LAYERS)
    ]
    pool.v_scale_buffer = [
        torch.zeros(shape, dtype=torch.float8_e8m0fnu) for _ in range(LAYERS)
    ]
    return pool


class TestMXFP8ScaleTransferBuffers(unittest.TestCase):
    def test_interleaved_item_len_is_one_page_row(self):
        pool = _pool(interleaved=True)

        ptrs, lens, item_lens = pool.get_kv_scale_buf_infos()

        self.assertEqual(len(ptrs), 2 * LAYERS)
        self.assertNotIn(0, lens)
        row = pool.k_scale_buffer[0][0].nbytes
        self.assertEqual(item_lens[0], row)
        self.assertEqual(lens[0] // item_lens[0], pool.k_scale_buffer[0].shape[0])

    def test_flat_item_len_covers_a_whole_page(self):
        """A flat buffer is indexed per slot, so a page's worth is page_size rows."""
        pool = _pool(interleaved=False)

        _, lens, item_lens = pool.get_kv_scale_buf_infos()

        row = pool.k_scale_buffer[0][0].nbytes
        self.assertEqual(item_lens[0], row * PAGE_SIZE)
        self.assertEqual(lens[0] // item_lens[0], 4)


if __name__ == "__main__":
    unittest.main()
