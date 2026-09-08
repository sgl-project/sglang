import unittest

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    get_compress_state_ring_size,
    get_compress_state_write_pad,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCompressStateWritePad(CustomTestCase):
    """The pad bounds how many speculative draft tokens a compress-state ring can serve.

    Mirrors `mtp_pad` in `c_plan.cuh`; `DSV4PoolConfigurator` rejects a larger draft
    count at startup.
    """

    def test_pad_is_zero_without_speculation(self):
        """A non-speculative ring is exactly one window wide: nothing rolls back."""
        for compress_ratio in (2, 4, 128):
            ring_size = get_compress_state_ring_size(compress_ratio, False)
            with self.subTest(cr=compress_ratio, ring=ring_size):
                self.assertEqual(
                    get_compress_state_write_pad(compress_ratio, ring_size), 0
                )

    def test_pad_matches_speculative_ring_capacity(self):
        """`ring_size - window_size + 2`, with window = 2*cr for the overlapping c4."""
        for compress_ratio, expected in ((4, 10), (128, 130)):
            ring_size = get_compress_state_ring_size(compress_ratio, True)
            with self.subTest(cr=compress_ratio, ring=ring_size):
                self.assertEqual(
                    get_compress_state_write_pad(compress_ratio, ring_size), expected
                )

    def test_ratio_two_ring_covers_the_draft_window(self):
        """One pair without speculation; with it, the smallest power-of-two ring
        wider than the draft window, so a regenerated position overwrites its
        rejected draft before the position after it reads the slot before it."""
        self.assertEqual(get_compress_state_ring_size(2, False), 2)
        for num_draft_tokens in range(1, 17):
            ring_size = get_compress_state_ring_size(2, True, num_draft_tokens)
            with self.subTest(num_draft_tokens=num_draft_tokens):
                self.assertEqual(ring_size & (ring_size - 1), 0)
                self.assertGreaterEqual(ring_size, 2 + num_draft_tokens)
                self.assertLess(ring_size // 2, 2 + num_draft_tokens)
                self.assertGreaterEqual(
                    get_compress_state_write_pad(2, ring_size), num_draft_tokens
                )

    def test_pad_is_zero_for_rings_below_one_window(self):
        """Online c128 collapses the ring to 1; the pad must clamp instead of going
        negative."""
        self.assertEqual(get_compress_state_write_pad(128, 1), 0)


if __name__ == "__main__":
    unittest.main()
