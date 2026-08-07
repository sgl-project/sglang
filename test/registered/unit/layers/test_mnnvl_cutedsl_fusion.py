import unittest

from sglang.srt.layers import mnnvl_cutedsl_fusion as cute_dsl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, stage="base-a")

# FlashInfer's own tuned GB300 presets, which the derivation must reproduce
# exactly at the hidden size they were measured at.
_GB300_H8192_CONSUMER_THREADS = 512
_GB300_H8192_VECTORS_PER_THREAD = 2
_GB300_H8192_TP8_ALL_REDUCE_REDUCTION_WARPS = 2
_GB300_H8192_TP8_FINALIZE_REDUCTION_WARPS = 1

_WARP_SIZE = 32
_VEC_BF16 = 8


class TestHTShardDerivation(unittest.TestCase):
    """The HT persistent kernel rejects a tuning whose shard split does not
    divide the token, so these values cannot be guessed at runtime; they are
    derived from the kernel's divisibility rules in ``_ht_shard_split`` /
    ``_ht_reduction_warps``."""

    def test_reproduces_flashinfer_h8192_presets(self):
        """The derivation must be an extension of FlashInfer's tuning, not a
        replacement: at H=8192 it has to land on the shipped GB300 values."""
        self.assertEqual(
            cute_dsl._ht_shard_split(8192),
            (_GB300_H8192_CONSUMER_THREADS, _GB300_H8192_VECTORS_PER_THREAD),
        )
        self.assertEqual(
            cute_dsl._ht_reduction_warps(8192, 8, preferred=2),
            _GB300_H8192_TP8_ALL_REDUCE_REDUCTION_WARPS,
        )
        self.assertEqual(
            cute_dsl._ht_reduction_warps(8192, 8, preferred=1),
            _GB300_H8192_TP8_FINALIZE_REDUCTION_WARPS,
        )

    def test_split_satisfies_kernel_constraints(self):
        """Every split the derivation returns must satisfy all four rules the
        HT device kernel enforces, or workspace construction raises."""
        for hidden_size in (2048, 4096, 6144, 7168, 8192, 12288, 16384):
            split = cute_dsl._ht_shard_split(hidden_size)
            if split is None:
                continue
            consumer_threads, vectors_per_thread = split
            packs = hidden_size // _VEC_BF16
            with self.subTest(hidden_size=hidden_size):
                self.assertEqual(consumer_threads % _WARP_SIZE, 0)
                self.assertEqual(packs % consumer_threads, 0)
                shard_elements = consumer_threads * _VEC_BF16 * vectors_per_thread
                self.assertEqual(hidden_size % shard_elements, 0)
                # Consumers plus the loader, publisher, and reduction warps
                # must fit one CUDA block.
                self.assertLessEqual(consumer_threads + 3 * _WARP_SIZE, 1024)

    def test_reduction_warps_never_exceed_the_tuned_preference(self):
        """Stepping *up* from FlashInfer's tuned warp count would silently
        retune a kernel we have no measurements for."""
        for hidden_size in (4096, 6144, 8192, 16384):
            for tp_size in (2, 4, 8, 16):
                for preferred in (1, 2):
                    warps = cute_dsl._ht_reduction_warps(
                        hidden_size, tp_size, preferred
                    )
                    if warps is None:
                        continue
                    with self.subTest(h=hidden_size, tp=tp_size, preferred=preferred):
                        self.assertLessEqual(warps, preferred)
                        packs_per_shard = (hidden_size // _VEC_BF16) // tp_size
                        self.assertEqual(packs_per_shard % (warps * _WARP_SIZE), 0)

    def test_unsupported_shapes_fall_back(self):
        """A hidden size the HT kernel cannot shard must return None so the
        caller keeps the mnnvl backend, rather than building a workspace that
        raises mid-startup."""
        # 2880 leaves 360 vectors per token: no warp-multiple consumer count
        # divides it with two or more vectors per thread.
        self.assertIsNone(cute_dsl._ht_shard_split(2880))
        # H=7168 at TP8 gives a 112-vector reduction shard, which no warp
        # count divides.
        self.assertIsNone(cute_dsl._ht_reduction_warps(7168, 8, preferred=2))


if __name__ == "__main__":
    unittest.main()
