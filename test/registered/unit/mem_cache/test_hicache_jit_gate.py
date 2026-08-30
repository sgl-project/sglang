import unittest

from sglang.kernels.jit.utils import is_hip_runtime
from sglang.kernels.ops.kvcache.hicache import (
    GROUP_BYTES,
    WARP_THREADS,
    _default_unroll,
    _tiles_across_lanes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# MLA's kv_lora (512) + rope (64) row, one byte per element under fp8_e4m3.
MLA_FP8_ROW_BYTES = 576

# The two tables GROUP_BYTES resolves to. Named here rather than read off the
# runtime so a runner of either kind covers both.
ROCM_GROUPS = (128, 64, 32, 16)
OTHER_GROUPS = (128,)


class TestHiCacheJitGate(CustomTestCase):
    """The gate deciding whether a row size can use the HiCache read JIT.

    A size the gate rejects falls back to the non-JIT path silently, so a wrong
    gate costs throughput without failing anything.
    """

    def test_group_table_matches_the_runtime(self):
        self.assertEqual(GROUP_BYTES, ROCM_GROUPS if is_hip_runtime() else OTHER_GROUPS)

    def test_mla_fp8_row_reaches_the_jit_only_with_the_narrow_rounds(self):
        # 128 does not divide 576, so before the narrow rounds the read JIT was
        # silently off on every DeepSeek/GLM-shaped model served with
        # --kv-cache-dtype fp8_e4m3.
        unroll = _default_unroll(MLA_FP8_ROW_BYTES)
        self.assertTrue(
            _tiles_across_lanes(MLA_FP8_ROW_BYTES, unroll, ROCM_GROUPS),
            "576 B row should tile as 64 B x 9",
        )
        self.assertFalse(
            _tiles_across_lanes(MLA_FP8_ROW_BYTES, unroll, OTHER_GROUPS),
            "platforms keeping the 128 B round must still reject 576 B",
        )

    def test_sizes_128_divides_are_unaffected_by_the_narrow_rounds(self):
        # The narrow rounds must not change the shape of anything that already
        # worked: 128 stays first choice on every platform.
        for element_size in (128, 256, 512, 1024, 2048):
            for unroll in (1, 2, 4):
                with self.subTest(element_size=element_size, unroll=unroll):
                    self.assertTrue(
                        _tiles_across_lanes(element_size, unroll, OTHER_GROUPS)
                    )
                    self.assertTrue(
                        _tiles_across_lanes(element_size, unroll, ROCM_GROUPS)
                    )
                    self.assertIn(128 // (WARP_THREADS // unroll), (4, 8, 16))

    def test_every_accepted_round_yields_a_package_the_kernel_instantiates(self):
        # Mirrors get_mem_package() in kvcacheio/hicache.cuh, which only
        # instantiates uint1/uint2/uint4.
        for unroll in (1, 2, 4):
            num_threads = WARP_THREADS // unroll
            for group in ROCM_GROUPS:
                if group % num_threads or not _tiles_across_lanes(
                    group, unroll, (group,)
                ):
                    continue
                with self.subTest(unroll=unroll, group=group):
                    self.assertIn(group // num_threads, (4, 8, 16))


if __name__ == "__main__":
    unittest.main()
