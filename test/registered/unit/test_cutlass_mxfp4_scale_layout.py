"""Load-time scale contract of the SM90 mixed-input MXFP4 grouped GEMM.

Both facts pinned here live in the CUDA collective, not in the Python file:

- The mainloop TMA-loads the scale operand as ``Array<uint8_t, 4>`` indexed by
  (K tile, N), so ``_pack_mxfp4_scales_for_cutlass`` must turn the checkpoint's
  ``[E, N, K/32]`` into ``[E, K/128, N*4]`` with the four groups of a 128-wide K
  tile adjacent.
- Its E2M1 decode never renormalizes, leaving every value at ``value * 2^-126``,
  so the repacked byte must carry the checkpoint exponent plus 126 -- a byte
  ``b`` scales by ``2^(b - 253)``.
"""

import unittest

import torch

from sglang.srt.layers.quantization.mxfp4_cutlass_moe import (
    _pack_mxfp4_scales_for_cutlass,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCutlassMxfp4ScaleLayout(CustomTestCase):
    def test_packs_k_tile_groups_adjacent_and_biases_exponent(self):
        num_experts, n, num_groups = 2, 6, 8
        torch.manual_seed(0)
        scales = torch.randint(0, 129, (num_experts, n, num_groups), dtype=torch.uint8)

        out = _pack_mxfp4_scales_for_cutlass(scales)

        self.assertEqual(tuple(out.shape), (num_experts, num_groups // 4, n * 4))
        for e in range(num_experts):
            for group in range(num_groups):
                for row in range(n):
                    got = int(out[e, group // 4, 4 * row + group % 4])
                    self.assertEqual(got, int(scales[e, row, group]) + 126)

    def test_rejects_scales_the_bf16_exponent_field_cannot_hold(self):
        # 129 + 126 == 255, which is Inf/NaN once the collective widens the byte
        # into a bf16 exponent, poisoning every accumulator the group touches.
        scales = torch.full((1, 4, 4), 129, dtype=torch.uint8)
        with self.assertRaises(NotImplementedError):
            _pack_mxfp4_scales_for_cutlass(scales)

    def test_rejects_k_not_tiled_by_128(self):
        # 6 groups == K 192; the collective reads whole 128-wide K tiles.
        scales = torch.zeros((1, 4, 6), dtype=torch.uint8)
        with self.assertRaises(NotImplementedError):
            _pack_mxfp4_scales_for_cutlass(scales)


if __name__ == "__main__":
    unittest.main(verbosity=2)
