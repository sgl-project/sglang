# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""KV tile size for the Lean decode kernel must fit the device's shared memory.

Lean stages only the K tile in LDS: `_LEAN_NUM_STAGES` copies of a
`[BLOCK_DMODEL, BLOCK_N]` bf16 tile. At head dim 128 the HIP default used to be
`BLOCK_N = 128`, i.e. 2 * 128 * 128 * 2 = 65536 B on top of the compiler's own
scratch, which does not fit the 64 KiB LDS of gfx1151 (RDNA3.5) and made the
launch raise OutOfResources while capturing the decode graph. The tile is now
chosen from the device budget instead of being hard-coded, so a device that
cannot hold the largest tile still gets the largest one it can hold.

The device budget is passed in rather than read from the GPU so this stays a
CPU test.

    python -m pytest test/registered/unit/layers/attention/test_lean_decode_block_n.py -v
"""

import unittest

from sglang.kernels.ops.attention import decode_attention as da
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# Shared memory per workgroup, in bytes, of the parts this budget has to cover.
GFS1151_LDS = 64 * 1024  # gfx1151 / RDNA3.5: 66560 B tile does not fit
MI300X_LDS = 160 * 1024  # CDNA3: holds the largest tile with room to spare

_STANDARD_HEAD_DIM = 128
_MLA_HEAD_DIMS = (288, 576)


class TestLeanDecodeBlockN(unittest.TestCase):
    def setUp(self):
        self._is_hip = da._is_hip
        self._lean_lds_limit = da._lean_lds_limit

    def tearDown(self):
        da._is_hip = self._is_hip
        da._lean_lds_limit = self._lean_lds_limit

    def _block_n(self, lds_bytes, head_dim, is_hip=True):
        da._is_hip = is_hip
        da._lean_lds_limit = lambda: lds_bytes
        return da._lean_decode_block_n(head_dim)

    def test_head_dim_128_fits_a_64_kib_device(self):
        # 128 would ask for 65536 B; the slack leaves 64512, so the tile halves.
        self.assertEqual(self._block_n(GFS1151_LDS, _STANDARD_HEAD_DIM), 64)

    def test_head_dim_128_keeps_the_large_tile_when_it_fits(self):
        self.assertEqual(self._block_n(MI300X_LDS, _STANDARD_HEAD_DIM), 128)

    def test_mla_head_dims_keep_their_small_tile(self):
        for head_dim in _MLA_HEAD_DIMS:
            for lds in (GFS1151_LDS, MI300X_LDS):
                self.assertEqual(self._block_n(lds, head_dim), 16)

    def test_cuda_is_unchanged(self):
        self.assertEqual(self._block_n(GFS1151_LDS, _STANDARD_HEAD_DIM, is_hip=False), 64)

    def test_every_choice_fits_the_budget(self):
        for lds in (48 * 1024, GFS1151_LDS, 96 * 1024, MI300X_LDS):
            for head_dim in (64, 128, 192, 256):
                block_n = self._block_n(lds, head_dim)
                self.assertIn(block_n, (16, 32, 64, 128))
                self.assertLessEqual(
                    da._LEAN_NUM_STAGES * block_n * head_dim * 2,
                    lds - da._LEAN_LDS_SLACK,
                    f"head_dim={head_dim} block_n={block_n} lds={lds}",
                )


if __name__ == "__main__":
    unittest.main()
