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
"""RDNA3.5 (gfx115x) extend-attention tiles at head_dim 256.

The prefix loop is fastest at BLOCK_M 64, the causal extend loop at 128; the switch is at a
cached prefix of a quarter of the new tokens, both always with 8 warps.

    python -m pytest test/registered/unit/layers/attention/test_gfx115_extend_tiles.py -v
"""

import unittest

from sglang.kernels.ops.attention.extend_attention import _gfx115_head_dim_256_tiles
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGfx115ExtendTiles(CustomTestCase):
    def test_no_prefix_takes_tall_tile(self):
        self.assertEqual(_gfx115_head_dim_256_tiles(0, 8192), (128, 8))
        self.assertEqual(_gfx115_head_dim_256_tiles(0, 1), (128, 8))

    def test_small_prefix_takes_tall_tile(self):
        self.assertEqual(_gfx115_head_dim_256_tiles(1024, 8192), (128, 8))  # 1/8
        self.assertEqual(
            _gfx115_head_dim_256_tiles(1023, 4096), (128, 8)
        )  # just under 1/4

    def test_quarter_prefix_takes_short_tile(self):
        # the crossover itself goes to BLOCK_M 64
        self.assertEqual(_gfx115_head_dim_256_tiles(1024, 4096), (64, 8))

    def test_large_prefix_takes_short_tile(self):
        self.assertEqual(_gfx115_head_dim_256_tiles(1024, 2048), (64, 8))  # 1/2
        self.assertEqual(_gfx115_head_dim_256_tiles(24576, 8192), (64, 8))  # 3x


if __name__ == "__main__":
    unittest.main()
