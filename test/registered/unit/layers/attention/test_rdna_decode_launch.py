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
"""RDNA (gfx110x, gfx115x, gfx120x) launch config for the grouped decode-attention stage 1.

Dropping the CDNA hints is a 1.4-2.7x win only where the kernel is memory-bound with
enough work in flight; on MLA, fp8 KV and tiny grids the hints win and must stay, and at
head_dim 256 dropping them is neutral.

    python -m pytest test/registered/unit/layers/attention/test_rdna_decode_launch.py -v
"""

import unittest

import torch

from sglang.kernels.ops.attention.decode_attention import _rdna_grouped_num_warps
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _warps(
    block_dmodel=128,
    block_dpe=0,
    kv_dtype=torch.bfloat16,
    batch=16,
    head_tiles=8,
):
    return _rdna_grouped_num_warps(
        block_dmodel=block_dmodel,
        block_dpe=block_dpe,
        kv_dtype=kv_dtype,
        batch=batch,
        head_tiles=head_tiles,
    )


class TestRdnaGroupedDecodeLaunch(CustomTestCase):
    def test_gqa_16bit_kv_drops_hints(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.assertEqual(_warps(block_dmodel=64, kv_dtype=dtype), 2)
                self.assertEqual(_warps(block_dmodel=128, kv_dtype=dtype), 2)

    def test_head_dim_256_keeps_hints(self):
        self.assertIsNone(_warps(block_dmodel=256))
        self.assertIsNone(_warps(block_dmodel=256, batch=64))

    def test_mla_keeps_hints(self):
        # Lk 576 and 288 are compute-bound, where the occupancy hint wins
        self.assertIsNone(
            _warps(block_dmodel=512, block_dpe=64, head_tiles=1, batch=64)
        )
        self.assertIsNone(
            _warps(block_dmodel=256, block_dpe=32, head_tiles=1, batch=64)
        )

    def test_fp8_kv_keeps_hints(self):
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            with self.subTest(dtype=dtype):
                self.assertIsNone(_warps(kv_dtype=dtype))

    def test_other_kv_dtypes_keep_hints(self):
        for dtype in (torch.float32, torch.uint8):
            with self.subTest(dtype=dtype):
                self.assertIsNone(_warps(kv_dtype=dtype))

    def test_tiny_grid_keeps_hints(self):
        # batch * head_tiles < 8: one long-context request on a 2-KV-head model
        self.assertIsNone(_warps(batch=1, head_tiles=2))
        self.assertIsNone(_warps(batch=3, head_tiles=2))
        self.assertIsNone(_warps(batch=1, head_tiles=4))
        self.assertEqual(_warps(batch=4, head_tiles=2), 2)
        self.assertEqual(_warps(batch=1, head_tiles=8), 2)


if __name__ == "__main__":
    unittest.main()
