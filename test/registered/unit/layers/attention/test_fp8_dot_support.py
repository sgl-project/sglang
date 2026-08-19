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
"""The gfx950 guard on fp8 x fp8 tl.dot reduction widths.

An fp8 x fp8 dot narrower than 128 crashes Triton's AMD backend on gfx950
(``PassManager::run failed`` out of ``ConvertTritonAMDGPUToLLVM``; upstream fix
triton-lang/triton#8278 is not in the 3.4 ROCm wheels). Every Triton attention
kernel that reads a KV pool feeds P @ V a reduction of BLOCK_N, so an fp8 pool
on gfx950 hits that width -- which is what these tests pin, alongside the
neighbouring Q @ K dot staying wide enough to keep the fp8 fast path.

    python -m pytest test/registered/unit/layers/attention/test_fp8_dot_support.py -v
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import extend_attention as ea
from sglang.kernels.ops.attention import fp8_dot_support
from sglang.kernels.ops.attention.decode_attention import _fwd_grouped_kernel_stage1
from sglang.kernels.ops.attention.fp8_dot_support import dot_in_kv_dtype
from sglang.kernels.ops.attention.verify_mla import _verify_mla_prefix_stage1
from sglang.kernels.ops.attention.verify_splitkv import _verify_prefix_stage1
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Qwen3.5's TP-local head, the shape that first exercised an fp8 pool on MI35x.
QWEN3_5_HEAD_DIM = 256


class TestFp8DotSupport(CustomTestCase):
    def test_narrow_fp8_reductions_rejected_on_gfx950(self):
        with patch.object(fp8_dot_support, "_is_gfx95", True):
            for width in (16, 32, 64, 96):
                with self.subTest(width=width):
                    self.assertFalse(
                        dot_in_kv_dtype(torch.float8_e4m3fn, width),
                        "an fp8 dot narrower than the scaled MFMA crashes gfx950",
                    )
            for width in (128, 256, 512):
                with self.subTest(width=width):
                    self.assertTrue(dot_in_kv_dtype(torch.float8_e4m3fn, width))

    def test_only_fp8_pools_are_narrowed_and_only_on_gfx950(self):
        # A bf16 pool never promotes to the scaled MFMA, and no other arch does.
        with patch.object(fp8_dot_support, "_is_gfx95", True):
            self.assertTrue(dot_in_kv_dtype(torch.bfloat16, 64))
            self.assertTrue(dot_in_kv_dtype(torch.float16, 64))
        with patch.object(fp8_dot_support, "_is_gfx95", False):
            self.assertTrue(dot_in_kv_dtype(torch.float8_e4m3fn, 64))
            self.assertTrue(dot_in_kv_dtype(torch.float8_e5m2, 64))

    def test_extend_attention_gfx950_tiles_need_the_guard(self):
        # The tile sizes that crashed: the P @ V reduction is BLOCK_N, which no
        # gfx950 fp8 tile config reaches 128 with, while Q @ K reduces
        # BLOCK_DMODEL and stays on the fp8 matrix core.
        with patch.object(ea, "_is_hip", True), patch.object(ea, "_is_gfx95", True):
            block_dmodel, block_dpe, _, _, block_n, _ = (
                ea._get_block_sizes_for_extend_attention(
                    QWEN3_5_HEAD_DIM, QWEN3_5_HEAD_DIM
                )
            )
        self.assertEqual(block_dpe, 0)
        with patch.object(fp8_dot_support, "_is_gfx95", True):
            self.assertTrue(dot_in_kv_dtype(torch.float8_e4m3fn, block_dmodel))
            self.assertFalse(dot_in_kv_dtype(torch.float8_e4m3fn, block_n))

    def test_pool_reading_kernels_expose_the_per_dot_flags(self):
        # A new dot over a pool tile has to declare its own verdict; defaulting
        # to the cast-down path is what crashes.
        qk_pe_pv = (
            "QK_DOT_IN_KV_DTYPE",
            "QK_PE_DOT_IN_KV_DTYPE",
            "PV_DOT_IN_KV_DTYPE",
        )
        qk_pv = ("QK_DOT_IN_KV_DTYPE", "PV_DOT_IN_KV_DTYPE")
        for name, kernel, flags in (
            ("_fwd_kernel", ea._fwd_kernel, qk_pe_pv),
            ("_fwd_kernel_unified", ea._fwd_kernel_unified, qk_pe_pv),
            ("_verify_mla_prefix_stage1", _verify_mla_prefix_stage1, qk_pe_pv),
            ("_verify_prefix_stage1", _verify_prefix_stage1, qk_pv),
            ("_fwd_grouped_kernel_stage1", _fwd_grouped_kernel_stage1, qk_pv),
        ):
            params = {p.name: p for p in kernel.params}
            for flag in flags:
                with self.subTest(kernel=name, flag=flag):
                    self.assertIn(flag, params)
                    self.assertTrue(params[flag].is_constexpr)


if __name__ == "__main__":
    unittest.main()
