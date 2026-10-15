# Copyright 2023-2024 SGLang Team
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

import unittest

import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.quantization.fp8_utils import unshuffle_aiter_fp8_weight
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(is_hip(), "requires ROCm AITER")
class TestAiterFp8Utils(CustomTestCase):
    def test_unshuffle_weight_round_trip(self):
        from aiter.ops.shuffle import shuffle_weight

        for shape in ((32, 64), (2, 32, 64)):
            with self.subTest(shape=shape):
                logical = (
                    torch.arange(
                        torch.Size(shape).numel(), device="cuda", dtype=torch.float32
                    )
                    .remainder(7)
                    .to(torch.float8_e4m3fn)
                    .reshape(shape)
                )
                shuffled = shuffle_weight(logical, layout=(16, 16))

                torch.testing.assert_close(
                    unshuffle_aiter_fp8_weight(shuffled), logical
                )

    def test_cp_row_slice_preserves_aiter_layout(self):
        from aiter.ops.shuffle import shuffle_weight

        tp_size = 4
        rank = 2
        n, k = 32, 128
        logical_weight = (
            torch.arange(n * k, device="cuda", dtype=torch.float32).remainder(7) - 3
        ).to(torch.float8_e4m3fn)
        logical_weight = logical_weight.reshape(n, k)
        shuffled_weight = shuffle_weight(logical_weight, layout=(16, 16))
        context = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
        context.decode_tp_rank = rank
        context.decode_tp_size = tp_size

        expected = context._slice(logical_weight, dim=1)
        naive = context._slice(shuffled_weight, dim=1)
        self.assertFalse(torch.equal(unshuffle_aiter_fp8_weight(naive), expected))

        actual = context._slice_aiter_bpreshuffled_weight(shuffled_weight)
        torch.testing.assert_close(
            actual,
            shuffle_weight(expected, layout=(16, 16)),
        )

    @unittest.skipUnless(is_gfx95_supported(), "gfx950 MXFP8 routes")
    def test_cp_slice_gfx95_mxfp8_routes(self):
        """A rank's sliced 32x32-block linear (DeepSeek-V4.1) must compute the rank's
        slice of the logical GEMM on every gfx950 MXFP8 route, not only the Triton one."""
        from sglang.kernels.ops.quantization.mxfp8_aiter_gfx95 import (
            aiter_mxfp8_blockscaled_linear,
        )
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            dequant_block_fp8_weight_to_bf16,
            dot_scaled_mxfp8_blockscaled_linear,
        )
        from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
            mxfp8_native_blockscaled_linear,
            prepare_mxfp8_native_weight,
            ue8m0_weight_scale,
        )
        from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
        from sglang.srt.layers.quantization.fp8_utils import (
            block_fp8_scale_to_mxfp8_e8m0,
        )

        tp_size, rank, n, k, m = 4, 2, 256, 512, 8
        gen = torch.Generator(device="cuda").manual_seed(0)
        weight = torch.randn(n, k, device="cuda", generator=gen).to(torch.float8_e4m3fn)
        exponents = torch.randint(
            -4, 3, (n // 32, k // 32), device="cuda", generator=gen
        )
        scale = torch.pow(2.0, exponents.float())
        # On the fp8 grid, so the routes' per-32 activation quant is exact.
        x = torch.randn(m, k, device="cuda", generator=gen).to(torch.float8_e4m3fn)
        x = x.to(torch.bfloat16)
        reference = dequant_block_fp8_weight_to_bf16(weight, scale, (32, 32))
        context = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
        context.decode_tp_rank = rank
        context.decode_tp_size = tp_size
        context._slice_cache = {}

        shuffled, scale_native = prepare_mxfp8_native_weight(weight, scale, (32, 32))
        routes = {
            "aiter": (
                weight,
                "weight_scale_mx_e8m0",
                ue8m0_weight_scale(scale),
                aiter_mxfp8_blockscaled_linear,
            ),
            "native": (
                shuffled.view(torch.float8_e4m3fn),
                "weight_scale_mx_e8m0",
                scale_native,
                mxfp8_native_blockscaled_linear,
            ),
            "dot_scaled": (
                weight,
                "weight_scale_inv_mx",
                block_fp8_scale_to_mxfp8_e8m0(scale, (n, k), [32, 32]),
                dot_scaled_mxfp8_blockscaled_linear,
            ),
        }
        for cls, dim in ((ColumnParallelLinear, 0), (RowParallelLinear, 1)):
            chunk = (n if dim == 0 else k) // tp_size
            span = slice(rank * chunk, (rank + 1) * chunk)
            x_rank = x if dim == 0 else x[:, span].contiguous()
            ref_weight = reference[span] if dim == 0 else reference[:, span]
            expected = x_rank.float() @ ref_weight.float().t()
            for route, (w, scale_attr, route_scale, linear_fn) in routes.items():
                with self.subTest(linear=cls.__name__, route=route):
                    layer = cls.__new__(cls)
                    torch.nn.Module.__init__(layer)
                    layer.weight = w
                    layer.weight_scale_inv = scale
                    setattr(layer, scale_attr, route_scale)
                    layer.weight_bf16 = reference
                    for obj, attr_name, attr_dim in context._get_linear_attrs(layer):
                        context._activate(obj, attr_name, attr_dim)
                    torch.testing.assert_close(layer.weight_bf16, ref_weight)
                    # The native kernels read the lane-ordered weight as bytes (fp8_hip).
                    w_rank = (
                        layer.weight.view(torch.uint8)
                        if route == "native"
                        else layer.weight
                    )
                    out = linear_fn(x_rank, w_rank, getattr(layer, scale_attr))
                    torch.testing.assert_close(
                        out.float(), expected, atol=0.1, rtol=0.01
                    )


if __name__ == "__main__":
    unittest.main()
