# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Fresh-region CSA with the Step5 execution and E4M3 format contract.

Mathematics/format contract follows Step5 csa_compact_update_sm90_gqa.py and
fp_compat.py; allocation, staging and the source SW128 layout are not ported.
The source rounds the first product before combining it with fresh +0 state;
the remaining seven numerator updates are fused. These rounding boundaries
apply to fresh complete regions; accumulated/partial state is a separate path.
"""

import triton
import triton.language as tl


@triton.jit
def _step5_csa_region_kernel(index_key, compression_logits, out):
    region = tl.program_id(0).to(tl.int64)
    channel = tl.arange(0, 256)
    base = region * (8 * 256) + channel

    # A region-wide scalar maximum, including every token/channel. The source
    # floor makes substituting per-channel maxima a mathematical change.
    local_max = tl.full((256,), -float("inf"), tl.float32)
    for offset in tl.static_range(8):
        logit = tl.load(compression_logits + base + offset * 256).to(tl.float32)
        local_max = tl.maximum(local_max, logit)
    maximum = tl.max(local_max, axis=0)

    numerator = tl.full((256,), 0.0, tl.float32)
    denominator = tl.full((256,), 0.0, tl.float32)
    for offset in tl.static_range(8):
        logit = tl.load(compression_logits + base + offset * 256).to(tl.float32)
        value = tl.load(index_key + base + offset * 256).to(tl.float32)
        weight = tl.exp2((logit - maximum) * 1.4426950408889634)
        if offset == 0:
            # Preserve both RN boundaries without libdevice's default FTZ.
            numerator = tl.inline_asm_elementwise(
                asm="""
                {
                    .reg .f32 product;
                    mul.rn.f32 product, $1, $2;
                    add.rn.f32 $0, product, 0f00000000;
                }
                """,
                constraints="=f,f,f",
                args=[weight, value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        else:
            numerator = tl.fma(weight, value, numerator)
        denominator = denominator + weight
    mean = tl.div_rn(numerator, tl.maximum(denominator, 1.0e-20))
    mean = tl.where(denominator > 0.0, mean, 0.0)

    # Retain the BF16 rounding boundary before the source's exact satfinite
    # E4M3 conversion; casting directly from FP32 would change tie cases.
    rounded = mean.to(tl.bfloat16).to(tl.float32)
    packed = tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b16 pair;
            .reg .f32 zero;
            mov.f32 zero, 0f00000000;
            cvt.rn.satfinite.e4m3x2.f32 pair, zero, $1;
            cvt.u32.u16 $0, pair;
        }
        """,
        constraints="=r,f",
        args=[rounded],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    tl.store(out + region * 256 + channel, packed.to(tl.uint8))


def _launch_step5_csa_region(index_key, compression_logits, out) -> None:
    # Public metadata validation has already established complete contiguous
    # regions and disjoint output storage. No runtime metadata cache is needed.
    _step5_csa_region_kernel[(index_key.shape[0],)](
        index_key,
        compression_logits,
        out,
        num_warps=8,
        num_stages=1,
        enable_fp_fusion=True,
    )
