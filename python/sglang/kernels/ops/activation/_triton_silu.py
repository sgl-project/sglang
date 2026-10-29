# SPDX-License-Identifier: Apache-2.0
"""Private Triton activation helpers with CUDA fast-math rounding."""

import triton
import triton.language as tl


@triton.jit
def _silu_fast_math(gate):
    # Matches CUDA silu_and_mul, whose --use_fast_math build computes
    # __fdividef(x, 1 + __expf(-x)); accurate exp/div can move BF16 results.
    # Keep SiLU in FP32; callers apply the multiply before the final BF16 store.
    # __expf(-x) == ex2.approx(x * -log2(e)); 0fBFB8AA3B is -log2(e).
    exp_neg_gate = tl.inline_asm_elementwise(
        "{ mul.ftz.f32 $0, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 $0, $0; }",
        "=f,f",
        [gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;",
        "=f,f,f",
        [gate, 1.0 + exp_neg_gate],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
