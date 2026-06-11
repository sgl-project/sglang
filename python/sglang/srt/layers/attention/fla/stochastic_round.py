# -*- coding: utf-8 -*-
"""Hardware stochastic rounding for the native narrow SSM/GDN state store.

When ``--mamba-ssm-enable-stochastic-rounding`` is set, the fla recurrence kernels
round the fp32 accumulated recurrent state to the narrow storage dtype
(fp16 / bf16) **stochastically** instead of round-to-nearest, at the final-state
store. This is the lossy fp32->narrow step for ``--mamba-ssm-dtype``.

Hardware path only: ``cvt.rs.{f16x2,bf16x2}.f32`` (PTX stochastic-rounding convert,
SM100+ / Blackwell), fed by per-element Philox random bits.

There is **no software fallback**: if ``cvt.rs`` is unsupported on the GPU arch the
kernel simply fails to compile (the combination is rejected at startup in
``ServerArgs``; see ``--mamba-ssm-enable-stochastic-rounding``).
"""

import triton
import triton.language as tl


@triton.jit
def convert_rs_fp16x2(x, rand):
    """fp32 -> fp16 with hardware stochastic rounding (``cvt.rs.f16x2.f32``).

    Processes two f32 lanes -> one f16x2 per asm invocation (``pack=2``); the
    random bits drive the stochastic rounding decision.
    """
    return tl.inline_asm_elementwise(
        asm="""{
        cvt.rs.f16x2.f32 $0, $2, $1, $3;
        }""",
        constraints=("=r,r,r,r,r"),
        args=(x, rand),
        dtype=tl.float16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def convert_rs_bf16x2(x, rand):
    """fp32 -> bf16 with hardware stochastic rounding (``cvt.rs.bf16x2.f32``)."""
    return tl.inline_asm_elementwise(
        asm="""{
        cvt.rs.bf16x2.f32 $0, $2, $1, $3;
        }""",
        constraints=("=r,r,r,r,r"),
        args=(x, rand),
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def rs_round_state(b_h, seed, offsets, DTYPE: tl.constexpr, PHILOX_ROUNDS: tl.constexpr):
    """Stochastically round fp32 tile ``b_h`` to ``DTYPE`` (fp16 or bf16).

    ``seed`` is a scalar per-call Philox seed (loaded from a device tensor so it is
    CUDA-graph capturable and advances per replay); ``offsets`` are per-element
    Philox counters that decorrelate lanes. ``DTYPE`` is the destination
    (cache) element type, passed as ``ptr.dtype.element_ty``.
    """
    rand = tl.randint(seed, offsets, PHILOX_ROUNDS)
    if DTYPE == tl.float16:
        return convert_rs_fp16x2(b_h, rand)
    else:
        tl.static_assert(
            DTYPE == tl.bfloat16,
            "GDN stochastic rounding requires the SSM cache dtype to be fp16 or bf16",
        )
        return convert_rs_bf16x2(b_h, rand)
