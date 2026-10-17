# SPDX-License-Identifier: Apache-2.0
"""Native cache and stream binding for the source Step5 Q/K norm kernel."""

import cutlass
from cutlass import Float32, Int32, cute

from .compile_cache import cached_compile_function
from .qknorm_rope import FusedQKNormRope


@cached_compile_function
def compile_qknorm(
    rotary_width: int,
    device_key: tuple[str, int, tuple[int, int]],
    logical_q_heads: int = 8,
    store_cache: bool = False,
    gate: bool = False,
):
    device_type, _, capability = device_key
    if device_type != "cuda" or capability[0] not in (9, 10):
        raise ValueError("Step5 Q/K norm port supports Hopper and Blackwell")
    if rotary_width not in (64, 192):
        raise ValueError("Step5 main RoPE widths are 64 (DSA) and 192 (SWA)")

    if logical_q_heads not in (8, 16):
        raise ValueError("Step5 main Q heads must be 8 or 16")
    q_width = logical_q_heads * 192
    qkv_width = q_width + 384
    input_width = qkv_width + logical_q_heads if gate else qkv_width

    def matrix(width, stride=None):
        return cute.runtime.make_fake_tensor(
            cutlass.BFloat16,
            (cute.sym_int(), width),
            (width if stride is None else stride, 1),
            assumed_align=16,
        )

    def weight():
        return cute.runtime.make_fake_tensor(Float32, (192,), (1,), assumed_align=16)

    positions = cute.runtime.make_fake_tensor(
        cutlass.Int64, (cute.sym_int(),), (1,), assumed_align=8
    )
    loc = (
        cute.runtime.make_fake_tensor(
            cutlass.Int64, (cute.sym_int(),), (1,), assumed_align=8
        )
        if store_cache
        else None
    )
    k_cache = matrix(192) if store_cache else None
    v_cache = matrix(192) if store_cache else None
    # Source rotary_dim denotes HALF the rotated width. The input and outputs
    # use packed 2D QKV/heads; only token count and cache capacity are dynamic.
    return cute.compile(
        FusedQKNormRope(
            cutlass.BFloat16,
            qkv_width,
            192,
            logical_q_heads,
            1,
            rotary_width // 2,
            input_size=input_width,
        ),
        matrix(input_width),
        weight(),
        weight(),
        matrix(rotary_width // 2, rotary_width),
        matrix(rotary_width // 2, rotary_width),
        positions,
        matrix(q_width),
        matrix(192),
        matrix(192),
        matrix(logical_q_heads) if gate else None,
        k_cache,
        v_cache,
        loc,
        Int32(192),
        Int32(logical_q_heads),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Float32(1e-5),
        Float32(1),
        options="--enable-tvm-ffi",
    )
