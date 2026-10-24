import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_fused_inplace_qknorm_rope,
    fused_inplace_qknorm_rope,
    fused_qknorm_rope_out_of_place,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def test_out_of_place_qknorm_rope_matches_inplace_and_keeps_inputs() -> None:
    """The out-of-place variant (strided fused-qkv views in, contiguous copies
    out) is bit-equal to the in-place kernel and leaves its inputs untouched;
    VDN-H3's linear branch reads the raw q/k after it."""
    T, H, D, R = 512, 4, 128, 96
    if not can_use_fused_inplace_qknorm_rope(
        D, R, True, torch.bfloat16, torch.bfloat16, True
    ):
        pytest.skip("fused qknorm+rope JIT kernel unavailable")
    g = torch.Generator(device="cpu").manual_seed(0)
    qkv = torch.randn(T, 3 * H * D, generator=g).to("cuda", torch.bfloat16)
    q = qkv[:, : H * D].view(T, H, D)
    k = qkv[:, H * D : 2 * H * D].view(T, H, D)
    qw = (torch.rand(D, generator=g) + 0.5).to("cuda", torch.bfloat16)
    kw = (torch.rand(D, generator=g) + 0.5).to("cuda", torch.bfloat16)
    freqs = torch.randn(T, R // 2, generator=g).to("cuda")
    cache = torch.cat((freqs.cos(), freqs.sin()), -1).to(torch.bfloat16).contiguous()
    pos = torch.arange(T, device="cuda")
    kwargs = dict(
        is_neox=True, eps=1e-5, head_dim=D, rope_dim=R, round_norm_before_rope=True
    )
    q_ref, k_ref = q.clone(), k.clone()
    fused_inplace_qknorm_rope(q_ref, k_ref, qw, kw, cache, pos, **kwargs)
    q_out = torch.empty(T, H, D, device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty_like(q_out)
    before = qkv.clone()
    fused_qknorm_rope_out_of_place(q, k, q_out, k_out, qw, kw, cache, pos, **kwargs)
    assert torch.equal(qkv, before)
    assert torch.equal(q_out, q_ref) and torch.equal(k_out, k_ref)


# each case JIT-builds two modules, so these cover the branches, not the product:
# H3's own config both ways, the FP32-cache, FP16, int32 and rope-width paths, then
# the other layouts that run two rows per warp (FLUX / Qwen-Image / FLUX 3 / Krea2 /
# Joy take the interleaved FP32 path; LongCat the rounded interleaved full-width one)
@pytest.mark.parametrize(
    "out_of_place,rope_dim,is_neox,round_norm,full_width,dtype,cache_dtype,pos_dtype",
    [
        (False, 96, True, True, False, torch.bfloat16, torch.bfloat16, torch.int64),
        (True, 96, True, True, False, torch.bfloat16, torch.bfloat16, torch.int64),
        (True, 32, True, True, False, torch.bfloat16, torch.float32, torch.int32),
        (False, 128, True, True, False, torch.float16, torch.float16, torch.int32),
        (False, 128, False, False, False, torch.bfloat16, torch.float32, torch.int64),
        (True, 128, False, False, False, torch.bfloat16, torch.float32, torch.int64),
        (False, 128, False, True, True, torch.bfloat16, torch.bfloat16, torch.int64),
        (False, 120, False, True, False, torch.bfloat16, torch.float32, torch.int32),
        (True, 64, False, False, True, torch.float16, torch.float16, torch.int32),
        (False, 128, True, True, True, torch.bfloat16, torch.bfloat16, torch.int64),
        (False, 64, False, False, False, torch.bfloat16, torch.bfloat16, torch.int64),
        (True, 96, False, False, True, torch.bfloat16, torch.float32, torch.int32),
    ],
)
def test_two_rows_per_warp_match_one_row_per_warp(
    out_of_place: bool,
    rope_dim: int,
    is_neox: bool,
    round_norm: bool,
    full_width: bool,
    dtype: torch.dtype,
    cache_dtype: torch.dtype,
    pos_dtype: torch.dtype,
) -> None:
    """head_dim 128 runs two rows per warp; its bytes must equal the
    one-row-per-warp kernel's on every arch, RoPE layout and rounding mode."""
    from sglang.kernels.ops.diffusion.rope.qknorm_rope_jit import (
        _jit_qknorm_rope_module,
    )

    T, H, D = 1037, 7, 128  # an odd row count leaves the last warp one row
    g = torch.Generator(device="cpu").manual_seed(rope_dim)
    qkv = (3 * torch.randn(T, 3 * H * D, generator=g)).to("cuda", dtype)
    q = qkv[:, : H * D].view(T, H, D)
    k = qkv[:, H * D : 2 * H * D].view(T, H, D)
    qw = (torch.rand(D, generator=g) + 0.5).to("cuda", dtype)
    kw = (torch.rand(D, generator=g) + 0.5).to("cuda", dtype)
    freqs = 50 * torch.randn(T, rope_dim if full_width else rope_dim // 2, generator=g)
    cache = torch.cat((freqs.cos(), freqs.sin()), -1).to("cuda", cache_dtype)
    pos = torch.randperm(T, generator=g).to("cuda", pos_dtype)
    outputs = []
    for half_warp in (False, True):
        module = _jit_qknorm_rope_module(
            D,
            rope_dim,
            is_neox,
            dtype,
            cache_dtype,
            round_norm,
            False,
            full_width,
            out_of_place,
            half_warp=half_warp,
        )
        if out_of_place:
            q_out = torch.empty(T, H, D, device="cuda", dtype=dtype)
            k_out = torch.empty_like(q_out)
            module.qknorm_rope_out_of_place(
                q, k, q_out, k_out, qw, kw, cache, pos, 1e-5
            )
            outputs.append((q_out, k_out))
        else:
            q2, k2 = q.clone(), k.clone()
            module.qknorm_rope(q2, k2, qw, kw, cache, pos, 1e-5)
            outputs.append((q2, k2))
    (q_ref, k_ref), (q_new, k_new) = outputs
    assert torch.equal(q_ref, q_new) and torch.equal(k_ref, k_new)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
