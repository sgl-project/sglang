import itertools

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.quantization.per_token_group_quant_8bit_v2 import (
    per_token_group_quant_8bit_v2,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

try:
    from sgl_kernel import sgl_per_token_group_quant_8bit  # AOT v2 reference op
except ImportError:
    sgl_per_token_group_quant_8bit = None

if sgl_per_token_group_quant_8bit is None and not torch.cuda.is_available():
    pytest.skip("sgl_kernel AOT reference op is unavailable", allow_module_level=True)
if sgl_per_token_group_quant_8bit is None:
    raise ImportError("sgl_kernel AOT reference op is unavailable")

from sglang.kernels.ops.quantization.fp8_kernel import (  # noqa: E402
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
    fp8_min,
)

G = 128


def _alloc(x_shape, scale_ue8m0):
    """Pre-allocated (zeroed) output_q + output_s for a given input/output shape.
    Zeroing makes the unwritten (padding / aligned) regions compare equal."""
    x_q = torch.zeros(x_shape, device="cuda", dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=x_shape,
        device="cuda",
        group_size=G,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=scale_ue8m0,
    )
    x_s.zero_()
    return x_q, x_s


V2_QUANT_CASES = get_ci_test_range(
    list(
        itertools.product(
            [torch.bfloat16, torch.float16],
            [1, 7, 38, 64, 333],
            [128, 2048, 4096, 7168],
            [False, True],
            [False, True],
        )
    ),
    [
        (torch.bfloat16, 1, 128, False, False),
        (torch.bfloat16, 17, 1536, False, False),
        (torch.bfloat16, 38, 7168, False, True),
        (torch.bfloat16, 38, 4096, False, True),
        (torch.bfloat16, 64, 4096, True, True),
        (torch.float16, 333, 2048, True, False),
    ],
)


@pytest.mark.parametrize(
    "dtype,num_tokens,hidden,fuse_silu_and_mul,scale_ue8m0", V2_QUANT_CASES
)
def test_v2_jit_matches_aot(dtype, num_tokens, hidden, fuse_silu_and_mul, scale_ue8m0):
    """JIT v2 must be bit-exact with the AOT v2 across vanilla/silu and float/ue8m0
    scales (NaiveScheduler)."""
    torch.manual_seed(
        hidden + num_tokens + int(fuse_silu_and_mul) + 7 * int(scale_ue8m0)
    )
    in_hidden = hidden * (2 if fuse_silu_and_mul else 1)
    x = torch.randn(num_tokens, in_hidden, device="cuda", dtype=dtype)
    out_shape = (num_tokens, hidden)

    q_ref, s_ref = _alloc(out_shape, scale_ue8m0)
    sgl_per_token_group_quant_8bit(
        x,
        q_ref,
        s_ref,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0,
        fuse_silu_and_mul,
        None,
        enable_v2=True,
    )

    x_q, x_s = _alloc(out_shape, scale_ue8m0)
    per_token_group_quant_8bit_v2(
        x,
        x_q,
        x_s,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
    )
    torch.cuda.synchronize()

    assert torch.equal(x_q.view(torch.int8), q_ref.view(torch.int8)), "fp8 codes differ"
    assert torch.equal(x_s, s_ref), "scales differ"


# NOTE: "row-major + scale_ue8m0=True" names two different formats:
#   1. packed int32 [T, ceil(G/4)] (4 exponent bytes per int32) -- supported by
#      the JIT per_token_group_quant kernel and pinned bit-exact in test_per_token_group_quant
#      (test_v3_ue8m0_row_packed_bitexact);
#   2. fp32 [T, G] storing power-of-two VALUES (the deep_gemm.fp8_einsum
#      format) -- v2-only. No srt caller requests it (production ties
#      scale_ue8m0 and column_major_scales to the same DEEPGEMM_SCALE_UE8M0
#      flag), so the srt entry `sglang_per_token_group_quant_fp8`, which now
#      routes to the JIT kernel, rejects it loudly instead of allocating an fp32
#      buffer the kernel cannot fill. The v2 JIT kernel itself still implements it and is
#      covered by test_v2_jit_matches_aot above.


# Masked (EP-MoE) path: the v2 op only has a masked scheduler for the
# column-major + ue8m0 + fused-silu+mul + masked combination. Input is 3D
# [num_experts, tokens_padded, hidden*2]; only tokens < masked_m[e] are processed
# (padding left untouched → zeros in both). Compare JIT vs AOT bit-exact.
MASKED_V2_CASES = get_ci_test_range(
    # MaskedLayoutScheduler requires hidden / group_size to be divisible by 16.
    list(itertools.product([2, 5], [2048, 4096, 8192], [128, 384])),
    [(2, 2048, 128), (5, 4096, 384), (2, 8192, 128)],
)


@pytest.mark.parametrize("num_experts,hidden,tokens_pad", MASKED_V2_CASES)
def test_v2_jit_masked_matches_aot(num_experts, hidden, tokens_pad):
    torch.manual_seed(num_experts * 1000 + hidden + tokens_pad)
    x = torch.randn(
        num_experts, tokens_pad, hidden * 2, device="cuda", dtype=torch.bfloat16
    )
    masked_m = torch.randint(
        0, tokens_pad + 1, (num_experts,), device="cuda", dtype=torch.int32
    )
    out_shape = (num_experts, tokens_pad, hidden)

    q_ref, s_ref = _alloc(out_shape, scale_ue8m0=True)
    sgl_per_token_group_quant_8bit(
        x,
        q_ref,
        s_ref,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        True,
        True,
        masked_m,
        enable_v2=True,
    )

    x_q, x_s = _alloc(out_shape, scale_ue8m0=True)
    per_token_group_quant_8bit_v2(
        x,
        x_q,
        x_s,
        G,
        1e-10,
        float(fp8_min),
        float(fp8_max),
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_m=masked_m,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        x_q.view(torch.int8), q_ref.view(torch.int8)
    ), "masked fp8 differ"
    assert torch.equal(x_s, s_ref), "masked scales differ"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
