"""Strong correctness test for the JIT per_token_group_quant_8bit kernel.

The JIT kernel is a drop-in, faster replacement for the production AOT kernel
``sgl_per_token_group_quant_8bit`` (v2). Since both round the scale identically
(ceil(log2) for UE8M0, multiply-by-reciprocal otherwise) the JIT output is
byte-identical to the AOT v2 output, except at rare exact fp8 rounding ties where
AOT's -use_fast_math reciprocal and the JIT's precise division round to adjacent
codes (allowed within 1 ULP). This asserts that across all supported variants:

  * row-major float scale
  * column-major float scale
  * column-major UE8M0 (int32-packed) scale  -- the MXFP8 path, incl. group=32
    (the MiniMax-M3 case the previous test skipped)

over group sizes {16, 32, 64, 128} and a wide range of (num_tokens, hidden_dim),
including the [8192, 6144] prefill shape.
"""

import itertools
import sys

import pytest
import torch

from sglang.srt.utils import is_hip

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

from sgl_kernel import sgl_per_token_group_quant_8bit as aot_per_token_group_quant_8bit

from sglang.jit_kernel.per_token_group_quant_8bit import (
    per_token_group_quant_8bit as jit_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)
# No register_amd_ci: this test compares against the AOT sgl_kernel v2 kernel,
# which is not built in the ROCm extension (common_extension_rocm.cc).

# (column_major_scales, scale_tma_aligned, scale_ue8m0)
LAYOUTS = [
    (False, False, False),  # row-major float
    (True, False, False),  # column-major float
    (True, True, False),  # column-major float, tma-aligned
    (True, True, True),  # column-major UE8M0 (mxfp8)
]

CONFIGS = list(
    itertools.product(
        [1, 4, 16, 64, 127, 128, 512, 1024, 4096, 8192],  # num_tokens
        [512, 1536, 2048, 4096, 6144, 7168, 16384],  # hidden_dim
        [16, 32, 64, 128],  # group_size
        LAYOUTS,
        [fp8_type_],
    )
)


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, layout, dst_dtype", CONFIGS
)
def test_jit_matches_aot_v2_byte_identical(
    num_tokens, hidden_dim, group_size, layout, dst_dtype
):
    column_major_scales, scale_tma_aligned, scale_ue8m0 = layout

    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if scale_ue8m0 and arch_major <= 9:
        pytest.skip("UE8M0 fusion is Blackwell-only")
    if hidden_dim % group_size != 0:
        pytest.skip("hidden_dim must be divisible by group_size")

    torch.manual_seed(num_tokens * 131 + hidden_dim + group_size)
    x = (torch.randn(num_tokens, hidden_dim, device="cuda", dtype=torch.bfloat16)) * 3.0

    fp8_max = torch.finfo(dst_dtype).max
    fp8_min = -fp8_max

    def _alloc():
        q = torch.empty_like(x, dtype=dst_dtype)
        s = create_per_token_group_quant_fp8_output_scale(
            x_shape=x.shape,
            device=x.device,
            group_size=group_size,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
        )
        return q, s

    q_aot, s_aot = _alloc()
    aot_per_token_group_quant_8bit(
        x,
        q_aot,
        s_aot,
        group_size,
        1e-10,
        fp8_min,
        fp8_max,
        scale_ue8m0,
        False,
        None,
        enable_v2=True,
    )

    q_jit, s_jit = _alloc()
    jit_per_token_group_quant_8bit(
        x, q_jit, s_jit, group_size, 1e-10, fp8_min, fp8_max, scale_ue8m0=scale_ue8m0
    )

    # Quantized values must match the AOT v2 kernel bit-for-bit, EXCEPT at exact
    # fp8 rounding ties. AOT v2 (sgl-kernel) is built with -use_fast_math, so its
    # MAX/amax reciprocal is approximate, while this JIT kernel uses precise
    # division; a value landing exactly on an fp8 midpoint can therefore round to
    # adjacent codes -- a deterministic 1-ULP difference (the JIT value is the more
    # accurate one). Allow such isolated 1-ULP ties (same sign, adjacent fp8 byte)
    # but reject anything larger or systemic.
    qj = q_jit.view(torch.uint8)
    qa = q_aot.view(torch.uint8)
    if not torch.equal(qj, qa):
        mism = qj != qa
        bj = qj[mism].to(torch.int16)
        ba = qa[mism].to(torch.int16)
        same_sign = (bj & 0x80) == (ba & 0x80)
        one_ulp = (bj - ba).abs() == 1
        assert bool(
            (same_sign & one_ulp).all()
        ), f"q mismatch > 1 fp8 ULP {num_tokens=} {hidden_dim=} {group_size=} {layout=}"
        assert mism.float().mean() < 0.01, (
            f"too many fp8 ties ({int(mism.sum())}/{mism.numel()}) "
            f"{num_tokens=} {hidden_dim=} {group_size=} {layout=}"
        )

    # Scales: int32-packed for UE8M0, float otherwise -- both must match exactly
    # on the valid (num_tokens) region.
    if scale_ue8m0:
        assert torch.equal(
            s_jit[:num_tokens].reshape(num_tokens, -1).view(torch.int32),
            s_aot[:num_tokens].reshape(num_tokens, -1).view(torch.int32),
        ), f"ue8m0 scale mismatch {num_tokens=} {hidden_dim=} {group_size=}"
    else:
        assert torch.equal(
            s_jit[:num_tokens].float(), s_aot[:num_tokens].float()
        ), f"float scale mismatch {num_tokens=} {hidden_dim=} {group_size=}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
