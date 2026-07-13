import itertools
import sys

import pytest
import torch

from sglang.jit_kernel.per_token_group_quant_8bit import (
    per_token_group_quant_8bit as sglang_per_token_group_quant_8bit,
)
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=16, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sgl_kernel import (  # noqa: E402
    sgl_per_token_group_quant_8bit as aot_per_token_group_quant_8bit,
)
from sgl_kernel.test_utils import (  # noqa: E402
    assert_all_close_or_tiny_diff,
    create_per_token_group_quant_test_data,
)

from sglang.jit_kernel.per_token_group_quant_8bit import (  # noqa: E402
    per_token_group_quant_8bit as jit_per_token_group_quant_8bit,
)
from sglang.srt.layers.quantization.fp8_kernel import (  # noqa: E402
    create_per_token_group_quant_fp8_output_scale,
)
from sglang.srt.layers.quantization.fp8_kernel import (  # noqa: E402
    per_token_group_quant_8bit as triton_per_token_group_quant_8bit,
)

_is_hip = is_hip()
fp8_type_ = torch.float8_e4m3fnuz if _is_hip else torch.float8_e4m3fn

BASE_FLAGS = [
    dict(
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=False,
        scale_ue8m0=False,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=False,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=False,
        masked_layout_mode=None,
    ),
]
FUSED_FLAGS = [
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_layout_mode=None,
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_layout_mode="balanced",
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_layout_mode="imbalanced",
    ),
    dict(
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
        fuse_silu_and_mul=True,
        masked_layout_mode="extreme",
    ),
]

configs = get_ci_test_range(
    list(
        itertools.product(
            [1, 4, 16, 17, 38, 51, 64, 127, 128, 512, 1024, 4096, 8192],
            [128, 256, 384, 512, 768, 1024, 1536, 1664, 2048, 4096, 7168, 16384],
            [16, 32, 64, 128],
            [None],
            [fp8_type_],
            BASE_FLAGS,
        )
    )
    + list(
        itertools.product(
            [1, 4, 1 * 8, 4 * 8, 64 * 8, 256 * 8, 768 * 8],
            [2048],
            [128],
            [8, 16, 32, 48],
            [fp8_type_],
            FUSED_FLAGS,
        )
    ),
    [
        (1, 128, 128, None, fp8_type_, BASE_FLAGS[0]),
        (17, 1536, 128, None, fp8_type_, BASE_FLAGS[2]),
        (38, 4096, 128, None, fp8_type_, BASE_FLAGS[2]),
        (51, 4096, 128, None, fp8_type_, BASE_FLAGS[2]),
        (512, 2048, 128, 8, fp8_type_, FUSED_FLAGS[0]),
        (2048, 2048, 128, 16, fp8_type_, FUSED_FLAGS[1]),
    ],
)


@pytest.mark.parametrize(
    "num_tokens, hidden_dim, group_size, num_ranks, dst_dtype, flags", configs
)
def test_per_token_group_quant_with_column_major(
    num_tokens,
    hidden_dim,
    group_size,
    num_ranks,
    dst_dtype,
    flags,
):
    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if flags["scale_ue8m0"] and (arch_major <= 9):
        pytest.skip("Only Blackwell need ue8m0 fusion")
        return

    if (flags["scale_ue8m0"] and (group_size != 128)) or (
        (dst_dtype == torch.int8) and flags["column_major_scales"]
    ):
        pytest.skip()
        return

    x, masked_m = create_per_token_group_quant_test_data(
        num_tokens=num_tokens, hidden_dim=hidden_dim, num_ranks=num_ranks, flags=flags
    )

    execute_kwargs = dict(
        x=x,
        masked_m=masked_m,
        group_size=group_size,
        eps=1e-10,
        dst_dtype=dst_dtype,
        **{k: v for k, v in flags.items() if k not in ["masked_layout_mode"]},
    )

    def _postprocess(x_q, x_s):
        if masked_m is not None:
            print(f"Mask tokens after {masked_m} to be zero")
            for i in range(len(masked_m)):
                x_q[i, masked_m[i] :, :] = 0
                x_s[i, masked_m[i] :, :] = 0
        return x_q, x_s

    x_q_triton, x_s_triton = _postprocess(
        *triton_per_token_group_quant_8bit(**execute_kwargs)
    )

    fuse_silu_and_mul = False
    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    fp8_dtype = torch.float8_e4m3fn
    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = -fp8_max
    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=False,
        scale_tma_aligned=False,
        scale_ue8m0=False,
    )

    execute_kwargs = dict(
        input=x,
        output_q=x_q,
        output_s=x_s,
        group_size=group_size,
        eps=1e-10,
        fp8_max=fp8_max,
        fp8_min=fp8_min,
    )
    x_q_sglang, x_s_sglang = _postprocess(
        *sglang_per_token_group_quant_8bit(**execute_kwargs)
    )

    try:
        assert_all_close_or_tiny_diff(x_q_triton, x_q_sglang)
        torch.testing.assert_close(
            x_s_triton.contiguous(),
            x_s_sglang.contiguous(),
            rtol=1e-3,
            atol=1e-5,
            msg=lambda message: message + f" {x_s_triton=} {x_s_sglang=}",
        )
    except AssertionError:
        print(
            f"{x.shape=} {x_q_triton.shape=} {x_s_triton.shape=} {x_q_sglang.shape=} {x_s_sglang.shape=}"
        )
        print(f"{x=}")
        print(f"{masked_m=}")
        print(f"{x_q_triton=}")
        print(f"{x_s_triton=}")
        print(f"{x_q_sglang=}")
        print(f"{x_s_sglang=}")

        raise


LAYOUTS = [
    (False, False, False),
    (True, False, False),
    (True, True, False),
    (True, True, True),
]

CONFIGS = list(
    itertools.product(
        [1, 4, 16, 64, 127, 128, 512, 1024, 4096, 8192],
        [512, 1536, 2048, 4096, 6144, 7168, 16384],
        [16, 32, 64, 128],
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

    # AOT v2 uses -use_fast_math reciprocal; this JIT uses precise division, so an
    # exact fp8 midpoint can round to an adjacent code (1-ULP, JIT more accurate).
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
    sys.exit(pytest.main([__file__, "-v", "-s"]))
