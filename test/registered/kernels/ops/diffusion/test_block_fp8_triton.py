"""Block-FP8 producers against the unfused bf16 kernel followed by
``sglang_per_token_group_quant_fp8``: fp8 payload and the packed UE8M0 scales
DeepGEMM takes, bit for bit."""

import sys

import pytest
import torch

from sglang.kernels.ops.diffusion import (
    can_use_indexed_scale_shift_block_fp8,
    indexed_scale_shift_bf16_,
    indexed_scale_shift_block_fp8,
)
from sglang.kernels.ops.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN = 5376
GROUP = 128
MODALITIES = 9


def _unfused(x, shift, scale, indices):
    modulated = indexed_scale_shift_bf16_(x.clone(), shift, scale, indices)
    # What deepgemm_w8a8_block_fp8_linear_with_fallback quantizes with; its
    # scales are MN-major, the fused kernel's the same values row-major.
    q, s = sglang_per_token_group_quant_fp8(
        modulated,
        GROUP,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return q, s.contiguous()


def _fused(x, shift, scale, indices):
    return indexed_scale_shift_block_fp8(x, shift, scale, indices, group_size=GROUP)


def _params(g, scale=0.5):
    return tuple(
        (torch.randn(MODALITIES, HIDDEN, device="cuda", generator=g) * scale).to(
            torch.bfloat16
        )
        for _ in range(2)
    )


def _case(name):
    g = torch.Generator(device="cuda").manual_seed(0)
    shift, scale = _params(g)
    zeros = torch.zeros(MODALITIES, HIDDEN, device="cuda", dtype=torch.bfloat16)
    if name == "normal":
        x = torch.randn(2048, HIDDEN, device="cuda", generator=g) * 3
        return x.to(torch.bfloat16), shift, scale
    if name == "heavy_tailed":
        x = torch.randn(2048, HIDDEN, device="cuda", generator=g) * torch.exp(
            torch.randn(2048, 1, device="cuda", generator=g) * 6
        )
        return x.to(torch.bfloat16), shift, scale
    if name == "zero_and_tiny":
        # Whole zero rows put amax on the 1e-10 floor; 1e-30 rows sit far below
        # e4m3's smallest subnormal, where flush-to-zero would differ first.
        x = torch.zeros(256, HIDDEN, device="cuda", dtype=torch.bfloat16)
        x[1::2] = (torch.randn(128, HIDDEN, device="cuda", generator=g) * 1e-30).to(
            torch.bfloat16
        )
        return x, zeros, zeros
    assert name == "infinite"
    x = torch.randn(64, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    x[3, 7] = float("inf")
    x[5, 300] = float("-inf")
    return x, zeros, zeros


@pytest.mark.parametrize(
    "case", ["normal", "heavy_tailed", "zero_and_tiny", "infinite"]
)
def test_indexed_scale_shift_block_fp8_is_bit_exact(case: str) -> None:
    x, shift, scale = _case(case)
    indices = torch.randint(0, MODALITIES, (x.shape[0],), device="cuda")
    expected_q, expected_scale = _unfused(x, shift, scale, indices)
    q, q_scale = _fused(x, shift, scale, indices)
    assert q_scale.shape == expected_scale.shape
    assert torch.equal(q.view(torch.uint8), expected_q.view(torch.uint8))
    assert torch.equal(q_scale.view(torch.int32), expected_scale.view(torch.int32))


def test_indexed_scale_shift_block_fp8_leaves_its_input_alone() -> None:
    x, shift, scale = _case("normal")
    before = x.clone()
    _fused(x, shift, scale, torch.zeros(x.shape[0], dtype=torch.long, device="cuda"))
    assert torch.equal(x, before)


def test_indexed_scale_shift_block_fp8_refuses_what_the_unfused_path_would_not_fuse() -> (
    None
):
    # The unfused modulation falls back to torch arithmetic for these, so a
    # kernel that ran anyway would silently disagree with it.
    x, shift, scale = _case("normal")
    indices = torch.zeros(x.shape[0], dtype=torch.long, device="cuda")
    for args in (
        (x.float(), shift, scale),
        (x.t().contiguous().t(), shift, scale),
        (x, shift.float(), scale),
    ):
        assert not can_use_indexed_scale_shift_block_fp8(*args, group_size=GROUP)
        with pytest.raises(ValueError):
            _fused(*args, indices)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
