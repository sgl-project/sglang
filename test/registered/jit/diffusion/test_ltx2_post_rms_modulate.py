import sys

import pytest
import torch

from sglang.jit_kernel.diffusion.ltx2_post_rms_modulate import (
    can_use_ltx2_post_rms_dual_modulate_cuda,
    can_use_ltx2_post_rms_modulate_cuda,
    ltx2_post_rms_dual_modulate_cuda,
    ltx2_post_rms_modulate_cuda,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _require_cuda_b200() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("LTX2 post-RMS modulation CUDA path is validated on B200")


@pytest.fixture(autouse=True)
def cuda_setup():
    _require_cuda_b200()
    torch.cuda.manual_seed(20260704)


def _make_param(
    batch: int, seq: int, hidden: int, param_seq: int, *, non_contiguous: bool
) -> torch.Tensor:
    if non_contiguous:
        packed = torch.randn(
            batch, param_seq, 3, hidden, device="cuda", dtype=torch.bfloat16
        )
        return packed.unbind(dim=2)[1]
    return torch.randn(batch, param_seq, hidden, device="cuda", dtype=torch.bfloat16)


def _reference(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale) + shift


@pytest.mark.parametrize(
    "batch,seq,hidden,param_seq,non_contiguous",
    [
        (1, 161, 4096, 1, False),
        (1, 529, 4096, 529, True),
        (2, 7, 2048, 1, True),
    ],
)
@torch.no_grad()
def test_ltx2_post_rms_modulate_matches_torch_exactly(
    batch: int,
    seq: int,
    hidden: int,
    param_seq: int,
    non_contiguous: bool,
) -> None:
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    scale = _make_param(batch, seq, hidden, param_seq, non_contiguous=non_contiguous)
    shift = _make_param(batch, seq, hidden, param_seq, non_contiguous=non_contiguous)

    assert can_use_ltx2_post_rms_modulate_cuda(x, scale, shift)
    actual = ltx2_post_rms_modulate_cuda(x, scale, shift)
    expected = _reference(x, scale, shift)
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


@torch.no_grad()
def test_ltx2_post_rms_dual_modulate_matches_torch_exactly() -> None:
    batch, seq, hidden = 1, 32640, 4096
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    scale0 = _make_param(batch, seq, hidden, seq, non_contiguous=True)
    shift0 = _make_param(batch, seq, hidden, seq, non_contiguous=True)
    scale1 = _make_param(batch, seq, hidden, seq, non_contiguous=True)
    shift1 = _make_param(batch, seq, hidden, seq, non_contiguous=True)

    assert can_use_ltx2_post_rms_dual_modulate_cuda(x, scale0, shift0, scale1, shift1)
    actual0, actual1 = ltx2_post_rms_dual_modulate_cuda(
        x, scale0, shift0, scale1, shift1
    )
    expected0 = _reference(x, scale0, shift0)
    expected1 = _reference(x, scale1, shift1)
    torch.cuda.synchronize()

    assert torch.equal(actual0, expected0)
    assert torch.equal(actual1, expected1)


@torch.no_grad()
def test_ltx2_post_rms_modulate_rejects_unsupported_inputs() -> None:
    x = torch.randn(1, 4, 4096, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)

    assert can_use_ltx2_post_rms_modulate_cuda(x, scale, shift)
    assert not can_use_ltx2_post_rms_modulate_cuda(x.float(), scale, shift)
    assert not can_use_ltx2_post_rms_modulate_cuda(x, scale.transpose(-1, -2), shift)


def test_ltx2_post_rms_modulate_custom_op_torch_compile_fullgraph() -> None:
    x = torch.randn(1, 5, 2048, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(1, 1, 2048, device="cuda", dtype=torch.bfloat16)
    shift = torch.randn(1, 1, 2048, device="cuda", dtype=torch.bfloat16)

    compiled = torch.compile(ltx2_post_rms_modulate_cuda, fullgraph=True)
    actual = compiled(x, scale, shift)
    expected = _reference(x, scale, shift)
    torch.cuda.synchronize()
    assert torch.equal(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
