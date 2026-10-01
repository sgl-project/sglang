from collections.abc import Iterator

import pytest
import torch

from sglang.kernels.ops.elementwise.elementwise import fused_gate_sigmoid_mul_add
from sglang.srt.batch_invariant_ops import (
    disable_batch_invariant_mode,
    enable_batch_invariant_mode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def batch_invariant_mode() -> Iterator[None]:
    enable_batch_invariant_mode()
    try:
        yield
    finally:
        disable_batch_invariant_mode()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mm_out_batch_invariance(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    a = torch.randn(257, 10240, dtype=dtype, device="cuda")
    b = torch.randn(320, 10240, dtype=dtype, device="cuda").T
    reference = torch.mm(a[:1], b)
    for rows in (1, 16, 257):
        output = torch.empty(rows, 320, dtype=dtype, device="cuda")
        result = torch.mm(a[:rows], b, out=output)
        assert result is output
        torch.testing.assert_close(output[:1], reference, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_expert_gate_batch_invariance(dtype: torch.dtype) -> None:
    torch.manual_seed(42)
    hidden = torch.randn(256, 2560, dtype=dtype, device="cuda")
    weight = torch.randn(2560, dtype=dtype, device="cuda") / 2560**0.5
    shared = torch.randn_like(hidden)
    residual = torch.randn_like(hidden)
    reference = residual.clone()
    fused_gate_sigmoid_mul_add(hidden, weight, shared, reference)
    for copies in (4, 8):
        output = residual.repeat(copies, 1)
        fused_gate_sigmoid_mul_add(
            hidden.repeat(copies, 1), weight, shared.repeat(copies, 1), output
        )
        torch.testing.assert_close(output[:256], reference, rtol=0, atol=0)
