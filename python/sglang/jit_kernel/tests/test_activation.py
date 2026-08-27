import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.activation import SUPPORTED_ACTIVATIONS, run_activation
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="nightly-kernel-1-gpu", nightly=True)


OPS = SUPPORTED_ACTIVATIONS
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
SHAPES = get_ci_test_range(
    full_range=[
        (7, 16),
        (83, 1024),
        (3, 5, 16),
        (2, 3, 512),
        (1, 17, 4096),
        *[(2**x, 2048) for x in range(0, 15, 2)],
        *[(2**x, 65536) for x in range(0, 5, 2)],
    ],
    ci_range=[(7, 16), (2, 3, 512)],
)


def _reference(op_name: str, x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    lhs = x[..., :d].float()
    rhs = x[..., d:]
    if op_name == "silu":
        act = F.silu(lhs)
    elif op_name == "gelu":
        act = F.gelu(lhs, approximate="none")
    else:
        act = F.gelu(lhs, approximate="tanh")
    return act.to(dtype=x.dtype) * rhs


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    return 1e-2, 1e-2


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_activation_correctness(
    op_name: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    x = torch.randn(shape, dtype=dtype, device="cuda")
    out = run_activation(op_name, x, None)
    expected = _reference(op_name, x)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_activation_out_param(
    op_name: str, dtype: torch.dtype, shape: tuple[int, ...]
) -> None:
    x = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty(shape[:-1] + (shape[-1] // 2,), dtype=dtype, device="cuda")
    result = run_activation(op_name, x, out)
    assert result is out
    expected = _reference(op_name, x)
    atol, rtol = _tolerances(dtype)
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)


FILTER_SHAPES = get_ci_test_range(
    full_range=[(83, 1024), (256, 2048), (1024, 4096)],
    ci_range=[(83, 1024)],
)
EXPERT_STEPS = [1, 16]


@pytest.mark.parametrize("op_name", OPS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", FILTER_SHAPES)
@pytest.mark.parametrize("expert_step", EXPERT_STEPS)
def test_activation_filter_expert(
    op_name: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
    expert_step: int,
) -> None:
    """expert_ids[token // expert_step] == -1 must leave the output row untouched."""
    num_tokens = shape[0]
    x = torch.randn(shape, dtype=dtype, device="cuda")
    # Pre-fill out with a sentinel so we can detect untouched rows.
    sentinel = float("nan")
    out = torch.full(
        shape[:-1] + (shape[-1] // 2,),
        sentinel,
        dtype=dtype,
        device="cuda",
    )

    num_groups = (num_tokens + expert_step - 1) // expert_step
    expert_ids = torch.randint(
        low=0, high=8, size=(num_groups,), dtype=torch.int32, device="cuda"
    )
    skip_mask = torch.rand(num_groups, device="cuda") < 0.4
    expert_ids[skip_mask] = -1

    result = run_activation(op_name, x, out, expert_ids, expert_step)
    assert result is out

    token_skip = skip_mask[torch.arange(num_tokens, device="cuda") // expert_step]
    expected = _reference(op_name, x)
    atol, rtol = _tolerances(dtype)

    kept = ~token_skip
    if kept.any():
        torch.testing.assert_close(out[kept], expected[kept], atol=atol, rtol=rtol)
    if token_skip.any():
        assert torch.isnan(
            out[token_skip]
        ).all(), "filter_expert kernel touched rows whose expert_id is -1"


@pytest.mark.parametrize("op_name", OPS)
def test_activation_filter_expert_all_skipped(op_name: str) -> None:
    """If every expert id is -1, the output must be left entirely untouched."""
    shape = (32, 512)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    out = torch.full(
        shape[:-1] + (shape[-1] // 2,),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    expert_ids = torch.full((shape[0],), -1, dtype=torch.int32, device="cuda")
    run_activation(op_name, x, out, expert_ids, 1)
    assert torch.isnan(out).all()


@pytest.mark.parametrize("op_name", OPS)
def test_activation_filter_expert_none_skipped(op_name: str) -> None:
    """No -1 in expert_ids must yield bit-identical output to the unfiltered path."""
    shape = (64, 512)
    dtype = torch.bfloat16
    x = torch.randn(shape, dtype=dtype, device="cuda")
    expert_ids = torch.zeros((shape[0],), dtype=torch.int32, device="cuda")
    out_filtered = run_activation(op_name, x, None, expert_ids, 1)
    out_unfiltered = run_activation(op_name, x, None)
    torch.testing.assert_close(out_filtered, out_unfiltered, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
