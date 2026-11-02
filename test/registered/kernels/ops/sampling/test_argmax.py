import sys

import pytest
import torch

from sglang.kernels.ops.sampling import argmax as sampling_argmax
from sglang.kernels.ops.sampling.argmax_jit import (
    argmax,
    can_argmax,
    create_argmax_workspace,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def _reference(logits: torch.Tensor) -> torch.Tensor:
    # numpy's argmax is defined to return the first maximum, which is the
    # tie-break the kernel promises; torch.argmax only documents it loosely.
    return torch.from_numpy(logits.float().cpu().numpy().argmax(axis=1)).to(torch.int32)


def _check(logits: torch.Tensor, **kwargs) -> None:
    got = argmax(logits, **kwargs)
    assert got.dtype == kwargs.get("index_dtype", torch.int32)
    torch.testing.assert_close(got.cpu().to(torch.int32), _reference(logits))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batch_size,vocab_size", [(1, 262144), (17, 32000)])
def test_argmax_index_dtype(index_dtype, batch_size, vocab_size) -> None:
    """int64 indices, for callers matching torch.argmax's return type."""
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32, device="cuda")
    _check(logits, index_dtype=index_dtype)
    out = torch.empty(batch_size, dtype=index_dtype, device="cuda")
    assert argmax(logits, out=out) is out
    torch.testing.assert_close(out.cpu().to(torch.int32), _reference(logits))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size", [1, 3, 32, 257])
@pytest.mark.parametrize("vocab_size", [32, 1024, 32000, 128256])
def test_argmax_correctness(dtype, batch_size, vocab_size) -> None:
    _check(torch.randn(batch_size, vocab_size, dtype=dtype, device="cuda"))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch_size,vocab_size", [(1, 262144), (2, 262144)])
def test_argmax_multi_split(dtype, batch_size, vocab_size) -> None:
    """A tall, narrow batch splits each row across blocks and takes the atomic path."""
    _check(torch.randn(batch_size, vocab_size, dtype=dtype, device="cuda"))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("vocab_size,padded", [(1024, 2048), (128256, 151936)])
def test_argmax_padded_rows(dtype, vocab_size, padded) -> None:
    """Rows are read at `stride(0)`, which a padded buffer makes wider than `vocab_size`.

    Deriving the row offset from the width instead reads every row but the first
    from the wrong place, and only a stride > width catches it.
    """
    full = torch.randn(17, padded, dtype=dtype, device="cuda")
    _check(full[:, :vocab_size])


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_ties_take_lowest_index(dtype) -> None:
    logits = torch.zeros(4, 4096, dtype=dtype, device="cuda")
    logits[:, 100:200] = 1.0
    got = argmax(logits)
    assert torch.all(got == 100)


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_signed_zero_ties_across_splits(dtype) -> None:
    # -0.0 and +0.0 tie; the row is long enough to be split across CTAs.
    logits = torch.full((1, 262144), -1.0, dtype=dtype, device="cuda")
    logits[0, 0] = -0.0
    logits[0, 131072] = 0.0
    _check(logits)


def test_argmax_misaligned_view_falls_back() -> None:
    # Row stride and vocab size are vector multiples, but the base address is not.
    logits = torch.randn(2, 1032, device="cuda")[:, 1:1025]
    assert not can_argmax(logits)
    got = sampling_argmax(logits)
    torch.testing.assert_close(got.cpu().to(torch.int32), _reference(logits))


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_all_equal(dtype) -> None:
    got = argmax(torch.full((8, 8192), -2.5, dtype=dtype, device="cuda"))
    assert torch.all(got == 0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_masked_rows(dtype) -> None:
    """Grammar masking drives whole rows to -inf; the one live token must win."""
    logits = torch.full((6, 65536), float("-inf"), dtype=dtype, device="cuda")
    live = torch.arange(6, device="cuda") * 9001 + 5
    logits[torch.arange(6, device="cuda"), live] = 1.0
    torch.testing.assert_close(argmax(logits).cpu(), live.to(torch.int32).cpu())


@pytest.mark.parametrize("dtype", DTYPES)
def test_argmax_all_neg_inf(dtype) -> None:
    got = argmax(torch.full((4, 4096), float("-inf"), dtype=dtype, device="cuda"))
    assert torch.all(got == 0)


def test_argmax_repeated_calls_reuse_workspace() -> None:
    """The finalize kernel must hand each slot back zeroed, or call N+1 sees stale maxima."""
    for _ in range(8):
        _check(torch.randn(2, 262144, dtype=torch.bfloat16, device="cuda"))


def test_argmax_out_param() -> None:
    logits = torch.randn(64, 32000, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(64, dtype=torch.int32, device="cuda")
    assert argmax(logits, out=out) is out
    torch.testing.assert_close(out.cpu(), _reference(logits))


def test_argmax_explicit_workspace() -> None:
    logits = torch.randn(2, 262144, dtype=torch.bfloat16, device="cuda")
    workspace = create_argmax_workspace(2, logits.device)
    for _ in range(3):
        _check(logits, workspace=workspace)
    assert torch.all(workspace == 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
