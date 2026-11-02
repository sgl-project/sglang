"""sam_rel_pos_bias must stay bit-identical to the eager decomposed-bias sum."""

import sys

import pytest
import torch

from sglang.kernels.ops.elementwise.sam_rel_pos_bias import (
    can_use_sam_rel_pos_bias,
    sam_rel_pos_bias,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.sam import sam_rel_pos_bias_reference

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _inputs(batch=2, queries=64, key_size=40, dtype=torch.bfloat16, device="cuda"):
    rel_h = torch.randn(batch, 12, queries, key_size, 1, dtype=dtype, device=device)
    rel_w = torch.randn(batch, 12, queries, 1, key_size, dtype=dtype, device=device)
    return rel_h, rel_w


@pytest.mark.parametrize(
    "batch,queries,key_size",
    [(8, 534, 40), (8, 1600, 40), (2, 1024, 64)],
)
def test_sam_rel_pos_bias_bit_identity(batch: int, queries: int, key_size: int):
    torch.manual_seed(20260923)
    rel_h, rel_w = _inputs(batch, queries, key_size)
    expected = sam_rel_pos_bias_reference(rel_h, rel_w)
    actual = sam_rel_pos_bias(rel_h, rel_w)
    assert torch.equal(actual, expected)


def test_query_chunks_stay_on_the_kernel():
    """The SAM attention slices the query axis into chunks; slices must stay eligible."""
    rel_h, rel_w = _inputs(queries=1600)
    for s, e in ((0, 256), (256, 1333), (1333, 1600)):
        h, w = rel_h[:, :, s:e], rel_w[:, :, s:e]
        assert can_use_sam_rel_pos_bias(h, w)
        assert torch.equal(sam_rel_pos_bias(h, w), sam_rel_pos_bias_reference(h, w))


def _strided_rel_w():
    rel_h, _ = _inputs()
    storage = torch.randn(2, 12, 64, 1, 80, dtype=torch.bfloat16, device="cuda")
    return rel_h, storage[..., ::2]


def _misaligned_rel_w():
    rel_h, _ = _inputs()
    storage = torch.randn(2, 12, 64, 1, 48, dtype=torch.bfloat16, device="cuda")
    return rel_h, storage[..., 1:41]


@pytest.mark.parametrize(
    "make",
    [
        lambda: _inputs(dtype=torch.float16),
        lambda: _inputs(dtype=torch.float32),
        lambda: _inputs(device="cpu"),
        lambda: _inputs(key_size=12),
        _strided_rel_w,
        _misaligned_rel_w,
    ],
    ids=["fp16", "fp32", "cpu", "key_width_12", "strided_rel_w", "misaligned_rel_w"],
)
def test_unsupported_inputs_take_the_eager_path(make):
    """Inputs the kernel rejects (or would misread) must take the eager sum."""
    rel_h, rel_w = make()
    assert not can_use_sam_rel_pos_bias(rel_h, rel_w)


@pytest.mark.parametrize(
    "make",
    [lambda: _inputs(dtype=torch.float16), _strided_rel_w, _misaligned_rel_w],
    ids=["fp16", "strided_rel_w", "misaligned_rel_w"],
)
def test_kernel_rejects_unsupported_inputs(make):
    """A non-unit key-width stride used to be read as contiguous and give a wrong bias."""
    rel_h, rel_w = make()
    with pytest.raises(Exception, match="Tensor match failed"):
        sam_rel_pos_bias(rel_h, rel_w)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
