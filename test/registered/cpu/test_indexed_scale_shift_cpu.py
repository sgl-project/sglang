import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.kernels.ops.diffusion.modulate.indexed_modulation_triton import (
    can_use_indexed_scale_shift_bf16_cpu,
    indexed_scale_shift_bf16_,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-b-test-cpu")


def _make_chunk_view(rows: int, hidden: int, chunk_index: int) -> torch.Tensor:
    storage = torch.randn(rows, hidden * 6, dtype=torch.bfloat16)
    return storage.chunk(6, dim=-1)[chunk_index]


def _reference(x, shift, scale, indices):
    gathered_scale = scale.index_select(0, indices)
    gathered_shift = shift.index_select(0, indices)
    one_plus_scale = (1.0 + gathered_scale.float()).to(torch.bfloat16)
    scaled = (x.float() * one_plus_scale.float()).to(torch.bfloat16)
    return (scaled.float() + gathered_shift.float()).to(torch.bfloat16)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_indexed_scale_shift_cpu_exact_bf16_parity(index_dtype):
    rows, hidden, table_rows = 257, 5376, 19
    x = torch.randn(rows, hidden, dtype=torch.bfloat16)
    shift = _make_chunk_view(table_rows, hidden, 1)
    scale = _make_chunk_view(table_rows, hidden, 4)
    indices = torch.randint(0, table_rows, (rows,), dtype=index_dtype)
    assert can_use_indexed_scale_shift_bf16_cpu(x, shift, scale, indices)

    out = x.clone()
    result = indexed_scale_shift_bf16_(out, shift, scale, indices)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.view(torch.int16), _reference(x, shift, scale, indices).view(torch.int16))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_indexed_scale_shift_cpu_zero_rows_noop(index_dtype):
    hidden = 5376
    x = torch.empty((0, hidden), dtype=torch.bfloat16)
    shift = _make_chunk_view(4, hidden, 0)
    scale = _make_chunk_view(4, hidden, 2)
    indices = torch.empty((0,), dtype=index_dtype)
    result = indexed_scale_shift_bf16_(x, shift, scale, indices)
    assert result.data_ptr() == x.data_ptr()
    assert result.shape == x.shape


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_indexed_scale_shift_cpu_out_of_range_raises(index_dtype):
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    shift = _make_chunk_view(3, 64, 0)
    scale = _make_chunk_view(3, 64, 1)
    indices = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0], dtype=index_dtype)
    with pytest.raises(RuntimeError, match=r"indices\[6\].*out of range"):
        torch.ops.sgl_kernel.indexed_scale_shift_bf16_(x, shift, scale, indices)


def test_indexed_scale_shift_cpu_rejects_strided_table():
    x = torch.randn(33, 96, dtype=torch.bfloat16)
    shift = torch.randn(7, 192, dtype=torch.bfloat16)[:, ::2]
    scale = torch.randn(7, 192, dtype=torch.bfloat16)[:, ::2]
    indices = torch.randint(0, 7, (33,), dtype=torch.int64)
    assert not can_use_indexed_scale_shift_bf16_cpu(x, shift, scale, indices)
    expected = _reference(x, shift, scale, indices)
    result = indexed_scale_shift_bf16_(x, shift, scale, indices)
    assert result.data_ptr() == x.data_ptr()
    torch.testing.assert_close(x, expected, atol=0, rtol=0)