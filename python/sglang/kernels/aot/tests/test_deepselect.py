"""Correctness tests for the DeepSelect AOT operator."""

import inspect

import pytest
import torch


def _has_deepselect() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from sgl_kernel import is_deepselect_supported
    except ImportError:
        return False
    return is_deepselect_supported()


pytestmark = pytest.mark.skipif(
    not _has_deepselect(), reason="requires a compatible sgl-kernel DeepSelect build"
)


def _aligned_input(rows: int, width: int, dtype: torch.dtype) -> torch.Tensor:
    alignment = 1024 // dtype.itemsize
    padded_width = (width + alignment - 1) // alignment * alignment
    return torch.randn((rows, padded_width), dtype=dtype, device="cuda")[:, :width]


def test_deepselect_reports_current_architecture():
    from sgl_kernel import (
        get_deepselect_supported_architectures,
        is_deepselect_supported,
    )

    assert get_deepselect_supported_architectures()
    assert is_deepselect_supported()


def test_deepselect_rejects_non_cuda_device():
    from sgl_kernel import is_deepselect_supported

    assert not is_deepselect_supported(torch.device("cpu"))


def test_deepselect_topk_matches_official_signature():
    from sgl_kernel.deepselect import topk

    assert list(inspect.signature(topk).parameters) == [
        "input",
        "topk",
        "sorted",
        "begin",
        "end",
        "indices_type",
        "sorted_index",
        "hint",
        "output_idx",
        "output_idx_offset",
        "idx_oob_fill_value",
        "value_oob_fill_value",
        "return_value",
        "abort_when_nan_found",
    ]


@pytest.mark.parametrize(
    "case,error",
    [
        pytest.param("dtype", "input dtype must be", id="dtype"),
        pytest.param("dimension", "input must be a 2D tensor", id="dimension"),
        pytest.param("topk", "topk must be in", id="topk"),
        pytest.param("begin", "`begin` is not supported", id="begin"),
        pytest.param("index-dtype", "output_index dtype must be", id="index-dtype"),
    ],
)
def test_deepselect_validation_is_owned_by_cpp(case, error):
    from sgl_kernel.deepselect import topk

    input = _aligned_input(1, 1024, torch.float32)
    kwargs = {}
    if case == "dtype":
        input = _aligned_input(1, 1024, torch.float16)
    elif case == "dimension":
        input = input[0]
    elif case == "topk":
        kwargs["topk"] = 0
    elif case == "begin":
        kwargs["begin"] = torch.zeros(1, dtype=torch.int32, device="cuda")
    elif case == "index-dtype":
        kwargs["indices_type"] = torch.float16

    with pytest.raises(RuntimeError, match=error):
        topk(input, kwargs.pop("topk", 512), **kwargs)


@pytest.mark.parametrize(
    "dtype,index_dtype,rows,width,topk",
    [
        pytest.param(torch.float32, torch.int32, 3, 16385, 512, id="fp32-i32-512"),
        pytest.param(torch.float32, torch.int64, 3, 32771, 513, id="fp32-i64-1024"),
        pytest.param(torch.bfloat16, torch.int32, 3, 32771, 513, id="bf16-i32-1024"),
        pytest.param(
            torch.bfloat16,
            torch.int64,
            1,
            131073,
            1024,
            id="bf16-i64-cluster",
        ),
        pytest.param(torch.bfloat16, torch.int32, 2, 65539, 2048, id="bf16-i32-4096"),
    ],
)
def test_deepselect_topk_varlen_matches_torch(dtype, index_dtype, rows, width, topk):
    """BF16/FP32 rows honor the per-row exclusive end position."""
    from sgl_kernel.deepselect import topk as deepselect_topk

    torch.manual_seed(913)
    input = _aligned_input(rows, width, dtype)
    length_candidates = [max(0, topk - 7), min(width, topk + 11), width - 3]
    if rows == 1:
        length_candidates = [width - 3]
    lengths = torch.tensor(
        [length_candidates[i % len(length_candidates)] for i in range(rows)],
        dtype=torch.int32,
        device="cuda",
    )
    values, indices = deepselect_topk(
        input, topk, end=lengths, indices_type=index_dtype
    )

    assert values is not None
    assert values.dtype == dtype
    assert indices.dtype == index_dtype
    assert values.stride(0) * values.element_size() % 32 == 0
    assert indices.stride(0) * indices.element_size() % 32 == 0
    for row, length in enumerate(lengths.tolist()):
        selected = min(length, topk)
        row_indices = indices[row, :selected].long()
        assert torch.all((row_indices >= 0) & (row_indices < length))
        torch.testing.assert_close(
            input[row].gather(0, row_indices), values[row, :selected], rtol=0, atol=0
        )
        expected = torch.topk(input[row, :length], selected, sorted=False).values
        torch.testing.assert_close(
            values[row, :selected].sort().values,
            expected.sort().values,
            rtol=0,
            atol=0,
        )
        if selected < topk:
            assert torch.all(torch.isneginf(values[row, selected:]))
            assert torch.all(indices[row, selected:] == 2147483647)


def test_deepselect_topk_indices_only_with_offsets():
    from sgl_kernel.deepselect import topk

    input = _aligned_input(2, 16385, torch.bfloat16)
    lengths = torch.tensor([500, 16380], dtype=torch.int32, device="cuda")
    offsets = torch.tensor([1000, 2000], dtype=torch.int32, device="cuda")
    values, indices = topk(
        input,
        512,
        end=lengths,
        indices_type=torch.int32,
        sorted_index=True,
        output_idx_offset=offsets,
        idx_oob_fill_value=-1,
        return_value=False,
    )

    assert values is None
    for row, length in enumerate(lengths.tolist()):
        selected = min(length, 512)
        valid_indices = indices[row, :selected] - offsets[row]
        assert torch.all(valid_indices[:-1] <= valid_indices[1:])
        assert torch.all((valid_indices >= 0) & (valid_indices < length))
        if selected < 512:
            assert torch.all(indices[row, selected:] == -1)


def test_deepselect_topk_sorted_fp32():
    from sgl_kernel.deepselect import topk

    input = _aligned_input(2, 16385, torch.float32)
    values, indices = topk(input, 512, sorted=True)
    assert values is not None
    assert torch.all(values[:, :-1] >= values[:, 1:])
    torch.testing.assert_close(input.gather(1, indices), values, rtol=0, atol=0)


def test_deepselect_topk_empty_batch():
    """CUDA-graph padding can produce zero rows; this must be a no-op, not a zero-grid launch."""
    from sgl_kernel.deepselect import topk

    input = torch.empty((0, 1024), dtype=torch.bfloat16, device="cuda")
    values, indices = topk(input, 512)
    assert values is not None
    assert values.shape == (0, 512)
    assert indices.shape == (0, 512)
