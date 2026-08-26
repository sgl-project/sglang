import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_ROOT = Path(__file__).resolve().parents[5]
_MODULE_PATH = _ROOT / "python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py"


class _KernelStub:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def __getitem__(self, _grid):
        # Wrapper tests exercise allocation/validation/address contracts.  The
        # actual Triton math is covered by the SM90 kernel test and H20 gate.
        def launch(*args, **kwargs):
            self.calls.append((args, kwargs))

        return launch


def _load_dequant_wrapper_without_triton_runtime():
    triton = types.ModuleType("triton")
    triton.jit = lambda function: _KernelStub(function)
    triton.cdiv = (
        lambda numerator, denominator: (numerator + denominator - 1) // denominator
    )
    language = types.ModuleType("triton.language")
    language.constexpr = object()
    triton.language = language

    previous_triton = sys.modules.get("triton")
    previous_language = sys.modules.get("triton.language")
    sys.modules["triton"] = triton
    sys.modules["triton.language"] = language
    try:
        spec = importlib.util.spec_from_file_location(
            "dsa_dequant_wrapper", _MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_triton is None:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = previous_triton
        if previous_language is None:
            sys.modules.pop("triton.language", None)
        else:
            sys.modules["triton.language"] = previous_language


_MODULE = _load_dequant_wrapper_without_triton_runtime()


def _inputs(num_pool_rows=4, num_selected_rows=2):
    pool = torch.empty(
        (num_pool_rows, 1, 656),
        dtype=torch.float8_e4m3fn,
    )
    page_table = torch.arange(num_selected_rows, dtype=torch.int32)
    return pool, page_table


def test_paged_dequant_returns_the_exact_preallocated_output_buffer():
    pool, page_table = _inputs()
    out = torch.empty((2, 1, 576), dtype=torch.bfloat16)

    result = _MODULE.dequantize_k_cache_paged(pool, page_table, out=out)

    assert result is out
    assert result.data_ptr() == out.data_ptr()


@pytest.mark.parametrize(
    ("out", "match"),
    [
        (torch.empty((3, 1, 576), dtype=torch.bfloat16), "shape"),
        (torch.empty((2, 1, 576), dtype=torch.float16), "dtype"),
        (torch.empty((2, 1, 1152), dtype=torch.bfloat16)[:, :, ::2], "contiguous"),
    ],
)
def test_paged_dequant_rejects_incompatible_output_buffers(out, match):
    pool, page_table = _inputs()

    with pytest.raises(ValueError, match=match):
        _MODULE.dequantize_k_cache_paged(pool, page_table, out=out)


def test_paged_dequant_allocates_when_output_buffer_is_omitted():
    pool, page_table = _inputs()

    result = _MODULE.dequantize_k_cache_paged(pool, page_table)

    assert result.shape == (2, 1, 576)
    assert result.dtype == torch.bfloat16
    assert result.is_contiguous()


def test_paged_dequant_accepts_device_valid_row_count_without_host_slicing():
    pool, page_table = _inputs(num_selected_rows=4)
    out = torch.empty((4, 1, 576), dtype=torch.bfloat16)
    num_valid_rows = torch.tensor([2], dtype=torch.int32)

    result = _MODULE.dequantize_k_cache_paged(
        pool,
        page_table,
        out=out,
        num_valid_rows=num_valid_rows,
    )

    assert result is out
    _, launch_kwargs = _MODULE._dequantize_k_cache_paged_device_extent_kernel.calls[-1]
    assert launch_kwargs["GRID_ROWS"] == page_table.numel()


@pytest.mark.parametrize(
    ("num_valid_rows", "match"),
    [
        (torch.tensor([1], dtype=torch.int64), "dtype"),
        (torch.tensor([1, 2], dtype=torch.int32), "one element"),
    ],
)
def test_paged_dequant_rejects_invalid_device_valid_row_count(num_valid_rows, match):
    pool, page_table = _inputs()

    with pytest.raises(ValueError, match=match):
        _MODULE.dequantize_k_cache_paged(
            pool,
            page_table,
            num_valid_rows=num_valid_rows,
        )
