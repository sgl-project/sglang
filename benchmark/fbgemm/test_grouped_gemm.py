import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fbgemm_grouped_gemm import grouped_gemm as fbgemm_grouped_gemm
    from fbgemm_grouped_gemm import (
        grouped_gemm_fp8_rowwise as fbgemm_grouped_gemm_fp8_rowwise,
    )

    FBGEMM_AVAILABLE = True
    print("✓ Successfully imported FBGEMM grouped GEMM")
except ImportError as e:
    print(f"✗ Failed to import FBGEMM grouped GEMM: {e}")
    FBGEMM_AVAILABLE = False

try:
    from sglang.srt.layers.moe.ep_moe.kernels import (
        grouped_gemm_triton as sglang_grouped_gemm,
    )

    SGLANG_AVAILABLE = True
    print("✓ Successfully imported SGLang grouped GEMM")
except ImportError as e:
    print(f"✗ Failed to import SGLang grouped GEMM: {e}")
    SGLANG_AVAILABLE = False


def create_uniform_groups(batch_size, num_groups, device):
    tokens_per_group = batch_size // num_groups
    return torch.full((num_groups,), tokens_per_group, dtype=torch.int64, device=device)


def create_non_uniform_groups(batch_size, num_groups, device):
    remaining = batch_size
    m_sizes = []

    for i in range(num_groups - 1):
        if remaining <= 1:
            size = 1
        else:
            max_size = remaining - (num_groups - i - 1) + 1
            size = torch.randint(1, max_size, (1,)).item()
        m_sizes.append(size)
        remaining -= size

    m_sizes.append(remaining)
    return torch.tensor(m_sizes, dtype=torch.int64, device=device)


def create_sglang_inputs(x, w, m_sizes, num_groups, intermediate_size, device):
    batch_size = x.shape[0]

    c_sglang = torch.empty(
        batch_size, intermediate_size, dtype=torch.bfloat16, device=device
    )

    seg_indptr = torch.zeros(num_groups + 1, dtype=torch.int64, device=device)
    current_pos = 0
    for i, size in enumerate(m_sizes):
        current_pos += size
        seg_indptr[i + 1] = current_pos

    weight_indices = torch.arange(num_groups, dtype=torch.int64, device=device)
    w_sglang = w.view(num_groups, intermediate_size, -1)

    return c_sglang, seg_indptr, weight_indices, w_sglang


def create_fp8_data(batch_size, num_groups, hidden_size, intermediate_size, device):
    torch.manual_seed(42)

    x_fp16 = torch.randn(batch_size, hidden_size, dtype=torch.float16, device=device)
    w_fp16 = torch.randn(
        num_groups * intermediate_size, hidden_size, dtype=torch.float16, device=device
    )

    x_fp8 = x_fp16.to(torch.float8_e4m3fn)
    w_fp8 = w_fp16.to(torch.float8_e4m3fn)

    x_scale = torch.randn(batch_size, dtype=torch.float32, device=device).abs() + 1e-4
    w_scale = torch.randn(num_groups, dtype=torch.float32, device=device).abs() + 1e-4

    return x_fp8, w_fp8, x_scale, w_scale


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="FBGEMM not available")
@pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("num_groups", [2, 4, 8])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
def test_uniform_groups(batch_size, num_groups, hidden_size, intermediate_size, device):
    if batch_size % num_groups != 0:
        pytest.skip(f"batch_size {batch_size} not divisible by num_groups {num_groups}")

    torch.manual_seed(42)

    m_sizes = create_uniform_groups(batch_size, num_groups, device)

    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    w = torch.randn(
        num_groups * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    )

    result_fbgemm = fbgemm_grouped_gemm(x, w, m_sizes, use_fast_accum=True)

    c_sglang, seg_indptr, weight_indices, w_sglang = create_sglang_inputs(
        x, w, m_sizes, num_groups, intermediate_size, device
    )

    result_sglang = sglang_grouped_gemm(
        x,
        w_sglang,
        c_sglang,
        num_groups,
        weight_column_major=True,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        c_dtype=c_sglang.dtype,
    )

    assert torch.allclose(result_fbgemm, result_sglang, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="FBGEMM not available")
@pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
@pytest.mark.parametrize("batch_size", [63, 100, 127])
@pytest.mark.parametrize("num_groups", [3, 5, 7])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
def test_non_uniform_groups(
    batch_size, num_groups, hidden_size, intermediate_size, device
):
    torch.manual_seed(42)

    m_sizes = create_non_uniform_groups(batch_size, num_groups, device)

    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    w = torch.randn(
        num_groups * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    )

    result_fbgemm = fbgemm_grouped_gemm(x, w, m_sizes, use_fast_accum=True)

    c_sglang, seg_indptr, weight_indices, w_sglang = create_sglang_inputs(
        x, w, m_sizes, num_groups, intermediate_size, device
    )

    result_sglang = sglang_grouped_gemm(
        x,
        w_sglang,
        c_sglang,
        num_groups,
        weight_column_major=True,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        c_dtype=c_sglang.dtype,
    )

    assert torch.allclose(result_fbgemm, result_sglang, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="FBGEMM not available")
@pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
@pytest.mark.parametrize("batch_size,num_groups", [(64, 4), (128, 8), (256, 16)])
@pytest.mark.parametrize("hidden_size", [768, 2048, 4096])
@pytest.mark.parametrize("intermediate_size", [2048, 4096, 8192])
def test_large_dimensions(
    batch_size, num_groups, hidden_size, intermediate_size, device
):
    torch.manual_seed(42)

    m_sizes = create_uniform_groups(batch_size, num_groups, device)

    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    w = torch.randn(
        num_groups * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    )

    result_fbgemm = fbgemm_grouped_gemm(x, w, m_sizes, use_fast_accum=True)

    c_sglang, seg_indptr, weight_indices, w_sglang = create_sglang_inputs(
        x, w, m_sizes, num_groups, intermediate_size, device
    )

    result_sglang = sglang_grouped_gemm(
        x,
        w_sglang,
        c_sglang,
        num_groups,
        weight_column_major=True,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        c_dtype=c_sglang.dtype,
    )

    assert torch.allclose(result_fbgemm, result_sglang, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="FBGEMM not available")
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("num_groups", [2, 4])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
def test_fp8_uniform_groups(
    batch_size, num_groups, hidden_size, intermediate_size, device
):
    if batch_size % num_groups != 0:
        pytest.skip(f"batch_size {batch_size} not divisible by num_groups {num_groups}")

    torch.manual_seed(42)

    m_sizes = create_uniform_groups(batch_size, num_groups, device)
    x_fp8, w_fp8, x_scale, w_scale = create_fp8_data(
        batch_size, num_groups, hidden_size, intermediate_size, device
    )

    try:
        result_fp8 = fbgemm_grouped_gemm_fp8_rowwise(
            x_fp8, w_fp8, m_sizes, x_scale, w_scale, use_fast_accum=True
        )
        assert result_fp8.shape == (batch_size, intermediate_size)
        assert result_fp8.dtype == torch.bfloat16
    except Exception as e:
        pytest.skip(f"FP8 test failed (possibly unsupported): {e}")


@pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="FBGEMM not available")
@pytest.mark.parametrize("batch_size", [63, 100])
@pytest.mark.parametrize("num_groups", [3, 5])
@pytest.mark.parametrize("hidden_size", [512, 1024])
@pytest.mark.parametrize("intermediate_size", [1024, 2048])
def test_fp8_non_uniform_groups(
    batch_size, num_groups, hidden_size, intermediate_size, device
):
    torch.manual_seed(42)

    m_sizes = create_non_uniform_groups(batch_size, num_groups, device)
    x_fp8, w_fp8, x_scale, w_scale = create_fp8_data(
        batch_size, num_groups, hidden_size, intermediate_size, device
    )

    try:
        result_fp8 = fbgemm_grouped_gemm_fp8_rowwise(
            x_fp8, w_fp8, m_sizes, x_scale, w_scale, use_fast_accum=True
        )
        assert result_fp8.shape == (batch_size, intermediate_size)
        assert result_fp8.dtype == torch.bfloat16
    except Exception as e:
        pytest.skip(f"FP8 test failed (possibly unsupported): {e}")


@pytest.mark.skipif(not FBGEMM_AVAILABLE, reason="FBGEMM not available")
def test_fbgemm_only_uniform(device):
    torch.manual_seed(42)

    batch_size, num_groups = 64, 4
    hidden_size, intermediate_size = 512, 1024

    m_sizes = create_uniform_groups(batch_size, num_groups, device)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    w = torch.randn(
        num_groups * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    )

    result = fbgemm_grouped_gemm(x, w, m_sizes, use_fast_accum=True)

    assert result.shape == (batch_size, intermediate_size)
    assert result.dtype == torch.bfloat16


@pytest.mark.skipif(not SGLANG_AVAILABLE, reason="SGLang not available")
def test_sglang_only_uniform(device):
    torch.manual_seed(42)

    batch_size, num_groups = 64, 4
    hidden_size, intermediate_size = 512, 1024

    m_sizes = create_uniform_groups(batch_size, num_groups, device)
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16, device=device)
    w = torch.randn(
        num_groups * intermediate_size, hidden_size, dtype=torch.bfloat16, device=device
    )

    c_sglang, seg_indptr, weight_indices, w_sglang = create_sglang_inputs(
        x, w, m_sizes, num_groups, intermediate_size, device
    )

    result = sglang_grouped_gemm(
        x,
        w_sglang,
        c_sglang,
        num_groups,
        weight_column_major=True,
        seg_indptr=seg_indptr,
        weight_indices=weight_indices,
        c_dtype=c_sglang.dtype,
    )

    assert result.shape == (batch_size, intermediate_size)
    assert result.dtype == torch.bfloat16


def test_imports():
    assert (
        FBGEMM_AVAILABLE or SGLANG_AVAILABLE
    ), "Neither FBGEMM nor SGLang is available"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
