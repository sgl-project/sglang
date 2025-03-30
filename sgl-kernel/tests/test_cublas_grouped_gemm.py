import pytest
import torch
from sgl_kernel import cublas_grouped_gemm


def torch_grouped_gemm(a_array, b_array, out_dtype):
    return [torch.matmul(a, b.t()).to(out_dtype) for a, b in zip(a_array, b_array)]


skip_condition = not torch.cuda.is_available() or (
    torch.version.cuda is None
    or tuple(map(int, torch.version.cuda.split("."))) < (12, 5)
)


m_values = [1, 16, 32, 256, 1024]
n_values = [2, 16, 128, 256, 4096]
k_values = [3, 16, 32, 512, 8192]


@pytest.mark.skipif(
    skip_condition, reason="CUDA not available or CUDA version lower than 12.5"
)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", m_values)
@pytest.mark.parametrize("N", n_values)
@pytest.mark.parametrize("K", k_values)
def test_grouped_gemm_accuracy(out_dtype, M, N, K):
    try:
        a = torch.randn((M, K), device="cuda", dtype=out_dtype) * 5
        b = torch.randn((N, K), device="cuda", dtype=out_dtype) * 5
        expected = torch.matmul(a, b.t()).to(out_dtype)

        a_array = [a]
        b_array = [b]
        c_array = [torch.empty((M, N), device="cuda", dtype=out_dtype)]

        result_torch = torch_grouped_gemm(a_array, b_array, out_dtype)[0]
        cublas_grouped_gemm(a_array, b_array, c_array, out_dtype)

        torch.testing.assert_close(result_torch, expected)
        torch.testing.assert_close(c_array[0], expected)

    except torch.cuda.OutOfMemoryError:
        pytest.skip(f"Skipping M={M}, N={N}, K={K} due to OOM")


if __name__ == "__main__":
    pytest.main([__file__])
