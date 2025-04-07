import pytest
import torch
from sgl_kernel import cublas_grouped_gemm


def torch_grouped_gemm(a_array, b_array, out_dtype):
    return [torch.matmul(a, b.t()).to(out_dtype) for a, b in zip(a_array, b_array)]


skip_condition = not torch.cuda.is_available() or (
    torch.version.cuda is None
    or tuple(map(int, torch.version.cuda.split("."))) < (12, 5)
)


@pytest.mark.skipif(
    skip_condition, reason="CUDA not available or CUDA version lower than 12.5"
)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 16, 32, 256, 1024])
@pytest.mark.parametrize("N", [2, 16, 128, 256, 4096])
@pytest.mark.parametrize("K", [3, 16, 32, 512, 8192])
def test_grouped_gemm_accuracy(out_dtype, M, N, K):
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


if __name__ == "__main__":
    pytest.main([__file__])
