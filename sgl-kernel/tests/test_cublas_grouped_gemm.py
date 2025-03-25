import pytest
import torch
from sgl_kernel import cublas_grouped_gemm


def torch_grouped_gemm(a_array, b_array, out_dtype):
    c_array = []
    for a, b in zip(a_array, b_array):
        c_array.append(torch.matmul(a, b.t()).to(out_dtype))
    return c_array


# skip if CUDA is not available or CUDA < 12.5
skip_condition = not torch.cuda.is_available() or (
    torch.version.cuda is None
    or tuple(map(int, torch.version.cuda.split("."))) < (12, 5)
)


@pytest.mark.skipif(
    skip_condition, reason="CUDA not available or CUDA version lower than 12.5"
)
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16])
def test_grouped_gemm_accuracy(out_dtype):
    Ms = [1, 16, 32, 256, 1024]
    Ns = [2, 16, 128, 256, 4096]
    Ks = [3, 16, 32, 512, 8192]
    group_count = len(Ms)

    a_array = []
    b_array = []
    c_array_cublas = []

    for i in range(group_count):
        M, N, K = Ms[i], Ns[i], Ks[i]
        a_array.append(torch.randn((M, K), device="cuda", dtype=out_dtype) * 5)
        b_array.append(torch.randn((N, K), device="cuda", dtype=out_dtype) * 5)
        c_array_cublas.append(torch.empty((M, N), device="cuda", dtype=out_dtype))

    c_array_torch = torch_grouped_gemm(a_array, b_array, out_dtype)
    cublas_grouped_gemm(a_array, b_array, c_array_cublas, out_dtype)

    for i in range(group_count):
        torch.testing.assert_close(c_array_torch[i], c_array_cublas[i])
        print(f"M={Ms[i]}, N={Ns[i]}, K={Ks[i]}, out_dtype={out_dtype}: OK")


if __name__ == "__main__":
    pytest.main([__file__])
