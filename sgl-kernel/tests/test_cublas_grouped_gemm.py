import unittest

import torch
from sgl_kernel import cublas_grouped_gemm


def torch_grouped_gemm(a_array, b_array, out_dtype):
    c_array = []
    for a, b in zip(a_array, b_array):
        c_array.append(torch.matmul(a, b.t()).to(out_dtype))
    return c_array


class TestGroupedGemm(unittest.TestCase):
    def _test_accuracy(self, Ms, Ns, Ks, out_dtype):
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
            M, N, K = Ms[i], Ns[i], Ks[i]
            torch.testing.assert_close(c_array_torch[i], c_array_cublas[i])
            print(f"M={M}, N={N}, K={K}, out_dtype={out_dtype}: OK")

    def test_accuracy(self):
        Ms = [1, 16, 32, 256, 1024]
        Ns = [2, 16, 128, 256, 4096]
        Ks = [3, 16, 32, 512, 8192]
        out_dtypes = [torch.float16, torch.bfloat16]
        for out_dtype in out_dtypes:
            self._test_accuracy(Ms, Ns, Ks, out_dtype)


if __name__ == "__main__":
    if torch.cuda.is_available():
        cuda_version = tuple(map(int, torch.version.cuda.split(".")))
        if cuda_version >= (12, 5):
            unittest.main()
        else:
            print(f"Cuda version {cuda_version} lower than 12.5, not executing tests.")
