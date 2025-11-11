import itertools
import unittest
from typing import List, Tuple

import torch
from deep_gemm import fp8_gemm_nt

from sglang.test.test_utils import CustomTestCase

_is_cuda = torch.cuda.is_available() and torch.version.cuda


# Modify form DeepGEMM Blackwell
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def per_token_group_quant_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_quant_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def ceil_to_ue8m0(x: torch.Tensor):
    assert x.view(-1).amax().item() > 0
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_token_group_quant_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    return (x_view * (1.0 / sf.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), sf


def per_block_quant_mxfp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


# For test
def native_w8a8_block_fp8_matmul(A, B, As, Bs, block_size, output_dtype=torch.float16):
    """This function performs matrix multiplication with block-wise quantization using native torch.

    It takes two input tensors `A` and `B` with scales `As` and `Bs`.
    The output is returned in the specified `output_dtype`.
    """

    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]
    assert (A.shape[-1] + block_k - 1) // block_k == As.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, A.shape[-1])
    As = As.reshape(M, As.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [A[:, i * block_k : min((i + 1) * block_k, K)] for i in range(k_tiles)]
    B_tiles = [
        [
            B[
                j * block_n : min((j + 1) * block_n, N),
                i * block_k : min((i + 1) * block_k, K),
            ]
            for i in range(k_tiles)
        ]
        for j in range(n_tiles)
    ]
    C_tiles = [C[:, j * block_n : min((j + 1) * block_n, N)] for j in range(n_tiles)]
    As_tiles = [As[:, i : i + 1] for i in range(k_tiles)]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = As_tiles[i] * Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


def block_quant_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """This function converts block-wise quantization to unquantized.
    The inputs are block-wise quantization tensor `x_q_block`, block-wise quantization scale
    and the block size.
    The output is an unquantized tensor with dtype.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = torch.empty_like(x_q_block, dtype=dtype)

    for j in range(n_tiles):
        for i in range(k_tiles):
            x_q_block_tile = x_q_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            x_dq_block_tile = x_dq_block[
                j * block_n : min((j + 1) * block_n, n),
                i * block_k : min((i + 1) * block_k, k),
            ]
            x_dq_block_tile[:, :] = x_q_block_tile.to(torch.float32) * x_s[j][i]

    return x_dq_block


class TestDeepGemmBlackwell(CustomTestCase):

    if not _is_cuda:
        OUT_DTYPES = [torch.float32, torch.half, torch.bfloat16]
        M = [1, 7, 83, 512, 2048]
        NKs = [
            (N, K)
            for N in [128, 512, 1024, 4096, 7748, 13824]
            for K in [256, 4096, 5120, 3884, 13824]
        ]
        # BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
        BLOCK_SIZE = [[128, 128]]
        SEEDS = [0]
    else:
        # use practical shape in DeepSeek V3 for test
        OUT_DTYPES = [torch.bfloat16]
        M = [64, 128, 512, 1024, 4096]
        NKs = [
            (2112, 7168),
            (1536, 7168),
            # (3072, 1536),
            # (24576, 7168),
            # (4096, 512),
            # (7168, 2048),
            # (4608, 7168),
            # (512, 7168),
            # (7168, 2304),
            # (7168, 512),
        ]
        BLOCK_SIZE = [[128, 128]]
        SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _test_deep_gemm_blackwell(self, M, NK, block_size, out_dtype, seed):
        N, K = NK
        torch.manual_seed(seed)

        A = torch.empty((M, K), dtype=torch.bfloat16).normal_(0, 0.2)
        B = torch.empty((N, K), dtype=torch.bfloat16).normal_(0, 0.2)

        A_q, A_s = per_token_group_quant_fp8(A)
        B_q, B_s = per_block_quant_fp8(B)

        A_dq = block_quant_dequant(A_q, A_s, [1, block_size[1]], out_dtype)
        B_dq = block_quant_dequant(B_q, B_s, block_size, out_dtype)

        A_qu = per_token_group_quant_mxfp8(A_dq)
        B_qu = per_block_quant_mxfp8(B_dq)
        out = None

        with torch.inference_mode():
            ref_out = native_w8a8_block_fp8_matmul(
                A_q, B_q, A_s, B_s, block_size, out_dtype
            )
            out = torch.empty_like(ref_out)
            fp8_gemm_nt(A_qu, B_qu, out)

        torch.testing.assert_close(out, ref_out, atol=1e-1, rtol=1e-2)

    def test_deep_gemm_blackwell(self):
        for params in itertools.product(
            self.M,
            self.NKs,
            self.BLOCK_SIZE,
            self.OUT_DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                NKs=params[1],
                block_size=params[2],
                out_dtype=params[3],
                seed=params[4],
            ):
                self._test_deep_gemm_blackwell(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
