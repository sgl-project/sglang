import itertools
import unittest

import torch

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.quantization.fp8_kernel import (
    per_tensor_quant_mla_fp8,
    per_token_group_quant_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
    static_quant_fp8,
    w8a8_block_fp8_matmul,
)
from sglang.srt.layers.quantization.fp8_utils import input_to_float8
from sglang.test.test_utils import CustomTestCase

_is_cuda = torch.cuda.is_available() and torch.version.cuda


# For test
def native_per_token_group_quant_fp8(
    x, group_size, eps=1e-10, dtype=torch.float8_e4m3fn
):
    """Function to perform per-token-group quantization on an input tensor `x` using native torch.

    It converts the tensor values into float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    Note that only `torch.float8_e4m3fn` is supported for now.
    """
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


class TestPerTokenGroupQuantFP8(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16, torch.float32]
    NUM_TOKENS = [7, 83, 2048]
    D = [512, 4096, 5120, 13824]
    GROUP_SIZE = [64, 128, 256, 512]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _per_token_group_quant_fp8(self, num_tokens, d, dtype, group_size, seed):
        torch.manual_seed(seed)

        x = torch.rand(num_tokens, d, dtype=dtype)

        with torch.inference_mode():
            ref_out, ref_scale = native_per_token_group_quant_fp8(x, group_size)
            out, scale = per_token_group_quant_fp8(x, group_size)

        self.assertTrue(
            torch.allclose(out.to(torch.float32), ref_out.to(torch.float32), rtol=0.20)
        )
        self.assertTrue(torch.allclose(scale, ref_scale))

    def test_per_token_group_quant_fp8(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.D,
            self.DTYPES,
            self.GROUP_SIZE,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                d=params[1],
                dtype=params[2],
                group_size=params[3],
                seed=params[4],
            ):
                self._per_token_group_quant_fp8(*params)


# For test
def native_static_quant_fp8(x, x_s, dtype=torch.float8_e4m3fn):
    """Function to perform static quantization on an input tensor `x` using native torch.

    It converts the tensor values into float8 values and returns the
    quantized tensor along with the scaling factor used for quantization.
    """
    assert x.is_contiguous(), "`x` is not contiguous"
    assert x_s.numel() == 1, "only supports per-tensor scale"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // x.shape[-1], x.shape[-1])
    x_s_inv = 1.0 / x_s
    x_q = (x_ * x_s_inv).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)

    return x_q, x_s


class TestStaticQuantFP8(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16, torch.float32]
    NUM_TOKENS = [7, 83, 2048]
    D = [512, 4096, 5120, 13824]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _static_quant_fp8(self, num_tokens, d, dtype, seed):
        torch.manual_seed(seed)

        x = torch.rand(num_tokens, d, dtype=dtype)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        x_s = x.max() / fp8_max

        with torch.inference_mode():
            ref_out, _ = native_static_quant_fp8(x, x_s)
            out, _ = static_quant_fp8(x, x_s, repeat_scale=True)

        self.assertTrue(
            torch.allclose(out.to(torch.float32), ref_out.to(torch.float32), rtol=0.50)
        )

    def test_static_quant_fp8(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.D,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                d=params[1],
                dtype=params[2],
                seed=params[3],
            ):
                self._static_quant_fp8(*params)


class TestPerTensorQuantMlaFP8(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16, torch.float32]
    NUM_TOKENS = [7, 83, 2048]
    D = [512, 4096, 5120, 13824]
    LAST_D_EXT = [1024, 0]
    LAST_D = [512]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _per_tensor_quant_mla_fp8(self, num_tokens, d, last_d_ext, last_d, dtype, seed):
        torch.manual_seed(seed)

        x = torch.rand(
            (num_tokens, d // last_d, last_d + last_d_ext),
            dtype=dtype,
        )
        x_sub, _ = x.split([last_d, last_d_ext], dim=-1)

        with torch.inference_mode():
            ref_out, ref_s = input_to_float8(x_sub.transpose(0, 1))
            out, out_s = per_tensor_quant_mla_fp8(x_sub.transpose(0, 1))

        self.assertTrue(out.is_contiguous())
        self.assertTrue(
            torch.allclose(out.to(torch.float32), ref_out.to(torch.float32), rtol=0.50)
        )
        self.assertTrue(
            torch.allclose(out_s.to(torch.float32), ref_s.to(torch.float32))
        )

    def test_per_tensor_quant_mla_fp8(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.D,
            self.LAST_D_EXT,
            self.LAST_D,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                d=params[1],
                last_d_ext=params[2],
                last_d=params[3],
                dtype=params[4],
                seed=params[5],
            ):
                self._per_tensor_quant_mla_fp8(*params)


class TestPerTokenGroupQuantMlaDeepGemmMaskedFP8(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16, torch.float32]
    B = [128]
    NUM_TOKENS = [7, 83, 2048, 1024 * 16]
    D = [512, 128]
    GROUP_SIZE = [128]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _per_token_group_quant_mla_deep_gemm_masked_fp8(
        self, b, num_tokens, d, dtype, group_size, seed
    ):
        torch.manual_seed(seed)

        x = torch.rand(b, num_tokens, d, dtype=dtype)

        with torch.inference_mode():
            ref_out, ref_scale = native_per_token_group_quant_fp8(x, group_size, 1e-12)
            out, scale, _, _, _ = per_token_group_quant_mla_deep_gemm_masked_fp8(
                x, group_size
            )
            out = out[:, :num_tokens, :]
            scale = scale[:, :num_tokens, :]

        self.assertTrue(
            torch.allclose(
                out.to(torch.float32), ref_out.to(torch.float32), rtol=0.20, atol=1e-2
            )
        )
        self.assertTrue(torch.allclose(scale, ref_scale))

    def test_per_token_group_quant_mla_deep_gemm_masked_fp8(self):
        for params in itertools.product(
            self.B,
            self.NUM_TOKENS,
            self.D,
            self.DTYPES,
            self.GROUP_SIZE,
            self.SEEDS,
        ):
            with self.subTest(
                b=params[0],
                num_tokens=params[1],
                d=params[2],
                dtype=params[3],
                group_size=params[4],
                seed=params[5],
            ):
                self._per_token_group_quant_mla_deep_gemm_masked_fp8(*params)


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


class TestW8A8BlockFP8Matmul(CustomTestCase):

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
            (3072, 1536),
            (24576, 7168),
            (4096, 512),
            (7168, 2048),
            (4608, 7168),
            (512, 7168),
            (7168, 2304),
            (7168, 512),
        ]
        BLOCK_SIZE = [[128, 128]]
        SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _w8a8_block_fp8_matmul(self, M, NK, block_size, out_dtype, seed):
        N, K = NK
        torch.manual_seed(seed)
        # NOTE(HandH1998): to avoid overflow when out_dtype = torch.half
        factor_for_scale = 1e-2
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
        A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
        B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        block_n, block_k = block_size[0], block_size[1]
        n_tiles = (N + block_n - 1) // block_n
        k_tiles = (K + block_k - 1) // block_k

        As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
        Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale

        with torch.inference_mode():
            ref_out = native_w8a8_block_fp8_matmul(
                A_fp8, B_fp8, As, Bs, block_size, out_dtype
            )
            out = w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

        self.assertTrue(
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / torch.mean(torch.abs(ref_out.to(torch.float32)))
            < 0.001
        )

    def test_w8a8_block_fp8_matmul(self):
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
                self._w8a8_block_fp8_matmul(*params)


# For test
def torch_w8a8_block_fp8_moe(a, w1, w2, w1_s, w2_s, score, topk, block_shape):
    """This function performs fused moe with block-wise quantization using native torch."""

    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)

    _, block_k = block_shape[0], block_shape[1]
    a_q, a_s = native_per_token_group_quant_fp8(a, block_k)
    # NOTE(HandH1998): Since "index_cuda" not implemented for 'Float8_e4m3fn', we need to cast `float8`` to `float32``.
    a_q = a_q.to(torch.float32)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            inter_out = native_w8a8_block_fp8_matmul(
                a_q[mask], w1[i], a_s[mask], w1_s[i], block_shape, output_dtype=a.dtype
            )
            act_out = SiluAndMul().forward_native(inter_out)
            act_out_q, act_out_s = native_per_token_group_quant_fp8(act_out, block_k)
            act_out = act_out.to(torch.float32)
            out[mask] = native_w8a8_block_fp8_matmul(
                act_out_q, w2[i], act_out_s, w2_s[i], block_shape, output_dtype=a.dtype
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


class TestW8A8BlockFP8FusedMoE(CustomTestCase):
    DTYPES = [torch.float32, torch.half, torch.bfloat16]
    M = [1, 33, 64, 222, 1024 * 128]
    N = [128, 1024, 2048]
    K = [256, 4096, 5120]
    E = [8, 24]
    TOP_KS = [2, 6]
    BLOCK_SIZE = [[64, 64], [64, 128], [128, 64], [128, 128]]
    # BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _w8a8_block_fp8_fused_moe(self, M, N, K, E, topk, block_size, dtype, seed):
        torch.manual_seed(seed)
        # NOTE(HandH1998): to avoid overflow when out_dtype = torch.half
        factor_for_scale = 1e-2
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        a = torch.randn((M, K), dtype=dtype) / 10

        w1_fp32 = (torch.rand((E, 2 * N, K), dtype=torch.float32) - 0.5) * 2 * fp8_max
        w1 = w1_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w2_fp32 = (torch.rand((E, K, N), dtype=torch.float32) - 0.5) * 2 * fp8_max
        w2 = w2_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        block_n, block_k = block_size[0], block_size[1]
        n_tiles_w1 = (2 * N + block_n - 1) // block_n
        n_tiles_w2 = (K + block_n - 1) // block_n
        k_tiles_w1 = (K + block_k - 1) // block_k
        k_tiles_w2 = (N + block_k - 1) // block_k

        w1_s = (
            torch.rand((E, n_tiles_w1, k_tiles_w1), dtype=torch.float32)
            * factor_for_scale
        )
        w2_s = (
            torch.rand((E, n_tiles_w2, k_tiles_w2), dtype=torch.float32)
            * factor_for_scale
        )

        score = torch.randn((M, E), dtype=dtype)

        with torch.inference_mode():
            ref_out = torch_w8a8_block_fp8_moe(
                a, w1, w2, w1_s, w2_s, score, topk, block_size
            )
            topk_output = select_experts(
                hidden_states=a,
                router_logits=score,
                topk_config=TopKConfig(top_k=topk, renormalize=False),
            )
            out = fused_moe(
                a,
                w1,
                w2,
                topk_output,
                use_fp8_w8a8=True,
                w1_scale=w1_s,
                w2_scale=w2_s,
                block_shape=block_size,
            )

        self.assertTrue(
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / torch.mean(torch.abs(ref_out.to(torch.float32)))
            < 0.02
        )

    def test_w8a8_block_fp8_fused_moe(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.TOP_KS,
            self.BLOCK_SIZE,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                E=params[3],
                topk=params[4],
                block_size=params[5],
                dtype=params[6],
                seed=params[7],
            ):
                self._w8a8_block_fp8_fused_moe(*params)


# For test
def torch_w8a8_block_fp8_bmm(a, a_s, w, w_s, block_shape, out_dtype):
    """This function performs bmm with block-wise quantization using native torch."""

    B, N, _ = w.shape
    _, M, _ = a.shape
    out = torch.empty((B, M, N), dtype=out_dtype, device=a.device)

    for i in range(B):
        out[i] = native_w8a8_block_fp8_matmul(
            a[i], w[i], a_s[i], w_s[i], block_shape, output_dtype=out_dtype
        )

    return out


class TestW8A8BlockFP8BatchedDeepGemm(CustomTestCase):
    DTYPES = [torch.bfloat16]
    M = [1, 33, 64, 222, 8192]
    N = [128, 512]
    K = [128, 512]
    BATCH = [128]
    BLOCK_SIZE = [[128, 128]]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        try:
            import deep_gemm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("DeepGEMM is not available")
        torch.set_default_device("cuda")

    def _w8a8_block_fp8_batched_deep_gemm(self, M, N, K, B, block_size, dtype, seed):
        torch.manual_seed(seed)
        factor_for_scale = 1e-2
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        a_fp32 = torch.randn((B, M, K), dtype=torch.float32) / 10
        a = a_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        w_fp32 = (torch.rand((B, N, K), dtype=torch.float32) - 0.5) * 2 * fp8_max
        w = w_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        block_n, block_k = block_size[0], block_size[1]
        n_tiles_w = (N + block_n - 1) // block_n
        k_tiles_w = (K + block_k - 1) // block_k

        w_s = (
            torch.rand((B, n_tiles_w, k_tiles_w), dtype=torch.float32)
            * factor_for_scale
        )
        a_s = torch.rand((B, M, k_tiles_w), dtype=torch.float32) * factor_for_scale

        ae = a.new_empty(B, (M + 255) // 256 * 256, K)
        ae_s = a_s.new_empty(B, (M + 255) // 256 * 256, k_tiles_w)
        oe = torch.empty((B, (M + 255) // 256 * 256, N), dtype=dtype)
        ae[:, :M, :] = a
        ae_s[:, :M, :] = a_s

        masked_m = torch.full((B,), M, dtype=torch.int)
        expected_m = M
        lhs = (
            ae,
            ae_s,
        )
        rhs = (
            w,
            w_s,
        )

        from deep_gemm import fp8_m_grouped_gemm_nt_masked

        with torch.inference_mode():
            ref_out = torch_w8a8_block_fp8_bmm(a, a_s, w, w_s, block_size, dtype)
            fp8_m_grouped_gemm_nt_masked(lhs, rhs, oe, masked_m, expected_m)
            out = oe[:, :M, :]

        self.assertTrue(
            torch.mean(torch.abs(out.to(torch.float32) - ref_out.to(torch.float32)))
            / torch.mean(torch.abs(ref_out.to(torch.float32)))
            < 0.0001
        )

    def test_w8a8_block_fp8_batched_deep_gemm(self):

        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.BATCH,
            self.BLOCK_SIZE,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                B=params[3],
                block_size=params[4],
                dtype=params[5],
                seed=params[6],
            ):
                self._w8a8_block_fp8_batched_deep_gemm(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
