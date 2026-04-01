"""
Unit tests for FP4 GEMM backends used in modelopt quantization.
Note, if you add a new backend for FP4 GEMM, it should be included here.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, suite="stage-b-test-4-gpu-b200")

SM = get_device_sm()
SKIP_REASON = "Test requires CUDA SM 100 or higher"

# Deepseek-R1 TP = 4/8 + DP 4/8
WEIGHT_SHAPES = [
    (1024, 3584),
    (7168, 256),
    (7168, 2304),
    (9216, 3584),
    (512, 3584),
    (7168, 128),
    (7168, 1152),
    (4608, 3584),
]

BATCH_SIZES = [1, 4, 8, 16, 32, 128, 256, 512, 1024, 4096, 8192]


def _run_flashinfer_fp4_gemm(m, n, k, res_dtype, backend):
    from flashinfer import SfLayout, mm_fp4, nvfp4_quantize

    if not mm_fp4.is_backend_supported(backend, SM):
        raise unittest.SkipTest(f"Backend '{backend}' not supported on SM{SM}")
    if backend == "trtllm" and res_dtype == torch.float16:
        raise unittest.SkipTest("trtllm backend does not support float16 output")

    x_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    global_sf_x = (448.0 * 6) / x_bf16.float().abs().nan_to_num().max()
    global_sf_w = (448.0 * 6) / w_bf16.float().abs().nan_to_num().max()

    x_fp4, x_sf = nvfp4_quantize(
        x_bf16, global_sf_x, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    w_fp4, w_sf = nvfp4_quantize(
        w_bf16,
        global_sf_w,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=(backend == "trtllm"),
    )

    alpha = 1.0 / (global_sf_x * global_sf_w)
    reference = torch.mm(x_bf16, w_bf16.T)

    res = torch.empty(m, n, device="cuda", dtype=res_dtype)
    mm_fp4(
        x_fp4,
        w_fp4.T,
        x_sf,
        w_sf.T,
        alpha,
        res_dtype,
        res,
        block_size=16,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=True,
        skip_check=False,
    )

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), res.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.98, (
        f"Cosine similarity {cos_sim:.4f} < 0.98 for "
        f"backend={backend}, m={m}, n={n}, k={k}, dtype={res_dtype}"
    )


def _run_sglang_cutlass_fp4_gemm(m, n, k, res_dtype):
    from sglang.jit_kernel.nvfp4 import cutlass_scaled_fp4_mm, scaled_fp4_quant

    x_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)

    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

    global_sf_x = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / x_bf16.float().abs().amax()
    global_sf_w = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w_bf16.float().abs().amax()
    alpha = 1.0 / (global_sf_x * global_sf_w)

    x_fp4, x_sf = scaled_fp4_quant(x_bf16, global_sf_x)
    w_fp4, w_sf = scaled_fp4_quant(w_bf16, global_sf_w)

    reference = torch.mm(x_bf16, w_bf16.T)

    out = cutlass_scaled_fp4_mm(x_fp4, w_fp4, x_sf, w_sf, alpha, res_dtype)

    cos_sim = F.cosine_similarity(
        reference.reshape(-1).float(), out.reshape(-1).float(), dim=0
    )
    assert cos_sim > 0.98, (
        f"Cosine similarity {cos_sim:.4f} < 0.98 for "
        f"sglang cutlass, m={m}, n={n}, k={k}, dtype={res_dtype}"
    )


@unittest.skipIf(SM < 100, SKIP_REASON)
class TestFP4GemmFlashinferCutlass(unittest.TestCase):

    def test_cutlass(self):
        for n, k in WEIGHT_SHAPES:
            for m in BATCH_SIZES:
                with self.subTest(m=m, n=n, k=k):
                    _run_flashinfer_fp4_gemm(m, n, k, torch.bfloat16, "cutlass")


@unittest.skipIf(SM < 100, SKIP_REASON)
class TestFP4GemmFlashinferCudnn(unittest.TestCase):

    def test_cudnn(self):
        for n, k in WEIGHT_SHAPES:
            for m in BATCH_SIZES:
                with self.subTest(m=m, n=n, k=k):
                    _run_flashinfer_fp4_gemm(m, n, k, torch.bfloat16, "cudnn")


@unittest.skipIf(SM < 100, SKIP_REASON)
class TestFP4GemmFlashinferTrtllm(unittest.TestCase):

    def test_trtllm(self):
        for n, k in WEIGHT_SHAPES:
            for m in BATCH_SIZES:
                with self.subTest(m=m, n=n, k=k):
                    _run_flashinfer_fp4_gemm(m, n, k, torch.bfloat16, "trtllm")


@unittest.skipIf(SM < 100, SKIP_REASON)
class TestFP4GemmFlashinferAuto(unittest.TestCase):

    def test_auto(self):
        for n, k in WEIGHT_SHAPES:
            for m in BATCH_SIZES:
                with self.subTest(m=m, n=n, k=k):
                    _run_flashinfer_fp4_gemm(m, n, k, torch.bfloat16, "auto")


@unittest.skipIf(SM < 100, SKIP_REASON)
class TestFP4GemmSglangCutlass(unittest.TestCase):

    def test_sglang_cutlass(self):
        for n, k in WEIGHT_SHAPES:
            for m in BATCH_SIZES:
                with self.subTest(m=m, n=n, k=k):
                    _run_sglang_cutlass_fp4_gemm(m, n, k, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
