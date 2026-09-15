import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import sglang.kernels.ops.quantization.fp8_kernel as fp8_kernel
from sglang.kernels.ops.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    w8a8_block_fp8_matmul,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

from sglang.srt.utils import get_device, is_cuda, is_xpu

_is_cuda = is_cuda()
_is_xpu = is_xpu()

device = get_device()


class TestFP8Base(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.M = 256
        # test non-aligned
        cls.N = 1024 + 64
        cls.K = 512
        cls.group_size = 128
        cls.quant_type = torch.float8_e4m3fn
        cls.output_type = torch.bfloat16

    @staticmethod
    def _make_A(M, K, group_size, out_dtype):
        quant_A = torch.rand(
            M, K // group_size, group_size, dtype=torch.float32, device=device
        )
        # -1 ~ 1
        quant_A = quant_A * 2 - 1
        # scaling abs max to fmax
        finfo = torch.finfo(out_dtype)
        fmax = finfo.max
        scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
        quant_A *= scaling
        quant_A = quant_A.to(out_dtype).to(torch.float32)

        # create scale and A
        scale = torch.rand(M, K // group_size, dtype=torch.float32, device=device)
        scale /= fmax
        A = quant_A * scale[..., None]

        A = A.reshape(M, K)
        quant_A = quant_A.reshape(M, K).to(out_dtype)
        return A, quant_A, scale

    @staticmethod
    def _make_B(K, N, group_size, out_dtype):
        def _aligned_size(a, b):
            return (a + b - 1) // b * b

        K_aligned = _aligned_size(K, group_size)
        N_aligned = _aligned_size(N, group_size)

        quant_B = torch.rand(
            K_aligned // group_size,
            group_size,
            N_aligned // group_size,
            group_size,
            dtype=torch.float32,
            device=device,
        )
        quant_B = quant_B * 2 - 1

        # scaling abs max to fmax
        finfo = torch.finfo(out_dtype)
        fmax = finfo.max
        scaling = fmax / quant_B.abs().amax((1, 3), keepdim=True)
        quant_B *= scaling
        quant_B = quant_B.to(out_dtype).to(torch.float32)

        scale = torch.rand(
            K_aligned // group_size,
            1,
            N_aligned // group_size,
            1,
            dtype=torch.float32,
            device=device,
        )
        scale /= fmax

        B = quant_B * scale

        B = B.reshape(K_aligned, N_aligned)[:K, :N]
        quant_B = quant_B.reshape(K_aligned, N_aligned).to(out_dtype)[:K, :N]
        scale = scale.reshape(K_aligned // group_size, N_aligned // group_size)
        return B, quant_B, scale


class TestPerTokenGroupQuantFP8(TestFP8Base):
    def test_per_token_group_quant_fp8(self):
        if _is_cuda and torch.cuda.get_device_capability()[0] < 9:
            return

        A, A_quant_gt, scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        A_quant, scale = per_token_group_quant_fp8(
            x=A.to(torch.bfloat16), group_size=self.group_size
        )
        torch.testing.assert_close(scale, scale_gt)
        diff = (A_quant.to(torch.float16) - A_quant_gt.to(torch.float16)).abs()
        diff_count = (diff > 1e-5).count_nonzero()
        assert diff_count / diff.numel() < 1e-4


class TestW8A8BlockFP8Matmul(TestFP8Base):
    def test_w8a8_block_fp8_matmul(self):
        if _is_cuda and torch.cuda.get_device_capability()[0] < 9:
            return
        elif _is_xpu:
            # XPU doesn't provide traditional capability info like CUDA
            pass
        else:
            return

        A, A_quant_gt, A_scale_gt = self._make_A(
            M=self.M, K=self.K, group_size=self.group_size, out_dtype=self.quant_type
        )
        B, B_quant_gt, B_scale_gt = self._make_B(
            K=self.K, N=self.N, group_size=self.group_size, out_dtype=self.quant_type
        )
        C_gt = A.to(self.output_type) @ B.to(self.output_type)
        C = w8a8_block_fp8_matmul(
            A=A_quant_gt,
            B=B_quant_gt.T.contiguous(),
            As=A_scale_gt,
            Bs=B_scale_gt.T.contiguous(),
            block_size=[128, 128],
            output_dtype=self.output_type,
        )
        torch.testing.assert_close(C, C_gt, atol=0.5, rtol=1e-4)


@unittest.skipUnless(_is_cuda, "Generic CUDA block-FP8 regression")
class TestW8A8BlockFP8TileValidation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] < 9:
            raise unittest.SkipTest("Requires SM90 or newer")

    @staticmethod
    def _config(bk):
        return dict(
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=bk,
            GROUP_SIZE_M=1,
            num_warps=4,
            num_stages=4,
        )

    @staticmethod
    def _inputs(group_k, k, m=3, n=35, dtype=torch.bfloat16):
        groups = (k + group_k - 1) // group_k
        a = torch.ones((m, k), device=device).to(torch.float8_e4m3fn)
        b = torch.ones((n, k), device=device).to(torch.float8_e4m3fn)
        a_scale = torch.tensor([4.0**i for i in range(groups)], device=device)
        b_scale = torch.tensor([2.0**i for i in range(groups)], device=device)
        a_scale = a_scale.repeat(m, 1)
        b_scale = b_scale.repeat((n + 31) // 32, 1)
        value = sum(min(group_k, k - i * group_k) * 8.0**i for i in range(groups))
        expected = torch.full((m, n), value, device=device, dtype=dtype)
        return a, b, a_scale, b_scale, expected

    def test_supported_tiles(self):
        cases = [
            (32, 32, 17),
            (32, 32, 64),
            (32, 32, 65),
            (64, 32, 129),
            (128, 32, 288),
            (128, 64, 288),
            (128, 128, 288),
        ]
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for group_k, bk, k in cases:
                with self.subTest(group_k=group_k, bk=bk, k=k, dtype=dtype):
                    a, b, a_s, b_s, expected = self._inputs(group_k, k, dtype=dtype)
                    with patch.object(
                        fp8_kernel,
                        "get_w8a8_block_fp8_configs",
                        return_value={3: self._config(bk)},
                    ):
                        out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                            a, b, a_s, b_s, [32, group_k], output_dtype=dtype
                        )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_rejects_cross_group_tile(self):
        a, b, a_s, b_s, _ = self._inputs(32, 64, m=16, n=32)
        with patch.object(
            fp8_kernel,
            "get_w8a8_block_fp8_configs",
            return_value={16: self._config(64)},
        ):
            with self.assertRaisesRegex(ValueError, "BLOCK_SIZE_K.*group_k"):
                fp8_kernel.w8a8_block_fp8_matmul_triton(a, b, a_s, b_s, [32, 32])

    def test_torch_compile_default_config(self):
        a, b, a_s, b_s, expected = self._inputs(32, 65)
        compiled = torch.compile(
            fp8_kernel.w8a8_block_fp8_matmul_triton, fullgraph=True
        )
        out = compiled(a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_rejects_invalid_tiles(self):
        for group_k, bk in [(32, 64), (32, 128), (96, 64), (32, 0), (32, -32)]:
            with self.subTest(group_k=group_k, bk=bk):
                a, b, a_s, b_s, _ = self._inputs(group_k, group_k * 2)
                with patch.object(
                    fp8_kernel,
                    "get_w8a8_block_fp8_configs",
                    return_value={3: self._config(bk)},
                ):
                    with self.assertRaisesRegex(ValueError, "BLOCK_SIZE_K.*group_k"):
                        fp8_kernel.w8a8_block_fp8_matmul_triton(
                            a, b, a_s, b_s, [32, group_k]
                        )

    def test_default_config_and_cuda_graph(self):
        a, b, a_s, b_s, expected = self._inputs(32, 65)
        with patch.object(fp8_kernel, "get_w8a8_block_fp8_configs", return_value=None):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fp8_kernel.w8a8_block_fp8_matmul_triton(
                        a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
                    )
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                    a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
                )
            graph.replay()
            torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_config_file_validation(self):
        a, b, a_s, b_s, expected = self._inputs(32, 64)
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory) / "configs"
            configs.mkdir()
            config_file = configs / (
                "N=35,K=64,device_name=Test_GPU,dtype=fp8_w8a8,"
                "block_shape=[32, 32].json"
            )
            with (
                patch.object(
                    fp8_kernel, "__file__", str(Path(directory) / "fp8_kernel.py")
                ),
                patch.object(fp8_kernel, "get_device_name", return_value="Test GPU"),
            ):
                try:
                    for bk in (32, 64, 0, -32):
                        config_file.write_text(json.dumps({"3": self._config(bk)}))
                        fp8_kernel.get_w8a8_block_fp8_configs.cache_clear()
                        if bk == 32:
                            out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                                a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
                            )
                            torch.testing.assert_close(out, expected, rtol=0, atol=0)
                        else:
                            with self.assertRaisesRegex(ValueError, "BLOCK_SIZE_K"):
                                fp8_kernel.w8a8_block_fp8_matmul_triton(
                                    a, b, a_s, b_s, [32, 32]
                                )
                finally:
                    fp8_kernel.get_w8a8_block_fp8_configs.cache_clear()

    def test_config_normalization_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = Path(directory) / "configs"
            configs.mkdir()
            config_file = configs / (
                "N=35,K=256,device_name=Test_GPU,dtype=fp8_w8a8,"
                "block_shape=[32, 128].json"
            )
            with (
                patch.object(
                    fp8_kernel, "__file__", str(Path(directory) / "fp8_kernel.py")
                ),
                patch.object(fp8_kernel, "get_device_name", return_value="Test GPU"),
            ):
                try:
                    # Preserve valid CUDA tiles and existing normalization policy.
                    for is_cuda, tile, expected_tile in [
                        (True, 64, 64),
                        (True, 48, 128),
                        (False, 64, 128),
                    ]:
                        config_file.write_text(json.dumps({"3": self._config(tile)}))
                        fp8_kernel.get_w8a8_block_fp8_configs.cache_clear()
                        with patch.object(fp8_kernel, "_is_cuda", is_cuda):
                            loaded = fp8_kernel.get_w8a8_block_fp8_configs(
                                35, 256, 32, 128
                            )
                        self.assertEqual(loaded[3]["BLOCK_SIZE_K"], expected_tile)
                finally:
                    fp8_kernel.get_w8a8_block_fp8_configs.cache_clear()

    def test_tuner_direct_call(self):
        path = Path(__file__).resolve().parents[3] / (
            "benchmark/kernels/quantization/tuning_block_wise_kernel.py"
        )
        spec = importlib.util.spec_from_file_location("block_fp8_tuner_test", path)
        tuner = importlib.util.module_from_spec(spec)
        # Importing the CLI must not change the test process's start method.
        with patch("multiprocessing.set_start_method"):
            spec.loader.exec_module(tuner)
        for group_k, bk in [(32, 32), (128, 64), (32, 64)]:
            a, b, a_s, b_s, expected = self._inputs(group_k, group_k * 2)
            if bk > group_k:
                with self.assertRaisesRegex(ValueError, "BLOCK_SIZE_K.*group_k"):
                    tuner.w8a8_block_matmul(
                        a, b, a_s, b_s, [32, group_k], self._config(bk)
                    )
            else:
                out = tuner.w8a8_block_matmul(
                    a, b, a_s, b_s, [32, group_k], self._config(bk), torch.bfloat16
                )
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
        # The shared tuning entry also accepts INT8; its dispatch is unchanged.
        a, b, a_s, b_s, expected = self._inputs(128, 256)
        out = tuner.w8a8_block_matmul(
            a.to(torch.int8),
            b.to(torch.int8),
            a_s,
            b_s,
            [32, 128],
            self._config(64),
            torch.bfloat16,
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
