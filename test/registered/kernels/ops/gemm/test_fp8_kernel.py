import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.kernels.ops.gemm.fp8_kernel as fp8_kernel
from sglang.kernels.ops.gemm.fp8_kernel import w8a8_block_fp8_matmul
from sglang.srt.utils import get_device, is_cuda, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.fp8 import TestFP8Base
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=75, stage="base-b", runner_config="1-gpu-large")


device = get_device()
_is_cuda = is_cuda()
_is_xpu = is_xpu()


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


@unittest.skipUnless(_is_cuda, "CUDA block-FP8 tile validation")
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

    def test_cross_group_tile(self):
        a, b, a_s, b_s, _ = self._inputs(32, 64, m=16, n=32)
        with patch.object(
            fp8_kernel,
            "get_w8a8_block_fp8_configs",
            return_value={16: self._config(64)},
        ):
            out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
            )
        # 32 * 1 * 1 + 32 * 4 * 2; one cross-group dot would return 64.
        torch.testing.assert_close(out, torch.full_like(out, 288), rtol=0, atol=0)

    def test_torch_compile_default_config(self):
        a, b, a_s, b_s, expected = self._inputs(32, 65)
        compiled = torch.compile(
            fp8_kernel.w8a8_block_fp8_matmul_triton, fullgraph=True
        )
        out = compiled(a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_rejects_invalid_tiles(self):
        for group_k, bk in [
            (32, 48),
            (96, 128),
            (32, 512),
            (96, 64),
            (32, 0),
            (32, -32),
        ]:
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
                        if bk in (32, 64):
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

    def test_dispatch_rejects_invalid_tiles_before_launch(self):
        # gfx1250 is a host-dispatch check, not an AMD numerical test.
        routes = [
            (False, False, {}, "_w8a8_block_fp8_matmul"),
            (True, False, {"SWAP_AB": True}, "_w8a8_block_fp8_matmul_hopper"),
            (True, False, {"SPLIT_K": 2}, "_w8a8_block_fp8_matmul_hopper"),
            (False, True, {}, "_w8a8_block_fp8_matmul_gfx1250"),
        ]
        a, b, a_s, b_s, _ = self._inputs(32, 64)
        for sm90, gfx1250, extra, kernel_name in routes:
            tiles = (0, -32, 48, 512)
            if kernel_name != "_w8a8_block_fp8_matmul":
                tiles += (64, 128, 256)
            for bk in tiles:
                with self.subTest(kernel=kernel_name, extra=extra, bk=bk):
                    config = {**self._config(bk), **extra}
                    with (
                        patch.object(
                            fp8_kernel,
                            "get_w8a8_block_fp8_configs",
                            return_value={3: config},
                        ),
                        patch.object(
                            fp8_kernel,
                            "get_platform",
                            return_value=SimpleNamespace(is_sm90=sm90),
                        ),
                        patch.object(fp8_kernel, "_is_gfx1250", gfx1250),
                        patch.object(fp8_kernel, kernel_name) as kernel,
                        patch.object(fp8_kernel.torch, "empty") as allocate_partials,
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "BLOCK_SIZE_K.*group_k"
                        ):
                            fp8_kernel.w8a8_block_fp8_matmul_triton(
                                a, b, a_s, b_s, [32, 32]
                            )
                        kernel.__getitem__.assert_not_called()
                        allocate_partials.assert_not_called()

    def test_hopper_supported_tiles(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("Hopper numerical regression requires SM90")
        # Include a split starting inside a quantization group, masked tails,
        # and splits with no K tiles. Compare independently constructed values.
        for group_k, bk, k in [(32, 32, 65), (128, 32, 288), (128, 64, 288)]:
            for swap_ab, split_k in [(True, 1), (False, 2), (True, 2), (True, 16)]:
                with self.subTest(
                    group_k=group_k, bk=bk, k=k, swap_ab=swap_ab, split_k=split_k
                ):
                    a, b, a_s, b_s, expected = self._inputs(group_k, k)
                    config = {
                        **self._config(bk),
                        "SWAP_AB": swap_ab,
                        "SPLIT_K": split_k,
                    }
                    with patch.object(
                        fp8_kernel,
                        "get_w8a8_block_fp8_configs",
                        return_value={3: config},
                    ):
                        out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                            a, b, a_s, b_s, [32, group_k], output_dtype=torch.bfloat16
                        )
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_linear_quantized_and_prequantized_inputs(self):
        from sglang.srt.layers.quantization.fp8_utils import (
            per_token_group_quant_fp8,
            triton_w8a8_block_fp8_linear,
        )

        gen = torch.Generator(device=device).manual_seed(43)
        x = torch.randn((3, 64), dtype=torch.bfloat16, device=device, generator=gen)
        weight = torch.randn((35, 64), device=device, generator=gen).to(
            torch.float8_e4m3fn
        )
        scales = torch.rand((2, 2), device=device, generator=gen) + 0.25
        q, q_scale = per_token_group_quant_fp8(x, 32, column_major_scales=False)
        bias = torch.randn((35,), dtype=torch.bfloat16, device=device, generator=gen)
        # Use an FP64 oracle, independent of the process's FP32/TF32 setting.
        dequant_x = q.double() * q_scale.double().repeat_interleave(32, 1)
        dequant_w = weight.double() * scales.double().repeat_interleave(32, 0)[
            :35
        ].repeat_interleave(32, 1)
        reference = (dequant_x @ dequant_w.T).to(torch.bfloat16) + bias
        for bk in (32, 64, 128):
            with patch.object(
                fp8_kernel,
                "get_w8a8_block_fp8_configs",
                return_value={3: self._config(bk)},
            ):
                for inp, inp_scale in [(x, None), (q, q_scale)]:
                    with self.subTest(bk=bk, prequantized=inp_scale is not None):
                        out = triton_w8a8_block_fp8_linear(
                            inp,
                            weight,
                            [32, 32],
                            scales,
                            input_scale=inp_scale,
                            bias=bias,
                        )
                        torch.testing.assert_close(out, reference, rtol=0.01, atol=0.02)

    def test_tuner_direct_call(self):
        path = Path(__file__).resolve().parents[5] / (
            "benchmark/kernels/quantization/tuning_block_wise_kernel.py"
        )
        spec = importlib.util.spec_from_file_location("block_fp8_tuner_test", path)
        tuner = importlib.util.module_from_spec(spec)
        # Importing the CLI must not change the test process's start method.
        with patch("multiprocessing.set_start_method"):
            spec.loader.exec_module(tuner)
        for group_k, bk in [(32, 32), (128, 64), (32, 64)]:
            a, b, a_s, b_s, expected = self._inputs(group_k, group_k * 2)
            out = tuner.w8a8_block_matmul(
                a, b, a_s, b_s, [32, group_k], self._config(bk), torch.bfloat16
            )
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self.assertEqual(
            {c["BLOCK_SIZE_K"] for c in tuner.get_tuning_configs(32, "fp8")},
            {32, 64, 128},
        )
        for kind, group in [("int8", 128), ("fp8", 96)]:
            configs = tuner.get_tuning_configs(group, kind)
            self.assertTrue(all(group % c["BLOCK_SIZE_K"] == 0 for c in configs))
        with patch.object(tuner, "is_cuda", return_value=False):
            configs = tuner.get_tuning_configs(64, "fp8")
            self.assertEqual({c["BLOCK_SIZE_K"] for c in configs}, {64})
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

    def test_unrolled_group_tails(self):
        # Dyadic inputs/scales permit exact FP64 reference comparisons. Padding
        # scale storage with NaNs also detects reads of masked-off K groups.
        gen = torch.Generator(device=device).manual_seed(29)
        for group_k, bk in [
            (32, 64),
            (32, 128),
            (32, 256),
            (64, 128),
            (64, 256),
            (128, 256),
        ]:
            for k in [17, group_k + 1, 288, 576]:
                m, n = 3, 35
                groups = (k + group_k - 1) // group_k
                a = (
                    torch.randint(-4, 5, (m, k), device=device, generator=gen).float()
                    / 4
                ).to(torch.float8_e4m3fn)
                b = (
                    torch.randint(-4, 5, (n, k), device=device, generator=gen).float()
                    / 4
                ).to(torch.float8_e4m3fn)
                for layout in ["compact", "padded", "column"]:

                    def scales(rows):
                        if layout == "column":
                            storage = torch.full(
                                (groups + 8, rows), float("nan"), device=device
                            )
                            view = storage[:groups].t()
                        elif layout == "padded":
                            storage = torch.full(
                                (rows, groups + 8), float("nan"), device=device
                            )
                            view = storage[:, :groups]
                        else:
                            view = torch.empty((rows, groups), device=device)
                        view.copy_(
                            2.0
                            ** torch.randint(
                                -1, 2, view.shape, device=device, generator=gen
                            ).float()
                        )
                        return view

                    a_s, b_s = scales(m), scales(2)
                    da = a.double() * a_s.double().repeat_interleave(group_k, 1)[:, :k]
                    db = (
                        b.double()
                        * b_s.double()
                        .repeat_interleave(32, 0)[:n]
                        .repeat_interleave(group_k, 1)[:, :k]
                    )
                    ref = da @ db.t()
                    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
                        with self.subTest(
                            group_k=group_k, bk=bk, k=k, layout=layout, dtype=dtype
                        ):
                            with patch.object(
                                fp8_kernel,
                                "get_w8a8_block_fp8_configs",
                                return_value={m: self._config(bk)},
                            ):
                                out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                                    a, b, a_s, b_s, [32, group_k], output_dtype=dtype
                                )
                            torch.testing.assert_close(
                                out, ref.to(dtype), rtol=0, atol=0
                            )

    def test_config_selection_boundaries(self):
        configs = {2: self._config(32), 16: self._config(128)}
        for m in (8, 9, 10, 24):
            with self.subTest(m=m):
                a, b, a_s, b_s, expected = self._inputs(32, 65, m=m)
                selected = []

                def observe(m, n, config):
                    selected.append(config["BLOCK_SIZE_K"])
                    return fp8_kernel._w8a8_block_fp8_matmul

                with (
                    patch.object(
                        fp8_kernel, "get_w8a8_block_fp8_configs", return_value=configs
                    ),
                    patch.object(
                        fp8_kernel,
                        "select_w8a8_block_fp8_matmul_kernel",
                        side_effect=observe,
                    ),
                ):
                    out = fp8_kernel.w8a8_block_fp8_matmul_triton(
                        a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16
                    )
                key = min(configs, key=lambda value: abs(value - m))
                self.assertEqual(selected, [configs[key]["BLOCK_SIZE_K"]])
                torch.testing.assert_close(out, expected, rtol=0, atol=0)

    def test_unrolled_compile_and_graph(self):
        a, b, a_s, b_s, expected = self._inputs(32, 65)
        configs = {2: self._config(32), 16: self._config(128)}
        with patch.object(
            fp8_kernel,
            "get_w8a8_block_fp8_configs",
            new=lambda *args: configs,
        ):
            compiled = torch.compile(
                fp8_kernel.w8a8_block_fp8_matmul_triton, fullgraph=True
            )
            for m in (3, 9, 10, 16):
                a, b, a_s, b_s, expected = self._inputs(32, 65, m=m)
                out = compiled(a, b, a_s, b_s, [32, 32], output_dtype=torch.bfloat16)
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
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

    def test_unrolled_rejects_unsupported_storage_and_backend(self):
        a, b, a_s, b_s, _ = self._inputs(32, 64)
        with patch.object(
            fp8_kernel, "get_w8a8_block_fp8_configs", return_value={3: self._config(64)}
        ):
            with self.assertRaises(NotImplementedError):
                fp8_kernel.w8a8_block_fp8_matmul_triton(
                    a, b, a_s.half(), b_s.half(), [32, 32]
                )
            # Structurally valid packed scales reach the new path's dtype guard.
            packed_a = torch.ones((3, 1), device=device, dtype=torch.int32)
            packed_b = torch.ones((35, 1), device=device, dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, "FP32 scale storage"):
                fp8_kernel.w8a8_block_fp8_matmul_triton(
                    a, b, packed_a, packed_b, [32, 32]
                )
        with patch.object(fp8_kernel, "_is_cuda", False):
            with self.assertRaisesRegex(ValueError, "BLOCK_SIZE_K.*group_k"):
                fp8_kernel._select_w8a8_block_fp8_generic_kernel(
                    32, self._config(64), a_s, b_s
                )

    def test_unrolled_selector_keeps_other_kernels_strict(self):
        scales = SimpleNamespace(dtype=torch.float32)
        for group_k, bk in [(32, 64), (32, 128), (64, 128), (128, 256)]:
            with self.subTest(group_k=group_k, bk=bk):
                config = self._config(bk)
                # The original validator must not gain CUDA-only exceptions:
                # Hopper and gfx1250 still use it before their launches.
                with self.assertRaisesRegex(ValueError, "BLOCK_SIZE_K.*group_k"):
                    fp8_kernel._validate_w8a8_block_fp8_config(group_k, config)
                self.assertIs(
                    fp8_kernel._select_w8a8_block_fp8_generic_kernel(
                        group_k, config, scales, scales
                    ),
                    fp8_kernel._w8a8_block_fp8_matmul_k_groups,
                )
                for extra in [
                    {"SWAP_AB": True},
                    {"SWAP_AB": False},
                    {"SPLIT_K": 2},
                    {"SPLIT_K": 1},
                    {"SWAP_AB": True, "SPLIT_K": 2},
                ]:
                    with self.subTest(extra=extra):
                        with self.assertRaisesRegex(ValueError, "SWAP_AB.*SPLIT_K"):
                            fp8_kernel._select_w8a8_block_fp8_generic_kernel(
                                group_k, {**config, **extra}, scales, scales
                            )
        # Storage restrictions are specific to the new unrolled path.
        packed = SimpleNamespace(dtype=torch.int32)
        self.assertIs(
            fp8_kernel._select_w8a8_block_fp8_generic_kernel(
                128, self._config(64), packed, packed
            ),
            fp8_kernel._w8a8_block_fp8_matmul,
        )


if __name__ == "__main__":
    unittest.main()
