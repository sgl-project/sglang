"""Numerics for the FP8 dense-linear GEMM backends (--fp8-gemm-backend).

Real layer path vs a dequantized-reference matmul, in four formats: FP8
blockwise, MXFP8, 32-wide-K ue8m0 block FP8 served as MXFP8, and per-tensor
FP8 (auto dispatch). Backend sets adapt to the device SM, so one file covers
SM90 / SM100 / SM120.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.quantization import fp8_utils
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import Fp8GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from sglang.srt.layers.quantization.mxfp8_input import Mxfp8SwizzledInput
from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.layer_ut_utils import (
    assert_output_close,
    init_single_process_dist,
    load_linear_weights,
    make_tp1_column_parallel_linear,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=112, stage="base-b", runner_config="4-gpu-b200")
register_cuda_ci(est_time=11, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-large")

FP8_MAX = 448.0

# (M, N, K), N and K multiples of the (128, 128) weight block.
FP8_BLOCK_SHAPES = [
    (64, 512, 512),
    (5, 384, 896),
    (128, 1024, 1024),
]

# (M, N, K); K must be a multiple of 256 (flashinfer trtllm mxfp8 requirement).
MXFP8_SHAPES = [
    (64, 512, 512),
    (5, 384, 768),
]

# (M, N, K); N % 64 == 0 and K % 128 == 0 for the FlashInfer MXFP8 scale swizzle.
BLOCK32_SHAPES = [
    (64, 512, 512),
    (5, 384, 768),
]

# (M, N, K); per-tensor has no block-alignment constraints.
PER_TENSOR_SHAPES = [
    (64, 512, 512),
    (5, 384, 896),
]


def _fp8_block_backends():
    sm = get_device_sm()
    if 100 <= sm < 110:
        return ["triton", "deep_gemm", "flashinfer_trtllm", "flashinfer_cutlass"]
    if sm >= 120:
        # cutlass is the SM120-only explicit backend; the trtllm / deepgemm
        # kernels do not support consumer Blackwell.
        return ["triton", "cutlass"]
    if sm == 90:
        # flashinfer_deepgemm (swapAB) is SM90-only.
        return ["triton", "deep_gemm", "flashinfer_deepgemm"]
    return []


def _mxfp8_backends():
    # MXFP8 linear is validated on SM100/103 only.
    if get_device_sm() in (100, 103):
        return [
            "auto",
            "flashinfer_trtllm",
            "flashinfer_cutlass",
            "flashinfer_cutedsl",
        ]
    return []


def _quantize_fp8_blockwise(w: torch.Tensor, block: int = 128):
    """Per (block, block) tile fp8 quantization; returns checkpoint-format
    (w_fp8 [N, K], scale_inv fp32 [N/block, K/block]) and the dequant reference."""
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-12)
    scale = amax / FP8_MAX
    w_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    w_dequant = (w_fp8.float() * scale[:, None, :, None]).reshape(n, k)
    return w_fp8.reshape(n, k), scale, w_dequant


def _block32_backends():
    # The block-fp8-as-MXFP8 route takes the FlashInfer CUTLASS / CuTe-DSL MXFP8
    # kernels only, on SM100/103.
    if get_device_sm() in (100, 103):
        return ["flashinfer_cutlass", "flashinfer_cutedsl"]
    return []


def _quantize_fp8_block32_ue8m0(w: torch.Tensor, block: int = 32):
    """Per (block, block) tile fp8 quantization with power-of-two scales; returns
    checkpoint-format (w_fp8 [N, K], scale e8m0 [N/block, K/block]) and the
    dequant reference."""
    n, k = w.shape
    tiles = w.float().reshape(n // block, block, k // block, block)
    amax = tiles.abs().amax(dim=(1, 3)).clamp(min=1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    w_fp8 = (tiles / scale[:, None, :, None]).to(torch.float8_e4m3fn)
    w_dequant = (w_fp8.float() * scale[:, None, :, None]).reshape(n, k)
    return w_fp8.reshape(n, k), scale.to(torch.float8_e8m0fnu), w_dequant


def _quantize_mxfp8(w: torch.Tensor, block: int = 32):
    """Per (1, block) group e8m0 quantization; returns checkpoint-format
    (w_fp8 [N, K], scale uint8 [N, K/block]) and the dequant reference."""
    n, k = w.shape
    groups = w.float().reshape(n, k // block, block)
    amax = groups.abs().amax(dim=-1).clamp(min=1e-12)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(min=-127, max=127)
    scale = torch.pow(2.0, exp)
    w_fp8 = (groups / scale[..., None]).to(torch.float8_e4m3fn)
    w_dequant = (w_fp8.float() * scale[..., None]).reshape(n, k)
    scale_e8m0 = (exp + 127).to(torch.uint8)
    return w_fp8.reshape(n, k), scale_e8m0, w_dequant


def _make_linear(quant_config, n: int, k: int):
    return make_tp1_column_parallel_linear(
        quant_config, n, k, skip_block_quant_check=True
    )


class _LinearBackendCheck(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        init_single_process_dist()

    def _check_backend(self, backend: str, allowed, shapes, build_layer):
        if backend not in allowed:
            self.skipTest(f"{backend} not in SM{get_device_sm()} backend set")
        torch.manual_seed(7)
        for m, n, k in shapes:
            with self.subTest(backend=backend, shape=(m, n, k)):
                with mock.patch.object(
                    fp8_utils,
                    "FP8_GEMM_RUNNER_BACKEND",
                    Fp8GemmRunnerBackend(backend),
                ):
                    layer, w_dequant = build_layer(n, k)
                    layer.quant_method.process_weights_after_loading(layer)

                    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / 10
                    out, _ = layer(x)

                    ref = x.float() @ w_dequant.T
                    # atol covers single-element UE8M0 scale-rounding outliers
                    # (deep_gemm); a wrong kernel/layout fails by orders more.
                    assert_output_close(self, out, ref, rtol=5e-2, atol=1e-1)


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestFp8BlockwiseLinearBackends(_LinearBackendCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale_inv, w_dequant = _quantize_fp8_blockwise(w)
        load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_inv)
        return layer, w_dequant

    def _run(self, backend: str):
        self._check_backend(
            backend, _fp8_block_backends(), FP8_BLOCK_SHAPES, self._build_layer
        )

    def test_triton(self):
        self._run("triton")

    def test_deep_gemm(self):
        self._run("deep_gemm")

    def test_flashinfer_trtllm(self):
        self._run("flashinfer_trtllm")

    def test_flashinfer_cutlass(self):
        self._run("flashinfer_cutlass")

    def test_flashinfer_deepgemm(self):
        self._run("flashinfer_deepgemm")

    def test_cutlass(self):
        self._run("cutlass")


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestMxfp8LinearBackends(_LinearBackendCheck):
    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            use_mxfp8=True,
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        w_fp8, scale_e8m0, w_dequant = _quantize_mxfp8(w)
        load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_e8m0)
        return layer, w_dequant

    def _run(self, backend: str):
        self._check_backend(backend, _mxfp8_backends(), MXFP8_SHAPES, self._build_layer)

    def test_flashinfer_trtllm(self):
        self._run("flashinfer_trtllm")

    def test_flashinfer_cutlass(self):
        self._run("flashinfer_cutlass")

    def test_flashinfer_cutedsl(self):
        self._run("flashinfer_cutedsl")

    def test_auto(self):
        if "auto" not in _mxfp8_backends():
            self.skipTest(f"auto not in SM{get_device_sm()} MXFP8 backend set")
        with mock.patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.AUTO,
        ):
            self.assertEqual(
                fp8_utils.resolve_mxfp8_dense_gemm_backend(),
                fp8_utils.Mxfp8DenseGemmBackend.FLASHINFER_CUTEDSL,
            )
        self._run("auto")

    @unittest.skipUnless(get_device_sm() >= 100, "Requires Blackwell FlashInfer")
    def test_auto_falls_back_when_cutedsl_is_unsupported(self):
        with (
            mock.patch.object(
                fp8_utils,
                "FP8_GEMM_RUNNER_BACKEND",
                Fp8GemmRunnerBackend.AUTO,
            ),
            mock.patch.object(fp8_utils, "get_device_sm", return_value=107),
            mock.patch.object(
                fp8_utils._raw_flashinfer_mm_mxfp8,
                "is_backend_supported",
                return_value=False,
            ) as is_backend_supported,
        ):
            self.assertEqual(
                fp8_utils.resolve_mxfp8_dense_gemm_backend(),
                fp8_utils.Mxfp8DenseGemmBackend.FLASHINFER_CUTLASS,
            )
            is_backend_supported.assert_called_once_with("cute-dsl", 107)


def _build_block32_layer(n: int, k: int, keep_plain_weight_layout: bool = False):
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
        scale_fmt="ue8m0",
    )
    layer = _make_linear(quant_config, n, k)
    if keep_plain_weight_layout:
        layer.keep_plain_weight_layout = True
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
    w_fp8, scale_e8m0, w_dequant = _quantize_fp8_block32_ue8m0(w)
    load_linear_weights(layer, weight=w_fp8, weight_scale_inv=scale_e8m0)
    return layer, w_dequant


class TestBlockFp8AsMxfp8Linear(_LinearBackendCheck):
    """A 32-wide-K ue8m0 block-fp8 weight served through the MXFP8 GEMMs."""

    _build_layer = staticmethod(_build_block32_layer)

    def _run(self, backend: str):
        self._check_backend(
            backend, _block32_backends(), BLOCK32_SHAPES, self._build_layer
        )

    def test_flashinfer_cutlass(self):
        self._run("flashinfer_cutlass")

    def test_flashinfer_cutedsl(self):
        self._run("flashinfer_cutedsl")

    def test_mxfp8_view_and_swizzled_input(self):
        if "flashinfer_cutedsl" not in _block32_backends():
            self.skipTest(f"cutedsl not in SM{get_device_sm()} backend set")
        from sglang.kernels.ops.attention.dsv4.wo_a import (
            _quantize_partial,
            _wo_a_reduce,
        )

        torch.manual_seed(7)
        with mock.patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL,
        ):
            n, k = 512, 2048
            layer, _ = self._build_layer(n, k)
            layer.quant_method.process_weights_after_loading(layer)
            self.assertTrue(layer.quant_method.block_fp8_as_mxfp8)
            self.assertTrue(layer.block_fp8_mxfp8_ready)
            # Block scales stay in place for the Triton fallback and raw readers.
            self.assertEqual(tuple(layer.weight_scale_inv.shape), (n // 32, k // 32))
            self.assertIsNotNone(layer.weight_scale_inv_swizzled)

            # A prequantized 128x4-swizzled MXFP8 activation must give the same
            # output as the bf16 input the layer quantizes itself.
            rows = 6
            partial = torch.randn(8, rows, 2, k // 2, device="cuda")
            bf16 = torch.empty(rows, k, dtype=torch.bfloat16, device="cuda")
            _wo_a_reduce[(rows * 8,)](partial, bf16, rows * k, num_warps=4)
            q, s = _quantize_partial(partial)
            swizzled = layer.quant_method.apply(layer, Mxfp8SwizzledInput(q, s))
            plain = layer.quant_method.apply(layer, bf16)
            torch.testing.assert_close(swizzled, plain, rtol=0, atol=0)

            # A layer that keeps the plain weight layout has no MXFP8 view.
            plain_layer, _ = self._build_layer(n, k, keep_plain_weight_layout=True)
            plain_layer.quant_method.process_weights_after_loading(plain_layer)
            self.assertFalse(plain_layer.block_fp8_mxfp8_ready)
            with self.assertRaises(ValueError):
                plain_layer.quant_method.apply(plain_layer, Mxfp8SwizzledInput(q, s))


@unittest.skipUnless(
    "flashinfer_cutedsl" in _block32_backends(),
    "block-fp8-as-MXFP8 prefill tuning needs the FlashInfer CuTe-DSL kernel",
)
class TestBlockFp8AsMxfp8PrefillAutotune(_LinearBackendCheck):
    """The startup hook that tunes those layers for the prefill M buckets."""

    def setUp(self):
        super().setUp()
        patcher = mock.patch.object(
            fp8_utils,
            "FP8_GEMM_RUNNER_BACKEND",
            Fp8GemmRunnerBackend.FLASHINFER_CUTEDSL,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        torch.manual_seed(7)

    @staticmethod
    def _ready_layer(n: int, k: int, keep_plain_weight_layout: bool = False):
        layer, _ = _build_block32_layer(n, k, keep_plain_weight_layout)
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_model_hook_deduplicates_ready_block_fp8_weights(self):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        layers = torch.nn.ModuleList()
        methods = []
        for _ in range(2):
            layer = self._ready_layer(128, 128)
            methods.append(layer.quant_method)
            layer.quant_method.apply = mock.Mock()
            layers.append(layer)
        # An unprepared layer intentionally has no swizzled scale buffer.
        fallback = self._ready_layer(128, 128, keep_plain_weight_layout=True)
        layers.append(fallback)
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"), model=layers
        )
        count = DeepseekV4ForCausalLM.autotune_prefill_kernels(
            model, 4096, dtype=torch.bfloat16
        )
        self.assertEqual(count, 1)
        methods[0].apply.assert_called_once()
        self.assertEqual(methods[0].apply.call_args.args[1].shape, (4096, 128))
        methods[1].apply.assert_not_called()
        for method in methods:
            self.assertEqual(method.mxfp8_prefill_autotune_min_tokens, 4096)
        self.assertIsNone(fallback.quant_method.mxfp8_prefill_autotune_min_tokens)

    def test_block_fp8_dispatch_keeps_decode_and_determinism_pinned(self):
        layer = self._ready_layer(128, 128)
        method = layer.quant_method
        method.mxfp8_prefill_autotune_min_tokens = 4096
        call = mock.Mock(return_value=torch.empty(0))
        method.w8a8_mxfp8_linear = call
        for rows, invariant, deterministic, expected in (
            (6, False, False, None),
            (4096, False, False, False),
            (4096, True, False, True),
            (4096, False, True, True),
        ):
            with self.subTest(
                rows=rows, invariant=invariant, deterministic=deterministic
            ):
                with (
                    mock.patch(
                        "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
                        return_value=invariant,
                    ),
                    mock.patch(
                        "sglang.srt.runtime_context.get_exec",
                        return_value=SimpleNamespace(
                            deterministic=SimpleNamespace(
                                enable_deterministic_inference=deterministic
                            )
                        ),
                    ),
                ):
                    method.apply(layer, torch.empty(rows, 128, device="cuda"))
                self.assertEqual(call.call_args.kwargs.get("pin_tactic"), expected)

    def test_prefill_tuning_leaves_decode_bit_identical(self):
        """Tuning the prefill buckets must not move the decode tactic: below the
        stamped min_tokens the output has to stay bit-for-bit what it was."""
        from flashinfer.autotuner import autotune

        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        runtime_patch = mock.patch(
            "sglang.srt.runtime_context.get_exec",
            return_value=SimpleNamespace(
                deterministic=SimpleNamespace(enable_deterministic_inference=False)
            ),
        )
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)
        layer = self._ready_layer(1792, 5120)
        method = layer.quant_method
        x = torch.randn(6, 5120, device="cuda", dtype=torch.bfloat16)
        original = method.apply(layer, x)
        model = SimpleNamespace(
            config=SimpleNamespace(model_type="deepseek_v41"),
            model=torch.nn.ModuleList([layer]),
        )
        with autotune(True):
            DeepseekV4ForCausalLM.autotune_prefill_kernels(
                model, 4096, dtype=torch.bfloat16
            )
        self.assertEqual(method.mxfp8_prefill_autotune_min_tokens, 4096)
        torch.testing.assert_close(method.apply(layer, x), original, rtol=0, atol=0)


@unittest.skipIf(get_device_sm() < 90, "FP8 GEMM backends require SM90+")
class TestModeloptFp8PerTensorLinear(_LinearBackendCheck):
    """Per-tensor FP8 (ModelOptFp8LinearMethod, static scales) on the auto
    dispatch path -- the checkpoint style of nvidia/*-FP8 models."""

    @staticmethod
    def _build_layer(n: int, k: int):
        quant_config = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True, packed_modules_mapping={}
        )
        layer = _make_linear(quant_config, n, k)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / 10
        scale = (w.float().abs().max() / FP8_MAX).clamp(min=1e-12)
        w_fp8 = (w.float() / scale).to(torch.float8_e4m3fn)
        # 0-dim scales exercise weight_loader_v2's scalar reshape branch.
        load_linear_weights(
            layer,
            weight=w_fp8,
            weight_scale=scale,
            input_scale=torch.tensor(1.0 / FP8_MAX, device="cuda"),
        )
        w_dequant = w_fp8.float() * scale
        return layer, w_dequant

    def test_auto(self):
        self._check_backend("auto", ["auto"], PER_TENSOR_SHAPES, self._build_layer)


if __name__ == "__main__":
    unittest.main()
