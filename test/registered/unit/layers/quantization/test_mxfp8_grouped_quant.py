import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization import mxfp8_grouped_quant as grouped_quant
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMxfp8GroupedQuantDispatch(CustomTestCase):
    def setUp(self):
        super().setUp()
        grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = None
        grouped_quant._load_flashinfer_mxfp8_grouped_quant.cache_clear()
        grouped_quant._resolve_grouped_quant_impl.cache_clear()

    def tearDown(self):
        grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = None
        grouped_quant._load_flashinfer_mxfp8_grouped_quant.cache_clear()
        grouped_quant._resolve_grouped_quant_impl.cache_clear()
        super().tearDown()

    def test_initialize_sets_backend(self):
        server_args = types.SimpleNamespace(mxfp8_grouped_quant_backend="native")

        grouped_quant.initialize_mxfp8_grouped_quant_config(server_args)

        self.assertTrue(grouped_quant.get_mxfp8_grouped_quant_backend().is_native())

    def test_native_backend_calls_sgl_kernel(self):
        calls = []
        fake_sgl_kernel = types.SimpleNamespace(
            es_sm100_mxfp8_blockscaled_grouped_quant=lambda *args: calls.append(args)
        )
        grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = (
            grouped_quant.Mxfp8GroupedQuantBackend.NATIVE
        )

        with patch.dict(sys.modules, {"sgl_kernel": fake_sgl_kernel}):
            grouped_quant.mxfp8_grouped_quant(1, 2, 3, 4, 5, 6)

        self.assertEqual(calls, [(1, 2, 3, 4, 5, 6)])

    def test_flashinfer_backend_calls_flashinfer_kernel(self):
        calls = []
        fake_kernel_module = types.SimpleNamespace(
            mxfp8_grouped_quantize_cutile=lambda *args: calls.append(args),
        )
        fake_cutile_module = types.SimpleNamespace(
            is_cuda_tile_available=lambda: True,
        )
        grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = (
            grouped_quant.Mxfp8GroupedQuantBackend.FLASHINFER
        )

        fake_modules = {
            "flashinfer": types.ModuleType("flashinfer"),
            "flashinfer.cutile": fake_cutile_module,
            "flashinfer.quantization": types.ModuleType("flashinfer.quantization"),
            "flashinfer.quantization.kernels": types.ModuleType(
                "flashinfer.quantization.kernels"
            ),
            "flashinfer.quantization.kernels.cutile": types.ModuleType(
                "flashinfer.quantization.kernels.cutile"
            ),
            "flashinfer.quantization.kernels.cutile.mxfp8_grouped_quantize_cutile": fake_kernel_module,
        }
        with patch.dict(sys.modules, fake_modules):
            grouped_quant.mxfp8_grouped_quant(1, 2, 3, 4, 5, 6)

        self.assertEqual(calls, [(1, 2, 3, 4, 5, 6)])

    def test_availability_false_when_symbol_missing(self):
        # Mismatched FlashInfer build: the kernel module exists but does not
        # define `mxfp8_grouped_quantize_cutile`.
        fake_cutile_module = types.SimpleNamespace(
            is_cuda_tile_available=lambda: True,
        )
        fake_kernel_module = types.ModuleType(
            "flashinfer.quantization.kernels.cutile.mxfp8_grouped_quantize_cutile"
        )

        fake_modules = {
            "flashinfer": types.ModuleType("flashinfer"),
            "flashinfer.cutile": fake_cutile_module,
            "flashinfer.quantization": types.ModuleType("flashinfer.quantization"),
            "flashinfer.quantization.kernels": types.ModuleType(
                "flashinfer.quantization.kernels"
            ),
            "flashinfer.quantization.kernels.cutile": types.ModuleType(
                "flashinfer.quantization.kernels.cutile"
            ),
            "flashinfer.quantization.kernels.cutile.mxfp8_grouped_quantize_cutile": fake_kernel_module,
        }
        with patch.dict(sys.modules, fake_modules):
            self.assertFalse(
                grouped_quant.is_flashinfer_mxfp8_grouped_quant_available()
            )

    def test_auto_uses_flashinfer_when_available(self):
        calls = []
        grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = (
            grouped_quant.Mxfp8GroupedQuantBackend.AUTO
        )

        with (
            patch.object(
                grouped_quant,
                "is_flashinfer_mxfp8_grouped_quant_available",
                return_value=True,
            ),
            patch.object(
                grouped_quant,
                "_flashinfer_mxfp8_grouped_quant",
                side_effect=lambda *args: calls.append(("flashinfer", args)),
            ),
            patch.object(
                grouped_quant,
                "_native_mxfp8_grouped_quant",
                side_effect=lambda *args: calls.append(("native", args)),
            ),
        ):
            grouped_quant.mxfp8_grouped_quant(1, 2, 3, 4, 5, 6)

        self.assertEqual(calls, [("flashinfer", (1, 2, 3, 4, 5, 6))])

    def test_auto_falls_back_to_native_when_flashinfer_unavailable(self):
        calls = []
        grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = (
            grouped_quant.Mxfp8GroupedQuantBackend.AUTO
        )

        with (
            patch.object(
                grouped_quant,
                "is_flashinfer_mxfp8_grouped_quant_available",
                return_value=False,
            ),
            patch.object(
                grouped_quant,
                "_flashinfer_mxfp8_grouped_quant",
                side_effect=lambda *args: calls.append(("flashinfer", args)),
            ),
            patch.object(
                grouped_quant,
                "_native_mxfp8_grouped_quant",
                side_effect=lambda *args: calls.append(("native", args)),
            ),
        ):
            grouped_quant.mxfp8_grouped_quant(1, 2, 3, 4, 5, 6)

        self.assertEqual(calls, [("native", (1, 2, 3, 4, 5, 6))])

    def test_flashinfer_backend_raises_when_kernel_is_missing(self):
        grouped_quant.MXFP8_GROUPED_QUANT_BACKEND = (
            grouped_quant.Mxfp8GroupedQuantBackend.FLASHINFER
        )

        with (
            patch.object(
                grouped_quant.importlib,
                "import_module",
                side_effect=ImportError("missing"),
            ),
            self.assertRaisesRegex(RuntimeError, "--mxfp8-grouped-quant-backend"),
        ):
            grouped_quant.mxfp8_grouped_quant(1, 2, 3, 4, 5, 6)


if __name__ == "__main__":
    unittest.main()
