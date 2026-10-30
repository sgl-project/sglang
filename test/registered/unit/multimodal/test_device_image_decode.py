"""CPU regressions for device JPEG policy, backend choice, and model overrides."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import io
import types
import unittest
from unittest.mock import Mock, patch

import torch
from PIL import Image

from sglang.srt.hardware_backend.npu import jpeg
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.utils import common


def jpeg_bytes():
    output = io.BytesIO()
    Image.new("RGB", (64, 32), (31, 127, 235)).save(
        output, format="JPEG", subsampling=2
    )
    return output.getvalue()


class TestImageDecodeBackend(unittest.TestCase):
    def test_published_runtime_device_takes_priority_over_available_hardware(self):
        context = Mock()
        context.is_config_namespace_published.return_value = True
        for backend in ("cpu", "cuda", "npu", "xpu"):
            with (
                self.subTest(backend=backend),
                patch.object(common, "get_context", return_value=context),
                patch.object(
                    common,
                    "get_runtime_device",
                    return_value=types.SimpleNamespace(device=backend),
                ),
                patch.object(common, "is_cpu", side_effect=AssertionError("CPU probe")),
                patch.object(
                    common, "is_cuda", side_effect=AssertionError("CUDA probe")
                ),
                patch.object(common, "is_npu", side_effect=AssertionError("NPU probe")),
                patch.object(
                    torch,
                    "npu",
                    types.SimpleNamespace(
                        is_available=Mock(side_effect=AssertionError("NPU probe"))
                    ),
                    create=True,
                ),
            ):
                self.assertEqual(common._image_decode_backend(), backend)
        context.is_config_namespace_published.assert_called_with("device")

    def test_standalone_cpu_selection_does_not_probe_accelerators(self):
        context = Mock()
        context.is_config_namespace_published.return_value = False
        with (
            patch.object(common, "get_context", return_value=context),
            patch.object(
                common,
                "get_runtime_device",
                side_effect=AssertionError("unpublished config"),
            ),
            patch.object(common, "is_cpu", return_value=True),
            patch.object(common, "is_cuda", side_effect=AssertionError("CUDA probe")),
            patch.object(common, "is_npu", side_effect=AssertionError("NPU probe")),
            patch.object(
                torch,
                "npu",
                types.SimpleNamespace(
                    is_available=Mock(side_effect=AssertionError("NPU probe"))
                ),
                create=True,
            ),
        ):
            self.assertEqual(common._image_decode_backend(), "cpu")

    def test_standalone_cuda_does_not_probe_npu(self):
        context = Mock()
        context.is_config_namespace_published.return_value = False
        with (
            patch.object(common, "get_context", return_value=context),
            patch.object(common, "is_cpu", return_value=False),
            patch.object(common, "is_cuda", return_value=True),
            patch.object(common, "is_npu", side_effect=RuntimeError("NPU unavailable")),
            patch.object(
                torch,
                "npu",
                types.SimpleNamespace(
                    is_available=Mock(side_effect=AssertionError("NPU probe"))
                ),
                create=True,
            ),
        ):
            self.assertEqual(common._image_decode_backend(), "cuda")

    def test_standalone_npu_and_no_accelerator(self):
        context = Mock()
        context.is_config_namespace_published.return_value = False
        for available in (False, True):
            with (
                self.subTest(npu_available=available),
                patch.object(common, "get_context", return_value=context),
                patch.object(common, "is_cpu", return_value=False),
                patch.object(common, "is_cuda", return_value=False),
                patch.object(
                    common, "is_npu", side_effect=AssertionError("throwing NPU helper")
                ),
                patch.object(
                    torch,
                    "npu",
                    types.SimpleNamespace(is_available=lambda: available),
                    create=True,
                ),
            ):
                self.assertEqual(
                    common._image_decode_backend(), "npu" if available else "cpu"
                )


class TestDeviceImageDecodeDispatch(unittest.TestCase):
    def setUp(self):
        self.data = jpeg_bytes()

    def test_false_never_resolves_or_probes_device(self):
        with (
            patch.object(
                common,
                "_image_decode_backend",
                side_effect=AssertionError("backend resolution"),
            ),
            patch.object(common, "is_cpu", side_effect=AssertionError("CPU probe")),
            patch.object(common, "is_cuda", side_effect=AssertionError("CUDA probe")),
            patch.object(common, "is_npu", side_effect=RuntimeError("NPU unavailable")),
            patch.object(jpeg, "decode_jpeg") as decode,
        ):
            result, _ = common.load_image(self.data, device_image_decode=False)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (64, 32))
        decode.assert_not_called()

    def test_registered_but_unavailable_npu_keeps_standalone_png_on_pil(self):
        output = io.BytesIO()
        Image.new("RGB", (64, 32), (31, 127, 235)).save(output, format="PNG")
        context = Mock()
        context.is_config_namespace_published.return_value = False
        with (
            patch.object(common, "get_context", return_value=context),
            patch.object(common, "is_cpu", return_value=False),
            patch.object(common, "is_cuda", return_value=False),
            patch.object(common, "is_npu", side_effect=RuntimeError("NPU unavailable")),
            patch.object(
                torch,
                "npu",
                types.SimpleNamespace(is_available=lambda: False),
                create=True,
            ),
            patch.object(jpeg, "decode_jpeg") as decode,
        ):
            result, _ = common.load_image(output.getvalue())
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.getpixel((0, 0)), (31, 127, 235))
        decode.assert_not_called()

    def test_explicit_npu_runtime_does_not_hide_device_failure(self):
        context = Mock()
        context.is_config_namespace_published.return_value = True
        with (
            patch.object(common, "get_context", return_value=context),
            patch.object(
                common,
                "get_runtime_device",
                return_value=types.SimpleNamespace(device="npu"),
            ),
            patch.object(common, "is_cpu", side_effect=AssertionError("CPU probe")),
            patch.object(common, "is_cuda", side_effect=AssertionError("CUDA probe")),
            patch.object(
                torch,
                "npu",
                types.SimpleNamespace(is_available=lambda: False),
                create=True,
            ),
            patch.object(
                jpeg,
                "decode_jpeg",
                side_effect=RuntimeError("selected NPU unavailable"),
            ) as decode,
        ):
            with self.assertRaisesRegex(RuntimeError, "selected NPU unavailable"):
                common.load_image(self.data)
        decode.assert_called_once_with(self.data)

    def test_backend_and_mode_matrix(self):
        expected = torch.zeros((3, 32, 64), dtype=torch.uint8)
        cuda_module = types.ModuleType("torchvision.io")
        fancy_module = types.ModuleType("sglang.srt.utils.nvjpeg_decoder")
        for backend in ("cpu", "cuda", "npu"):
            for mode in (False, True, "nvjpeg_fancy"):
                cuda_decode = Mock(return_value=expected)
                fancy_decode = Mock(return_value=expected)
                cuda_module.decode_jpeg = cuda_decode
                fancy_module.decode_jpeg_with_fancy_upsampling = fancy_decode
                with (
                    self.subTest(backend=backend, mode=mode),
                    patch.object(common, "_image_decode_backend", return_value=backend),
                    patch.object(common, "is_cuda", return_value=backend == "cuda"),
                    patch.object(
                        common, "is_npu", side_effect=AssertionError("second NPU probe")
                    ),
                    patch.object(
                        jpeg, "decode_jpeg", return_value=expected
                    ) as npu_decode,
                    patch.dict(
                        "sys.modules",
                        {
                            "torchvision.io": cuda_module,
                            "sglang.srt.utils.nvjpeg_decoder": fancy_module,
                        },
                    ),
                ):
                    result, _ = common.load_image(self.data, device_image_decode=mode)
                use_npu = backend == "npu" and mode is True
                use_cuda = backend == "cuda" and mode is True
                use_fancy = backend == "cuda" and mode == "nvjpeg_fancy"
                if use_npu or use_cuda or use_fancy:
                    self.assertIs(result, expected)
                else:
                    self.assertIsInstance(result, Image.Image)
                    with Image.open(io.BytesIO(self.data)) as reference:
                        self.assertEqual(result.tobytes(), reference.tobytes())
                if use_npu:
                    npu_decode.assert_called_once_with(self.data)
                else:
                    npu_decode.assert_not_called()
                if use_cuda:
                    cuda_decode.assert_called_once()
                    encoded = cuda_decode.call_args.args[0]
                    self.assertEqual(encoded.dtype, torch.uint8)
                    self.assertEqual(bytes(encoded.tolist()), self.data)
                    self.assertEqual(cuda_decode.call_args.kwargs, {"device": "cuda"})
                else:
                    cuda_decode.assert_not_called()
                if use_fancy:
                    fancy_decode.assert_called_once_with(self.data)
                else:
                    fancy_decode.assert_not_called()

    def test_npu_unsupported_result_keeps_pil_pixels(self):
        with (
            patch.object(common, "_image_decode_backend", return_value="npu"),
            patch.object(jpeg, "decode_jpeg", return_value=None) as decode,
        ):
            result, _ = common.load_image(self.data)
        decode.assert_called_once_with(self.data)
        self.assertIsInstance(result, Image.Image)
        with Image.open(io.BytesIO(self.data)) as reference:
            self.assertEqual(result.tobytes(), reference.tobytes())

    def test_npu_execution_failure_remains_visible(self):
        with (
            patch.object(common, "_image_decode_backend", return_value="npu"),
            patch.object(
                jpeg, "decode_jpeg", side_effect=RuntimeError("device failed")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "device failed"):
                common.load_image(self.data)

    def test_pil_input_never_reenters_device_decode(self):
        image = Image.new("RGB", (64, 32))
        with patch.object(
            common, "_load_image", side_effect=AssertionError("redecode")
        ):
            result, _ = common.load_image(image, device_image_decode=True)
        self.assertIs(result, image)


class TestProcessorDecodePolicyMigration(unittest.TestCase):
    def test_legacy_override_has_explicit_migration_error(self):
        for old_mode in (False, True, "nvjpeg_fancy"):
            with self.subTest(mode=old_mode):
                with self.assertRaisesRegex(
                    TypeError, r"rename .*gpu_image_decode to device_image_decode"
                ):
                    type(
                        "LegacyProcessor",
                        (BaseMultimodalProcessor,),
                        {"gpu_image_decode": old_mode},
                    )

    def test_legacy_override_is_not_silently_accepted_with_new_name(self):
        with self.assertRaisesRegex(TypeError, r"rename .*gpu_image_decode"):
            type(
                "ConflictingProcessor",
                (BaseMultimodalProcessor,),
                {"gpu_image_decode": False, "device_image_decode": True},
            )

    def test_legacy_mixin_override_has_explicit_migration_error(self):
        class LegacyMixin:
            gpu_image_decode = False

        with self.assertRaisesRegex(TypeError, r"rename .*gpu_image_decode"):
            type("CustomProcessor", (LegacyMixin, BaseMultimodalProcessor), {})

    def test_new_override_survives_multiple_inheritance_levels(self):
        for mode in (False, True, "nvjpeg_fancy"):
            with self.subTest(mode=mode):
                parent = type(
                    "NewProcessor",
                    (BaseMultimodalProcessor,),
                    {"device_image_decode": mode},
                )
                child = type("ChildProcessor", (parent,), {})
                self.assertEqual(child.device_image_decode, mode)


if __name__ == "__main__":
    unittest.main()
