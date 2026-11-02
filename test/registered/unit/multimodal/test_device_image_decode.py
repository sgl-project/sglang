"""CPU routing regressions for independent GPU and NPU JPEG policies."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import importlib
import io
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock, patch

import torch
from PIL import Image

from sglang.srt.hardware_backend.npu import jpeg
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
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

    def test_both_disabled_do_not_call_a_decoder(self):
        with (
            patch.object(common, "_image_decode_backend", return_value="npu"),
            patch.object(common, "is_cuda", side_effect=AssertionError("CUDA probe")),
            patch.object(jpeg, "decode_jpeg") as decode,
        ):
            result, _ = common.load_image(
                self.data, gpu_image_decode=False, npu_image_decode=False
            )
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (64, 32))
        decode.assert_not_called()

    def test_backend_and_independent_policy_matrix(self):
        expected = torch.zeros((3, 32, 64), dtype=torch.uint8)
        cuda_module = types.ModuleType("torchvision.io")
        fancy_module = types.ModuleType("sglang.srt.utils.nvjpeg_decoder")
        for backend in ("cpu", "cuda", "npu"):
            for gpu_mode in (False, True, "nvjpeg_fancy"):
                for npu_mode in (False, True):
                    cuda_decode = Mock(return_value=expected)
                    fancy_decode = Mock(return_value=expected)
                    cuda_module.decode_jpeg = cuda_decode
                    fancy_module.decode_jpeg_with_fancy_upsampling = fancy_decode
                    with (
                        self.subTest(backend=backend, gpu=gpu_mode, npu=npu_mode),
                        patch.object(
                            common, "_image_decode_backend", return_value=backend
                        ),
                        patch.object(common, "is_cuda", return_value=backend == "cuda"),
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
                        result, _ = common.load_image(
                            self.data, gpu_mode, npu_image_decode=npu_mode
                        )
                    use_npu = backend == "npu" and npu_mode
                    use_cuda = backend == "cuda" and gpu_mode is True
                    use_fancy = backend == "cuda" and gpu_mode == "nvjpeg_fancy"
                    if use_npu or use_cuda or use_fancy:
                        self.assertIs(result, expected)
                    else:
                        self.assertIsInstance(result, Image.Image)
                        with Image.open(io.BytesIO(self.data)) as reference:
                            self.assertEqual(result.tobytes(), reference.tobytes())
                    self.assertEqual(npu_decode.call_count, int(use_npu))
                    self.assertEqual(cuda_decode.call_count, int(use_cuda))
                    self.assertEqual(fancy_decode.call_count, int(use_fancy))
                    if use_npu:
                        npu_decode.assert_called_once_with(self.data)
                    if use_cuda:
                        encoded = cuda_decode.call_args.args[0]
                        self.assertEqual(encoded.dtype, torch.uint8)
                        self.assertEqual(bytes(encoded.tolist()), self.data)
                        self.assertEqual(
                            cuda_decode.call_args.kwargs, {"device": "cuda"}
                        )
                    if use_fancy:
                        fancy_decode.assert_called_once_with(self.data)

    def test_selected_npu_never_tries_cuda_even_when_cuda_is_available(self):
        for gpu_mode in (False, True, "nvjpeg_fancy"):
            for npu_mode in (False, True):
                with (
                    self.subTest(gpu=gpu_mode, npu=npu_mode),
                    patch.object(common, "_image_decode_backend", return_value="npu"),
                    patch.object(
                        common, "is_cuda", side_effect=AssertionError("CUDA probe")
                    ),
                    patch.object(jpeg, "decode_jpeg", return_value=None) as decode,
                ):
                    result, _ = common.load_image(
                        self.data, gpu_mode, npu_image_decode=npu_mode
                    )
                self.assertIsInstance(result, Image.Image)
                self.assertEqual(decode.call_count, int(npu_mode))
                with Image.open(io.BytesIO(self.data)) as reference:
                    self.assertEqual(result.tobytes(), reference.tobytes())

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
                common.load_image(self.data, npu_image_decode=True)
        decode.assert_called_once_with(self.data)

    def test_published_cpu_runtime_keeps_legacy_cuda_dispatch(self):
        expected = torch.zeros((3, 32, 64), dtype=torch.uint8)
        module = types.ModuleType("torchvision.io")
        for npu_mode in (False, True):
            module.decode_jpeg = Mock(return_value=expected)
            with (
                self.subTest(npu=npu_mode),
                patch.object(common, "_image_decode_backend", return_value="cpu"),
                patch.object(common, "is_cuda", return_value=True),
                patch.object(
                    jpeg, "decode_jpeg", side_effect=AssertionError("NPU decode")
                ),
                patch.dict("sys.modules", {"torchvision.io": module}),
            ):
                result, _ = common.load_image(
                    self.data, True, npu_image_decode=npu_mode
                )
            self.assertIs(result, expected)
            module.decode_jpeg.assert_called_once()

    def test_gpu_decode_errors_keep_original_pil_fallback(self):
        cuda_module = types.ModuleType("torchvision.io")
        fancy_module = types.ModuleType("sglang.srt.utils.nvjpeg_decoder")
        for gpu_mode in (True, "nvjpeg_fancy"):
            for npu_mode in (False, True):
                cuda_module.decode_jpeg = Mock(
                    side_effect=RuntimeError("nvJPEG failure")
                )
                fancy_module.decode_jpeg_with_fancy_upsampling = Mock(
                    side_effect=RuntimeError("fancy failure")
                )
                with (
                    self.subTest(gpu=gpu_mode, npu=npu_mode),
                    patch.object(common, "_image_decode_backend", return_value="cuda"),
                    patch.object(common, "is_cuda", return_value=True),
                    patch.object(
                        jpeg, "decode_jpeg", side_effect=AssertionError("NPU decode")
                    ),
                    patch.object(common, "_warn_fancy_jpeg_fallback") as fancy_warning,
                    patch.dict(
                        "sys.modules",
                        {
                            "torchvision.io": cuda_module,
                            "sglang.srt.utils.nvjpeg_decoder": fancy_module,
                        },
                    ),
                ):
                    result, _ = common.load_image(
                        self.data, gpu_mode, npu_image_decode=npu_mode
                    )
                self.assertIsInstance(result, Image.Image)
                self.assertEqual(
                    fancy_warning.call_count, int(gpu_mode == "nvjpeg_fancy")
                )
                with Image.open(io.BytesIO(self.data)) as reference:
                    self.assertEqual(result.tobytes(), reference.tobytes())

    def test_old_gpu_positional_and_keyword_calls_remain_supported(self):
        with patch.object(common, "_image_decode_backend", return_value="cpu"):
            positional, _ = common.load_image(self.data, False)
            keyword, _ = common.load_image(self.data, gpu_image_decode=False)
            private = common._load_image(self.data, "", False)
            self.assertEqual(positional.tobytes(), keyword.tobytes())
            self.assertEqual(positional.tobytes(), private.tobytes())
        with self.assertRaises(TypeError):
            common.load_image(self.data, False, True)

    def test_non_jpeg_and_invalid_image_keep_pil_results(self):
        output = io.BytesIO()
        Image.new("RGB", (64, 32), (31, 127, 235)).save(output, format="PNG")
        with (
            patch.object(common, "_image_decode_backend", return_value="cuda"),
            patch.object(common, "is_cuda", return_value=True),
            patch.object(jpeg, "decode_jpeg", side_effect=AssertionError("NPU decode")),
        ):
            image, _ = common.load_image(output.getvalue(), npu_image_decode=True)
            self.assertEqual(image.getpixel((0, 0)), (31, 127, 235))
            with self.assertRaisesRegex(ValueError, "Could not decode image"):
                common.load_image(b"invalid", npu_image_decode=True)

    def test_pil_input_never_reenters_device_decode(self):
        image = Image.new("RGB", (64, 32))
        with patch.object(
            common, "_load_image", side_effect=AssertionError("redecode")
        ):
            result, _ = common.load_image(
                image, gpu_image_decode=True, npu_image_decode=True
            )
        self.assertIs(result, image)


class TestProcessorIoDevice(unittest.TestCase):
    def test_both_submit_paths_use_each_instances_npu_device(self):
        state = threading.local()

        @contextmanager
        def device(index):
            previous = getattr(state, "device", 0)
            state.device = index
            try:
                yield
            finally:
                state.device = previous

        class Processor(BaseMultimodalProcessor):
            def _get_npu_image_decode_mode(self):
                return True

            @classmethod
            def _load_single_item(cls, data, modality, *args, **kwargs):
                assert kwargs["gpu_image_decode"] is True
                assert kwargs["npu_image_decode"] is True
                return getattr(state, "device", 0)

            async def process_mm_data_async(self, *args, **kwargs):
                raise NotImplementedError

        with (
            patch.object(
                torch, "npu", types.SimpleNamespace(device=device), create=True
            ),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            for index in (1, 0):
                processor = object.__new__(Processor)
                processor.server_args = types.SimpleNamespace(
                    device="npu", base_gpu_id=index
                )
                processor.io_executor = pool
                simple = processor._submit_mm_data_loading_tasks_simple(
                    [b"image"], Modality.IMAGE, None, True
                )
                self.assertEqual(simple[0][2].result(), index)
                futures, _ = processor.submit_data_loading_tasks(
                    ["<image>"],
                    MultimodalSpecialTokens(image_token="<image>"),
                    {Modality.IMAGE: iter([b"image"])},
                )
                self.assertEqual(futures[0].result(), index)
            self.assertEqual(
                pool.submit(lambda: getattr(state, "device", 0)).result(), 0
            )

    def test_other_backends_disabled_npu_and_other_modalities_do_not_bind(self):
        class Processor(BaseMultimodalProcessor):
            def _get_npu_image_decode_mode(self):
                return self.npu_image_decode

            @classmethod
            def _load_single_item(cls, data, modality, *args, **kwargs):
                return data

            async def process_mm_data_async(self, *args, **kwargs):
                raise NotImplementedError

        for backend, npu_mode, modality in [
            ("cpu", True, Modality.IMAGE),
            ("cuda", True, Modality.IMAGE),
            ("npu", False, Modality.IMAGE),
            ("npu", True, Modality.AUDIO),
            ("npu", True, Modality.VIDEO),
        ]:
            processor = object.__new__(Processor)
            processor.server_args = types.SimpleNamespace(device=backend, base_gpu_id=1)
            processor.npu_image_decode = npu_mode
            with (
                self.subTest(backend=backend, npu=npu_mode, modality=modality),
                patch.object(
                    torch,
                    "npu",
                    types.SimpleNamespace(
                        device=Mock(
                            side_effect=AssertionError("unexpected NPU binding")
                        )
                    ),
                    create=True,
                ),
            ):
                self.assertEqual(
                    processor._load_single_item_for_processor(b"data", modality),
                    b"data",
                )


class TestIndependentProcessorPolicies(unittest.TestCase):
    def test_unknown_product_loaders_preserve_gpu_and_disable_npu(self):
        from sglang.srt.disaggregation.encoder.preprocessor import EncoderPreprocessor
        from sglang.srt.multimodal.cache import snapshot_media
        from sglang.srt.multimodal.media_artifacts import MediaArtifactCacheMixin
        from sglang.srt.multimodal.processors.transformers_auto import (
            TransformersAutoMultimodalProcessor,
        )

        class Processor(MediaArtifactCacheMixin, BaseMultimodalProcessor):
            async def process_mm_data_async(self, *args, **kwargs):
                raise NotImplementedError

        image = Image.new("RGB", (2, 2))
        for backend in ("cpu", "cuda", "npu"):
            for gpu_mode in (False, True, "nvjpeg_fancy"):
                with self.subTest(backend=backend, gpu=gpu_mode):
                    processor = object.__new__(Processor)
                    processor.server_args = types.SimpleNamespace(
                        device=backend, base_gpu_id=0
                    )
                    processor.gpu_image_decode = gpu_mode
                    processor.npu_image_decode = (
                        True  # Unknown paths remain unqualified.
                    )
                    with patch(
                        "sglang.srt.multimodal.processors.base_processor.load_image",
                        return_value=(image, None),
                    ) as load:
                        processor._load_single_item_for_processor(
                            b"jpeg", Modality.IMAGE
                        )
                        load.assert_called_once_with(
                            b"jpeg", gpu_mode, npu_image_decode=False
                        )
                    with patch(
                        "sglang.srt.multimodal.media_artifacts.base.load_image",
                        return_value=(image, None),
                    ) as load:
                        processor.decode_media_snapshot(
                            snapshot_media(b"jpeg"), Modality.IMAGE
                        )
                        load.assert_called_once_with(
                            b"jpeg", gpu_mode, npu_image_decode=False
                        )
                    generic = object.__new__(TransformersAutoMultimodalProcessor)
                    generic.server_args = types.SimpleNamespace(device=backend)
                    generic.gpu_image_decode = gpu_mode
                    with patch(
                        "sglang.srt.multimodal.processors.transformers_auto.load_image",
                        return_value=(image, None),
                    ) as load:
                        generic._load_images([b"jpeg"])
                        load.assert_called_once_with(
                            b"jpeg", gpu_mode, npu_image_decode=False
                        )
                    encoder = object.__new__(EncoderPreprocessor)
                    encoder.device = backend
                    encoder.use_image_processor_gpu = True
                    encoder.encoder_media_processor_config = types.SimpleNamespace(
                        image_decode_mode=gpu_mode, preserve_media_metadata=False
                    )
                    with patch(
                        "sglang.srt.disaggregation.encoder.preprocessor.load_image",
                        return_value=(image, None),
                    ) as load:
                        encoder._load_single_item(b"jpeg", Modality.IMAGE)
                        load.assert_called_once_with(
                            b"jpeg", gpu_mode, npu_image_decode=False
                        )

    def test_fingerprint_records_both_policies_without_changing_class(self):
        class Processor(BaseMultimodalProcessor):
            def _get_npu_image_decode_mode(self):
                return self.npu_image_decode

            async def process_mm_data_async(self, *args, **kwargs):
                raise NotImplementedError

        payloads = []
        for enabled in (False, True):
            processor = object.__new__(Processor)
            processor.server_args = types.SimpleNamespace(device="npu")
            processor.npu_image_decode = enabled
            processor._processor = object()
            processor.image_processor_backend = "auto"
            processor.mm_feature_transport = "cpu"
            processor.image_config = processor.video_config = (
                processor.audio_config
            ) = {}
            payloads.append(processor.preprocess_fingerprint_payload())
        self.assertIs(payloads[0]["gpu_image_decode"], True)
        self.assertIs(payloads[1]["gpu_image_decode"], True)
        self.assertIs(payloads[0]["npu_image_decode"], False)
        self.assertIs(payloads[1]["npu_image_decode"], True)
        self.assertNotEqual(payloads[0], payloads[1])
        self.assertIs(Processor.gpu_image_decode, True)
        self.assertIs(Processor.npu_image_decode, False)
        original_payload = {
            key: value
            for key, value in payloads[0].items()
            if key != "npu_image_decode"
        }
        for backend in ("cpu", "cuda"):
            processor.server_args.device = backend
            with patch.object(
                processor,
                "_get_npu_image_decode_mode",
                side_effect=AssertionError("CPU/CUDA must not evaluate NPU policy"),
            ):
                self.assertEqual(
                    processor.preprocess_fingerprint_payload(), original_payload
                )

    def test_existing_gpu_overrides_and_mixins_remain_supported(self):
        for mode in (False, True, "nvjpeg_fancy"):
            parent = type(
                "ExistingProcessor",
                (BaseMultimodalProcessor,),
                {"gpu_image_decode": mode},
            )
            child = type("ChildProcessor", (parent,), {})
            mixin = type("ExistingMixin", (), {"gpu_image_decode": mode})
            mixed = type("MixedProcessor", (mixin, BaseMultimodalProcessor), {})
            self.assertEqual(child.gpu_image_decode, mode)
            self.assertEqual(mixed.gpu_image_decode, mode)
            self.assertIs(child.npu_image_decode, False)
            with patch(
                "sglang.srt.multimodal.processors.base_processor.load_image",
                return_value=(Image.new("RGB", (2, 2)), None),
            ) as load:
                child._load_single_item(b"jpeg", Modality.IMAGE)
            load.assert_called_once_with(b"jpeg", mode, npu_image_decode=False)

    def test_janus_phi_keep_original_gpu_and_npu_pil_policy(self):
        for module_name, class_name in (
            ("janus_pro", "JanusProImageProcessor"),
            ("phi4mm", "Phi4MMMultimodalProcessor"),
        ):
            processor_type = getattr(
                importlib.import_module(
                    f"sglang.srt.multimodal.processors.{module_name}"
                ),
                class_name,
            )
            self.assertIs(processor_type.gpu_image_decode, True)
            self.assertIs(processor_type.npu_image_decode, False)
            for backend in ("cuda", "npu"):
                processor = object.__new__(processor_type)
                processor.server_args = types.SimpleNamespace(
                    device=backend, base_gpu_id=0
                )
                self.assertIs(processor._get_npu_image_decode_mode(), False)
                with patch(
                    "sglang.srt.multimodal.processors.base_processor.load_image",
                    return_value=(Image.new("RGB", (2, 2)), None),
                ) as load:
                    processor._load_single_item_for_processor(b"jpeg", Modality.IMAGE)
                load.assert_called_once_with(b"jpeg", True, npu_image_decode=False)

    def test_qwen_qualification_is_independent_of_gpu_policy(self):
        from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

        image_type = type(
            "Qwen2VLImageProcessor",
            (),
            {"__module__": "transformers.models.qwen2_vl.image_processing_qwen2_vl"},
        )
        hf_type = type(
            "Qwen2_5_VLProcessor",
            (),
            {"__module__": "transformers.models.qwen2_5_vl.processing_qwen2_5_vl"},
        )
        image = image_type()
        image.backend = "torchvision"
        image.patch_size, image.temporal_patch_size, image.merge_size = 14, 2, 2
        image.do_resize = image.do_rescale = image.do_normalize = (
            image.do_convert_rgb
        ) = True
        image.resample = 3
        image.size = types.SimpleNamespace(shortest_edge=3136, longest_edge=12845056)
        image.image_mean = (0.48145466, 0.4578275, 0.40821073)
        image.image_std = (0.26862954, 0.26130258, 0.27577711)
        image.rescale_factor = 1 / 255
        hf = hf_type()
        hf.image_processor = image
        processor = object.__new__(QwenVLImageProcessor)
        processor.server_args = types.SimpleNamespace(device="npu", base_gpu_id=0)
        processor.hf_config = types.SimpleNamespace(model_type="qwen2_5_vl")
        processor._processor = hf
        processor.image_config = {}
        for gpu_mode in (False, True, "nvjpeg_fancy"):
            processor.gpu_image_decode = gpu_mode
            self.assertIs(processor._get_npu_image_decode_mode(), True)
            self.assertEqual(processor.gpu_image_decode, gpu_mode)
            from sglang.srt.multimodal.cache import snapshot_media

            with patch.object(
                torch,
                "npu",
                types.SimpleNamespace(device=lambda index: nullcontext()),
                create=True,
            ):
                with patch(
                    "sglang.srt.multimodal.processors.base_processor.load_image",
                    return_value=(Image.new("RGB", (2, 2)), None),
                ) as load:
                    processor._load_single_item_for_processor(b"jpeg", Modality.IMAGE)
                    load.assert_called_once_with(
                        b"jpeg", gpu_mode, npu_image_decode=True
                    )
                with patch(
                    "sglang.srt.multimodal.media_artifacts.base.load_image",
                    return_value=(Image.new("RGB", (2, 2)), None),
                ) as load:
                    processor.decode_media_snapshot(
                        snapshot_media(b"jpeg"), Modality.IMAGE
                    )
                    load.assert_called_once_with(
                        b"jpeg", gpu_mode, npu_image_decode=True
                    )
        processor.npu_image_decode = False
        self.assertIs(processor._get_npu_image_decode_mode(), False)
        processor.npu_image_decode = True
        for model_type in ("qwen3_vl", "qwen2_vl", "qwen2_5_omni"):
            processor.hf_config.model_type = model_type
            self.assertIs(processor._get_npu_image_decode_mode(), False)
        processor.hf_config.model_type = "qwen2_5_vl"
        processor.image_config = {"patch_size": 16}
        self.assertIs(processor._get_npu_image_decode_mode(), False)
        processor.image_config = {}
        image.backend = "pil"
        self.assertIs(processor._get_npu_image_decode_mode(), False)
        image.backend = "torchvision"
        image.do_resize = False
        self.assertIs(processor._get_npu_image_decode_mode(), False)
        image.do_resize = True

        class UnknownQwenSubclass(QwenVLImageProcessor):
            pass

        unknown = object.__new__(UnknownQwenSubclass)
        unknown.__dict__.update(processor.__dict__)
        self.assertIs(unknown._get_npu_image_decode_mode(), False)
        for backend in ("cuda", "cpu"):
            processor.server_args.device = backend
            processor._processor = None
            self.assertIs(processor._get_npu_image_decode_mode(), False)
        self.assertIs(QwenVLImageProcessor.gpu_image_decode, True)

    def test_gemma_glm_qualification_keeps_the_gpu_policy_independent(self):
        from sglang.srt.multimodal.processors.gemma3 import Gemma3SGLangImageProcessor
        from sglang.srt.multimodal.processors.glm4v import Glm4vImageProcessor

        cases = (
            (
                Gemma3SGLangImageProcessor,
                "gemma3",
                "Gemma3Processor",
                "transformers.models.gemma3.processing_gemma3",
                "Gemma3ImageProcessor",
                "transformers.models.gemma3.image_processing_gemma3",
                "torchvision",
            ),
            (
                Glm4vImageProcessor,
                "glm4v",
                "Glm4vProcessor",
                "transformers.models.glm4v.processing_glm4v",
                "Glm4vImageProcessor",
                "transformers.models.glm4v.image_processing_glm4v",
                "torchvision",
            ),
            (
                Glm4vImageProcessor,
                "glm4v",
                "Glm4vProcessor",
                "transformers.models.glm4v.processing_glm4v",
                "Glm4vImageProcessorPil",
                "transformers.models.glm4v.image_processing_pil_glm4v",
                "pil",
            ),
        )
        for (
            wrapper,
            model_type,
            hf_name,
            hf_module,
            image_name,
            image_module,
            backend,
        ) in cases:
            with self.subTest(wrapper=wrapper.__name__, backend=backend):
                image = type(image_name, (), {"__module__": image_module})()
                image.backend = backend
                image.do_resize = image.do_rescale = image.do_normalize = True
                image.rescale_factor = 1 / 255
                if model_type == "gemma3":
                    image.size = types.SimpleNamespace(height=896, width=896)
                    image.resample = 2
                    image.do_convert_rgb = image.do_pan_and_scan = None
                    image.image_mean = image.image_std = (0.5, 0.5, 0.5)
                else:
                    image.size = types.SimpleNamespace(
                        shortest_edge=12544, longest_edge=9633792
                    )
                    image.patch_size, image.temporal_patch_size, image.merge_size = (
                        14,
                        2,
                        2,
                    )
                    image.do_convert_rgb = True
                    image.image_mean = (0.48145466, 0.4578275, 0.40821073)
                    image.image_std = (0.26862954, 0.26130258, 0.27577711)
                hf = type(hf_name, (), {"__module__": hf_module})()
                hf.image_processor = image
                processor = object.__new__(wrapper)
                processor.server_args = types.SimpleNamespace(device="npu")
                processor.hf_config = types.SimpleNamespace(model_type=model_type)
                processor._processor = hf
                processor.image_config = {}
                processor.image_processor_backend = "auto"
                processor.disable_fast_image_processor = False
                for gpu_mode in (False, True, "nvjpeg_fancy"):
                    processor.gpu_image_decode = gpu_mode
                    self.assertIs(processor._get_npu_image_decode_mode(), True)
                    self.assertEqual(processor.gpu_image_decode, gpu_mode)
                processor.npu_image_decode = False
                self.assertIs(processor._get_npu_image_decode_mode(), False)
                processor.npu_image_decode = True
                processor.hf_config.model_type = "unverified"
                self.assertIs(processor._get_npu_image_decode_mode(), False)
                processor.hf_config.model_type = model_type
                processor.image_config = {"size": 512}
                self.assertIs(processor._get_npu_image_decode_mode(), False)
                processor.image_config = {}
                unknown_type = type("UnknownSubclass", (wrapper,), {})
                unknown = object.__new__(unknown_type)
                unknown.__dict__.update(processor.__dict__)
                self.assertIs(unknown._get_npu_image_decode_mode(), False)
                image.backend = "custom"
                self.assertIs(processor._get_npu_image_decode_mode(), False)
                for serving_backend in ("cuda", "cpu"):
                    processor.server_args.device = serving_backend
                    self.assertIs(processor._get_npu_image_decode_mode(), False)
                self.assertIs(wrapper.gpu_image_decode, True)

    def test_kimi_and_embedding_overrides_forward_both_policies(self):
        from sglang.srt.multimodal.processors.embedding_gemma2 import (
            EmbeddingGemma2SGLangProcessor,
        )
        from sglang.srt.multimodal.processors.kimi_common import KimiGridMMDataMixin

        class KimiProcessor(KimiGridMMDataMixin, BaseMultimodalProcessor):
            async def process_mm_data_async(self, *args, **kwargs):
                raise NotImplementedError

        for processor_type in (KimiProcessor, EmbeddingGemma2SGLangProcessor):
            image = Image.new("RGB", (2, 2))
            with patch(
                "sglang.srt.multimodal.processors.base_processor.load_image",
                return_value=(image, None),
            ) as load:
                result = processor_type._load_single_item(
                    b"jpeg",
                    Modality.IMAGE,
                    gpu_image_decode=False,
                    npu_image_decode=True,
                )
            self.assertIs(result, image)
            load.assert_called_once_with(b"jpeg", False, npu_image_decode=True)

    def test_auto_pil_backend_stages_only_npu_tensors(self):
        class InputTensor:
            def cpu(self):
                copies.append(self)
                return cpu_copy

        class HFProcessor:
            image_processor = types.SimpleNamespace(backend="pil")

            def __call__(self, **kwargs):
                captured.append(kwargs["images"])
                return {}

        class Processor(BaseMultimodalProcessor):
            async def process_mm_data_async(self, *args, **kwargs):
                raise NotImplementedError

        for device in ("npu", "cuda", "cpu"):
            with self.subTest(device=device):
                copies, captured = [], []
                cpu_copy, non_tensor = object(), object()
                decoded = InputTensor()
                decoded.device = types.SimpleNamespace(type=device)
                original = [decoded, non_tensor]
                processor = object.__new__(Processor)
                processor._processor = HFProcessor()
                processor._tokenizer = object()
                processor._tokenizer_auto_adds_specials = False
                processor.image_processor_backend = "auto"
                processor.disable_fast_image_processor = False
                processor.mm_feature_transport = "cpu"
                processor.image_config = {}
                processor.FEATURE_NAMES = []
                with patch(
                    "sglang.srt.multimodal.processors.base_processor.torch.Tensor",
                    InputTensor,
                ):
                    processor.process_mm_data("test", images=original)
                self.assertEqual(copies, [decoded] if device == "npu" else [])
                self.assertIs(captured[0][0], cpu_copy if device == "npu" else decoded)
                self.assertIs(captured[0][1], non_tensor)
                self.assertIs(original[0], decoded)


if __name__ == "__main__":
    unittest.main()
