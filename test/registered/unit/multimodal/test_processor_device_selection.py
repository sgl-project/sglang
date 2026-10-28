"""Where fast visual preprocessing runs, and who gets to decide.

The device comes from the processor's own ServerArgs (regression: it once read
the published global ServerArgs, so every processor answered with one
process-wide device, which the encode-server DP workers cannot express). One
resolver, `_resolve_mm_preprocessing_device`, ranks `--mm-preprocessing-device`
over the platform; `process_mm_data` hands its answer
to the HF processor call and the worker-count decision reads the same answer.
"""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.runtime_context import get_mm, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

BASE = "sglang.srt.multimodal.processors.base_processor"


class _Processor:
    pass


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class _CpuVideoProcessor(_StubProcessor):
    video_preprocessing_device = "cpu"


def _make(cls=_StubProcessor, **fields):
    """Both surfaces, because the device decision reads both.

    `base_gpu_id` is the instance's own -- two engines in one process keep
    different ones, which `test_publishing_another_config_does_not_move_the_device`
    pins -- so it stays on the record the processor holds. `rl_on_policy_target`
    is the process's, so it is published.
    """
    server_args = ServerArgs(model_path="dummy", **fields)
    publish(server_args, role="tokenizer")
    processor = cls.__new__(cls)
    processor.server_args = server_args
    return processor


class TestResolveMmPreprocessingDevice(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _device(self, processor, cuda_alike=True, device_type="cuda", **platform):
        flags = {"_is_cpu": False, "_is_xpu": False, "_is_npu": False}
        flags.update(platform)
        platforms = SimpleNamespace(
            current_platform=SimpleNamespace(
                is_cuda_alike=lambda: cuda_alike,
                device_type=device_type,
            )
        )
        with patch.multiple(BASE, platforms=platforms, **flags):
            return processor._resolve_mm_preprocessing_device(_Processor())

    def test_device_follows_the_instance_base_gpu_id(self):
        self.assertEqual(self._device(_make(base_gpu_id=3)), "cuda:3")

    def test_engines_in_one_process_keep_their_own_device(self):
        first, second = _make(base_gpu_id=0), _make(base_gpu_id=5)
        self.assertEqual(self._device(first), "cuda:0")
        self.assertEqual(self._device(second), "cuda:5")

    def test_publishing_another_config_does_not_move_the_device(self):
        from sglang.srt.runtime_context import get_context

        processor = _make(base_gpu_id=2)
        override = get_context().override_server_args(base_gpu_id=7)
        override.install()
        self.addCleanup(override.restore)
        self.assertEqual(self._device(processor), "cuda:2")

    def test_rl_on_policy_target_forces_cpu(self):
        processor = _make(base_gpu_id=3, rl_on_policy_target="fsdp")
        self.assertEqual(self._device(processor), "cpu")

    def test_cpu_and_xpu_platforms_win_over_base_gpu_id(self):
        processor = _make(base_gpu_id=3)
        self.assertEqual(self._device(processor, _is_cpu=True), "cpu")
        self.assertEqual(self._device(processor, _is_xpu=True), "xpu")

    def test_npu_glm4v_leaves_the_device_unset(self):
        class Glm4vProcessor:
            pass

        processor = _make(base_gpu_id=3)
        with patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=True):
            device = processor._resolve_mm_preprocessing_device(Glm4vProcessor())
        self.assertIsNone(device)

    def test_platform_respects_cuda_compatibility(self):
        processor = _make(base_gpu_id=3)
        for cuda_alike, device_type, expected in (
            (False, "other", None),
            (True, "cuda", "cuda:3"),
            (True, "musa", "musa:3"),
        ):
            with self.subTest(cuda_alike=cuda_alike, device_type=device_type):
                self.assertEqual(
                    self._device(
                        processor,
                        cuda_alike=cuda_alike,
                        device_type=device_type,
                    ),
                    expected,
                )

    def test_the_choice_is_pure_and_applies_no_npu_patches(self):
        """The constructor sizes the worker pool from the choice, so it must
        not import or apply the NPU processor patches; those run on the first
        processor call through `_resolve_mm_preprocessing_device`."""
        processor = _make(base_gpu_id=3)
        with (
            patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=True),
            patch.object(
                BaseMultimodalProcessor, "_apply_npu_processor_patches"
            ) as patches,
        ):
            self.assertEqual(processor._mm_preprocessing_device_choice(), "npu")
            self.assertFalse(patches.called)
            processor._resolve_mm_preprocessing_device(_Processor())
            self.assertTrue(patches.called)

    def test_invalid_server_setting_is_rejected(self):
        from sglang.srt.arg_groups.validation_hook import (
            validate_mm_preprocessing_device,
        )

        for bad in ("cpuu", "gpu", ""):
            with self.subTest(setting=bad):
                cfg = SimpleNamespace(mm_preprocessing_device=bad, device="cuda")
                with self.assertRaisesRegex(ValueError, "mm-preprocessing-device"):
                    validate_mm_preprocessing_device(cfg)
        with self.assertRaisesRegex(ValueError, "requires --device=cuda"):
            validate_mm_preprocessing_device(
                SimpleNamespace(mm_preprocessing_device="cuda", device="cpu")
            )
        for ok in ("auto", "cpu", "cuda"):
            validate_mm_preprocessing_device(
                SimpleNamespace(mm_preprocessing_device=ok, device="cuda")
            )

    def test_server_cpu_overrides_the_serving_gpu(self):
        processor = _make(base_gpu_id=3, mm_preprocessing_device="cpu")
        self.assertEqual(self._device(processor), "cpu")

    def test_server_cuda_uses_the_worker_gpu_ordinal(self):
        processor = _make(base_gpu_id=3, mm_preprocessing_device="cuda")
        self.assertEqual(self._device(processor), "cuda:3")

    def test_server_cpu_wins_over_the_platform(self):
        processor = _make(base_gpu_id=3, mm_preprocessing_device="cpu")
        self.assertEqual(self._device(processor, _is_xpu=True), "cpu")

    def test_server_auto_preserves_the_platform_default(self):
        self.assertEqual(ServerArgs(model_path="dummy").mm_preprocessing_device, "auto")
        auto = dict(base_gpu_id=3, mm_preprocessing_device="auto")
        self.assertEqual(self._device(_make(**auto)), "cuda:3")

    def test_resolved_setting_controls_decode_and_processor_together(self):
        processor = _make(base_gpu_id=3)
        with get_mm().override(mm_preprocessing_device="cpu"):
            self.assertEqual(self._device(processor), "cpu")
            self.assertFalse(processor._resolve_gpu_image_decode())
        self.assertEqual(self._device(processor), "cuda:3")

    def test_device_choice_changes_the_artifact_fingerprint(self):
        from sglang.srt.multimodal.cache.identity import build_processor_fingerprint

        fingerprints = []
        for setting in ("auto", "cpu", "auto"):
            processor = _make(mm_preprocessing_device=setting)
            processor._processor = _Processor()
            # Even a model without GPU decode can preprocess on CPU or GPU.
            processor.gpu_image_decode = False
            processor.image_processor_backend = "torchvision"
            processor.mm_feature_transport = "cpu"
            processor.image_config = {}
            processor.video_config = {}
            processor.audio_config = {}
            fingerprints.append(
                build_processor_fingerprint(
                    processor, SimpleNamespace(to_dict=lambda: {"model_type": "vlm"})
                )
            )
        self.assertNotEqual(fingerprints[0], fingerprints[1])
        self.assertEqual(fingerprints[0], fingerprints[2])


class _FastImageProcessor:
    pass


def _fast_path_processor(cls=_StubProcessor, **fields):
    processor = _make(cls, base_gpu_id=0, **fields)
    processor.disable_fast_image_processor = False
    processor._processor = SimpleNamespace(image_processor=_FastImageProcessor())
    return processor


@contextmanager
def _gpu_platform():
    with (
        patch.multiple(
            BASE,
            _is_cpu=False,
            _is_xpu=False,
            _is_npu=False,
            platforms=SimpleNamespace(
                current_platform=SimpleNamespace(
                    is_cuda_alike=lambda: True, device_type="cuda"
                )
            ),
        ),
        patch(f"{BASE}.BaseImageProcessor", _FastImageProcessor),
    ):
        yield


class TestPreprocessingCompetesWithTheScheduler(CustomTestCase):
    def _competes(self, processor):
        with _gpu_platform():
            competes = processor._preprocessing_competes_with_the_scheduler()
            workers = processor._resolve_auto_mm_processor_worker_num()
        return competes, workers

    def test_fast_processor_on_the_serving_gpu_competes(self):
        self.assertEqual(self._competes(_fast_path_processor()), (True, 1))

    def test_server_cpu_does_not_compete(self):
        processor = _fast_path_processor(mm_preprocessing_device="cpu")
        self.assertEqual(self._competes(processor), (False, 2))

    def test_server_cuda_competes(self):
        processor = _fast_path_processor(mm_preprocessing_device="cuda")
        self.assertEqual(self._competes(processor), (True, 1))

    def test_pil_backend_never_competes(self):
        processor = _fast_path_processor()
        processor.disable_fast_image_processor = True
        self.assertEqual(self._competes(processor), (False, 2))


class _Feature:
    def __init__(self, events):
        self.events = events

    def to(self, device):
        self.events.append(("copy", device))
        return self


class TestProcessMmDataDevice(CustomTestCase):
    """The device the HF processor call actually receives."""

    def _harness(self, cls=_StubProcessor, **fields):
        events = []
        feature = _Feature(events)

        class Processor:
            image_processor = _FastImageProcessor()
            tokenizer = SimpleNamespace(bos_token=None)

            def __call__(self, **kwargs):
                events.append(("call", kwargs.get("device"), sorted(kwargs)))
                return {"pixel_values": feature}

        processor = _make(cls, base_gpu_id=0, **fields)
        processor.mm_feature_transport = "cpu"
        processor.precompute_hash_before_cpu_transfer = False
        processor._processor = Processor()
        processor._tokenizer = processor._processor.tokenizer
        processor._tokenizer_auto_adds_specials = False
        processor.disable_fast_image_processor = False
        processor.image_config = {}
        processor.video_config = {}
        processor.audio_config = {}
        processor.FEATURE_NAMES = ["pixel_values"]
        return processor, events

    @contextmanager
    def _cuda_pool_spy(self):
        with (
            _gpu_platform(),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool") as mem_pool,
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=nullcontext()),
            patch(f"{BASE}.torch.Tensor", _Feature),
        ):
            yield mem_pool

    def _call_device(self, events):
        (call,) = [event for event in events if event[0] == "call"]
        return call[1], call[2]

    def test_unpinned_call_goes_to_the_serving_gpu_through_the_pool(self):
        processor, events = self._harness()
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", images=["image"])
        self.assertEqual(self._call_device(events)[0], "cuda:0")
        self.assertTrue(mem_pool.called)

    def test_server_cpu_reaches_the_call_and_skips_the_pool(self):
        processor, events = self._harness(mm_preprocessing_device="cpu")
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", images=["image"])
        self.assertEqual(self._call_device(events)[0], "cpu")
        self.assertFalse(mem_pool.called)
        self.assertIn(("copy", "cpu"), events)

    def test_server_cuda_reaches_the_call(self):
        processor, events = self._harness(mm_preprocessing_device="cuda")
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", images=["image"])
        self.assertEqual(self._call_device(events)[0], "cuda:0")
        self.assertTrue(mem_pool.called)

    def test_video_only_request_takes_the_same_device(self):
        """The device is a top-level kwarg of the whole call, so video moves too."""
        processor, events = self._harness(mm_preprocessing_device="cpu")
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", videos=["video"])
        device, kwargs = self._call_device(events)
        self.assertEqual(device, "cpu")
        self.assertIn("videos", kwargs)
        self.assertNotIn("images", kwargs)
        self.assertFalse(mem_pool.called)

    def test_cpu_video_override_does_not_enter_a_cuda_pool(self):
        """The active call device, not the earlier image default, owns the pool."""
        processor, events = self._harness(_CpuVideoProcessor)
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", videos=["video"])
        self.assertEqual(self._call_device(events)[0], "cpu")
        self.assertFalse(mem_pool.called)

    def test_pil_backend_passes_no_device(self):
        processor, events = self._harness(mm_preprocessing_device="cpu")
        processor.disable_fast_image_processor = True
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", images=["image"])
        device, kwargs = self._call_device(events)
        self.assertIsNone(device)
        self.assertNotIn("device", kwargs)
        self.assertFalse(mem_pool.called)

    def test_gemma4_delegated_call_follows_the_setting(self):
        """Gemma4 overrides ``process_mm_data`` only to prepare its inputs and
        then delegates to the base call, so the setting reaches that call and
        the worker count follows it without a marker override."""
        from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor

        for setting, device, competes in (
            ("auto", "cuda:0", True),
            ("cpu", "cpu", False),
        ):
            with self.subTest(setting=setting):
                processor, events = self._harness(
                    Gemma4SGLangProcessor, mm_preprocessing_device=setting
                )
                with self._cuda_pool_spy():
                    processor.process_mm_data("t", images=["image"])
                    self.assertEqual(
                        processor._preprocessing_competes_with_the_scheduler(),
                        competes,
                    )
                self.assertEqual(self._call_device(events)[0], device)


class TestGpuImageDecodeFollowsThePlacement(CustomTestCase):
    def _decode(self, processor):
        with patch.multiple(
            BASE,
            _is_cpu=False,
            _is_xpu=False,
            _is_npu=False,
            platforms=SimpleNamespace(
                current_platform=SimpleNamespace(
                    is_cuda_alike=lambda: True, device_type="cuda"
                )
            ),
        ):
            return processor._resolve_gpu_image_decode()

    def test_server_cpu_turns_gpu_decode_off(self):
        self.assertTrue(_StubProcessor.gpu_image_decode)
        self.assertFalse(self._decode(_make(mm_preprocessing_device="cpu")))

    def test_other_placements_keep_the_class_default(self):
        class NoGpuDecode(_StubProcessor):
            gpu_image_decode = False

        for setting in ("auto", "cuda"):
            with self.subTest(setting=setting):
                self.assertTrue(self._decode(_make(mm_preprocessing_device=setting)))
                self.assertFalse(
                    self._decode(_make(NoGpuDecode, mm_preprocessing_device=setting))
                )

    def test_load_single_item_uses_decode_mode_and_forces_pil_decode(self):
        image = SimpleNamespace(mode="RGB", load=Mock())
        with patch(f"{BASE}.load_image", return_value=(image, None)) as load_image:
            _StubProcessor._load_single_item(
                "data", Modality.IMAGE, gpu_image_decode=False
            )
            _StubProcessor._load_single_item("data", Modality.IMAGE)
        self.assertEqual(
            [call.args[1] for call in load_image.call_args_list], [False, True]
        )
        self.assertEqual(image.load.call_count, 2)


class TestFastImageProcessorMemoryPool(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _processor(self, *, transport="cpu", precompute_hash=False):
        processor = _make(base_gpu_id=0)
        processor.mm_feature_transport = transport
        processor.precompute_hash_before_cpu_transfer = precompute_hash
        return processor

    def test_pool_is_limited_to_immediate_cpu_transport(self):
        cases = (
            (self._processor(), "cuda:0", True),
            (self._processor(transport="cuda_ipc"), "cuda:0", False),
            (self._processor(transport="cuda_vmm"), "cuda:0", False),
            (self._processor(precompute_hash=True), "cuda:0", False),
            (self._processor(), "cpu", False),
            (self._processor(), None, False),
        )
        for processor, device, expected in cases:
            with (
                self.subTest(device=device, transport=processor.mm_feature_transport),
                patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
                patch(f"{BASE}.torch.cuda.MemPool", return_value="pool") as mem_pool,
                patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=nullcontext()),
            ):
                with processor._temporary_fast_processor_cuda_pool(device):
                    pass
                self.assertEqual(mem_pool.called, expected)

    def test_processor_call_uses_private_pool_until_cpu_copy_finishes(self):
        class ImageProcessor:
            pass

        class Feature:
            def to(self, device):
                events.append(("copy", device))

        feature = Feature()

        class Processor:
            image_processor = ImageProcessor()
            tokenizer = SimpleNamespace(bos_token=None)

            def __call__(self, **kwargs):
                events.append(("call", kwargs["device"]))
                return {"pixel_values": feature}

        events = []
        processor = self._processor()
        processor._processor = Processor()
        processor._tokenizer = processor._processor.tokenizer
        processor._tokenizer_auto_adds_specials = False
        processor.disable_fast_image_processor = False
        processor.image_config = {}
        processor.video_config = {}
        processor.audio_config = {}
        processor.FEATURE_NAMES = ["pixel_values"]

        class PoolContext:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("exit")

        with (
            patch.multiple(
                BASE,
                _is_cpu=False,
                _is_xpu=False,
                _is_npu=False,
                platforms=SimpleNamespace(
                    current_platform=SimpleNamespace(
                        is_cuda_alike=lambda: True,
                        device_type="cuda",
                    )
                ),
            ),
            patch(f"{BASE}.BaseImageProcessor", ImageProcessor),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool"),
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=PoolContext()),
            patch(f"{BASE}.torch.Tensor", Feature),
        ):
            processor.process_mm_data("test", images=["image"])

        self.assertEqual(
            events,
            ["enter", ("call", "cuda:0"), ("copy", "cpu"), "exit"],
        )


class TestModelProcessorsUseTheResolver(CustomTestCase):
    """Model processors that build their own fast processor call pass the same
    resolved device (regression: they called a helper the resolver replaced,
    so the first image request raised AttributeError)."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _qwen(self, **fields):
        from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

        calls = []

        class ImageProcessor(_FastImageProcessor):
            def __call__(self, images, **kwargs):
                calls.append(kwargs.get("device"))
                return {
                    "pixel_values": torch.zeros(4 * len(images), 2),
                    "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
                }

        processor = _make(QwenVLImageProcessor, base_gpu_id=0, **fields)
        processor.disable_fast_image_processor = False
        processor.image_config = {}
        processor.mm_feature_transport = "cpu"
        processor.precompute_hash_before_cpu_transfer = False
        processor.mm_preprocess_cache = SimpleNamespace(enabled=False)
        processor._processor = SimpleNamespace(image_processor=ImageProcessor())
        processor._tokenizer = None
        return processor, calls

    def _qwen_artifact_device(self, **fields):
        from sglang.srt.multimodal.media_artifacts import MediaArtifactInput

        processor, calls = self._qwen(**fields)
        entry = MediaArtifactInput(
            content_digest="sha256:" + "0" * 64,
            artifact_key="sha256:" + "1" * 64,
            modality=Modality.IMAGE,
            media="image",
        )
        with (
            _gpu_platform(),
            patch(
                "sglang.srt.multimodal.processors.qwen_vl.BaseImageProcessor",
                _FastImageProcessor,
            ),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool"),
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=nullcontext()),
        ):
            (artifact,) = processor.prepare_artifact_batch([entry])
        self.assertEqual(tuple(artifact.feature.shape), (4, 2))
        (device,) = calls
        return device

    def test_qwen_artifact_preprocessing_follows_the_resolver(self):
        for setting, expected in (("auto", "cuda:0"), ("cpu", "cpu")):
            with self.subTest(setting=setting):
                self.assertEqual(
                    self._qwen_artifact_device(mm_preprocessing_device=setting),
                    expected,
                )

    def _bailing_call_device(self, **fields):
        from sglang.srt.multimodal.processors.bailing_mm import (
            BailingMMMultimodalProcessor,
        )

        calls = []

        class Processor:
            image_processor = _FastImageProcessor()

            def __call__(self, **kwargs):
                calls.append(kwargs.get("device"))
                return {"pixel_values": torch.zeros(4, 2)}

        processor = _make(BailingMMMultimodalProcessor, base_gpu_id=0, **fields)
        processor.mm_feature_transport = "cpu"
        processor.FEATURE_NAMES = ["pixel_values"]
        processor._processor = Processor()
        processor._tokenizer = None
        with (
            _gpu_platform(),
            patch(
                "sglang.srt.multimodal.processors.bailing_mm.BaseImageProcessor",
                _FastImageProcessor,
            ),
        ):
            processor.process_mm_data("t", images=["image"])
            choice = processor._mm_preprocessing_device_choice()
        (device,) = calls
        self.assertEqual(device, choice)
        return device

    def test_bailing_visual_call_follows_the_resolver(self):
        for setting, expected in (("auto", "cuda:0"), ("cpu", "cpu")):
            with self.subTest(setting=setting):
                self.assertEqual(
                    self._bailing_call_device(mm_preprocessing_device=setting),
                    expected,
                )

    def test_ernie_cpu_setting_reaches_preprocessing_and_preserves_normalization(self):
        """ERNIE's hardcoded CUDA call used to ignore explicit CPU placement."""
        from sglang.srt.multimodal.processors.ernie45_vl import (
            Ernie4_5_VLImageProcessor,
        )

        calls = []
        pixels = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        class Processor:
            image_processor = _FastImageProcessor()

            def __call__(self, **kwargs):
                calls.append(kwargs.get("device"))
                return {
                    "images": pixels.clone(),
                    "grid_thw": torch.tensor([[1, 1, 2]]),
                }

        for setting, expected in (("cpu", "cpu"), ("auto", "cuda:3")):
            with self.subTest(setting=setting):
                processor = _make(
                    Ernie4_5_VLImageProcessor,
                    base_gpu_id=3,
                    mm_preprocessing_device=setting,
                )
                processor._processor = Processor()
                processor._tokenizer = None
                processor.disable_fast_image_processor = False
                processor.image_config = {}
                processor.use_cuda_ipc = False
                processor.FEATURE_NAMES = ["pixel_values", "pixel_values_videos"]
                processor.image_processor = SimpleNamespace(
                    image_mean=[0.5] * 3,
                    image_std=[0.25] * 3,
                    rescale_factor=1 / 255,
                )
                processor.hf_config = SimpleNamespace(
                    dtype=torch.float32,
                    vision_config=SimpleNamespace(patch_size=2),
                )
                with (
                    _gpu_platform(),
                    patch("sglang.srt.multimodal.processors.ernie45_vl._is_npu", False),
                ):
                    result = processor.process_mm_data("t", images=["image"])
                self.assertEqual(calls[-1], expected)
                torch.testing.assert_close(
                    result["pixel_values"], (pixels / 255 - 0.5) / 0.25
                )
                self.assertEqual(result["image_grid_thw"].tolist(), [[1, 1, 2]])
                self.assertNotIn("pixel_values_videos", result)


if __name__ == "__main__":
    unittest.main()
