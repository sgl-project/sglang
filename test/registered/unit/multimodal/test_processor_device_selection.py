"""Where fast visual preprocessing runs, and who gets to decide.

The device comes from the processor's own ServerArgs (regression: it once read
the published global ServerArgs, so every processor answered with one
process-wide device, which the encode-server DP workers cannot express). One
resolver, `_resolve_mm_preprocessing_device`, ranks `--mm-preprocessing-device`
over the class default over the platform; `process_mm_data` hands its answer
to the HF processor call and the worker-count decision reads the same answer.
"""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BASE = "sglang.srt.multimodal.processors.base_processor"


class _Processor:
    pass


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class _CpuDefaultProcessor(_StubProcessor):
    mm_preprocessing_device = "cpu"


class _Cuda1DefaultProcessor(_StubProcessor):
    mm_preprocessing_device = "cuda:1"


def _make(cls=_StubProcessor, **fields):
    processor = cls.__new__(cls)
    processor.server_args = ServerArgs(model_path="dummy", **fields)
    return processor


class TestResolveMmPreprocessingDevice(CustomTestCase):
    def _device(self, processor, **platform):
        flags = {"_is_cpu": False, "_is_xpu": False, "_is_npu": False}
        flags.update(platform)
        with patch.multiple(BASE, **flags):
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

    def test_class_default_replaces_the_serving_gpu_on_cuda_platforms(self):
        self.assertIsNone(_StubProcessor.mm_preprocessing_device)
        self.assertEqual(self._device(_make(base_gpu_id=3)), "cuda:3")
        self.assertEqual(
            self._device(_make(_CpuDefaultProcessor, base_gpu_id=3)), "cpu"
        )
        self.assertEqual(
            self._device(_make(_Cuda1DefaultProcessor, base_gpu_id=3)), "cuda:1"
        )

    def test_class_default_yields_to_cpu_xpu_and_npu_platforms(self):
        processor = _make(_Cuda1DefaultProcessor, base_gpu_id=3)
        self.assertEqual(self._device(processor, _is_cpu=True), "cpu")
        self.assertEqual(self._device(processor, _is_xpu=True), "xpu")
        with patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=True):
            self.assertEqual(processor._mm_preprocessing_device_choice(), "npu")

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

    def test_server_cpu_wins_over_the_class_default(self):
        processor = _make(
            _Cuda1DefaultProcessor, base_gpu_id=3, mm_preprocessing_device="cpu"
        )
        self.assertEqual(self._device(processor), "cpu")

    def test_server_cuda_is_the_serving_gpu_and_wins_over_the_class_default(self):
        processor = _make(
            _CpuDefaultProcessor, base_gpu_id=3, mm_preprocessing_device="cuda"
        )
        self.assertEqual(self._device(processor), "cuda:3")

    def test_server_cpu_wins_over_the_platform(self):
        processor = _make(base_gpu_id=3, mm_preprocessing_device="cpu")
        self.assertEqual(self._device(processor, _is_xpu=True), "cpu")

    def test_server_auto_defers_to_class_default_then_platform(self):
        self.assertEqual(ServerArgs(model_path="dummy").mm_preprocessing_device, "auto")
        auto = dict(base_gpu_id=3, mm_preprocessing_device="auto")
        self.assertEqual(self._device(_make(_CpuDefaultProcessor, **auto)), "cpu")
        self.assertEqual(self._device(_make(**auto)), "cuda:3")


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
        patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=False),
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

    def test_class_cpu_default_does_not_compete(self):
        processor = _fast_path_processor(_CpuDefaultProcessor)
        self.assertEqual(self._competes(processor), (False, 2))

    def test_server_cpu_does_not_compete(self):
        processor = _fast_path_processor(mm_preprocessing_device="cpu")
        self.assertEqual(self._competes(processor), (False, 2))

    def test_server_cuda_competes_despite_a_class_cpu_default(self):
        processor = _fast_path_processor(
            _CpuDefaultProcessor, mm_preprocessing_device="cuda"
        )
        self.assertEqual(self._competes(processor), (True, 1))

    def test_pil_backend_never_competes(self):
        processor = _fast_path_processor()
        processor.disable_fast_image_processor = True
        self.assertEqual(self._competes(processor), (False, 2))

    def test_the_choice_override_is_what_counts(self):
        """A subclass that places the work itself reports it via the choice."""

        class PlacesOnTheGpu(_CpuDefaultProcessor):
            def _mm_preprocessing_device_choice(self):
                return "cuda:0"

        processor = _fast_path_processor(PlacesOnTheGpu)
        self.assertEqual(self._competes(processor), (True, 1))

    def test_a_class_placing_its_own_call_is_sized_for_the_platform(self):
        """Neither the server setting nor a class default reaches the fast
        processor call of a class with its own ``process_mm_data``, so the
        worker count assumes the platform device (the serving GPU here)."""

        class OwnPlacement(_CpuDefaultProcessor):
            def process_mm_data(self, *args, **kwargs):
                raise NotImplementedError

        processor = _fast_path_processor(OwnPlacement, mm_preprocessing_device="cpu")
        self.assertTrue(processor._places_preprocessing_itself())
        self.assertEqual(self._competes(processor), (True, 1))
        self.assertFalse(_fast_path_processor()._places_preprocessing_itself())


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

    def test_class_cpu_default_reaches_the_call_and_skips_the_pool(self):
        processor, events = self._harness(_CpuDefaultProcessor)
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", images=["image"])
        self.assertEqual(self._call_device(events)[0], "cpu")
        self.assertFalse(mem_pool.called)
        self.assertIn(("copy", "cpu"), events)

    def test_server_cpu_reaches_the_call_and_skips_the_pool(self):
        processor, events = self._harness(mm_preprocessing_device="cpu")
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", images=["image"])
        self.assertEqual(self._call_device(events)[0], "cpu")
        self.assertFalse(mem_pool.called)

    def test_server_cuda_reaches_the_call_over_a_class_cpu_default(self):
        processor, events = self._harness(
            _CpuDefaultProcessor, mm_preprocessing_device="cuda"
        )
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

    def test_pil_backend_passes_no_device(self):
        processor, events = self._harness(mm_preprocessing_device="cpu")
        processor.disable_fast_image_processor = True
        with self._cuda_pool_spy() as mem_pool:
            processor.process_mm_data("t", images=["image"])
        device, kwargs = self._call_device(events)
        self.assertIsNone(device)
        self.assertNotIn("device", kwargs)
        self.assertFalse(mem_pool.called)


class TestGpuImageDecodeFollowsThePlacement(CustomTestCase):
    def _decode(self, processor):
        with patch.multiple(BASE, _is_cpu=False, _is_xpu=False, _is_npu=False):
            return processor._resolve_gpu_image_decode()

    def test_server_cpu_turns_gpu_decode_off(self):
        self.assertTrue(_StubProcessor.gpu_image_decode)
        self.assertFalse(self._decode(_make(mm_preprocessing_device="cpu")))

    def test_class_cpu_default_turns_gpu_decode_off(self):
        self.assertFalse(self._decode(_make(_CpuDefaultProcessor)))
        # The server setting outranks the class default in both directions.
        self.assertTrue(
            self._decode(_make(_CpuDefaultProcessor, mm_preprocessing_device="cuda"))
        )

    def test_other_placements_keep_the_class_default(self):
        class NoGpuDecode(_StubProcessor):
            gpu_image_decode = False

        for setting in ("auto", "cuda"):
            with self.subTest(setting=setting):
                self.assertTrue(self._decode(_make(mm_preprocessing_device=setting)))
                self.assertFalse(
                    self._decode(_make(NoGpuDecode, mm_preprocessing_device=setting))
                )

    def test_load_single_item_uses_the_instance_mode(self):
        image = SimpleNamespace(mode="RGB")
        with patch(f"{BASE}.load_image", return_value=(image, None)) as load_image:
            _StubProcessor._load_single_item(
                "data", Modality.IMAGE, gpu_image_decode=False
            )
            _StubProcessor._load_single_item("data", Modality.IMAGE)
        self.assertEqual(
            [call.args[1] for call in load_image.call_args_list], [False, True]
        )


class TestFastImageProcessorMemoryPool(CustomTestCase):
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


if __name__ == "__main__":
    unittest.main()
