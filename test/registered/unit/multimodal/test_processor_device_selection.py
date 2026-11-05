"""The fast-image-processor device comes from the processor's own ServerArgs.

Regression: the device decision read the published global ServerArgs, so every
processor answered with one process-wide device. The encode-server DP workers
each drive their own GPU, which no process-global value can express — the
device has to come from what the worker was handed.
"""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.runtime_context import publish, reset_context
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


def _make(**fields):
    """Both surfaces, because the device decision reads both.

    `base_gpu_id` is the instance's own -- two engines in one process keep
    different ones, which `test_publishing_another_config_does_not_move_the_device`
    pins -- so it stays on the record the processor holds. `rl_on_policy_target`
    is the process's, so it is published.
    """
    server_args = ServerArgs(model_path="dummy", **fields)
    publish(server_args, role="tokenizer")
    processor = _StubProcessor.__new__(_StubProcessor)
    processor.server_args = server_args
    return processor


class TestFastImageProcessorDevice(CustomTestCase):
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
            return processor._fast_image_processor_device(_Processor())

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
            device = processor._fast_image_processor_device(Glm4vProcessor())
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


class TestForceCpuImagePreprocessing(CustomTestCase):
    """SGLANG_FORCE_CPU_IMAGE_PREPROCESSING keeps image decode and the fast
    image processor off the GPU, so tokenizer workers open no CUDA context."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _force(self, on):
        from sglang.srt.environ import envs

        return envs.SGLANG_FORCE_CPU_IMAGE_PREPROCESSING.override(on)

    def _cuda(self, nvidia=True):
        """A CUDA-like platform: NVIDIA CUDA, or ROCm (``nvidia=False``)."""
        platforms = SimpleNamespace(
            current_platform=SimpleNamespace(
                is_cuda_alike=lambda: True,
                is_cuda=lambda: nvidia,
                device_type="cuda",
            )
        )
        return patch.multiple(
            BASE,
            platforms=platforms,
            _is_cpu=False,
            _is_xpu=False,
            _is_npu=False,
        )

    def _with_fast_processor(self, **fields):
        from transformers import BaseImageProcessor

        processor = _make(base_gpu_id=3, **fields)
        processor.disable_fast_image_processor = False
        processor._processor = SimpleNamespace(image_processor=BaseImageProcessor())
        return processor

    def _construct(self, cls=None, tokenizer_worker_num=1, _processor=None):
        from unittest.mock import MagicMock

        cls = cls or _StubProcessor
        server_args = ServerArgs(
            model_path="dummy",
            tokenizer_worker_num=tokenizer_worker_num,
            mm_process_config={},
        )
        publish(server_args, role="tokenizer")
        with patch.object(cls, "__abstractmethods__", set()):
            processor = cls(
                hf_config=MagicMock(),
                server_args=server_args,
                _processor=_processor if _processor is not None else MagicMock(),
                transport_mode=None,
            )
        self.addCleanup(processor.shutdown)
        return processor

    def test_fast_processor_runs_on_the_cpu(self):
        # An explicit "cpu", not None: the processor would follow its input.
        processor = self._with_fast_processor()
        with self._cuda():
            with self._force(False):
                default = processor._fast_image_processor_device(_Processor())
            with self._force(True):
                forced = processor._fast_image_processor_device(_Processor())
        self.assertEqual(default, "cuda:3")
        self.assertEqual(forced, "cpu")

    def test_cpu_preprocessing_does_not_compete_with_the_scheduler(self):
        processor = self._with_fast_processor()
        with self._cuda():
            with self._force(False):
                default = processor._preprocessing_competes_with_the_scheduler()
            with self._force(True):
                forced = processor._preprocessing_competes_with_the_scheduler()
        self.assertTrue(default)
        self.assertFalse(forced)

    def test_instance_decodes_on_the_cpu(self):
        with self._cuda():
            with self._force(False):
                default = self._construct().gpu_image_decode
            with self._force(True):
                forced = self._construct().gpu_image_decode
        self.assertTrue(default)
        self.assertFalse(forced)
        self.assertTrue(_StubProcessor.gpu_image_decode)  # class left alone

    def test_loader_skips_nvjpeg(self):
        from sglang.srt.multimodal.modality import Modality

        class _FancyDecode(_StubProcessor):
            gpu_image_decode = "nvjpeg_fancy"

        class _CpuDecode(_StubProcessor):
            gpu_image_decode = False

        image = SimpleNamespace(mode="RGB", load=lambda: None)
        calls = {}
        for force in (False, True):
            with (
                self._cuda(),
                self._force(force),
                patch(f"{BASE}.load_image", return_value=(image, None)) as load,
            ):
                for cls in (_StubProcessor, _FancyDecode, _CpuDecode):
                    cls._load_single_item(b"jpeg", Modality.IMAGE)
            calls[force] = [c.args[1] for c in load.call_args_list]
        self.assertEqual(calls[False], [True, "nvjpeg_fancy", False])
        self.assertEqual(calls[True], [False, False, False])

    def test_warns_only_when_several_workers_use_the_gpu(self):
        from transformers import BaseImageProcessor

        class _CpuDecode(_StubProcessor):
            gpu_image_decode = False

        fast = SimpleNamespace(image_processor=BaseImageProcessor(), tokenizer=None)

        def warnings(force, workers, cls=None, _processor=None):
            with (
                self._cuda(),
                self._force(force),
                patch(f"{BASE}.logger.warning") as warning,
            ):
                self._construct(cls, workers, _processor)
            return [c.args[0] % c.args[1:] for c in warning.call_args_list]

        decode_only = warnings(False, 8)
        self.assertEqual(len(decode_only), 1)
        self.assertIn("8 tokenizer workers each decode JPEG images on", decode_only[0])
        self.assertNotIn("fast image processor", decode_only[0])
        self.assertIn("SGLANG_FORCE_CPU_IMAGE_PREPROCESSING=1", decode_only[0])
        fast_only = warnings(False, 8, _CpuDecode, fast)
        self.assertEqual(len(fast_only), 1)
        self.assertIn("each run the fast image processor on", fast_only[0])
        self.assertEqual(warnings(True, 8), [])
        self.assertEqual(warnings(False, 1), [])
        with (
            self._cuda(nvidia=False),
            self._force(False),
            patch(f"{BASE}.logger.warning") as warning,
        ):
            self._construct(tokenizer_worker_num=8)
            self._construct(_CpuDecode, 8, fast)
        self.assertFalse(warning.called)

    def test_other_platforms_keep_their_placement(self):
        """Honored on NVIDIA CUDA only; e.g. ROCm keeps the device path."""
        from sglang.srt.multimodal.modality import Modality

        processor = self._with_fast_processor()
        image = SimpleNamespace(mode="RGB", load=lambda: None)
        with (
            self._cuda(nvidia=False),
            self._force(True),
            patch(f"{BASE}.load_image", return_value=(image, None)) as load,
        ):
            device = processor._fast_image_processor_device(_Processor())
            competes = processor._preprocessing_competes_with_the_scheduler()
            decode = self._construct().gpu_image_decode
            _StubProcessor._load_single_item(b"jpeg", Modality.IMAGE)
        self.assertEqual(device, "cuda:3")
        self.assertTrue(competes)
        self.assertTrue(decode)
        self.assertEqual(load.call_args.args[1], True)


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


if __name__ == "__main__":
    unittest.main()
