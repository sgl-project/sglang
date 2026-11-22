"""The fast-image-processor device comes from the processor's own ServerArgs.

Regression: the device decision read the published global ServerArgs, so every
processor answered with one process-wide device. The encode-server DP workers
each drive their own GPU, which no process-global value can express — the
device has to come from what the worker was handed.
"""

import asyncio
import unittest
import weakref
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch
from transformers import BaseImageProcessor, PreTrainedTokenizerFast

from sglang.srt.environ import envs
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.srt.multimodal.token_space_mm_process.processor import (
    TokenSpaceProcessor,
)
from sglang.srt.multimodal.token_space_mm_process.serving_processor import (
    TokenSpaceSGLangProcessor,
)
from sglang.srt.multimodal.token_space_mm_process.serving_spec import (
    TokenSpaceServingSpec,
)
from sglang.srt.multimodal.transport.feature_transport import (
    MultimodalFeatureTransport,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

BASE = "sglang.srt.multimodal.processors.base_processor"
PLACEMENT = "sglang.srt.multimodal.serving_preprocessing"


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


def _make_wrapper(**fields):
    wrapper = TokenSpaceSGLangProcessor.__new__(TokenSpaceSGLangProcessor)
    wrapper._server_args = _make(**fields).server_args
    return wrapper


def _gpu_placement():
    """Patch placement so the fast image processor picks cuda:0 on any host."""
    return patch.multiple(
        PLACEMENT,
        _is_cpu=False,
        _is_xpu=False,
        _is_npu=False,
        platforms=SimpleNamespace(
            current_platform=SimpleNamespace(
                is_cuda_alike=lambda: True, device_type="cuda"
            )
        ),
    )


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
        with patch.multiple(PLACEMENT, platforms=platforms, **flags):
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

    def test_npu_glm4v_preprocesses_on_cpu(self):
        class Glm4vProcessor:
            pass

        processor = _make(base_gpu_id=3)
        with patch.multiple(PLACEMENT, _is_cpu=False, _is_xpu=False, _is_npu=True):
            self.assertEqual(
                processor._fast_image_processor_device(Glm4vProcessor()), "cpu"
            )

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


class TestTokenSpaceDevices(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def test_npu_wrapper_keeps_ordinary_processor_on_npu(self):
        wrapper = _make_wrapper(base_gpu_id=3)
        processor = _Processor()
        processor.image_processor = object.__new__(BaseImageProcessor)
        wrapper._core = SimpleNamespace(_processor=processor)
        wrapper._disable_fast_image_processor = False
        wrapper._rl_on_policy_target = None
        with (
            patch.multiple(PLACEMENT, _is_cpu=False, _is_xpu=False, _is_npu=True),
            patch(
                "sglang.srt.multimodal.token_space_mm_process.serving_processor.apply_npu_image_preprocess_patches"
            ),
        ):
            self.assertEqual(wrapper._serving_image_processor_device(), "npu")

    def test_wrapper_fills_the_image_device_unless_pil_decodes(self):
        for use_pil, expected in ((False, "cuda:3"), (True, None)):
            wrapper = _make_wrapper(base_gpu_id=3)
            wrapper._core = SimpleNamespace(
                _processor=SimpleNamespace(
                    image_processor=object.__new__(BaseImageProcessor)
                )
            )
            wrapper._disable_fast_image_processor = use_pil
            wrapper._rl_on_policy_target = None
            platforms = SimpleNamespace(
                current_platform=SimpleNamespace(
                    is_cuda_alike=lambda: True, device_type="cuda"
                )
            )
            with (
                self.subTest(use_pil=use_pil),
                patch.multiple(
                    PLACEMENT,
                    platforms=platforms,
                    _is_cpu=False,
                    _is_xpu=False,
                    _is_npu=False,
                ),
            ):
                self.assertEqual(wrapper._serving_image_processor_device(), expected)

    def test_wrapper_media_threads_start_on_the_instance_gpu(self):
        """Pool threads otherwise decode media on cuda:0 whatever base_gpu_id is."""
        server_args = ServerArgs(
            model_path="dummy",
            base_gpu_id=3,
            mm_process_config={},
            enable_token_space_processor=True,
            mm_processor_worker_num=1,
        )
        publish(server_args, role="test")
        with patch(
            "sglang.srt.multimodal.token_space_mm_process.serving_processor.is_cuda_alike",
            return_value=True,
        ):
            wrapper = TokenSpaceSGLangProcessor.for_strategy_and_serving_spec(
                TokenSpaceProcessStrategy, TokenSpaceServingSpec
            )(
                None,
                server_args,
                PreTrainedTokenizerFast(tokenizer_object=_empty_tokenizer()),
                None,
            )
        self.addCleanup(wrapper.shutdown)
        self.assertEqual(wrapper._core.io_executor._initializer.args, (3,))

    def test_video_requests_use_the_configured_video_device(self):
        # As in process_mm_data: a video request runs all its media on that device.
        for config_device, spec_device, expected in (
            (None, None, None),
            (None, "cpu", "cpu"),
            ("cpu", None, "cpu"),
            ("cuda:1", "cpu", "cuda:1"),
        ):

            class _Spec(TokenSpaceServingSpec):
                video_preprocessing_device = spec_device

            mm_process_config = (
                {}
                if config_device is None
                else {"video_preprocessing_device": config_device}
            )
            server_args = ServerArgs(
                model_path="dummy",
                mm_process_config=mm_process_config,
                enable_token_space_processor=True,
                mm_processor_worker_num=1,
            )
            publish(server_args, role="test")
            wrapper = TokenSpaceSGLangProcessor.for_strategy_and_serving_spec(
                TokenSpaceProcessStrategy, _Spec
            )(
                None,
                server_args,
                PreTrainedTokenizerFast(tokenizer_object=_empty_tokenizer()),
                None,
            )
            self.addCleanup(wrapper.shutdown)
            wrapper._image_device = "cuda:5"
            for video_data, image_device, video_device in (
                (["video"], expected or "cuda:5", expected),
                (None, "cuda:5", None),
            ):
                with (
                    self.subTest(
                        config_device=config_device,
                        spec_device=spec_device,
                        video=bool(video_data),
                    ),
                    patch.object(wrapper._core, "process", AsyncMock()) as process,
                ):
                    asyncio.run(
                        wrapper.process_mm_data_async(
                            image_data=["image"], video_data=video_data, input_ids=[1]
                        )
                    )
                self.assertEqual(process.call_args.kwargs["image_device"], image_device)
                self.assertEqual(process.call_args.kwargs["video_device"], video_device)

    def test_videos_decode_on_cpu_only_when_their_device_is_cpu(self):
        """The decode device can change pixels, so the trainer must decode as rollout did."""
        processor = TokenSpaceProcessor.for_strategy(TokenSpaceProcessStrategy)(
            None,
            PreTrainedTokenizerFast(tokenizer_object=_empty_tokenizer()),
            processor_config=MultimodalProcessorConfig(mm_processor_worker_num=1),
        )
        self.addCleanup(processor.shutdown)
        for video_device, use_gpu in ((None, True), ("cuda:3", True), ("cpu", False)):
            with (
                self.subTest(video_device=video_device),
                patch(
                    "sglang.srt.multimodal.multimodal_processor_mixin.load_video"
                ) as load_video,
                self.assertRaises(_Stop),
                patch.object(processor, "_list_media_items", side_effect=_Stop),
            ):
                asyncio.run(
                    processor.process(
                        None, ["clip.mp4"], input_ids=[1], video_device=video_device
                    )
                )
            load_video.assert_called_once_with("clip.mp4", use_gpu=use_gpu)


class _Stop(Exception):
    pass


def _empty_tokenizer():
    from tokenizers import Tokenizer, models

    return Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))


class TestFastImageProcessorMemoryPool(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _processor(self, *, transport="cpu", precompute_hash=False):
        processor = _make(base_gpu_id=0)
        processor._feature_transport = MultimodalFeatureTransport(
            transport, base_gpu_id=0, skip_mm_pool=True
        )
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
                self.subTest(
                    device=device,
                    transport=processor._feature_transport.mm_feature_transport,
                ),
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
                PLACEMENT,
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


class _ImageProcessor:
    pass


def _image_processor_stub(gpu_error=None, *, retry=True):
    """A base processor whose HF call raises `gpu_error()` off the CPU, if given."""
    calls = []
    held = []

    class Held:
        pass

    class Processor:
        image_processor = _ImageProcessor()
        tokenizer = SimpleNamespace(bos_token=None)

        def __call__(self, **kwargs):
            calls.append(kwargs["device"])
            if gpu_error is not None and kwargs["device"] != "cpu":
                # Stands in for the GPU tensors the failed attempt allocated.
                gpu_tensors = Held()
                held.append(weakref.ref(gpu_tensors))
                raise gpu_error()
            return {"input_ids": [1]}

    processor = _make(base_gpu_id=0)
    processor._feature_transport = MultimodalFeatureTransport(
        "cpu", base_gpu_id=0, skip_mm_pool=True
    )
    processor.precompute_hash_before_cpu_transfer = False
    processor._processor = Processor()
    processor._tokenizer = processor._processor.tokenizer
    processor._tokenizer_auto_adds_specials = False
    processor.disable_fast_image_processor = False
    processor.retry_on_cpu_after_gpu_oom = retry
    processor.video_preprocessing_device = None
    processor.image_config = {}
    processor.video_config = {}
    processor.audio_config = {}
    processor.FEATURE_NAMES = ["pixel_values"]
    return processor, calls, held


class TestRetryOnCpuAfterGpuOom(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _run(self, processor, empty_cache=None):
        with (
            _gpu_placement(),
            patch(f"{BASE}.BaseImageProcessor", _ImageProcessor),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool"),
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.empty_cache", side_effect=empty_cache),
        ):
            return processor.process_mm_data("test", images=["image"])

    def test_gpu_oom_retries_on_cpu(self):
        """A CUDA OOM raised outside the caching allocator (cuBLAS, a kernel
        launch) arrives as a RuntimeError and must take the CPU retry too."""
        for name, error in [
            ("allocator", lambda: torch.cuda.OutOfMemoryError("CUDA out of memory")),
            ("cuda call", lambda: RuntimeError("CUDA error: out of memory")),
        ]:
            with self.subTest(name):
                processor, calls, _ = _image_processor_stub(error)
                result = self._run(processor)
                self.assertEqual(calls, ["cuda:0", "cpu"])
                self.assertEqual(result, {"input_ids": [1]})

    def test_other_errors_are_not_retried(self):
        cases = [
            ("illegal address", RuntimeError("an illegal memory access"), True),
            ("retry disabled", torch.cuda.OutOfMemoryError("oom"), False),
        ]
        for name, error, retry in cases:
            with self.subTest(name):
                processor, calls, _ = _image_processor_stub(lambda: error, retry=retry)
                with self.assertRaises(RuntimeError):
                    self._run(processor)
                self.assertEqual(calls, ["cuda:0"])

    def test_failed_attempt_is_released_before_the_cpu_retry(self):
        """Retrying inside the except block kept the failed GPU attempt's frames
        alive, so empty_cache freed nothing and the CPU retry ran with that GPU
        memory still held."""
        processor, calls, held = _image_processor_stub(
            lambda: torch.cuda.OutOfMemoryError("oom")
        )
        released_at_empty_cache = []
        self._run(
            processor,
            empty_cache=lambda: released_at_empty_cache.append(held[0]() is None),
        )
        self.assertEqual(released_at_empty_cache, [True])
        self.assertEqual(calls, ["cuda:0", "cpu"])


class TestFreeMemoryRouting(CustomTestCase):
    """SGLANG_MM_GPU_PREPROCESS_RESERVE_MB sends image preprocessing straight to
    the CPU when the shared GPU lacks headroom, instead of failing there first."""

    MiB = 1024 * 1024

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)

    def _run(self, *, reserve_mb, free_mib, estimate_mib=0):
        processor, calls, _ = _image_processor_stub(retry=False)
        processor._estimate_gpu_preprocess_bytes = lambda images, kw: (
            estimate_mib * self.MiB
        )
        with (
            envs.SGLANG_MM_GPU_PREPROCESS_RESERVE_MB.override(reserve_mb),
            _gpu_placement(),
            patch(f"{BASE}.BaseImageProcessor", _ImageProcessor),
            patch(f"{BASE}.torch.cuda.device", return_value=nullcontext()),
            patch(f"{BASE}.torch.cuda.MemPool", return_value="pool"),
            patch(f"{BASE}.torch.cuda.use_mem_pool", return_value=nullcontext()),
            patch(
                f"{PLACEMENT}.torch.cuda.mem_get_info",
                return_value=(free_mib * self.MiB, 0),
            ) as mem_get_info,
            patch(f"{PLACEMENT}.torch.cuda.memory_reserved", return_value=0),
            patch(f"{PLACEMENT}.torch.cuda.memory_allocated", return_value=0),
        ):
            processor.process_mm_data("test", images=["image"])
        return calls, mem_get_info

    def test_disabled_reserve_never_queries_the_gpu(self):
        calls, mem_get_info = self._run(reserve_mb=0, free_mib=1)
        self.assertEqual(calls, ["cuda:0"])
        mem_get_info.assert_not_called()

    def test_routes_by_free_memory_against_estimate_plus_reserve(self):
        cases = [
            ("below the reserve", 1024, 4096, 0, ["cpu"]),
            ("reserve met, estimate not", 4500, 4096, 512, ["cpu"]),
            ("estimate plus reserve met", 8192, 4096, 512, ["cuda:0"]),
        ]
        for name, free_mib, reserve_mb, estimate_mib, expected in cases:
            with self.subTest(name):
                calls, _ = self._run(
                    reserve_mb=reserve_mb,
                    free_mib=free_mib,
                    estimate_mib=estimate_mib,
                )
                self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
