import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=1, suite="stage-b-test-1-gpu-small-amd")


class TestMmProcessConfigValidation(unittest.TestCase):
    """Server-args validation for mm_process_config."""

    def _validate_config(self, mm_process_config):
        args = ServerArgs(model_path="dummy", mm_process_config=mm_process_config)
        args._handle_multimodal()
        return args

    def test_valid_config_accepted(self):
        args = self._validate_config({"image": {"max_pixels": 5000000}})
        self.assertEqual(args.mm_process_config, {"image": {"max_pixels": 5000000}})

    def test_empty_config_accepted(self):
        args = self._validate_config({})
        self.assertEqual(args.mm_process_config, {})

    def test_none_config_defaults_to_empty_dict(self):
        args = self._validate_config(None)
        # None is kept as-is for dummy models (default happens after early return)
        # but for real models it would be set to {}
        self.assertIsNone(args.mm_process_config)

    def test_top_level_non_dict_rejected(self):
        with self.assertRaises(TypeError) as ctx:
            self._validate_config("bad")
        self.assertIn("mm_process_config must be a dict", str(ctx.exception))

    def test_modality_non_dict_rejected_image(self):
        with self.assertRaises(TypeError) as ctx:
            self._validate_config({"image": "bad"})
        self.assertIn("mm_process_config['image'] must be a dict", str(ctx.exception))

    def test_modality_non_dict_rejected_video(self):
        with self.assertRaises(TypeError) as ctx:
            self._validate_config({"video": 123})
        self.assertIn("mm_process_config['video'] must be a dict", str(ctx.exception))

    def test_modality_non_dict_rejected_audio(self):
        with self.assertRaises(TypeError) as ctx:
            self._validate_config({"audio": [1, 2]})
        self.assertIn("mm_process_config['audio'] must be a dict", str(ctx.exception))

    def test_multi_modality_config_accepted(self):
        config = {
            "image": {"max_pixels": 1048576},
            "video": {"max_pixels": 602112},
            "audio": {"sample_rate": 16000},
        }
        args = self._validate_config(config)
        self.assertEqual(args.mm_process_config, config)


class TestBaseProcessorConfigExtraction(unittest.TestCase):
    """Verify BaseMultimodalProcessor.__init__ extracts configs from server_args."""

    def _make_processor(
        self,
        mm_process_config,
        mm_processor_worker_num=0,
        mm_io_worker_num=0,
    ):
        """Create a BaseMultimodalProcessor via the real __init__ with mocked deps."""
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        server_args = MagicMock()
        server_args.mm_process_config = mm_process_config
        server_args.mm_processor_worker_num = mm_processor_worker_num
        server_args.mm_io_worker_num = mm_io_worker_num

        hf_config = MagicMock()
        mock_hf_processor = MagicMock()

        # Call real __init__ so we test actual config extraction
        with patch.object(BaseMultimodalProcessor, "__abstractmethods__", set()):
            proc = BaseMultimodalProcessor(
                hf_config=hf_config,
                server_args=server_args,
                _processor=mock_hf_processor,
                transport_mode=None,
            )
        return proc

    def test_configs_extracted(self):
        config = {
            "image": {"max_pixels": 5000000},
            "video": {"fps": 3},
            "audio": {"sample_rate": 16000},
        }
        proc = self._make_processor(config)
        self.assertEqual(proc.image_config, {"max_pixels": 5000000})
        self.assertEqual(proc.video_config, {"fps": 3})
        self.assertEqual(proc.audio_config, {"sample_rate": 16000})

    def test_empty_config_yields_empty_dicts(self):
        proc = self._make_processor({})
        self.assertEqual(proc.image_config, {})
        self.assertEqual(proc.video_config, {})
        self.assertEqual(proc.audio_config, {})

    def test_model_specific_auto_worker_count_enables_executor(self):
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_IO_WORKERS", None)
            with patch.object(
                BaseMultimodalProcessor, "auto_mm_processor_worker_num", 4
            ), patch.object(
                BaseMultimodalProcessor, "auto_mm_io_worker_num", 16
            ), patch.object(
                BaseMultimodalProcessor, "supports_mm_processor_concurrency", True
            ):
                proc = self._make_processor({})
        try:
            self.assertEqual(proc.mm_processor_worker_num, 4)
            self.assertEqual(proc.mm_io_worker_num, 16)
            self.assertIsNotNone(proc.mm_processor_executor)
        finally:
            proc.mm_processor_executor.shutdown()

    def test_explicit_single_worker_disables_executor(self):
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        with patch.object(BaseMultimodalProcessor, "auto_mm_processor_worker_num", 4):
            proc = self._make_processor({}, mm_processor_worker_num=1)
        self.assertEqual(proc.mm_processor_worker_num, 1)
        self.assertIsNone(proc.mm_processor_executor)

    def test_parallel_workers_require_processor_support(self):
        proc = self._make_processor({}, mm_processor_worker_num=2)
        self.assertEqual(proc.mm_processor_worker_num, 1)
        self.assertIsNone(proc.mm_processor_executor)

    def test_explicit_io_worker_count_overrides_auto(self):
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        with patch.object(BaseMultimodalProcessor, "auto_mm_io_worker_num", 16):
            proc = self._make_processor({}, mm_io_worker_num=6)
        self.assertEqual(proc.mm_io_worker_num, 6)


class TestMultimodalFeatureTransportRuntime(unittest.TestCase):
    @staticmethod
    def _server_args(mm_feature_transport):
        return SimpleNamespace(
            mm_feature_transport=mm_feature_transport,
            keep_mm_feature_on_device=False,
            disable_fast_image_processor=False,
            skip_tokenizer_init=False,
            mm_process_config={},
            mm_processor_worker_num=0,
            mm_io_worker_num=0,
            tokenizer_worker_num=1,
            base_gpu_id=2,
        )

    @staticmethod
    def _processor():
        processor = MagicMock()
        processor.tokenizer.encode.return_value = []
        return processor

    def test_cuda_ipc_pool_uses_resolved_server_arg(self):
        # The processor module can be imported before this instance is built;
        # transport policy must still resolve from the instance's ServerArgs.
        from sglang.srt.multimodal.processors import base_processor

        with patch.object(
            base_processor.BaseMultimodalProcessor, "__abstractmethods__", set()
        ), patch.object(base_processor, "MmItemMemoryPool") as memory_pool:
            processor = base_processor.BaseMultimodalProcessor(
                hf_config=MagicMock(),
                server_args=self._server_args("cuda_ipc"),
                _processor=self._processor(),
                transport_mode=None,
            )

        self.assertEqual(processor.mm_feature_transport, "cuda_ipc")
        self.assertTrue(processor.use_cuda_ipc)
        memory_pool.assert_called_once()

    def test_cpu_transport_does_not_allocate_ipc_pool(self):
        from sglang.srt.multimodal.processors import base_processor

        with patch.object(
            base_processor.BaseMultimodalProcessor, "__abstractmethods__", set()
        ), patch.object(base_processor, "MmItemMemoryPool") as memory_pool:
            processor = base_processor.BaseMultimodalProcessor(
                hf_config=MagicMock(),
                server_args=self._server_args("cpu"),
                _processor=self._processor(),
                transport_mode=None,
            )

        self.assertEqual(processor.mm_feature_transport, "cpu")
        self.assertFalse(processor.use_cuda_ipc)
        memory_pool.assert_not_called()


class TestMultimodalProcessorConcurrency(unittest.IsolatedAsyncioTestCase):
    async def test_dedicated_executor_runs_processor_off_event_loop(self):
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )
        from sglang.srt.multimodal.processors.executor import (
            MultimodalProcessorExecutor,
        )

        with patch.object(
            BaseMultimodalProcessor, "__abstractmethods__", set()
        ), patch.object(BaseMultimodalProcessor, "__init__", lambda self: None):
            processor = BaseMultimodalProcessor()

        processor.mm_processor_executor = MultimodalProcessorExecutor(
            SimpleNamespace(tokenizer=object()), max_workers=2
        )
        processor.process_and_combine_mm_data = MagicMock(
            side_effect=lambda *_args, **_kwargs: threading.current_thread().name
        )
        try:
            thread_name = await processor.process_and_combine_mm_data_async(
                MagicMock(), MagicMock(), marker=True
            )
        finally:
            processor.mm_processor_executor.shutdown()

        self.assertTrue(thread_name.startswith("sglang-mm-processor"))
        processor.process_and_combine_mm_data.assert_called_once()
        self.assertTrue(
            processor.process_and_combine_mm_data.call_args.kwargs["marker"]
        )

    async def test_single_worker_preserves_synchronous_path(self):
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        with patch.object(
            BaseMultimodalProcessor, "__abstractmethods__", set()
        ), patch.object(BaseMultimodalProcessor, "__init__", lambda self: None):
            processor = BaseMultimodalProcessor()

        processor.mm_processor_executor = None
        processor.process_and_combine_mm_data = MagicMock(return_value="synchronous")

        result = await processor.process_and_combine_mm_data_async(
            MagicMock(), MagicMock()
        )

        self.assertEqual(result, "synchronous")
        processor.process_and_combine_mm_data.assert_called_once()

    async def test_worker_reuses_precreated_private_processor_clone(self):
        from sglang.srt.multimodal.processors.executor import (
            MultimodalProcessorExecutor,
        )

        executor = MultimodalProcessorExecutor(object(), max_workers=2)
        return_processor = lambda *, processor: processor
        try:
            first = await executor.run(return_processor)
            second = await executor.run(return_processor)
        finally:
            executor.shutdown()

        self.assertIs(first, second)

    async def test_replacement_worker_lazily_clones_processor(self):
        from sglang.srt.multimodal.processors import executor as executor_module

        source_processor = object()
        replacement_clone = SimpleNamespace(tokenizer=object())
        with patch.object(
            executor_module.copy,
            "deepcopy",
            return_value=replacement_clone,
        ) as deepcopy:
            executor = executor_module.MultimodalProcessorExecutor(
                source_processor, max_workers=2
            )
            executor._processor_clones.clear()
            return_processor = lambda *, processor: processor
            try:
                first = await executor.run(return_processor)
                second = await executor.run(return_processor)
            finally:
                executor.shutdown()

        self.assertIs(first, replacement_clone)
        self.assertIs(first, second)
        self.assertEqual(deepcopy.call_count, 3)


class TestProcessMmDataKwargs(unittest.TestCase):
    """Verify process_mm_data injects per-modality kwargs correctly."""

    def _make_base_processor(self, mm_process_config):
        """Create a BaseMultimodalProcessor with process_mm_data testable."""
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        server_args = MagicMock()
        server_args.mm_process_config = mm_process_config
        server_args.mm_feature_transport = "cpu"
        server_args.disable_fast_image_processor = True
        server_args.keep_mm_feature_on_device = True
        server_args.skip_tokenizer_init = False

        mock_processor = MagicMock()
        mock_processor.__class__.__name__ = "TestProcessor"
        # Capture kwargs passed to __call__
        captured_kwargs = {}

        def capture_call(**kwargs):
            captured_kwargs.update(kwargs)
            return {}

        mock_processor.__call__ = MagicMock(side_effect=capture_call)

        with patch.object(BaseMultimodalProcessor, "__abstractmethods__", set()):
            with patch.object(BaseMultimodalProcessor, "__init__", lambda self: None):
                proc = BaseMultimodalProcessor()

        proc.server_args = server_args
        proc.keep_mm_feature_on_device = server_args.keep_mm_feature_on_device
        proc.mm_feature_transport = server_args.mm_feature_transport
        proc.use_cuda_ipc = False
        proc.disable_fast_image_processor = server_args.disable_fast_image_processor
        proc.skip_tokenizer_init = server_args.skip_tokenizer_init
        proc._processor = mock_processor
        proc._tokenizer = MagicMock()
        proc._tokenizer_auto_adds_specials = False
        proc.image_config = mm_process_config.get("image", {})
        proc.video_config = mm_process_config.get("video", {})
        proc.audio_config = mm_process_config.get("audio", {})
        proc.FEATURE_NAMES = []

        return proc, mock_processor, captured_kwargs

    def test_images_kwargs_injected(self):
        config = {"image": {"max_pixels": 5000000}}
        proc, mock_proc, _ = self._make_base_processor(config)

        proc.process_mm_data("test", images=["img1"])

        call_kwargs = mock_proc.__call__.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("images_kwargs"), {"max_pixels": 5000000}
        )

    def test_videos_kwargs_injected(self):
        config = {"video": {"fps": 3, "max_frames": 60}}
        proc, mock_proc, _ = self._make_base_processor(config)

        proc.process_mm_data("test", videos=["vid1"])

        call_kwargs = mock_proc.__call__.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("videos_kwargs"), {"fps": 3, "max_frames": 60}
        )

    def test_preprocessed_video_config_is_filtered_before_single_call(self):
        config = {
            "video": {
                "fps": 3,
                "max_frames": 60,
                "do_normalize": False,
            }
        }
        proc, mock_proc, _ = self._make_base_processor(config)

        proc.process_mm_data(
            "test",
            videos=["vid1"],
            processor_video_config={"do_normalize": False},
        )

        self.assertEqual(mock_proc.__call__.call_count, 1)
        self.assertEqual(
            mock_proc.__call__.call_args.kwargs.get("videos_kwargs"),
            {"do_normalize": False},
        )

    def test_processor_error_is_not_retried(self):
        proc, mock_proc, _ = self._make_base_processor({"video": {"max_frames": 60}})
        mock_proc.__call__.side_effect = ValueError("processor failure")

        with self.assertRaisesRegex(ValueError, "processor failure"):
            proc.process_mm_data("test", videos=["vid1"])

        self.assertEqual(mock_proc.__call__.call_count, 1)

    def test_no_collision_with_overlapping_keys(self):
        """Core test: image and video both have max_pixels but stay separate."""
        config = {
            "image": {"max_pixels": 1048576},
            "video": {"max_pixels": 602112},
        }
        proc, mock_proc, _ = self._make_base_processor(config)

        proc.process_mm_data("test", images=["img1"], videos=["vid1"])

        call_kwargs = mock_proc.__call__.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("images_kwargs"), {"max_pixels": 1048576}
        )
        self.assertEqual(
            call_kwargs.kwargs.get("videos_kwargs"), {"max_pixels": 602112}
        )

    def test_empty_config_no_kwargs_injected(self):
        proc, mock_proc, _ = self._make_base_processor({})

        proc.process_mm_data("test", images=["img1"])

        call_kwargs = mock_proc.__call__.call_args
        self.assertNotIn("images_kwargs", call_kwargs.kwargs)

    def test_audio_kwargs_preserved_with_config(self):
        """audio_config merges with existing truncation=False."""
        config = {"audio": {"sample_rate": 16000}}
        proc, mock_proc, _ = self._make_base_processor(config)
        # Simulate a processor that uses singular "audio" key
        mock_proc.__class__.__name__ = "Gemma3nProcessor"

        proc.process_mm_data("test", audios=["aud1"])

        call_kwargs = mock_proc.__call__.call_args
        audio_kw = call_kwargs.kwargs.get("audio_kwargs", {})
        self.assertFalse(audio_kw.get("truncation", True))
        self.assertEqual(audio_kw.get("sample_rate"), 16000)


class TestOverrideProcessorsConfigInjection(unittest.TestCase):
    """Regression tests for processors that override process_mm_data."""

    def _make_override_processor(self, processor_cls, mm_process_config):
        """Create an override processor with mocked dependencies."""
        server_args = MagicMock()
        server_args.mm_process_config = mm_process_config
        server_args.mm_feature_transport = "cpu"
        server_args.disable_fast_image_processor = True
        server_args.keep_mm_feature_on_device = False
        server_args.skip_tokenizer_init = False

        mock_hf_processor = MagicMock()
        mock_hf_processor.__class__.__name__ = "TestProcessor"
        # Ernie processor accesses result["images"] after __call__,
        # so return {"images": None} to pass the None-guard safely.
        mock_hf_processor.__call__ = MagicMock(return_value={"images": None})

        with patch.object(processor_cls, "__init__", lambda self: None):
            proc = processor_cls()

        proc.server_args = server_args
        proc.keep_mm_feature_on_device = server_args.keep_mm_feature_on_device
        proc.mm_feature_transport = server_args.mm_feature_transport
        proc.use_cuda_ipc = False
        proc.disable_fast_image_processor = server_args.disable_fast_image_processor
        proc.skip_tokenizer_init = server_args.skip_tokenizer_init
        proc._processor = mock_hf_processor
        proc.image_config = mm_process_config.get("image", {})
        proc.video_config = mm_process_config.get("video", {})
        proc.audio_config = mm_process_config.get("audio", {})
        proc.FEATURE_NAMES = []

        return proc, mock_hf_processor

    def test_ernie45_vl_injects_images_kwargs(self):
        from sglang.srt.multimodal.processors.ernie45_vl import (
            Ernie4_5_VLImageProcessor,
        )

        config = {"image": {"max_pixels": 2000000}, "video": {"max_pixels": 500000}}
        proc, mock_proc = self._make_override_processor(
            Ernie4_5_VLImageProcessor, config
        )

        proc.process_mm_data("test", images=["img1"], videos=["vid1"])

        call_kwargs = mock_proc.__call__.call_args
        self.assertEqual(
            call_kwargs.kwargs.get("images_kwargs"), {"max_pixels": 2000000}
        )
        self.assertEqual(
            call_kwargs.kwargs.get("videos_kwargs"), {"max_pixels": 500000}
        )

    def test_midashenglm_injects_audio_kwargs(self):
        from sglang.srt.multimodal.processors.midashenglm import (
            MiDashengLMMultimodalProcessor,
        )

        config = {"audio": {"sample_rate": 16000}}
        proc, mock_proc = self._make_override_processor(
            MiDashengLMMultimodalProcessor, config
        )

        proc.process_mm_data("test", audios=["aud1"])

        call_kwargs = mock_proc.__call__.call_args
        audio_kw = call_kwargs.kwargs.get("audio_kwargs", {})
        self.assertFalse(audio_kw.get("truncation", True))
        self.assertEqual(audio_kw.get("sample_rate"), 16000)

    def test_midashenglm_user_config_overrides_truncation(self):
        """User config can override the default truncation=False."""
        from sglang.srt.multimodal.processors.midashenglm import (
            MiDashengLMMultimodalProcessor,
        )

        config = {"audio": {"truncation": True}}
        proc, mock_proc = self._make_override_processor(
            MiDashengLMMultimodalProcessor, config
        )

        proc.process_mm_data("test", audios=["aud1"])

        call_kwargs = mock_proc.__call__.call_args
        audio_kw = call_kwargs.kwargs.get("audio_kwargs", {})
        # User config can override truncation if they explicitly set it
        self.assertTrue(audio_kw.get("truncation"))


class TestQwenVideoConfigRouting(unittest.TestCase):
    def test_preprocessed_video_drops_sglang_owned_config(self):
        from sglang.srt.multimodal.processors.qwen_vl import (
            _get_processor_video_config,
        )

        video_config = {
            "fps": 3,
            "nframes": 12,
            "max_frames": 60,
            "max_pixels": 500000,
            "do_normalize": False,
        }

        processor_config = _get_processor_video_config(video_config, [{"fps": 30.0}])

        self.assertEqual(processor_config, {"do_normalize": False})

    def test_unprocessed_video_uses_original_config(self):
        from sglang.srt.multimodal.processors.qwen_vl import (
            _get_processor_video_config,
        )

        video_config = {"fps": 3, "max_frames": 60}

        self.assertIsNone(_get_processor_video_config(video_config, None))
        self.assertIsNone(_get_processor_video_config(video_config, [None]))


class TestDoubleBosGuard(unittest.TestCase):
    """Regression test for the multimodal double-BOS bug.

    Repro condition (Cohere2 / Llama3-LLaVA-Next family):
      - tokenizer.encode("") returns [bos_id]  (auto-adds specials), AND
      - chat template renders the BOS string as a literal at the start.

    Without the guard in BaseMultimodalProcessor, the inner processor.__call__
    on the rendered prompt would auto-prepend a second BOS, producing 2 leading
    BOS tokens vs the HF reference's 1.
    """

    def test_guard_passes_add_special_tokens_false_on_bug_condition(self):
        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        server_args = MagicMock()
        server_args.mm_process_config = {}
        server_args.mm_processor_worker_num = 0
        server_args.mm_io_worker_num = 0
        server_args.mm_feature_transport = "cpu"
        server_args.disable_fast_image_processor = True
        server_args.keep_mm_feature_on_device = True

        mock_hf_processor = MagicMock()
        mock_hf_processor.__class__.__name__ = "TestProcessor"
        mock_hf_processor.__call__ = MagicMock(return_value={})
        mock_hf_processor.tokenizer.encode = MagicMock(return_value=[2])
        mock_hf_processor.tokenizer.bos_token = "<BOS>"

        with patch.object(BaseMultimodalProcessor, "__abstractmethods__", set()):
            proc = BaseMultimodalProcessor(
                hf_config=MagicMock(),
                server_args=server_args,
                _processor=mock_hf_processor,
                transport_mode=None,
            )
        proc.FEATURE_NAMES = []

        proc.process_mm_data("<BOS>hello", images=["img1"])

        call_kwargs = mock_hf_processor.__call__.call_args.kwargs
        self.assertEqual(call_kwargs.get("add_special_tokens"), False)


if __name__ == "__main__":
    unittest.main()
