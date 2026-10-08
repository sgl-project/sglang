"""A fresh trainer uses shared processor pools without loading inference modules.

fresh Python -> resource-free Qwen member + lightweight host
                     |
host resources -> default: one worker; explicit: requested clone workers
                     |
PNG path + explicit devices -> host pools -> member -> BatchFeature
                     |
CPU pool + token expansion -> shutdown -> executor submission rejected

Serving Base, model wrappers, ServerArgs, NPU patches and kernel imports are forbidden.
Caller devices pass unchanged; omitted devices retain HF defaults.
Original ImageData wrappers carry caller options before loading.
Source configs stay outside loaded media and reach the cloned processor unchanged.
"""

import os
import subprocess
import sys
import textwrap

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_trainer_uses_shared_pools_without_inference_imports():
    script = textwrap.dedent(
        """
        import asyncio
        import concurrent.futures
        import importlib.abc
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        forbidden = (
            "sglang.kernels",
            "sglang.srt.managers",
            "sglang.srt.model_executor",
            "sglang.srt.model_loader",
            "sglang.srt.models",
            "sglang.srt.layers",
            "sglang.srt.distributed",
            "sglang.srt.server_args",
            "sglang.srt.multimodal.processors.base_processor",
            "sglang.srt.multimodal.processors.qwen_vl",
            "sglang.srt.hardware_backend.npu",
            "sgl_kernel",
            "flashinfer",
        )

        class RejectInferenceImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                assert not any(
                    fullname == prefix or fullname.startswith(prefix + ".")
                    for prefix in forbidden
                ), fullname

        sys.meta_path.insert(0, RejectInferenceImports())

        from sglang.srt.multimodal.media_processor import (
            TokenSpaceMMProcessor, MultimodalProcessorMixin, get_media_source_configs,
        )
        from sglang.srt.multimodal.processors.processor_config import MultimodalProcessorConfig
        from sglang.srt.multimodal.token_space.qwen_vl import QwenTokenSpaceProcessStrategy
        from sglang.srt.utils import ImageData

        import torch
        from PIL import Image
        from tokenizers import Tokenizer, models
        from transformers import (
            BatchFeature, PreTrainedTokenizerFast, Qwen2VLImageProcessor,
            Qwen3VLConfig, Qwen3VLProcessor, Qwen3VLVideoProcessor,
        )

        tokens = [
            "[UNK]", "[PAD]", "<|image_pad|>", "<|video_pad|>",
            "<|vision_start|>", "<|vision_end|>",
        ]
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(models.WordLevel(
                {token: index for index, token in enumerate(tokens)},
                unk_token="[UNK]",
            )),
            unk_token="[UNK]",
            pad_token="[PAD]",
            additional_special_tokens=tokens[2:],
        )
        hf_processor = Qwen3VLProcessor(
            tokenizer=tokenizer,
            image_processor=Qwen2VLImageProcessor(
                size={"shortest_edge": 56**2, "longest_edge": 56**2},
            ),
            video_processor=Qwen3VLVideoProcessor(
                patch_size=14,
                size={"shortest_edge": 56**2, "longest_edge": 56**2},
            ),
        )
        # HF may probe torch_npu availability during import; processing must not import it.
        forbidden += ("torch_npu",)

        # A member has no resource lifecycle; only its host may create pools.
        with (
            patch.object(
                MultimodalProcessorMixin, "_initialize_processor", side_effect=AssertionError,
            ),
            patch.object(concurrent.futures, "ThreadPoolExecutor", side_effect=AssertionError),
            patch.object(concurrent.futures, "ProcessPoolExecutor", side_effect=AssertionError),
        ):
            member = QwenTokenSpaceProcessStrategy(Qwen3VLConfig(), hf_processor)
        assert not isinstance(member, MultimodalProcessorMixin)

        # Training owns its worker count even when the model declares a serving default.
        worker_cases = [({}, 1)]
        for requested, expected in ((0, 1), (1, 1), (3, 3)):
            worker_cases.append(({
                "processor_config": MultimodalProcessorConfig(
                    mm_processor_worker_num=requested, cpu_worker_num=1,
                ),
            }, expected))
        for options, expected_workers in worker_cases:
            with (
                patch.object(
                    concurrent.futures, "ThreadPoolExecutor",
                    wraps=concurrent.futures.ThreadPoolExecutor,
                ) as thread_pool,
                patch.object(
                    concurrent.futures, "ProcessPoolExecutor",
                    wraps=concurrent.futures.ProcessPoolExecutor,
                ) as cpu_pool,
            ):
                configured = TokenSpaceMMProcessor(
                    Qwen3VLConfig(), hf_processor, QwenTokenSpaceProcessStrategy, **options,
                )
            try:
                assert thread_pool.call_count == 1 + (expected_workers > 1), options
                assert cpu_pool.call_count == 1, options
                assert configured.mm_processor_worker_num == expected_workers, options
                executor = configured.mm_processor_executor
                if expected_workers == 1:
                    assert executor is None, options
                else:
                    assert executor._executor._max_workers == expected_workers, options
            finally:
                configured.shutdown()

        processor = TokenSpaceMMProcessor(
            Qwen3VLConfig(
                image_token_id=2, video_token_id=3,
                vision_start_token_id=4, vision_end_token_id=5,
            ),
            hf_processor,
            QwenTokenSpaceProcessStrategy,
            processor_config=MultimodalProcessorConfig(
                mm_io_worker_num=2, mm_processor_worker_num=2,
                cpu_worker_num=1, cpu_process_start_method="spawn",
            ),
        )

        token_space_process_strategy = processor.token_space_process_strategy

        async def process_image(image_path):
            sources = [ImageData(str(image_path), preprocess_kwargs={"do_rescale": False})]
            images, _, _ = await processor._load_media_lists(
                sources, None, None, audio_sample_rate=None, discard_alpha_channel=True
            )
            assert isinstance(images[0], Image.Image)
            return await processor.process_media_async(
                images=images, image_device="cpu",
                image_source_configs=get_media_source_configs(sources),
            )

        try:
            with tempfile.TemporaryDirectory() as directory:
                image_path = Path(directory) / "image.png"
                image = Image.new("RGB", (56, 56), (17, 39, 81))
                image.save(image_path)
                with patch.object(
                    token_space_process_strategy, "process_image",
                    wraps=token_space_process_strategy.process_image,
                ) as wrapped_process_image:
                    features = asyncio.run(process_image(image_path))
                assert wrapped_process_image.call_args.args[1] is not hf_processor
                expected = hf_processor.image_processor(image, do_rescale=False, return_tensors="pt")
                assert isinstance(features, BatchFeature)
                assert set(features) == set(expected)
                for name in expected:
                    assert torch.equal(features[name], expected[name]), name
                batch = asyncio.run(processor.load_and_process_token_space_async(
                    [ImageData(str(image_path), preprocess_kwargs={"do_rescale": False})],
                    input_ids=[4, 2, 5], image_device="cpu",
                ))
                assert batch["input_ids"].tolist() == [[4, 2, 2, 2, 2, 5]]
                assert batch["attention_mask"].tolist() == [[1] * 6]
                for name in expected:
                    assert torch.equal(batch[name], expected[name]), name
            expanded = token_space_process_strategy.expand_media_tokens(
                features, input_ids=[4, 2, 5], processor=hf_processor
            )
            assert expanded == [4, 2, 2, 2, 2, 5]

            # Device forwarding must not install serving NPU patches or choose a GPU.
            with (
                patch.object(
                    token_space_process_strategy, "process_image",
                    side_effect=lambda image, hf, source_config, **kwargs: {
                        "image_device": kwargs.get("device"),
                    },
                ),
                patch.object(
                    token_space_process_strategy, "process_video",
                    side_effect=lambda video, hf, source_config, **kwargs: {
                        "video_device": kwargs.get("device"),
                    },
                ),
            ):
                for image_device, video_device, expected in (
                    (None, None, {"image_device": None, "video_device": None}),
                    ("cpu", "meta", {"image_device": "cpu", "video_device": "meta"}),
                    ("meta", None, {"image_device": "meta", "video_device": "meta"}),
                ):
                    captured = asyncio.run(processor.process_media_async(
                        images=[image], videos=["video"],
                        image_device=image_device, video_device=video_device,
                    ))
                    assert dict(captured) == expected
            assert processor.cpu_executor.submit(sum, [2, 3]).result(timeout=20) == 5
            assert not hasattr(processor, "runtime_context")
            assert not hasattr(processor, "server_args")
            assert not hasattr(processor, "mm_preprocess_cache")
            assert not torch.cuda.is_initialized()
            assert not torch.distributed.is_initialized()
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in sys.modules for prefix in forbidden
            )
        finally:
            processor.shutdown()
        for executor in (processor.io_executor, processor.cpu_executor):
            try:
                executor.submit(sum, [2, 3])
            except RuntimeError:
                pass
            else:
                raise AssertionError("Processor shutdown left an executor running")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "SGLANG_USE_CPU_ENGINE": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
