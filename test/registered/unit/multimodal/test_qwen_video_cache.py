"""Real CPU video decoding and HF preprocessing through Qwen's cache entry point."""

import asyncio
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import torch
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLConfig,
    Qwen2_5_VLProcessor,
    Qwen2VLVideoProcessor,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.multimodal import video_cache
from sglang.srt.multimodal.processors import base_processor, qwen_vl
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import VideoData, video_decoder
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

VIDEO_TOKEN = "<|vision_start|><|video_pad|><|vision_end|>"
PROMPT = f"hello {VIDEO_TOKEN} describe"


def _write_video(path):
    """A moving square makes frame-sampling changes observable without downloads."""
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt = 56, 56, "yuv420p"
        for index in range(16):
            pixels = np.zeros((56, 56, 3), dtype=np.uint8)
            pixels[:, :, 0] = 192
            pixels[8:24, index * 2 : index * 2 + 16] = [0, 0, 255]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def _make_processor(*, video_mean=None):
    # Adapt the repository's rust/qwen/_fixtures.py tokenizer, while using the
    # actual Qwen2.5 processor and config needed for video temporal metadata.
    vocab = [
        "<unk>",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        "hello",
        "<|video_pad|>",
        "<pad>",
        "describe",
        "again",
    ]
    backend = Tokenizer(
        WordLevel({token: i for i, token in enumerate(vocab)}, unk_token=vocab[0])
    )
    backend.pre_tokenizer, backend.decoder = WhitespaceSplit(), decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=vocab[0],
        pad_token=vocab[6],
        additional_special_tokens=vocab[1:4] + [vocab[5]],
    )
    video_options = {} if video_mean is None else {"image_mean": video_mean}
    hf_processor = Qwen2_5_VLProcessor(
        image_processor=Qwen2VLImageProcessor(min_pixels=784, max_pixels=3136),
        video_processor=Qwen2VLVideoProcessor(
            size={"shortest_edge": 784, "longest_edge": 3136},
            do_sample_frames=False,
            **video_options,
        ),
        tokenizer=tokenizer,
    )
    config = Qwen2_5_VLConfig(
        architectures=["Qwen2_5_VLForConditionalGeneration"],
        vision_start_token_id=1,
        image_token_id=2,
        vision_end_token_id=3,
        video_token_id=5,
        vision_config={"spatial_merge_size": 2, "tokens_per_second": 2},
    )
    args = ServerArgs(
        model_path="dummy",
        model_impl="sglang",
        mm_feature_transport="cpu",
        mm_preprocess_cache_size_mb=32,
        trust_mm_cache_ids=True,
        disable_fast_image_processor=True,
        mm_io_worker_num=2,
        allowed_media_domains=[],
        mm_process_config={
            "video": {
                "fps": 2,
                "min_frames": 2,
                "max_frames": 8,
                "min_pixels": 784,
                "max_pixels": 3136,
            }
        },
    )
    publish(args, role="tokenizer")
    return qwen_vl.QwenVLImageProcessor(
        config, args, hf_processor, None, skip_mm_pool=True
    )


class TestQwenVideoCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.video_path = str(Path(cls.directory.name) / "clip.mp4")
        _write_video(cls.video_path)

    def setUp(self):
        self.addCleanup(reset_context)
        # Only select CPU allocation/decoder dispatch. Decord, frame sampling,
        # resizing, the HF processor, token expansion and M-RoPE remain real.
        for target in (
            patch.object(qwen_vl, "is_cpu", return_value=True),
            patch.object(video_decoder, "_is_cpu_engine", return_value=True),
        ):
            target.start()
            self.addCleanup(target.stop)
        self.processor = _make_processor()
        self.addCleanup(self.processor.shutdown)

    def _request(self, *, identity="clip-v1", path=None, options=None, tenant="a"):
        source = VideoData(
            path or self.video_path, preprocess_kwargs=options, cache_id=identity
        )
        return GenerateReqInput(text=PROMPT, video_data=[source], cache_salt=tenant)

    async def _process(self, request=None, *, prompt=PROMPT, processor=None):
        return await (processor or self.processor).process_mm_data_async(
            None, prompt, request or self._request()
        )

    def _assert_parity(self, actual, reference):
        self.assertEqual(actual.input_ids, reference.input_ids)
        self.assertEqual(
            [offset for item in actual.mm_items for offset in item.offsets],
            [offset for item in reference.mm_items for offset in item.offsets],
        )
        # The old path groups videos in one MM item; the cache composes one
        # artifact per video. The model consumes concatenated features/grids.
        for attribute in ("feature", "video_grid_thw"):
            torch.testing.assert_close(
                torch.cat([getattr(item, attribute) for item in actual.mm_items]),
                torch.cat([getattr(item, attribute) for item in reference.mm_items]),
                rtol=0,
                atol=0,
            )
        for attribute in ("mrope_positions", "mrope_position_delta"):
            torch.testing.assert_close(
                getattr(actual, attribute),
                getattr(reference, attribute),
                rtol=0,
                atol=0,
            )

    def test_cold_and_warm_match_real_uncached_video_before_io_and_hash(self):
        async def run():
            for prompt in (PROMPT, self.processor._tokenizer.encode(PROMPT)):
                with self.subTest(prompt_type=type(prompt).__name__):
                    self.processor.clear_preprocess_cache()
                    reference = await self._process(
                        self._request(identity=None), prompt=prompt
                    )
                    with (
                        patch.object(
                            video_cache, "load_video", wraps=video_cache.load_video
                        ) as load,
                        patch.object(
                            qwen_vl, "preprocess_video", wraps=qwen_vl.preprocess_video
                        ) as preprocess,
                        patch.object(
                            self.processor,
                            "process_mm_data",
                            wraps=self.processor.process_mm_data,
                        ) as hf,
                    ):
                        cold = await self._process(prompt=prompt)
                        self._assert_parity(cold, reference)
                        self.assertEqual(
                            (load.call_count, preprocess.call_count, hf.call_count),
                            (1, 1, 1),
                        )
                        # A hit must neither touch the now-unreadable source nor
                        # hash the tensor again; identity hashing is still cheap.
                        with patch.object(
                            video_cache,
                            "resolve_multimodal_item_hash",
                            side_effect=AssertionError("warm hit hashed media"),
                        ):
                            warm = await self._process(
                                self._request(path=self.video_path + ".missing"),
                                prompt=prompt,
                            )
                        self._assert_parity(warm, reference)
                        self.assertEqual(cold.padded_input_ids, warm.padded_input_ids)
                        self.assertEqual(
                            (load.call_count, preprocess.call_count, hf.call_count),
                            (1, 1, 1),
                        )

        asyncio.run(run())

    def test_prompt_recomposition_and_mixed_ids_do_not_mutate_cached_artifacts(self):
        async def run():
            request = self._request()
            request.video_data.append(VideoData(self.video_path))
            prompt = f"hello {VIDEO_TOKEN} describe {VIDEO_TOKEN}"
            reference_request = self._request(identity=None)
            reference_request.video_data.append(VideoData(self.video_path))
            reference = await self._process(reference_request, prompt=prompt)
            with patch.object(
                video_cache, "load_video", wraps=video_cache.load_video
            ) as load:
                cold = await self._process(request, prompt=prompt)
                self._assert_parity(cold, reference)
                self.assertEqual(load.call_count, 2)
                cold.mm_items[0].feature.zero_()
                cold.mm_items[0].video_grid_thw.zero_()
                warm = await self._process(request, prompt=prompt)
                self._assert_parity(warm, reference)
                self.assertEqual(load.call_count, 3)
                changed_prompt = f"again hello {VIDEO_TOKEN}"
                changed_reference = await self._process(
                    self._request(identity=None), prompt=changed_prompt
                )
                changed = await self._process(prompt=changed_prompt)
                self._assert_parity(changed, changed_reference)
                self.assertEqual(load.call_count, 3)

        asyncio.run(run())

    def test_sampling_and_tenant_changes_miss_but_equal_fps_hits(self):
        async def run():
            with patch.object(
                video_cache, "load_video", wraps=video_cache.load_video
            ) as load:
                original = await self._process()
                equal = await self._process(self._request(options={"fps": 2.0}))
                self._assert_parity(equal, original)
                self.assertEqual(load.call_count, 1)
                for options in ({"fps": 4}, {"nframes": 8}, {"max_frames": 2}):
                    with self.subTest(options=options):
                        before = load.call_count
                        changed = await self._process(self._request(options=options))
                        self.assertEqual(load.call_count, before + 1)
                        self.assertNotEqual(
                            changed.mm_items[0].feature.shape,
                            original.mm_items[0].feature.shape,
                        )
                other_tenant = await self._process(
                    self._request(tenant="another-tenant")
                )
                self.assertEqual(load.call_count, 5)
                self.assertNotEqual(
                    original.padded_input_ids, other_tenant.padded_input_ids
                )

        asyncio.run(run())

    def test_real_hf_processor_configuration_partitions_shared_cache(self):
        async def run():
            other = _make_processor(video_mean=[0.1, 0.2, 0.3])
            self.addCleanup(other.shutdown)
            other.mm_preprocess_cache = self.processor.mm_preprocess_cache
            self.assertNotEqual(
                other.processor_fingerprint, self.processor.processor_fingerprint
            )
            with patch.object(
                video_cache, "load_video", wraps=video_cache.load_video
            ) as load:
                original = await self._process()
                changed = await self._process(processor=other)
                self.assertEqual(load.call_count, 2)
                self.assertFalse(
                    torch.equal(
                        original.mm_items[0].feature, changed.mm_items[0].feature
                    )
                )
                self.assertNotEqual(original.padded_input_ids, changed.padded_input_ids)
                self._assert_parity(await self._process(), original)
                self.assertEqual(load.call_count, 2)

        asyncio.run(run())

    def test_cancelled_first_request_does_not_cancel_shared_preprocessing(self):
        async def run():
            started, release = asyncio.Event(), asyncio.Event()
            real_preprocess = qwen_vl.preprocess_video

            async def delayed(decoder, **kwargs):
                started.set()
                await release.wait()
                return await real_preprocess(decoder, **kwargs)

            with (
                patch.object(qwen_vl, "preprocess_video", side_effect=delayed),
                patch.object(
                    video_cache, "load_video", wraps=video_cache.load_video
                ) as load,
            ):
                first = asyncio.create_task(self._process())
                await asyncio.wait_for(started.wait(), 10)
                second = asyncio.create_task(self._process())
                await asyncio.sleep(0)
                self.assertEqual(
                    self.processor.mm_preprocess_cache.singleflight_joins, 1
                )
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                release.set()
                result = await asyncio.wait_for(second, 10)
                self._assert_parity(await self._process(), result)
                self.assertEqual(load.call_count, 1)

        asyncio.run(run())

    def test_shared_preprocess_failure_is_propagated_and_retry_is_not_poisoned(self):
        async def run():
            started, release = asyncio.Event(), asyncio.Event()
            real_load = video_cache.load_video
            closed = []

            def tracked_load(*args, **kwargs):
                decoder = real_load(*args, **kwargs)
                original_close = decoder.close

                def close():
                    closed.append(True)
                    original_close()

                decoder.close = close
                return decoder

            async def fail(decoder, **kwargs):
                started.set()
                await release.wait()
                raise ValueError("injected preprocessing failure")

            with (
                patch.object(video_cache, "load_video", side_effect=tracked_load),
                patch.object(qwen_vl, "preprocess_video", side_effect=fail),
            ):
                first = asyncio.create_task(self._process())
                await asyncio.wait_for(started.wait(), 10)
                second = asyncio.create_task(self._process())
                await asyncio.sleep(0)
                release.set()
                errors = await asyncio.gather(first, second, return_exceptions=True)
                self.assertTrue(all(isinstance(error, ValueError) for error in errors))
                self.assertTrue(all("injected" in str(error) for error in errors))
                self.assertEqual(len(closed), 1)
                self.assertEqual(len(self.processor.mm_preprocess_cache), 0)
            retry = await self._process()
            self._assert_parity(await self._process(), retry)

        asyncio.run(run())

    def test_cancelled_mixed_request_closes_untagged_inflight_decoder(self):
        async def run():
            await self._process()
            started, release, closed = (
                threading.Event(),
                threading.Event(),
                threading.Event(),
            )
            real_load = video_cache.load_video

            def delayed_load(*args, **kwargs):
                decoder = real_load(*args, **kwargs)
                original_close = decoder.close

                def close():
                    original_close()
                    closed.set()

                decoder.close = close
                started.set()
                if not release.wait(10):
                    decoder.close()
                    raise TimeoutError("test did not release the video reader")
                return decoder

            request = self._request()
            request.video_data.append(VideoData(self.video_path))
            with patch.object(video_cache, "load_video", side_effect=delayed_load):
                task = asyncio.create_task(
                    self._process(request, prompt=f"{VIDEO_TOKEN} {VIDEO_TOKEN}")
                )
                try:
                    self.assertTrue(await asyncio.to_thread(started.wait, 10))
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                finally:
                    release.set()
                self.assertTrue(await asyncio.to_thread(closed.wait, 10))

        asyncio.run(run())

    def test_real_decode_failure_does_not_poison_caller_identity(self):
        async def run():
            broken = Path(self.directory.name) / "broken.mp4"
            broken.write_bytes(b"not a video container")
            with self.assertRaises(ValueError):
                await self._process(self._request(path=str(broken)))
            self.assertEqual(len(self.processor.mm_preprocess_cache), 0)
            retry = await self._process()
            self._assert_parity(await self._process(), retry)

        asyncio.run(run())

    def test_eviction_and_flush_force_real_reprocessing(self):
        async def run():
            cache = self.processor.mm_preprocess_cache
            with patch.object(
                video_cache, "load_video", wraps=video_cache.load_video
            ) as load:
                first = await self._process()
                # Exercise byte accounting of actual video tensors rather than
                # relying only on the generic cache's max-entry bound.
                self.assertGreater(cache.current_size_bytes, 0)
                cache.max_size_bytes = cache.current_size_bytes + 64
                await self._process(self._request(identity="clip-v2"))
                self.assertEqual(len(cache), 1)
                self.assertEqual(cache.evictions, 1)
                self._assert_parity(await self._process(), first)
                self.assertEqual(load.call_count, 3)
                self.processor.clear_preprocess_cache()
                self.assertEqual(cache.current_size_bytes, 0)
                self._assert_parity(await self._process(), first)
                self.assertEqual(load.call_count, 4)

        asyncio.run(run())

    def test_flush_during_preprocessing_does_not_repopulate_cache(self):
        async def run():
            started, release = asyncio.Event(), asyncio.Event()
            real_preprocess = qwen_vl.preprocess_video

            async def delayed(decoder, **kwargs):
                started.set()
                await release.wait()
                return await real_preprocess(decoder, **kwargs)

            with patch.object(qwen_vl, "preprocess_video", side_effect=delayed):
                first = asyncio.create_task(self._process())
                await asyncio.wait_for(started.wait(), 10)
                self.processor.clear_preprocess_cache()
                release.set()
                result = await asyncio.wait_for(first, 10)
            self.assertEqual(len(self.processor.mm_preprocess_cache), 0)
            with patch.object(
                video_cache, "load_video", wraps=video_cache.load_video
            ) as load:
                self._assert_parity(await self._process(), result)
                self.assertEqual(load.call_count, 1)

        asyncio.run(run())

    def test_requests_without_ids_keep_the_uncached_path(self):
        async def run():
            with (
                patch.object(
                    base_processor, "load_video", wraps=base_processor.load_video
                ) as load,
                patch.object(
                    video_cache,
                    "process_cached_qwen_video",
                    side_effect=AssertionError("no-ID request entered cache path"),
                ),
            ):
                request = self._request(identity=None)
                first = await self._process(request)
                self._assert_parity(await self._process(request), first)
                self.assertEqual(load.call_count, 2)
                self.assertEqual(len(self.processor.mm_preprocess_cache), 0)

        asyncio.run(run())

    def test_invalid_requests_fail_before_reading_video(self):
        async def run():
            requests = []
            for identity in ("", " ", "x" * 257, 42):
                requests.append((self._request(identity=identity), PROMPT))
            for tenant in (None, "", " "):
                requests.append((self._request(tenant=tenant), PROMPT))
            for options in (
                {"fps": 2, "nframes": 4},
                {"fps": float("nan")},
                {"max_frames": 0},
                {"unknown": 1},
                [],
            ):
                requests.append((self._request(options=options), PROMPT))
            requests.extend(
                [(self._request(), "hello"), (self._request(), VIDEO_TOKEN * 2)]
            )
            for name in ("image_data", "audio_data", "mm_hashes"):
                request = self._request()
                setattr(request, name, ["unsupported"])
                requests.append((request, PROMPT))
            with patch.object(
                video_cache, "load_video", side_effect=AssertionError("unexpected I/O")
            ):
                for request, prompt in requests:
                    with self.subTest(request=request, prompt=prompt):
                        with self.assertRaises(ValueError):
                            await self._process(request, prompt=prompt)
                for attribute, value in (
                    ("model_type", "qwen3_vl"),
                    ("supports_video_cache_ids", False),
                ):
                    with patch.object(self.processor, attribute, value):
                        with self.assertRaises(ValueError):
                            await self._process()
                self.processor.mm_preprocess_cache.max_size_bytes = 0
                with self.assertRaises(ValueError):
                    await self._process()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
