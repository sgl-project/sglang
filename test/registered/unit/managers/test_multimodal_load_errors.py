import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import ImageData, load_image
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


BAD_IMAGE_DATA_URI = (
    "data:image/jpeg;base64,"
    "R0lGODlhEAAQAPZAAI6Ojp2dnampqaysrLW1tbW1tre3t7i4uLm5ubi4vbq6v7+/v8PDw8fHysvLy8zMzM3Nzc/Pz8/P19LS0tTU1NXV1dXV1tfX19fX2NbW2tfX2tDQ3NfX3dnZ2dvb29vb3tzc3N3d3d/f39nZ4Nra4dvb59ra6t7e6d3d7N/f7d7e7+Dg4ODg4ePj4+Pj5OTk5OXl5eTk5+bm5unp6e3t7e/v7/Dw8PHx8fLy8vPz8/T09Pj4+Pr6+vz8/P39/f7+/v///wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAAAAAAALAAAAAAQABAAAAeLgECCg4SFhDstFRAQDhAULTuFNBEVNj1APTs4FBE0gzidPKKjojQQOIIUKzk2ra04ryIUggw2M7cxJB4ZFL0iDIILNDEtHxnHyMcLgg4UuR8kJNDSJA0OgjQIHCgl3d4lEgSogi0EDQoo6CYmGwkELoU6AigC9AMIEzmGQAEBHvqGAPz9G0iwoEF9gQAAOw=="
)


class _TestMultimodalProcessor(BaseMultimodalProcessor):
    gpu_image_decode = False

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class _FakeServing(OpenAIServingBase):
    def __init__(self, exc):
        tokenizer_manager = SimpleNamespace(
            server_args=SimpleNamespace(),
            request_logger=SimpleNamespace(log_requests=False, log_requests_level=0),
        )
        super().__init__(tokenizer_manager)
        self.exc = exc

    def _request_id_prefix(self):
        return "chatcmpl-"

    def _convert_to_internal_request(self, request, raw_request=None):
        return object(), request

    async def _handle_non_streaming_request(
        self, adapted_request, request, raw_request
    ):
        raise self.exc


class MultimodalLoadErrorTestCase(unittest.TestCase):
    def _make_processor(self, *, skip_tokenizer_init):
        return _TestMultimodalProcessor(
            hf_config=None,
            server_args=SimpleNamespace(
                mm_process_config={},
                keep_mm_feature_on_device=False,
                skip_tokenizer_init=skip_tokenizer_init,
            ),
            _processor=SimpleNamespace(),
            transport_mode=None,
        )

    async def _load_bad_media(self, *, skip_tokenizer_init, prompt, tokens, **mm_data):
        processor = self._make_processor(skip_tokenizer_init=skip_tokenizer_init)
        try:
            await processor.load_mm_data(
                prompt=prompt,
                multimodal_tokens=tokens,
                **mm_data,
            )
        finally:
            processor.io_executor.shutdown(wait=True)
            processor.cpu_executor.shutdown(wait=True)

    async def _load_bad_image(self, *, skip_tokenizer_init):
        tokens = MultimodalSpecialTokens(image_token="<image>").build(SimpleNamespace())
        await self._load_bad_media(
            skip_tokenizer_init=skip_tokenizer_init,
            prompt="<image>",
            tokens=tokens,
            image_data=[ImageData(url=BAD_IMAGE_DATA_URI)],
        )

    async def _load_bad_audio(self, *, skip_tokenizer_init):
        tokens = MultimodalSpecialTokens(audio_token="<audio>").build(SimpleNamespace())
        await self._load_bad_media(
            skip_tokenizer_init=skip_tokenizer_init,
            prompt="<audio>",
            tokens=tokens,
            audio_data=[b"not audio"],
        )

    async def _load_bad_video(self, *, skip_tokenizer_init):
        tokens = MultimodalSpecialTokens(video_token="<video>").build(SimpleNamespace())
        await self._load_bad_media(
            skip_tokenizer_init=skip_tokenizer_init,
            prompt="<video>",
            tokens=tokens,
            video_data=[b"not video"],
        )

    def _assert_maps_to_http_400(self, exc):
        response = asyncio.run(
            _FakeServing(exc).handle_request(
                SimpleNamespace(stream=False),
                raw_request=None,
            )
        )

        body = json.loads(response.body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(body["code"], 400)
        self.assertEqual(body["type"], "BadRequest")

    def test_bad_image_fast_loader_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "broken data stream"):
            asyncio.run(self._load_bad_image(skip_tokenizer_init=False))

    def test_bad_image_legacy_loader_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "broken data stream"):
            asyncio.run(self._load_bad_image(skip_tokenizer_init=True))

    def test_bad_audio_fast_loader_raises_value_error(self):
        with self.assertRaises(ValueError):
            asyncio.run(self._load_bad_audio(skip_tokenizer_init=False))

    def test_bad_audio_legacy_loader_raises_value_error(self):
        with self.assertRaises(ValueError):
            asyncio.run(self._load_bad_audio(skip_tokenizer_init=True))

    def test_bad_video_fast_loader_raises_value_error(self):
        with self.assertRaises(ValueError):
            asyncio.run(self._load_bad_video(skip_tokenizer_init=False))

    def test_bad_video_legacy_loader_raises_value_error(self):
        with self.assertRaises(ValueError):
            asyncio.run(self._load_bad_video(skip_tokenizer_init=True))

    def test_cpu_image_decode_unidentified_image_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Invalid image data"):
            load_image(b"not an image", gpu_image_decode=False)

    def test_cpu_image_decode_broken_stream_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "broken data stream"):
            load_image(ImageData(url=BAD_IMAGE_DATA_URI), gpu_image_decode=False)

    def test_bad_media_value_errors_map_to_http_400(self):
        loaders = [
            self._load_bad_image,
            self._load_bad_audio,
            self._load_bad_video,
        ]
        for loader in loaders:
            with self.subTest(loader=loader.__name__):
                try:
                    asyncio.run(loader(skip_tokenizer_init=False))
                except ValueError as exc:
                    self._assert_maps_to_http_400(exc)
                else:
                    self.fail(f"{loader.__name__} did not raise ValueError")


if __name__ == "__main__":
    unittest.main()
