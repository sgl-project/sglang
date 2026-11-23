"""M3.1 media resize, per-request limits, and token expansion."""

import asyncio
import base64
import io
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch
from PIL import Image
from transformers import MiniMaxM3VLImageProcessor, ProcessorMixin

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageContentImageURL,
    ChatCompletionMessageContentVideoURL,
)
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.models.minimax_vl_common import (
    DEFERRED_IMAGE_KEY,
    MiniMaxDeferredImage,
    _stack_pixel_values,
    finish_deferred_image,
)
from sglang.srt.multimodal.cache import snapshot_media
from sglang.srt.multimodal.media_io import ImageData, VideoData
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.minimax_m3_vl import (
    MAX_IMAGE_PIXELS,
    MiniMaxM3VLProcessor,
    _compute_aligned_size,
    _compute_sampled_frame_indices,
    _expand_media_tokens_scan,
    _LoadedImage,
    expand_media_tokens,
    preprocess_for_encoder,
    resize_images_for_request,
    video_frame_timestamps,
)
from sglang.srt.multimodal.processors.minimax_m31_media import (
    MiniMaxM31ImageProcessor,
    MiniMaxM31VideoProcessor,
)
from sglang.srt.parser.jinja_template_utils import process_content_for_template_format
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FACTOR = 28
# From the MiniMax-M3.1 checkpoint's preprocessor_config.json.
M3_VL_PREPROCESSOR = {
    "patch_size": 14,
    "merge_size": 2,
    "temporal_patch_size": 2,
    "max_pixels": 451584,
    "size": {"height": 672, "width": 672},
}


def _image(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


class FakeDecoder:
    def __init__(self, num_frames, fps, height, width):
        self.num_frames = num_frames
        self.avg_fps = fps
        self.frame_shape = (height, width)
        self.get_frames_as_tensor = Mock(
            side_effect=lambda indices: torch.zeros(
                len(indices), height, width, 3, dtype=torch.uint8
            )
        )

    def __len__(self):
        return self.num_frames


class TestMiniMaxM31Resize(CustomTestCase):
    def test_aligned_size(self):
        for (height, width, cap), expected in (
            # The cap rounds down to the factor: 500 -> 476.
            ((2000, 2000, 500), (476, 476)),
            # A short side under 112 is enlarged to 112.
            ((50, 100, 1008), (112, 224)),
            # Shrinking a long side over the cap takes precedence over the short-side floor.
            ((50, 2000, 1008), (28, 1008)),
            # An upscale that would push the long side past the cap stops at the cap.
            ((50, 1000, 1008), (56, 1008)),
            # Sides round to the nearest factor; floor would give 980.
            ((345, 1000, 1008), (336, 1008)),
        ):
            with self.subTest(size=(height, width), cap=cap):
                self.assertEqual(
                    _compute_aligned_size(height, width, FACTOR, cap), expected
                )
        with self.assertRaises(ValueError):
            _compute_aligned_size(10, 2100, FACTOR, 1008)

    def test_rectangular_image_matches_checkpoint_resize(self):
        image = _image(500, 900, seed=42)
        processor = MiniMaxM31ImageProcessor(max_size=(672, 672))
        actual = processor(images=[image], return_tensors="pt")
        expected = MiniMaxM3VLImageProcessor()(
            images=[image.resize((392, 672), Image.Resampling.BICUBIC)],
            do_resize=False,
            return_tensors="pt",
        )
        self.assertEqual(actual.image_grid_thw.tolist(), [[1, 48, 28]])
        self.assertEqual(processor.get_number_of_image_patches(900, 500), 48 * 28)
        torch.testing.assert_close(actual.pixel_values, expected.pixel_values)

    def test_video_and_image_keep_the_same_spatial_layout(self):
        frame = np.random.default_rng(42).integers(
            0, 256, (100, 160, 3), dtype=np.uint8
        )
        image = MiniMaxM31ImageProcessor(max_size=(112, 112))(
            images=[Image.fromarray(frame)], return_tensors="pt"
        )
        video = MiniMaxM31VideoProcessor(max_size=(112, 112))(
            videos=[np.stack([frame, frame])],
            do_sample_frames=False,
            return_tensors="pt",
        )
        torch.testing.assert_close(video.video_grid_thw, image.image_grid_thw)
        torch.testing.assert_close(video.pixel_values_videos, image.pixel_values)


class _ProcessorCase(CustomTestCase):
    def setUp(self):
        self.image_processor = MiniMaxM31ImageProcessor(max_size=(672, 672))
        processor = object.__new__(MiniMaxM3VLProcessor)
        processor._tokenizer = None
        processor.IM_TOKEN_ID = 10
        processor.VIDEO_TOKEN_ID = 11
        processor.IM_START_TOKEN_ID = 12
        processor.IM_END_TOKEN_ID = 13
        processor.mm_tokens = None
        processor.image_config = {}
        processor.video_config = {}
        processor.video_fps = 1.0
        processor.video_frame_max_size = None
        processor.video_max_frames = None
        processor._processor = SimpleNamespace(
            image_processor=self.image_processor,
            video_processor=SimpleNamespace(
                patch_size=14, merge_size=2, max_size=(672, 672)
            ),
        )
        processor.process_and_combine_mm_data_async = AsyncMock(
            return_value=([], [1, 2, 3], {})
        )
        processor.io_executor = ThreadPoolExecutor(2)
        self.addCleanup(processor.io_executor.shutdown)
        processor.deferred_image = None
        self.processor = processor

    def run_processor(self, images=(), image_data=None, videos=(), video_data=None):
        self.processor.load_mm_data = AsyncMock(
            return_value=SimpleNamespace(images=list(images), videos=list(videos))
        )
        asyncio.run(
            self.processor.process_mm_data_async(
                image_data=image_data,
                audio_data=None,
                input_text="prompt",
                request_obj=SimpleNamespace(video_data=video_data),
            )
        )
        call = self.processor.process_and_combine_mm_data_async.call_args
        return call.kwargs["base_output"], call.kwargs


class TestImageLimits(_ProcessorCase):
    def test_image_without_option_uses_default_long_side(self):
        # The verifier expects 3000x2000 at the default tier to become 2016x1344.
        output, kwargs = self.run_processor([_image(3000, 2000)], [ImageData(url="x")])
        self.assertTrue(kwargs["images_resized"])
        self.assertEqual(output.images[0].size, (2016, 1344))

    def test_max_long_side_pixel_values(self):
        self.assertEqual(3584 * 3584, MAX_IMAGE_PIXELS)
        for size, value, expected in (
            ((1500, 1000), 112, (112, 84)),
            ((3584, 3584), 3584, (3584, 3584)),
        ):
            with self.subTest(value=value):
                output, _ = self.run_processor(
                    [Image.new("RGB", size)],
                    [ImageData(url="x", max_long_side_pixel=value)],
                )
                self.assertEqual(output.images[0].size, expected)
        for value in (0, 84, 130, 3612, 1.5, "504", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_processor(
                    [_image(64, 64)], [ImageData(url="x", max_long_side_pixel=value)]
                )

    def test_image_is_resized_the_same_alone_or_mixed(self):
        """An image without max_long_side_pixel gets the same pixels whether or not
        another image in the request sets it."""
        plain = _image(1920, 1080, seed=1)
        alone, _ = self.run_processor([plain], [ImageData(url="a")])
        mixed, _ = self.run_processor(
            [plain, _image(1500, 1000)],
            [ImageData(url="a"), ImageData(url="b", max_long_side_pixel=252)],
        )
        self.assertTrue(np.array_equal(mixed.images[0], alone.images[0]))

    def test_server_max_pixels_keeps_hf_resize(self):
        """With a server image max_pixels, an image without max_long_side_pixel gets
        the HF processor's pixels; one with the option still follows the size spec."""
        # Built the way get_processor loads this checkpoint: its stored size never
        # drove the resize and is dropped there.
        image_processor = MiniMaxM3VLImageProcessor(
            **{**M3_VL_PREPROCESSOR, "size": None}
        )
        self.processor._processor.image_processor = image_processor
        self.processor.image_config = {"max_pixels": 200704}
        plain = _image(1920, 1080, seed=1)
        output, _ = self.run_processor(
            [plain, _image(1500, 1000)],
            [ImageData(url="a"), ImageData(url="b", max_long_side_pixel=252)],
        )
        # A per-call pixel cap goes through size; max_pixels only applies at init.
        expected = image_processor(
            images=[plain],
            return_tensors="pt",
            size={
                "shortest_edge": image_processor.size["shortest_edge"],
                "longest_edge": 200704,
            },
        )
        actual = image_processor(
            images=[output.images[0]], return_tensors="pt", do_resize=False
        )
        self.assertTrue(torch.equal(actual.pixel_values, expected.pixel_values))
        self.assertEqual(output.images[1].size, (252, 168))

    def test_precomputed_image_is_not_resized(self):
        image = object()
        output, _ = self.run_processor([image], [ImageData(url="x")])
        self.assertIs(output.images[0], image)

    def test_concurrent_resize_matches_encoder_path(self):
        image_processor = MiniMaxM3VLImageProcessor(
            **{**M3_VL_PREPROCESSOR, "size": None}
        )
        self.processor._processor.image_processor = image_processor
        self.processor.image_config = {"max_pixels": 200704}
        images = [_image(1920, 1080, seed=1), _image(1500, 1000, seed=2)] * 2
        image_data = [
            ImageData(url="a"),
            ImageData(url="b", max_long_side_pixel=252),
            ImageData(url="c", max_long_side_pixel=1008),
            ImageData(url="d"),
        ]
        output, _ = self.run_processor(images, image_data)
        expected = resize_images_for_request(
            images, image_data, image_processor=image_processor, max_pixels=200704
        )
        self.assertEqual(
            [image.size for image in output.images],
            [(588, 336), (252, 168), (1008, 560), (532, 364)],
        )
        for actual, reference in zip(output.images, expected):
            self.assertTrue(np.array_equal(actual, reference))


def _png_bytes(width, height, color=(40, 80, 120), mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


class TestImageIdentity(_ProcessorCase):
    def setUp(self):
        super().setUp()
        self.processor.processor_fingerprint = "sha256:" + "ab" * 32

        async def combine(*, base_output, **kwargs):
            items = []
            for image in base_output.images:
                width, height = image.size
                items.append(
                    MultimodalDataItem(
                        modality=Modality.IMAGE,
                        feature=torch.zeros(1),
                        model_specific_data={
                            "image_grid_thw": torch.tensor(
                                [[1, height // 14, width // 14]]
                            )
                        },
                    )
                )
            return items, torch.tensor([1]), {}

        self.processor.process_and_combine_mm_data_async = combine

    def identities(self, images, limits=None):
        loaded = [
            MiniMaxM3VLProcessor._load_single_item(image, Modality.IMAGE)
            for image in images
        ]
        self.processor.load_mm_data = AsyncMock(
            return_value=SimpleNamespace(images=loaded, videos=[])
        )
        output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=[
                    ImageData(url="x", max_long_side_pixel=limit)
                    for limit in limits or [None] * len(images)
                ],
                audio_data=None,
                input_text=[1],
                request_obj=SimpleNamespace(video_data=None),
            )
        )
        return [item.identity for item in output.mm_items]

    def test_loader_hashes_the_bytes_it_decodes(self):
        payload = _png_bytes(64, 48)
        url = "data:image/png;base64," + base64.b64encode(payload).decode()
        for data in (payload, url, ImageData(url=url, max_long_side_pixel=504)):
            with self.subTest(data=type(data).__name__):
                loaded = MiniMaxM3VLProcessor._load_single_item(data, Modality.IMAGE)
                self.assertIsInstance(loaded, _LoadedImage)
                self.assertEqual(
                    loaded.content_digest, snapshot_media(payload).content_digest
                )
                self.assertTrue(
                    np.array_equal(loaded.image, Image.open(io.BytesIO(payload)))
                )

    def test_identity_covers_bytes_options_and_config(self):
        first, second = _png_bytes(3000, 2000), _png_bytes(3000, 2000, (0, 0, 0))
        small = _png_bytes(640, 480)
        [plain] = self.identities([first])
        self.assertEqual(len(plain), 32)
        self.assertEqual(self.identities([first]), [plain])
        self.assertEqual(self.identities([first], [2016]), [plain])
        self.assertNotEqual(self.identities([second]), [plain])
        self.assertNotEqual(self.identities([first], [1008]), [plain])
        self.assertEqual(
            self.identities([first, small]), self.identities([small, first])[::-1]
        )
        self.processor.processor_fingerprint = "sha256:" + "cd" * 32
        self.assertNotEqual(self.identities([first]), [plain])

    def test_inputs_without_digests_keep_feature_hash(self):
        self.processor.load_mm_data = AsyncMock(
            return_value=SimpleNamespace(images=[_image(64, 64)], videos=[])
        )
        output = asyncio.run(
            self.processor.process_mm_data_async(
                image_data=[ImageData(url="x")],
                audio_data=None,
                input_text=[1],
                request_obj=SimpleNamespace(video_data=None),
            )
        )
        self.assertIsNone(output.mm_items[0].identity)
        bundled = MultimodalDataItem(
            modality=Modality.IMAGE,
            model_specific_data={"image_grid_thw": torch.tensor([[1, 4, 4]] * 2)},
        )
        self.processor._set_image_identities(
            [bundled],
            content_digests=[snapshot_media(_png_bytes(8, 8)).content_digest] * 2,
            resize_specs=[("max_long_side_pixel", 2016)] * 2,
        )
        self.assertIsNone(bundled.identity)

    def test_malformed_image_fails_its_own_load(self):
        payload = b"\x89PNG\r\n\x1a\nnot an image"
        with self.assertRaises(ValueError):
            MiniMaxM3VLProcessor._load_single_item(payload, Modality.IMAGE)


class TestPreprocessThreads(CustomTestCase):
    def test_preprocessing_threads_are_scoped_to_the_call(self):
        processor = object.__new__(MiniMaxM3VLProcessor)
        processor._tokenizer = None
        processor.IM_TOKEN_ID = 10
        processor.VIDEO_TOKEN_ID = 11
        processor.IM_START_TOKEN_ID = 12
        processor.image_config = {}
        processor.preprocess_num_threads = 4
        seen = []

        def image_processor(**kwargs):
            seen.append(torch.get_num_threads())
            return {"image_grid_thw": [[1, 2, 4]]}

        processor._processor = SimpleNamespace(
            image_processor=image_processor,
            tokenizer=Mock(encode=Mock(return_value=[12, 10, 10, 13])),
        )
        processor._image_placeholder_text = lambda grid: "image"
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        for worker_threads in (1, 16):
            torch.set_num_threads(worker_threads)
            processor.process_mm_data(None, images=[object()], input_ids=[10])
            self.assertEqual(torch.get_num_threads(), worker_threads)
        self.assertEqual(seen, [4, 16])


class TestVideoLimits(_ProcessorCase):
    def test_video_options(self):
        for item, fps, size in (
            ("v", 1.0, (308, 448)),
            (VideoData(url="v", preprocess_kwargs={"fps": 0.2}), 0.2, (308, 448)),
            (
                VideoData(
                    url="v", preprocess_kwargs={"fps": 5}, max_long_side_pixel=252
                ),
                5.0,
                (168, 252),
            ),
        ):
            with self.subTest(item=item):
                output, kwargs = self.run_processor(
                    videos=[FakeDecoder(30, 10.0, 300, 450)], video_data=[item]
                )
                expected = _compute_sampled_frame_indices(30, 10.0, fps)
                self.assertEqual(
                    kwargs["video_metadata"][0]["frames_indices"], expected
                )
                self.assertEqual(
                    tuple(output.videos[0].shape), (len(expected), 3, *size)
                )
        for item in (
            VideoData(url="v", preprocess_kwargs={"fps": 0.1}),
            VideoData(url="v", preprocess_kwargs={"fps": 5.5}),
            VideoData(url="v", preprocess_kwargs={"fps": 0}),
            VideoData(url="v", preprocess_kwargs={"fps": True}),
            VideoData(url="v", max_long_side_pixel=0),
        ):
            with self.subTest(item=item), self.assertRaises(ValueError):
                self.run_processor(
                    videos=[FakeDecoder(30, 10.0, 56, 56)], video_data=[item]
                )

    def test_video_pixel_limit_rejects_before_decode(self):
        # 60 s at fps=5 -> 301 frames of 1008x1008 > MAX_VIDEO_PIXELS.
        decoder = FakeDecoder(num_frames=1800, fps=30.0, height=1008, width=1008)
        decoder.get_frames_as_tensor.side_effect = AssertionError(
            "Frames must not be decoded"
        )
        with self.assertRaises(ValueError):
            self.run_processor(
                videos=[decoder],
                video_data=[
                    VideoData(
                        url="v", preprocess_kwargs={"fps": 5}, max_long_side_pixel=1008
                    )
                ],
            )


class TestRequestPlumbing(CustomTestCase):
    def test_fields_reach_media_data(self):
        image_data, video_data = [], []
        image_part = ChatCompletionMessageContentImageURL(
            url="img", max_long_side_pixel=504
        )
        video_parts = (
            ChatCompletionMessageContentVideoURL(
                url="vid", fps=2.0, max_long_side_pixel=252
            ).model_dump(),
            ChatCompletionMessageContentVideoURL(
                url="vid", max_long_side_pixel=252
            ).model_dump(),
            {"url": "vid"},
        )
        process_content_for_template_format(
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": image_part.model_dump()}]
                + [{"type": "video_url", "video_url": part} for part in video_parts],
            },
            "openai",
            image_data,
            video_data,
            [],
            [],
        )
        self.assertEqual(image_data[0].max_long_side_pixel, 504)
        self.assertEqual(
            video_data,
            [
                VideoData(
                    url="vid", preprocess_kwargs={"fps": 2.0}, max_long_side_pixel=252
                ),
                VideoData(url="vid", max_long_side_pixel=252),
                "vid",
            ],
        )


class TestMiniMaxM31Processor(CustomTestCase):
    def setUp(self):
        self.processor = object.__new__(MiniMaxM3VLProcessor)
        self.processor._tokenizer = None
        self.processor.IM_TOKEN_ID = 10
        self.processor.VIDEO_TOKEN_ID = 11
        self.processor.IM_START_TOKEN_ID = 12
        self.processor.IM_END_TOKEN_ID = 13
        self.processor.image_config = {}
        self.processor.video_config = {}
        self.image = [12, 10, 10, 13]
        self.video = [20, 12, 11, 11, 13, 21, 12, 11, 11, 13]
        tokenizer = Mock()
        tokenizer.encode.side_effect = lambda text, **kw: {
            "image": self.image,
            "video": self.video,
        }[text]
        tokenizer.decode.side_effect = AssertionError("Text must not be decoded")
        self.processor._processor = SimpleNamespace(
            tokenizer=tokenizer,
            image_processor=Mock(return_value={"image_grid_thw": [[1, 2, 4]]}),
            video_processor=Mock(return_value={"video_grid_thw": [[2, 2, 4]]}),
        )
        self.processor._image_placeholder_text = lambda grid: "image"
        self.processor._video_placeholder_text = lambda grid, timestamps: "video"

    def process(self, ids, **kwargs):
        return self.processor.process_mm_data(None, input_ids=ids, **kwargs)[
            "input_ids"
        ][0].tolist()

    def test_expanded_media_is_not_expanded_twice(self):
        ids = [101] + self.image + [102] + self.video + [103]
        self.assertEqual(self.process(ids, images=[object()], videos=[object()]), ids)
        self.assertEqual(
            self.process(
                [101] + self.image + [11, 103], images=[object()], videos=[object()]
            ),
            [101] + self.image + self.video + [103],
        )

    def test_media_mismatches_are_rejected(self):
        for ids in ([101], [10, 10], [12, 10, 13]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.process(ids, images=[object()])

    def test_processor_without_replace_hooks_expands_token_ids(self):
        """A checkpoint processor that defines its own __call__ but not the optional
        replace_image_token hook must still expand token-ID image prompts."""
        del self.processor._image_placeholder_text
        hf_processor = object.__new__(ProcessorMixin)
        hf_processor.tokenizer = Mock(encode=Mock(return_value=self.image))
        hf_processor.image_processor = Mock(
            return_value={"image_grid_thw": [[1, 2, 4]]}, merge_size=2
        )
        self.processor._processor = hf_processor
        self.assertEqual(
            self.process([101, 10, 103], images=[object()]),
            [101] + self.image + [103],
        )

    def test_resized_images_are_not_resized_again(self):
        """An image already resized for the request keeps its size; the processor's
        672 cap would otherwise shrink 1008x672 to 672x448."""
        self.processor._processor.image_processor = MiniMaxM31ImageProcessor(
            max_size=(672, 672)
        )
        result = self.processor.process_mm_data(
            None, images=[_image(1008, 672)], input_ids=[10], images_resized=True
        )
        self.assertEqual(result["image_grid_thw"].tolist(), [[1, 48, 72]])

    def test_epd_rebuild_matches_colocated_expansion(self):
        """EPD prompts expand like colocated ones, and encoder rows split per item and video frame."""
        self.processor._video_placeholder_text = lambda grid, timestamps: "video"
        ids = [101, 10, 102, 11, 103]
        output = self.processor.get_validated_mm_data(
            ids,
            {
                Modality.IMAGE: torch.zeros(2, 1),
                Modality.VIDEO: torch.zeros(4, 1),
            },
            img_grid_thw=torch.tensor([[1, 2, 4]]),
            video_grid_thw=torch.tensor([[2, 2, 4]]),
        )
        self.assertEqual(
            output.input_ids, self.process(ids, images=[object()], videos=[object()])
        )
        self.assertEqual(
            [item.offsets for item in output.mm_items],
            [[(2, 3)], [(8, 9), (13, 14)]],
        )


class TestMiniMaxM31FeatureTransportDtype(CustomTestCase):
    def test_model_dtype_transport_keeps_encoder_input(self):
        """Features cast to the model dtype before transport must reach the vision
        encoder bit for bit as the fp32 features did, at half the bytes."""
        processor = object.__new__(MiniMaxM3VLProcessor)
        processor._tokenizer = None
        processor.feature_dtype = torch.bfloat16
        pixel_values = torch.randn(
            256, 1176, generator=torch.Generator().manual_seed(0)
        )
        item = SimpleNamespace(feature=pixel_values.clone(), model_specific_data={})
        processor._postprocess_mm_items_before_transport([item], images=None)

        self.assertEqual(item.feature.dtype, torch.bfloat16)
        weight = torch.empty(1, dtype=torch.bfloat16)
        vision_tower = SimpleNamespace(
            vision_model=SimpleNamespace(
                embeddings=SimpleNamespace(
                    patch_embedding=SimpleNamespace(weight=weight)
                )
            ),
            dtype=torch.bfloat16,
        )
        self.assertTrue(
            torch.equal(
                _stack_pixel_values(vision_tower, [item]),
                _stack_pixel_values(
                    vision_tower,
                    [SimpleNamespace(feature=pixel_values, model_specific_data={})],
                ),
            )
        )


class TestMiniMaxM31EncoderPreprocessing(CustomTestCase):
    def test_image_placeholder_matches_hf_expansion(self):
        """Image text built without the processor's hook must equal the HF expansion."""
        from transformers import MiniMaxM3VLProcessor as HFMiniMaxM3VLProcessor

        hf_processor = object.__new__(HFMiniMaxM3VLProcessor)
        hf_processor.image_processor = MiniMaxM31ImageProcessor(max_size=(112, 112))
        processor = object.__new__(MiniMaxM3VLProcessor)
        processor._tokenizer = None
        processor._processor = hf_processor
        grid = torch.tensor([[1, 4, 6]])
        self.assertEqual(
            processor._image_placeholder_text(grid[0]),
            hf_processor.replace_image_token({"image_grid_thw": grid}, 0),
        )

    def test_video_placeholder_matches_hf_expansion(self):
        """EPD video text built from encoder timestamps must equal the HF expansion."""
        from transformers import MiniMaxM3VLProcessor as HFMiniMaxM3VLProcessor

        hf_processor = object.__new__(HFMiniMaxM3VLProcessor)
        hf_processor.video_processor = MiniMaxM31VideoProcessor(max_size=(112, 112))
        processor = object.__new__(MiniMaxM3VLProcessor)
        processor._tokenizer = None
        processor._processor = hf_processor
        grid = torch.tensor([[3, 4, 6]])
        for metadata in (
            SimpleNamespace(fps=2.0, frames_indices=[0, 3, 7, 9, 12]),
            None,
        ):
            timestamps = (
                None
                if metadata is None
                else video_frame_timestamps(
                    vars(metadata), 3, hf_processor.video_processor.temporal_patch_size
                )
            )
            with self.subTest(metadata=metadata):
                self.assertEqual(
                    processor._video_placeholder_text(grid[0], timestamps),
                    hf_processor.replace_video_token(
                        {"video_grid_thw": grid, "video_metadata": [metadata]}, 0
                    ),
                )

    def test_encoder_images_keep_per_item_resize(self):
        """The encoder must honor each image's max_long_side_pixel like the colocated path."""
        image = _image(500, 900)
        output = preprocess_for_encoder(
            [{"type": "image", "image": image, "max_long_side_pixel": 448}, image],
            Modality.IMAGE,
            processor=SimpleNamespace(
                image_processor=MiniMaxM31ImageProcessor(max_size=(672, 672))
            ),
            mm_process_config={},
        )
        self.assertEqual(output["image_grid_thw"].tolist(), [[1, 32, 18], [1, 64, 36]])


class TestMiniMaxM31Placeholders(CustomTestCase):
    def test_only_m3_media_tokens_are_placeholders(self):
        """User text such as "<image>" must not count as a media placeholder."""

        def fake_base_init(self, *args, **kwargs):
            self.video_config = {}
            self.mm_processor_worker_num = 1

        tokenizer = Mock(convert_tokens_to_ids=Mock(return_value=1))
        disagg = SimpleNamespace(
            disaggregation_mode="null", disaggregation_decode_enable_radix_cache=False
        )
        with (
            patch.object(BaseMultimodalProcessor, "__init__", fake_base_init),
            patch(
                "sglang.srt.multimodal.processors.minimax_m3_vl.get_disagg",
                return_value=disagg,
            ),
            patch(
                "sglang.srt.multimodal.processors.minimax_m3_vl.get_serving",
                return_value=SimpleNamespace(tokenizer_worker_num=1),
            ),
        ):
            processor = MiniMaxM3VLProcessor(
                None, None, SimpleNamespace(tokenizer=tokenizer)
            )
        prompt = (
            "<image> <|image|> <|image_pad|> <video> <|video|> ]<]image[>[ ]<]video[>["
        )
        self.assertEqual(
            processor.mm_tokens.get_combined_regex().findall(prompt),
            ["]<]image[>[", "]<]video[>["],
        )


_FAKE_SPECIALS = {
    MiniMaxM3VLProcessor.IMAGE_START_TOKEN: 12,
    MiniMaxM3VLProcessor.IMAGE_TOKEN: 10,
    MiniMaxM3VLProcessor.IMAGE_END_TOKEN: 13,
    MiniMaxM3VLProcessor.VIDEO_TOKEN: 11,
}


def _fake_encode(text, **kwargs):
    pattern = "|".join(re.escape(token) for token in _FAKE_SPECIALS) + "|."
    return [
        _FAKE_SPECIALS[piece] if piece in _FAKE_SPECIALS else ord(piece) + 1000
        for piece in re.findall(pattern, text, flags=re.S)
    ]


def _png_data_url(width, height, mode="RGB", color=(40, 80, 120)):
    return (
        "data:image/png;base64,"
        + base64.b64encode(_png_bytes(width, height, color, mode)).decode()
    )


def _fake_processor(image_processor):
    processor = object.__new__(MiniMaxM3VLProcessor)
    processor.IM_TOKEN_ID = 10
    processor.VIDEO_TOKEN_ID = 11
    processor.IM_START_TOKEN_ID = 12
    processor.IM_END_TOKEN_ID = 13
    processor.image_config = {}
    processor.video_config = {}
    processor.io_executor = ThreadPoolExecutor(2)
    tokenizer = Mock(bos_token=None)
    tokenizer.encode.side_effect = _fake_encode
    processor._tokenizer = tokenizer
    processor._tokenizer_auto_adds_specials = False
    processor._processor = SimpleNamespace(
        tokenizer=tokenizer, image_processor=image_processor
    )
    return processor


class TestDeferredImagePreprocessing(CustomTestCase):
    def setUp(self):
        processor = _fake_processor(
            MiniMaxM3VLImageProcessor(**{**M3_VL_PREPROCESSOR, "size": None})
        )
        self.addCleanup(processor.io_executor.shutdown)
        processor._feature_transport = SimpleNamespace(mm_feature_transport="cpu")
        processor.feature_dtype = torch.float32
        self.processor = processor

    def enable(self):
        with patch(
            "sglang.srt.multimodal.processors.minimax_m3_vl.is_cuda",
            return_value=True,
        ):
            self.assertIsInstance(self.processor.deferred_image, MiniMaxDeferredImage)

    def finish(self, mm_items, device):
        return torch.cat(
            [
                finish_deferred_image(
                    item.feature,
                    item.model_specific_data[DEFERRED_IMAGE_KEY],
                    torch.device(device),
                ).cpu()
                for item in mm_items
            ]
        )

    def test_deferred_images_match_processor_output(self):
        self.enable()
        images = [
            _image(588, 336, seed=1),
            _image(252, 168, seed=2),
            _image(28, 5600, seed=3),
            _image(1008, 756, seed=4),
        ]
        ids = [101, 10, 102, 10, 103, 10, 10, 104]
        expected = self.processor.process_mm_data(
            None, images=images, input_ids=list(ids), images_resized=True
        )
        mm_items, input_ids = asyncio.run(
            self.processor._defer_image_preprocessing(images, ids)
        )
        self.assertEqual(input_ids, expected["input_ids"][0].tolist())
        self.assertEqual(
            [item.offsets[0] for item in mm_items],
            BaseMultimodalProcessor.get_mm_items_offset(expected["input_ids"][0], 10),
        )
        self.assertTrue(
            torch.equal(
                torch.cat([item.image_grid_thw for item in mm_items]),
                expected["image_grid_thw"],
            )
        )
        self.assertEqual(mm_items[0].feature.dtype, torch.uint8)
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                self.assertTrue(
                    torch.equal(
                        self.finish(mm_items, device),
                        expected["pixel_values"],
                    )
                )

    def test_request_path_matches_processor_path(self):
        self.enable()
        sources = [
            _png_data_url(3000, 2000),
            _png_data_url(333, 517, mode="RGBA", color=(10, 200, 30, 90)),
            _png_data_url(1000, 50, mode="L", color=77),
            _png_data_url(640, 480, mode="P", color=5),
        ]
        image_data = [ImageData(url=url) for url in sources]
        prompt = "a]<]image[>[b]<]image[>[c]<]image[>[d]<]image[>[e"

        def run(deferred, input_text):
            self.processor.__dict__["deferred_image"] = deferred
            return asyncio.run(
                self.processor.process_mm_data_async(
                    image_data=image_data,
                    audio_data=None,
                    input_text=input_text,
                    request_obj=SimpleNamespace(video_data=None),
                )
            )

        async def combine(*, base_output, images_resized, input_ids, **kwargs):
            result = self.processor.process_mm_data(
                None,
                images=base_output.images,
                input_ids=input_ids,
                images_resized=images_resized,
            )
            items = [
                MultimodalDataItem(
                    modality=Modality.IMAGE,
                    feature=feature,
                    offsets=[offset],
                    model_specific_data={"image_grid_thw": grid[None]},
                )
                for feature, grid, offset in zip(
                    result["pixel_values"].split(
                        result["image_grid_thw"].prod(-1).tolist()
                    ),
                    result["image_grid_thw"],
                    BaseMultimodalProcessor.get_mm_items_offset(
                        result["input_ids"][0], 10
                    ),
                )
            ]
            return items, result["input_ids"][0], result

        async def load(*, image_data, **kwargs):
            return SimpleNamespace(
                images=[
                    MiniMaxM3VLProcessor._load_single_item(item, Modality.IMAGE)
                    for item in image_data
                ],
                videos=[],
            )

        self.processor.process_and_combine_mm_data_async = combine
        self.processor.load_mm_data = load
        self.processor.processor_fingerprint = "sha256:" + "ab" * 32
        deferred = self.processor.deferred_image
        ids = _fake_encode(prompt)
        expected = run(None, ids)
        for input_text in (ids, prompt):
            with self.subTest(input_text=type(input_text).__name__):
                actual = run(deferred, input_text)
                self.assertEqual(actual.input_ids, expected.input_ids)
                self.assertEqual(
                    [item.offsets for item in actual.mm_items],
                    [item.offsets for item in expected.mm_items],
                )
                self.assertEqual(
                    [item.identity for item in actual.mm_items],
                    [item.identity for item in expected.mm_items],
                )
                self.assertTrue(
                    all(
                        DEFERRED_IMAGE_KEY in item.model_specific_data
                        for item in actual.mm_items
                    )
                )
                self.assertTrue(
                    torch.equal(
                        self.finish(actual.mm_items, "cpu"),
                        torch.cat([item.feature for item in expected.mm_items]),
                    )
                )

    def test_unsupported_image_options_keep_processor_path(self):
        with patch(
            "sglang.srt.multimodal.processors.minimax_m3_vl.is_cuda",
            return_value=True,
        ):
            for image_config, supported in (
                ({"max_pixels": 200704}, True),
                ({"image_mean": [0.5, 0.5, 0.5]}, False),
            ):
                with self.subTest(image_config=image_config):
                    self.processor.__dict__.pop("deferred_image", None)
                    self.processor.image_config = image_config
                    self.assertEqual(
                        self.processor.deferred_image is not None, supported
                    )


class TestDecodeImageSizes(CustomTestCase):
    def setUp(self):
        self.processor = _fake_processor(MiniMaxM31ImageProcessor(max_size=(672, 672)))
        self.addCleanup(self.processor.io_executor.shutdown)

    def test_header_sizes_match_pixel_path(self):
        sizes = [(3000, 2000), (50, 1000), (640, 480), (100, 100), (1500, 1000)]
        image_data = [
            ImageData(url=_png_data_url(*size), max_long_side_pixel=limit)
            for size, limit in zip(sizes, [None, None, 252, None, 1008])
        ]
        ids = [101]
        for _ in sizes:
            ids += [10, 102]
        for max_pixels in (None, 451584):
            with self.subTest(max_pixels=max_pixels):
                self.processor.image_config = (
                    {} if max_pixels is None else {"max_pixels": max_pixels}
                )
                output = asyncio.run(
                    self.processor._process_image_sizes(image_data, list(ids))
                )
                resized = resize_images_for_request(
                    [Image.new("RGB", size) for size in sizes],
                    image_data,
                    image_processor=self.processor._processor.image_processor,
                    max_pixels=max_pixels,
                )
                expected = self.processor.process_mm_data(
                    None, images=resized, input_ids=list(ids), images_resized=True
                )
                expected_ids = expected["input_ids"][0]
                self.assertEqual(output.input_ids, expected_ids.tolist())
                self.assertEqual(
                    [item.image_grid_thw[0].tolist() for item in output.mm_items],
                    expected["image_grid_thw"].tolist(),
                )
                self.assertEqual(
                    [item.offsets[0] for item in output.mm_items],
                    BaseMultimodalProcessor.get_mm_items_offset(expected_ids, 10),
                )

    def test_decode_rejects_malformed_image(self):
        with self.assertRaises(ValueError):
            asyncio.run(
                self.processor._process_image_sizes(
                    [ImageData(url="data:image/png;base64,bm90IGFuIGltYWdl")], [10]
                )
            )

    def test_vectorized_expansion_matches_scan(self):
        rng = np.random.default_rng(0)
        image = [12, 10, 10, 13]
        video = [20, 12, 11, 13, 12, 11, 13]
        for trial in range(200):
            ids = rng.integers(100, 110, rng.integers(0, 40)).tolist()
            for token in rng.choice([10, 11], rng.integers(0, 5)).tolist():
                ids.insert(int(rng.integers(0, len(ids) + 1)), token)
            replacements = {
                10: [image] * int(rng.integers(0, 4)),
                11: [video] * int(rng.integers(0, 3)),
            }
            results = []
            for expand in (expand_media_tokens, _expand_media_tokens_scan):
                try:
                    results.append(expand(ids, replacements, im_start_id=12))
                except ValueError as error:
                    results.append(str(error))
            self.assertEqual(results[0], results[1], msg=f"trial {trial}: {ids}")


class TestVisionTokenBlocks(CustomTestCase):
    def test_blocked_grouped_fn_equals_whole(self):
        """Blocking the ViT MLP and projector must not change the output (up to GEMM rounding)."""
        from sglang.srt.models.minimax_vl_common import _run_in_token_blocks

        torch.manual_seed(0)
        x = torch.randn(1, 1000, 16)
        linear = torch.nn.Linear(16, 8)

        def merge(t):  # folds each run of 4 consecutive tokens, like the patch merger
            return linear(t).unflatten(-2, (-1, 4)).mean(-2)

        # 1000 tokens in blocks of 64 leave an uneven last block of 40.
        torch.testing.assert_close(_run_in_token_blocks(merge, x, 64), merge(x))


if __name__ == "__main__":
    unittest.main()
