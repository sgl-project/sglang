"""GLM-V image preprocessing reuse, over a tiny offline processor on CPU."""

import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from tokenizers import Tokenizer, decoders
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    Glm4vConfig,
    Glm4vProcessor,
    Glm4vVideoProcessor,
    Glm5NextConfig,
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
    PreTrainedTokenizerFast,
)
from transformers.models.glm4v.image_processing_glm4v import (
    Glm4vImageProcessor as HfGlm4vImageProcessor,
)

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# On a machine that reports CUDA, importing the processor reaches sgl_kernel.
maybe_stub_sgl_kernel()

from sglang.srt.multimodal.processors.glm4v import (  # noqa: E402
    TEXT_LAYOUT_TOKENIZER_KWARGS,
    Glm4vImageProcessor,
    tokenizer_sets_text_layout,
)
from sglang.srt.runtime_context import publish, reset_context  # noqa: E402
from sglang.srt.server_args import ServerArgs  # noqa: E402

VOCAB = (
    "<unk>",
    "<|begin_of_image|>",
    "<|image|>",
    "<|end_of_image|>",
    "<|begin_of_video|>",
    "<|video|>",
    "<|end_of_video|>",
    "hello",
    "world",
    "<pad>",
)
IMAGE_MARKER = "<|begin_of_image|><|image|><|end_of_image|>"
PROMPT = f"hello {IMAGE_MARKER} world"
OTHER_PROMPT = f"world {IMAGE_MARKER} hello"
FLAVORS = ("glm5_next", "glm4v")


def _encoded(image: Image.Image, fmt: str, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, fmt, **save_kwargs)
    return buffer.getvalue()


def _jpeg(color=(200, 30, 30), size=(84, 56)) -> bytes:
    return _encoded(Image.new("RGB", size, color), "JPEG")


def _translucent_png(size=(84, 56)) -> bytes:
    """An image smart_to_rgb composites over a background instead of dropping alpha."""
    return _encoded(Image.new("RGBA", size, (200, 30, 30, 90)), "PNG")


def _oriented_jpeg(orientation: int, size=(84, 56)) -> bytes:
    """One pixel buffer under an EXIF orientation that swaps its sides."""
    gradient = Image.new("RGB", size)
    gradient.putdata(
        [
            (x * 3 % 256, y * 5 % 256, (x + y) % 256)
            for y in range(size[1])
            for x in range(size[0])
        ]
    )
    exif = Image.Exif()
    exif[0x0112] = orientation
    return _encoded(gradient, "JPEG", exif=exif)


def _rotated_jpeg(size=(84, 56)) -> bytes:
    """An image whose EXIF orientation swaps its sides, and so its encoder grid."""
    return _oriented_jpeg(6, size)


def _oriented_image(orientation: int, size=(84, 56)) -> Image.Image:
    """The same buffer as a decoded PIL image, the shape the Python API accepts."""
    return Image.open(io.BytesIO(_oriented_jpeg(orientation, size)))


class _AcceleratorTensor(torch.Tensor):
    """Stands in for an accelerator processor output on a CPU runner.

    It reports a CUDA device so the device-dependent branches run where no
    accelerator is available. Only ``cpu()`` sheds the claim, which is what
    the processor must call before an artifact reaches the cache.
    """

    @staticmethod
    def __new__(cls, data: torch.Tensor) -> "_AcceleratorTensor":
        return torch.Tensor._make_subclass(cls, data)

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", 0)

    def cpu(self) -> torch.Tensor:
        return self.detach().as_subclass(torch.Tensor).to("cpu")


def _tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            {token: index for index, token in enumerate(VOCAB)}, unk_token=VOCAB[0]
        )
    )
    backend.pre_tokenizer, backend.decoder = WhitespaceSplit(), decoders.Fuse()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token=VOCAB[0],
        pad_token="<pad>",
        additional_special_tokens=list(VOCAB[1:7]),
    )


def _hf_processor(flavor: str):
    tokenizer = _tokenizer()
    if flavor == "glm5_next":
        return Glm5NextProcessor(
            image_processor=Glm5NextImageProcessor(),
            video_processor=Glm5NextVideoProcessor(),
            tokenizer=tokenizer,
        )
    return Glm4vProcessor(
        image_processor=HfGlm4vImageProcessor(),
        video_processor=Glm4vVideoProcessor(),
        tokenizer=tokenizer,
    )


def _hf_config(flavor: str):
    config_cls = Glm5NextConfig if flavor == "glm5_next" else Glm4vConfig
    name = "Glm5Next" if flavor == "glm5_next" else "Glm4v"
    return config_cls(
        architectures=[f"{name}ForConditionalGeneration"],
        image_token_id=VOCAB.index("<|image|>"),
        video_token_id=VOCAB.index("<|video|>"),
        image_start_token_id=VOCAB.index("<|begin_of_image|>"),
        image_end_token_id=VOCAB.index("<|end_of_image|>"),
        video_start_token_id=VOCAB.index("<|begin_of_video|>"),
        video_end_token_id=VOCAB.index("<|end_of_video|>"),
        vision_config={"spatial_merge_size": 2},
    )


def _request(**overrides) -> SimpleNamespace:
    fields = {"video_data": None, "audio_data": None, "mm_content_hashes": None}
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _Glm4vArtifactTestCase(CustomTestCase):
    def processor(self, flavor: str, cache_size_mb: int) -> Glm4vImageProcessor:
        server_args = ServerArgs(
            model_path="dummy",
            # "auto" sends the fast image processor to cuda:<base_gpu_id>, which
            # the CPU runner cannot do, and model_impl "auto" would resolve the
            # config this fixture builds by hand.
            model_impl="sglang",
            image_processor_backend="pil",
            mm_feature_transport="cpu",
            mm_process_config={},
            mm_io_worker_num=1,
            mm_processor_worker_num=1,
            tokenizer_worker_num=1,
            mm_preprocess_cache_size_mb=cache_size_mb,
        )
        publish(server_args, role="tokenizer")
        self.addCleanup(reset_context)
        processor = Glm4vImageProcessor(
            _hf_config(flavor),
            server_args,
            _hf_processor(flavor),
            None,
            skip_mm_pool=True,
        )
        self.addCleanup(processor.shutdown)
        return processor

    @staticmethod
    def process(processor, image_data, input_text=PROMPT, **request_fields):
        return asyncio.run(
            processor.process_mm_data_async(
                image_data=image_data,
                input_text=input_text,
                request_obj=_request(**request_fields),
            )
        )

    def assert_same_request(self, expected, actual):
        """Every value the scheduler and the encoder read from the output."""
        self.assertEqual(list(expected.input_ids), list(actual.input_ids))
        self.assertEqual(len(expected.mm_items), len(actual.mm_items))
        self.assertEqual(expected.im_token_id, actual.im_token_id)
        self.assertEqual(expected.video_token_id, actual.video_token_id)
        self.assertTrue(torch.equal(expected.mrope_positions, actual.mrope_positions))
        self.assertEqual(
            int(expected.mrope_position_delta), int(actual.mrope_position_delta)
        )
        for expected_item, actual_item in zip(expected.mm_items, actual.mm_items):
            self.assertEqual(expected_item.offsets, actual_item.offsets)
            self.assertTrue(torch.equal(expected_item.feature, actual_item.feature))
            self.assertTrue(
                torch.equal(
                    expected_item.model_specific_data["image_grid_thw"],
                    actual_item.model_specific_data["image_grid_thw"],
                )
            )


class TestGlm4vImageArtifacts(_Glm4vArtifactTestCase):
    def test_cached_image_composes_the_full_path_request(self):
        """A reused image is preprocessed once and still builds the same request.

        The artifact path rebuilds prompt ids, offsets and M-RoPE itself instead
        of reading them from the HF processor, and decodes from a content
        snapshot rather than through the loader, so each of those can drift from
        the full path on its own.
        """
        images = {
            "jpeg": _jpeg(),
            "translucent png": _translucent_png(),
            "exif orientation": _rotated_jpeg(),
        }
        for flavor in FLAVORS:
            for name, image in images.items():
                with self.subTest(flavor=flavor, image=name):
                    uncached = self.process(
                        self.processor(flavor, cache_size_mb=0), [image], PROMPT
                    )
                    processor = self.processor(flavor, cache_size_mb=64)
                    with patch.object(
                        processor,
                        "prepare_artifact_batch",
                        wraps=processor.prepare_artifact_batch,
                    ) as prepare:
                        cached = self.process(processor, [image], PROMPT)
                        reused = self.process(processor, [image], OTHER_PROMPT)
                    self.assert_same_request(uncached, cached)
                    self.assertEqual(prepare.call_count, 1)
                    # The second prompt expands elsewhere but reuses the image.
                    self.assertNotEqual(list(cached.input_ids), list(reused.input_ids))
                    self.assertEqual(
                        cached.mm_items[0].identity, reused.mm_items[0].identity
                    )

    def test_images_of_one_request_keep_their_own_grids_and_offsets(self):
        """Per-image artifacts must not blur together within one request.

        The processor returns one flat patch tensor for the batch, so splitting
        it by the wrong grid, or reusing one artifact for a repeated image
        without its own offsets, stays invisible for a single image.
        """
        wide, tall = _jpeg(), _jpeg(color=(30, 30, 200), size=(112, 70))
        prompt = f"{IMAGE_MARKER} hello {IMAGE_MARKER}"
        for flavor in FLAVORS:
            for name, image_data in (
                ("different images", [wide, tall]),
                ("one image twice", [wide, wide]),
            ):
                with self.subTest(flavor=flavor, images=name):
                    uncached = self.process(
                        self.processor(flavor, cache_size_mb=0), image_data, prompt
                    )
                    processor = self.processor(flavor, cache_size_mb=64)
                    with patch.object(
                        processor,
                        "prepare_artifact_batch",
                        wraps=processor.prepare_artifact_batch,
                    ) as prepare:
                        cached = self.process(processor, image_data, prompt)
                    self.assert_same_request(uncached, cached)
                    # A repeated image is one cache miss, preprocessed once.
                    self.assertEqual(prepare.call_count, 1)

    def test_only_its_own_pixels_are_charged_to_the_preprocess_cache(self):
        """Each artifact owns its patches, so the budget is not multiplied.

        The processor emits the batch as one tensor and the cache charges an
        entry for its whole storage, so retaining a slice of that tensor charges
        every image of the request for all of them.
        """
        processor = self.processor("glm5_next", cache_size_mb=64)
        images = [_jpeg(), _jpeg(color=(30, 30, 200), size=(112, 70))]
        output = self.process(processor, images, f"{IMAGE_MARKER} hello {IMAGE_MARKER}")
        own_bytes = sum(
            item.feature.numel() * item.feature.element_size()
            for item in output.mm_items
        )
        cache = processor.mm_preprocess_cache
        self.assertEqual(len(cache), len(images))
        self.assertLess(cache.current_size_bytes, own_bytes * 3 // 2)

    def test_exif_orientation_is_part_of_a_decoded_image_identity(self):
        """Two orientations of one pixel buffer are two different images.

        A decoded image is keyed by its pixels, mode, size and palette, none of
        which the EXIF orientation changes, while preprocessing applies that
        orientation. Keying on the raw buffer therefore serves the first
        orientation's features, grid and M-RoPE to the second.
        """
        for flavor in FLAVORS:
            with self.subTest(flavor=flavor):
                uncached = self.processor(flavor, cache_size_mb=0)
                expected = [
                    self.process(uncached, [_oriented_image(orientation)])
                    for orientation in (1, 6)
                ]
                # The two orientations must not agree, or nothing is being read.
                self.assertNotEqual(
                    expected[0]
                    .mm_items[0]
                    .model_specific_data["image_grid_thw"]
                    .tolist(),
                    expected[1]
                    .mm_items[0]
                    .model_specific_data["image_grid_thw"]
                    .tolist(),
                )
                processor = self.processor(flavor, cache_size_mb=64)
                for orientation, want in zip((1, 6), expected):
                    self.assert_same_request(
                        want, self.process(processor, [_oriented_image(orientation)])
                    )

    def test_an_accelerator_output_is_cached_with_a_usable_feature(self):
        """A cached artifact keeps the feature the next request needs.

        cache_value never retains an accelerator tensor, so an artifact built
        from one is stored without its feature and every later request
        preprocesses the image again. Only moving the processor output to the
        host first keeps the entry usable, which a CPU runner cannot show
        without a stand-in for that output.
        """
        processor = self.processor("glm5_next", cache_size_mb=64)
        image_processor = processor._processor.image_processor

        def on_accelerator(**kwargs):
            output = image_processor(**kwargs)
            output["pixel_values"] = _AcceleratorTensor(output["pixel_values"])
            return output

        with patch.object(
            processor._processor, "image_processor", side_effect=on_accelerator
        ) as patched:
            patched.merge_size = image_processor.merge_size
            first = self.process(processor, [_jpeg()])
            second = self.process(processor, [_jpeg()])

        self.assertEqual(patched.call_count, 1)
        self.assertEqual(first.mm_items[0].feature.device.type, "cpu")
        self.assert_same_request(first, second)
        (stored,) = [
            entry.value for entry in processor.mm_preprocess_cache._entries.values()
        ]
        self.assertTrue(stored.cache_value().has_feature)

    def test_requests_the_artifact_path_cannot_compose_keep_the_full_path(self):
        """The full path loads media the artifact path does not read.

        It is the only path that loads video and audio, accepts preprocessed
        inputs, and tolerates a prompt whose markers do not match its images,
        so admitting any of those to the artifact path drops or misplaces media.
        """
        image = _jpeg()
        cases = {
            "video data": ([image], PROMPT, 64, {"video_data": ["clip.mp4"]}),
            "audio data": ([image], PROMPT, 64, {"audio_data": ["speech.wav"]}),
            "video marker": ([image], f"{IMAGE_MARKER} <|video|>", 64, {}),
            "fewer markers than images": ([image, image], PROMPT, 64, {}),
            "preprocessed input": (
                [{"format": "precomputed_embedding", "feature": torch.zeros(4, 8)}],
                PROMPT,
                64,
                {},
            ),
            "cache disabled": ([image], PROMPT, 0, {}),
        }
        for name, (image_data, prompt, cache_size_mb, fields) in cases.items():
            with self.subTest(case=name):
                processor = self.processor("glm5_next", cache_size_mb=cache_size_mb)
                with patch.object(processor, "prepare_artifact_batch") as prepare:
                    try:
                        self.process(processor, image_data, prompt, **fields)
                    except ValueError:
                        # A prompt the full path also rejects still must not
                        # have reached the artifact path.
                        pass
                prepare.assert_not_called()

    def test_inputs_the_artifact_path_cannot_reproduce_keep_the_full_path(self):
        """The full path decides these token sequences on its own.

        It keeps a pre-tokenized prompt verbatim only while
        SGLANG_MM_AVOID_RETOKENIZE is on and its token-count helper agrees with
        the processor, and decodes and tokenizes again otherwise; and a
        tokenizer carrying truncation settings drops text that the artifact
        path, which tokenizes before it expands the image token, would keep.
        Expanding the given ids reproduces neither, so admitting them changes a
        request's tokens.
        """
        image = _jpeg()
        processor = self.processor("glm5_next", cache_size_mb=64)
        self.assertFalse(processor._tokenizer_sets_text_layout)
        with patch.object(
            processor, "prepare_artifact_batch", wraps=processor.prepare_artifact_batch
        ) as prepare:
            self.process(processor, [image], [7, 1, 2, 3, 8])
        prepare.assert_not_called()

        processor = self.processor("glm5_next", cache_size_mb=64)
        processor._tokenizer_sets_text_layout = True
        with patch.object(
            processor, "prepare_artifact_batch", wraps=processor.prepare_artifact_batch
        ) as prepare:
            self.process(processor, [image], PROMPT)
        prepare.assert_not_called()

    def test_only_settings_that_drop_or_pad_text_decline_the_artifact_path(self):
        """The guard covers the settings HF merges, and only those that bite.

        _merge_kwargs takes them from the tokenizer's own init kwargs, so a
        checkpoint that saves one must be recognized without the request naming
        it. Published GLM and Qwen checkpoints carry a bare `max_length`, which
        takes effect only alongside truncation, so treating it as decisive
        would disable the cache for every real model.
        """
        declines = (
            {"truncation": True},
            {"truncation": "longest_first", "max_length": 30},
            {"padding": "max_length", "max_length": 30},
            {"pad_to_multiple_of": 32},
            # The full path passes padding=True itself, so a saved False here
            # does not stop it from padding to a multiple.
            {"padding": False, "pad_to_multiple_of": 32},
        )
        keeps = (
            {"max_length": 30},
            {"stride": 0},
            {"padding": False},
            {"padding": True},
            {"truncation": False},
            {"pad_to_multiple_of": 1},
        )
        for init_kwargs, expected in [(k, True) for k in declines] + [
            (k, False) for k in keeps
        ]:
            with self.subTest(init_kwargs=init_kwargs):
                self.assertEqual(tokenizer_sets_text_layout(init_kwargs), expected)
        self.assertEqual(
            TEXT_LAYOUT_TOKENIZER_KWARGS,
            ("truncation", "padding", "pad_to_multiple_of"),
        )

    def test_a_checkpoints_own_setting_reaches_the_guard(self):
        """The setting is read at construction and keeps the request whole.

        A tokenizer that pads to a multiple only bites because the full path
        passes `padding=True` of its own, so the saved value alone does not
        show up anywhere the request can be inspected later.
        """
        for setting in ({"truncation": True}, {"pad_to_multiple_of": 32}):
            with self.subTest(setting=setting):
                processor = self.processor("glm5_next", cache_size_mb=64)
                self.assertFalse(processor._tokenizer_sets_text_layout)
                processor._tokenizer.init_kwargs.update(setting)
                reloaded = Glm4vImageProcessor(
                    _hf_config("glm5_next"),
                    processor.server_args,
                    processor._processor,
                    None,
                    skip_mm_pool=True,
                )
                self.addCleanup(reloaded.shutdown)
                self.assertTrue(reloaded._tokenizer_sets_text_layout)
                with patch.object(
                    reloaded,
                    "prepare_artifact_batch",
                    wraps=reloaded.prepare_artifact_batch,
                ) as prepare:
                    self.process(reloaded, [_jpeg()], PROMPT)
                prepare.assert_not_called()
                for key in setting:
                    processor._tokenizer.init_kwargs.pop(key, None)


if __name__ == "__main__":
    unittest.main()
