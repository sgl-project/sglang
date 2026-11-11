"""CPU parity of the Inkling token-space strategy with the legacy processor; no weights.

serving   served output == legacy assemble: items, offsets, features, audio_end_id;
          IDs equal once the sentinels are mapped to the placeholders; PNG, JPEG,
          16 kHz mono and 22.05 kHz stereo WAV, so decode and resampling are the legacy ones
          -> test_serving_output_matches_legacy_processor
sources   bytes, a data URL and a path read the same media
          -> test_media_sources_read_the_same_bytes
turns     a later turn adds one audio behind the expanded prefix
          -> test_partial_expansion_adds_the_new_turns_media
towers    a text-only checkpoint rejects media
          -> test_text_only_checkpoint_rejects_media
trainer   the strategy module imports without serving modules
          -> test_strategy_imports_without_serving_modules
"""

import asyncio
import base64
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import msgspec
import numpy as np
import soundfile as sf
import torch
from PIL import Image
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from sglang.srt.configs.inkling import InklingMMConfig
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.inkling import InklingMultimodalProcessor
from sglang.srt.multimodal.token_space_mm_process.models.inkling import (
    AUDIO_PLACEHOLDER,
    IMAGE_PLACEHOLDER,
    InklingTokenSpaceProcessStrategy,
    InklingTokenSpaceServingSpec,
)
from sglang.srt.multimodal.token_space_mm_process.sglang import (
    TokenSpaceSGLangProcessor,
)
from sglang.srt.parser.inkling_tokenizer import (
    AUDIO_END,
    AUDIO_TOKEN_ID,
    CONTENT_AUDIO_INPUT,
    CONTENT_IMAGE,
    IMAGE_TOKEN_ID,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

ARCH = "InklingForConditionalGeneration"
# The checkpoint's ids: the placeholders come from its chat template, the rest are Inkling's
# protocol constants, which the legacy processor uses whatever the tokenizer says.
VOCAB = {
    "[UNK]": 0,
    "hi": 1,
    "there": 2,
    IMAGE_PLACEHOLDER: 200054,
    AUDIO_PLACEHOLDER: 200053,
    AUDIO_END: 200043,
    CONTENT_IMAGE: 200005,
    CONTENT_AUDIO_INPUT: 200020,
}
IMAGE, AUDIO = VOCAB[IMAGE_PLACEHOLDER], VOCAB[AUDIO_PLACEHOLDER]
IMAGE_START, AUDIO_START, AUDIO_STOP = (
    VOCAB[CONTENT_IMAGE],
    VOCAB[CONTENT_AUDIO_INPUT],
    VOCAB[AUDIO_END],
)
TO_SENTINEL = {IMAGE: IMAGE_TOKEN_ID, AUDIO: AUDIO_TOKEN_ID}

InklingTokenSpaceProcessor = TokenSpaceSGLangProcessor.for_strategy_and_serving_spec(
    InklingTokenSpaceProcessStrategy, InklingTokenSpaceServingSpec
)


def _tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]")),
        unk_token="[UNK]",
    )


def _config(*, towers=True):
    width = 16 if towers else None
    return InklingMMConfig(
        architectures=[ARCH],
        vision_config={"decoder_dmodel": width, "patch_size": 8},
        audio_config={
            "decoder_dmodel": width,
            "n_mel_bins": 80,
            "mel_vocab_size": 16,
            "dmel_min_value": -7.0,
            "dmel_max_value": 2.0,
        },
    )


def _image(width, height, seed, format):
    pixels = (np.random.RandomState(seed).rand(height, width, 3) * 255).astype("uint8")
    buffer = io.BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, format=format)
    return buffer.getvalue()


def _wav(samples, seed, *, sample_rate=16000, channels=1):
    buffer = io.BytesIO()
    pcm = np.random.RandomState(seed).uniform(-0.5, 0.5, (samples, channels))
    sf.write(buffer, pcm.astype("float32"), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _sentinel_ids(input_ids):
    return [TO_SENTINEL.get(token_id, token_id) for token_id in input_ids]


class TestInklingTokenSpace(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        self.server_args = ServerArgs(
            model_path="dummy",
            model_impl="sglang",
            mm_process_config={},
            mm_feature_transport="cpu",
            mm_preprocess_cache_size_mb=0,
            disable_fast_image_processor=True,
            mm_processor_worker_num=1,
            enable_token_space_processor=True,
        )
        publish(self.server_args, role="test")
        self.tokenizer = _tokenizer()
        self.images = [_image(16, 16, 1, "PNG"), _image(24, 8, 2, "JPEG")]
        self.audios = [_wav(1600, 3), _wav(5000, 4, sample_rate=22050, channels=2)]

    def _processors(self, config=None):
        config = config or _config()
        # The Python image processor, which the strategy uses; the Rust one is a legacy-only switch.
        with envs.SGLANG_INKLING_RS_MM_PREPROCESS.override(False):
            legacy = InklingMultimodalProcessor(
                config, self.server_args, self.tokenizer, None
            )
        self.addCleanup(legacy.shutdown)
        served = InklingTokenSpaceProcessor(
            config, self.server_args, self.tokenizer, None
        )
        self.addCleanup(served.shutdown)
        return legacy, served

    def _legacy_output(self, legacy, input_ids, *, images=(), audios=()):
        return asyncio.run(
            legacy.process_mm_data_async(
                image_data=list(images),
                audio_data=list(audios),
                input_text=input_ids,
                request_obj=GenerateReqInput(input_ids=input_ids),
            )
        )

    def assert_same_media(self, served, legacy):
        self.assertEqual(_sentinel_ids(served.input_ids), legacy.input_ids)
        self.assertEqual(served.audio_end_id, legacy.audio_end_id)
        self.assertEqual(len(served.mm_items), len(legacy.mm_items))
        for item, reference in zip(served.mm_items, legacy.mm_items):
            self.assertEqual(item.modality, reference.modality)
            self.assertEqual(item.offsets, reference.offsets)
            self.assertEqual(item.feature.dtype, reference.feature.dtype)
            self.assertTrue(torch.equal(item.feature, reference.feature))
            item, reference = (msgspec.structs.asdict(i) for i in (item, reference))
            self.assertEqual(
                item["model_specific_data"], reference["model_specific_data"]
            )

    def test_serving_output_matches_legacy_processor(self):
        """A PNG, a JPEG, a 16 kHz mono and a 22.05 kHz stereo clip, in mixed order."""
        legacy, served = self._processors()
        input_ids = [
            IMAGE_START, IMAGE, 1, AUDIO_START, AUDIO, AUDIO_STOP, 2,
            IMAGE_START, IMAGE, AUDIO_START, AUDIO, AUDIO_STOP,
        ]  # fmt: skip

        served_output = asyncio.run(
            served.process_mm_data_async(
                image_data=self.images, audio_data=self.audios, input_text=input_ids
            )
        )
        legacy_output = self._legacy_output(
            legacy, _sentinel_ids(input_ids), images=self.images, audios=self.audios
        )

        self.assert_same_media(served_output, legacy_output)
        # Items follow the modality order of the lists, one span per media.
        self.assertEqual(
            [item.modality for item in served_output.mm_items],
            [Modality.IMAGE, Modality.IMAGE, Modality.AUDIO, Modality.AUDIO],
        )
        # The placeholder ids differ by design: in vocabulary here, sentinels in legacy.
        self.assertEqual(served_output.im_token_id, IMAGE)
        self.assertEqual(legacy_output.im_token_id, IMAGE_TOKEN_ID)

    def test_media_sources_read_the_same_bytes(self):
        """Bytes, a data URL and a file path give the legacy features."""
        legacy, served = self._processors()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.wav"
            path.write_bytes(self.audios[1])
            data_url = (
                "data:image/png;base64," + base64.b64encode(self.images[0]).decode()
            )
            input_ids = [
                IMAGE_START, IMAGE, AUDIO_START, AUDIO, AUDIO_STOP, AUDIO_START, AUDIO, AUDIO_STOP,
            ]  # fmt: skip
            served_output = asyncio.run(
                served.process_mm_data_async(
                    image_data=[data_url],
                    audio_data=[self.audios[0], str(path)],
                    input_text=input_ids,
                )
            )
        legacy_output = self._legacy_output(
            legacy, _sentinel_ids(input_ids), images=self.images[:1], audios=self.audios
        )
        self.assert_same_media(served_output, legacy_output)

    def test_partial_expansion_adds_the_new_turns_media(self):
        """The second turn resends every audio; only the placeholder after the prefix expands."""
        legacy, served = self._processors()
        first_turn = asyncio.run(
            served.process_mm_data_async(
                audio_data=self.audios[:1],
                input_text=[AUDIO_START, AUDIO, AUDIO_STOP, 1],
            )
        )
        second_turn_ids = first_turn.input_ids + [2, AUDIO_START, AUDIO, AUDIO_STOP]
        second_turn = asyncio.run(
            served.process_mm_data_async(
                audio_data=self.audios,
                input_text=second_turn_ids,
                request_obj=GenerateReqInput(
                    input_ids=second_turn_ids,
                    mm_token_expansion_start_len=len(first_turn.input_ids),
                ),
            )
        )
        legacy_output = self._legacy_output(
            legacy,
            [AUDIO_START, AUDIO_TOKEN_ID, AUDIO_STOP, 1, 2, AUDIO_START, AUDIO_TOKEN_ID, AUDIO_STOP],
            audios=self.audios,
        )  # fmt: skip
        self.assert_same_media(second_turn, legacy_output)

    def test_text_only_checkpoint_rejects_media(self):
        _, served = self._processors(_config(towers=False))
        with self.assertRaisesRegex(ValueError, "no audio tower"):
            asyncio.run(
                served.process_mm_data_async(
                    audio_data=self.audios[:1],
                    input_text=[AUDIO_START, AUDIO, AUDIO_STOP],
                )
            )
        with self.assertRaisesRegex(ValueError, "no vision tower"):
            asyncio.run(
                served.process_mm_data_async(
                    image_data=self.images[:1], input_text=[IMAGE_START, IMAGE]
                )
            )

    def test_strategy_imports_without_serving_modules(self):
        """Trainers import the strategy; the serving spec defers its scheduler import."""
        loaded = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys\n"
                "import sglang.srt.multimodal.token_space_mm_process.models.inkling\n"
                "print(sorted(m for m in sys.modules if m.startswith(("
                "'sglang.srt.managers', 'sglang.srt.models', 'sglang.srt.server_args', "
                "'sglang.srt.multimodal.processors.base_processor'))))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(loaded, "[]")


if __name__ == "__main__":
    unittest.main()
