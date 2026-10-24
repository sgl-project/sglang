"""The multimodal processor call must not reconfigure the shared tokenizer.

``BaseMultimodalProcessor.process_mm_data`` runs the HF processor on the event
loop, and the processor's tokenizer is the same object the OpenAI handlers use
from executor threads. Passing ``padding=True`` for the processor's one text made
transformers call the backend's ``enable_padding``, and the next plain encode
called ``no_padding``. With fastokens>=0.3.3 that write deadlocks against an
encode in flight on another thread, which wedges the HTTP server. Padding one
text never changes its ids, so the call has to leave the backend's padding state
alone while every other modality keeps the kwargs it got before.
"""

import copy
import unittest

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    GlmAsrProcessor,
    PreTrainedTokenizerFast,
    WhisperFeatureExtractor,
)
from transformers.models.gemma4.feature_extraction_gemma4 import (
    Gemma4AudioFeatureExtractor,
)
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
from transformers.models.gemma4.video_processing_gemma4 import Gemma4VideoProcessor

from sglang.srt.configs.qwen3_asr import Qwen3ASRProcessor
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    _single_text_padding_kwargs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_GEMMA4_TOKENS = {
    "image_token": "<image>",
    "boi_token": "<boi>",
    "eoi_token": "<eoi>",
    "audio_token": "<audio>",
    "boa_token": "<boa>",
    "eoa_token": "<eoa>",
}
_QWEN_AUDIO_TOKENS = ["<|audio_bos|>", "<|AUDIO|>", "<|audio_eos|>", "<|audio_pad|>"]
_WORDS = ["describe", "this", "audio"]

IMAGE_PROMPT = "<image> describe this"
AUDIO_PROMPT = "<audio> describe this audio"


def _tokenizer() -> PreTrainedTokenizerFast:
    """A word-level fast tokenizer with Gemma 4's and Qwen audio's special tokens."""
    tokens = ["<pad>", "<unk>", *_GEMMA4_TOKENS.values(), *_QWEN_AUDIO_TOKENS]
    vocab = {token: i for i, token in enumerate(tokens + _WORDS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        extra_special_tokens=_GEMMA4_TOKENS,
    )
    tokenizer.add_tokens(_QWEN_AUDIO_TOKENS, special_tokens=True)
    return tokenizer


def _gemma4_processor() -> Gemma4Processor:
    """Gemma 4's processor pads text by default, so leaving padding out is not enough."""
    return Gemma4Processor(
        feature_extractor=Gemma4AudioFeatureExtractor(),
        image_processor=Gemma4ImageProcessor(),
        tokenizer=_tokenizer(),
        video_processor=Gemma4VideoProcessor(),
    )


def _clip(seconds: float) -> np.ndarray:
    return np.zeros(int(16000 * seconds), dtype=np.float32)


class _StubProcessor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _wrap(hf_processor) -> BaseMultimodalProcessor:
    """A BaseMultimodalProcessor around a real HF processor, without a server."""
    processor = _StubProcessor.__new__(_StubProcessor)
    processor._processor = hf_processor
    processor._tokenizer = hf_processor.tokenizer
    processor._tokenizer_auto_adds_specials = False
    processor.disable_fast_image_processor = True
    processor.mm_feature_transport = "cpu"
    processor.image_config = {}
    processor.video_config = {}
    processor.audio_config = {}
    processor.FEATURE_NAMES = []
    return processor


class TestImageRequest(CustomTestCase):
    def setUp(self):
        self.hf_processor = _gemma4_processor()
        self.processor = _wrap(self.hf_processor)
        self.image = Image.new("RGB", (64, 64))

    def test_leaves_the_shared_tokenizer_unpadded(self):
        self.processor.process_mm_data(IMAGE_PROMPT, images=[self.image])

        self.assertIsNone(self.hf_processor.tokenizer.backend_tokenizer.padding)

    def test_ids_match_the_padded_call(self):
        result = self.processor.process_mm_data(IMAGE_PROMPT, images=[self.image])

        padded = self.hf_processor(
            text=[IMAGE_PROMPT], images=[self.image], padding=True, return_tensors="pt"
        )
        self.assertEqual(result["input_ids"].tolist(), padded["input_ids"].tolist())


class TestAudioRequest(CustomTestCase):
    def setUp(self):
        self.hf_processor = _gemma4_processor()
        self.processor = _wrap(self.hf_processor)

    def test_leaves_the_shared_tokenizer_unpadded(self):
        self.processor.process_mm_data(AUDIO_PROMPT, audios=[_clip(1.0)])

        self.assertIsNone(self.hf_processor.tokenizer.backend_tokenizer.padding)

    def test_features_and_ids_match_the_padded_call(self):
        prompt = "<audio> <audio> describe this audio"
        clips = [_clip(1.0), _clip(0.5)]

        result = self.processor.process_mm_data(prompt, audios=clips)

        padded = self.hf_processor(
            text=[prompt],
            audio=clips,
            padding=True,
            audio_kwargs={"truncation": False},
            return_tensors="pt",
        )
        torch.testing.assert_close(result["input_features"], padded["input_features"])
        self.assertEqual(result["input_ids"].tolist(), padded["input_ids"].tolist())


class TestAudioKwargsRouting(CustomTestCase):
    """Only the text's padding may change; audio keeps what ``padding=True`` gave it."""

    def test_transformers_processor_keeps_its_audio_kwargs(self):
        # GLM-ASR defaults audio padding to "max_length" and hands the merged
        # audio kwargs to its feature extractor, so losing the old flat
        # padding=True would change the extractor's input.
        hf_processor = GlmAsrProcessor(
            feature_extractor=WhisperFeatureExtractor(sampling_rate=16000),
            tokenizer=_tokenizer(),
            audio_token="<|AUDIO|>",
        )
        call_kwargs = {"audio_kwargs": {"truncation": False}}

        before = hf_processor._merge_kwargs(
            hf_processor.valid_processor_kwargs,
            tokenizer_init_kwargs=hf_processor.tokenizer.init_kwargs,
            padding=True,
            **copy.deepcopy(call_kwargs),
        )
        after = hf_processor._merge_kwargs(
            hf_processor.valid_processor_kwargs,
            tokenizer_init_kwargs=hf_processor.tokenizer.init_kwargs,
            **_single_text_padding_kwargs(hf_processor, call_kwargs, has_audio=True),
        )

        self.assertEqual(after["audio_kwargs"], before["audio_kwargs"])
        self.assertIs(after["text_kwargs"].pop("padding"), False)
        before["text_kwargs"].pop("padding")
        self.assertEqual(after["text_kwargs"], before["text_kwargs"])

    def test_sglang_processor_keeps_its_audio_kwargs(self):
        # SGLang's Qwen3ASRProcessor hands audio_kwargs straight to its feature
        # extractor, which pads to Whisper's 30 s window (3000 frames) unless told
        # otherwise. A padding entry added to audio_kwargs would change that.
        hf_processor = Qwen3ASRProcessor(
            feature_extractor=WhisperFeatureExtractor(sampling_rate=16000),
            tokenizer=_tokenizer(),
        )

        result = _wrap(hf_processor).process_mm_data(
            "<|audio_pad|> describe this audio", audios=[_clip(1.0)]
        )

        self.assertEqual(result["input_features"].shape[-1], 3000)


if __name__ == "__main__":
    unittest.main()
