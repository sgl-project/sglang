"""Unit tests for ``BaseMultimodalProcessor.render_transcription_prompt``.

The transcription server renders an adapter's ``TranscriptionPrompt`` with the
model's chat template and sends it as the request text. These tests use a stub
template to check part order, the inline-string form, the fallback when a
template drops structured audio, and caching.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.entrypoints.openai.transcription_adapters import TranscriptionPrompt
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

AUDIO = "<|audio|>"


class _Processor(BaseMultimodalProcessor):
    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


def _render_parts(messages, **_):
    """Stand-in chat template for content parts: audio parts become the audio token."""
    out = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            content = "".join(
                AUDIO if part["type"] == "audio" else part["text"] for part in content
            )
        out.append(f"<{message['role']}>{content}</{message['role']}>")
    return "".join(out) + "<assistant>"


def _render_text(messages, **_):
    return "".join(f"[{m['role']}]{m['content']}" for m in messages) + "[assistant]"


def _make_processor(parts_template=_render_parts, audio_token=AUDIO):
    processor = object.__new__(_Processor)
    hf_processor = MagicMock()
    hf_processor.apply_chat_template.side_effect = parts_template
    hf_processor.tokenizer.apply_chat_template.side_effect = _render_text
    processor._processor = hf_processor
    processor._tokenizer = hf_processor.tokenizer
    processor.mm_tokens = SimpleNamespace(audio_token=audio_token)
    return processor


class TestRenderTranscriptionPrompt(CustomTestCase):
    def test_parts_follow_audio_order(self):
        processor = _make_processor()
        first = TranscriptionPrompt(
            "Transcribe.", audio_first=True, content_format="parts"
        )
        last = TranscriptionPrompt(
            "Transcribe.", audio_first=False, content_format="parts"
        )
        self.assertEqual(
            processor.render_transcription_prompt(first),
            f"<user>{AUDIO}Transcribe.</user><assistant>",
        )
        self.assertEqual(
            processor.render_transcription_prompt(last),
            f"<user>Transcribe.{AUDIO}</user><assistant>",
        )

    def test_text_inlines_the_audio_token_through_the_tokenizer_template(self):
        processor = _make_processor()
        prompt = TranscriptionPrompt("Transcribe.", system="Be terse.")
        self.assertEqual(
            processor.render_transcription_prompt(prompt),
            f"[system]Be terse.[user]{AUDIO}Transcribe.[assistant]",
        )
        processor._processor.apply_chat_template.assert_not_called()

    def test_template_that_drops_audio_falls_back_to_inline_text(self):
        drops_audio = lambda messages, **kw: "<user>Transcribe.</user>"  # noqa: E731
        processor = _make_processor(parts_template=drops_audio)
        prompt = TranscriptionPrompt("Transcribe.", content_format="parts")
        with patch("sglang.srt.multimodal.processors.base_processor.logger") as logger:
            rendered = processor.render_transcription_prompt(prompt)
        logger.warning.assert_called_once()
        self.assertEqual(rendered, f"[user]{AUDIO}Transcribe.[assistant]")

    def test_renders_each_prompt_once(self):
        processor = _make_processor()
        prompt = TranscriptionPrompt("Transcribe.", content_format="parts")
        for _ in range(3):
            processor.render_transcription_prompt(prompt)
        self.assertEqual(processor._processor.apply_chat_template.call_count, 1)

    def test_no_audio_token_is_an_error(self):
        processor = _make_processor(audio_token=None)
        with self.assertRaises(ValueError):
            processor.render_transcription_prompt(TranscriptionPrompt("Transcribe."))


if __name__ == "__main__":
    unittest.main()
