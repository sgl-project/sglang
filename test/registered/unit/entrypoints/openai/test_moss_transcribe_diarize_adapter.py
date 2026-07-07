"""Unit tests for the MOSS transcription adapter."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from unittest.mock import Mock

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionUsage,
)
from sglang.srt.entrypoints.openai.transcription_adapters import resolve_adapter
from sglang.srt.entrypoints.openai.transcription_adapters.moss_transcribe_diarize import (
    MossTranscribeDiarizeAdapter,
)
from sglang.srt.multimodal.processors.moss_transcribe_diarize import (
    DEFAULT_TRANSCRIBE_DIARIZE_PROMPT,
    MossTranscribeDiarizeMultimodalProcessor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestMossTranscribeDiarizeAdapter(CustomTestCase):
    def test_adapter_registered_by_architecture(self):
        adapter = resolve_adapter(["MossTranscribeDiarizeForConditionalGeneration"])
        self.assertIsInstance(adapter, MossTranscribeDiarizeAdapter)

    def test_build_sampling_params_uses_transcription_defaults(self):
        params = MossTranscribeDiarizeAdapter().build_sampling_params(
            TranscriptionRequest(temperature=0.0)
        )
        self.assertEqual(params["temperature"], 0.0)
        self.assertEqual(params["max_new_tokens"], 20480)

    def test_build_sampling_params_uses_moss_max_new_tokens_override(self):
        params = MossTranscribeDiarizeAdapter().build_sampling_params(
            TranscriptionRequest(temperature=0.0, max_new_tokens=32)
        )
        self.assertEqual(params["max_new_tokens"], 32)

    def test_postprocess_strips_chat_special_tokens(self):
        text = "[0.00][S01]你好[1.00]<|im_end|>"
        self.assertEqual(
            MossTranscribeDiarizeAdapter().postprocess_text(text),
            "[0.00][S01]你好[1.00]",
        )

    def test_verbose_response_parses_diarized_timestamp_segments(self):
        text = "[0.41][S01]有一个人前来买瓜[1.65]" "[12.24][S02]生意行吗你们哥俩[13.82]"
        resp = MossTranscribeDiarizeAdapter().build_verbose_response(
            TranscriptionRequest(language=None, audio_duration_s=13.9),
            text,
            ret={},
            tokenizer=Mock(),
            usage=TranscriptionUsage(seconds=14),
        )

        self.assertEqual(resp.language, "auto")
        self.assertEqual(resp.duration, 13.9)
        self.assertEqual(resp.text, text)
        self.assertEqual(len(resp.segments), 2)
        self.assertEqual(resp.segments[0].start, 0.41)
        self.assertEqual(resp.segments[0].end, 1.65)
        self.assertEqual(resp.segments[0].text, "[S01]有一个人前来买瓜")
        self.assertEqual(resp.segments[1].start, 12.24)
        self.assertEqual(resp.segments[1].end, 13.82)
        self.assertEqual(resp.segments[1].text, "[S02]生意行吗你们哥俩")

    def test_verbose_response_parses_segments_with_optional_whitespace(self):
        text = (
            "[0.41] [S01] 有一个人前来买瓜 [1.65]\n"
            "[12.24] [S02] 生意行吗你们哥俩 [13.82]"
        )
        resp = MossTranscribeDiarizeAdapter().build_verbose_response(
            TranscriptionRequest(audio_duration_s=13.9),
            text,
            ret={},
            tokenizer=Mock(),
            usage=TranscriptionUsage(seconds=14),
        )

        self.assertEqual(len(resp.segments), 2)
        self.assertEqual(resp.segments[0].start, 0.41)
        self.assertEqual(resp.segments[0].end, 1.65)
        self.assertEqual(resp.segments[0].text, "[S01]有一个人前来买瓜")
        self.assertEqual(resp.segments[1].start, 12.24)
        self.assertEqual(resp.segments[1].end, 13.82)
        self.assertEqual(resp.segments[1].text, "[S02]生意行吗你们哥俩")

    def test_verbose_response_falls_back_to_full_audio_segment(self):
        resp = MossTranscribeDiarizeAdapter().build_verbose_response(
            TranscriptionRequest(audio_duration_s=1.23),
            "没有时间戳的文本",
            ret={},
            tokenizer=Mock(),
            usage=TranscriptionUsage(seconds=1),
        )
        self.assertEqual(len(resp.segments), 1)
        self.assertEqual(resp.segments[0].id, 0)
        self.assertEqual(resp.segments[0].start, 0.0)
        self.assertEqual(resp.segments[0].end, 1.23)
        self.assertEqual(resp.segments[0].text, "[S01]没有时间戳的文本")

    def test_verbose_response_fallback_preserves_existing_speaker(self):
        resp = MossTranscribeDiarizeAdapter().build_verbose_response(
            TranscriptionRequest(audio_duration_s=1.23),
            "[S02]没有时间戳的文本",
            ret={},
            tokenizer=Mock(),
            usage=TranscriptionUsage(seconds=1),
        )
        self.assertEqual(resp.segments[0].text, "[S02]没有时间戳的文本")

    def test_verbose_response_keeps_empty_segments_for_empty_text(self):
        resp = MossTranscribeDiarizeAdapter().build_verbose_response(
            TranscriptionRequest(audio_duration_s=1.0),
            "   ",
            ret={},
            tokenizer=Mock(),
            usage=TranscriptionUsage(seconds=1),
        )
        self.assertEqual(resp.segments, [])


class TestMossTranscribeDiarizePrompt(CustomTestCase):
    def test_empty_input_uses_default_transcription_prompt(self):
        proc = MossTranscribeDiarizeMultimodalProcessor.__new__(
            MossTranscribeDiarizeMultimodalProcessor
        )
        proc._processor = Mock()
        proc._processor.apply_chat_template = Mock(return_value="rendered prompt")

        self.assertEqual(proc._build_prompt(""), "rendered prompt")
        messages = proc._processor.apply_chat_template.call_args.args[0]
        self.assertEqual(
            messages[0]["content"][1]["text"], DEFAULT_TRANSCRIBE_DIARIZE_PROMPT
        )

    def test_explicit_prompt_is_preserved(self):
        proc = MossTranscribeDiarizeMultimodalProcessor.__new__(
            MossTranscribeDiarizeMultimodalProcessor
        )
        proc._processor = Mock()
        proc._processor.apply_chat_template = Mock(return_value="rendered prompt")

        proc._build_prompt("自定义转写要求")
        messages = proc._processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["content"][1]["text"], "自定义转写要求")


if __name__ == "__main__":
    unittest.main()
