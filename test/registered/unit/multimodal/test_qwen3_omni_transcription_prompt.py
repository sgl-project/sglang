"""Unit tests for the default ASR prompt the Qwen3-Omni processor builds for
``/v1/audio/transcriptions`` requests, which arrive with empty text, and for
the helper that pops the adapter-forwarded language out of ``sampling_params``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _chat_processor():
    """A processor stub whose chat template returns the conversation itself."""
    processor = MagicMock()
    processor.apply_chat_template.side_effect = lambda conversation, **_: conversation
    return processor


def _user_text(conversation):
    (message,) = conversation
    return [part for part in message["content"] if part["type"] == "text"][0]["text"]


class TestPopTranscriptionLanguage(CustomTestCase):
    def test_pops_language_so_sampling_params_stays_valid(self):
        request = SimpleNamespace(
            sampling_params={"language": "en", "max_new_tokens": 8}
        )
        self.assertEqual(
            BaseMultimodalProcessor._pop_transcription_language(request), "en"
        )
        self.assertEqual(request.sampling_params, {"max_new_tokens": 8})

    def test_missing_language_or_params(self):
        pop = BaseMultimodalProcessor._pop_transcription_language
        self.assertIsNone(pop(SimpleNamespace(sampling_params={})))
        self.assertIsNone(pop(SimpleNamespace(sampling_params=None)))
        self.assertIsNone(pop(None))


class TestQwen3OmniTranscriptionPrompt(CustomTestCase):
    def setUp(self):
        self.processor = object.__new__(QwenVLImageProcessor)
        self.processor._processor = _chat_processor()

    def _prompt(self, input_text, language):
        request = SimpleNamespace(sampling_params={"language": language})
        prompt = self.processor._build_omni_transcription_prompt(input_text, request)
        self.assertNotIn("language", request.sampling_params)
        return prompt

    def test_empty_text_uses_model_card_prompt(self):
        conversation = self._prompt([], "en")
        self.assertEqual(
            _user_text(conversation), "Transcribe the English audio into text."
        )
        # Audio first, then the instruction; no system turn.
        self.assertEqual(
            [part["type"] for part in conversation[0]["content"]], ["audio", "text"]
        )
        self.assertEqual(
            _user_text(self._prompt("", "zh")), "请将这段中文语音转换为纯文本。"
        )
        self.assertEqual(
            _user_text(self._prompt("", None)), "Transcribe the audio into text."
        )

    def test_caller_text_is_kept(self):
        self.assertEqual(self._prompt("Describe the clip.", "en"), "Describe the clip.")
        # Chat requests may arrive as token ids; load_mm_data handles those.
        self.assertEqual(self._prompt([151644, 872], None), [151644, 872])


if __name__ == "__main__":
    unittest.main()
