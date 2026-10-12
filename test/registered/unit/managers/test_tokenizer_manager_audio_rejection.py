"""
Unit tests for early audio rejection in TokenizerManager._tokenize_one_request.

A request carrying audio that the model cannot understand (no audio tower,
e.g. a non-audio variant of a multimodal family) must be rejected with a
per-request error at tokenization time. Without the guard, the audio reaches
the scheduler, where get_audio_feature() raises inside model.forward and
kills the scheduler process.

Covers:
  - audio + non-audio model -> ValueError naming the served model
  - rejection happens before validation/dispatch
  - empty audio payloads (None, [], [None], [[None]]) are not rejected
  - audio + audio-capable model proceeds
  - the --language-model-only message takes precedence over the audio one
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tm() -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.model_config = MagicMock()
    tm.model_config.is_audio_understandable_model = False
    tm.model_config.hf_config.architectures = ["Gemma4ForConditionalGeneration"]
    tm.served_model_name = "test-model"
    tm.mm_processor = None
    return tm


def _make_request(audio_data):
    return GenerateReqInput(input_ids=[1, 2, 3], audio_data=audio_data)


def _run_tokenize(tm, request, language_model_only=False):
    async def drive():
        with patch(
            "sglang.srt.managers.tokenizer_manager.get_disagg",
            return_value=SimpleNamespace(language_model_only=language_model_only),
        ):
            return await tm._tokenize_one_request(request)

    return asyncio.run(drive())


class TestAudioRejectionInTokenizeOneRequest(CustomTestCase):
    def test_audio_rejected_with_served_model_name(self):
        tm = _make_tm()
        tm._validate_one_request = Mock()
        request = _make_request(["x.wav"])

        with self.assertRaisesRegex(
            ValueError, "Audio inputs are not supported by test-model"
        ):
            _run_tokenize(tm, request)

        tm._validate_one_request.assert_not_called()

    def test_empty_audio_payload_is_not_rejected(self):
        """Empty audio payloads carry no audio; an is-not-None check would
        wrongly reject them (batch normalization fills audio_data with
        [None] * n for text-only batches)."""
        for payload in (None, [], [None], [[None]]):
            with self.subTest(payload=payload):
                tm = _make_tm()
                sentinel = object()
                tm._validate_one_request = Mock()
                tm._create_tokenized_object = Mock(return_value=sentinel)

                self.assertIs(_run_tokenize(tm, _make_request(payload)), sentinel)

    def test_audio_allowed_for_audio_capable_model(self):
        tm = _make_tm()
        tm.model_config.is_audio_understandable_model = True
        sentinel = object()
        tm._validate_one_request = Mock()
        tm._create_tokenized_object = Mock(return_value=sentinel)

        self.assertIs(_run_tokenize(tm, _make_request(["x.wav"])), sentinel)

    def test_language_model_only_message_takes_precedence(self):
        tm = _make_tm()
        request = _make_request(["x.wav"])

        with self.assertRaisesRegex(ValueError, "--language-model-only"):
            _run_tokenize(tm, request, language_model_only=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
