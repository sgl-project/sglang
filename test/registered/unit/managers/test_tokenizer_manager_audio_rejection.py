"""
Unit tests for early audio rejection in TokenizerManager._tokenize_one_request.

A request carrying audio for a model that declares audio support but has no
audio tower (e.g. a non-audio variant of a multimodal family such as
gemma-4-31B-it) must be rejected with a per-request error at tokenization
time. Without the guard, the audio items reach the scheduler, where
get_audio_feature() raises inside model.forward and kills the scheduler
process.

Covers:
  - audio + audio-declaring model without an audio tower -> ValueError
    naming the served model, raised before validation/dispatch
  - empty audio payloads (None, [], [None], [[None]]) are not rejected
  - audio + audio-capable model proceeds
  - models whose processor does not declare audio (or have no processor)
    keep their existing behavior and are not rejected
  - the --language-model-only message takes precedence over the audio one
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_mm_processor(declares_audio: bool):
    return SimpleNamespace(
        mm_tokens=SimpleNamespace(audio_token="<|audio|>" if declares_audio else None),
        prefer_tokenized_input=False,
    )


def _make_tm(declares_audio: bool = True) -> TokenizerManager:
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.model_config = MagicMock()
    tm.model_config.is_audio_understandable_model = False
    tm.model_config.hf_config.architectures = ["Gemma4ForConditionalGeneration"]
    tm.served_model_name = "test-model"
    tm.max_req_input_len = 8192
    tm.mm_processor = _make_mm_processor(declares_audio)
    return tm


def _make_request(audio_data):
    return GenerateReqInput(input_ids=[1, 2, 3], audio_data=audio_data)


def _run_tokenize(tm, request, language_model_only=False):
    async def drive():
        with patch(
            "sglang.srt.managers.tokenizer_manager.get_disagg",
            return_value=SimpleNamespace(
                language_model_only=language_model_only,
                language_only=False,
                encoder_transfer_backend="zmq_to_scheduler",
            ),
        ):
            return await tm._tokenize_one_request(request)

    return asyncio.run(drive())


def _stub_pipeline_past_guard(tm: TokenizerManager, sentinel: object) -> None:
    """Stub the pipeline past the audio guard so these tests assert on the
    guard boundary without a live runtime context (mm limits and the mm
    processor pipeline need scheduler-side context bags)."""
    tm._validate_mm_limits = Mock()
    tm._normalize_mm_content_hashes = Mock()
    tm._validate_one_request = Mock()
    tm._create_tokenized_object = Mock(return_value=sentinel)
    if tm.mm_processor is not None:
        tm.mm_processor.process_mm_data_async = AsyncMock(return_value=None)


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
                _stub_pipeline_past_guard(tm, sentinel)

                self.assertIs(_run_tokenize(tm, _make_request(payload)), sentinel)

    def test_audio_allowed_for_audio_capable_model(self):
        tm = _make_tm()
        tm.model_config.is_audio_understandable_model = True
        sentinel = object()
        _stub_pipeline_past_guard(tm, sentinel)

        self.assertIs(_run_tokenize(tm, _make_request(["x.wav"])), sentinel)

    def test_audio_not_rejected_when_audio_is_not_declared(self):
        """Processors without an audio token never carry audio to the
        scheduler (the base processor drops it), so such models keep their
        existing behavior: the request still goes through the multimodal
        path and is not rejected. Text-only models without a multimodal
        processor never enter that path at all."""
        for mm_processor, mm_path_entered in (
            (_make_mm_processor(False), True),
            (None, False),
        ):
            with self.subTest(mm_processor=mm_processor):
                tm = _make_tm(declares_audio=False)
                tm.mm_processor = mm_processor
                sentinel = object()
                _stub_pipeline_past_guard(tm, sentinel)

                self.assertIs(_run_tokenize(tm, _make_request(["x.wav"])), sentinel)
                if mm_path_entered:
                    tm._validate_mm_limits.assert_called_once()
                else:
                    tm._validate_mm_limits.assert_not_called()

    def test_language_model_only_message_takes_precedence(self):
        tm = _make_tm()
        request = _make_request(["x.wav"])

        with self.assertRaisesRegex(ValueError, "--language-model-only"):
            _run_tokenize(tm, request, language_model_only=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
