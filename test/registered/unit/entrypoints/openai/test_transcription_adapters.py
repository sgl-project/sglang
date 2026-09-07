"""Unit tests for the transcription-adapter registry and the ASR/audio adapters.

Covers the pieces that are easy to regress and cheap to check on CPU:

* ``resolve_adapter`` maps each real HF architecture string to the intended
  adapter class (guards the substring-matching resolver against collisions when
  new keys are added), and an unknown arch falls back to Whisper.
* The decoder-only speech-LM adapters (GLM-ASR, Qwen2-Audio, Granite Speech)
  keep language auto-detection off and each implement ``build_sampling_params``
  (duration-scaled ``max_new_tokens`` above a per-model floor) and
  ``build_verbose_response`` (no segments, per-model default language).
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.transcription_adapters import (
    GlmAsrAdapter,
    GraniteSpeechAdapter,
    Qwen2AudioAdapter,
    WhisperAdapter,
    resolve_adapter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


# Per-second scaling rate every speech-LM adapter uses for max_new_tokens.
_TOKENS_PER_SECOND = 15

# (arch string, adapter class, expected floor max_new_tokens, expected response language)
# Response language is always None: these adapters neither honor a request-side
# hint nor detect+report a language, so they must not claim one.
_SPEECH_LM_CASES = [
    ("GlmAsrForConditionalGeneration", GlmAsrAdapter, 448, None),
    ("Qwen2AudioForConditionalGeneration", Qwen2AudioAdapter, 448, None),
    ("GraniteSpeechForConditionalGeneration", GraniteSpeechAdapter, 448, None),
]


class TestTranscriptionAdapterResolution(CustomTestCase):
    def test_resolves_expected_adapter_per_architecture(self):
        for arch, cls, _, _ in _SPEECH_LM_CASES:
            with self.subTest(arch=arch):
                self.assertIsInstance(resolve_adapter([arch]), cls)

    def test_unknown_architecture_falls_back_to_whisper(self):
        # Whisper is the registered default; a non-ASR arch must not match one
        # of the speech-LM keys via substring.
        self.assertIsInstance(
            resolve_adapter(["SomeUnknownForConditionalGeneration"]), WhisperAdapter
        )
        self.assertIsInstance(
            resolve_adapter(["WhisperForConditionalGeneration"]), WhisperAdapter
        )
        self.assertIsInstance(resolve_adapter([]), WhisperAdapter)


class TestSpeechLMAdapterContract(CustomTestCase):
    def _request(self, temperature=0.0, duration=3.0, language=None):
        return TranscriptionRequest(
            model="m",
            temperature=temperature,
            audio_duration_s=duration,
            language=language,
        )

    def test_language_detection_disabled(self):
        for _, cls, _, _ in _SPEECH_LM_CASES:
            with self.subTest(adapter=cls.__name__):
                self.assertFalse(cls().supports_language_detection)

    def test_sampling_params_short_clip_uses_floor(self):
        # A 3s clip scales to well under any adapter's floor, so max_new_tokens
        # must stay pinned at the per-model floor (guards the max(...) floor).
        req = self._request(temperature=0.2, duration=3.0)
        for _, cls, max_new_tokens, _ in _SPEECH_LM_CASES:
            with self.subTest(adapter=cls.__name__):
                self.assertEqual(
                    cls().build_sampling_params(req),
                    {"temperature": 0.2, "max_new_tokens": max_new_tokens},
                )

    def test_sampling_params_long_clip_scales_with_duration(self):
        # A long clip must lift max_new_tokens above the floor so the transcript
        # isn't silently truncated. Each adapter scales at _TOKENS_PER_SECOND, so
        # 600s -> 9000 tokens, well above every per-model floor.
        req = self._request(temperature=0.0, duration=600.0)
        expected = int(600.0 * _TOKENS_PER_SECOND)
        for _, cls, floor, _ in _SPEECH_LM_CASES:
            with self.subTest(adapter=cls.__name__):
                params = cls().build_sampling_params(req)
                self.assertEqual(params["max_new_tokens"], expected)
                self.assertGreater(params["max_new_tokens"], floor)

    def test_sampling_params_at_floor_scale_boundary(self):
        # Pin the floor<->scale crossover (max_new_tokens = max(floor,
        # int(duration * rate))). The knee is at floor/rate seconds; bracket it
        # by +/-1s. Just below must stay at the floor; just above must switch to
        # the scaled value. A max()->min() swap or a wrong comparison at the
        # knee would flip one of these and is invisible to the deep-floor (3s)
        # and deep-scaled (600s) cases.
        for _, cls, floor, _ in _SPEECH_LM_CASES:
            knee_s = floor / _TOKENS_PER_SECOND
            below_s = knee_s - 1.0
            above_s = knee_s + 1.0
            with self.subTest(adapter=cls.__name__):
                below = cls().build_sampling_params(self._request(duration=below_s))
                self.assertEqual(below["max_new_tokens"], floor)

                above = cls().build_sampling_params(self._request(duration=above_s))
                self.assertEqual(
                    above["max_new_tokens"], int(above_s * _TOKENS_PER_SECOND)
                )
                self.assertGreater(above["max_new_tokens"], floor)

    def test_verbose_response_language_unset_and_has_no_segments(self):
        for _, cls, _, expected_language in _SPEECH_LM_CASES:
            with self.subTest(adapter=cls.__name__):
                resp = cls().build_verbose_response(
                    self._request(duration=3.456), "hello", {}, None, None
                )
                self.assertEqual(resp.language, expected_language)
                self.assertEqual(resp.duration, 3.46)
                self.assertEqual(resp.text, "hello")
                self.assertEqual(resp.segments, [])

    def test_verbose_response_does_not_claim_request_language(self):
        # Even when the client sends a language hint, these adapters neither
        # honor nor detect it, so the response must not claim it (guards against
        # re-introducing a misleading echo).
        for _, cls, _, _ in _SPEECH_LM_CASES:
            with self.subTest(adapter=cls.__name__):
                resp = cls().build_verbose_response(
                    self._request(language="fr"), "bonjour", {}, None, None
                )
                self.assertIsNone(resp.language)

    def test_glm_postprocess_strips_assistant_prefix(self):
        # GLM-ASR wraps the transcript in an assistant preamble + quotes;
        # postprocess_text must strip it (mirrors HF strip_prefix=True) so it
        # doesn't leak into the transcript and inflate WER.
        adapter = GlmAsrAdapter()
        self.assertEqual(
            adapter.postprocess_text(
                'The spoken content of the audio is "hello world".'
            ),
            "hello world",
        )
        self.assertEqual(
            adapter.postprocess_text("The transcription of the audio is 'bonjour'."),
            "bonjour",
        )
        # A raw transcript with no preamble must pass through unchanged.
        self.assertEqual(
            adapter.postprocess_text("just the transcript"), "just the transcript"
        )


if __name__ == "__main__":
    unittest.main()
