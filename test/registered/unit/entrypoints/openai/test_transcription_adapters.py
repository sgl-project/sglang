"""Unit tests for the transcription-adapter registry and the ASR/audio adapters.

Covers the pieces that are easy to regress and cheap to check on CPU:

* ``resolve_adapter`` maps each real HF architecture string to the intended
  adapter class (guards the substring-matching resolver against collisions when
  new keys are added), and an unknown arch falls back to ``SpeechLMAdapter``.
* The decoder-only speech-LM adapters (GLM-ASR, Qwen2-Audio, Granite Speech)
  keep language auto-detection off and each implement ``build_sampling_params``
  (duration-scaled ``max_new_tokens`` above a per-model floor) and
  ``build_verbose_response`` (no segments, per-model default language).
* ``build_prompt``: every speech-LM adapter yields a fully formatted ASR prompt
  for any request language, the migrated adapters keep the exact instruction
  their processors used before, and decoder-prefix models (Whisper, Qwen3-ASR,
  MiMo-V2-ASR) keep the prompt in the processor.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import TranscriptionRequest
from sglang.srt.entrypoints.openai.transcription_adapters import (
    Gemma4Adapter,
    GlmAsrAdapter,
    GraniteSpeechAdapter,
    MiMoV2ASRAdapter,
    Qwen2AudioAdapter,
    Qwen3ASRAdapter,
    Qwen3OmniAdapter,
    SpeechLMAdapter,
    TranscriptionPrompt,
    WhisperAdapter,
    resolve_adapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.base import _ADAPTER_REGISTRY
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

    def test_resolves_gemma4_and_qwen3_omni(self):
        for arch, cls in (
            # Both the encoder-free unified 12B and the conformer E2B/E4B archs.
            ("Gemma4UnifiedForConditionalGeneration", Gemma4Adapter),
            ("Gemma4ForConditionalGeneration", Gemma4Adapter),
            ("Qwen3OmniMoeForConditionalGeneration", Qwen3OmniAdapter),
        ):
            with self.subTest(arch=arch):
                self.assertIs(type(resolve_adapter([arch])), cls)

    def test_unknown_architecture_falls_back_to_speech_lm_default(self):
        # The Whisper adapter adds sampling keys only the Whisper processor
        # consumes, so it must not be the default; an unmatched speech LM gets
        # the generic prompt instead.
        self.assertIs(
            type(resolve_adapter(["SomeUnknownForConditionalGeneration"])),
            SpeechLMAdapter,
        )
        self.assertIs(type(resolve_adapter([])), SpeechLMAdapter)
        self.assertIsInstance(
            resolve_adapter(["WhisperForConditionalGeneration"]), WhisperAdapter
        )


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


def _speech_lm_adapter_classes():
    return sorted(
        {cls for cls in _ADAPTER_REGISTRY.values() if issubclass(cls, SpeechLMAdapter)},
        key=lambda cls: cls.__name__,
    )


class TestTranscriptionPrompt(CustomTestCase):
    def _request(self, language=None, duration=3.0, temperature=0.0):
        return TranscriptionRequest(
            model="m",
            temperature=temperature,
            audio_duration_s=duration,
            language=language,
        )

    def test_every_speech_lm_adapter_formats_its_prompt(self):
        # Unknown or name-less languages fall back to the no-language prompt, and
        # no template placeholder may survive into the model input.
        for cls in _speech_lm_adapter_classes():
            for language in (None, "en", "English", "zh", "yue", "not-a-language"):
                with self.subTest(adapter=cls.__name__, language=language):
                    prompt = cls().build_prompt(self._request(language=language))
                    self.assertIsInstance(prompt, TranscriptionPrompt)
                    self.assertTrue(prompt.instruction.strip())
                    self.assertNotIn("{", prompt.instruction)
                    self.assertIn(prompt.content_format, ("parts", "text"))

    def test_decoder_prefix_models_keep_the_prompt_in_the_processor(self):
        for cls in (WhisperAdapter, Qwen3ASRAdapter, MiMoV2ASRAdapter):
            with self.subTest(adapter=cls.__name__):
                self.assertIsNone(cls().build_prompt(self._request(language="en")))

    def test_migrated_adapters_keep_their_processor_prompts(self):
        # These instructions were rendered by the processors before; any change
        # alters the model input, so they must stay byte-identical.
        cases = [
            (
                Qwen2AudioAdapter,
                "Transcribe the audio. Output only the exact transcription, "
                "with no preamble, prefix, commentary, or quotation marks.",
                "parts",
            ),
            (GlmAsrAdapter, "Transcribe the audio.", "parts"),
            (
                GraniteSpeechAdapter,
                "Transcribe the speech into written text. Output only the exact "
                "transcription, with no preamble, commentary, quotation marks, or "
                "explanation.",
                "text",
            ),
        ]
        for cls, instruction, content_format in cases:
            with self.subTest(adapter=cls.__name__):
                prompt = cls().build_prompt(self._request(language="en"))
                self.assertEqual(
                    prompt,
                    TranscriptionPrompt(
                        instruction=instruction,
                        audio_first=True,
                        system=None,
                        content_format=content_format,
                    ),
                )

    def test_qwen3_omni_names_the_language(self):
        adapter = Qwen3OmniAdapter()
        instruction = lambda lang: (
            adapter.build_prompt(  # noqa: E731
                self._request(language=lang)
            ).instruction
        )
        self.assertEqual(instruction("en"), "Transcribe the English audio into text.")
        self.assertEqual(
            instruction("English"), "Transcribe the English audio into text."
        )
        self.assertEqual(instruction("zh"), "请将这段中文语音转换为纯文本。")
        self.assertEqual(instruction(None), "Transcribe the audio into text.")
        self.assertEqual(instruction("yue"), "Transcribe the audio into text.")
        resp = adapter.build_verbose_response(
            self._request(language="fr"), "bonjour", {}, None, None
        )
        self.assertEqual(resp.language, "fr")

    def test_gemma4_prompt_and_sampling(self):
        adapter = Gemma4Adapter()
        prompt = adapter.build_prompt(self._request())
        self.assertFalse(prompt.audio_first)
        self.assertEqual(prompt.content_format, "parts")
        self.assertTrue(
            prompt.instruction.startswith("Transcribe the following speech")
        )
        # Token 100 is <|channel> in every Gemma 4 tokenizer; opening a thought
        # channel is how the empty-block loop starts.
        self.assertEqual(
            adapter.build_sampling_params(self._request(temperature=0.2)),
            {"temperature": 0.2, "max_new_tokens": 448, "logit_bias": {"100": -100.0}},
        )
        self.assertEqual(adapter.max_audio_clip_s, 30.0)
        self.assertIsNone(SpeechLMAdapter().max_audio_clip_s)


if __name__ == "__main__":
    unittest.main()
