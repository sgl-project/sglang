"""Shared utilities and test mixin for ASR (transcription) tests."""

import unittest
from dataclasses import dataclass
from typing import Any, List, Optional

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    download_audio_bytes,
    parse_sse_stream,
    popen_launch_server,
)

# Sentinel for _post_transcribe(language=...). ``_USE_CASE_LANG`` means
# fall back to ``audio_case.language``; an explicit ``None`` *omits* the
# field (e.g. to exercise Whisper's fused auto-detect path).
_USE_CASE_LANG: Any = object()


@dataclass
class AudioTestCase:
    """Describes one audio sample and its expected keywords."""

    url: str
    keywords: List[str]
    min_keyword_matches: int = 1
    local_cache_path: Optional[str] = None
    language: Optional[str] = None
    mime_type: str = "audio/wav"
    filename: str = "audio.wav"


class ASRTestBase(CustomTestCase):
    """Base class for /v1/audio/transcriptions tests.

    Subclasses must set:
        model: str                  - HuggingFace model name
        served_model_name: str      - --served-model-name value
        audio_cases: list[AudioTestCase] - at least one audio test case
        extra_args: list[str]       - additional server launch args (default [])

    Optional overrides:
        streaming_exact_match: bool - True if streaming output should exactly
                                      match non-streaming (default True).
                                      Set False for chunk-based streaming.
    """

    # Prevent pytest from collecting this abstract base class directly.
    # __init_subclass__ restores collection for concrete subclasses.
    __test__ = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__test__ = True

    model: str = ""
    served_model_name: str = ""
    audio_cases: List[AudioTestCase] = []
    extra_args: list = []
    streaming_exact_match: bool = True

    @classmethod
    def setUpClass(cls):
        if cls is ASRTestBase:
            raise unittest.SkipTest("ASRTestBase is an abstract base class")

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--served-model-name",
            cls.served_model_name,
        ] + list(cls.extra_args)
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.client = openai.Client(api_key="sk-test", base_url=cls.base_url + "/v1")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # ---- helpers ----

    def _get_audio_file(self, audio_case: AudioTestCase):
        """Download audio and return a file tuple for the openai SDK."""
        audio_bytes = download_audio_bytes(audio_case.url, audio_case.local_cache_path)
        return (audio_case.filename, audio_bytes, audio_case.mime_type)

    def _transcribe(self, audio_case: AudioTestCase) -> str:
        """Non-streaming transcription via openai SDK. Returns text."""
        audio_file = self._get_audio_file(audio_case)
        kwargs = {"model": self.served_model_name, "file": audio_file}
        if audio_case.language:
            kwargs["language"] = audio_case.language
        result = self.client.audio.transcriptions.create(**kwargs)
        # SDK returns str when response_format is not set
        if isinstance(result, str):
            return result
        return result.text

    def _post_transcribe(
        self,
        audio_case: AudioTestCase,
        *,
        stream: bool = False,
        language: Any = _USE_CASE_LANG,
        response_format: Optional[str] = None,
        timestamp_granularities: Optional[List[str]] = None,
        timeout: float = 180,
    ) -> requests.Response:
        """Raw POST to ``/v1/audio/transcriptions``.

        Centralises the request shape (file upload + form data) so subclasses
        and helpers don't redo it. Use this when you need access to the raw
        JSON body (``verbose_json``, segments, etc.) or to omit ``language``
        entirely — both of which the openai SDK path can't easily express.

        ``language`` semantics:

        * ``_USE_CASE_LANG`` (default) — use ``audio_case.language``.
        * ``None`` — omit the field (e.g. for fused auto-detect tests).
        * explicit string — override.
        """
        audio_bytes = download_audio_bytes(audio_case.url, audio_case.local_cache_path)
        if language is _USE_CASE_LANG:
            language = audio_case.language
        data = {"model": self.served_model_name}
        if stream:
            data["stream"] = "true"
        if language is not None:
            data["language"] = language
        if response_format is not None:
            data["response_format"] = response_format
        if timestamp_granularities is not None:
            data["timestamp_granularities[]"] = timestamp_granularities
        response = requests.post(
            self.base_url + "/v1/audio/transcriptions",
            files={"file": (audio_case.filename, audio_bytes, audio_case.mime_type)},
            data=data,
            timeout=timeout,
            stream=stream,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def _transcribe_json(self, audio_case: AudioTestCase, **kwargs: Any) -> dict:
        """Non-streaming raw POST returning the parsed JSON body.

        ``**kwargs`` forwards to :py:meth:`_post_transcribe`.
        """
        return self._post_transcribe(audio_case, stream=False, **kwargs).json()

    def _transcribe_stream(self, audio_case: AudioTestCase, **kwargs: Any):
        """Streaming transcription, returns ``(events, full_text)``.

        Uses raw requests because sglang's streaming format
        (``choices[0].delta.content``) differs from the OpenAI SDK's
        TranscriptionTextDeltaEvent shape. ``**kwargs`` forwards to
        :py:meth:`_post_transcribe`.
        """
        response = self._post_transcribe(audio_case, stream=True, **kwargs)
        return parse_sse_stream(response)

    @staticmethod
    def _iter_deltas(events: List[dict]) -> List[str]:
        """Extract non-empty ``delta.content`` strings from SSE events."""
        out: List[str] = []
        for event in events:
            delta = event.get("choices", [{}])[0].get("delta") or {}
            content = delta.get("content")
            if content:
                out.append(content)
        return out

    def _assert_keywords(self, text, audio_case: AudioTestCase, label=""):
        """Assert that text contains enough expected keywords."""
        # For Chinese keywords, match directly; for others, lowercase
        has_cjk = any("\u4e00" <= c <= "\u9fff" for c in audio_case.keywords[0])
        check_text = text if has_cjk else text.lower()
        matches = [kw for kw in audio_case.keywords if kw in check_text]
        self.assertGreaterEqual(
            len(matches),
            audio_case.min_keyword_matches,
            f"{label}Expected at least {audio_case.min_keyword_matches} of "
            f"{audio_case.keywords}, found {matches}. Full text: {text}",
        )

    # ---- shared test methods ----

    def test_non_streaming(self):
        """Non-streaming transcription returns valid text with expected content."""
        for case in self.audio_cases:
            with self.subTest(url=case.url):
                text = self._transcribe(case)
                self.assertGreater(len(text), 0, "Transcription should not be empty")
                self._assert_keywords(text, case, "[non-streaming] ")

    def test_streaming(self):
        """Streaming transcription returns SSE events with expected content."""
        for case in self.audio_cases:
            with self.subTest(url=case.url):
                events, full_text = self._transcribe_stream(case)
                self.assertGreater(
                    len(events), 0, "Should receive at least one SSE event"
                )
                self.assertGreater(
                    len(full_text), 0, "Assembled text should not be empty"
                )

                last_event = events[-1]
                self.assertEqual(
                    last_event["choices"][0].get("finish_reason"),
                    "stop",
                    "Last event should have finish_reason='stop'",
                )

                self._assert_keywords(full_text, case, "[streaming] ")

    def test_streaming_event_format(self):
        """Verify each SSE event has the expected structure."""
        case = self.audio_cases[0]
        events, _ = self._transcribe_stream(case)
        for i, event in enumerate(events):
            self.assertIn("id", event, f"Event {i} missing 'id'")
            self.assertIn("choices", event, f"Event {i} missing 'choices'")
            self.assertEqual(
                len(event["choices"]), 1, f"Event {i} should have exactly 1 choice"
            )
            self.assertIn(
                "delta", event["choices"][0], f"Event {i} choice missing 'delta'"
            )

    def test_streaming_vs_nonstreaming(self):
        """Compare streaming and non-streaming output."""
        case = self.audio_cases[0]
        non_stream_text = self._transcribe(case).strip()
        _, stream_text = self._transcribe_stream(case)
        stream_text = stream_text.strip()

        self.assertGreater(len(non_stream_text), 0)
        self.assertGreater(len(stream_text), 0)

        if self.streaming_exact_match:
            self.assertEqual(
                non_stream_text,
                stream_text,
                f"Streaming text differs from non-streaming.\n"
                f"  non-stream: {non_stream_text[:200]}\n"
                f"  stream:     {stream_text[:200]}",
            )
        else:
            ns_words = set(non_stream_text.lower().split())
            st_words = set(stream_text.lower().split())
            if ns_words:
                overlap = len(ns_words & st_words) / len(ns_words)
                self.assertGreater(
                    overlap,
                    0.8,
                    f"Word overlap too low ({overlap:.0%}). "
                    f"Non-stream: {non_stream_text[:100]}, "
                    f"Stream: {stream_text[:100]}",
                )
