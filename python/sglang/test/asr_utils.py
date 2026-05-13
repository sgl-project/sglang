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


@dataclass
class AudioTestCase:
    """Describes one audio sample and its expected keywords."""

    url: str
    keywords: List[str]
    min_keyword_matches: int = 1
    local_cache_path: Optional[str] = None
    language: Optional[str] = None
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

    def _transcribe(self, audio_case: AudioTestCase) -> str:
        """Non-streaming transcription via openai SDK. Returns text."""
        audio_bytes = download_audio_bytes(audio_case.url, audio_case.local_cache_path)
        kwargs = {
            "model": self.served_model_name,
            "file": (audio_case.filename, audio_bytes),
        }
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
        language: Optional[str] = None,
        response_format: Optional[str] = None,
        timestamp_granularities: Optional[List[str]] = None,
        timeout: float = 180,
    ) -> requests.Response:
        """Raw POST to ``/v1/audio/transcriptions``.

        Centralises the request shape (file upload + form data) so subclasses
        and helpers don't redo it. Use this when you need access to the raw
        JSON body (``verbose_json``, segments, etc.) or to omit ``language``
        entirely — both of which the openai SDK path can't easily express.

        ``language=None`` omits the field; pass a string to set it. The
        ``_transcribe_json`` / ``_transcribe_stream`` wrappers default to
        ``audio_case.language`` when the caller doesn't specify one.
        """
        audio_bytes = download_audio_bytes(audio_case.url, audio_case.local_cache_path)
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
            files={"file": (audio_case.filename, audio_bytes)},
            data=data,
            timeout=timeout,
            stream=stream,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def _transcribe_json(self, audio_case: AudioTestCase, **kwargs: Any) -> dict:
        """Non-streaming raw POST returning the parsed JSON body.

        Defaults ``language`` to ``audio_case.language`` when the caller
        doesn't pass one; pass ``language=None`` explicitly to omit.
        ``**kwargs`` forwards to :py:meth:`_post_transcribe`.
        """
        kwargs.setdefault("language", audio_case.language)
        return self._post_transcribe(audio_case, stream=False, **kwargs).json()

    def _transcribe_stream(self, audio_case: AudioTestCase, **kwargs: Any):
        """Streaming transcription, returns ``(events, full_text)``.

        Uses raw requests because sglang's streaming format
        (``choices[0].delta.content``) differs from the OpenAI SDK's
        TranscriptionTextDeltaEvent shape. Defaults ``language`` to
        ``audio_case.language`` when the caller doesn't pass one;
        ``**kwargs`` forwards to :py:meth:`_post_transcribe`.
        """
        kwargs.setdefault("language", audio_case.language)
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

    def test_multiple_sequential_requests(self):
        """Sequential non-streaming requests on the same audio yield identical text."""
        case = self.audio_cases[0]
        results = []
        for _ in range(3):
            text = self._transcribe(case)
            self.assertGreater(len(text), 0)
            results.append(text)
        for i in range(1, len(results)):
            self.assertEqual(
                results[0],
                results[i],
                f"Transcription {i + 1} differs from first transcription",
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
