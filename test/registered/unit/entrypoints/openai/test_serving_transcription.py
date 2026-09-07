"""Unit tests for OpenAIServingTranscription's streaming fused-autodetect path.

Exercises the streaming handler: buffer deltas until the forced-prefix
sentinel lands, emit the scrubbed user-visible text, and never leak
Whisper special tokens. Covers both streaming modes — cumulative
(``incremental_streaming_output=False``, the default) and incremental
(``incremental_streaming_output=True``).

The tests mock ``TokenizerManager.generate_request`` to yield synthetic
``text`` chunks for each of the happy, abort, and boundary cases.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import io
import json
import unittest
from typing import List
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import soundfile as sf

from sglang.srt.entrypoints.openai.protocol import (
    TranscriptionRequest,
    TranscriptionResponse,
)
from sglang.srt.entrypoints.openai.serving_transcription import (
    OpenAIServingTranscription,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _chunk(text: str, finish: str = None) -> dict:
    """Shape of what TokenizerManager.generate_request yields per step."""
    return {
        "text": text,
        "meta_info": {
            "finish_reason": {"type": finish} if finish else None,
        },
    }


class _MockTokenizerManager:
    """Minimal mock satisfying OpenAIServingTranscription.__init__ and stream loop."""

    def __init__(self, stream_chunks: List[dict]):
        self.model_config = Mock()
        self.model_config.hf_config = Mock()
        self.model_config.hf_config.architectures = ["WhisperForConditionalGeneration"]
        # Not a real ServerArgs, so base class sets allowed_custom_labels=None.
        # Default tests assume cumulative-text streaming (the sglang upstream
        # default); tests for incremental_streaming_output=True override this.
        self.server_args = Mock(
            incremental_streaming_output=False,
            asr_max_concurrent_sessions=32,
        )
        self.tokenizer = Mock()
        self._stream_chunks = stream_chunks

    def generate_request(self, adapted_request, raw_request):
        chunks = self._stream_chunks

        async def gen():
            for c in chunks:
                yield c

        return gen()

    def create_abort_task(self, adapted_request):
        return None


def _deltas_from_sse(sse_lines: List[str]) -> List[str]:
    """Extract ``choices[0].delta.content`` strings from a list of SSE frames."""
    out = []
    for line in sse_lines:
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in obj.get("choices", []):
            content = (choice.get("delta") or {}).get("content")
            if content:
                out.append(content)
    return out


class TestStreamingFusedAutodetect(CustomTestCase):
    """_generate_transcription_stream with _fused_autodetect=True."""

    def _run_stream(
        self, chunks: List[dict], fused: bool = True, ts_variant: bool = False
    ):
        tm = _MockTokenizerManager(chunks)
        serving = OpenAIServingTranscription(tm)

        kwargs = {"model": "whisper", "stream": True}
        if ts_variant:
            kwargs["timestamp_granularities"] = ["segment"]
        request = TranscriptionRequest(**kwargs)
        if fused:
            request._fused_autodetect = True
            request._fused_ts_variant = ts_variant
        adapted = GenerateReqInput(text="", modalities=["audio"])
        raw_request = Mock()

        async def drive():
            frames = []
            async for frame in serving._generate_transcription_stream(
                adapted, request, raw_request
            ):
                frames.append(frame)
            return frames

        loop = get_or_create_event_loop()
        frames = loop.run_until_complete(drive())
        return request, frames

    def test_prefix_stripped_and_language_extracted(self):
        chunks = [
            _chunk("<|en|>"),
            _chunk("<|en|><|transcribe|>"),
            _chunk("<|en|><|transcribe|><|notimestamps|>"),
            _chunk("<|en|><|transcribe|><|notimestamps|> Hello"),
            _chunk("<|en|><|transcribe|><|notimestamps|> Hello world", finish="stop"),
        ]
        request, frames = self._run_stream(chunks)
        deltas = _deltas_from_sse(frames)
        self.assertEqual(deltas, ["Hello", " world"])
        self.assertEqual(request.language, "en")
        # No delta ever starts with the forced prefix or leading whitespace.
        self.assertFalse(any("<|" in d for d in deltas))
        self.assertFalse(deltas[0].startswith(" "))

    def test_non_english_language_extracted(self):
        chunks = [
            _chunk("<|zh|><|transcribe|><|notimestamps|>你好"),
            _chunk("<|zh|><|transcribe|><|notimestamps|>你好世界", finish="stop"),
        ]
        request, frames = self._run_stream(chunks)
        self.assertEqual(request.language, "zh")
        self.assertEqual(_deltas_from_sse(frames), ["你好", "世界"])

    def test_fsm_abort_before_sentinel_emits_error_frame(self):
        # Sentinel never arrives; stream terminates on finish_reason. The
        # handler must surface this as a real SSE error frame so the client
        # can distinguish "detection failed" from "silent audio with zero
        # transcription". language stays unset.
        chunks = [
            _chunk("<|en|>"),
            _chunk("<|en|><|transcribe|>", finish="length"),
        ]
        request, frames = self._run_stream(chunks)
        self.assertEqual(_deltas_from_sse(frames), [])
        error_frames = [f for f in frames if f.startswith("data: ") and '"error"' in f]
        self.assertTrue(
            error_frames, f"expected an SSE error frame, got frames={frames!r}"
        )
        self.assertIn("language auto-detect failed", error_frames[0])
        self.assertIsNone(request.language)

    def test_non_fused_stream_passes_through(self):
        # When _fused_autodetect is False, no buffering or anchoring happens.
        chunks = [
            _chunk("Hello"),
            _chunk("Hello world", finish="stop"),
        ]
        request, frames = self._run_stream(chunks, fused=False)
        self.assertEqual(_deltas_from_sse(frames), ["Hello", " world"])

    def test_streaming_ts_variant_sentinel_at_chunk_boundary(self):
        # The <|0.00|> sentinel can land in its own chunk ahead of any
        # transcription text, and the trailing-space arrives later. The
        # handler must buffer silently until a non-whitespace char shows
        # up (so the first delta doesn't leak a leading space) and then
        # scrub subsequent embedded timestamp tokens.
        chunks = [
            _chunk("<|en|>"),
            _chunk("<|en|><|transcribe|>"),
            _chunk("<|en|><|transcribe|><|0.00|>"),  # sentinel alone
            _chunk("<|en|><|transcribe|><|0.00|> "),  # + whitespace only
            _chunk("<|en|><|transcribe|><|0.00|> Hello"),  # first word
            _chunk("<|en|><|transcribe|><|0.00|> Hello<|5.00|> World"),
            _chunk(
                "<|en|><|transcribe|><|0.00|> Hello<|5.00|> World<|endoftext|>",
                finish="stop",
            ),
        ]
        request, frames = self._run_stream(chunks, ts_variant=True)
        deltas = _deltas_from_sse(frames)
        self.assertEqual(request.language, "en")
        self.assertFalse(any("<|" in d for d in deltas))
        # No delta starts with a leading space (the one Whisper emits
        # between <|0.00|> and "Hello" was consumed by the defer-on-
        # whitespace path).
        self.assertFalse(deltas[0].startswith(" "))
        self.assertEqual("".join(deltas), "Hello World")

    def test_streaming_timestamps_variant_scrubs_embedded_segment_tokens(self):
        # Streaming + timestamp_granularities + language=None uses the fused
        # timestamps variant (<|0.00|> sentinel). Segment-boundary tokens
        # <|5.00|>, <|10.00|> land mid-stream; each delta must have them
        # scrubbed before reaching the client. Auto-detection still works
        # — the SSE stream carries clean text, and callers who want
        # segment timing can use response_format=verbose_json which builds
        # segments from output_ids on a separate path.
        chunks = [
            _chunk("<|en|><|transcribe|><|0.00|> Hello"),
            _chunk("<|en|><|transcribe|><|0.00|> Hello<|5.00|> World"),
            _chunk(
                "<|en|><|transcribe|><|0.00|> Hello<|5.00|> World<|10.00|><|endoftext|>",
                finish="stop",
            ),
        ]
        request, frames = self._run_stream(chunks, ts_variant=True)
        deltas = _deltas_from_sse(frames)
        self.assertEqual(request.language, "en")
        self.assertFalse(any("<|" in d for d in deltas))
        self.assertEqual("".join(deltas), "Hello World")

    def test_trailing_endoftext_scrubbed_from_last_delta(self):
        # skip_special_tokens=False means the detokenizer may emit
        # <|endoftext|> at the tail. The fused streaming path must scrub it
        # per-delta so clients never see special tokens in SSE chunks.
        chunks = [
            _chunk("<|en|><|transcribe|><|notimestamps|> Hello"),
            _chunk(
                "<|en|><|transcribe|><|notimestamps|> Hello world<|endoftext|>",
                finish="stop",
            ),
        ]
        _, frames = self._run_stream(chunks)
        deltas = _deltas_from_sse(frames)
        self.assertEqual(deltas, ["Hello", " world"])
        self.assertFalse(any("<|" in d for d in deltas))


class _MockChunkTokenizerManager:
    """Mock TM scripting one result list per dispatched request, in order."""

    def __init__(self, results_per_request: List):
        self.model_config = Mock()
        self.model_config.hf_config = Mock()
        self.model_config.hf_config.architectures = ["WhisperForConditionalGeneration"]
        self.server_args = Mock(
            incremental_streaming_output=False,
            asr_max_concurrent_sessions=32,
        )
        self.request_logger = Mock(log_requests=False)
        self.tokenizer = Mock()
        self.requests: List[GenerateReqInput] = []
        self.aborted: List[str] = []
        self._results = results_per_request
        self.active_dispatches = 0
        self.max_active_dispatches = 0

    def generate_request(self, adapted_request, raw_request):
        idx = len(self.requests)
        # Mimic the real generate_request assigning a rid (via
        # normalize_batch_and_arguments) so abort-by-rid is exercisable.
        adapted_request.rid = f"rid{idx}"
        self.requests.append(adapted_request)
        results = self._results[idx]

        async def gen():
            self.active_dispatches += 1
            self.max_active_dispatches = max(
                self.max_active_dispatches, self.active_dispatches
            )
            # Give concurrently scheduled generators a chance to overlap at
            # dispatch, then record that this request reached the engine.
            await asyncio.sleep(0)
            self.active_dispatches -= 1
            for r in results:
                # A bare exception entry simulates a chunk request failing.
                if isinstance(r, BaseException):
                    raise r
                yield r

        return gen()

    def abort_request(self, rid: str = "", abort_all: bool = False):
        self.aborted.append(rid)

    def create_abort_task(self, adapted_request):
        return None


def _long_wav_bytes(duration_s: float = 65.0) -> bytes:
    """A 16 kHz tone with silence gaps inside each 30 s stride's split-search
    window, so the energy-aware splitter cuts at ~29.2 s and ~58.5 s."""
    sr = 16000
    t = np.arange(int(duration_s * sr)) / sr
    wav = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    for gap_start, gap_end in ((29.2, 29.5), (58.5, 58.8)):
        wav[int(gap_start * sr) : int(gap_end * sr)] = 0.0
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


class TestLongAudioChunkedNonStreaming(CustomTestCase):
    """Audio longer than Whisper's 30 s window must be split into chunk
    requests and the transcripts stitched in order — without chunking the
    feature extractor silently truncates everything past 30 s."""

    def _create_transcription(self, tm, audio_bytes, language="en", **kwargs):
        serving = OpenAIServingTranscription(tm)
        loop = get_or_create_event_loop()
        return loop.run_until_complete(
            serving.create_transcription(
                audio_data=audio_bytes,
                model="whisper",
                language=language,
                response_format=kwargs.pop("response_format", "json"),
                temperature=0.0,
                stream=False,
                raw_request=Mock(),
                **kwargs,
            )
        )

    def test_long_audio_is_chunked_and_stitched(self):
        texts = [" part one.", " part two.", " part three."]
        tm = _MockChunkTokenizerManager(
            [
                [{"text": t, "meta_info": {"finish_reason": {"type": "stop"}}}]
                for t in texts
            ]
        )
        result = self._create_transcription(tm, _long_wav_bytes(65.0))

        self.assertIsInstance(result, TranscriptionResponse, result)
        # In-order plain concatenation (vLLM-parity stitching).
        self.assertEqual(result.text, " part one. part two. part three.")
        self.assertEqual(result.usage.seconds, 65)
        self.assertEqual(len(tm.requests), 3)
        self.assertEqual(tm.max_active_dispatches, 1)

        # Each chunk request is independent: own audio payload, own
        # sampling_params dict (the multimodal processor pops keys out of
        # it per request), stream=False, audio modality.
        params_ids = {id(req.sampling_params) for req in tm.requests}
        self.assertEqual(len(params_ids), len(tm.requests))
        total_samples = 0
        for req in tm.requests:
            self.assertFalse(req.stream)
            self.assertEqual(req.modalities, ["audio"])
            data, sr = sf.read(io.BytesIO(req.audio_data), dtype="float32")
            self.assertEqual(sr, 16000)
            self.assertLessEqual(len(data), 30 * 16000)
            total_samples += len(data)
        self.assertEqual(total_samples, 65 * 16000)

    def test_chunk_failure_returns_error_and_stops_dispatch(self):
        # When one chunk request fails, create_transcription must return an
        # error response (not a partial transcript), abort the in-flight
        # request, and leave later chunks undispatched.
        tm = _MockChunkTokenizerManager(
            [
                [
                    {
                        "text": " part one.",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
                [ValueError("chunk boom")],
                [
                    {
                        "text": " part three.",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
            ]
        )
        result = self._create_transcription(tm, _long_wav_bytes(65.0))
        # Error response, not a TranscriptionResponse.
        self.assertNotIsInstance(result, TranscriptionResponse)
        # Sequential dispatch means the third chunk never reaches the engine.
        self.assertEqual(len(tm.requests), 2)
        self.assertIn(tm.requests[-1].rid, tm.aborted)
        self.assertEqual(tm.max_active_dispatches, 1)

    def test_split_failure_returns_error_without_dispatch(self):
        tm = _MockChunkTokenizerManager([])
        with patch(
            "sglang.srt.entrypoints.openai.serving_transcription."
            "split_audio_energy_aware",
            side_effect=RuntimeError("decode failed"),
        ):
            result = self._create_transcription(tm, _long_wav_bytes(65.0))

        self.assertEqual(result.status_code, 400)
        self.assertIn("Failed to split audio", json.loads(result.body)["message"])
        self.assertEqual(tm.requests, [])

    def test_short_audio_stays_unchunked(self):
        tm = _MockChunkTokenizerManager(
            [[{"text": " short.", "meta_info": {"finish_reason": {"type": "stop"}}}]]
        )
        result = self._create_transcription(tm, _long_wav_bytes(10.0))
        self.assertIsInstance(result, TranscriptionResponse, result)
        self.assertEqual(result.text, " short.")
        self.assertEqual(len(tm.requests), 1)

    def test_chunked_fused_autodetect_first_chunk_language_wins(self):
        # language=None → fused auto-detect per chunk. Each chunk carries
        # its own forced prefix; the stitched text must strip all of them,
        # and the reported language comes from the first chunk.
        tm = _MockChunkTokenizerManager(
            [
                [
                    {
                        "text": "<|en|><|transcribe|><|notimestamps|> part one.",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
                [
                    {
                        "text": "<|fr|><|transcribe|><|notimestamps|> part two.",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
                [
                    {
                        "text": "<|en|><|transcribe|><|notimestamps|> part three.",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
            ]
        )
        result = self._create_transcription(tm, _long_wav_bytes(65.0), language=None)
        self.assertIsInstance(result, TranscriptionResponse, result)
        self.assertEqual(result.text, "part one. part two. part three.")
        self.assertNotIn("<|", result.text)
        self.assertEqual(len(tm.requests), 3)
        # Every chunk request kept the fused regex constraint.
        for req in tm.requests:
            self.assertIn("regex", req.sampling_params)

    def test_chunked_fused_spaceless_script_not_space_joined(self):
        # zh/ja/th chunk texts carry no boundary whitespace; stitching must
        # not inject an ASCII space the model never emitted.
        tm = _MockChunkTokenizerManager(
            [
                [
                    {
                        "text": "<|zh|><|transcribe|><|notimestamps|>你好",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
                [
                    {
                        "text": "<|zh|><|transcribe|><|notimestamps|>世界",
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
            ]
        )
        result = self._create_transcription(tm, _long_wav_bytes(40.0), language=None)
        self.assertIsInstance(result, TranscriptionResponse, result)
        self.assertEqual(result.text, "你好世界")
        self.assertEqual(len(tm.requests), 2)

    def test_chunked_fused_language_uses_first_nonempty_chunk(self):
        tm = _MockChunkTokenizerManager(
            [
                [
                    {
                        "text": "<|fr|><|transcribe|><|notimestamps|>",
                        "output_ids": [],
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
                [
                    {
                        "text": "<|en|><|transcribe|><|notimestamps|> Hello",
                        "output_ids": [],
                        "meta_info": {"finish_reason": {"type": "stop"}},
                    }
                ],
            ]
        )
        result = self._create_transcription(
            tm,
            _long_wav_bytes(40.0),
            language=None,
            response_format="verbose_json",
        )

        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.language, "en")


class TestLongAudioChunkedStreaming(CustomTestCase):
    """_generate_long_audio_stream: chunks transcribed sequentially, deltas
    emitted in audio order, exactly one finish frame."""

    def _run_stream(self, results_per_request, fused=False, n_chunks=2):
        tm = _MockChunkTokenizerManager(results_per_request)
        serving = OpenAIServingTranscription(tm)
        request = TranscriptionRequest(model="whisper", stream=True)
        if fused:
            request._fused_autodetect = True
            request._fused_ts_variant = False
        request._audio_chunks = [b"chunk%d" % i for i in range(n_chunks)]
        # Streaming has no segment timing, so the offsets are intentionally
        # not set here — only the non-streaming verbose_json path reads them.
        adapted = GenerateReqInput(
            text="", modalities=["audio"], sampling_params={"temperature": 0.0}
        )
        raw_request = Mock()
        raw_request.is_disconnected = AsyncMock(return_value=False)

        async def drive():
            frames = []
            async for frame in serving._generate_long_audio_stream(
                adapted, request, raw_request
            ):
                frames.append(frame)
            return frames

        loop = get_or_create_event_loop()
        return tm, request, loop.run_until_complete(drive())

    @staticmethod
    def _finish_reasons(frames: List[str]) -> List[str]:
        out = []
        for line in frames:
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            obj = json.loads(line[len("data: ") :])
            for choice in obj.get("choices", []):
                if choice.get("finish_reason"):
                    out.append(choice["finish_reason"])
        return out

    def test_chunks_streamed_in_order_with_single_finish(self):
        tm, _, frames = self._run_stream(
            [
                [_chunk(" Hello"), _chunk(" Hello world", finish="stop")],
                [_chunk(" Again"), _chunk(" Again done", finish="stop")],
            ]
        )
        self.assertEqual(
            _deltas_from_sse(frames), [" Hello", " world", " Again", " done"]
        )
        self.assertEqual(self._finish_reasons(frames), ["stop"])
        self.assertEqual(frames[-1], "data: [DONE]\n\n")
        # Both chunk requests were dispatched and streamed.
        self.assertEqual(len(tm.requests), 2)
        self.assertTrue(all(req.stream for req in tm.requests))

    def test_disconnect_between_chunks_stops_and_aborts(self):
        # Client disconnects after the first chunk: the second chunk is
        # never dispatched, and the in-flight request from chunk 0 is
        # already done so nothing is left decoding.
        tm = _MockChunkTokenizerManager(
            [
                [_chunk(" Hello", finish="stop")],
                [_chunk(" world", finish="stop")],
            ]
        )
        serving = OpenAIServingTranscription(tm)
        request = TranscriptionRequest(model="whisper", stream=True)
        request._audio_chunks = [b"chunk0", b"chunk1"]
        adapted = GenerateReqInput(
            text="", modalities=["audio"], sampling_params={"temperature": 0.0}
        )
        raw_request = Mock()
        # Connected for the first chunk, disconnected before the second.
        raw_request.is_disconnected = AsyncMock(side_effect=[False, True])

        async def drive():
            return [
                f
                async for f in serving._generate_long_audio_stream(
                    adapted, request, raw_request
                )
            ]

        frames = get_or_create_event_loop().run_until_complete(drive())
        self.assertEqual(_deltas_from_sse(frames), [" Hello"])
        # Only the first chunk was dispatched.
        self.assertEqual(len(tm.requests), 1)

    def test_abnormal_chunk_finish_reason_not_masked(self):
        # A non-final chunk truncated at the token cap (finish="length")
        # must surface in the single final frame even though later chunks
        # stop cleanly — otherwise silently missing transcript content
        # reads as success.
        _, _, frames = self._run_stream(
            [
                [_chunk(" Hello", finish="length")],
                [_chunk(" world", finish="stop")],
            ]
        )
        self.assertEqual(_deltas_from_sse(frames), [" Hello", " world"])
        self.assertEqual(self._finish_reasons(frames), ["length"])

    def test_fused_chunks_strip_prefixes_and_preserve_boundary_space(self):
        tm, request, frames = self._run_stream(
            [
                [
                    _chunk("<|fr|><|transcribe|>"),
                    _chunk(
                        "<|fr|><|transcribe|><|notimestamps|> Bonjour", finish="stop"
                    ),
                ],
                [
                    _chunk(
                        "<|fr|><|transcribe|><|notimestamps|> le monde",
                        finish="stop",
                    )
                ],
            ],
            fused=True,
        )
        deltas = _deltas_from_sse(frames)
        self.assertFalse(any("<|" in d for d in deltas))
        # No leading space at stream start; the later chunk's own leading
        # space is the seam separator.
        self.assertFalse(deltas[0].startswith(" "))
        self.assertEqual("".join(deltas), "Bonjour le monde")
        self.assertEqual(request.language, "fr")
        self.assertEqual(self._finish_reasons(frames), ["stop"])

    def test_fused_spaceless_script_chunks_not_space_joined(self):
        # zh/ja/th transcripts carry no boundary whitespace; the seam must
        # not inject an ASCII space the model never emitted.
        _, request, frames = self._run_stream(
            [
                [_chunk("<|zh|><|transcribe|><|notimestamps|>你好", finish="stop")],
                [_chunk("<|zh|><|transcribe|><|notimestamps|>世界", finish="stop")],
            ],
            fused=True,
        )
        self.assertEqual("".join(_deltas_from_sse(frames)), "你好世界")
        self.assertEqual(request.language, "zh")

    def test_fused_leading_silence_uses_first_nonempty_chunk_language(self):
        _, request, frames = self._run_stream(
            [
                [_chunk("<|fr|><|transcribe|><|notimestamps|>", finish="stop")],
                [
                    _chunk(
                        "<|en|><|transcribe|><|notimestamps|> Hello",
                        finish="stop",
                    )
                ],
            ],
            fused=True,
        )
        self.assertEqual("".join(_deltas_from_sse(frames)), "Hello")
        self.assertEqual(request.language, "en")


class TestStreamingIncrementalOutputMode(CustomTestCase):
    """Server runs with ``incremental_streaming_output=True``.

    In that mode each chunk's ``content["text"]`` is the new delta from the
    detokenizer, not the cumulative text. The handler must accumulate
    locally into ``cumulative_text`` — otherwise the subsequent
    ``visible[len(visible_buffer):]`` slice would strip characters the
    server already sent as a delta.
    """

    def _run_incremental_stream(self, chunk_deltas, fused=False):
        """Server in incremental mode: yield per-chunk delta, not cumulative."""
        chunks = [
            _chunk(d, finish=("stop" if i == len(chunk_deltas) - 1 else None))
            for i, d in enumerate(chunk_deltas)
        ]
        tm = _MockTokenizerManager(chunks)
        tm.server_args = Mock(
            incremental_streaming_output=True,
            asr_max_concurrent_sessions=32,
        )
        serving = OpenAIServingTranscription(tm)

        request = TranscriptionRequest(model="whisper", stream=True)
        if fused:
            request._fused_autodetect = True
        adapted = GenerateReqInput(text="", modalities=["audio"])

        async def drive():
            frames = []
            async for f in serving._generate_transcription_stream(
                adapted, request, Mock()
            ):
                frames.append(f)
            return frames

        return request, get_or_create_event_loop().run_until_complete(drive())

    def test_incremental_non_fused_emits_each_delta_verbatim(self):
        # sglang.private default: each content["text"] IS the new delta, so
        # the handler should NOT slice it. Client should see exactly what
        # the detokenizer emitted.
        deltas_in = [" The", " President", ":", " Thank", " you"]
        _, frames = self._run_incremental_stream(deltas_in, fused=False)
        self.assertEqual(_deltas_from_sse(frames), deltas_in)

    def test_incremental_fused_autodetect_still_strips_prefix(self):
        # Incremental + fused: the handler must accumulate to find the
        # sentinel, then emit only the post-prefix portion per chunk.
        deltas_in = [
            "<|en|>",
            "<|transcribe|>",
            "<|notimestamps|>",
            " Hello",
            " world",
        ]
        request, frames = self._run_incremental_stream(deltas_in, fused=True)
        emitted = _deltas_from_sse(frames)
        # Prefix never leaks, and concat matches the expected transcription.
        self.assertFalse(any("<|" in d for d in emitted))
        self.assertEqual("".join(emitted), "Hello world")
        self.assertEqual(request.language, "en")


if __name__ == "__main__":
    unittest.main()
