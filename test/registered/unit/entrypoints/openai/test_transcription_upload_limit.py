"""Unit tests for the transcription upload size cap (sglang#43709).

The /v1/audio/transcriptions endpoint used to buffer the whole multipart
upload in host memory via ``await file.read()`` before any validation ran.
These tests pin the new contract: over-limit uploads are rejected with a 413
— via the declared-size precheck when the client is honest, and via the
chunked read when the declared size is missing or lies — and the serving
layer is never reached for rejected uploads.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.entrypoints.http_server import (
    _read_upload_with_limit,
    openai_v1_audio_transcriptions,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MIB = 1024 * 1024


class _FakeUploadFile:
    """starlette UploadFile semantics: read(size) returns up to size bytes."""

    def __init__(self, data: bytes, declared_size=None):
        self._data = data
        self.size = declared_size
        self._pos = 0

    async def read(self, size: int = -1):
        if size is None or size < 0:
            size = len(self._data) - self._pos
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


def _fake_mm(limit_mib: int):
    return SimpleNamespace(audio_transcription_max_file_size_mb=limit_mib)


def _raw_request(serving_result=None):
    serving = Mock()
    serving.create_transcription = AsyncMock(return_value=serving_result)
    return (
        SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(openai_serving_transcription=serving)
            )
        ),
        serving.create_transcription,
    )


class ReadUploadWithLimitTest(CustomTestCase):
    def test_under_limit_upload_is_returned_intact(self):
        import asyncio

        data = b"ab" * 1024  # 2 KiB
        out = asyncio.run(_read_upload_with_limit(_FakeUploadFile(data), 4 * _MIB))
        self.assertEqual(out, data)

    def test_over_limit_stream_returns_none(self):
        import asyncio

        # 3 MiB of data against a 2 MiB cap; the declared size is absent so
        # only the chunked read can catch it.
        data = b"x" * (3 * _MIB)
        self.assertIsNone(
            asyncio.run(_read_upload_with_limit(_FakeUploadFile(data), 2 * _MIB))
        )

    def test_exact_limit_passes(self):
        import asyncio

        data = b"x" * (2 * _MIB)
        out = asyncio.run(_read_upload_with_limit(_FakeUploadFile(data), 2 * _MIB))
        self.assertEqual(len(out), 2 * _MIB)

    def test_empty_upload_returns_empty_bytes(self):
        import asyncio

        self.assertEqual(
            asyncio.run(_read_upload_with_limit(_FakeUploadFile(b""), _MIB)), b""
        )


class TranscriptionUploadLimitTest(CustomTestCase):
    def _run(self, file, limit_mib, serving_result=None):
        import asyncio

        raw_request, serving = _raw_request(serving_result)
        with patch(
            "sglang.srt.entrypoints.http_server.get_mm",
            return_value=_fake_mm(limit_mib),
        ):
            response = asyncio.run(
                openai_v1_audio_transcriptions(
                    raw_request=raw_request, file=file, response_format="json"
                )
            )
        return response, serving

    def test_declared_over_limit_rejects_before_reading(self):
        file = _FakeUploadFile(b"anything", declared_size=30 * _MIB)
        response, serving = self._run(file, limit_mib=25)
        self.assertEqual(response.status_code, 413)
        self.assertIn(
            "--audio-transcription-max-file-size-mb",
            response.body.decode(),
        )
        serving.assert_not_awaited()

    def test_lying_declared_size_is_caught_while_streaming(self):
        # Declared 1 MiB, actually 30 MiB: only the chunked read catches it.
        file = _FakeUploadFile(b"x" * (30 * _MIB), declared_size=_MIB)
        response, serving = self._run(file, limit_mib=25)
        self.assertEqual(response.status_code, 413)
        serving.assert_not_awaited()

    def test_under_limit_upload_reaches_the_serving_layer(self):
        data = b"real audio bytes"
        file = _FakeUploadFile(data, declared_size=len(data))
        response, serving = self._run(file, limit_mib=25, serving_result="ok")
        self.assertEqual(response, "ok")
        serving.assert_awaited_once()
        self.assertEqual(serving.await_args.kwargs["audio_data"], data)

    def test_zero_limit_disables_the_cap(self):
        data = b"x" * (40 * _MIB)
        file = _FakeUploadFile(data, declared_size=len(data))
        response, serving = self._run(file, limit_mib=0, serving_result="ok")
        self.assertEqual(response, "ok")
        self.assertEqual(serving.await_args.kwargs["audio_data"], data)

    def test_invalid_response_format_still_rejected_first(self):
        import asyncio

        raw_request, serving = _raw_request()
        response = asyncio.run(
            openai_v1_audio_transcriptions(
                raw_request=raw_request,
                file=_FakeUploadFile(b"x"),
                response_format="srt",
            )
        )
        self.assertEqual(response.status_code, 400)
        serving.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
