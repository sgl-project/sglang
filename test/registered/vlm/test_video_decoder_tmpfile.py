"""Tests for the temporary file VideoDecoderWrapper writes for byte sources."""

import errno
import os
import sys
import types
from unittest import mock

import pytest

from sglang.srt.utils import video_decoder as video_decoder_mod
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


PAYLOAD = b"\x00\x01\x02\x03" * 4096


@pytest.fixture
def decord_backend(monkeypatch):
    """Force the decord code path with a stub module, and capture the temp path."""
    recorded = {}

    class StubVideoReader:
        def __init__(self, path, ctx=None):
            recorded["decoded_path"] = path
            recorded["decoded_size"] = os.path.getsize(path)

        def __len__(self):
            return 0

    stub = types.ModuleType("decord")
    stub.VideoReader = StubVideoReader
    stub.cpu = lambda _idx: None
    monkeypatch.setitem(sys.modules, "decord", stub)
    monkeypatch.setattr(video_decoder_mod, "_BACKEND", "decord")

    import tempfile as _tempfile

    real = _tempfile.mkstemp

    def capturing_mkstemp(*args, **kwargs):
        fd, path = real(*args, **kwargs)
        recorded["tmp_path"] = path
        return fd, path

    monkeypatch.setattr(_tempfile, "mkstemp", capturing_mkstemp)
    yield recorded
    path = recorded.get("tmp_path")
    if path and os.path.exists(path):
        os.unlink(path)


def test_byte_source_is_written_in_full(decord_backend):
    """A short write(2) must not truncate the video.

    The kernel caps a single write(2) (0x7ffff000 bytes on Linux) and reports a
    short count instead of raising, so a bare os.write() silently truncates
    large videos. Simulate that cap by making os.write report a short count.
    """
    real_write = os.write

    def short_write(fd, data):
        # Mimic the kernel writing only part of a large buffer.
        return real_write(fd, data[: len(data) // 4])

    with mock.patch.object(os, "write", side_effect=short_write):
        decoder = VideoDecoderWrapper(PAYLOAD)

    assert decord_backend["decoded_size"] == len(PAYLOAD)
    decoder.close()


def test_tmp_file_not_orphaned_when_write_fails(decord_backend):
    """A failed write must not leave the temp file behind forever.

    close() can only remove what _tmp_path names, so if the path is recorded
    only after a successful write, an ENOSPC leaves the .mp4 on disk with
    nothing left holding a reference to it. __del__ is not a reliable cleanup
    path for an object whose __init__ raised, so __init__ must remove it.
    """
    enospc = OSError(errno.ENOSPC, "No space left on device")

    class FailingWriter:
        def __init__(self, fd, *args, **kwargs):
            self._fd = fd

        def write(self, _data):
            raise enospc

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            os.close(self._fd)
            return False

    def failing_write(fd, data):
        raise enospc

    with (
        mock.patch.object(os, "write", side_effect=failing_write),
        mock.patch.object(os, "fdopen", side_effect=FailingWriter),
    ):
        with pytest.raises(OSError):
            VideoDecoderWrapper(PAYLOAD)

    tmp_path = decord_backend["tmp_path"]
    assert not os.path.exists(
        tmp_path
    ), f"temp file {tmp_path} was orphaned after a failed write"
