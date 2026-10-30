# SPDX-License-Identifier: Apache-2.0
"""The video-encoder registry is the seam ``EncodeWhileDecoding`` resolves through."""

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages import video_encoders as ve


class _Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.written = []
        self.closed = False

    def write(self, frames):
        self.written.append(frames)

    def close(self):
        self.closed = True
        return ["/tmp/out.mp4"]

    def abort(self):
        pass


def test_default_name_resolves_to_builtin(monkeypatch):
    """No name means the built-in encoder, i.e. today's behaviour."""
    seen = {}

    def fake_open(**kwargs):
        seen.update(kwargs)
        return _Recorder(**kwargs)

    monkeypatch.setattr(ve, "_default_factory", lambda: fake_open)
    encoder = ve.open_video_encoder(
        None, save_file_path="/tmp/x.mp4", height=64, width=64
    )
    assert isinstance(encoder, _Recorder)
    assert seen["save_file_path"] == "/tmp/x.mp4"
    assert ve.get_video_encoder_factory(None) is fake_open


def test_registered_encoder_is_selected_by_name():
    ve.register_video_encoder("recorder-test", _Recorder, replace=True)
    encoder = ve.open_video_encoder("recorder-test", save_file_path="/tmp/y.mp4")
    assert isinstance(encoder, _Recorder)
    assert "recorder-test" in ve.available_video_encoders()


def test_unknown_name_names_the_alternatives():
    with pytest.raises(ValueError) as error:
        ve.open_video_encoder("no-such-encoder")
    assert "no-such-encoder" in str(error.value)
    assert "ffmpeg" in str(error.value)


def test_registration_does_not_shadow_by_default():
    ve.register_video_encoder("dup-test", _Recorder, replace=True)
    with pytest.raises(ValueError):
        ve.register_video_encoder("dup-test", _Recorder)
    ve.register_video_encoder("dup-test", _Recorder, replace=True)


def test_builtin_encoder_satisfies_the_protocol():
    """The default encoder must actually conform, not just be adapted into place."""
    from sglang.multimodal_gen.runtime.entrypoints.utils import CudaVideoEncoder

    for method in ("write", "close", "abort"):
        assert callable(getattr(CudaVideoEncoder, method)), method


def test_protocol_matches_the_builtin_signatures():
    """close() takes no arguments and returns nothing, as the built-in does."""
    import inspect

    from sglang.multimodal_gen.runtime.entrypoints.utils import CudaVideoEncoder

    for method in ("write", "close", "abort"):
        builtin = inspect.signature(getattr(CudaVideoEncoder, method))
        declared = inspect.signature(getattr(ve.VideoEncoder, method))
        assert list(builtin.parameters) == list(declared.parameters), method


def test_finish_reports_a_different_path_once(monkeypatch):
    """An encoder may write elsewhere, but exactly one path per request."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        encode_while_decoding as ewd,
    )

    class _Elsewhere:
        def write(self, frames):
            pass

        def close(self):
            return "/tmp/actually-written.mp4"

        def abort(self):
            pass

    stage = object.__new__(ewd.EncodeWhileDecoding)
    stage._encoder = _Elsewhere()
    stage._failed = False
    stage._writes_file = True
    stage.save_file_path = "/tmp/asked-for.mp4"
    stage._batch = type("B", (), {})()
    assert stage.finish() == ["/tmp/actually-written.mp4"]


def test_finish_falls_back_to_the_opened_path(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        encode_while_decoding as ewd,
    )

    class _Builtin:
        def write(self, frames):
            pass

        def close(self):
            return None

        def abort(self):
            pass

    stage = object.__new__(ewd.EncodeWhileDecoding)
    stage._encoder = _Builtin()
    stage._failed = False
    stage._writes_file = True
    stage.save_file_path = "/tmp/asked-for.mp4"
    stage._batch = type("B", (), {})()
    assert stage.finish() == ["/tmp/asked-for.mp4"]


def test_default_encoder_can_be_chosen_without_a_server_arg():
    ve.register_video_encoder("alternative", _Recorder, replace=True)
    ve.set_default_video_encoder("alternative")
    try:
        assert ve.default_video_encoder() == "alternative"
        assert ve.get_video_encoder_factory(None) is _Recorder
    finally:
        ve.set_default_video_encoder(None)
    assert ve.default_video_encoder() == ve.DEFAULT_VIDEO_ENCODER


def test_choosing_an_unknown_default_raises():
    with pytest.raises(ValueError):
        ve.set_default_video_encoder("does-not-exist")
    assert ve.default_video_encoder() == ve.DEFAULT_VIDEO_ENCODER


def test_stage_opens_the_configured_encoder():
    """The configured default is what the stage opens, with no server arg involved."""
    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        encode_while_decoding as ewd,
    )

    seen = {}

    class _Spy:
        def write(self, frames):
            pass

        def close(self):
            return None

        def abort(self):
            pass

    def factory(**kwargs):
        seen.update(kwargs)
        return _Spy()

    ve.register_video_encoder("spy-open", factory, replace=True)
    ve.set_default_video_encoder("spy-open")
    try:
        stage = object.__new__(ewd.EncodeWhileDecoding)
        stage.save_file_path = "/tmp/spy.mp4"
        stage._batch = type(
            "B",
            (),
            {
                "fps": 24,
                "num_frames": 4,
                "output_compression": None,
                "x264_preset": None,
            },
        )()
        stage._audio = None
        stage._audio_sample_rate = None
        stage._encoder = None
        stage._failed = False
        stage._writes_file = True

        stage._encoder = stage._open(torch.zeros(3, 4, 16, 16))
        assert seen["save_file_path"] == "/tmp/spy.mp4"
        assert seen["fps"] == 24
        assert seen["width"] == 16 and seen["height"] == 16
    finally:
        ve.set_default_video_encoder(None)


def test_multiple_chunks_arrive_as_frame_runs():
    """write() is called once per finished run, each [3, T, H, W]."""
    import torch

    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        encode_while_decoding as ewd,
    )

    class _Recorder:
        def __init__(self):
            self.chunks = []

        def write(self, frames):
            self.chunks.append(frames)

        def close(self):
            return None

        def abort(self):
            pass

    encoder = _Recorder()
    stage = object.__new__(ewd.EncodeWhileDecoding)
    stage._batch = type(
        "B",
        (),
        {
            "fps": 24,
            "num_frames": 9,
            "output_compression": None,
            "x264_preset": None,
            "extra": {},
        },
    )()
    stage._finish_frames = None
    stage._audio = None
    stage._audio_sample_rate = None
    stage._encoder = encoder
    stage._failed = False
    stage._writes_file = True
    stage.save_file_path = "/tmp/chunks.mp4"

    stage(torch.zeros(1, 3, 9, 16, 16))

    assert len(encoder.chunks) >= 1
    for chunk in encoder.chunks:
        assert chunk.shape[0] == 3, "each write is [3, T, H, W]"
        assert chunk.shape[-2:] == (16, 16)
    assert sum(c.shape[1] for c in encoder.chunks) == 9


def test_builtin_receives_the_compression_settings(monkeypatch):
    """The default factory must forward x264_preset and output_compression."""
    from sglang.multimodal_gen.runtime.entrypoints import utils as entry_utils

    seen = {}

    def fake_open(**kwargs):
        seen.update(kwargs)
        return _Recorder()

    monkeypatch.setattr(entry_utils.CudaVideoEncoder, "open", staticmethod(fake_open))

    factory = ve.get_video_encoder_factory(None)
    factory(
        save_file_path="/tmp/x.mp4",
        device="cuda:0",
        height=320,
        width=480,
        fps=24,
        num_frames=25,
        output_compression=19,
        x264_preset="ultrafast",
        max_queued_frames=17,
    )

    assert seen["x264_preset"] == "ultrafast"
    assert seen["output_compression"] == 19
    assert seen["num_frames"] == 25
    assert seen["max_queued_frames"] == 17
