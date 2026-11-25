# SPDX-License-Identifier: Apache-2.0
"""Client-supplied media URLs (video_path, input_reference/image_path, control_path,
model-specific inputs) must not make the server fetch non-public addresses, or hand
FFmpeg protocol-prefixed strings to ``av.open``. Public http(s) URLs keep working
through ``download_remote_media``.
"""

import asyncio
import io
import unittest.mock as mock
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from PIL import Image

from sglang.multimodal_gen.runtime.entrypoints.openai import utils as openai_utils
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _save_url_image_to_path,
    _validate_remote_media_paths,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    _iter_video_frames_pyav,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.multiview import (
    media_hw,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
    minimax_h3_localize_material_uri,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.audio import (
    extract_reference_video_audio,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.utils import vision
from sglang.srt.multimodal.media_io import configure_media_url_security

PRIVATE_URL = "http://127.0.0.1:19881/path1_target?run_id=test"
METADATA_URL = "http://169.254.169.254/latest/meta-data"
PUBLIC_VIDEO_URL = "http://93.184.216.34/clip.mp4"


@pytest.fixture(autouse=True)
def _reset_media_url_policy():
    configure_media_url_security([], max_file_size_mb=64)
    yield
    configure_media_url_security([], max_file_size_mb=64)


def _fake_av():
    fake_av = mock.MagicMock()
    container = mock.MagicMock()
    stream = mock.MagicMock()
    container.streams.video = [stream]
    container.decode.return_value = iter([])
    fake_av.open.return_value.__enter__.return_value = container
    return fake_av


class TestEntrypointMediaValidation:
    @pytest.mark.parametrize(
        "fields",
        [
            {"input_reference": METADATA_URL},
            {"control_path": [PUBLIC_VIDEO_URL, PRIVATE_URL]},
        ],
        ids=["input_reference", "control_path_list"],
    )
    def test_private_url_is_rejected(self, fields):
        with pytest.raises(HTTPException) as excinfo:
            _validate_remote_media_paths(fields)
        assert excinfo.value.status_code == 400
        assert "disallowed media URL" in excinfo.value.detail

    @pytest.mark.parametrize(
        "value",
        [
            "file:///etc/passwd",
            "file:/etc/passwd",
            "concat:clip1.mp4|clip2.mp4",
            "ftp://93.184.216.34/clip.mp4",
            "subfile,,start,0,end,99,,:http:169.254.169.254/latest",
        ],
    )
    def test_non_http_scheme_is_rejected(self, value):
        with pytest.raises(HTTPException) as excinfo:
            _validate_remote_media_paths({"video_path": value})
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"video_path": PUBLIC_VIDEO_URL},
            {"video_path": "/srv/media/clip.mp4"},
            {"video_path": "relative/clip.mp4"},
            {"video_path": "C:\\media\\clip.mp4"},
            {"image_path": "data:image/png;base64,iVBORw0KGgo="},
            {"control_path": None},
            {},
        ],
    )
    def test_legitimate_values_pass(self, kwargs):
        _validate_remote_media_paths(kwargs)


class TestLoadVideoDownload:
    def test_public_url_is_read_from_the_downloaded_bytes(self):
        read_bytes = []

        def read_frames(path, max_frames):
            with open(path, "rb") as f:
                read_bytes.append(f.read())
            return ["frame"]

        with (
            mock.patch.object(
                vision, "download_remote_media", return_value=b"fake-mp4"
            ) as download,
            mock.patch.object(vision, "_read_video_frames", side_effect=read_frames),
        ):
            assert vision.load_video(PUBLIC_VIDEO_URL) == ["frame"]
        assert download.call_args.args[0] == PUBLIC_VIDEO_URL
        assert read_bytes == [b"fake-mp4"]


class TestAvOpenGuard:
    @pytest.mark.parametrize(
        "source", ["file:///etc/passwd", "concat:clip1.mp4|clip2.mp4"]
    )
    def test_protocol_scheme_is_rejected_before_av_open(self, source):
        fake_av = _fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="Unsupported video path scheme"):
                list(_iter_video_frames_pyav(source, 1))
        fake_av.open.assert_not_called()

    def test_private_http_url_is_rejected_before_av_open(self):
        fake_av = _fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="Could not open video source"):
                list(_iter_video_frames_pyav(PRIVATE_URL, 1))
        fake_av.open.assert_not_called()

    def test_local_path_still_reaches_av_open(self, tmp_path):
        fake_av = _fake_av()
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"fixture")
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            assert list(_iter_video_frames_pyav(str(path), 1)) == []
        source = fake_av.open.call_args.args[0]
        assert source.name == str(path) and source.closed
        assert fake_av.open.call_args.kwargs == {
            "options": {"protocol_whitelist": "data"}
        }

    def test_approved_download_hands_decoder_local_bytes(self):
        fake_av = _fake_av()
        opened = []

        def open_stream(stream, *, options):
            assert stream.seekable() and stream.read() == b"fixture clip"
            assert options == {"protocol_whitelist": "data"}
            opened.append(stream)
            return fake_av.open.return_value

        fake_av.open.side_effect = open_stream
        with (
            mock.patch.dict("sys.modules", {"av": fake_av}),
            mock.patch.object(
                vision, "download_remote_media", return_value=b"fixture clip"
            ) as download,
        ):
            assert list(_iter_video_frames_pyav(PUBLIC_VIDEO_URL, 1)) == []
        assert download.call_args.args[0] == PUBLIC_VIDEO_URL
        assert len(opened) == 1


class TestSaveUrlImage:
    def test_extension_is_sniffed_from_the_downloaded_bytes(self, tmp_path):
        png = io.BytesIO()
        Image.new("RGB", (2, 2)).save(png, format="PNG")
        target = tmp_path / "image"
        with mock.patch.object(
            openai_utils, "download_remote_media", return_value=png.getvalue()
        ):
            saved = asyncio.run(
                _save_url_image_to_path("http://93.184.216.34/image", str(target))
            )
        assert saved == str(target) + ".png"
        assert (tmp_path / "image.png").read_bytes() == png.getvalue()


class TestModelSpecificFetchPaths:
    """Each loader fetches through the shared policy, so a private URL fails there."""

    def test_cosmos3_multiview_does_not_hand_av_a_url(self):
        fake_av = _fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="non-public IP"):
                media_hw(PRIVATE_URL)
        fake_av.open.assert_not_called()

    def test_wan_s2v_remote_input_is_checked(self, tmp_path):
        stage = object.__new__(WanS2VBeforeDenoisingStage)
        with (
            mock.patch("tempfile.tempdir", str(tmp_path)),
            pytest.raises(ValueError, match="non-public IP"),
        ):
            stage._get_single_path(PRIVATE_URL, "image_path")

    def test_wan_animate_2_audio_does_not_hand_av_a_url(self):
        fake_av = _fake_av()
        warnings = []
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            result = extract_reference_video_audio(
                PRIVATE_URL,
                log_info=lambda msg, *args: None,
                log_warning=lambda msg, *args: warnings.append(msg % args),
            )
        assert result == (None, None)
        assert len(warnings) == 1
        fake_av.open.assert_not_called()

    def test_minimax_h3_http_material_is_checked(self, tmp_path):
        batch = SimpleNamespace(extra={})
        with (
            mock.patch("tempfile.tempdir", str(tmp_path)),
            pytest.raises(ValueError, match="non-public IP"),
        ):
            minimax_h3_localize_material_uri(
                batch, PRIVATE_URL, condition_type="image", condition_index=0
            )
