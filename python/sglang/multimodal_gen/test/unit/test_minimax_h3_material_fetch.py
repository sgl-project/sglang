# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 pre-queue probes fetch a request's remote materials concurrently.

No network and no ffmpeg: an in-process origin stands in for urlopen, and the
audio and video files hold the ffprobe report they are probed for.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    material_io,
    prequeue,
    video_adapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
    MINIMAX_H3_MATERIAL_CACHE_EXTRA_KEY,
    MINIMAX_H3_MATERIAL_PROBE_EXTRA_KEY,
    MINIMAX_H3_TEMP_DIRS_EXTRA_KEY,
)

# a wait that only expires when the code under test is broken
_WAIT_S = 10.0


def _container(format_name: str, *streams: dict) -> bytes:
    report = {
        "streams": streams,
        "format": {"format_name": format_name, "duration": "1.0"},
    }
    return json.dumps(report).encode()


@pytest.fixture(scope="module")
def media(tmp_path_factory) -> dict[str, Path]:
    image = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(image, format="PNG")
    video = {
        "codec_type": "video",
        "width": 64,
        "height": 48,
        "avg_frame_rate": "24/1",
        "duration": "1.0",
    }
    audio = {
        "codec_type": "audio",
        "sample_rate": "16000",
        "channels": 1,
        "duration": "1.0",
    }
    files = {
        "image.png": image.getvalue(),
        "video.mp4": _container("mov,mp4,m4a,3gp,3g2,mj2", video),
        "video_audio.mp4": _container("mov,mp4,m4a,3gp,3g2,mj2", video, audio),
        "audio.wav": _container("wav", audio),
        "garbage.png": b"not media",
        "garbage.wav": b"not media",
        "empty.wav": b"",
    }
    root = tmp_path_factory.mktemp("media")
    for name, body in files.items():
        (root / name).write_bytes(body)
    return {name: root / name for name in files}


class _Origin:
    """Stands in for urlopen: serves the media and counts requests in flight."""

    def __init__(self, media: dict[str, Path]) -> None:
        self.files = {name: path.read_bytes() for name, path in media.items()}
        self.lock = threading.Lock()
        self.linger_s = 0.0
        self.expect(1)

    def expect(self, in_flight: int) -> None:
        """Answer no request until `in_flight` of them are open at once."""

        self.answer_at = in_flight
        self.answering = threading.Event()
        self.hits: dict[str, int] = {}
        self.in_flight = 0
        self.max_in_flight = 0

    def urlopen(self, uri: str, timeout: float) -> io.BytesIO:
        with self.lock:
            self.hits[uri] = self.hits.get(uri, 0) + 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            if self.in_flight >= self.answer_at:
                self.answering.set()
        try:
            if not self.answering.wait(_WAIT_S):
                # fail this request; the ones after it need not wait as long again
                self.answering.set()
                raise AssertionError("the downloads did not overlap")
            time.sleep(self.linger_s)
            body = self.files.get(Path(urllib.parse.urlsplit(uri).path).name)
            if body is None:
                raise urllib.error.HTTPError(uri, 404, "Not Found", None, None)
            return io.BytesIO(body)
        finally:
            with self.lock:
                self.in_flight -= 1


@pytest.fixture
def origin(monkeypatch, media) -> _Origin:
    origin = _Origin(media)
    from contextlib import contextmanager

    @contextmanager
    def open_material(uri, timeout):
        with origin.urlopen(uri, timeout) as response:
            yield {}, iter(lambda: response.read(65536), b"")

    monkeypatch.setattr(material_io, "open_remote_media", open_material)
    monkeypatch.setattr(
        material_io, "_ffprobe_media", lambda path: json.loads(Path(path).read_bytes())
    )
    return origin


@pytest.fixture(autouse=True)
def temp_root(monkeypatch, tmp_path) -> Path:
    # request material workdirs are created here, so leftovers are visible
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))
    return root


def _plan(media, *spec) -> SimpleNamespace:
    """spec: (condition_type, file, None for a local path or a tag of a distinct URL)."""

    materials = []
    for index, (condition_type, name, tag) in enumerate(spec):
        uri = (
            str(media[name]) if tag is None else f"https://media.test/{name}?sig={tag}"
        )
        materials.append(
            SimpleNamespace(
                condition_index=index, condition_type=condition_type, uri=uri
            )
        )
    return SimpleNamespace(materials=materials)


def _finish_in_reverse(patch, plan) -> None:
    """Hold each URI's first probe until the next URI's first probe has finished.

    The threads then finish in reverse request order, so results and errors
    equal the sequential path's only if they are taken in request order.
    """

    first_by_uri = {}
    for material in plan.materials:
        first_by_uri.setdefault(material.uri, material.condition_index)
    firsts = list(first_by_uri.values())
    finished = {index: threading.Event() for index in firsts}
    following = dict(zip(firsts, firsts[1:]))
    probe = prequeue.minimax_h3_probe_material

    def held(batch, uri, *, condition_type, condition_index):
        try:
            return probe(
                batch,
                uri,
                condition_type=condition_type,
                condition_index=condition_index,
            )
        finally:
            later = following.get(condition_index)
            if later is not None and not finished[later].wait(_WAIT_S):
                for event in finished.values():
                    event.set()
                raise AssertionError("the probes did not run concurrently")
            if condition_index in finished:
                finished[condition_index].set()

    patch.setattr(prequeue, "minimax_h3_probe_material", held)


@contextlib.contextmanager
def _path(monkeypatch, origin, plan, *, concurrent: bool):
    """Take the sequential path, or the concurrent one under its worst timing."""

    remote = {m.uri for m in plan.materials if m.uri.startswith("https:")}
    origin.expect(len(remote) if concurrent else 1)
    with monkeypatch.context() as patch:
        if concurrent:
            _finish_in_reverse(patch, plan)
        else:
            patch.setattr(prequeue, "_is_concurrent_material_uri", lambda uri: False)
        yield


def _probe(monkeypatch, origin, plan, *, concurrent: bool) -> dict:
    """Facts, localized files and request caches, with the random workdir masked."""

    batch = SimpleNamespace(extra={})
    with _path(monkeypatch, origin, plan, concurrent=concurrent):
        results = prequeue._probe_materials(batch, plan)
    workdirs = batch.extra.get(MINIMAX_H3_TEMP_DIRS_EXTRA_KEY, {}).get("material", [])
    assert len(workdirs) <= 1

    def mask(value):
        if workdirs and isinstance(value, str):
            return value.replace(workdirs[0], "<workdir>")
        return value

    files = {}
    if workdirs:
        files = {path.name: path.read_bytes() for path in Path(workdirs[0]).iterdir()}
    cache = batch.extra.get(MINIMAX_H3_MATERIAL_CACHE_EXTRA_KEY, {})
    return {
        "results": [{key: mask(value) for key, value in r.items()} for r in results],
        "files": files,
        "cache": {uri: mask(path) for uri, path in cache.items()},
        "probe": batch.extra.get(MINIMAX_H3_MATERIAL_PROBE_EXTRA_KEY),
        "extra_keys": sorted(batch.extra),
    }


def test_local_and_http_materials_are_probed_concurrently() -> None:
    assert prequeue._is_concurrent_material_uri("/data/ref.mp4")
    assert prequeue._is_concurrent_material_uri("file:///data/ref.png")
    assert prequeue._is_concurrent_material_uri("https://example.com/ref.mp4")
    assert not prequeue._is_concurrent_material_uri("s3://bucket/ref.mp4")
    assert not prequeue._is_concurrent_material_uri("data:image/png;base64,AAAA")
    assert not prequeue._is_concurrent_material_uri(None)


@pytest.mark.parametrize("count", [2, 3, 4])
def test_remote_materials_are_fetched_concurrently(
    monkeypatch, origin, media, count
) -> None:
    spec = [
        ("image", "image.png", "a"),
        ("video", "video.mp4", "b"),
        ("audio", "audio.wav", "c"),
        ("video_audio", "video_audio.mp4", "d"),
    ][:count]
    plan = _plan(media, *spec)
    mkdtemp = tempfile.mkdtemp

    def slow_mkdtemp(*args, **kwargs):
        # widen the window in which the threads find no request workdir yet
        time.sleep(0.05)
        return mkdtemp(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkdtemp", slow_mkdtemp)
    sequential = _probe(monkeypatch, origin, plan, concurrent=False)
    assert origin.max_in_flight == 1
    concurrent = _probe(monkeypatch, origin, plan, concurrent=True)
    assert origin.max_in_flight == count
    assert concurrent == sequential
    assert concurrent["files"] == {
        f"condition_{index:04d}{Path(name).suffix}": origin.files[name]
        for index, (_, name, _) in enumerate(spec)
    }


def test_a_request_downloads_on_at_most_eight_threads(origin, media) -> None:
    cap = prequeue._PROBE_MAX_THREADS
    assert cap == 8
    plan = _plan(media, *(("image", "image.png", tag) for tag in range(cap + 2)))
    origin.expect(cap)
    # time for a thread beyond the cap to open its request while these are held
    origin.linger_s = 0.05
    results = prequeue._probe_materials(SimpleNamespace(extra={}), plan)
    assert origin.max_in_flight == cap
    assert [Path(r["local_path"]).name for r in results] == [
        f"condition_{index:04d}.png" for index in range(cap + 2)
    ]
    assert origin.hits == {material.uri: 1 for material in plan.materials}


@pytest.mark.parametrize(
    "spec",
    [
        [("image", "image.png", "a")],
        [("video", "video.mp4", None), ("audio", "audio.wav", "a")],
        [
            ("image", "image.png", "a"),
            ("image", "image.png", None),
            ("video_audio", "video_audio.mp4", "b"),
        ],
        [
            ("audio", "audio.wav", None),
            ("video", "video.mp4", "a"),
            ("image", "image.png", "b"),
            ("video_audio", "video_audio.mp4", None),
        ],
        # one URI twice: fetched once, the repeat reads the first condition's file
        [
            ("video_audio", "video_audio.mp4", "a"),
            ("image", "image.png", "b"),
            ("audio", "video_audio.mp4", "a"),
        ],
    ],
)
def test_mixed_materials_match_the_sequential_probes(
    monkeypatch, origin, media, spec
) -> None:
    plan = _plan(media, *spec)
    sequential = _probe(monkeypatch, origin, plan, concurrent=False)
    concurrent = _probe(monkeypatch, origin, plan, concurrent=True)
    assert origin.hits == {
        material.uri: 1
        for material in plan.materials
        if material.uri.startswith("https:")
    }
    assert concurrent == sequential
    assert [r["condition_type"] for r in concurrent["results"]] == [
        condition_type for condition_type, _, _ in spec
    ]


@pytest.mark.parametrize(
    "spec, error_type, message",
    [
        # condition 1's 404 is reported although condition 2 fails to decode first
        (
            [
                ("image", "image.png", "a"),
                ("video", "missing.mp4", "b"),
                ("image", "garbage.png", "c"),
            ],
            urllib.error.HTTPError,
            "HTTP Error 404: Not Found",
        ),
        (
            [("audio", "empty.wav", "a"), ("video", "missing.mp4", "b")],
            ValueError,
            "HTTP material body is empty",
        ),
        (
            [
                ("video", "video.mp4", None),
                ("image", "image.png", "a"),
                ("audio", "garbage.wav", "b"),
            ],
            ValueError,
            "MiniMax H3 media material is invalid",
        ),
        # the repeated URI fails as an image, ahead of condition 3's 404
        (
            [
                ("video_audio", "video_audio.mp4", "a"),
                ("image", "image.png", "b"),
                ("image", "video_audio.mp4", "a"),
                ("video", "missing.mp4", "c"),
            ],
            ValueError,
            "MiniMax H3 image material is invalid",
        ),
    ],
)
def test_failures_raise_the_first_failing_condition_and_leave_no_files(
    monkeypatch, origin, media, temp_root, spec, error_type, message
) -> None:
    plan = _plan(media, *spec)
    monkeypatch.setattr(prequeue, "minimax_h3_plan_from_batch", lambda batch: plan)
    errors = []
    for concurrent in (False, True):
        batch = SimpleNamespace(extra={})
        with _path(monkeypatch, origin, plan, concurrent=concurrent):
            with pytest.raises(error_type, match=message) as info:
                prequeue.minimax_h3_prepare_for_queue(batch)
        errors.append(info.value)
        assert batch.extra == {}
        assert list(temp_root.iterdir()) == []
    sequential, concurrent = errors
    assert type(concurrent) is type(sequential)
    assert str(concurrent) == str(sequential)


def test_a_playlist_url_is_staged_as_the_condition_type_and_read_as_a_file(
    monkeypatch, media
) -> None:
    @contextlib.contextmanager
    def open_playlist(uri, timeout):
        headers = {"Content-Type": "application/vnd.apple.mpegurl"}
        yield headers, iter([media["video.mp4"].read_bytes()])

    monkeypatch.setattr(material_io, "open_remote_media", open_playlist)
    monkeypatch.setattr(
        material_io, "_ffprobe_media", lambda path: json.loads(Path(path).read_bytes())
    )
    path = material_io.minimax_h3_localize_material_uri(
        SimpleNamespace(extra={}),
        "https://media.test/clip.m3u8",
        condition_type="video",
        condition_index=0,
    )
    assert Path(path).name == "condition_0000.mp4"

    opened = []

    def av_open(target, **kwargs):
        opened.append(kwargs)
        return contextlib.nullcontext(
            SimpleNamespace(streams=[], format=SimpleNamespace(name="mp4"), duration=0)
        )

    monkeypatch.setattr(
        video_adapter,
        "av",
        SimpleNamespace(open=av_open, error=SimpleNamespace(FFmpegError=OSError)),
    )
    video_adapter._read_output_media(path)
    assert opened == [{"options": {"protocol_whitelist": "file"}}]
