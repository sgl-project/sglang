# SPDX-License-Identifier: Apache-2.0
"""Opt-in SR checkpoint test covering real JSON and multipart video requests."""

import io
import os
import shlex
import subprocess
import time

import av
import numpy as np
import pytest
import requests

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import (
    extract_audio_pcm_from_video_bytes,
    get_dynamic_server_port,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_KANDINSKY6_SR_TEST_MODEL"),
    reason="set SGLANG_KANDINSKY6_SR_TEST_MODEL for full-checkpoint HTTP tests",
)


@pytest.fixture(scope="module")
def source_videos(tmp_path_factory):
    directory = tmp_path_factory.mktemp("kandinsky6_sr_source")
    source = directory / "source.mp4"
    silent = directory / "silent.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=384x256:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=44100",
            "-frames:v",
            "33",
            "-t",
            str(33 / 24),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(source),
        ],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "copy",
            "-an",
            str(silent),
        ],
        check=True,
        timeout=60,
    )
    return source, silent


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    directory = tmp_path_factory.mktemp("kandinsky6_sr_server")
    manager = ServerManager(
        model=os.environ["SGLANG_KANDINSKY6_SR_TEST_MODEL"],
        port=get_dynamic_server_port(),
        extra_args=(
            "--performance-mode speed --attention-backend fa --warmup-mode off "
            f"--input-save-path {shlex.quote(str(directory / 'uploads'))} "
            f"--output-path {shlex.quote(str(directory / 'outputs'))} "
            + os.environ.get("SGLANG_KANDINSKY6_SR_TEST_SERVER_ARGS", "--num-gpus 1")
        ),
    )
    context = manager.start()
    try:
        yield context
    finally:
        context.cleanup()


def _generate(
    server, source, *, scale=2, tiles_batch_size=1, upload=False, has_audio=True
):
    url = f"http://127.0.0.1:{server.port}/v1/videos"
    payload = {
        "sr_resolution_scale": scale,
        "sr_tiles_batch_size": tiles_batch_size,
        "num_inference_steps": 2,
        "seed": 42,
    }
    with requests.Session() as client:
        if upload:
            with source.open("rb") as stream:
                response = client.post(
                    url,
                    data=payload,
                    files={"video_reference": (source.name, stream, "video/mp4")},
                    timeout=120,
                )
        else:
            response = client.post(
                url, json=payload | {"video_path": str(source)}, timeout=120
            )
        assert response.ok, response.text
        job_url = f"{url}/{response.json()['id']}"
        try:
            deadline = time.monotonic() + 300
            while True:
                response = client.get(job_url, timeout=30)
                assert response.ok, response.text
                job = response.json()
                assert job["status"] not in {"failed", "cancelled", "deleted"}, job
                if job["status"] == "completed":
                    break
                assert time.monotonic() < deadline, (job, server.log_tail())
                time.sleep(0.5)
            response = client.get(f"{job_url}/content", timeout=60)
            assert response.ok, response.text
            content = response.content
        finally:
            deleted = client.delete(job_url, timeout=30)
            assert deleted.ok, deleted.text
    with av.open(io.BytesIO(content)) as video:
        assert bool(video.streams.audio) == has_audio
        assert float(video.streams.video[0].average_rate) == 24
        frames = np.stack(
            [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
        )
    assert frames.shape == (33, int(256 * scale), int(384 * scale), 3)
    assert frames.std() > 1, "degenerate video"
    audio = extract_audio_pcm_from_video_bytes(content) if has_audio else np.empty(0)
    assert np.isfinite(audio).all()
    if has_audio:
        assert audio.size > 0
        assert np.sqrt(np.mean(np.square(audio, dtype=np.float64))) > 1e-5
    return frames, audio


def test_sr_requests_preserve_video_and_audio(server, source_videos):
    source, silent = source_videos
    first = _generate(server, source)
    uploaded = _generate(server, source, upload=True)
    for actual, expected in zip(uploaded, first, strict=True):
        np.testing.assert_array_equal(actual, expected)
    _generate(server, source, scale=4, upload=True)
    _generate(server, source, scale=2.25, tiles_batch_size=2)
    silent_output = _generate(server, silent, upload=True, has_audio=False)
    np.testing.assert_array_equal(silent_output[0], first[0])
    restored = _generate(server, source)
    for actual, expected in zip(restored, first, strict=True):
        np.testing.assert_array_equal(actual, expected)
