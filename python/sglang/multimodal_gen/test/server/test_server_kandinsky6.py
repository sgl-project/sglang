# SPDX-License-Identifier: Apache-2.0
"""Opt-in checkpoint test; SGLANG_KANDINSKY6_TEST_SERVER_ARGS selects topology."""

import io
import json
import os
import sys
import time
from pathlib import Path

import av
import numpy as np
import pytest
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from safetensors.torch import save_file

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import (
    extract_audio_pcm_from_video_bytes,
    get_dynamic_server_port,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("SGLANG_KANDINSKY6_TEST_MODEL"),
    reason="set SGLANG_KANDINSKY6_TEST_MODEL for full-checkpoint HTTP tests",
)


@pytest.fixture(scope="module")
def server():
    manager = ServerManager(
        model=os.environ["SGLANG_KANDINSKY6_TEST_MODEL"],
        port=get_dynamic_server_port(),
        extra_args=(
            "--performance-mode speed --attention-backend fa --warmup-mode off "
            + os.environ.get("SGLANG_KANDINSKY6_TEST_SERVER_ARGS", "--num-gpus 1")
        ),
    )
    context = manager.start()
    try:
        yield context
    finally:
        context.cleanup()


def _generate_video(server, *, image=None, size="384x256", prompt=None, **overrides):
    url = f"http://127.0.0.1:{server.port}/v1/videos"
    payload = {
        "prompt": prompt or "A red ceramic teapot on a wooden table. Quiet jazz music.",
        "size": size,
        "num_frames": 17,
        "fps": 24,
        "num_inference_steps": 2,
        "seed": 42,
    } | overrides
    with requests.Session() as client:
        if image is None:
            response = client.post(url, json=payload, timeout=120)
        else:
            response = client.post(
                url,
                data=payload,
                files={"input_reference": ("reference.png", image, "image/png")},
                timeout=120,
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
            client.delete(job_url, timeout=30)
    with av.open(io.BytesIO(content)) as video:
        frames = np.stack(
            [frame.to_ndarray(format="rgb24") for frame in video.decode(video=0)]
        )
    audio = extract_audio_pcm_from_video_bytes(content)
    width, height = map(int, size.split("x"))
    assert frames.shape == (17, height, width, 3)
    assert frames.std() > 1, "degenerate video"
    assert audio.size > 0 and np.isfinite(audio).all()
    assert np.sqrt(np.mean(np.square(audio, dtype=np.float64))) > 1e-5, "silent audio"
    return frames, audio


def test_repeated_requests_and_conditioning_isolation(server):
    reference = Image.new("RGB", (384, 256), "white")
    ImageDraw.Draw(reference).rectangle((96, 64, 288, 192), fill="red")
    buffer = io.BytesIO()
    reference.save(buffer, format="PNG")

    first = _generate_video(server)
    repeated = _generate_video(server)
    for actual, expected in zip(repeated, first, strict=True):
        np.testing.assert_array_equal(actual, expected)

    conditioned = _generate_video(server, image=buffer.getvalue())
    conditioned_repeat = _generate_video(server, image=buffer.getvalue())
    for actual, expected in zip(conditioned_repeat, conditioned, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(conditioned[0], first[0]), (
        "image conditioning was ignored"
    )

    _generate_video(server, size="256x384", prompt="A blue glass vase. Rain sounds.")
    restored = _generate_video(server)
    for actual, expected in zip(restored, first, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_request_cache_dit_refresh_and_opt_out(server):
    baseline = _generate_video(server, num_inference_steps=6, enable_cache_dit=False)
    no_skip = _generate_video(
        server,
        num_inference_steps=6,
        enable_cache_dit=True,
        cache_dit_params={"max_warmup_steps": 6},
    )
    for actual, expected in zip(no_skip, baseline, strict=True):
        np.testing.assert_array_equal(actual, expected)

    cached_params = dict(
        num_inference_steps=6,
        enable_cache_dit=True,
        cache_dit_params={
            "max_warmup_steps": 1,
            "scm_compute_bins": [1, 1, 1],
            "scm_cache_bins": [1, 1, 1],
            "scm_policy": "static",
        },
    )
    first = _generate_video(server, **cached_params)
    assert any(not np.array_equal(a, b) for a, b in zip(first, baseline, strict=True))
    _generate_video(
        server,
        size="256x384",
        prompt="A blue glass vase. Rain sounds.",
        **cached_params,
    )
    repeated = _generate_video(server, **cached_params)
    for actual, expected in zip(repeated, first, strict=True):
        np.testing.assert_array_equal(actual, expected)
    restored = _generate_video(server, num_inference_steps=6, enable_cache_dit=False)
    for actual, expected in zip(restored, baseline, strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.fixture(scope="module")
def lora_adapter(tmp_path_factory):
    model = os.environ["SGLANG_KANDINSKY6_TEST_MODEL"]
    config_path = Path(model) / "transformer" / "config.json"
    if not config_path.is_file():
        config_path = Path(hf_hub_download(model, "transformer/config.json"))
    config = json.loads(config_path.read_text())
    layers = {
        "visual_transformer_blocks.0.video_dec_block.feed_forward.net.0.proj": (
            config["ff_dim"],
            config["model_dim"],
        ),
        "visual_transformer_blocks.0.audio_dec_block.feed_forward.net.2": (
            config["model_dim_a"],
            config["ff_dim_a"],
        ),
        "video_time_embeddings.timestep_embedder.linear_1": (
            config["time_dim"],
            config["model_dim"],
        ),
    }
    generator = torch.Generator().manual_seed(827)
    weights = {}
    for name, (out_dim, in_dim) in layers.items():
        for suffix, shape in (("A", (2, in_dim)), ("B", (out_dim, 2))):
            weights[f"transformer.{name}.lora_{suffix}.weight"] = (
                torch.randn(shape, generator=generator) * 0.05
            )
    path = tmp_path_factory.mktemp("kandinsky6_lora") / "adapter.safetensors"
    save_file(weights, path)
    return str(path)


@pytest.mark.parametrize("merge_mode", ["dynamic", "merge"])
def test_lora_scaling_repeatability_and_restore(server, lora_adapter, merge_mode):
    base_url = f"http://127.0.0.1:{server.port}/v1"
    baseline = _generate_video(server, enable_cache_dit=False)

    def set_strength(strength):
        response = requests.post(
            f"{base_url}/set_lora",
            json={
                "lora_nickname": "kandinsky6_synthetic",
                "lora_path": lora_adapter,
                "target": "transformer",
                "strength": strength,
                "merge_mode": merge_mode,
            },
            timeout=180,
        )
        assert response.ok, response.text

    try:
        set_strength(1.0)
        adapted = _generate_video(server, enable_cache_dit=False)
        for actual, expected in zip(adapted, baseline, strict=True):
            assert not np.array_equal(actual, expected), (
                "LoRA did not change the output"
            )
        repeated = _generate_video(server, enable_cache_dit=False)
        for actual, expected in zip(repeated, adapted, strict=True):
            np.testing.assert_array_equal(actual, expected)
        set_strength(0.0)
        zero = _generate_video(server, enable_cache_dit=False)
        for actual, expected in zip(zero, baseline, strict=True):
            np.testing.assert_array_equal(actual, expected)
        set_strength(0.5)
        scaled = _generate_video(server, enable_cache_dit=False)
        assert any(
            not np.array_equal(a, b) for a, b in zip(scaled, adapted, strict=True)
        )
        set_strength(1.0)
        restored_adapter = _generate_video(server, enable_cache_dit=False)
        for actual, expected in zip(restored_adapter, adapted, strict=True):
            np.testing.assert_array_equal(actual, expected)
    finally:
        response = requests.post(
            f"{base_url}/unmerge_lora_weights",
            json={"target": "transformer"},
            timeout=180,
        )
        assert response.ok, response.text
    restored_base = _generate_video(server, enable_cache_dit=False)
    for actual, expected in zip(restored_base, baseline, strict=True):
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
