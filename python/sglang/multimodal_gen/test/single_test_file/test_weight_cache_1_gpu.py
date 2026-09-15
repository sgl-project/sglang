# SPDX-License-Identifier: Apache-2.0
"""Real Wan HTTP recovery, parity, readiness and immutable-weight API tests.

Uses the normal diffusion ServerManager and video request/validation helpers.
Run with pytest. SGLANG_WEIGHT_CACHE_TEST_MODEL may select a local published
mirror; the default is the pinned Wan2.1 1.3B HF snapshot.
"""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import pytest
import requests
from openai import OpenAI

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerManager,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionSamplingParams
from sglang.multimodal_gen.test.test_utils import (
    compute_psnr,
    compute_ssim,
    extract_key_frames_from_video,
    get_video_frame_count,
)
from sglang.srt.utils.network import get_free_port
from sglang.weight_cache_common.liveness import ProcessIdentity

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA weight-cache adapter"
)


class TimedServerManager(ServerManager):
    def start(self):
        self.started = time.perf_counter()
        self.readiness = {}
        return super().start()

    def _wait_for_ready(self, process, stdout_path):
        stopped = threading.Event()

        def observe_liveness():
            while not stopped.is_set() and process.poll() is None:
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{self.port}/liveness", timeout=0.5
                    )
                    if response.status_code == 200:
                        self.readiness["liveness"] = time.perf_counter() - self.started
                        return
                except requests.RequestException:
                    pass
                stopped.wait(0.1)

        observer = threading.Thread(target=observe_liveness)
        observer.start()
        try:
            super()._wait_for_ready(process, stdout_path)
            self.readiness["health"] = time.perf_counter() - self.started
        finally:
            stopped.set()
            observer.join(timeout=2)
        assert "liveness" in self.readiness, "HTTP health without observed liveness"


def _wait_for(process, predicate, *, timeout=180):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert process.poll() is None, (
            f"Process {process.pid} exited: {process.returncode}"
        )
        assert time.monotonic() < deadline, f"Timed out waiting for {process.pid}"
        time.sleep(0.1)


def _start_owner(model, socket_path, env, log):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sglang.multimodal_gen.runtime.weight_cache.daemon",
            "--model-path",
            model,
            "--weight-cache-socket",
            str(socket_path),
        ],
        env={**os.environ, **env},
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    ready = socket_path.with_suffix(".ready")

    def published():
        if not ready.exists():
            return False
        text = ready.read_text()
        if not text.startswith(f"pid={process.pid}\n"):
            return False  # A stale predecessor is not readiness.
        generation = json.loads(text.split("\n", 1)[1])["generation"]
        return generation["producer"]["pid"] == process.pid

    try:
        _wait_for(process, published)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
        raise
    return process


def _stop_owner(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            raise


def _generate(context, model, case):
    params = DiffusionSamplingParams(
        output_size="832x480",
        prompt="A small boat sails across a calm lake at sunrise.",
        num_frames=9,
        extras={"num_inference_steps": 4, "seed": 42},
    )
    generate = get_generate_fn(model, "video", params)
    # No notifications/uploads as a side effect of this correctness test.
    with (
        OpenAI(
            base_url=f"http://127.0.0.1:{context.port}/v1",
            api_key="EMPTY",
            timeout=120,
            max_retries=0,
        ) as client,
        patch(
            "sglang.multimodal_gen.test.server.test_server_utils.upload_file_to_slack",
            return_value=False,
        ),
    ):
        _, content = generate(case, client)
    return content, extract_key_frames_from_video(content, num_frames=9)


def _assert_parity(actual, reference):
    assert len(actual) == len(reference) == 3
    for frame, expected in zip(actual, reference):
        assert frame.shape == expected.shape == (480, 832, 3)
        assert compute_ssim(frame, expected) >= 0.999
        assert compute_psnr(frame, expected) >= 50


def _assert_mutations_rejected(context):
    for endpoint, payload in (
        ("v1/set_lora", {"lora_nickname": "forbidden", "lora_path": "/must-not-open"}),
        ("v1/merge_lora_weights", {"target": "all"}),
        ("v1/unmerge_lora_weights", {"target": "all"}),
        ("update_weights_from_disk", {"model_path": "/must-not-open"}),
        (
            "update_weights_from_tensor",
            {"serialized_named_tensors": ["must-not-deserialize"]},
        ),
        ("release_memory_occupation", {}),
        ("resume_memory_occupation", {}),
    ):
        response = requests.post(
            f"http://127.0.0.1:{context.port}/{endpoint}", json=payload, timeout=20
        )
        assert response.status_code >= 400, (endpoint, response.text)
        assert "weight-cache" in response.text.lower(), (endpoint, response.text)


def _weights_checksum(context, *, timeout=60):
    response = requests.post(
        f"http://127.0.0.1:{context.port}/get_weights_checksum",
        json={"module_names": ["transformer"]},
        timeout=timeout,
    )
    response.raise_for_status()
    checksum = response.json()["transformer"]
    assert re.fullmatch(r"[0-9a-f]{64}", checksum), response.text
    return checksum


def check_wan_weight_cache_recovery(tmp_path, warmup, *, count=5):
    model = maybe_download_model(
        os.environ.get(
            "SGLANG_WEIGHT_CACHE_TEST_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        ),
        force_diffusers_model=True,
        revision="0fad780a534b6463e45facd96134c9f345acfa5b",
    )
    records = []
    reference = None
    weight_checksum = None
    with tempfile.TemporaryDirectory(prefix="sgl-wc-") as runtime_dir:
        socket_path = Path(runtime_dir) / "owner.sock"
        env = {"SGLANG_DIFFUSION_WEIGHT_CACHE_DIR": runtime_dir}
        base_flags = f"--num-gpus 1 --warmup-mode {warmup}"
        if warmup == "server":
            base_flags += (
                " --warmup-resolutions 832x480 --warmup-num-frames 9 --warmup-steps 1"
            )
        for index in range(count):
            manager = TimedServerManager(
                model, get_free_port(), extra_args=base_flags, env_vars=env
            )
            context = manager.start()
            try:
                assert f'"warmup_mode": "{warmup}"' in context.stdout_file.read_text()
                checksum = _weights_checksum(context)
                if weight_checksum is None:
                    weight_checksum = checksum
                assert checksum == weight_checksum
                content, frames = _generate(
                    context, model, f"cache-baseline-{warmup}-{index}"
                )
                if reference is None:
                    reference = frames
                    reference_path = tmp_path / f"reference-{warmup}.mp4"
                    reference_path.write_bytes(content)
                    assert get_video_frame_count(str(reference_path)) == 9
                else:
                    _assert_parity(frames, reference)
                records.append({"mode": "off", **manager.readiness})
            finally:
                context.cleanup()
                context.process.wait(timeout=20)

        owner = None
        context = None
        with (tmp_path / f"owner-{warmup}.log").open("w") as log:
            try:
                owner = _start_owner(model, socket_path, env, log)
                for index in range(count):
                    manager = TimedServerManager(
                        model,
                        get_free_port(),
                        extra_args=f"{base_flags} --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}",
                        env_vars=env,
                    )
                    context = manager.start()
                    assert (
                        f'"warmup_mode": "{warmup}"' in context.stdout_file.read_text()
                    )
                    assert _weights_checksum(context) == weight_checksum
                    _assert_mutations_rejected(context)
                    content, frames = _generate(
                        context, model, f"cache-restart-{warmup}-{index}"
                    )
                    _assert_parity(frames, reference)
                    assert _weights_checksum(context) == weight_checksum
                    video_path = tmp_path / f"cached-{warmup}-{index}.mp4"
                    video_path.write_bytes(content)
                    assert get_video_frame_count(str(video_path)) == 9
                    text = context.stdout_file.read_text()
                    found = re.findall(
                        r"\[WeightCache\] transformer imported in ([\d.]+)s", text
                    )
                    assert len(found) == 1, context.log_tail()
                    assert float(found[0]) < 2, found
                    assert "Using module transformer already provided" in text
                    assert "[ComponentLoader] transformer materialized" not in text
                    records.append(
                        {
                            "mode": "client",
                            "component": float(found[0]),
                            **manager.readiness,
                        }
                    )
                    assert owner.poll() is None
                    if index != count - 1:
                        # Existing ServerContext cleanup kills the client tree:
                        # every next iteration is a crash/restart under one owner.
                        context.cleanup()
                        context.process.wait(timeout=20)
                        context = None

                workers = [
                    ProcessIdentity.read(p.pid)
                    for p in psutil.Process(context.process.pid).children(
                        recursive=True
                    )
                    if p.name().startswith("sgl_diffusion")
                ]
                assert len(workers) == 1, workers
                owner.send_signal(signal.SIGTERM if warmup == "off" else signal.SIGKILL)
                deadline = time.monotonic() + 10
                while any(p.is_alive() for p in workers):
                    assert time.monotonic() < deadline, (
                        "Attached worker survived owner death"
                    )
                    time.sleep(0.1)
                owner.wait(timeout=20)
                context.cleanup()
                context.process.wait(timeout=20)
                context = None
                # Reclaim the killed generation's stale files without --force.
                owner = _start_owner(model, socket_path, env, log)
                _stop_owner(owner)
                assert not socket_path.exists()
                assert not socket_path.with_suffix(".ready").exists()
                missing = TimedServerManager(
                    model,
                    get_free_port(),
                    wait_deadline=90,
                    extra_args=f"{base_flags} --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}",
                    env_vars=env,
                )
                with pytest.raises(
                    RuntimeError, match="FileNotFoundError|No such file"
                ):
                    missing.start()
            finally:
                if context is not None:
                    context.cleanup()
                    context.process.wait(timeout=20)
                _stop_owner(owner)
    summary = {"warmup": warmup, "samples": records, "weight_checksum": weight_checksum}
    for mode in ("off", "client"):
        subset = [r for r in records if r["mode"] == mode]
        summary[mode] = {
            name: {
                "median": float(np.median([r[name] for r in subset])),
                "p90": float(np.percentile([r[name] for r in subset], 90)),
            }
            for name in ("liveness", "health")
        }
    # A model/host-derived regression bound, not a claim of startup speedup.
    assert summary["client"]["health"]["p90"] < summary["off"]["health"]["p90"] * 1.25
    (tmp_path / f"readiness-{warmup}.json").write_text(json.dumps(summary, indent=2))
    print("WEIGHT_CACHE_HTTP_RESULT", json.dumps(summary), flush=True)


@pytest.mark.parametrize("warmup", ["off", "server"])
def test_wan_weight_cache_recovery(tmp_path, warmup):
    check_wan_weight_cache_recovery(tmp_path, warmup, count=5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
