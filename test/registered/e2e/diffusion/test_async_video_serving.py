# SPDX-License-Identifier: Apache-2.0
"""Real HTTP/GPU lifecycle regression with a small Wan checkpoint.

CI runs on diffusion partition 0. One GPU covers single/batch2; two GPUs
cover DP2, Ulysses2, TP2 and CFG2. Use ``-k dp2`` to select one topology.
"""

import hashlib
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import av
import psutil
import pytest
import requests
import torch

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import terminate_and_kill_process_tree

register_cuda_ci(est_time=180, stage="base-b", runner_config="diffusion-1-gpu-h100")
register_cuda_ci(est_time=480, stage="base-b", runner_config="diffusion-2-gpu-h100")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(
        os.environ.get("DIFFUSION_PARTITION_ID", "0") != "0",
        reason="lifecycle regression runs on diffusion partition 0",
    ),
]


@pytest.fixture(
    params=[
        ("single", 1, ""),
        ("batch2", 1, "--batching-max-size 2 --batching-delay-ms 30"),
        ("dp2", 2, "--dp-size 2"),
        ("sp2", 2, "--ulysses-degree 2"),
        ("tp2", 2, "--tp-size 2 --ulysses-degree 1"),
        ("cfg2", 2, "--enable-cfg-parallel --ulysses-degree 1"),
    ],
    ids=lambda config: config[0],
)
def server(request, tmp_path):
    mode, num_gpus, extra = request.param
    if torch.cuda.device_count() < num_gpus:
        pytest.skip(f"{mode} needs {num_gpus} GPUs")
    if num_gpus == 1 and torch.cuda.device_count() > 1:
        pytest.skip("single-GPU cases run in the one-GPU lane")
    ctx = ServerManager(
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        port=get_dynamic_server_port(),
        wait_deadline=600,
        extra_args=(
            f"--model-type diffusion --num-gpus {num_gpus} {extra} "
            f"--warmup-mode off --scheduler-rpc-timeout 30 "
            f"--strict-ports --output-path {tmp_path / 'videos'}"
        ),
    ).start()
    try:
        yield ctx, mode, num_gpus
    finally:
        terminate_and_kill_process_tree(ctx.process)
        ctx.cleanup()


@contextmanager
def _paused(workers):
    paused = []
    try:
        for worker in workers:
            worker.suspend()
            paused.append(worker)
        yield
    finally:
        for worker in paused:
            if worker.is_running():
                worker.resume()


def test_video_jobs_survive_concurrency_timeouts_and_deletion(
    server, tmp_path, record_property
):
    ctx, mode, num_gpus = server
    base = f"http://127.0.0.1:{ctx.port}"

    def call(method, route, expected=200, **kwargs):
        response = requests.request(method, base + route, timeout=10, **kwargs)
        assert response.status_code == expected, (route, response.text, ctx.log_tail())
        return response

    def submit(expected=200, **overrides):
        body = dict(
            prompt="A small sailboat on a calm blue lake in daylight.",
            size="512x320",
            num_frames=17,
            num_inference_steps=4,
            fps=8,
            seed=42,
            n=1,
        )
        body.update(overrides)
        response = call("POST", "/v1/videos", expected=expected, json=body)
        return response.json()["id"] if expected == 200 else None

    def wait(job_ids, expected="completed"):
        pending = set(job_ids)
        history = {job_id: ("queued", 0) for job_id in job_ids}
        deadline = time.monotonic() + 90
        while pending and time.monotonic() < deadline:
            for job_id in list(pending):
                job = call("GET", f"/v1/videos/{job_id}").json()
                status, progress = job["status"], job["progress"]
                old_status, old_progress = history[job_id]
                assert type(progress) is int and old_progress <= progress <= 100, job
                assert not (old_status == "in_progress" and status == "queued"), job
                assert (progress == 100) == (status == "completed"), job
                history[job_id] = (status, progress)
                if status in ("completed", "failed"):
                    assert status == expected, (job, ctx.log_tail())
                    pending.remove(job_id)
            call("GET", "/liveness")
            time.sleep(0.02)
        assert not pending, (pending, history, ctx.log_tail())

    def decoded_hash(job_id, variant=0):
        response = call(
            "GET", f"/v1/videos/{job_id}/content", params={"variant": variant}
        )
        digest = hashlib.sha256()
        with av.open(io.BytesIO(response.content)) as video:
            frames = 0
            for frame in video.decode(video=0):
                pixels = frame.to_ndarray(format="rgb24")
                assert pixels.shape == (320, 512, 3)
                digest.update(pixels.tobytes())
                frames += 1
        assert frames == 17
        return digest.hexdigest()

    # warm every replica and keep a serial reference, not a dynamic-batch reference
    for _ in range(num_gpus if mode == "dp2" else 1):
        baseline = submit()
        wait([baseline])
    reference = decoded_hash(baseline)
    record_property("decoded_reference_sha256", reference)
    children = psutil.Process(ctx.process.pid).children(recursive=True)
    workers = [
        child
        for child in children
        if child.cmdline()[0].startswith("sgl_diffusion::scheduler")
    ]
    assert len(workers) == num_gpus, [(p.pid, p.cmdline()) for p in children]

    with _paused(workers):
        # a deterministic queue barrier avoids racing deletion against a fast model
        deleted = submit()
        call("DELETE", f"/v1/videos/{deleted}")
        call("GET", f"/v1/videos/{deleted}", expected=404)
        with ThreadPoolExecutor(4) as pool:
            jobs = list(
                pool.map(lambda n: submit(n=n, num_inference_steps=12), [1, 2, 1, 2])
            )
        for job_id in jobs:
            call("GET", f"/v1/videos/{job_id}/content", expected=404)
        call("GET", "/health")

    wait(jobs)
    for job_id, outputs in zip(jobs, [1, 2, 1, 2]):
        hashes = [decoded_hash(job_id, variant) for variant in range(outputs)]
        assert len(set(hashes)) == outputs
    call("GET", f"/v1/videos/{deleted}", expected=404)
    call(
        "GET", f"/v1/videos/{baseline}/content", params={"variant": "99"}, expected=404
    )

    # fail output persistence after admission, not input validation or GPU compute
    bad_output = tmp_path / "unwritable-output"
    bad_output.mkdir()
    with _paused(workers):
        failed_save = submit(output_path=str(bad_output))
        bad_output.rmdir()
        bad_output.touch()
    try:
        wait([failed_save], expected="failed")
        call("GET", f"/v1/videos/{failed_save}/content", expected=404)
        job = call("GET", f"/v1/videos/{failed_save}").json()
        assert str(bad_output) in job["error"]["message"], job
    finally:
        bad_output.unlink()

    failed_jobs = {}
    with _paused(workers):
        timed_out = [submit() for _ in range(num_gpus if mode == "dp2" else 1)]
        wait(timed_out, expected="failed")
        for job_id in timed_out:
            call("GET", f"/v1/videos/{job_id}/content", expected=404)
            job = call("GET", f"/v1/videos/{job_id}").json()
            assert "Scheduler did not respond in time" in job["error"]["message"], job
            failed_jobs[job_id] = job

    # timeout ends the RPC, not GPU work: late results must not revive failed jobs
    for _ in range(3 if mode == "single" else num_gpus):
        recovered = submit()
        wait([recovered])
        assert decoded_hash(recovered) == reference
    for job_id in timed_out:
        job = call("GET", f"/v1/videos/{job_id}").json()
        assert job == failed_jobs[job_id] and job["progress"] < 100, job
        call("GET", f"/v1/videos/{job_id}/content", expected=404)
    assert decoded_hash(baseline) == reference
    call("GET", "/health")
    if mode == "batch2":
        assert "Processed dynamic batch of 2/2" in ctx.stdout_file.read_text()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
