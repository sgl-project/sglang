"""Data-parallel serving must route, replicate, and agree.

Launches one server with --dp-size 2 (one GPU per replica) and checks the three
properties that make DP real rather than a parsed flag: both replica drivers
bind their own ingress, both replicas serve traffic (round-robin means two
sequential requests land on different replicas), and a fixed seed produces the
same image bytes from either replica.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_dp_serving_2_gpu.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import unittest
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_MODEL = "Tongyi-MAI/Z-Image-Turbo"
_PORT = 30811
_STARTUP_TIMEOUT_S = 1200


def _post_generation(prompt: str, seed: int) -> dict:
    payload = json.dumps(
        {
            "prompt": prompt,
            "size": "512x512",
            "seed": seed,
            "num_inference_steps": 20,
            "response_format": "b64_json",
        }
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{_PORT}/v1/images/generations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _image_md5(response: dict) -> str:
    b64 = response["data"][0]["b64_json"]
    return hashlib.md5(base64.b64decode(b64)).hexdigest()


class TestDpServingTwoGpu(CustomTestCase):
    def test_two_replicas_serve_and_agree(self):
        if not current_platform.is_cuda():
            self.skipTest("DP e2e is exercised on CUDA")
        if torch.cuda.device_count() < 2:
            self.skipTest("needs 2 GPUs")

        log_path = Path("/tmp/dp_serving_test.log")
        fh = open(log_path, "w")
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.multimodal_gen.runtime.entrypoints.cli.main",
                "serve",
                "--model-path",
                _MODEL,
                "--num-gpus",
                "2",
                "--dp-size",
                "2",
                "--enable-cfg-parallel",
                "false",
                "--port",
                str(_PORT),
            ],
            stdout=fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            env=os.environ.copy(),
        )
        try:
            deadline = time.monotonic() + _STARTUP_TIMEOUT_S
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{_PORT}/health", timeout=5
                    ) as resp:
                        if resp.status == 200:
                            break
                except Exception:
                    time.sleep(5)
            else:
                self.fail(f"server not healthy in {_STARTUP_TIMEOUT_S}s")

            # warm each replica once so first-request costs stay out of timing
            for _ in range(2):
                _post_generation("warm pear", seed=7)

            # same seed through two round-robined replicas -> identical bytes
            first = _post_generation("a pear on a table", seed=42)
            second = _post_generation("a pear on a table", seed=42)
            self.assertEqual(_image_md5(first), _image_md5(second))

            log = log_path.read_text(errors="ignore")
            self.assertIn("dp replica 0) bind", log)
            self.assertIn("dp replica 1) bind", log)

            # distribution, not just agreement: a concurrent pair must run in
            # about one request's wall time. Two requests serialized on a
            # single replica would take ~2x the single-request time, so the
            # 1.6x bound separates the behaviors with margin for jitter.
            t0 = time.monotonic()
            single = _post_generation("a pear on a table", seed=42)
            single_s = time.monotonic() - t0
            self.assertTrue(single["data"])

            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(_post_generation, "a pear on a table", 42)
                    for _ in range(2)
                ]
                results = [f.result() for f in futures]
            pair_s = time.monotonic() - t0
            for r in results:
                self.assertTrue(r["data"])
            self.assertLess(
                pair_s,
                1.6 * single_s,
                f"concurrent pair took {pair_s:.2f}s vs single {single_s:.2f}s; "
                "requests are serializing on one replica",
            )
        finally:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            fh.close()


if __name__ == "__main__":
    unittest.main()
