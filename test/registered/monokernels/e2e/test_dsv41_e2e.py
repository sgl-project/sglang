# SPDX-License-Identifier: Apache-2.0
"""Serving tests of DeepSeek-V4.1's monokernels (gfx950, TP4, DSpark): steps that cross the largest width and come
back, and a wait stalled on one rank mid-request. Set MONO_MODEL to the checkpoint path."""

import concurrent.futures as cf
import json
import os
import re
import signal
import subprocess
import sys
import time
import unittest
import urllib.request

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=3600, suite="nightly-amd-8-gpu-mi35x", nightly=True)

MODEL = os.environ.get("MONO_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash")
PORT = int(os.environ.get("MONO_PORT", "31170"))
ARGS = (
    f"--model-path {MODEL} --tp 4 --port {PORT} --trust-remote-code --ep-size 1 --mem-fraction-static 0.70 "
    "--chunked-prefill-size 16384 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 "
    "--max-running-requests 16 --cuda-graph-max-bs-decode 64 --watchdog-timeout 3600 "
    "--cuda-graph-backend-prefill disabled --enable-decoder-swa-bounded-replay --fp8-gemm-backend aiter "
    "--enforce-shared-experts-fusion --enable-monokernel"
)


class Server:
    def __init__(self, log_name, env=None):
        self.log = open(log_name, "w")
        self.p = subprocess.Popen(
            f"{sys.executable} -m sglang.launch_server {ARGS}", shell=True, env={**os.environ, **(env or {})},
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip

    def ready(self, timeout=3600):
        t0 = time.time()
        while time.time() - t0 < timeout and self.p.poll() is None:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
                return True
            except Exception:
                time.sleep(10)
        return False

    def text(self):
        self.log.flush()
        return open(self.log.name, errors="replace").read()

    def stop(self):
        if self.p.poll() is None:
            os.killpg(self.p.pid, signal.SIGTERM)
            try:
                self.p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(self.p.pid, signal.SIGKILL)


def complete(prompt, n=64, timeout=900):
    body = json.dumps({"prompt": prompt, "max_tokens": n, "temperature": 0}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        body,
        {"Content-Type": "application/json"},
    )
    out = json.load(urllib.request.urlopen(req, timeout=timeout))
    return out["choices"][0]["text"], out["choices"][0].get("finish_reason")


PROMPTS = [f"Question {i}: what is {i} times {i + 3}? Answer:" for i in range(16)]


class TestSwitchPoints(unittest.TestCase):
    def test_cross_the_largest_width_and_back(self):
        srv = Server("e2e_switch_points.log")
        try:
            self.assertTrue(srv.ready(), "server did not start")
            phases = {}
            for name, conc in (
                ("1", 1),
                ("8 (48 rows)", 8),
                ("12 (72 rows)", 12),
                ("1 again", 1),
            ):
                with cf.ThreadPoolExecutor(conc) as ex:
                    phases[name] = list(ex.map(complete, PROMPTS[:conc]))
                for text, finish in phases[name]:
                    self.assertTrue(text.strip(), f"empty answer in phase {name}")
                    self.assertIn(finish, ("stop", "length"))
            log = srv.text()
            self.assertRegex(log, r"captured target_verify 48 rows on monokernels")
            self.assertRegex(
                log, r"target_verify steps of (5[4-9]|[6-9]\d) rows take the model path"
            )
            self.assertEqual(
                phases["1"][0][0],
                phases["1 again"][0][0],
                "the same request differs after the switch",
            )
        finally:
            srv.stop()


class TestFault(unittest.TestCase):
    def test_stalled_wait_stops_every_rank(self):
        # 'pmix' is a mid-step wait: with rank 1 stalled there, no rank can finish that step
        after = int(os.environ.get("MONO_FAULT_AFTER", "4000"))
        srv = Server(
            "e2e_fault.log", env={"SGLANG_TEST_MONO_FAULT_INJECT": f"1:pmix:{after}"}
        )
        try:
            self.assertTrue(srv.ready(), "server did not start")
            answer = None
            t0 = time.time()
            try:
                answer = complete("Count from 1 to 400, one number a line:", n=2048)
            except Exception:
                pass
            srv.p.wait(timeout=300)
            self.assertIsNone(answer, "the stalled step's request got an answer")
            self.assertIsNotNone(srv.p.poll(), "the server is still running")
            log = srv.text()
            self.assertTrue(
                re.search(r"Memory access fault|HIP error|died|exited", log),
                "no stop in the log",
            )
            print(
                f"server stopped {time.time() - t0:.0f} s after the request", flush=True
            )
        finally:
            srv.stop()


if __name__ == "__main__":
    unittest.main()
