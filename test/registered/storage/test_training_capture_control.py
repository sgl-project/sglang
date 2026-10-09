"""Live capture control preserves generation and asynchronous Store ownership."""

import argparse
import importlib.util
import json
import shutil
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar
from unittest.mock import patch

import requests
import test_training_capture_runtime as runtime
import torch
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

CUDA_GRAPH_OVERLAP = False


@unittest.skipUnless(
    torch.cuda.is_available()
    and shutil.which("mooncake_master")
    and importlib.util.find_spec("mooncake"),
    "CUDA and Mooncake required",
)
class TestTrainingCaptureControl(CustomTestCase):
    headers: ClassVar = {"Authorization": "Bearer capture-control-test-admin"}

    @classmethod
    def setUpClass(cls):
        launch = runtime.popen_launch_server
        capture_launch = runtime.TestTrainingCaptureRuntime.launch_server

        def authenticated_server(*args, **kwargs):
            kwargs["other_args"] += ["--admin-api-key", "capture-control-test-admin"]
            if CUDA_GRAPH_OVERLAP:
                kwargs["other_args"].remove("--disable-overlap-schedule")
            return launch(*args, **kwargs)

        with (
            patch.object(runtime, "popen_launch_server", authenticated_server),
            patch.object(
                runtime.TestTrainingCaptureRuntime,
                "launch_server",
                side_effect=lambda path: capture_launch(
                    path, cuda_graph=CUDA_GRAPH_OVERLAP
                ),
            ),
        ):
            runtime.TestTrainingCaptureRuntime.setUpClass()
        cls.fixture = runtime.TestTrainingCaptureRuntime()

    @classmethod
    def tearDownClass(cls):
        runtime.TestTrainingCaptureRuntime.tearDownClass()

    def state(self):
        response = requests.get(
            self.fixture.url + "/server_info", headers=self.headers, timeout=10
        )
        response.raise_for_status()
        return response.json()["internal_states"][0]["training_capture"]

    def wait_until(self, predicate):
        deadline = time.monotonic() + 30
        while True:
            state = self.state()
            if predicate(state):
                return state
            self.assertLess(time.monotonic(), deadline, state)
            time.sleep(0.01)

    def control(self, action):
        response = requests.post(
            self.fixture.url + "/control_training_capture",
            json={"action": action},
            headers=self.headers,
            timeout=10,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["success"], result)
        state = result["results"][0]["state"]
        self.assertEqual(state["admission_paused"], action != "resume")
        self.assertIsNone(state["disabled_reason"])
        return state

    def generate(self, length):
        response = requests.post(
            self.fixture.url + "/generate",
            json={
                "input_ids": [100, 200, 300, 400] * 4,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": length,
                    "ignore_eos": True,
                },
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        tokens = response.json()["output_ids"]
        self.assertEqual(len(tokens), length)
        return tokens

    def test_pause_abort_resume_and_post_exit_readback(self):
        catalog = self.fixture.catalog
        self.wait_until(lambda s: s["states"].get("available", 0) > 0)
        for headers in ({}, {"Authorization": "Bearer wrong-key"}):
            response = requests.post(
                self.fixture.url + "/control_training_capture",
                json={"action": "abort"},
                headers=headers,
                timeout=10,
            )
            self.assertEqual(response.status_code, 401, response.text)
            self.assertFalse(self.state()["admission_paused"])
        baseline = self.generate(32)
        catalog.wait_publications(1)
        self.control("pause")
        before = self.state()["counters"]["admitted"]
        self.assertEqual(self.generate(32), baseline)
        self.assertEqual(self.state()["counters"]["admitted"], before)
        self.assertEqual(len(catalog.publications), 1)

        samples = [baseline]
        with ThreadPoolExecutor(max_workers=1) as executor:
            self.control("resume")
            self.wait_until(lambda s: s["states"].get("available", 0) > 0)
            pending = executor.submit(self.generate, 240)
            self.wait_until(lambda s: s["states"].get("active", 0) == 1)
            self.control("pause")
            samples.append(pending.result(timeout=120))
            catalog.wait_publications(2)
            self.assertEqual(self.generate(32), baseline)
            self.assertEqual(len(catalog.publications), 2)

            self.control("resume")
            self.wait_until(lambda s: s["states"].get("available", 0) > 0)
            pending = executor.submit(self.generate, 240)
            self.wait_until(lambda s: s["states"].get("active", 0) == 1)
            self.control("abort")
            self.assertEqual(pending.result(timeout=120), samples[1])
            self.wait_until(
                lambda s: not any(
                    count
                    for state, count in s["states"].items()
                    if state != "available"
                )
            )
            self.assertEqual(len(catalog.publications), 2)
            self.assertEqual(self.state()["counters"]["failed_operator_aborted"], 1)

        self.control("resume")
        self.wait_until(lambda s: s["states"].get("available", 0) > 0)
        samples.append(self.generate(32))
        self.assertEqual(samples[-1], baseline)
        publications = catalog.wait_publications(3)
        self.control("pause")
        for payload in ({"action": "invalid"}, {"action": True}, {}):
            response = requests.post(
                self.fixture.url + "/control_training_capture",
                json=payload,
                headers=self.headers,
                timeout=10,
            )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertTrue(self.state()["admission_paused"])
        final = self.wait_until(
            lambda s: not any(
                count for state, count in s["states"].items() if state != "available"
            )
        )
        self.assertEqual(final["host_pool"]["quarantined"], 0)
        self.assertFalse(catalog.errors)
        self.assertEqual(final["enable_overlap"], CUDA_GRAPH_OVERLAP)
        if CUDA_GRAPH_OVERLAP:
            self.assertGreater(final["counters"].get("cuda_graph_forwards", 0), 0)
            self.assertGreater(final["counters"].get("overlap_forwards", 0), 0)
        print(
            json.dumps(
                {"graph_overlap": CUDA_GRAPH_OVERLAP, "control_final_state": final}
            ),
            flush=True,
        )
        kill_process_tree(self.fixture.server.pid)
        self.fixture.server.wait(timeout=20)
        runtime.TestTrainingCaptureRuntime.server = None
        for publication, tokens in zip(publications, samples, strict=True):
            _, tensors = self.fixture.read_sample(publication)
            self.assertEqual(
                tensors["token_ids"].tolist(), [100, 200, 300, 400] * 4 + tokens
            )
            self.assertEqual(
                tensors["loss_mask"].tolist(), [0] * 16 + [1] * len(tokens)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=runtime.MODEL_PATH)
    parser.add_argument("--cuda-graph-overlap", action="store_true")
    args, remaining = parser.parse_known_args()
    runtime.MODEL_PATH = args.model_path
    CUDA_GRAPH_OVERLAP = args.cuda_graph_overlap
    unittest.main(argv=[__file__, *remaining])
