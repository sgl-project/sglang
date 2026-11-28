"""Exercise policy switching through a real single-GPU HTTP server.

Run this file directly for both normal and overlap scheduling. Set
SGLANG_TEST_MODEL to a local model path when reproducing outside CI.
"""

import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    ADMIN_HEADERS_FOR_TEST,
    ADMIN_LAUNCH_ARGS_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-small")


class PolicySwitchServerMixin:
    server_args = []

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            os.environ.get("SGLANG_TEST_MODEL", DEFAULT_SMALL_MODEL_NAME_FOR_TEST),
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--schedule-policy",
                "fcfs",
                "--mem-fraction-static",
                "0.7",
                "--chunked-prefill-size",
                "4096",
                "--disable-decode-cuda-graph",
                "--disable-prefill-cuda-graph",
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--enable-metrics",
                "--load-snapshot-publish-interval",
                "1",
                *ADMIN_LAUNCH_ARGS_FOR_TEST,
                *cls.server_args,
            ],
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, cls.process)

    def setUp(self):
        super().setUp()
        self.wait_idle()
        self.switch_policy("fcfs")
        # Warm the first prefill/decode before polling for a running blocker.
        self.check_generation(self.generate("warmup", 8), "warmup", 8)

    def post(self, path, payload, timeout=30):
        response = requests.post(
            self.base_url + path,
            json=payload,
            headers=ADMIN_HEADERS_FOR_TEST,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_load(self):
        response = requests.get(
            self.base_url + "/v1/loads",
            params={"include": "core"},
            timeout=10,
        )
        response.raise_for_status()
        loads = response.json()["loads"]
        self.assertEqual(len(loads), 1)
        return loads[0]

    def abort_requests(self, payload):
        # The abort endpoint acknowledges with an empty response body.
        response = requests.post(
            self.base_url + "/abort_request",
            json=payload,
            headers=ADMIN_HEADERS_FOR_TEST,
            timeout=30,
        )
        response.raise_for_status()

    def wait_load(self, predicate, *, unfinished=(), timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for future in unfinished:
                self.assertFalse(
                    future.done(),
                    "A request finished before the intended switch point",
                )
            load = self.get_load()
            if predicate(load):
                return load
            time.sleep(0.02)
        self.fail(f"Timed out waiting for the expected load: {load}")

    def wait_idle(self):
        self.wait_load(
            lambda load: load["num_running_reqs"] == load["num_waiting_reqs"] == 0
        )

    def switch_policy(self, name):
        result = self.post("/schedule_policy_switch", {"new_schedule_policy": name})
        self.assertTrue(result["success"], result)
        self.assertEqual(result["new_schedule_policy"], name)
        response = requests.get(self.base_url + "/server_info", timeout=10)
        response.raise_for_status()
        states = response.json()["internal_states"]
        self.assertEqual([state["schedule_policy"] for state in states], [name])
        return result

    def generate(self, rid, tokens, token=1000):
        return self.post(
            "/generate",
            {
                "rid": rid,
                "input_ids": [token] * 32,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "ignore_eos": True,
                },
            },
            timeout=180,
        )

    def check_generation(self, result, rid, tokens):
        meta = result["meta_info"]
        self.assertEqual(meta["id"], rid)
        self.assertEqual(meta["finish_reason"]["type"], "length", meta)
        self.assertEqual(meta["completion_tokens"], tokens)
        self.assertGreater(meta["forward_entry_time"], 0)
        return meta


class OrderedSwitchMixin:
    def check_order(self, *, restore_fcfs):
        prefix = self._testMethodName
        blocker_id = prefix + "-blocker"
        lengths = [("A", 16), ("B", 1024 if restore_fcfs else 128), ("C", 64)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            blocker = executor.submit(self.generate, blocker_id, 8192)
            futures = []
            try:
                self.wait_load(
                    lambda load: load["num_running_reqs"] == 1,
                    unfinished=[blocker],
                )
                for name, budget in lengths:
                    rid = prefix + "-" + name
                    future = executor.submit(self.generate, rid, budget)
                    futures.append((rid, budget, future))
                    self.wait_load(
                        lambda load: (
                            load["num_running_reqs"] == 1
                            and load["num_waiting_reqs"] == len(futures)
                        ),
                        unfinished=[blocker],
                    )

                # All three are queued while the original request keeps decoding.
                self.switch_policy("lof")
                self.assertFalse(blocker.done())
                self.abort_requests({"rid": blocker_id})
                self.assertEqual(
                    blocker.result(timeout=30)["meta_info"]["finish_reason"]["type"],
                    "abort",
                )
                if restore_fcfs:
                    self.wait_load(
                        lambda load: (
                            load["num_running_reqs"] == 1
                            and load["num_waiting_reqs"] == 2
                        ),
                        unfinished=[future for _, _, future in futures],
                    )
                    # B remains running; FCFS must restore the remaining A, C order.
                    self.switch_policy("fcfs")
                    self.assertFalse(any(future.done() for _, _, future in futures))

                entries = []
                for (name, _), (rid, budget, future) in zip(lengths, futures):
                    meta = self.check_generation(
                        future.result(timeout=120), rid, budget
                    )
                    entries.append((meta["forward_entry_time"], name))
                expected = ["B", "A", "C"] if restore_fcfs else ["B", "C", "A"]
                self.assertEqual([name for _, name in sorted(entries)], expected)
                self.assertEqual(len({stamp for stamp, _ in entries}), 3)
            finally:
                # Abort pending requests before the executor joins, including on failure.
                self.abort_requests({"abort_all": True})
        self.wait_idle()

    def test_switch_to_lof_orders_existing_waiting_requests(self):
        self.check_order(restore_fcfs=False)

    def test_switch_back_to_fcfs_preserves_running_request(self):
        self.check_order(restore_fcfs=True)


class TestPolicySwitchNormal(
    OrderedSwitchMixin, PolicySwitchServerMixin, CustomTestCase
):
    server_args = [
        "--disable-overlap-schedule",
        "--max-running-requests",
        "1",
        "--prefill-max-requests",
        "1",
    ]


class TestPolicySwitchOverlap(
    OrderedSwitchMixin, PolicySwitchServerMixin, CustomTestCase
):
    server_args = ["--max-running-requests", "1", "--prefill-max-requests", "1"]


class TestPolicySwitchConcurrent(PolicySwitchServerMixin, CustomTestCase):
    # Keep a backlog with several requests decoding, independent of GPU memory size.
    server_args = ["--max-running-requests", "4"]

    def test_switches_under_load_do_not_lose_requests(self):
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            for i in range(12):
                rid = f"concurrent-{i}"
                budget = 512 + 64 * (i % 4)
                future = executor.submit(self.generate, rid, budget, 1000 + i)
                futures.append((rid, budget, future))
            try:
                self.wait_load(
                    lambda load: (
                        load["num_running_reqs"] > 1 and load["num_waiting_reqs"] > 0
                    )
                )
                for policy in (
                    "lpm",
                    "lof",
                    "hrrn",
                    "shortest-prefill-first",
                    "dfs-weight",
                    "routing-key",
                    "random",
                    "uniboost",
                    "fcfs",
                ):
                    self.assertFalse(all(future.done() for _, _, future in futures))
                    self.switch_policy(policy)
                for rid, budget, future in futures:
                    self.check_generation(future.result(timeout=120), rid, budget)
            finally:
                self.abort_requests({"abort_all": True})
        self.wait_idle()


if __name__ == "__main__":
    unittest.main()
