from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=31, suite="base-a-test-cpu")

import multiprocessing
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch.distributed as dist

from sglang.srt.distributed import gated_launch
from sglang.srt.distributed.gated_launch import (
    POLL_INTERVAL_SECONDS,
    _GatedLaunchServer,
    _wait_until_activated,
    maybe_wait_for_gated_launch,
)
from sglang.srt.utils.network import get_open_port
from sglang.test.test_utils import CustomTestCase

_JOIN_TIMEOUT_SECONDS = 120


def _fake_world_group(*, rank: int, world_size: int, cpu_group=None):
    return SimpleNamespace(
        rank=rank,
        rank_in_group=rank,
        ranks=list(range(world_size)),
        world_size=world_size,
        cpu_group=cpu_group,
    )


def _activate_over_http(base_url: str, delay: float):
    time.sleep(delay)
    deadline = time.perf_counter() + 30
    while time.perf_counter() < deadline:
        try:
            if requests.post(f"{base_url}/gate/activate", timeout=1).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.1)


def _gated_launch_worker(
    rank: int, dist_port: int, gate_port: int, activate_after: float, out
):
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{dist_port}",
        rank=rank,
        world_size=2,
    )
    try:
        if rank == 0:
            threading.Thread(
                target=_activate_over_http,
                args=(f"http://127.0.0.1:{gate_port}", activate_after),
                daemon=True,
            ).start()

        world_group = _fake_world_group(
            rank=rank, world_size=2, cpu_group=dist.group.WORLD
        )
        started_at = time.perf_counter()
        with patch.object(gated_launch, "get_world_group", return_value=world_group):
            maybe_wait_for_gated_launch(host="127.0.0.1", port=gate_port)
        out.put((rank, time.perf_counter() - started_at))
    finally:
        dist.destroy_process_group()


class TestGatedLaunchServer(CustomTestCase):
    def setUp(self):
        self.server = _GatedLaunchServer()
        self.port = get_open_port()
        self.server.serve(host="127.0.0.1", port=self.port)
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._wait_until_listening()

    def _wait_until_listening(self):
        deadline = time.perf_counter() + 30
        while time.perf_counter() < deadline:
            try:
                requests.get(f"{self.base_url}/health", timeout=1)
                return
            except requests.exceptions.RequestException:
                time.sleep(0.1)
        self.fail(f"control server did not start listening on port {self.port}")

    def test_health_is_served_while_the_gate_is_still_closed(self):
        """The control port answers before activation so a caller can find it."""
        response = requests.get(f"{self.base_url}/health", timeout=5)

        self.assertEqual(response.status_code, 200)
        self.assertFalse(self.server.activated)

    def test_activate_flips_the_flag_and_stays_successful_when_repeated(self):
        """A retried activation still succeeds instead of erroring or toggling back."""
        first = requests.post(f"{self.base_url}/gate/activate", timeout=5)
        self.assertEqual(first.status_code, 200)
        self.assertTrue(self.server.activated)

        second = requests.post(f"{self.base_url}/gate/activate", timeout=5)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(self.server.activated)

    def test_activating_one_server_leaves_another_one_closed(self):
        """The route acts on its own server instead of process wide state."""
        other = _GatedLaunchServer()
        other_port = get_open_port()
        other.serve(host="127.0.0.1", port=other_port)

        requests.post(f"{self.base_url}/gate/activate", timeout=5)

        self.assertTrue(self.server.activated)
        self.assertFalse(other.activated)


class TestWaitUntilActivated(CustomTestCase):
    def test_single_rank_keeps_polling_until_the_flag_is_set(self):
        """A lone rank leaves the gate only after its own flag flips."""
        server = _GatedLaunchServer()
        activate_after = 2 * POLL_INTERVAL_SECONDS
        threading.Timer(
            activate_after, lambda: setattr(server, "activated", True)
        ).start()

        started_at = time.perf_counter()
        _wait_until_activated(
            world_group=_fake_world_group(rank=0, world_size=1), server=server
        )
        elapsed = time.perf_counter() - started_at

        self.assertGreaterEqual(elapsed, activate_after)

    def test_second_rank_learns_about_activation_through_the_cpu_group(self):
        """Driven through maybe_wait_for_gated_launch: the rank without the control server is released by the gloo broadcast."""
        context = multiprocessing.get_context("spawn")
        out = context.Queue()
        dist_port = get_open_port()
        gate_port = get_open_port()
        activate_after = 2 * POLL_INTERVAL_SECONDS

        processes = [
            context.Process(
                target=_gated_launch_worker,
                args=(rank, dist_port, gate_port, activate_after, out),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()

        elapsed_by_rank = {}
        for _ in processes:
            rank, elapsed = out.get(timeout=_JOIN_TIMEOUT_SECONDS)
            elapsed_by_rank[rank] = elapsed

        for process in processes:
            process.join(timeout=_JOIN_TIMEOUT_SECONDS)
        for process in processes:
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(sorted(elapsed_by_rank), [0, 1])
        self.assertGreaterEqual(elapsed_by_rank[1], activate_after)


class TestMaybeWaitForGatedLaunch(CustomTestCase):
    def setUp(self):
        self.addCleanup(setattr, gated_launch, "_instance", None)
        gated_launch._instance = None

    def test_an_unset_port_leaves_the_startup_path_untouched(self):
        """Without the flag the gate never reaches the distributed environment."""
        with patch.object(gated_launch, "get_world_group") as get_world_group:
            maybe_wait_for_gated_launch(host="127.0.0.1", port=None)

        get_world_group.assert_not_called()
        self.assertIsNone(gated_launch._instance)

    def test_a_second_call_in_the_same_process_does_not_gate_again(self):
        """A draft worker re-entering the init path must not wait a second time."""
        gated_launch._instance = _GatedLaunchServer()

        with patch.object(gated_launch, "get_world_group") as get_world_group:
            maybe_wait_for_gated_launch(host="127.0.0.1", port=get_open_port())

        get_world_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
