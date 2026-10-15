"""The registry's cross-list queries in each mode: the running requests."""

import unittest
from types import SimpleNamespace

from sglang.srt.dllm.manager import DllmManager
from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import prefill_owner_shell
from sglang.test.scheduler_request_lists import make_request_registry
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _dllm(**lists):
    manager = DllmManager.__new__(DllmManager)
    manager.waiting, manager.staging, manager.between_blocks = [], [], []
    manager.__dict__.update(lists)
    return manager


def _hisparse(staging):
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.staging = staging
    return coordinator


def _req(rid, lora_id=None):
    return SimpleNamespace(rid=rid, lora_id=lora_id)


# Each mode's lists of launched requests, by the owner holding them.
_RUNNING = {
    "dllm_staging": lambda req: dict(dllm=_dllm(staging=[req])),
    "dllm_between_blocks": lambda req: dict(dllm=_dllm(between_blocks=[req])),
    "hisparse_staging": lambda req: dict(hisparse_coordinator=_hisparse([req])),
    "pd_prefill_prefilling": lambda req: dict(
        disagg_prefill=prefill_owner_shell(prefilling=[req])
    ),
}


class TestRunningRequests(CustomTestCase):
    def test_each_mode_counts_its_launched_requests_as_running(self):
        for name, owners in _RUNNING.items():
            with self.subTest(name):
                req = _req("r")
                registry = make_request_registry(mode_owners=ModeOwners(**owners(req)))
                self.assertEqual(registry.running_reqs(), [req])


if __name__ == "__main__":
    unittest.main()
