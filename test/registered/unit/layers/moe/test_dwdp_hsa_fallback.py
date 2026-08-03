import pytest

from sglang.srt.layers.moe.dwdp.rocm_ipc import RocmIpcDwdpManager
from sglang.srt.layers.moe.dwdp.weight_manager import DWDPWeightManager
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=1, suite="base-c-test-cpu")


class _FakeCopyEngine:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def wait_all(self, tickets):
        self.calls.append(tuple(tickets))
        if self.fail:
            raise RuntimeError("injected drain failure")


@pytest.mark.parametrize("manager_type", [RocmIpcDwdpManager, DWDPWeightManager])
@pytest.mark.parametrize("drain_fails", [False, True])
def test_hsa_fallback_drains_other_slot(manager_type, drain_fails):
    manager = manager_type.__new__(manager_type)
    engine = _FakeCopyEngine(fail=drain_fails)
    manager._copy_engine = engine
    manager._copy_tickets = [[11, 12], [21, 22]]

    manager._disable_hsa_copy_engine(completed_slot=0)

    assert engine.calls == [(21, 22)]
    assert manager._copy_tickets == [[], []]
    assert manager._copy_engine is None
