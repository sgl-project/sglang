"""Bind delta admission to published ranks and the ordinary updater state."""

import io
import sys
from types import SimpleNamespace

from sglang.srt.weight_sync.gpu_delta_session import (
    GpuDeltaSchedulerControl,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_describe_binds_published_scheduler_ranks_without_legacy_ps(monkeypatch):
    rank = 7
    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
    from sglang.srt.server_args import ServerArgs

    reset_context()
    try:
        # Use the current launch-time configuration/rank derivation, rather than
        # adding the removed `ps` field to a fake scheduler.
        publish(
            ServerArgs(
                model_path="dummy",
                tp_size=8,
                dp_size=8,
                ep_size=8,
                enable_dp_attention=True,
            ),
            role="test",
            ranks=SpawnRanks(world_rank=rank, dp_rank=rank),
        )
        runner = SimpleNamespace(
            model=object(),
            weight_updater=SimpleNamespace(
                _assert_weight_cache_inactive=lambda _: None
            ),
        )
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_lora = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.rust_server = None
        scheduler.tp_worker = SimpleNamespace(model_runner=runner)
        assert not hasattr(scheduler, "ps")

        plan = {"tensors": []}
        backends = []

        def make_backend(model_runner, identity):
            assert model_runner is runner
            backends.append(identity.copy())
            return SimpleNamespace(describe=lambda: plan)

        monkeypatch.setitem(
            sys.modules,
            "sglang.srt.weight_sync.gpu_delta_layout",
            SimpleNamespace(GpuDeltaBackend=make_backend),
        )
        monkeypatch.setattr(
            "sglang.srt.model_executor.model_runner_components.weight_updater."
            "_unsupported_derived_weight_cache_error",
            lambda _: None,
        )
        # Linux starttime field; comm deliberately contains spaces and ')'.
        stat = "123 (scheduler worker)) " + " ".join(["0"] * 19 + ["456"])
        monkeypatch.setattr(
            "sglang.srt.weight_sync.gpu_delta_session.open",
            lambda path: io.StringIO(stat),
            raising=False,
        )
        cache_engines = []

        def cache_id(engine_id):
            cache_engines.append(engine_id)
            return "shared-host-test"

        monkeypatch.setattr(
            "sglang.srt.weight_sync.gpu_delta_host.host_cache_id",
            cache_id,
        )
        control = GpuDeltaSchedulerControl(scheduler)
        receipt = control._describe("engine-0")
        assert receipt["identity"] | {
            "rank_id": "ignored",
            "hostname": "ignored",
            "pid": 0,
        } == {
            "engine_id": "engine-0",
            "rank_id": "ignored",
            "hostname": "ignored",
            "host_cache_id": "shared-host-test",
            "pid": 0,
            "start_ticks": 456,
            "tp_rank": rank,
            "dp_rank": rank,
            "pp_rank": 0,
        }
        assert receipt["state"] == "IDLE" and receipt["version"] == 0
        assert receipt["plan"] == plan
        assert control._describe("engine-0") == receipt
        assert len(backends) == 1
        assert cache_engines == ["engine-0"]
        control.session._executor.shutdown(wait=True)
    finally:
        reset_context()
