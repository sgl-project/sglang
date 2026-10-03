"""The HTTP resume certificate must survive the scheduler IPC hop."""

import io
import sys
from types import SimpleNamespace

import pytest
from pydantic import TypeAdapter
from sglang.srt.managers.io_struct import (
    PrepareWeightsFromDeltaReqInput,
    ResumeWeightsFromDeltaReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.weight_sync.gpu_delta_session import (
    GpuDeltaSchedulerControl,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("rank", [0, 7])
def test_describe_binds_published_scheduler_ranks_without_legacy_ps(rank, monkeypatch):
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
        monkeypatch.setattr(
            "sglang.srt.weight_sync.gpu_delta_host.host_cache_id",
            lambda: "shared-host-test",
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
        with pytest.raises(ValueError, match="already bound"):
            control._describe("different-engine")
        control.session._executor.shutdown(wait=True)
    finally:
        reset_context()


def test_resume_certificate_survives_http_validation_and_ipc():
    receipts = [
        {
            "identity": {"engine_id": "engine-0", "rank_id": "original-rank-0"},
            "state": "APPLIED",
            "session_id": "publication-1",
            "target_version": 1,
        }
    ]
    request = TypeAdapter(ResumeWeightsFromDeltaReqInput).validate_python(
        {"session_id": "publication-1", "receipts": receipts}
    )
    received = msgpack_decode(msgpack_encode(request))
    assert isinstance(received, ResumeWeightsFromDeltaReqInput)
    assert received.session_id == "publication-1"
    assert received.receipts == receipts


@pytest.mark.parametrize(
    "legacy_session,offloaded", [(False, False), (True, False), (False, True)]
)
def test_prepare_uses_scheduler_updater_session_and_offload_state(
    legacy_session, offloaded
):
    who = {"engine_id": "engine-0", "rank_id": "original-0"}
    control = GpuDeltaSchedulerControl(
        SimpleNamespace(
            weight_updater=SimpleNamespace(
                _session=object() if legacy_session else None,
                offload_tags={"weights"} if offloaded else set(),
            )
        )
    )
    control.identity = who
    prepared = []

    def prepare(request):
        prepared.append(request)
        return {
            "identity": who,
            "session_id": request["session_id"],
            "state": "PREPARING",
        }

    def status(session_id):
        # Rejected before DeltaSession.prepare, so no session exists to inspect.
        raise ValueError("unknown delta session")

    control.session = SimpleNamespace(prepare=prepare, status=status)
    request = PrepareWeightsFromDeltaReqInput(
        session_id="publication-1",
        engine_id=who["engine_id"],
        manifest_path="/immutable/manifest.json",
        manifest_sha256="a" * 64,
        stream_id="run-1",
        base_version=0,
        target_version=1,
        plan_digest="b" * 64,
        participants=[who],
        host_tensor_names={"host": []},
    )
    result = control.handle(request)
    assert result.success is not (legacy_session or offloaded)
    assert bool(prepared) is result.success
    if not result.success:
        assert "another weight update or memory offload" in result.message
