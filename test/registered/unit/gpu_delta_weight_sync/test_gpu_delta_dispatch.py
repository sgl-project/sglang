"""CPU coverage of delta lease controls around the ordinary scheduler dispatcher."""

from array import array
from types import SimpleNamespace

import pytest
from sglang.srt.managers import io_struct as io
from sglang.srt.weight_sync import gpu_delta_session as delta_runtime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.utils import TypeBasedDispatcher

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.fixture
def dispatch(monkeypatch):
    events = []
    scheduler = SimpleNamespace(_engine_paused=False)

    def generate(request):
        events.append("generate")
        return request

    def pause(request):
        events.append(("reclaim", request.mode))
        scheduler._engine_paused = True
        return "paused"

    def resume(request):
        events.append(("resume", request))
        scheduler._engine_paused = False
        return "resumed"

    def mutate(request):
        events.append(("mutate", request))
        return "mutated"

    scheduler.continue_generation = resume
    control = delta_runtime.GpuDeltaSchedulerControl(scheduler)
    monkeypatch.setattr(
        delta_runtime, "GpuDeltaSchedulerControl", lambda owner: control
    )
    ordinary = TypeBasedDispatcher(
        [
            (io.TokenizedGenerateReqInput, generate),
            (io.PauseGenerationReqInput, pause),
            (io.ContinueGenerationReqInput, resume),
            (io.UpdateWeightFromDiskReqInput, mutate),
            (io.ReleaseMemoryOccupationReqInput, mutate),
        ]
    )
    return SimpleNamespace(
        scheduler=scheduler,
        control=control,
        events=events,
        ordinary=ordinary,
        wrapped=delta_runtime.with_gpu_delta_controls(scheduler, ordinary),
    )


def test_generation_handler_and_ordinary_pause_resume_are_preserved(dispatch):
    request = io.TokenizedGenerateReqInput(
        input_text="hello",
        input_ids=array("i", [1]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=None,  # Opaque to dispatch; no generation executes here.
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )
    # Calling through a wrapper could preserve outputs while regressing every
    # generation dispatch. The registered callable itself must stay unchanged.
    assert dispatch.wrapped._mapping[io.TokenizedGenerateReqInput] is (
        dispatch.ordinary._mapping[io.TokenizedGenerateReqInput]
    )
    assert dispatch.wrapped(request) is request
    assert dispatch.wrapped(io.PauseGenerationReqInput(mode="retract")) == "paused"
    assert dispatch.scheduler._engine_paused
    resume = io.ContinueGenerationReqInput(torch_empty_cache=True)
    assert dispatch.wrapped(resume) == "resumed"
    assert not dispatch.scheduler._engine_paused
    assert dispatch.events == ["generate", ("reclaim", "retract"), ("resume", resume)]


def test_competing_mutations_cannot_reach_ordinary_handlers_under_lease(dispatch):
    dispatch.control.session = SimpleNamespace(leased=True)
    rejected = dispatch.wrapped(io.UpdateWeightFromDiskReqInput(model_path="/next"))
    assert isinstance(rejected, io.UpdateWeightFromDiskReqOutput)
    assert not rejected.success
    assert "owns the model" in rejected.message
    # Empty-ACK APIs cannot report success for a refused mutation. The tokenizer
    # normally filters these; the scheduler still enforces the invariant.
    with pytest.raises(RuntimeError, match="bypassed.*dispatch guard"):
        dispatch.wrapped(io.ReleaseMemoryOccupationReqInput(tags=["weights"]))
    assert dispatch.events == []


def test_failed_reader_fence_preserves_pause_and_blocks_plain_resume(dispatch):
    def fence():
        assert dispatch.scheduler._engine_paused
        dispatch.events.append("reader fence")
        raise RuntimeError("readers still active")

    dispatch.scheduler.device_module = SimpleNamespace(synchronize=fence)
    dispatch.control.session = SimpleNamespace(
        leased=True, quiesce=lambda synchronize: synchronize()
    )
    dispatch.wrapped(io.PauseGenerationReqInput(mode="retract"))
    assert dispatch.scheduler._engine_paused
    assert dispatch.events == ["reader fence"]  # No ordinary KV reclamation.

    dispatch.wrapped(io.ContinueGenerationReqInput())
    dispatch.wrapped(
        io.ContinueGenerationReqInput(
            delta_session_id="publication-1",
            delta_commit_receipts=[{"state": "COMMITTED"}],
        )
    )
    assert dispatch.scheduler._engine_paused
    assert dispatch.events == ["reader fence"]


def test_certified_continue_authorizes_before_ordinary_resume(dispatch):
    dispatch.scheduler._engine_paused = True
    receipts = [{"state": "COMMITTED", "certificate": "all-original-ranks"}]

    def authorize(session_id, actual_receipts):
        assert session_id == "publication-1"
        assert actual_receipts == receipts
        assert dispatch.scheduler._engine_paused
        dispatch.events.append("authorized")

    def resumed(session_id):
        assert session_id == "publication-1"
        assert not dispatch.scheduler._engine_paused
        dispatch.events.append("receipt")
        return {"state": "RESUMED"}

    dispatch.control.session = SimpleNamespace(
        leased=True, authorize_resume=authorize, resumed=resumed
    )
    result = dispatch.wrapped(
        io.ContinueWeightsFromDeltaReqInput(
            rid="continue-1", session_id="publication-1", receipts=receipts
        )
    )
    assert result.success and result.rid == "continue-1"
    assert result.participant["state"] == "RESUMED"
    authorized, (operation, ordinary_request), recorded = dispatch.events
    assert (authorized, operation, recorded) == ("authorized", "resume", "receipt")
    assert isinstance(ordinary_request, io.ContinueGenerationReqInput)
    assert ordinary_request.delta_session_id == "publication-1"
    assert ordinary_request.torch_empty_cache is False
