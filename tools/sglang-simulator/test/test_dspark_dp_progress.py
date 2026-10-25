"""Joint causal protocols only; no native performance or qualified row fixture."""

import json
import sys
from dataclasses import replace
from types import ModuleType

import pytest
from sglang_simulator.simulation.sglang.dspark import (
    DSparkGroup,
    DSparkPhaseCost,
    DSparkSession,
    DSparkStep,
    DSparkStepTiming,
    DSparkTopology,
)
from sglang_simulator.time_predictor import dspark_golden_progress as golden
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_step import Cost, request, session

RUNTIME = '{"scope":"explicit CPU joint-cost context, not runtime qualification"}'


@pytest.fixture
def policy(monkeypatch):
    # Isolate the algorithm from the SDK's separate real receipt reader.
    class SourcePolicy:
        golden_law = golden.LAW
        sampler_capacities = (64, 64)
        source_identity_sha256 = "a" * 64

    module = ModuleType("infercast.sdk.native_dp_progress")
    module.NativeDPProgressPolicy = SourcePolicy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return SourcePolicy()


def before(counts=(1, 2), *, fid=0, mode="TARGET_VERIFY"):
    groups = []
    for rank, batch in enumerate(counts):
        requests = []
        for index in range(batch):
            maximum = 2 + rank * 10 + index
            recipe = json.dumps(
                {"ignore_eos": True, "is_normalized": True, "max_new_tokens": maximum},
                sort_keys=True,
                separators=(",", ":"),
            )
            requests.append(
                replace(
                    request(f"dp{rank}-{index}", prefix=512 + index * 37),
                    max_new_tokens=maximum,
                    sampling_params_json=recipe,
                )
            )
        # Cold independently in both groups. A peer's warm buffer is not a seed
        # for this local buffer; cost preparation must retain the full vector.
        groups.append(
            DSparkGroup(
                rank,
                mode if batch else "IDLE",
                tuple(requests),
                '{"draft_buffer_capacity":0,"draft_buffer_device_matches":false}',
            )
        )
    return DSparkStep(fid, DSparkTopology(2, 2), tuple(groups), RUNTIME)


class JointCost:
    calls = 0

    def predict_step(self, step):
        self.calls += 1
        # An explicit existing CPU-protocol cost, never an accepted latency.
        return DSparkStepTiming(
            step.forward_id,
            (DSparkPhaseCost("whole_worker", 0.01, "CPU whole-worker protocol"),),
        )


@pytest.mark.parametrize("counts", [(1, 0), (0, 1), (1, 2), (3, 4), (64, 64)])
def test_active_local_draw_shared_over_local_batch_idle_draws_nothing(
    policy, monkeypatch, counts
):
    model = golden.SourceGoldenDPProgress(
        policy, runtime_profile_json=RUNTIME, model_seed=42
    )
    step = before(counts)
    sha = golden.hashlib.sha256
    calls = []

    def counted(value):
        calls.append(json.loads(value))
        return sha(value)

    monkeypatch.setattr(golden.hashlib, "sha256", counted)
    result = model.predict_progress(step)
    assert len(calls) == sum(bool(n) for n in counts)
    assert [row["attention_dp_rank"] for row in calls] == [
        r for r, n in enumerate(counts) if n
    ]
    commits = dict(result.committed_tokens)
    assert tuple(commits) == tuple(req.rid for req in step.requests)
    for group in step.groups:
        assert len({commits[r.rid] for r in group.requests}) == int(
            bool(group.requests)
        )
    assert model.predict_progress(step) == result
    assert (
        model.identity["joint_assumption"]
        == "independent_local_hashes_not_native_rng_replay"
    )


def test_local_hashes_can_differ_without_claiming_native_rng_independence(policy):
    model = golden.SourceGoldenDPProgress(
        policy, runtime_profile_json=RUNTIME, model_seed=42
    )
    pairs = [
        tuple(
            value
            for _, value in model.predict_progress(
                before((1, 1), fid=i)
            ).committed_tokens
        )
        for i in range(100)
    ]
    assert any(a != b for a, b in pairs)
    assert set(pairs) <= {(3, 3), (3, 4), (4, 3), (4, 4)}


@pytest.mark.parametrize("counts", [(1, 0), (0, 1), (1, 2), (64, 64)])
def test_joint_session_costs_once_then_commits_every_request_atomically(policy, counts):
    model = golden.SourceGoldenDPProgress(
        policy, runtime_profile_json=RUNTIME, model_seed=42
    )
    cost = JointCost()
    cost.calls = 0
    step = before(counts)
    run = session(step, cost=cost, progress=model)
    old = dict(run._history)
    result = run.predict(step, local_attention_dp_rank=1)
    assert cost.calls == 1 and result.timing.seconds == 0.01
    assert run._history == old
    # Both local views share one global-FID cost/result; do not query twice.
    peer = replace(result, local_attention_dp_rank=0)
    assert peer.step is result.step and peer.timing is result.timing
    run.commit(result)
    for req, (_, commit) in zip(
        step.requests, result.progress.committed_tokens, strict=True
    ):
        assert (
            run.history_for(req.rid).target_prefix_len == req.target_prefix_len + commit
        )
    with pytest.raises(PredictorError, match="already committed"):
        run.commit(peer)
    assert cost.calls == 1


@pytest.mark.parametrize(
    "failure",
    ["cost", "last-recipe", "last-width", "last-emits", "missing-group", "mixed"],
)
def test_joint_failures_never_draw_or_modify_causal_history(
    policy, monkeypatch, failure
):
    model = golden.SourceGoldenDPProgress(
        policy, runtime_profile_json=RUNTIME, model_seed=42
    )
    step = before((1, 2))
    if failure == "missing-group":
        with pytest.raises(PredictorError, match="every actual attention-DP"):
            replace(step, groups=step.groups[:1])
        return
    if failure == "mixed":
        step = replace(
            step,
            groups=(replace(step.groups[0], forward_mode="EXTEND"), step.groups[1]),
        )
    elif failure in {"last-recipe", "last-width", "last-emits"}:
        last = step.requests[-1]
        if failure == "last-recipe":
            recipe = json.loads(last.sampling_params_json)
            recipe["ignore_eos"] = False
            last = replace(
                last,
                sampling_params_json=json.dumps(
                    recipe, sort_keys=True, separators=(",", ":")
                ),
            )
        elif failure == "last-width":
            last = replace(last, extend_len=5)
        else:
            last = replace(last, emits_output=False)
        step = replace(
            step,
            groups=(
                step.groups[0],
                replace(
                    step.groups[1], requests=step.groups[1].requests[:-1] + (last,)
                ),
            ),
        )

    def no_draw(_):
        pytest.fail("invalid/cost-missing joint work drew progress")

    monkeypatch.setattr(golden.hashlib, "sha256", no_draw)

    class MissingCost:
        def predict_step(self, _):
            raise PredictorError("data_unavailable", "missing whole-worker cost")

    run = session(
        step, cost=MissingCost() if failure == "cost" else JointCost(), progress=model
    )
    old = dict(run._history)
    with pytest.raises(PredictorError):
        run.predict(step)
    assert run._history == old and run._last_forward_id == -1


@pytest.mark.parametrize("kind", ["ADP1", "extra-phase", "missing-phase", "wrong-fid"])
def test_whole_worker_cost_is_exclusive_and_cannot_substitute_partial_boundaries(kind):
    step = before()
    cost = JointCost().predict_step(step)
    if kind == "ADP1":
        step = replace(step, topology=DSparkTopology(2, 1), groups=step.groups[:1])
    elif kind == "extra-phase":
        cost = replace(
            cost,
            phases=cost.phases + (DSparkPhaseCost("proposal", 0.01, "CPU partial"),),
        )
    elif kind == "missing-phase":
        cost = replace(cost, phases=(DSparkPhaseCost("proposal", 0.01, "CPU partial"),))
    else:
        cost = replace(cost, forward_id=1)
    with pytest.raises(PredictorError):
        cost.validate_for(step)


def test_prefill_preserves_local_zero_one_and_never_draws_decode(policy):
    class NoDecode:
        def predict_progress(self, _):
            pytest.fail("prefill drew decode progress")

    step = before((2, 0), mode="EXTEND")
    requests = tuple(
        replace(r, draft_valid_from=r.target_prefix_len, emits_output=bool(i))
        for i, r in enumerate(step.requests)
    )
    step = replace(
        step, groups=(replace(step.groups[0], requests=requests), step.groups[1])
    )
    run = DSparkSession(JointCost(), NoDecode())
    result = run.execute(step)
    assert result.progress.committed_tokens == (
        (requests[0].rid, 0),
        (requests[1].rid, 1),
    )


def test_old_adp1_phase_cost_protocol_remains_unchanged():
    from test_dspark_step import step

    before_step = step()
    original = Cost().predict_step(before_step)
    assert original.validate_for(before_step) is original
    assert original.seconds == pytest.approx(0.05)


@pytest.mark.parametrize("invalid", [True, {}, None])
def test_caller_approval_or_adp1_policy_cannot_replace_singleton_source(
    policy, invalid
):
    with pytest.raises(PredictorError, match="SDK-verified singleton"):
        golden.SourceGoldenDPProgress(
            invalid, runtime_profile_json=RUNTIME, model_seed=42
        )
