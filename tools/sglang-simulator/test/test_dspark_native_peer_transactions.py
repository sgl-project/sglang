"""Existing runtime/reference transactions over a bounded CPU transport double.

Fixture timing/progress and transport are protocol controls, not native costs,
source attestation or proof that the public ADP2 factory/clock is enabled.
"""

import copy
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.simulation.sglang.dspark import (
    STEP_ATTRIBUTE,
    DSparkPhaseCost,
    DSparkStepTiming,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_reference_driver import (
    DSparkReferenceDriver,
)
from sglang_simulator.simulation.sglang.dspark_runtime import canonical
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_dp_progress import policy as policy_fixture
from test_dspark_reference import native_page_sizing
from test_dspark_runtime import batch
from test_dspark_runtime import runtime as runtime_fixture

pytestmark = pytest.mark.usefixtures(native_page_sizing.__name__)


@pytest.fixture
def owners(monkeypatch):
    condition = threading.Condition()
    exchanges = {}
    events = []

    class NativeCPUGroup:
        world_size = 2
        cpu_group = object()

        def __init__(self, rank):
            self.ranks = [0, 1]
            self.rank_in_group = rank
            self.index = 0

        def all_gather_object(self, value):
            with condition:
                index = self.index
                self.index += 1
                rows = exchanges.setdefault(index, {})
                assert self.rank_in_group not in rows
                rows[self.rank_in_group] = copy.deepcopy(value)
                condition.notify_all()
                assert condition.wait_for(lambda: len(rows) == 2, timeout=3), (
                    index,
                    value,
                )
                return copy.deepcopy([rows[0], rows[1]])

    module = ModuleType("sglang.srt.distributed.parallel_state")
    module.GroupCoordinator = NativeCPUGroup
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runs = []
    for rank in range(2):
        runtime = runtime_fixture.__wrapped__(monkeypatch)
        runtime.topology = DSparkTopology(2, 2)
        runtime.reference_driver = DSparkReferenceDriver(runtime)

        def execution(*, group, reference_view, **kwargs):
            assert reference_view.local_attention_dp_rank == group.attention_dp_rank
            return {"local_reference_owner": group.attention_dp_rank}

        runtime.execution_resolver = execution
        runtime.session.cost_model.native_source_policy = NS(
            reference_needs_cpu_seq_lens=False
        )

        def cost(step, *, rank=rank, runtime=runtime):
            runtime.session.cost_model.calls += 1
            events.append(("cost", rank, step.forward_id))
            return DSparkStepTiming(
                step.forward_id,
                (DSparkPhaseCost("whole_worker", 0.01, "CPU protocol fixture"),),
            )

        runtime.session.cost_model.predict_step = cost
        runs.append(
            NS(
                runtime=runtime,
                driver=runtime.reference_driver,
                scheduler=NS(
                    enable_overlap=True,
                    future_map=NS(needs_cpu_seq_lens=False),
                    token_to_kv_pool_allocator=NS(page_size=256),
                ),
                group=NativeCPUGroup(rank),
                rank=rank,
                events=events,
            )
        )
    return runs


def both(owners, call):
    def invoke(run):
        try:
            return call(run)
        except BaseException as error:  # noqa: BLE001 - assert both peer outcomes
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke, run) for run in owners]
        return [future.result(timeout=8) for future in futures]


def passed(values):
    assert not any(isinstance(value, BaseException) for value in values), values
    return values


def native_batch(run, mode="EXTEND", *, size=1, prefix=0, extend=512, requests=None):
    if mode == "IDLE":
        return NS(
            device="cpu",
            forward_mode=NS(name=mode),
            reqs=[],
            seq_lens=None,
            seq_lens_cpu=None,
            spec_info=None,
        )
    parts = [batch(mode, prefix=prefix + i * 7, extend=extend) for i in range(size)]
    current = parts[0]
    current.reqs = [part.reqs[0] for part in parts] if requests is None else requests
    current.chunked_req = None
    current.token_to_kv_pool_allocator = NS(page_size=256)
    current.req_pool_indices = list(range(1, len(current.reqs) + 1))
    current.spec_info = None
    if mode == "EXTEND":
        current.prefix_lens = [prefix + i * 7 for i in range(size)]
        current.extend_lens = [extend] * size
        current.seq_lens = current.seq_lens_cpu = [
            p + extend for p in current.prefix_lens
        ]
        for index, req in enumerate(current.reqs):
            req.rid = f"dp{run.rank}-{index}"
            req.done = False
            req.finished = lambda req=req: req.done
            req.is_retracted = False
            req.kv.kv_committed_len = current.seq_lens[index]
    else:
        rows = [run.driver.adapter.ledger.request_for(req.rid) for req in requests]
        for req, row in zip(requests, rows, strict=True):
            req.kv.kv_committed_len = row.kv_committed_len
        current.seq_lens = current.seq_lens_cpu = [row.kv_committed_len for row in rows]
        from sglang_simulator.simulation.sglang.dspark_reference import (
            _decode_alloc_lens,
        )

        _, following, _ = _decode_alloc_lens(tuple(rows), reserve=12, page_size=256)
        current.spec_info = NS(future_indices=None, nxt_kv_lens_cpu=following)
    return current


def select(owners, batches, *, sync=(False, False)):
    def call(run):
        step = run.runtime.select_native_dp_peer_batch(
            run.scheduler,
            batches[run.rank],
            tp_group=run.group,
            attention_dp_rank=run.rank,
            native_forward_ct=100 + run.runtime._forward_id,
        )
        run.driver.overlap_decision(batches[run.rank], sync[run.rank])
        return step

    return both(owners, call)


def resolve(*, step, reference_views, local_executions):
    assert tuple(view.local_attention_dp_rank for view in reference_views) == (0, 1)
    assert tuple(row["local_reference_owner"] for row in local_executions) == (0, 1)
    # Explicit protocol-only resolver. A future source-led dynamic joint
    # execution projector is still required before public ADP2 activation.
    execution = canonical({"joint": local_executions})
    return replace(
        step,
        groups=tuple(replace(group, execution_json=execution) for group in step.groups),
    )


def forward(owners, batches, *, resolver=resolve, publish=None):
    def call(run):
        return run.runtime.forward_native_dp_peer(
            worker=NS(query_token_num=5, verify_num_draft_tokens=6),
            batch=batches[run.rank],
            tp_group=run.group,
            attention_dp_rank=run.rank,
            native_forward_ct=100 + run.runtime._forward_id,
            joint_execution_resolver=resolver,
            on_publish=(
                (lambda value: publish(run, value)) if publish is not None else None
            ),
        )

    return both(owners, call)


def process(run, current, result, *, finish=False):
    run.driver.before_result(current, result)
    prediction = getattr(result, STEP_ATTRIBUTE)
    counts = dict(prediction.progress.committed_tokens)
    for req in current.reqs:
        req.output_ids.extend([1] * counts[req.rid])
        if prediction.local_group.forward_mode == "TARGET_VERIFY":
            req.kv.kv_committed_len += counts[req.rid]
        req.done = finish
    run.driver.after_result()


def untouched(owners):
    for run in owners:
        assert run.runtime._forward_id == run.runtime._draft_buffer_capacity == 0
        assert run.runtime.session._history == {}
        assert run.runtime.session._last_forward_id == -1


def test_two_local_reference_owners_build_then_publish_one_joint_result(owners):
    batches = [native_batch(run, size=run.rank + 1) for run in owners]
    selected = passed(select(owners, batches))
    assert selected[0] == selected[1]
    for run in owners:
        original = run.runtime.results.build

        def build(*, batch, prediction, run=run, original=original):
            run.events.append(("build", run.rank, prediction.step.forward_id))
            return original(batch=batch, prediction=prediction)

        run.runtime.results.build = build

    def publish(run, value):
        assert {event[1] for event in run.events if event[0] == "build"} == {0, 1}
        assert all(peer.runtime.session._history == {} for peer in owners)
        run.events.append(("publish", run.rank, tuple(value)))

    results = passed(forward(owners, batches, publish=publish))
    predictions = [getattr(result, STEP_ATTRIBUTE) for result in results]
    assert predictions[0].step == predictions[1].step
    assert predictions[0].timing is not None
    assert [prediction.local_attention_dp_rank for prediction in predictions] == [0, 1]
    assert [len(result.new_seq_lens) for result in results] == [1, 2]
    assert [run.runtime.session.cost_model.calls for run in owners] == [1, 0]
    assert [run.runtime.session.progress_model.calls for run in owners] == [0, 0]
    assert owners[0].runtime.session._history == owners[1].runtime.session._history
    assert len(owners[0].runtime.session._history) == 3
    for run in owners:
        assert run.runtime._forward_id == 1
        assert run.runtime._draft_buffer_capacity == 0
        assert len(run.driver.adapter.ledger.requests) == run.rank + 1
        process(run, batches[run.rank], results[run.rank])

    decode = [
        native_batch(run, "DECODE", requests=batches[run.rank].reqs) for run in owners
    ]
    passed(select(owners, decode))
    results = passed(forward(owners, decode))
    assert [run.runtime.session.cost_model.calls for run in owners] == [2, 0]
    assert [run.runtime.session.progress_model.calls for run in owners] == [1, 0]
    assert [run.runtime._draft_buffer_capacity for run in owners] == [1, 2]
    for run, result in zip(owners, results, strict=True):
        process(run, decode[run.rank], result)
        assert run.driver.adapter.ledger.pending == ()
    assert owners[0].runtime.session._history == owners[1].runtime.session._history


@pytest.mark.parametrize("idle_rank", [0, 1])
def test_idle_participates_without_local_allocation_or_peer_buffer_warmth(
    owners, idle_rank
):
    batches = [
        native_batch(run, "IDLE" if run.rank == idle_rank else "EXTEND", size=2)
        for run in owners
    ]
    passed(select(owners, batches))
    results = passed(forward(owners, batches))
    for run in owners:
        process(run, batches[run.rank], results[run.rank])
    decode = [
        native_batch(
            run,
            "IDLE" if run.rank == idle_rank else "DECODE",
            requests=batches[run.rank].reqs,
        )
        for run in owners
    ]
    passed(select(owners, decode))
    results = passed(forward(owners, decode))
    assert results[idle_rank].new_seq_lens == []
    assert owners[idle_rank].runtime._draft_buffer_capacity == 0
    assert owners[1 - idle_rank].runtime._draft_buffer_capacity == 2
    assert owners[idle_rank].driver.adapter.ledger.requests == ()
    assert len(owners[idle_rank].runtime.session._history) == 2
    assert [run.runtime.session.progress_model.calls for run in owners] == [1, 0]


@pytest.mark.parametrize("failure", ["resolver", "cost", "build", "stale"])
def test_prepublication_failures_rendezvous_without_session_or_fid_changes(
    owners, failure
):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))

    def reject(*args, **kwargs):
        raise PredictorError("data_unavailable", "protocol rejection")

    resolver = resolve
    if failure == "resolver":
        resolver = reject
    elif failure == "cost":
        owners[0].runtime.session.cost_model.predict_step = reject
    elif failure == "build":
        owners[1].runtime.results.build = reject
    else:
        owners[1].driver.validate_owner = reject
    published = []
    results = forward(
        owners, batches, resolver=resolver, publish=lambda *v: published.append(v)
    )
    assert all(isinstance(result, PredictorError) for result in results), results
    untouched(owners)
    assert published == []
    assert all(run.driver.failure is not None for run in owners)
    assert [run.runtime.session.progress_model.calls for run in owners] == [0, 0]


@pytest.mark.parametrize("bad_rank", [0, 1])
def test_unilateral_source_prepare_failure_releases_both_peers(owners, bad_rank):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))
    batches[bad_rank].seq_lens = [999]
    results = forward(owners, batches)
    assert all(isinstance(result, PredictorError) for result in results), results
    untouched(owners)
    assert [run.runtime.session.cost_model.calls for run in owners] == [0, 0]


@pytest.mark.parametrize("bad_rank", [0, 1])
def test_postpublication_failure_is_terminal_on_both_owners(owners, bad_rank):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))

    def publish(run, value):
        if run.rank == bad_rank:
            raise RuntimeError("actual publication failed")

    results = forward(owners, batches, publish=publish)
    assert isinstance(results[bad_rank], RuntimeError)
    assert isinstance(results[1 - bad_rank], PredictorError)
    untouched(owners)
    assert all(run.runtime._peer_failure is not None for run in owners)
    assert all(run.driver.failure is not None for run in owners)
    again = forward(owners, batches)
    assert all(isinstance(result, PredictorError) for result in again)


def test_publication_cannot_mutate_replicated_causal_state_before_commit(owners):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))

    def publish(run, value):
        if run.rank == 1:
            run.runtime._forward_id = 9

    results = forward(owners, batches, publish=publish)
    assert all(isinstance(result, PredictorError) for result in results)
    assert all(run.runtime._peer_failure is not None for run in owners)
    assert all(run.runtime.session._last_forward_id == -1 for run in owners)
    assert owners[0].runtime._forward_id == 0
    # Do not disguise an external effect as rolled back or commit over it.
    assert owners[1].runtime._forward_id == 9


def test_drained_retirement_and_reset_are_explicit_replicated_events(owners):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))
    results = passed(forward(owners, batches))
    for run in owners:
        process(run, batches[run.rank], results[run.rank], finish=True)
        assert run.runtime._peer_retirements == {f"dp{run.rank}-0"}
        assert len(run.runtime.session._history) == 2

    def lifecycle(run, reset=False):
        return run.runtime.synchronize_native_dp_peer_lifecycle(
            tp_group=run.group, attention_dp_rank=run.rank, reset=reset
        )

    passed(both(owners, lifecycle))
    for run in owners:
        assert run.runtime.session._history == {}
        assert run.runtime._peer_retirements == set()
        assert run.runtime._forward_id == 1
        assert run.runtime.session._last_forward_id == 0
        with pytest.raises(PredictorError, match="explicit joint lifecycle"):
            run.runtime.clear_cache_pool()
    passed(both(owners, lambda run: lifecycle(run, reset=True)))
    assert all(run.driver.adapter.ledger.pending == () for run in owners)


def test_peer_cannot_retire_live_local_work_or_reset_unprocessed_results(owners):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))
    passed(forward(owners, batches))
    results = both(
        owners,
        lambda run: run.runtime.synchronize_native_dp_peer_lifecycle(
            tp_group=run.group, attention_dp_rank=run.rank, reset=True
        ),
    )
    assert all(isinstance(result, PredictorError) for result in results), results
    assert all(len(run.runtime.session._history) == 2 for run in owners)
    assert all(run.runtime._forward_id == 1 for run in owners)


def test_resolver_cannot_change_nonexecution_work_or_local_reference_owner(owners):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))

    def changed(**kwargs):
        resolved = resolve(**kwargs)
        return replace(resolved, forward_id=resolved.forward_id + 1)

    results = forward(owners, batches, resolver=changed)
    assert all(isinstance(result, PredictorError) for result in results), results
    untouched(owners)
    assert [run.runtime.session.cost_model.calls for run in owners] == [0, 0]


@pytest.mark.parametrize("sync", [(True, False), (False, True), (True, True)])
def test_each_native_overlap_boundary_retains_its_actual_output_snapshot(owners, sync):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))
    first = passed(forward(owners, batches))
    decode = [
        native_batch(run, "DECODE", requests=batches[run.rank].reqs) for run in owners
    ]
    passed(select(owners, decode, sync=sync))
    for run in owners:
        if sync[run.rank]:
            process(run, batches[run.rank], first[run.rank])
    second = passed(forward(owners, decode))
    step = getattr(second[0], STEP_ATTRIBUTE).step
    assert tuple(group.requests[0].output_tokens_before for group in step.groups) == (
        int(sync[0]),
        int(sync[1]),
    )
    for run in owners:
        if not sync[run.rank]:
            assert len(run.driver.adapter.ledger.pending) == 2
            process(run, batches[run.rank], first[run.rank])
        process(run, decode[run.rank], second[run.rank])
        assert run.driver.adapter.ledger.pending == ()
    assert owners[0].runtime.session._history == owners[1].runtime.session._history


def test_commit_failure_after_publication_is_sticky_without_fabricated_rollback(owners):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))

    def broken_commit(_):
        raise RuntimeError("replicated commit failed")

    owners[1].runtime.session.commit = broken_commit
    published = []
    results = forward(
        owners, batches, publish=lambda run, _: published.append(run.rank)
    )
    assert set(published) == {0, 1}
    assert isinstance(results[0], PredictorError)
    assert isinstance(results[1], RuntimeError)
    assert owners[0].runtime._forward_id == 1
    assert owners[1].runtime._forward_id == 0
    assert all(run.runtime._peer_failure is not None for run in owners)
    assert all(run.driver.failure is not None for run in owners)
    assert all(isinstance(value, PredictorError) for value in forward(owners, batches))


def test_replicated_history_mismatch_rejects_before_capture_or_cost(owners):
    owners[1].runtime.session._last_forward_id = 10
    results = select(owners, [native_batch(run) for run in owners])
    assert all(isinstance(value, PredictorError) for value in results)
    assert all(run.driver.selection is None for run in owners)
    assert [run.runtime.session.cost_model.calls for run in owners] == [0, 0]


def test_only_local_retirement_after_drain_forgets_the_same_global_rid(owners):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))
    results = passed(forward(owners, batches))
    for run in owners:
        process(run, batches[run.rank], results[run.rank], finish=run.rank == 0)
    assert owners[0].runtime._peer_retirements == {"dp0-0"}
    assert owners[1].runtime._peer_retirements == set()
    passed(
        both(
            owners,
            lambda run: run.runtime.synchronize_native_dp_peer_lifecycle(
                tp_group=run.group, attention_dp_rank=run.rank
            ),
        )
    )
    assert owners[0].runtime.session._history == owners[1].runtime.session._history
    assert tuple(owners[0].runtime.session._history) == ("dp1-0",)
    assert tuple(row.rid for row in owners[1].driver.adapter.ledger.requests) == (
        "dp1-0",
    )


def test_lifecycle_disagreement_cannot_clear_either_replicated_history(owners):
    batches = [native_batch(run) for run in owners]
    passed(select(owners, batches))
    results = passed(forward(owners, batches))
    for run in owners:
        process(run, batches[run.rank], results[run.rank])
    before = [dict(run.runtime.session._history) for run in owners]
    results = both(
        owners,
        lambda run: run.runtime.synchronize_native_dp_peer_lifecycle(
            tp_group=run.group, attention_dp_rank=run.rank, reset=run.rank == 0
        ),
    )
    assert all(isinstance(value, PredictorError) for value in results)
    assert [run.runtime.session._history for run in owners] == before


@pytest.mark.parametrize("idle_rank", [None, 0, 1])
def test_existing_local_hash_progress_draws_once_per_active_group_only(
    owners, monkeypatch, idle_rank
):
    from sglang_simulator.time_predictor import dspark_golden_progress as golden

    # The separate source receipt reader is already covered independently.
    # Reuse its explicitly declared policy fixture for algorithm-only controls.
    policy = policy_fixture.__wrapped__(monkeypatch)
    for run in owners:
        run.runtime.session.progress_model = golden.SourceGoldenDPProgress(
            policy,
            runtime_profile_json=run.runtime.runtime_profile_json,
            model_seed=42,
        )
    batches = [
        native_batch(run, "IDLE" if run.rank == idle_rank else "EXTEND", size=2)
        for run in owners
    ]
    passed(select(owners, batches))
    first = passed(forward(owners, batches))
    for run in owners:
        process(run, batches[run.rank], first[run.rank])
    decode = [
        native_batch(
            run,
            "IDLE" if run.rank == idle_rank else "DECODE",
            requests=batches[run.rank].reqs,
        )
        for run in owners
    ]
    passed(select(owners, decode))
    calls = []
    digest = golden.hashlib.sha256

    def counted(data):
        import json

        calls.append(json.loads(data))
        return digest(data)

    monkeypatch.setattr(golden.hashlib, "sha256", counted)
    results = passed(forward(owners, decode))
    assert [row["attention_dp_rank"] for row in calls] == [
        rank for rank in range(2) if rank != idle_rank
    ]
    assert (
        getattr(results[0], STEP_ATTRIBUTE).progress
        == getattr(results[1], STEP_ATTRIBUTE).progress
    )
    commits = dict(getattr(results[0], STEP_ATTRIBUTE).progress.committed_tokens)
    for run in owners:
        assert {commits[req.rid] for req in decode[run.rank].reqs} <= {3, 4}
        if run.rank != idle_rank:
            assert len({commits[req.rid] for req in decode[run.rank].reqs}) == 1
