"""Local reference ownership only; protocol costs are not timing evidence."""

import json
from dataclasses import replace

import pytest
from sglang_simulator.simulation.sglang.dspark import (
    DSparkGroup,
    DSparkPhaseCost,
    DSparkProgress,
    DSparkStep,
    DSparkStepResult,
    DSparkStepTiming,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_reference import (
    DSparkReferenceLedger,
    ReferenceCPUView,
)
from sglang_simulator.simulation.sglang.dspark_reference_adapter import (
    DSparkReferenceAdapter,
)
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_reference import native_page_sizing, seeded
from test_dspark_reference_adapter import bridge
from test_dspark_step import request

pytestmark = pytest.mark.usefixtures(native_page_sizing.__name__)


def joint_step(rows, *, fid=0, mode="TARGET_VERIFY", final=True):
    groups = []
    for rank, local in enumerate(rows):
        requests = tuple(
            replace(
                request(row.rid, prefix=row.published_seq_len),
                extend_len=6 if mode == "TARGET_VERIFY" else 16,
                output_tokens_before=(
                    row.processed_output_tokens if mode == "TARGET_VERIFY" else 0
                ),
                emits_output=final,
            )
            for row in local
        )
        groups.append(
            DSparkGroup(
                rank,
                mode if local else "IDLE",
                requests,
                '{"draft_buffer_capacity":0,"draft_buffer_device_matches":false,"path":"CPU fixture"}',
            )
        )
    return DSparkStep(
        fid,
        DSparkTopology(2, 2),
        tuple(groups),
        '{"source":"joint CPU protocol fixture; not attestation"}',
    )


def shared_result(before):
    # The common prediction is supplied once. No measured or modeled latency
    # claim is made by this reference-only protocol fixture.
    return DSparkStepResult(
        before,
        DSparkStepTiming(
            before.forward_id,
            (DSparkPhaseCost("whole_worker", 1, "CPU protocol fixture only"),),
        ),
        DSparkProgress(
            before.forward_id,
            tuple(
                (
                    r.rid,
                    (3 + group.attention_dp_rank)
                    if group.forward_mode == "TARGET_VERIFY"
                    else int(r.emits_output),
                )
                for group in before.groups
                for r in group.requests
            ),
            "CPU protocol fixture only; no RNG inference",
        ),
    )


def ledger_for(rank, rows):
    return DSparkReferenceLedger(requests=rows, local_attention_dp_rank=rank)


@pytest.mark.parametrize("idle_rank", [None, 0, 1])
def test_both_native_owners_retain_global_step_but_settle_only_local_requests(
    idle_rank,
):
    rows = (
        ()
        if idle_rank == 0
        else (seeded("a"), seeded("b", committed=245, published=245)),
        ()
        if idle_rank == 1
        else (seeded("c", committed=500, allocated=512, published=500),),
    )
    before = joint_step(rows)
    transactions = tuple(
        bridge(ledger_for(rank, local), before) for rank, local in enumerate(rows)
    )
    common = shared_result(before)
    for rank, (transaction, local) in enumerate(zip(transactions, rows, strict=True)):
        assert (
            transaction.step == before and transaction.step.topology is before.topology
        )
        assert transaction.step.groups[1 - rank] is before.groups[1 - rank]
        assert (
            transaction.prepared_ledger.prepared.local_group
            is transaction.step.groups[rank]
        )
        view = transaction.view
        assert view.local_attention_dp_rank == rank
        assert view.is_idle == (idle_rank == rank)
        assert view.seq_lens_cpu == tuple(row.kv_committed_len for row in local)
        if not local:
            assert view.nxt_kv_lens_cpu is None and view.planner_seq_lens_cpu == ()
        else:
            assert view.nxt_kv_lens_cpu == tuple(
                512 if row.kv_committed_len >= 245 else 256 for row in local
            )
        bound = transaction.bind_result(common)
        assert bound.step is transaction.step
        assert bound.timing is common.timing and bound.progress is common.progress
        assert bound.local_attention_dp_rank == rank
        assert bound.local_group.requests == before.groups[rank].requests
        complete = transaction.complete(bound, current_ledger=transaction.base_ledger)
        assert complete.pending == (bound,)
        assert tuple(row.rid for row in complete.requests) == tuple(
            row.rid for row in local
        )
        drained = complete.process_result(before.forward_id)
        for row in local:
            assert (
                complete.request_for(row.rid).published_seq_len
                == row.published_seq_len + 3 + rank
            )
            assert (
                complete.request_for(row.rid).kv_committed_len == row.kv_committed_len
            )
            assert (
                drained.request_for(row.rid).kv_committed_len
                == row.kv_committed_len + 3 + rank
            )
            assert (
                drained.request_for(row.rid).processed_output_tokens
                == row.processed_output_tokens + 3 + rank
            )
        assert drained.pending == () and drained.last_forward_id == before.forward_id
        assert transaction.base_ledger.requests == local
    assert common.local_attention_dp_rank == 0  # Shared input was not mutated.


@pytest.mark.parametrize("local_rank", [0, 1])
def test_local_prefill_chunks_and_bonus_do_not_admit_or_consume_peer_requests(
    local_rank,
):
    rows = ((seeded("a"),), (seeded("b", committed=500, allocated=512, published=500),))
    first = joint_step(rows, mode="EXTEND", final=False)
    base = ledger_for(local_rank, ())
    one = bridge(base, first)
    one_result = one.bind_result(shared_result(first))
    published = one.complete(one_result, current_ledger=base)
    rid = rows[local_rank][0].rid
    local = published.request_for(rid)
    assert local.published_seq_len == rows[local_rank][0].published_seq_len + 16
    assert not local.prefill_complete and local.processed_output_tokens == 0
    drained = published.process_result(0)
    next_rows = tuple(
        tuple(
            replace(
                row,
                kv_committed_len=row.published_seq_len + 16,
                kv_allocated_len=row.published_seq_len + 16,
                published_seq_len=row.published_seq_len + 16,
            )
            for row in group
        )
        for group in rows
    )
    second = joint_step(next_rows, fid=1, mode="EXTEND")
    two = bridge(drained, second)
    final = two.complete(two.bind_result(shared_result(second)), current_ledger=drained)
    assert final.request_for(rid).prefill_complete
    assert final.request_for(rid).processed_output_tokens == 0
    assert final.process_result(1).request_for(rid).processed_output_tokens == 1
    with pytest.raises(PredictorError, match="no explicit prefill admission"):
        final.request_for(rows[1 - local_rank][0].rid)


@pytest.mark.parametrize("rank", [0, 1])
def test_resolver_receives_selected_local_group_buffer_and_cpu_view_only(rank):
    rows = ((seeded("a"),), (seeded("b"),))
    before = joint_step(rows)
    seen = []

    def resolve(**kwargs):
        seen.append(kwargs)
        assert kwargs["group"] is before.groups[rank]
        assert kwargs["reference_view"].local_attention_dp_rank == rank
        return {"source_selected_local": rank}

    transaction = bridge(
        ledger_for(rank, rows[rank]),
        before,
        resolver=resolve,
        buffer_capacity=rank,
        buffer_device_matches=bool(rank),
    )
    assert len(seen) == 1
    assert transaction.step.groups[1 - rank] is before.groups[1 - rank]
    assert transaction.step.requests == before.requests
    execution = json.loads(transaction.step.groups[rank].execution_json)
    assert execution == {
        "source_selected_local": rank,
        "draft_buffer_capacity": rank,
        "draft_buffer_device_matches": bool(rank),
    }
    # A locally resolved step is not a substitute for a common resolved step.
    with pytest.raises(PredictorError, match="complete matching global step"):
        transaction.bind_result(shared_result(before))


def test_exchanged_peer_executions_bind_one_full_prediction_without_changing_local_views():
    rows = ((seeded("a"), seeded("b")), (seeded("c"),))
    before = joint_step(rows)
    transactions = tuple(
        bridge(
            ledger_for(rank, rows[rank]),
            before,
            resolver=lambda group, **kwargs: {
                "resolved_owner": group.attention_dp_rank
            },
            buffer_capacity=rank,
            buffer_device_matches=bool(rank),
        )
        for rank in (0, 1)
    )
    # The caller received both actual views. This fixture labels the common
    # execution JSON only; the SDK, not this adapter, validates a joint query.
    common_execution = json.dumps(
        {
            "joint_views": [
                transaction.view.seq_lens_cpu for transaction in transactions
            ]
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    agreed = replace(
        before,
        groups=tuple(
            replace(group, execution_json=common_execution) for group in before.groups
        ),
    )
    rebound = tuple(
        transaction.with_resolved_executions(agreed) for transaction in transactions
    )
    common = shared_result(agreed)
    for rank, transaction in enumerate(rebound):
        assert (
            transaction.step is agreed and transaction.view is transactions[rank].view
        )
        assert transaction.base_ledger is transactions[rank].base_ledger
        assert (
            transaction.prepared_ledger.requests
            == transactions[rank].prepared_ledger.requests
        )
        assert (
            transaction.prepared_ledger.pending
            is transactions[rank].prepared_ledger.pending
        )
        assert (
            transaction.prepared_ledger.prepared.seed_seq_lens_cpu
            is transactions[rank].prepared_ledger.prepared.seed_seq_lens_cpu
        )
        assert (
            transaction.prepared_ledger.prepared.nxt_kv_lens_cpu
            is transactions[rank].prepared_ledger.prepared.nxt_kv_lens_cpu
        )
        assert transaction.prepared_ledger.prepared.local_attention_dp_rank == rank
        result = transaction.bind_result(common)
        assert result.timing is common.timing and result.progress is common.progress
        assert result.local_group is agreed.groups[rank]
        completed = transaction.complete(result, current_ledger=transaction.base_ledger)
        assert tuple(row.rid for row in completed.requests) == tuple(
            row.rid for row in rows[rank]
        )
        assert transactions[rank].step.groups[1 - rank] is before.groups[1 - rank]


@pytest.mark.parametrize(
    "change",
    ["local-request", "peer-request", "peer-mode", "runtime", "fid", "topology"],
)
def test_execution_join_cannot_rewrite_frozen_before_work(change):
    before = joint_step(((seeded("a"),), (seeded("b"),)))
    transaction = bridge(ledger_for(1, (seeded("b"),)), before)
    if change == "local-request":
        group = before.groups[1]
        wrong = replace(
            before,
            groups=(
                before.groups[0],
                replace(
                    group,
                    requests=(replace(group.requests[0], output_tokens_before=2),),
                ),
            ),
        )
    elif change == "peer-request":
        peer = before.groups[0]
        wrong = replace(
            before,
            groups=(
                replace(
                    peer, requests=(replace(peer.requests[0], output_tokens_before=2),)
                ),
                before.groups[1],
            ),
        )
    elif change == "peer-mode":
        wrong = replace(
            before,
            groups=(replace(before.groups[0], forward_mode="EXTEND"), before.groups[1]),
        )
    elif change == "runtime":
        wrong = replace(before, runtime_profile_json='{"different":"runtime"}')
    elif change == "fid":
        wrong = replace(before, forward_id=1)
    else:
        wrong = replace(
            before,
            topology=DSparkTopology(4, 1),
            groups=(replace(before.groups[0], requests=before.requests),),
        )
    with pytest.raises(PredictorError, match="execution resolution changed frozen"):
        transaction.with_resolved_executions(wrong)
    assert transaction.base_ledger.prepared is None
    assert transaction.prepared_ledger.prepared.step is transaction.step


@pytest.mark.parametrize("idle_rank", [0, 1])
def test_idle_owner_has_no_local_allocation_or_relay(monkeypatch, idle_rank):
    from sglang_simulator.simulation.sglang import dspark_reference

    def must_not_allocate(*args, **kwargs):
        pytest.fail("idle peer allocated active peer KV")

    monkeypatch.setattr(dspark_reference, "_decode_alloc_lens", must_not_allocate)
    rows = ((), (seeded("b"),)) if idle_rank == 0 else ((seeded("a"),), ())
    ledger = ledger_for(idle_rank, ())
    before = joint_step(rows)
    prepared = ledger.prepare(before, page_size=256)
    assert prepared.prepared.step is before
    assert (
        prepared.prepared.seed_seq_lens_cpu
        == prepared.prepared.published_seq_lens
        == ()
    )
    assert prepared.prepared.nxt_kv_lens_cpu is None
    view = prepared.prepared.resolve(
        future_indices_ready=False, needs_cpu_seq_lens=None
    )
    assert view == ReferenceCPUView((), None, idle_rank, True)
    with pytest.raises(PredictorError, match="idle group cannot resolve"):
        prepared.prepared.resolve(future_indices_ready=True, needs_cpu_seq_lens=True)
    assert ledger.requests == () and ledger.prepared is None


@pytest.mark.parametrize("rank", [0, 1])
def test_peer_progress_skips_and_retirement_cannot_consume_local_ownership(rank):
    rows = ((seeded("a"),), (seeded("b"),))
    before = joint_step(rows)
    original = DSparkReferenceAdapter(ledger_for(rank, rows[rank]))
    prepared = original.prepare_current(before, page_size=256)
    common = shared_result(before)
    wrong = replace(common, local_attention_dp_rank=1 - rank)
    with pytest.raises(PredictorError, match="exact prepared modeled result"):
        prepared.publish_current(wrong)
    bound = replace(common, local_attention_dp_rank=rank)
    published = prepared.publish_current(bound)
    peer_rid = rows[1 - rank][0].rid
    with pytest.raises(PredictorError, match="skipped request ownership"):
        published.ledger.process_result(0, skipped_request_ids=(peer_rid,))
    with pytest.raises(PredictorError, match="no explicit prefill admission"):
        published.retire_request(peer_rid)
    with pytest.raises(PredictorError, match="matching previous result"):
        bridge(published.ledger, replace(before, forward_id=1), skipped=(peer_rid,))
    local_rid = rows[rank][0].rid
    retired = published.retire_request(local_rid)
    assert retired.retiring_request_ids == (local_rid,)
    drained = retired.process_head(0)
    assert drained.ledger.requests == drained.retiring_request_ids == ()
    assert drained.ledger.local_attention_dp_rank == rank
    assert original.ledger.requests == rows[rank]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("sync", [False, True])
def test_two_global_frames_freeze_local_vectors_before_source_selected_fifo_settlement(
    rank, sync
):
    rows = ((seeded("a"),), (seeded("b"),))
    first = joint_step(rows)
    transaction = bridge(ledger_for(rank, rows[rank]), first)
    published = transaction.complete(
        transaction.bind_result(shared_result(first)),
        current_ledger=transaction.base_ledger,
    )
    next_rows = tuple(
        tuple(
            replace(
                row,
                published_seq_len=row.published_seq_len + 3 + dp,
                processed_output_tokens=row.processed_output_tokens
                + (3 + dp if sync and dp == rank else 0),
            )
            for row in local
        )
        for dp, local in enumerate(rows)
    )
    second = joint_step(next_rows, fid=1)
    current = bridge(published, second, sync=sync)
    assert current.view.seq_lens_cpu == (240,)
    assert current.view.nxt_kv_lens_cpu == (256,)
    result = current.bind_result(shared_result(second))
    completed = current.complete(result, current_ledger=published)
    assert tuple(r.step.forward_id for r in completed.pending) == (1,)
    assert completed.request_for(rows[rank][0].rid).kv_committed_len == 243 + rank
    assert (
        completed.process_result(1).request_for(rows[rank][0].rid).kv_committed_len
        == 246 + 2 * rank
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("needs_cpu", [False, True])
def test_relay_uses_only_selected_published_lengths_and_local_reservations(
    rank, needs_cpu
):
    rows = ((seeded("a"),), (seeded("b", committed=500, allocated=512, published=500),))
    before = joint_step(rows)
    transaction = bridge(
        ledger_for(rank, rows[rank]),
        before,
        future_indices_ready=True,
        needs_cpu_seq_lens=needs_cpu,
    )
    assert transaction.view.seq_lens_cpu == (
        (rows[rank][0].published_seq_len,) if needs_cpu else None
    )
    assert transaction.view.nxt_kv_lens_cpu == ((256,) if rank == 0 else (512,))
    assert transaction.view.local_attention_dp_rank == rank
    assert transaction.step.groups[1 - rank] is before.groups[1 - rank]


@pytest.mark.parametrize("rank", [0, 1])
def test_idle_successor_settles_its_local_previous_result_without_consuming_peer_progress(
    rank,
):
    rows = ((seeded("a"),), (seeded("b"),))
    first = joint_step(rows)
    transaction = bridge(ledger_for(rank, rows[rank]), first)
    pending = transaction.complete(
        transaction.bind_result(shared_result(first)),
        current_ledger=transaction.base_ledger,
    )
    idle_rows = tuple(() if dp == rank else local for dp, local in enumerate(rows))
    successor = joint_step(idle_rows, fid=1)
    idle = bridge(pending, successor, sync=True)
    assert idle.view.is_idle and idle.view.seq_lens_cpu == ()
    local_rid = rows[rank][0].rid
    assert idle.prepared_ledger.request_for(local_rid).kv_committed_len == 243 + rank
    bound = idle.bind_result(shared_result(successor))
    assert bound.local_group.requests == () and bound.progress.committed_tokens
    published = idle.complete(bound, current_ledger=pending)
    settled = published.process_result(1)
    assert settled.request_for(local_rid) == published.request_for(local_rid)
    assert settled.pending == () and settled.last_forward_id == 1
    assert settled.forget(local_rid).requests == ()
    assert pending.request_for(local_rid).kv_committed_len == 240


@pytest.mark.parametrize("change", [True, -1, 2, 0.0])
def test_local_rank_must_be_a_real_declared_integer_owner(change):
    with pytest.raises(PredictorError):
        DSparkReferenceLedger(local_attention_dp_rank=change)
    with pytest.raises(PredictorError):
        ReferenceCPUView((240,), (256,), change)


def test_wrong_rank_pending_preparation_and_peer_admission_are_rejected():
    rows = ((seeded("a"),), (seeded("b"),))
    before = joint_step(rows)
    prepared = ledger_for(1, rows[1]).prepare(before, page_size=256)
    with pytest.raises(PredictorError):
        replace(prepared, local_attention_dp_rank=0)
    with pytest.raises(PredictorError):
        DSparkReferenceLedger(
            pending=(shared_result(before),),
            last_forward_id=0,
            local_attention_dp_rank=1,
        )
    with pytest.raises(PredictorError, match="peer request is already owned"):
        ledger_for(1, rows[0] + rows[1]).prepare(before, page_size=256)
    assert prepared.prepared.step is before and prepared.pending == ()


@pytest.mark.parametrize("change", ["peer-recipe", "peer-order", "runtime", "fid"])
def test_binding_does_not_hide_global_peer_or_runtime_mismatch(change):
    rows = ((seeded("a"), seeded("b")), (seeded("c"),))
    before = joint_step(rows)
    transaction = bridge(ledger_for(1, rows[1]), before)
    if change == "peer-recipe":
        peer = before.groups[0]
        request = replace(
            peer.requests[0],
            sampling_params_json='{"extra":true,"is_normalized":true,"max_new_tokens":20}',
        )
        wrong = replace(
            before,
            groups=(
                replace(peer, requests=(request, *peer.requests[1:])),
                before.groups[1],
            ),
        )
    elif change == "peer-order":
        wrong = replace(
            before,
            groups=(
                replace(before.groups[0], requests=before.groups[0].requests[::-1]),
                before.groups[1],
            ),
        )
    elif change == "runtime":
        wrong = replace(before, runtime_profile_json='{"source":"different"}')
    else:
        wrong = replace(before, forward_id=1)
    with pytest.raises(PredictorError, match="complete matching global step"):
        transaction.bind_result(shared_result(wrong))
    assert transaction.base_ledger.prepared is None


def test_local_allocation_snapshot_restores_only_its_selected_output_counts():
    rows = ((seeded("a"),), (seeded("b"),))
    before = joint_step(rows)
    base = ledger_for(1, rows[1])
    allocation = base.prepare(before, page_size=256)
    group = before.groups[1]
    worker = replace(
        before,
        groups=(
            before.groups[0],
            replace(
                group, requests=(replace(group.requests[0], output_tokens_before=5),)
            ),
        ),
    )
    transaction = bridge(base, worker, allocation_ledger=allocation)
    assert transaction.step.groups[0] is before.groups[0]
    assert transaction.step.groups[1].requests[0].output_tokens_before == 5
    assert transaction.view.seq_lens_cpu == (240,)
    peer = before.groups[0]
    wrong = replace(
        worker,
        groups=(
            replace(
                peer, requests=(replace(peer.requests[0], output_tokens_before=4),)
            ),
            worker.groups[1],
        ),
    )
    with pytest.raises(PredictorError, match="frozen selection"):
        bridge(base, wrong, allocation_ledger=allocation)
    with pytest.raises(PredictorError, match="allocation snapshot"):
        bridge(ledger_for(0, rows[0]), worker, allocation_ledger=allocation)
    assert allocation.prepared.step is before and base.prepared is None


@pytest.mark.parametrize("change", ["mixed", "batch65", "wrong-rank", "wrong-view"])
def test_joint_reference_failures_do_not_install_preparation_or_consume_progress(
    change,
):
    rows = ((seeded("a"),), (seeded("b"),))
    before = joint_step(rows)
    base = ledger_for(1, rows[1])
    if change == "mixed":
        before = replace(
            before,
            groups=(replace(before.groups[0], forward_mode="EXTEND"), before.groups[1]),
        )
    elif change == "batch65":
        before = replace(
            before,
            groups=(
                replace(
                    before.groups[0],
                    requests=tuple(request(str(i), prefix=240) for i in range(65)),
                ),
                before.groups[1],
            ),
        )
    transaction = None
    with pytest.raises(PredictorError):
        transaction = bridge(base, before)
        if change == "wrong-rank":
            transaction.complete(shared_result(transaction.step), current_ledger=base)
        else:
            replace(transaction, view=ReferenceCPUView((240,), (256,)))
    assert base.prepared is None and base.pending == () and base.requests == rows[1]
    if transaction is not None:
        assert transaction.prepared_ledger.pending == ()


def test_empty_vectors_require_an_explicit_idle_view_without_fake_peer_mirrors():
    assert ReferenceCPUView((), None, 1, True).planner_seq_lens_cpu == ()
    for args in [
        ((), None),
        (None, None, 1, True),
        ((), (), 1, True),
        ((240,), None, 1, True),
        ((240,), (256,), 1, 1),
    ]:
        with pytest.raises(PredictorError):
            ReferenceCPUView(*args)
