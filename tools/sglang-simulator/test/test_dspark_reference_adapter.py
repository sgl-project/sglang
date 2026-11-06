"""Explicit reference events only; CPU fixtures are not source qualification."""

from dataclasses import FrozenInstanceError, replace

import pytest
from sglang_simulator.simulation.sglang.dspark_reference import DSparkReferenceLedger
from sglang_simulator.simulation.sglang.dspark_reference_adapter import (
    DSparkReferenceAdapter,
    prepare_reference_step,
)
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_reference import (
    decode,
    first_published,
    modeled,
    native_page_sizing,
    prefill,
    seeded,
)
from test_dspark_step import session

pytestmark = pytest.mark.usefixtures(native_page_sizing.__name__)


def bridge(ledger, before, *, sync=False, skipped=(), resolver=None, **changes):
    arguments = {
        "page_size": 256,
        "future_indices_ready": False,
        "needs_cpu_seq_lens": None,
        "process_previous_before": sync,
        "skipped_previous_ids": skipped,
        "execution_resolver": resolver or (lambda **kwargs: {"path": "CPU fixture"}),
        "native_batch": None,
        "buffer_capacity": 0,
        "buffer_device_matches": False,
    }
    arguments.update(changes)
    return prepare_reference_step(ledger, before, **arguments)


def test_explicit_retirement_waits_for_both_pending_owners_and_skips_only_its_request():
    ledger = first_published()
    second = decode(1, prefix=246, output=1)
    ledger = ledger.prepare(second, page_size=256).publish(modeled(second))
    initial = DSparkReferenceAdapter(ledger=ledger)
    retiring = initial.retire_request("a")
    assert retiring.retiring_request_ids == ("a",)
    assert not retiring.ledger.request_for("a").active
    assert retiring.retire_request("a") == retiring
    one_pending = retiring.process_head(0)
    assert one_pending.retiring_request_ids == ("a",)
    assert one_pending.ledger.request_for("a").kv_committed_len == 240
    assert one_pending.ledger.request_for("a").processed_output_tokens == 1
    assert len(one_pending.ledger.pending) == 1
    with pytest.raises(PredictorError):
        one_pending.prepare_current(decode(2, prefix=248), page_size=256)
    drained = one_pending.process_head(1)
    assert drained.ledger.requests == drained.retiring_request_ids == ()
    with pytest.raises(PredictorError):
        drained.retire_request("a")
    assert initial.ledger.request_for("a").active
    assert len(initial.ledger.pending) == 2


@pytest.mark.parametrize("retire_before_publication", [False, True])
def test_existing_prefill_retirement_defers_forget_until_prepared_and_pending_drain(
    retire_before_publication,
):
    before = prefill(count=32, final=True)
    adapter = DSparkReferenceAdapter().prepare_current(before, page_size=256)
    if retire_before_publication:
        adapter = adapter.retire_request("a")
        assert adapter.ledger.prepared.step is before
    adapter = adapter.publish_current(modeled(before))
    if not retire_before_publication:
        adapter = adapter.retire_request("a")
    assert adapter.retiring_request_ids == ("a",)
    assert adapter.ledger.request_for("a").published_seq_len == 32
    assert adapter.ledger.request_for("a").processed_output_tokens == 0
    drained = adapter.process_head(0)
    assert drained.ledger.requests == () and drained.retiring_request_ids == ()
    assert drained.ledger.last_forward_id == 0
    reused = drained.prepare_current(prefill(1, count=8, final=True), page_size=256)
    assert reused.ledger.request_for("a").active
    assert reused.ledger.request_for("a").processed_output_tokens == 0


def test_natural_prefill_finish_settles_bonus_before_retirement():
    before = prefill(count=32, final=True)
    published = (
        DSparkReferenceAdapter()
        .prepare_current(before, page_size=256)
        .publish_current(modeled(before))
    )
    settled = published.process_head(0)
    assert settled.ledger.request_for("a").kv_committed_len == 32
    assert settled.ledger.request_for("a").processed_output_tokens == 1
    assert settled.retire_request("a").ledger.requests == ()


def test_natural_decode_finish_settles_head_then_skips_already_launched_successor():
    first = first_published(maximum=3)
    successor = decode(1, prefix=246, output=1, maximum=3)
    adapter = DSparkReferenceAdapter(ledger=first).prepare_current(
        successor, page_size=256
    )
    adapter = adapter.publish_current(modeled(successor))
    settled = adapter.process_head(0)
    assert settled.ledger.request_for("a").kv_committed_len == 246
    assert settled.ledger.request_for("a").processed_output_tokens == 7
    retiring = settled.retire_request("a")
    assert retiring.ledger.request_for("a").kv_committed_len == 246
    assert retiring.ledger.request_for("a").processed_output_tokens == 7
    assert len(retiring.ledger.pending) == 1
    drained = retiring.process_head(1)
    assert drained.ledger.requests == drained.retiring_request_ids == ()
    assert settled.ledger.request_for("a").active


def test_retirement_of_prepared_decode_preserves_its_frozen_view_and_publication():
    before = decode()
    adapter = DSparkReferenceAdapter(
        ledger=DSparkReferenceLedger(requests=(seeded(),))
    ).prepare_current(before, page_size=256)
    retiring = adapter.retire_request("a")
    view = retiring.resolve_current(future_indices_ready=False, needs_cpu_seq_lens=None)
    assert view.seq_lens_cpu == (240,) and view.nxt_kv_lens_cpu == (256,)
    published = retiring.publish_current(modeled(before, count=6))
    assert published.ledger.request_for("a").published_seq_len == 246
    assert not published.ledger.request_for("a").active
    assert published.process_head(0).ledger.requests == ()


def test_retirement_survives_prior_fifo_drain_while_current_work_is_still_prepared():
    before = decode(1, prefix=246)
    prepared = DSparkReferenceAdapter(ledger=first_published()).prepare_current(
        before, page_size=256
    )
    retired = prepared.retire_request("a").process_head(0)
    assert retired.ledger.pending == ()
    assert retired.ledger.prepared.step is before
    assert retired.retiring_request_ids == ("a",)
    assert not retired.ledger.request_for("a").active
    with pytest.raises(PredictorError):
        retired.drained_reset()
    published = retired.publish_current(modeled(before))
    assert published.retiring_request_ids == ("a",)
    drained = published.process_head(1)
    assert drained.ledger.requests == drained.retiring_request_ids == ()


def test_retirement_without_outstanding_ownership_forgets_immediately():
    adapter = DSparkReferenceAdapter(ledger=DSparkReferenceLedger(requests=(seeded(),)))
    retired = adapter.retire_request("a")
    assert retired.ledger.requests == retired.retiring_request_ids == ()
    with pytest.raises(PredictorError):
        adapter.retire_request("missing")
    assert adapter.ledger.request_for("a").active


def test_process_head_intersects_retirement_intents_with_exact_fifo_membership():
    first = prefill(rid="a", count=32, final=True)
    adapter = DSparkReferenceAdapter().prepare_current(first, page_size=256)
    adapter = adapter.publish_current(modeled(first))
    second = prefill(1, rid="b", count=16, final=True)
    adapter = adapter.prepare_current(second, page_size=256)
    adapter = adapter.publish_current(modeled(second)).retire_request("b")
    settled = adapter.process_head(0)
    assert settled.ledger.request_for("a").processed_output_tokens == 1
    assert settled.ledger.request_for("b").processed_output_tokens == 0
    assert settled.retiring_request_ids == ("b",)
    with pytest.raises(PredictorError):
        adapter.process_head(1)
    drained = settled.process_head(1)
    assert tuple(row.rid for row in drained.ledger.requests) == ("a",)
    assert drained.retiring_request_ids == ()


def test_drained_reset_preserves_forward_order_and_requires_explicit_ownership_drain():
    before = prefill(count=32, final=True)
    prepared = DSparkReferenceAdapter().prepare_current(before, page_size=256)
    with pytest.raises(PredictorError):
        prepared.drained_reset()
    published = prepared.publish_current(modeled(before))
    with pytest.raises(PredictorError):
        published.drained_reset()
    reset = published.process_head(0).drained_reset()
    assert reset.ledger.requests == reset.retiring_request_ids == ()
    assert reset.ledger.last_forward_id == 0
    with pytest.raises(PredictorError):
        reset.prepare_current(prefill(), page_size=256)
    assert reset.prepare_current(prefill(1), page_size=256).ledger.prepared is not None


@pytest.mark.parametrize("sync", [False, True])
def test_bridge_freezes_preparation_then_retires_only_the_preexisting_result(sync):
    base = first_published()
    # Worker-before-work output observes prior processing only at the explicit
    # sync boundary; allocation was prepared before that boundary in both cases.
    before = decode(1, prefix=246, output=7 if sync else 1)
    seen = []

    def resolve(**kwargs):
        seen.append(kwargs)
        assert kwargs["group"].requests is before.groups[0].requests
        assert kwargs["reference_view"].seq_lens_cpu == (240,)
        assert kwargs["reference_view"].nxt_kv_lens_cpu == (256,)
        return {"path": "CPU fixture"}

    transaction = bridge(base, before, sync=sync, resolver=resolve)
    assert transaction.base_ledger is base
    assert transaction.previous_forward_id == 0
    assert transaction.view.seq_lens_cpu == (240,)
    assert transaction.prepared_ledger.prepared.step is transaction.step
    assert transaction.prepared_ledger.prepared.seed_seq_lens_cpu == (240,)
    assert transaction.prepared_ledger.request_for("a").kv_committed_len == (
        246 if sync else 240
    )
    assert transaction.step.requests == before.requests
    assert transaction.step.requests[0].output_tokens_before == (7 if sync else 1)
    assert len(seen) == 1
    complete = transaction.complete(modeled(transaction.step), current_ledger=base)
    assert tuple(result.step.forward_id for result in complete.pending) == (1,)
    assert complete.request_for("a").kv_committed_len == 246
    assert complete.request_for("a").processed_output_tokens == 7
    assert complete.request_for("a").published_seq_len == 248
    assert base.request_for("a").kv_committed_len == 240
    assert tuple(result.step.forward_id for result in base.pending) == (0,)


def test_bridge_first_prefill_keeps_its_current_result_pending_until_explicit_process():
    base = DSparkReferenceLedger()
    transaction = bridge(base, prefill(count=32, final=True))
    assert transaction.previous_forward_id is None
    completed = transaction.complete(modeled(transaction.step), current_ledger=base)
    assert tuple(result.step.forward_id for result in completed.pending) == (0,)
    assert completed.request_for("a").processed_output_tokens == 0
    assert completed.process_result(0).request_for("a").processed_output_tokens == 1
    assert base.requests == ()


@pytest.mark.parametrize("needs_cpu", [False, True])
def test_bridge_transports_resolved_cpu_presence_without_changing_the_planner_vector(
    needs_cpu,
):
    base = first_published()
    seen = []

    def resolve(**kwargs):
        seen.append(kwargs["reference_view"])
        return {"path": "CPU fixture"}

    transaction = bridge(
        base,
        decode(1, prefix=246),
        resolver=resolve,
        future_indices_ready=True,
        needs_cpu_seq_lens=needs_cpu,
    )
    assert seen == [transaction.view]
    assert transaction.view.seq_lens_cpu == ((246,) if needs_cpu else None)
    assert transaction.view.nxt_kv_lens_cpu == (256,)
    assert transaction.view.planner_seq_lens_cpu == ((246,) if needs_cpu else (256,))


def test_bridge_final_step_changes_only_execution_and_preserves_exact_result_binding():
    import json

    base = first_published()
    before = decode(1, prefix=246)
    native = object()
    observed = []

    def resolve(**kwargs):
        observed.append(kwargs)
        return {
            "path": "CPU fixture",
            "draft_buffer_capacity": 9000,
            "draft_buffer_device_matches": True,
        }

    transaction = bridge(base, before, resolver=resolve, native_batch=native)
    final = transaction.step
    assert final is not before
    assert final.forward_id == before.forward_id
    assert final.topology is before.topology
    assert final.runtime_profile_json == before.runtime_profile_json
    assert final.groups[0].requests is before.groups[0].requests
    assert final.groups[0].forward_mode == before.groups[0].forward_mode
    assert final.groups[0].attention_dp_rank == before.groups[0].attention_dp_rank
    execution = json.loads(final.groups[0].execution_json)
    assert execution["draft_buffer_capacity"] == 0
    assert execution["draft_buffer_device_matches"] is False
    assert observed[0]["native_batch"] is native
    assert observed[0]["runtime_profile_json"] == before.runtime_profile_json
    assert observed[0]["buffer_capacity"] == 0
    assert observed[0]["buffer_device_matches"] is False
    with pytest.raises(PredictorError):
        transaction.complete(modeled(before), current_ledger=base)
    with pytest.raises(PredictorError):
        transaction.complete(modeled(replace(final)), current_ledger=base)
    completed = transaction.complete(modeled(final), current_ledger=base)
    assert completed.pending[0].step is final


@pytest.mark.parametrize("sync", [False, True])
def test_bridge_freezes_prior_retirement_before_current_modeled_progress(sync):
    base = first_published()
    transaction = bridge(
        base, decode(1, prefix=246, output=7 if sync else 1), sync=sync, skipped=("a",)
    )
    completed = transaction.complete(
        modeled(transaction.step, count=6), current_ledger=base
    )
    assert completed.request_for("a").published_seq_len == 252
    assert completed.request_for("a").kv_committed_len == 240
    assert completed.request_for("a").processed_output_tokens == 1
    assert not completed.request_for("a").active
    with pytest.raises(PredictorError):
        completed.process_result(1)
    assert completed.process_result(1, skipped_request_ids=("a",)).pending == ()


def test_bridge_cost_failure_and_stale_owner_cannot_consume_or_replace_the_base():
    base = first_published()
    before = decode(1, prefix=246)
    transaction = bridge(base, before)
    run = session(transaction.step)
    progress_calls = run.progress_model.calls

    def missing(_step):
        raise PredictorError("data_unavailable", "missing CPU fixture cost")

    run.cost_model.predict_step = missing
    with pytest.raises(PredictorError, match="missing CPU fixture cost"):
        run.predict(transaction.step)
    assert run.progress_model.calls == progress_calls
    assert transaction.base_ledger is base
    assert base.request_for("a").kv_committed_len == 240
    advanced = base.process_result(0)
    prediction = modeled(transaction.step)
    with pytest.raises(PredictorError):
        transaction.complete(prediction, current_ledger=advanced)
    with pytest.raises(PredictorError):
        transaction.complete(prediction, current_ledger=replace(base))
    completed = transaction.complete(prediction, current_ledger=base)
    assert tuple(result.step.forward_id for result in completed.pending) == (1,)
    with pytest.raises(FrozenInstanceError):
        transaction.process_previous_before = True


def test_bridge_resolver_failure_leaves_all_reference_events_uncommitted():
    base = first_published()

    def missing(**kwargs):
        raise PredictorError("data_unavailable", "source projection unavailable")

    with pytest.raises(PredictorError, match="source projection unavailable"):
        bridge(base, decode(1, prefix=246), sync=True, resolver=missing)
    assert base.prepared is None
    assert tuple(result.step.forward_id for result in base.pending) == (0,)
    assert base.request_for("a").kv_committed_len == 240
    assert bridge(base, decode(1, prefix=246)).previous_forward_id == 0


@pytest.mark.parametrize(
    "execution", [None, [], {"path": float("nan")}, {"path": object()}]
)
def test_bridge_requires_serializable_finite_execution_metadata(execution):
    base = first_published()
    with pytest.raises(PredictorError):
        bridge(base, decode(1, prefix=246), resolver=lambda **kwargs: execution)
    assert base.prepared is None
    assert base.request_for("a").kv_committed_len == 240


@pytest.mark.parametrize("sync", [None, 0, 1, "sync"])
def test_bridge_requires_explicit_boolean_source_sync_policy_before_resolution(sync):
    def must_not_resolve(**kwargs):
        pytest.fail("invalid source policy reached resolver")

    with pytest.raises(PredictorError):
        bridge(
            first_published(),
            decode(1, prefix=246),
            sync=sync,
            resolver=must_not_resolve,
        )


@pytest.mark.parametrize("skipped", [("missing",), ("a", "a"), ["a"], (True,)])
def test_bridge_validates_previous_skip_ownership_before_resolution(skipped):
    def must_not_resolve(**kwargs):
        pytest.fail("invalid prior skip ownership reached resolver")

    with pytest.raises(PredictorError):
        bridge(
            first_published(),
            decode(1, prefix=246),
            skipped=skipped,
            resolver=must_not_resolve,
        )


@pytest.mark.parametrize("changes", [{"sync": True}, {"skipped": ("a",)}])
def test_bridge_rejects_sync_or_skip_claim_without_an_existing_previous_result(changes):
    def must_not_resolve(**kwargs):
        pytest.fail("missing prior result reached resolver")

    with pytest.raises(PredictorError):
        bridge(DSparkReferenceLedger(), prefill(), resolver=must_not_resolve, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"execution_resolver": None},
        {"buffer_capacity": True},
        {"buffer_capacity": -1},
        {"buffer_device_matches": 0},
    ],
)
def test_bridge_rejects_missing_resolver_or_untyped_source_buffer_facts(changes):
    with pytest.raises(PredictorError):
        bridge(first_published(), decode(1, prefix=246), **changes)
