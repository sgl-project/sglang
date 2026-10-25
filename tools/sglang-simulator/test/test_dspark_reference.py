"""CPU event-order contracts; modeled fixtures do not qualify source execution."""

import importlib.util
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import ModuleType

import pytest
from sglang_simulator.simulation.sglang import dspark_reference as reference
from sglang_simulator.simulation.sglang.dspark import DSparkTopology
from sglang_simulator.simulation.sglang.dspark_reference import (
    DSparkReferenceLedger,
    PreparedReferenceBatch,
    ReferenceCPUView,
    ReferenceRequestKV,
)
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_step import Progress, request, session, step


@pytest.fixture(autouse=True)
def native_page_sizing(monkeypatch):
    """Exercise the native formula without importing a GPU runtime or copying it."""
    source = (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/mem_cache/allocation_sizing.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_reference_native_page_sizing", source
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError:
        # Only this lazy runtime leaf is unavailable on the CPU test host. The
        # pure sizing helper must not consult any of its runtime accessors.
        context = ModuleType("sglang.srt.runtime_context")

        def unavailable(*args, **kwargs):
            raise AssertionError("pure page sizing consulted native runtime state")

        for name in (
            "get_parallel",
            "get_schedule",
            "get_spec",
            "max_speculative_num_draft_tokens",
        ):
            setattr(context, name, unavailable)
        monkeypatch.setitem(sys.modules, context.__name__, context)
        spec.loader.exec_module(module)
    monkeypatch.setattr(
        reference, "_decode_alloc_lens", module.page_aligned_decode_alloc_lens
    )


def seeded(
    rid="a", *, committed=240, allocated=256, published=240, output=1, maximum=20
):
    """Explicit protocol seed, with no claim that these are observed source facts."""
    return ReferenceRequestKV(
        rid,
        committed,
        allocated,
        published,
        output,
        maximum,
        prefill_complete=True,
    )


def decode(fid=0, *, rid="a", prefix=240, output=1, maximum=20, requests=None):
    before = replace(
        request(rid, prefix=prefix),
        output_tokens_before=output,
        max_new_tokens=maximum,
        sampling_params_json=(
            '{"is_normalized":true,"max_new_tokens":' + str(maximum) + "}"
        ),
    )
    return step(fid=fid, requests=requests if requests is not None else (before,))


def prefill(fid=0, *, rid="a", prefix=0, count=240, final=False):
    before = decode(fid, rid=rid, prefix=prefix, output=0)
    group = replace(
        before.groups[0],
        forward_mode="EXTEND",
        requests=(replace(before.requests[0], extend_len=count, emits_output=final),),
    )
    return replace(before, groups=(group,))


def modeled(before, count=2):
    # Reuse the existing independent cost/progress fixtures and their explicit
    # fixture provenance. No timing or acceptance scalar is introduced here.
    progress = Progress()
    progress.count = count
    return session(before, progress=progress).predict(before)


def first_published(*, maximum=20):
    ledger = DSparkReferenceLedger(requests=(seeded(maximum=maximum),))
    before = decode(maximum=maximum)
    return ledger.prepare(before, page_size=256).publish(modeled(before, count=6))


def test_native_page_boundary_preserves_three_distinct_cpu_facts_and_sync_snapshot():
    initial = DSparkReferenceLedger(requests=(seeded(),))
    first = decode()
    prepared = initial.prepare(first, page_size=256)
    assert prepared.prepared.seed_seq_lens_cpu == (240,)
    assert prepared.prepared.nxt_kv_lens_cpu == (256,)
    published = prepared.publish(modeled(first, count=6))
    assert published.request_for("a").kv_committed_len == 240
    assert published.request_for("a").published_seq_len == 246
    assert published.request_for("a").processed_output_tokens == 1

    second = decode(1, prefix=246, output=7)
    before_sync = published.prepare(second, page_size=256)
    snapshot = before_sync.prepared
    assert snapshot.seed_seq_lens_cpu == (240,)
    assert snapshot.published_seq_lens == (246,)
    assert snapshot.nxt_kv_lens_cpu == (256,)
    assert snapshot.resolve(
        future_indices_ready=False, needs_cpu_seq_lens=None
    ) == ReferenceCPUView((240,), (256,))
    assert snapshot.resolve(
        future_indices_ready=True, needs_cpu_seq_lens=True
    ) == ReferenceCPUView((246,), (256,))
    absent = snapshot.resolve(future_indices_ready=True, needs_cpu_seq_lens=False)
    assert absent.seq_lens_cpu is None
    assert absent.planner_seq_lens_cpu == (256,)

    after_sync = before_sync.process_result(0)
    assert after_sync.request_for("a").kv_committed_len == 246
    assert after_sync.request_for("a").processed_output_tokens == 7
    assert after_sync.prepared is snapshot
    assert snapshot.seed_seq_lens_cpu == (240,)
    next_published = after_sync.publish(modeled(second))
    assert next_published.request_for("a").published_seq_len == 248
    third = decode(2, prefix=248, output=9)
    next_prepared = next_published.prepare(third, page_size=256)
    assert next_prepared.prepared.seed_seq_lens_cpu == (246,)
    assert next_prepared.prepared.nxt_kv_lens_cpu == (512,)
    assert initial.request_for("a") == seeded()
    assert prepared.prepared is not None
    assert published.pending[0].step is first


def test_ordinary_publish_then_retire_allows_two_results_but_requires_fifo():
    one = first_published()
    second = decode(1, prefix=246, output=7)
    two = one.prepare(second, page_size=256).publish(modeled(second))
    assert tuple(result.step.forward_id for result in two.pending) == (0, 1)
    with pytest.raises(PredictorError, match="preparation precedes retirement"):
        two.prepare(decode(2, prefix=248, output=9), page_size=256)
    with pytest.raises(PredictorError, match="FIFO head"):
        two.process_result(1)
    retired_first = two.process_result(0)
    assert retired_first.request_for("a").kv_committed_len == 246
    retired_second = retired_first.process_result(1)
    assert retired_second.request_for("a").kv_committed_len == 248
    assert retired_second.request_for("a").processed_output_tokens == 9
    assert retired_second.pending == ()
    with pytest.raises(PredictorError, match="FIFO head"):
        retired_second.process_result(1)
    assert len(two.pending) == 2


def test_publication_requires_the_exact_prepared_step_object_and_is_transactional():
    initial = DSparkReferenceLedger(requests=(seeded(),))
    before = decode()
    prepared = initial.prepare(before, page_size=256)
    equal_step = replace(before)
    assert equal_step == before and equal_step is not before
    with pytest.raises(PredictorError, match="exact prepared modeled result"):
        prepared.publish(modeled(equal_step))
    with pytest.raises(PredictorError, match="preparation precedes retirement"):
        prepared.prepare(decode(1), page_size=256)
    with pytest.raises(PredictorError, match="exact prepared modeled result"):
        initial.publish(modeled(before))
    assert initial.prepared is None and initial.pending == ()
    assert prepared.prepared.step is before
    assert prepared.publish(modeled(before)).pending[0].step is before


@pytest.mark.parametrize("unknown", [None, 0, 1, "ready"])
def test_relay_readiness_must_be_explicit_not_truthy(unknown):
    prepared = (
        first_published()
        .prepare(decode(1, prefix=246, output=7), page_size=256)
        .prepared
    )
    with pytest.raises(PredictorError, match="future-index readiness is unknown"):
        prepared.resolve(future_indices_ready=unknown, needs_cpu_seq_lens=True)


@pytest.mark.parametrize("unknown", [None, 0, 1, "cpu"])
def test_present_relay_requires_known_cpu_mirror_policy(unknown):
    prepared = (
        first_published()
        .prepare(decode(1, prefix=246, output=7), page_size=256)
        .prepared
    )
    with pytest.raises(PredictorError, match="CPU-mirror policy is unknown"):
        prepared.resolve(future_indices_ready=True, needs_cpu_seq_lens=unknown)


def test_absent_relay_keeps_seed_even_when_backend_needs_no_cpu_mirror():
    prepared = (
        first_published()
        .prepare(decode(1, prefix=246, output=7), page_size=256)
        .prepared
    )
    view = prepared.resolve(future_indices_ready=False, needs_cpu_seq_lens=False)
    assert view.seq_lens_cpu == view.planner_seq_lens_cpu == (240,)
    assert view.nxt_kv_lens_cpu == (256,)


@pytest.mark.parametrize("needs_cpu", [False, True])
def test_prefill_has_no_decode_relay_even_when_a_prior_chunk_was_published(needs_cpu):
    prepared = DSparkReferenceLedger().prepare(prefill(), page_size=256).prepared
    assert prepared.published_seq_lens == (None,)
    with pytest.raises(PredictorError, match="prefill does not resolve a decode relay"):
        prepared.resolve(future_indices_ready=True, needs_cpu_seq_lens=needs_cpu)
    assert prepared.resolve(
        future_indices_ready=False, needs_cpu_seq_lens=None
    ).seq_lens_cpu == (240,)
    middle_published = (
        DSparkReferenceLedger()
        .prepare(prepared.step, page_size=256)
        .publish(modeled(prepared.step))
    )
    final_prepared = middle_published.prepare(
        prefill(1, prefix=240, count=16, final=True), page_size=256
    ).prepared
    assert final_prepared.published_seq_lens == (240,)
    with pytest.raises(PredictorError, match="prefill does not resolve a decode relay"):
        final_prepared.resolve(future_indices_ready=True, needs_cpu_seq_lens=needs_cpu)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_native_allocation_and_relay_vectors_follow_selected_request_order(batch_size):
    rows = tuple(
        seeded(f"r{i}", committed=240 + i, published=240 + i) for i in range(8)
    )
    selected = tuple(reversed(rows))[:batch_size]
    requests = tuple(
        replace(request(row.rid, prefix=row.published_seq_len), output_tokens_before=1)
        for row in selected
    )
    before = decode(requests=requests)
    initial = DSparkReferenceLedger(requests=rows)
    prepared = initial.prepare(before, page_size=256)
    assert prepared.prepared.seed_seq_lens_cpu == tuple(
        row.kv_committed_len for row in selected
    )
    assert prepared.prepared.published_seq_lens == tuple(
        row.published_seq_len for row in selected
    )
    assert prepared.prepared.nxt_kv_lens_cpu == tuple(
        512 if row.kv_committed_len >= 245 else 256 for row in selected
    )
    for row in rows:
        if row not in selected:
            assert prepared.request_for(row.rid) is row
    assert initial.requests == rows


def test_bootstrap_middle_and_final_prefill_keep_allocation_publication_and_output_distinct():
    initial = DSparkReferenceLedger()
    middle = prefill()
    middle_prepared = initial.prepare(middle, page_size=256)
    row = middle_prepared.request_for("a")
    assert (row.kv_committed_len, row.kv_allocated_len) == (240, 240)
    assert row.published_seq_len is None and not row.prefill_complete
    assert middle_prepared.prepared.nxt_kv_lens_cpu is None
    middle_published = middle_prepared.publish(modeled(middle))
    assert middle_published.request_for("a").published_seq_len == 240
    with pytest.raises(PredictorError, match="decode differs"):
        middle_published.prepare(decode(1), page_size=256)

    final = prefill(1, prefix=240, count=16, final=True)
    final_prepared = middle_published.prepare(final, page_size=256)
    assert final_prepared.request_for("a").kv_committed_len == 256
    assert final_prepared.request_for("a").published_seq_len == 240
    synced = final_prepared.process_result(0)
    assert synced.prepared is final_prepared.prepared
    assert synced.request_for("a").kv_committed_len == 256
    final_published = synced.publish(modeled(final))
    row = final_published.request_for("a")
    assert row.prefill_complete and row.published_seq_len == 256
    assert row.processed_output_tokens == 0
    drained = final_published.process_result(1)
    assert drained.request_for("a").processed_output_tokens == 1
    assert drained.request_for("a").kv_committed_len == 256
    assert drained.prepare(
        decode(2, prefix=256), page_size=256
    ).prepared.nxt_kv_lens_cpu == (512,)
    with pytest.raises(PredictorError, match="chunk does not continue"):
        drained.prepare(prefill(2, prefix=256, count=8, final=True), page_size=256)


def test_explicit_prefix_admission_allocates_prefill_without_inventing_publication():
    before = prefill(prefix=384, count=128, final=True)
    prepared = DSparkReferenceLedger().prepare(before, page_size=256)
    row = prepared.request_for("a")
    assert row.kv_committed_len == row.kv_allocated_len == 512
    assert row.published_seq_len is None
    published = prepared.publish(modeled(before))
    assert published.request_for("a").published_seq_len == 512
    assert published.request_for("a").processed_output_tokens == 0
    assert published.process_result(0).request_for("a").processed_output_tokens == 1


def test_processing_preserves_full_native_overshoot_instead_of_budget_clipping():
    row = seeded(committed=100, published=100, maximum=2)
    before = decode(prefix=100, maximum=2)
    published = (
        DSparkReferenceLedger(requests=(row,))
        .prepare(before, page_size=256)
        .publish(modeled(before, count=6))
    )
    assert published.request_for("a").published_seq_len == 106
    processed = published.process_result(0)
    assert processed.request_for("a").kv_committed_len == 106
    assert processed.request_for("a").processed_output_tokens == 7
    assert published.request_for("a").kv_committed_len == 100
    assert published.request_for("a").processed_output_tokens == 1


def test_explicit_finished_skip_drains_then_allows_forget_clear_and_request_id_reuse():
    published = first_published(maximum=2)
    skipped = published.process_result(0, skipped_request_ids=("a",))
    assert skipped.pending == ()
    assert skipped.request_for("a").kv_committed_len == 240
    assert skipped.request_for("a").processed_output_tokens == 1
    assert not skipped.request_for("a").active
    with pytest.raises(PredictorError, match="requires explicit retirement"):
        skipped.prepare(decode(1, prefix=246), page_size=256)
    forgotten = skipped.forget("a")
    assert forgotten.requests == ()
    cleared = forgotten.clear()
    assert cleared.last_forward_id == 0
    with pytest.raises(PredictorError, match="preparation precedes retirement"):
        cleared.prepare(prefill(), page_size=256)
    reused = cleared.prepare(prefill(1, count=32, final=True), page_size=256)
    assert reused.request_for("a").published_seq_len is None
    assert reused.request_for("a").processed_output_tokens == 0
    assert (
        reused.publish(modeled(reused.prepared.step))
        .process_result(1)
        .request_for("a")
        .processed_output_tokens
        == 1
    )


def test_finished_skip_cannot_revive_a_later_pending_result_without_an_explicit_skip():
    one = first_published()
    second = decode(1, prefix=246, output=7)
    two = one.prepare(second, page_size=256).publish(modeled(second))
    skipped = two.process_result(0, skipped_request_ids=("a",))
    with pytest.raises(PredictorError, match="inactive pending request"):
        skipped.process_result(1)
    with pytest.raises(PredictorError, match="still owns prepared or pending work"):
        skipped.forget("a")
    drained = skipped.process_result(1, skipped_request_ids=("a",))
    assert drained.request_for("a").kv_committed_len == 240
    assert drained.request_for("a").processed_output_tokens == 1
    assert drained.forget("a").requests == ()


@pytest.mark.parametrize("phase", ["prepared", "published"])
def test_forget_and_clear_require_ownership_drain(phase):
    before = decode()
    initial = DSparkReferenceLedger(requests=(seeded(), seeded("other")))
    ledger = initial.prepare(before, page_size=256)
    if phase == "published":
        ledger = ledger.publish(modeled(before))
    with pytest.raises(PredictorError, match="still owns prepared or pending work"):
        ledger.forget("a")
    with pytest.raises(PredictorError, match="pending-result drain"):
        ledger.clear()
    assert ledger.forget("other").request_for("a") == ledger.request_for("a")
    assert initial.request_for("other").rid == "other"


@pytest.mark.parametrize("skipped", [("missing",), ("a", "a"), ["a"], (True,)])
def test_result_skips_require_unique_explicit_ids_from_the_head(skipped):
    ledger = first_published()
    with pytest.raises(PredictorError, match="skipped request ownership"):
        ledger.process_result(0, skipped_request_ids=skipped)
    assert len(ledger.pending) == 1
    assert ledger.request_for("a").kv_committed_len == 240


def test_prefill_cannot_overwrite_an_unprocessed_decode():
    ledger = first_published()
    with pytest.raises(PredictorError, match="cannot overwrite an unsettled decode"):
        ledger.prepare(prefill(1, prefix=246, count=32, final=True), page_size=256)


@pytest.mark.parametrize("page_size", [True, 0, -1, 256.0, None])
def test_reference_prepare_rejects_unknown_or_noninteger_page_size(page_size):
    ledger = DSparkReferenceLedger(requests=(seeded(),))
    with pytest.raises(PredictorError, match="B1-B64 page domain"):
        ledger.prepare(decode(), page_size=page_size)
    assert ledger.prepared is None


@pytest.mark.parametrize("tp,dp", [(2, 2)])
def test_reference_ledger_does_not_extend_to_other_supported_dspark_layouts(tp, dp):
    ledger = DSparkReferenceLedger(
        requests=(seeded(committed=512, allocated=512, published=512),)
    )
    with pytest.raises(PredictorError, match="TP2/TP4 ADP1/PP1"):
        ledger.prepare(step(tp, dp), page_size=256)


def test_reference_ledger_rejects_batch_65_and_mixed_mode():
    rows = tuple(seeded(str(i)) for i in range(65))
    before = decode(requests=tuple(request(row.rid, prefix=240) for row in rows))
    with pytest.raises(PredictorError, match="B1-B64"):
        DSparkReferenceLedger(requests=rows).prepare(before, page_size=256)
    before = decode()
    mixed = replace(before, groups=(replace(before.groups[0], forward_mode="MIXED"),))
    with pytest.raises(PredictorError, match="TP2/TP4 ADP1/PP1"):
        DSparkReferenceLedger(requests=(seeded(),)).prepare(mixed, page_size=256)


@pytest.mark.parametrize("change", [{"target_prefix_len": 239}, {"extend_len": 5}])
def test_decode_cannot_substitute_unpublished_history_or_other_work(change):
    initial = DSparkReferenceLedger(requests=(seeded(),))
    before = decode()
    modified = replace(before.requests[0], **change, draft_valid_to=239)
    before = replace(before, groups=(replace(before.groups[0], requests=(modified,)),))
    with pytest.raises(PredictorError, match="decode differs"):
        initial.prepare(before, page_size=256)


def test_decode_requires_admission_and_preserves_the_request_budget():
    with pytest.raises(PredictorError, match="no explicit prefill admission"):
        DSparkReferenceLedger().prepare(decode(), page_size=256)
    with pytest.raises(PredictorError, match="decode differs"):
        DSparkReferenceLedger(requests=(seeded(),)).prepare(
            decode(maximum=21), page_size=256
        )
    before = prefill()
    before = replace(
        before,
        groups=(
            replace(
                before.groups[0],
                requests=(replace(before.requests[0], output_tokens_before=1),),
            ),
        ),
    )
    with pytest.raises(PredictorError, match="invent processed output history"):
        DSparkReferenceLedger().prepare(before, page_size=256)


def test_already_processed_budget_completion_cannot_start_another_decode():
    published = first_published(maximum=2)
    settled = published.process_result(0)
    assert settled.request_for("a").processed_output_tokens == 7
    with pytest.raises(PredictorError, match="decode differs"):
        settled.prepare(decode(1, prefix=246, maximum=2), page_size=256)
    assert settled.pending == () and settled.prepared is None


def test_decode_cannot_disable_its_native_output_publication():
    before = decode()
    before = replace(
        before,
        groups=(
            replace(
                before.groups[0],
                requests=(replace(before.requests[0], emits_output=False),),
            ),
        ),
    )
    with pytest.raises(PredictorError, match="decode differs"):
        DSparkReferenceLedger(requests=(seeded(),)).prepare(before, page_size=256)


def test_final_prefill_cannot_publish_an_arbitrary_valid_progress_count():
    before = prefill(count=32, final=True)
    result = modeled(before)
    # DSpark's general typed progress permits this count, while the native
    # homogeneous prefill publication has exactly one bonus output token.
    result = replace(
        result,
        progress=replace(result.progress, committed_tokens=(("a", 2),)),
    )
    prepared = DSparkReferenceLedger().prepare(before, page_size=256)
    with pytest.raises(PredictorError, match="native zero/one bonus"):
        prepared.publish(result)
    assert prepared.request_for("a").published_seq_len is None
    assert prepared.publish(modeled(before)).request_for("a").published_seq_len == 32


@pytest.mark.parametrize(
    "change",
    [
        {"kv_committed_len": True},
        {"kv_committed_len": -1},
        {"kv_committed_len": 257},
        {"kv_allocated_len": 256.0},
        {"published_seq_len": 257},
        {"published_seq_len": False},
        {"processed_output_tokens": -1},
        {"max_new_tokens": 0},
        {"prefill_complete": 1},
        {"active": 1},
    ],
)
def test_seed_metadata_is_strict_and_immutable(change):
    row = seeded()
    with pytest.raises(PredictorError, match="metadata is inconsistent"):
        replace(row, **change)
    assert row.kv is row
    with pytest.raises(FrozenInstanceError):
        row.kv_committed_len = 246


@pytest.mark.parametrize(
    "vectors", [([240], (256,)), ((True,), (256,)), (None, None), ((240,), (256, 512))]
)
def test_reference_view_rejects_mutable_unknown_or_misaligned_cpu_vectors(vectors):
    with pytest.raises(PredictorError, match="immutable known vectors"):
        ReferenceCPUView(*vectors)


def test_prepared_vectors_cannot_hide_mutability_or_boolean_coordinates():
    prepared = (
        DSparkReferenceLedger(requests=(seeded(),))
        .prepare(decode(), page_size=256)
        .prepared
    )
    for changes in (
        {"seed_seq_lens_cpu": [240]},
        {"nxt_kv_lens_cpu": (True,)},
        {"published_seq_lens": [240]},
        {"published_seq_lens": (False,)},
    ):
        with pytest.raises(PredictorError, match="immutable ordered CPU metadata"):
            replace(prepared, **changes)
    assert type(prepared) is PreparedReferenceBatch
    with pytest.raises(FrozenInstanceError):
        prepared.seed_seq_lens_cpu = (246,)


def test_explicit_pending_seeds_require_unique_order_and_published_forward_ids():
    first = modeled(decode())
    second = modeled(decode(1))
    for pending, last_forward_id in (
        ((first,), -1),
        ((first, first), 0),
        ((second, first), 1),
        ((first, second, modeled(decode(2))), 2),
    ):
        with pytest.raises(PredictorError):
            DSparkReferenceLedger(
                requests=(seeded(),), pending=pending, last_forward_id=last_forward_id
            )
    with pytest.raises(PredictorError, match="typed unique requests"):
        DSparkReferenceLedger(requests=(seeded(), seeded()))


def test_tp4_b1_reference_uses_same_source_page_and_publication_order():
    original = DSparkReferenceLedger(requests=(seeded(),))
    before = decode()
    tp4 = replace(before, topology=DSparkTopology(4, 1))
    expected = original.prepare(before, page_size=256)
    actual = original.prepare(tp4, page_size=256)
    assert actual.prepared.seed_seq_lens_cpu == expected.prepared.seed_seq_lens_cpu
    assert actual.prepared.nxt_kv_lens_cpu == expected.prepared.nxt_kv_lens_cpu
    published = actual.publish(modeled(tp4, count=4))
    assert published.request_for("a").kv_committed_len == 240
    assert published.request_for("a").published_seq_len == 244
    assert published.process_result(0).request_for("a").kv_committed_len == 244


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("batch", [3, 4, 64])
def test_ordered_ragged_batch_uses_each_native_page_and_full_commit(tp, batch):
    rows = tuple(
        seeded(str(i), committed=240 + i, allocated=512, published=240 + i)
        for i in range(batch)
    )
    requests = tuple(request(row.rid, prefix=row.published_seq_len) for row in rows)
    before = replace(decode(requests=requests), topology=DSparkTopology(tp, 1))
    original = DSparkReferenceLedger(requests=rows)
    actual = original.prepare(before, page_size=256)
    assert actual.prepared.seed_seq_lens_cpu == tuple(240 + i for i in range(batch))
    view = actual.prepared.resolve(future_indices_ready=True, needs_cpu_seq_lens=False)
    assert view.seq_lens_cpu is None
    assert view.nxt_kv_lens_cpu == actual.prepared.nxt_kv_lens_cpu
    assert len(view.planner_seq_lens_cpu) == batch
    result = modeled(before, count=4)
    published = actual.publish(result)
    for row in rows:
        assert original.request_for(row.rid) == row
        assert published.request_for(row.rid).kv_committed_len == row.kv_committed_len
        assert (
            published.request_for(row.rid).published_seq_len
            == row.published_seq_len + 4
        )
    settled = published.process_result(before.forward_id)
    assert settled.pending == ()
    for row in rows:
        assert settled.request_for(row.rid).kv_committed_len == row.kv_committed_len + 4
        assert (
            settled.request_for(row.rid).processed_output_tokens
            == row.processed_output_tokens + 4
        )


@pytest.mark.parametrize("failure", ["last-prefix", "last-budget", "wrong-order"])
def test_large_batch_failure_preserves_immutable_reference_ledger(failure):
    rows = tuple(seeded(str(i)) for i in range(64))
    requests = tuple(request(row.rid, prefix=240) for row in rows)
    before = replace(decode(requests=requests), topology=DSparkTopology(4, 1))
    original = DSparkReferenceLedger(requests=rows)
    if failure == "wrong-order":
        prepared = original.prepare(before, page_size=256)
        reordered = replace(
            before, groups=(replace(before.groups[0], requests=requests[::-1]),)
        )
        with pytest.raises(PredictorError, match="exact prepared"):
            prepared.publish(modeled(reordered))
        assert prepared.prepared.step is before and prepared.pending == ()
    else:
        changes = (
            {"target_prefix_len": 239}
            if failure == "last-prefix"
            else {
                "max_new_tokens": 19,
                "sampling_params_json": '{"is_normalized":true,"max_new_tokens":19}',
            }
        )
        last = replace(requests[-1], **changes, draft_valid_to=239)
        changed = replace(
            before,
            groups=(replace(before.groups[0], requests=requests[:-1] + (last,)),),
        )
        with pytest.raises(PredictorError, match="decode differs"):
            original.prepare(changed, page_size=256)
    assert (
        original.requests == rows
        and original.prepared is None
        and original.pending == ()
    )


def test_reference_cpu_view_cannot_truncate_or_coerce_large_vectors():
    assert ReferenceCPUView(None, tuple(range(64))).planner_seq_lens_cpu == tuple(
        range(64)
    )
    for value in (tuple(range(65)), (True,) * 64, list(range(64))):
        with pytest.raises(PredictorError, match="immutable known vectors"):
            ReferenceCPUView(None, value)
