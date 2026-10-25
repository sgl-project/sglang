"""Explicit native DP source transactions through the existing CPU TP group.

The library joins both local reference owners, predicts once, builds both native
results, then publishes and replicates history. Public factory/clock guards stay
closed. A separate source-owned resolver must supply complete joint executions.
"""

import json
import time
from dataclasses import replace

from sglang_simulator.simulation.sglang.dspark import (
    DSparkGroup,
    DSparkStep,
    DSparkStepResult,
    DSparkTopology,
)
from sglang_simulator.simulation.sglang.dspark_runtime import canonical
from sglang_simulator.time_predictor.base import PredictorError


def _require(valid, message):
    if not valid:
        raise PredictorError("invalid_dspark_step", message)


def _native_group(tp_group):
    from sglang.srt.distributed.parallel_state import GroupCoordinator

    _require(
        type(tp_group) is GroupCoordinator
        and type(tp_group.world_size) is int
        and tp_group.world_size == 2
        and type(tp_group.ranks) is list
        and all(type(rank) is int for rank in tp_group.ranks)
        and len(tp_group.ranks) == len(set(tp_group.ranks)) == 2
        and type(tp_group.rank_in_group) is int
        and tp_group.rank_in_group in (0, 1)
        and tp_group.cpu_group is not None,
        "peer exchange requires the actual two-rank native CPU TP group",
    )


def _exchange(tp_group, event, make):
    """Rendezvous success or local failure through the existing native group."""
    failure = None
    value = None
    try:
        value = make()
    except BaseException as error:  # noqa: BLE001 - rendezvous, then re-raise unchanged
        failure = error
    own = {
        "event": event,
        "value": value,
        "error": None if failure is None else repr(failure),
    }
    rows = tp_group.all_gather_object(own)
    if failure is not None:
        raise failure
    _require(
        type(rows) is list
        and len(rows) == 2
        and all(
            type(row) is dict
            and set(row) == set(own)
            and row["event"] == event
            and row["error"] is None
            for row in rows
        ),
        "native DP peer failed at " + event,
    )
    _require(
        rows[tp_group.rank_in_group] == own,
        "native DP exchange changed its local value",
    )
    return tuple(row["value"] for row in rows)


def _history_context(runtime):
    return {
        "forward_id": runtime._forward_id,
        "last_committed": runtime.session._last_forward_id,
        "history": dict(runtime.session._history),
        "runtime": runtime.runtime_profile_json,
    }


def select_native_dp_peer_batch(
    runtime,
    scheduler,
    batch,
    *,
    tp_group,
    attention_dp_rank,
    native_forward_ct,
    prior_requests=(),
):
    """Bind actual selection/allocation before local source overlap decisions."""
    _native_group(tp_group)
    driver = runtime.reference_driver

    def entry():
        _require(
            driver is not None and runtime._peer_failure is None,
            "native DP selection requires its live reference driver",
        )
        driver.check()
        return _history_context(runtime)

    contexts = _exchange(tp_group, "selection entry", entry)
    _require(
        contexts[0] == contexts[1], "native DP histories/FIDs differ before selection"
    )
    before = capture_native_dp_peer_step(
        runtime,
        batch,
        tp_group=tp_group,
        attention_dp_rank=attention_dp_rank,
        native_forward_ct=native_forward_ct,
        reference=True,
    )
    candidate = None

    def select():
        nonlocal candidate
        candidate = driver.native_dp_selection(
            scheduler,
            batch,
            before,
            peer_context=(tp_group, attention_dp_rank, native_forward_ct),
            prior_requests=prior_requests,
        )
        return True

    _exchange(tp_group, "selection ready", select)
    driver.selection = candidate
    return before


def forward_native_dp_peer(
    runtime,
    *,
    worker,
    batch,
    tp_group,
    attention_dp_rank,
    native_forward_ct,
    joint_execution_resolver,
    on_publish=None,
):
    """One shared prediction, both local builds, then publication and commit.

    The explicit source-owned resolver supplies a complete resolved step after
    both local reference views. This neither fabricates an observation nor opens
    public routing/factory/clock guards.
    """
    _native_group(tp_group)
    started = time.perf_counter()
    driver = runtime.reference_driver
    transaction = result = candidate = bound = next_capacity = None
    try:

        def entry():
            _require(
                runtime._peer_failure is None and driver is not None,
                "native DP worker requires its live reference driver",
            )
            driver.check()
            selected = driver.selection
            _require(
                worker.query_token_num == 5
                and worker.verify_num_draft_tokens == 6
                and callable(joint_execution_resolver)
                and selected is not None
                and selected.peer_context
                == (tp_group, attention_dp_rank, native_forward_ct),
                "native DP worker/resolver has no matching source selection",
            )
            _require(
                not runtime._peer_retirements,
                "native DP retirement needs its explicit joint lifecycle event",
            )
            return _history_context(runtime)

        contexts = _exchange(tp_group, "worker entry", entry)
        _require(
            contexts[0] == contexts[1], "native DP histories/FIDs differ before work"
        )
        before = capture_native_dp_peer_step(
            runtime,
            batch,
            tp_group=tp_group,
            attention_dp_rank=attention_dp_rank,
            native_forward_ct=native_forward_ct,
            reference=True,
        )

        def prepare():
            nonlocal transaction
            transaction = driver.prepare_worker(batch, peer_step=before)
            return {
                "view": transaction.view,
                "execution": json.loads(
                    transaction.step.groups[attention_dp_rank].execution_json
                ),
            }

        views = _exchange(tp_group, "reference ready", prepare)

        def predict():
            if attention_dp_rank != 0:
                return None
            resolved = joint_execution_resolver(
                step=before,
                reference_views=tuple(row["view"] for row in views),
                local_executions=tuple(row["execution"] for row in views),
            )
            prediction = runtime.session.predict(
                transaction.with_resolved_executions(resolved).step
            )
            _require(
                tuple(cost.phase for cost in prediction.timing.phases)
                == ("whole_worker",),
                "native DP requires one exclusive whole-worker timing",
            )
            return prediction

        predictions = _exchange(tp_group, "joint prediction", predict)
        prediction = predictions[0]
        _require(
            type(prediction) is DSparkStepResult and predictions[1] is None,
            "native DP prediction must belong only to the leader",
        )

        def build():
            nonlocal transaction, result, candidate, bound, next_capacity
            transaction = transaction.with_resolved_executions(prediction.step)
            bound = transaction.bind_result(prediction)
            result = runtime.results.build(batch=batch, prediction=bound)
            candidate = driver.complete_candidate(transaction, bound)
            driver.validate_owner(transaction)
            next_capacity = runtime._draft_buffer_capacity
            # Pinned dspark_draft._run_forward uses bs=len(batch.seq_lens).
            # run_idle_participation never enters that persistent-buffer owner.
            if bound.local_group.forward_mode == "TARGET_VERIFY":
                next_capacity = max(next_capacity, len(batch.seq_lens))
            _require(
                _history_context(runtime) == contexts[attention_dp_rank],
                "native DP history changed during cost/build",
            )
            return True

        _exchange(tp_group, "native builds ready", build)
    except BaseException as error:
        if driver is not None:
            driver.fail("native DP preparation/cost/build", error)
        raise
    try:

        def publish():
            if on_publish is not None:
                on_publish(result.new_seq_lens)
            driver.validate_owner(transaction)
            _require(
                _history_context(runtime) == contexts[attention_dp_rank],
                "native DP history changed during publication",
            )
            return True

        _exchange(tp_group, "native publication", publish)

        def commit():
            runtime.session.commit(bound)
            driver.install(candidate)
            runtime._draft_buffer_capacity = next_capacity
            runtime._forward_id += 1
            return _history_context(runtime)

        contexts = _exchange(tp_group, "replicated commit", commit)
        _require(contexts[0] == contexts[1], "native DP replicated commits differ")
    except BaseException as error:
        runtime._peer_failure = repr(error)
        driver.fail("native DP publication/commit", error)
        raise
    from sglang_simulator.simulation.sglang.dspark import STEP_ATTRIBUTE

    setattr(
        result,
        STEP_ATTRIBUTE,
        replace(bound, prediction_wall_seconds=time.perf_counter() - started),
    )
    return result


def synchronize_native_dp_peer_lifecycle(
    runtime, *, tp_group, attention_dp_rank, reset=False
):
    """Apply source-owned drained retirement/reset to both existing Sessions."""
    _native_group(tp_group)
    driver = runtime.reference_driver
    candidate = None

    def prepare():
        nonlocal candidate
        _require(
            runtime._peer_failure is None
            and driver is not None
            and type(attention_dp_rank) is int
            and attention_dp_rank == tp_group.rank_in_group
            and type(reset) is bool,
            "invalid native DP lifecycle owner",
        )
        driver.check()
        _require(
            driver.selection is None and driver.processing is None,
            "native DP lifecycle still owns a source event",
        )
        candidate = driver.adapter.drained_reset() if reset else driver.adapter
        live = tuple(row.rid for row in driver.adapter.ledger.requests)
        _require(
            not set(runtime._peer_retirements) & set(live),
            "native DP retirement still owns local reference work",
        )
        return {
            "context": _history_context(runtime),
            "reset": reset,
            "retired": tuple(sorted(runtime._peer_retirements)),
            "live": live,
        }

    rows = _exchange(tp_group, "lifecycle ready", prepare)
    _require(
        rows[0]["context"] == rows[1]["context"]
        and rows[0]["reset"] == rows[1]["reset"],
        "native DP lifecycle histories/reset intent differ",
    )
    retired = set(rows[0]["retired"]) | set(rows[1]["retired"])
    _require(
        not retired & (set(rows[0]["live"]) | set(rows[1]["live"])),
        "native DP retirement still belongs to its peer",
    )

    def apply():
        if reset:
            runtime.session.clear()
            driver.adapter = candidate
        else:
            for rid in sorted(retired):
                runtime.session.forget(rid)
        runtime._peer_retirements.clear()
        return _history_context(runtime)

    try:
        contexts = _exchange(tp_group, "lifecycle applied", apply)
        _require(contexts[0] == contexts[1], "native DP lifecycle commits differ")
    except BaseException as error:
        runtime._peer_failure = repr(error)
        driver.fail("native DP lifecycle commit", error)
        raise


def capture_native_dp_peer_step(
    runtime, batch, *, tp_group, attention_dp_rank, native_forward_ct, reference=False
):
    """Capture both real partitions without using aggregate token-count columns.

    A native scheduler/worker adapter supplies its actual ``forward_ct`` and
    ``get_parallel().tp_group`` after native DP batch selection. Logical model
    FID and native scheduler FID are independently checked across both ranks.
    The caller still needs a joint-cost and publication/clock adapter before it
    can execute this unresolved step. One rank's local buffer is never copied
    into its peer's state.
    """
    # New native dependencies remain lazy: ADP1 and the library protocol do not
    # require torch, GroupCoordinator or the new SDK DP policy at import time.
    _native_group(tp_group)
    payload = dict.fromkeys(
        ("forward_id", "native_forward_ct", "runtime_profile_json", "group")
    )
    capture_error = None
    try:
        _require(
            runtime.topology == DSparkTopology(2, 2)
            and type(attention_dp_rank) is int
            and attention_dp_rank == tp_group.rank_in_group
            and type(native_forward_ct) is int
            and native_forward_ct >= 0
            and type(runtime._forward_id) is int
            and runtime._forward_id >= 0
            and type(runtime._draft_buffer_capacity) is int
            and runtime._draft_buffer_capacity >= 0,
            "peer exchange requires TP2/ADP2/PP1 and explicit before-work state",
        )
        payload.update(
            forward_id=runtime._forward_id,
            native_forward_ct=native_forward_ct,
            runtime_profile_json=runtime.runtime_profile_json,
        )
        local = runtime._capture_local_group(
            batch, attention_dp_rank=attention_dp_rank, reference=reference
        )
        local = replace(
            local,
            execution_json=canonical(
                {
                    "selection": "native_peer_before_work_unresolved",
                    "draft_buffer_capacity": runtime._draft_buffer_capacity,
                    "draft_buffer_device_matches": runtime._draft_buffer_capacity > 0,
                }
            ),
        )
        payload["group"] = local
    except Exception as error:  # noqa: BLE001 - rendezvous, then re-raise unchanged
        # Both real ranks must reach the rendezvous even if one local capture
        # fails. A missing group makes every participant reject before costs.
        capture_error = error
    peers = tp_group.all_gather_object(payload)
    if capture_error is not None:
        raise capture_error
    _require(type(peers) is list and len(peers) == 2, "missing native DP peer")
    for rank, peer in enumerate(peers):
        _require(
            type(peer) is dict
            and set(peer) == set(payload)
            and type(peer["forward_id"]) is int
            and peer["forward_id"] == runtime._forward_id
            and type(peer["native_forward_ct"]) is int
            and peer["native_forward_ct"] == native_forward_ct
            and peer["runtime_profile_json"] == runtime.runtime_profile_json
            and type(peer["group"]) is DSparkGroup
            and peer["group"].attention_dp_rank == rank,
            "native DP peer has different FID, runtime or ordered partition",
        )
        execution = json.loads(peer["group"].execution_json)
        _require(
            set(execution)
            == {
                "selection",
                "draft_buffer_capacity",
                "draft_buffer_device_matches",
            }
            and execution["selection"] == "native_peer_before_work_unresolved"
            and type(execution["draft_buffer_capacity"]) is int
            and execution["draft_buffer_capacity"] >= 0
            and type(execution["draft_buffer_device_matches"]) is bool
            and execution["draft_buffer_device_matches"]
            == (execution["draft_buffer_capacity"] > 0),
            "native DP peer omits its independent before-work buffer state",
        )
    _require(
        peers[attention_dp_rank] == payload,
        "native DP exchange changed its local captured partition",
    )
    step = DSparkStep(
        runtime._forward_id,
        runtime.topology,
        tuple(peer["group"] for peer in peers),
        runtime.runtime_profile_json,
    )
    _require(
        {group.forward_mode for group in step.groups} - {"IDLE"}
        in ({"EXTEND"}, {"TARGET_VERIFY"}),
        "native DP mixed-phase coordination remains unsupported",
    )
    return step
