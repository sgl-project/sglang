# SPDX-License-Identifier: Apache-2.0
"""Actual host/file policy movement and request-admission reference ownership."""

from array import array
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import test_kv_control_cache as cache_fixtures
import test_kv_control_storage as storage_helpers
import test_kv_control_storage_reads as reads
import test_kv_control_transfers as transfers
import test_kv_policy_runtime as retention_helpers
import test_kv_quota_guidance as requests

from sglang.srt.managers.kv_control.contracts import SessionTarget, Status
from sglang.srt.managers.kv_control.policy_runtime import (
    PolicyDefinition,
    RetentionOptions,
    ToolWaitDefinition,
)
from sglang.srt.managers.kv_control.service import Ack, ServiceConfig
from sglang.srt.managers.kv_control.tool_wait_policy import ToolWaitOptions
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")
owner = transfers.owner
FULL = ComponentType.FULL
FORCED = ToolWaitOptions(
    min_pressure=0, min_pause_seconds=0.001, initial_estimated_transfer_seconds=0.001
)


def config(*, tier="host", forced=True):
    options = (
        msgspec.structs.replace(FORCED, tier=tier)
        if forced
        else ToolWaitOptions(tier=tier)
    )
    return ServiceConfig(
        namespace="ns",
        engine_id="engine",
        policies=(
            PolicyDefinition(
                name="retain", options=RetentionOptions(priority=40, ttl_seconds=10)
            ),
            ToolWaitDefinition(name="tools", options=options),
        ),
    )


with_host = pytest.mark.parametrize("owner", [config()], indirect=True)


def initialize(owner):
    _, cache, service, dma = owner
    node = transfers.seed(cache)
    requests.task(service, "task")
    return node, SessionTarget(namespace="ns", session_id="task")


def signal(service, target, kind, *, message, pause=None):
    parent = retention_helpers.submit(
        service,
        target=target,
        signal=kind,
        message=message,
        **({"resume_after_seconds": pause} if pause is not None else {}),
    ).results[0]
    assert parent.status == Status.COMPLETED, parent
    return parent, tuple(
        service.control.registry.get(ref) for ref in parent.update.derived_operations
    )


def save(owner, target):
    _, cache, service, dma = owner
    _, children = signal(service, target, "tool_wait", message="pause", pause=10)
    (child,) = children
    service.progress()
    dma.finish()
    done = storage_helpers.wait(service, child)
    assert done.status == Status.COMPLETED, done
    return done


def begin_resume(owner):
    _, cache, service, dma = owner
    req = requests.request(service, "task")
    requests.observe(cache, service, (req,))
    assert not service.prepare_admission(req)
    reads.wait_for(service, lambda: bool(dma.pending))
    return req


@pytest.mark.parametrize(
    "owner", [config(), (config(tier="storage"), "file_namespaced")], indirect=True
)
def test_save_restore_priority_dependency_and_actual_prefill_handoff(owner):
    _, cache, service, dma = owner
    node, target = initialize(owner)
    save(owner, target)
    assert node.component_data[FULL].value is None
    if cache.enable_storage:
        assert cache.evict_host(4) == 4
    req = begin_resume(owner)
    # Retention was offered before COPY, but cannot touch an unpublished copy.
    parent = service.control.registry.query(
        req.kv_control_context, "bind-task"
    ).results[0]
    retention = next(
        service.control.registry.get(ref)
        for ref in parent.update.derived_operations
        if service.control.registry.get(ref).action_type == "sglang.set_priority"
    )
    assert retention.status == Status.QUEUED
    dma.finish()
    reads.wait_for(
        service,
        lambda: any(w.ready for w in service.policy_runtime.execution.works.values()),
    )
    requests.observe(cache, service, (req,))
    assert service.prepare_admission(req)
    current = service.control.executor.residency(
        service.control.index.resolve(cache_fixtures.target())
    )
    assert [item.effective_priority for item in current.items] == [40]
    req.init_next_round_input(cache)
    adder = PrefillAdder(
        1,
        cache,
        cache.token_to_kv_pool_allocator,
        SimpleNamespace(reqs=[]),
        1,
        64,
        None,
    )
    adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)
    assert req in adder.can_run_list and req.lock is not None
    try:
        service.policy_request_admitted(req)
        assert not service.control.executor.preparations
        assert service.policy_runtime.execution.stats["preparations_consumed"] == 1
        assert service.policy_runtime.execution.stats["consumed_prefix_tokens"] == 4
        cache.evict(EvictParams(num_tokens=4))
        assert cache.token_to_kv_pool_allocator.available_size() == 60
    finally:
        cache.unlock(req.lock)
        req.lock = None
    service.progress()
    assert not service.has_pending


@pytest.mark.parametrize(
    "owner,pause",
    [(config(forced=False), 10), (config(), 0.001), (config(), None)],
    indirect=["owner"],
)
def test_fit_short_and_unknown_pause_do_not_move(owner, pause):
    _, cache, service, dma = owner
    node, target = initialize(owner)
    parent, children = signal(service, target, "tool_wait", message="keep", pause=pause)
    assert not children and not dma.pending
    _, children = signal(service, target, "resume", message="resume")
    # Retention may act, but no placement operation was proposed.
    assert all(child.action_type == "sglang.set_priority" for child in children)
    assert node.component_data[FULL].value is not None
    assert not service.has_pending


@with_host
def test_abort_during_h2d_drains_before_retiring_and_does_not_apply_dependent_priority(
    owner,
):
    _, cache, service, dma = owner
    _, target = initialize(owner)
    save(owner, target)
    req = begin_resume(owner)
    copy_work = next(
        w
        for w in service.policy_runtime.execution.works.values()
        if w.entry.proposal.prepare_for_admission
    )
    copy_ref = copy_work.ref
    service.finish_request(req, aborted=True)
    assert service.control.registry.get(copy_ref).status == Status.CANCELLING
    assert service.has_pending and service.blocks_prefill
    dma.finish()
    reads.wait_for(service, lambda: not service.has_pending)
    assert service.control.registry.get(copy_ref).status == Status.CANCELLED
    assert (
        not service.policy_runtime.execution.works
        and not service.control.executor.preparations
    )


@with_host
def test_resume_cancels_pending_save_and_uses_actual_resident_prefix(owner):
    _, cache, service, dma = owner
    node, target = initialize(owner)
    _, children = signal(service, target, "tool_wait", message="pause", pause=10)
    move_ref = children[0].ref
    service.progress()
    assert dma.pending
    req = requests.request(service, "task")
    requests.observe(cache, service, (req,))
    assert service.prepare_admission(req)
    # No H2D was needed; the owner holds the real existing GPU prefix.
    assert len(dma.pending) == 1
    assert any(w.ready for w in service.policy_runtime.execution.works.values())
    dma.finish()
    service.progress()
    assert service.control.registry.get(move_ref).status == Status.CANCELLED
    assert node.component_data[FULL].value is not None
    service.finish_request(req, aborted=True)
    reads.wait_for(service, lambda: not service.has_pending)


@with_host
def test_expiry_and_decode_pressure_revoke_unconsumed_preparation(owner):
    case, cache, service, dma = owner
    _, target = initialize(owner)
    save(owner, target)
    req = begin_resume(owner)
    dma.finish()
    reads.wait_for(
        service,
        lambda: any(w.ready for w in service.policy_runtime.execution.works.values()),
    )
    service.control.executor.gpu_copy_capacity = lambda: -1
    service.progress()
    assert not service.control.executor.preparations
    assert not service.policy_runtime.execution.works
    assert service.policy_runtime.execution.stats["preparations_revoked"] == 1
    cache.evict(EvictParams(num_tokens=4))
    assert cache.token_to_kv_pool_allocator.available_size() == 64
    case.clock.now += 2
    assert service.prepare_admission(req)


@with_host
def test_reset_cancels_prediction_before_due_without_starting_io(owner):
    case, cache, service, dma = owner
    _, target = initialize(owner)
    save(owner, target)
    _, children = signal(
        service, target, "resume_prediction", message="predict", pause=20
    )
    assert any(child.action_type == "sglang.copy_to" for child in children)
    assert not dma.pending
    service.prepare_reset()
    service.progress()
    assert not service.has_pending and not service.policy_runtime.execution.works
    assert not dma.pending


@with_host
def test_whole_request_headroom_is_checked_before_prefix_copy(owner):
    _, cache, service, dma = owner
    _, target = initialize(owner)
    save(owner, target)
    other = cache.token_to_kv_pool_allocator.alloc(60)
    req = requests.request(service, "task")
    requests.observe(cache, service, (req,))
    try:
        assert not service.prepare_admission(req)
        service.progress()
        assert not dma.pending and not service.control.executor.preparations
        assert cache.token_to_kv_pool_allocator.available_size() == 4
        cache.token_to_kv_pool_allocator.free(other[:3])
        assert not service.prepare_admission(req)
        reads.wait_for(service, lambda: bool(dma.pending))
        dma.finish()
        reads.wait_for(
            service,
            lambda: any(
                w.ready for w in service.policy_runtime.execution.works.values()
            ),
        )
        assert cache.token_to_kv_pool_allocator.available_size() == 3
        service.finish_request(req, aborted=True)
        reads.wait_for(service, lambda: not service.has_pending)
    finally:
        cache.token_to_kv_pool_allocator.free(other[3:])


@with_host
def test_live_explicit_priority_vetoes_automatic_move(owner):
    case, cache, service, dma = owner
    node, target = initialize(owner)
    case.priority(service.control, value=90)
    _, children = signal(
        service, target, "tool_wait", message="paused-explicit", pause=10
    )
    assert len(children) == 1
    result = service.control.registry.get(children[0].ref)
    assert result.status == Status.DEFERRED and result.update.code == "explicit_intent"
    assert node.component_data[FULL].value is not None and not dma.pending
    assert not service.has_pending


@with_host
def test_acknowledged_copy_keeps_separate_preparation_until_task_cleanup(owner):
    _, cache, service, dma = owner
    _, target = initialize(owner)
    save(owner, target)
    _req = begin_resume(owner)
    dma.finish()
    reads.wait_for(
        service,
        lambda: any(w.ready for w in service.policy_runtime.execution.works.values()),
    )
    work = next(w for w in service.policy_runtime.execution.works.values() if w.ready)
    service.execute(Ack(context=work.ref.context, message_id=work.ref.message_id))
    assert work.lease.ready and service.has_pending
    finished = retention_helpers.submit(
        service,
        "finish_task",
        message="finish-after-ack",
        session_id="task",
        expected_revision=service.control.index.session_revision("task"),
    )
    assert finished.results[0].status == Status.COMPLETED
    assert not service.control.executor.preparations and not service.has_pending


@with_host
def test_explicit_move_revokes_preparation_and_preserves_its_own_intent(owner):
    _, cache, service, dma = owner
    _, target = initialize(owner)
    save(owner, target)
    _req = begin_resume(owner)
    dma.finish()
    reads.wait_for(
        service,
        lambda: any(w.ready for w in service.policy_runtime.execution.works.values()),
    )
    explicit = transfers.transfer(
        service, kind="move_to", selection=target, message="explicit"
    )
    assert not service.control.executor.preparations
    reads.wait_for(
        service, lambda: service.control.registry.get(explicit.ref).status.terminal
    )
    assert service.control.registry.get(explicit.ref).status == Status.COMPLETED
    assert cache.token_to_kv_pool_allocator.available_size() == 64
    assert not service.has_pending


@with_host
def test_actual_resume_supersedes_queued_prediction_without_poisoning_priority_dependency(
    owner,
):
    _, cache, service, dma = owner
    _, target = initialize(owner)
    save(owner, target)
    _, predicted = signal(
        service, target, "resume_prediction", message="old-prediction", pause=20
    )
    req = begin_resume(owner)
    assert service.control.registry.get(predicted[0].ref).status == Status.CANCELLED
    dma.finish()
    reads.wait_for(
        service,
        lambda: any(w.ready for w in service.policy_runtime.execution.works.values()),
    )
    parent = service.control.registry.query(
        req.kv_control_context, "bind-task"
    ).results[0]
    children = [
        service.control.registry.get(ref) for ref in parent.update.derived_operations
    ]
    assert all(child.status == Status.COMPLETED for child in children), children
    service.finish_request(req, aborted=True)
    reads.wait_for(service, lambda: not service.has_pending)


@pytest.mark.parametrize(
    "owner", [(config(tier="storage"), "file_namespaced")], indirect=True
)
def test_existing_inference_prefetch_publishes_before_policy_load_without_duplicate_query(
    owner,
):
    _, cache, service, dma = owner
    _, target = initialize(owner)
    save(owner, target)
    cache.evict_host(4)
    req = requests.request(service, "task")
    cache.prefetch_from_storage(
        req.cache_request_handle,
        cache.root_node_handle(),
        array("q", req.origin_input_ids),
        cache_salt=req.cache_salt,
        minimum_prefetch_tokens=1,
    )
    requests.observe(cache, service, (req,))
    assert req.cache_request_handle in cache.ongoing_prefetch
    assert not service.prepare_admission(req)
    assert not service.control.executor.storage_reads.jobs

    def inference_prefetch_finished():
        cache.drain_storage_control_queues()
        return req.cache_request_handle not in cache.ongoing_prefetch

    reads.wait_for(service, inference_prefetch_finished)
    cache.pop_prefetch_loaded_span(req.cache_request_handle)
    cache.discard_storage_prefetch_accounting(req.cache_request_handle)
    requests.observe(cache, service, (req,))
    with patch.object(
        cache.cache_controller,
        "_storage_hit_query",
        side_effect=AssertionError("duplicate cold query"),
    ):
        assert not service.prepare_admission(req)
        reads.wait_for(service, lambda: bool(dma.pending))
        dma.finish()
        reads.wait_for(
            service,
            lambda: any(
                w.ready for w in service.policy_runtime.execution.works.values()
            ),
        )
    service.finish_request(req, aborted=True)
    reads.wait_for(service, lambda: not service.has_pending)


@with_host
@pytest.mark.parametrize("explicit_priority", [False, True])
def test_attached_tool_save_waits_for_checkpoint_and_explicit_completion(
    owner, explicit_priority
):
    from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
    from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    _, cache, service, dma = owner
    node, target = initialize(owner)
    req = Req(
        rid="finishing",
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2, 3, 4, 9]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
    )
    service.prepare_request(req)
    actions = [
        KvHintAction(
            action_id="pause",
            action_type="sglang.signal",
            action_version="1.0",
            payload={
                "target": msgspec.json.decode(msgspec.json.encode(target)),
                "signal": "tool_wait",
                "resume_after_seconds": 10,
            },
        )
    ]
    if explicit_priority:
        actions.append(
            KvHintAction(
                action_id="explicit",
                action_type="sglang.set_priority",
                action_version="1.0",
                payload={
                    "target": "request_cached",
                    "location": {"tier": "gpu"},
                    "claim_id": "explicit",
                    "revision": 1,
                    "value": 90,
                    "ttl_seconds": 10,
                },
            )
        )
    envelope = KvHintsEnvelope(
        protocol_version="0.1", message_id="finish-pause", actions=actions
    )
    service.attach(req, envelope)
    context = req.kv_control_context
    results = service.control.registry.query(context, envelope.message_id).results
    (save_ref,) = results[0].update.derived_operations
    service.progress()
    assert service.control.registry.get(save_ref).status == Status.QUEUED
    assert not dma.pending
    req.init_next_round_input(cache)
    req.lock = cache.lock(node.id)
    try:
        req.finished_reason = FINISH_LENGTH(1)
        service.finish_request(req, committed_len=4, checkpointed=True)
    finally:
        cache.unlock(req.lock)
        req.lock = None
    if explicit_priority:
        service.progress()
        assert service.control.registry.get(save_ref).status == Status.CANCELLED
        assert not dma.pending and node.component_data[FULL].value is not None
        assert service.control.executor.claims.effective(node, FULL) == 90
    else:
        service.progress()
        assert dma.pending and node.component_data[FULL].lock_ref > 0
        dma.finish()
        reads.wait_for(service, lambda: not service.has_pending)
        assert service.control.registry.get(save_ref).status == Status.COMPLETED
        assert node.component_data[FULL].value is None
