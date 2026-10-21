# SPDX-License-Identifier: Apache-2.0
"""Actual request/tree/store keys must agree with logical control namespaces."""

import json
from array import array
from types import SimpleNamespace

import msgspec
import pytest
import test_kv_control_cache as fixtures
import torch

from sglang.srt.managers.kv_control.contracts import ControlError, Status, TokensTarget
from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt
from sglang.srt.managers.kv_control.service import KVControlService, ServiceConfig
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.utils import compute_node_hash_values
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture
def owner():
    assert torch.cuda.device_count() == 0
    case = fixtures.TestKVControlCache()
    case.setUp()
    try:
        cache, _ = case.make(session=False)
        cache.reset()

        def idle():
            return (
                len(cache.req_to_token_pool.free_slots) == cache.req_to_token_pool.size
            )

        service = KVControlService(
            cache,
            ServiceConfig(namespace="ns", engine_id="engine"),
            clock=case.clock,
            namespace_keys=True,
            idle_check=idle,
        )
        yield cache, service
        assert not service.has_pending
    finally:
        case.doCleanups()


def request(salt=None, session=None):
    return Req(
        rid="r",
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2, 3, 4]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
        cache_salt=salt,
        session=session,
    )


def complete(cache, service, req):
    service.prepare_request(req)
    req.output_ids = array("q", [5])
    req.finished_reason = FINISH_LENGTH(1)
    cache.req_to_token_pool.alloc([req])
    slots = cache.token_to_kv_pool_allocator.alloc(5)
    cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 5)), slots)
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 5
    release_kv_cache(req, cache, checkpoint=True)


def target(service, salt=None):
    return TokensTarget(
        namespace=service.config.namespace, token_ids=(1, 2, 3, 4, 5), cache_salt=salt
    )


def priority(service, selection="request_cached", context=None, message="keep"):
    selection = (
        selection
        if isinstance(selection, str)
        else msgspec.json.decode(msgspec.json.encode(selection))
    )
    result = service.control.submit(
        KvHintsEnvelope(
            protocol_version="0.1",
            message_id=message,
            actions=[
                KvHintAction(
                    action_id="a",
                    action_type="sglang.set_priority",
                    action_version="1.0",
                    payload={
                        "target": selection,
                        "location": {"tier": "gpu"},
                        "claim_id": "c",
                        "revision": 1,
                        "value": 90,
                        "ttl_seconds": 10,
                    },
                )
            ],
        ),
        context or service.control.context("router"),
    )
    return result.results[0]


@pytest.mark.parametrize("salt", [None, "tenant-a", "模型/租户", "a" * 64])
def test_ordinary_unhinted_request_and_independent_lookup_use_same_real_key(
    owner, salt
):
    from sglang.srt.managers.scheduler import Scheduler

    class TaggedReq(Req):
        pass

    class TaggedScheduler(Scheduler):
        def _create_generation_req(self, recv_req, *args, **kwargs):
            return TaggedReq(*args, **kwargs)

    cache, service = owner
    raw_node = fixtures.insert(cache, [1, 2, 3, 4, 5], salt=salt)
    scheduler = Scheduler.__new__(TaggedScheduler)
    scheduler.kv_control_service = service
    req = scheduler._new_generation_req(
        SimpleNamespace(kv_hints=None, mm_inputs=None),
        rid="r",
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2, 3, 4]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
        cache_salt=salt,
    )
    assert isinstance(req, TaggedReq)
    complete(cache, service, req)
    assert not hasattr(req, "kv_control_context")
    assert req.cache_salt == encode_cache_salt("ns", salt)
    assert req.kv_control_logical_cache_salt == salt
    result = priority(service, target(service, salt))
    assert result.status == Status.COMPLETED, result
    snapshot = service.control.executor.residency(
        service.control.index.resolve(target(service, salt))
    )
    assert snapshot.items and not snapshot.missing
    assert all(row.effective_priority == 90 for row in snapshot.items)
    assert (
        cache.tree_core.priority_claims.effective(raw_node, fixtures.ComponentType.FULL)
        is None
    )
    # Supplying an encoded physical salt as a logical salt cannot bypass the
    # namespace mapping or reach another tenant's data.
    encoded_as_logical = service.control.executor.residency(
        service.control.index.resolve(target(service, req.cache_salt))
    )
    assert not encoded_as_logical.items and encoded_as_logical.missing
    cache.sanity_check()


def test_attached_completion_uses_logical_salt_once_and_includes_generated_suffix(
    owner,
):
    cache, service = owner
    req = request("tenant-a")
    service.prepare_request(req)
    first = req.cache_salt
    service.prepare_request(req)
    assert req.cache_salt == first
    envelope = KvHintsEnvelope(
        protocol_version="0.1",
        message_id="attached",
        actions=[
            KvHintAction(
                action_id="a",
                action_type="sglang.set_priority",
                action_version="1.0",
                payload={
                    "target": "request_cached",
                    "location": {"tier": "gpu"},
                    "claim_id": "c",
                    "revision": 1,
                    "value": 90,
                    "ttl_seconds": 10,
                },
            )
        ],
    )
    service.attach(req, envelope)
    context = req.kv_control_context
    complete(cache, service, req)
    result = service.control.registry.query(context, "attached").results[0]
    assert result.status == Status.COMPLETED, result
    assert max(o.coverage.end_token for o in result.update.target_outcomes) == 5
    snap = service.control.executor.residency(
        service.control.index.resolve(target(service, "tenant-a"))
    )
    assert (
        snap.items
        and not snap.missing
        and all(i.effective_priority == 90 for i in snap.items)
    )


def test_new_namespace_cannot_match_old_real_entries_or_store_hashes(owner):
    cache, service = owner
    old_req = request("tenant-a")
    complete(cache, service, old_req)
    old_context = service.control.context("router")
    old_target = target(service, "tenant-a")
    priority(service, old_target)
    old_nodes = [
        n
        for n in cache.tree_core._node_arena.values()
        if n is not cache.tree_core.root_node
    ]

    def hashes(nodes):
        def depth(node):
            return len(node.key) + (
                depth(node.parent) if node.parent is not None else 0
            )

        result = []
        for node in sorted(nodes, key=depth):
            node.hash_value = compute_node_hash_values(node, 1)
            result.extend(node.hash_value)
        return result

    old_hashes = hashes(old_nodes)
    previous_epoch = service.control.registry.engine_epoch
    service.rebind_namespace("weights-v2")
    assert service.control.registry.engine_epoch != previous_epoch
    assert all(n.id in cache.tree_core._node_arena for n in old_nodes)
    new_target = target(service, "tenant-a")
    snapshot = service.control.executor.residency(
        service.control.index.resolve(new_target)
    )
    assert not snapshot.items and snapshot.missing
    with pytest.raises(ControlError) as error:
        service.control.registry.query(old_context, "keep")
    assert error.value.code == "stale_epoch"
    with pytest.raises(ControlError) as error:
        service.prepare_request(old_req)
    assert error.value.code == "stale_epoch"
    new_req = request("tenant-a")
    complete(cache, service, new_req)
    new_nodes = [
        n
        for n in cache.tree_core._node_arena.values()
        if n is not cache.tree_core.root_node and n.key.cache_salt == new_req.cache_salt
    ]
    new_hashes = hashes(new_nodes)
    assert new_nodes and set(old_hashes).isdisjoint(new_hashes)
    assert old_req.cache_salt != new_req.cache_salt
    assert all(
        cache.tree_core.priority_claims.effective(n, fixtures.ComponentType.FULL)
        is None
        for n in old_nodes
    )
    cache.sanity_check()


def test_rebind_checks_request_rows_and_correctness_refs_before_changing_state(owner):
    cache, service = owner
    req = request()
    complete(cache, service, req)
    node = next(
        n
        for n in cache.tree_core._node_arena.values()
        if n is not cache.tree_core.root_node
    )
    old = service.capabilities()
    lock = cache.lock(node.id)
    with pytest.raises(ControlError) as error:
        service.rebind_namespace("new")
    assert error.value.code == "busy" and service.capabilities() == old
    cache.unlock(lock)
    queued = request()
    cache.req_to_token_pool.alloc([queued])
    with pytest.raises(ControlError) as error:
        service.rebind_namespace("new")
    assert error.value.code == "busy" and service.capabilities() == old
    cache.req_to_token_pool.free(queued)
    with pytest.raises(ControlError) as error:
        service.rebind_namespace("ns")
    assert error.value.code == "conflict"


def test_streaming_session_private_kv_blocks_namespace_rotation_even_if_scheduler_idle(
    owner,
):
    cache, service = owner
    req = request(
        session=Session(capacity_of_str_len=0, session_id="s", streaming=True)
    )
    cache.req_to_token_pool.alloc([req])
    slots = cache.token_to_kv_pool_allocator.alloc(4)
    cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 4)), slots)
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 4
    cache.session.take(req)
    service.idle_check = lambda: True
    with pytest.raises(ControlError) as error:
        service.rebind_namespace("new")
    assert error.value.code == "busy" and "StreamingSession" in str(error.value)
    cache.session.release_session("s")


def test_flush_preserves_key_mapper_and_resource_approval(owner):
    cache, service = owner

    def budget():
        return 17

    service.control.executor.gpu_copy_capacity = budget
    encoded = service.cache_salt("tenant")
    cache.reset()
    service.reset_after_cache_flush()
    assert service.control.executor.gpu_copy_capacity is budget
    assert (
        service.control.executor._compatibility(target(service, "tenant")).cache_salt
        == encoded
    )
    assert service.capabilities()["cache_salt_namespace"] == "ns"


def test_encoding_is_unambiguous_and_matches_documented_wire_format():
    import hashlib

    body = ["sglang.kv.weight.v1", "权重", "租户"]
    expected = hashlib.sha256(
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    assert encode_cache_salt("权重", "租户") == expected
    assert encode_cache_salt("a", None) == encode_cache_salt("a", "")
    assert encode_cache_salt("a", "b:c") != encode_cache_salt("a:b", "c")
    assert len(encode_cache_salt("a" * 256, "b" * 256)) == 64
    with pytest.raises(ControlError):
        encode_cache_salt("ns", True)


def test_wrong_thread_namespace_change_cannot_rotate_records_before_rejection(owner):
    from concurrent.futures import ThreadPoolExecutor

    _, service = owner
    previous = service.capabilities()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(ControlError) as error:
            pool.submit(service.rebind_namespace, "other").result()
    assert error.value.code == "wrong_owner"
    assert service.capabilities() == previous
