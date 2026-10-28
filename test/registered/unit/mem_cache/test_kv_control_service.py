# SPDX-License-Identifier: Apache-2.0
"""Serving control, request completion and weight namespace isolation."""

import asyncio
import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import test_kv_control_cache as cache_fixtures
import torch
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    KVControlReqInput,
    KVControlReqOutput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.kv_control.contracts import Status, TokensTarget
from sglang.srt.managers.kv_control.http import install_routes
from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt
from sglang.srt.managers.kv_control.service import (
    KVControlService,
    ServiceConfig,
    dispatch,
    runtime_config,
)
from sglang.srt.managers.kv_control.weight_lifecycle import weight_namespace
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def envelope(*, attached=False, message="message", opaque=False, invalid=False):
    selection = "request_cached" if attached else msgspec.to_builtins(target())
    actions = [
        KvHintAction(
            action_id="keep",
            action_type="sglang.set_priority",
            action_version="1.0",
            payload={
                "target": selection,
                "location": {"tier": "gpu"},
                "claim_id": "keep",
                "revision": 1,
                "value": 90,
                "ttl_seconds": 10,
            },
        )
    ]
    if invalid:
        actions.append(
            KvHintAction(
                action_id="bad",
                action_type="sglang.set_priority",
                action_version="1.0",
                payload={"bad": "value"},
            )
        )
    if opaque:
        actions.append(
            KvHintAction(
                action_id="other",
                action_type="backend.opaque",
                action_version="8",
                payload={"private": [1, 2]},
            )
        )
    return KvHintsEnvelope(protocol_version="0.1", message_id=message, actions=actions)


def serving_request(rid="same", *, salt=None):
    return Req(
        rid=rid,
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2, 3, 4]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
        cache_salt=salt,
    )


@contextmanager
def service_owner(case, *, namespace_keys=False):
    cache, _ = case.make(session=False)
    cache.reset()
    service = KVControlService(
        cache,
        ServiceConfig(namespace="ns", engine_id="engine"),
        clock=case.clock,
        namespace_keys=namespace_keys,
        idle_check=lambda: (
            len(cache.req_to_token_pool.free_slots) == cache.req_to_token_pool.size
        ),
    )
    yield (cache, service)
    assert not service.has_pending


insert, match_len, target = (
    cache_fixtures.insert,
    cache_fixtures.match_len,
    cache_fixtures.target,
)


class TestKVControlService(cache_fixtures.KVCacheTestCase):
    def test_http_executes_real_priority_and_replay_and_epoch_checks(self):
        with service_owner(self) as owner:
            cache, service = owner
            insert(cache, [1, 2, 3, 4])
            insert(cache, [5, 6, 7, 8])

            class Manager:
                async def kv_control(self, command):
                    wire = msgpack_decode(
                        msgpack_encode(
                            KVControlReqInput(request_id="rpc", command=command)
                        )
                    )
                    reply = KVControlReqOutput(
                        request_id=wire.request_id,
                        result=dispatch(service, wire.command),
                    )
                    return msgpack_decode(msgpack_encode(reply)).result

            app = FastAPI()
            install_routes(app, lambda: Manager())

            async def call():
                async with AsyncClient(
                    transport=ASGITransport(app), base_url="http://owner"
                ) as client:
                    cap = (await client.get("/v1/kv/capabilities")).json()["data"]
                    assert (
                        cap["engine_id"] == "engine"
                        and cap["topology"] == "single_owner"
                    )
                    ctx = msgspec.to_builtins(service.control.context("router"))
                    body = {
                        "kind": "submit",
                        "context": ctx,
                        "envelope": msgspec.to_builtins(envelope()),
                    }
                    result = await client.post("/v1/kv/control", json=body)
                    assert result.status_code == 200
                    assert (
                        result.json()["data"]["results"][0]["update"]["status"]
                        == Status.COMPLETED.value
                    )
                    replayed = await client.post("/v1/kv/control", json=body)
                    assert replayed.json()["data"]["replayed"]
                    stale = await client.post(
                        "/v1/kv/control",
                        json={
                            "kind": "capacity",
                            "context": dict(ctx, engine_epoch="stale"),
                        },
                    )
                    assert stale.status_code == 409

            asyncio.run(call())
            cache.evict(EvictParams(num_tokens=4))
            assert match_len(cache, [1, 2, 3, 4]) == 4
            assert match_len(cache, [5, 6, 7, 8]) == 0

    def test_attached_fanout_lookup_does_not_mutate_shared_envelope(self):
        with service_owner(self) as owner:
            cache, service = owner
            insert(cache, [1, 2, 3, 4])
            hints = envelope(attached=True, opaque=True, invalid=True)
            original = msgspec.json.encode(hints)
            reqs = [serving_request(), serving_request()]
            for req in reqs:
                service.attach(req, hints)
                assert len(req.kv_hints.actions) == 1
                assert req.kv_hints.actions[0] is hints.actions[-1]
            assert msgspec.json.encode(hints) == original
            lookup = dispatch(
                service,
                {
                    "kind": "lookup",
                    "context": msgspec.to_builtins(service.control.context("request")),
                    "message_id": "message",
                },
            )
            refs = [message["results"][0]["ref"] for message in lookup["data"]]
            assert (
                len(refs) == 2
                and refs[0]["context"]["binding_id"] != refs[1]["context"]["binding_id"]
            )
            assert service.has_pending
            service.finish_request(reqs[0], aborted=True)
            reqs[1].finished_reason = FINISH_LENGTH(1)
            reqs[1].refresh_fill_ids()
            service.finish_request(reqs[1], checkpointed=True, committed_len=4)
            results = [
                service.control.registry.query(
                    msgspec.convert(
                        r["context"], type=type(service.control.context("request"))
                    ),
                    "message",
                ).results
                for r in refs
            ]
            assert results[0][0].status == Status.CANCELLED
            assert results[1][0].status == Status.COMPLETED
            assert all((batch[1].status == Status.REJECTED for batch in results))
            assert not service.has_pending

    def test_real_release_binds_after_checkpoint_before_unlock_and_keeps_salt(self):
        with service_owner(self) as owner:
            cache, service = owner
            req = serving_request(salt="tenant")
            req.output_ids = array("q", [5])
            req.finished_reason = FINISH_LENGTH(1)
            cache.req_to_token_pool.alloc([req])
            slots = cache.token_to_kv_pool_allocator.alloc(5)
            cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 5)), slots)
            req.kv.kv_committed_len = req.kv.kv_allocated_len = 5
            service.attach(req, envelope(attached=True))
            context = req.kv_control_context
            observed = []
            start = service.control.executor.start

            def inspect(work):
                observed.append(
                    (
                        req.kv.holds_kv,
                        req.lock is not None,
                        tuple(work.target.branches[0].content.token_ids),
                    )
                )
                return start(work)

            with patch.object(service.control.executor, "start", side_effect=inspect):
                release_kv_cache(req, cache, checkpoint=True)
            assert observed == [(True, True, (1, 2, 3, 4, 5))]
            result = service.control.registry.query(context, "message").results[0]
            assert result.status == Status.COMPLETED
            assert not req.kv.holds_kv and req.lock is None
            assert match_len(cache, [1, 2, 3, 4, 5], salt="tenant") == 5
            assert match_len(cache, [1, 2, 3, 4, 5]) == 0
            assert cache.token_to_kv_pool_allocator.available_size() == 59
            cache.sanity_check()

    def test_runtime_profile_rejects_unsupported_config(self):
        for fields in [{"tp_size": 2}, {"disable_radix_cache": True}]:
            with self.subTest(fields=fields):
                with get_context().override_server_args(
                    kv_control_config={"namespace": "ns"}, **fields
                ):
                    with pytest.raises(ValueError):
                        runtime_config()


def namespace_request(salt=None, session=None):
    return Req(
        rid="r",
        origin_input_text=None,
        origin_input_ids=array("q", [1, 2, 3, 4]),
        sampling_params=SamplingParams(max_new_tokens=1),
        vocab_size=128,
        cache_salt=salt,
        session=session,
    )


def complete_request(cache, service, req):
    service.prepare_request(req)
    req.output_ids = array("q", [5])
    req.finished_reason = FINISH_LENGTH(1)
    cache.req_to_token_pool.alloc([req])
    slots = cache.token_to_kv_pool_allocator.alloc(5)
    cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 5)), slots)
    req.kv.kv_allocated_len = req.kv.kv_committed_len = 5
    release_kv_cache(req, cache, checkpoint=True)


def namespace_target(service, salt=None):
    return TokensTarget(
        namespace=service.config.namespace, token_ids=(1, 2, 3, 4, 5), cache_salt=salt
    )


def namespace_priority(
    service, selection="request_cached", context=None, message="keep"
):
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


class TestKVControlNamespaces(cache_fixtures.KVCacheTestCase):
    def test_ordinary_unhinted_request_and_independent_lookup_use_same_real_key(self):
        with service_owner(self, namespace_keys=True) as owner:
            salt = "tenant-a"
            from sglang.srt.managers.scheduler import Scheduler

            class TaggedReq(Req):
                pass

            class TaggedScheduler(Scheduler):
                def _create_generation_req(self, recv_req, *args, **kwargs):
                    return TaggedReq(*args, **kwargs)

            cache, service = owner
            raw_node = cache_fixtures.insert(cache, [1, 2, 3, 4, 5], salt=salt)
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
            complete_request(cache, service, req)
            assert not hasattr(req, "kv_control_context")
            assert req.cache_salt == encode_cache_salt("ns", salt)
            assert req.kv_control_logical_cache_salt == salt
            result = namespace_priority(service, namespace_target(service, salt))
            assert result.status == Status.COMPLETED, result
            snapshot = service.control.executor.residency(
                service.control.index.resolve(namespace_target(service, salt))
            )
            assert snapshot.items and (not snapshot.missing)
            assert all((row.effective_priority == 90 for row in snapshot.items))
            assert (
                cache.tree_core.priority_claims.effective(
                    raw_node, cache_fixtures.ComponentType.FULL
                )
                is None
            )
            encoded_as_logical = service.control.executor.residency(
                service.control.index.resolve(namespace_target(service, req.cache_salt))
            )
            assert not encoded_as_logical.items and encoded_as_logical.missing
            cache.sanity_check()


class WeightRunner:
    def __init__(self, service):
        self.service = service
        self.initial_namespace = service.config.namespace
        self.weight = torch.tensor([1.0])
        self.weight_updater = self
        self.calls = []
        self.failure = None

    def begin_weight_update(self):
        assert self.service.weights.active
        assert self.service.config.namespace != self.initial_namespace
        self.calls.append("begin")

    def end_weight_update(self, *, run_post_load):
        assert self.service.weights.active
        self.calls.append(("end", run_post_load))

    def load(self, value):
        assert self.service.weights.active
        assert self.service.config.namespace != self.initial_namespace
        self.calls.append("load")
        self.weight.copy_(value)
        if self.failure == "partial":
            return (False, "injected partial update")
        if self.failure == "exception":
            raise RuntimeError("injected parameter exception")
        return (True, "loaded")

    def update_weights_from_disk(self, path, load_format, *, recapture_cuda_graph):
        return self.load(torch.tensor([2.0]))

    def update_weights_from_tensor(self, *, named_tensors, load_format):
        return self.load(named_tensors[0][1])


@contextmanager
def weight_system(owner):
    cache, service = owner
    service.weights.version = "v1"
    runner = WeightRunner(service)
    versions = []
    flushes = []

    def flush(**kwargs):
        assert service.weights.fence is None or service.weights.fence.quiescent
        flushes.append(kwargs)
        cache.reset()
        service.reset_after_cache_flush()
        return True

    scheduler = SimpleNamespace(
        kv_control_service=service,
        record_weight_version_change=lambda new_version: versions.append(new_version),
    )
    with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
        manager = SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=runner,
                weight_update_runners=lambda: [("target", runner)],
                deserialize_own_rank=lambda payload: payload,
            ),
            draft_worker=None,
            memory_saver_adapter=None,
            flush_cache=flush,
            is_fully_idle=service.idle_check,
            scheduler=scheduler,
        )
    with patch("torch.distributed.barrier"):
        yield (cache, service, manager, runner, versions, flushes)
    if service.weights.active:
        runner.weight.fill_(1)
        service.weights.request_finish(False)
        service.weights.mark_quiescent()
        service.weights.finish(False, synchronized=True)


def weight_request(**changes):
    fields = dict(
        weight_version="v2",
        flush_cache=True,
        torch_empty_cache=False,
        model_path="local",
        load_format=None,
        recapture_cuda_graph=False,
        selector="target",
        serialized_named_tensors=[("weight", torch.tensor([2.0]))],
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


class TestKVControlWeights(cache_fixtures.KVCacheTestCase):
    def test_standalone_mutates_parameters_after_isolation_and_never_reuses_old_keys(
        self,
    ):
        for flush_cache in (False, True):
            with self.subTest(flush_cache=flush_cache):
                with service_owner(self, namespace_keys=True) as owner:
                    with weight_system(owner) as system:
                        cache, service, manager, runner, versions, flushes = system
                        req = namespace_request("tenant")
                        complete_request(cache, service, req)
                        old_context = service.control.context("router")
                        old_namespace = service.config.namespace
                        old_nodes = {n.id for n in cache.tree_core._node_arena.values()}
                        output = manager.update_weights_from_disk(
                            weight_request(flush_cache=flush_cache)
                        )
                        assert output.success, output
                        assert runner.weight.item() == 2.0 and versions == ["v2"]
                        assert service.config.namespace == weight_namespace(
                            service.base_namespace, "version", "v2"
                        )
                        assert service.config.namespace != old_namespace and (
                            not service.weights.active
                        )
                        assert service.weights.version == "v2"
                        target = namespace_target(service, "tenant")
                        snapshot = service.control.executor.residency(
                            service.control.index.resolve(target)
                        )
                        assert not snapshot.items and snapshot.missing
                        assert bool(flushes) == flush_cache
                        if not flush_cache:
                            assert old_nodes.issubset(cache.tree_core._node_arena)
                        with pytest.raises(ValueError):
                            service.control.registry.validate_context(old_context)

    def test_multipart_holds_fence_and_defers_flush_until_end(self):
        with service_owner(self, namespace_keys=True) as owner:
            with weight_system(owner) as system:
                cache, service, manager, runner, versions, flushes = system
                assert manager.begin_weight_update(
                    BeginWeightUpdateReqInput(selector="target")
                ).success
                assert service.weights.active
                assert manager.update_weights_from_tensor(weight_request()).success
                assert runner.weight.item() == 2 and (not flushes)
                assert (
                    service.weights.active
                    and service.capabilities()["updating_weights"]
                )
                context = msgspec.to_builtins(service.control.context("router"))
                blocked = dispatch(
                    service,
                    {
                        "kind": "submit",
                        "context": context,
                        "envelope": {
                            "protocol_version": "0.1",
                            "message_id": "busy",
                            "actions": [],
                        },
                    },
                )
                assert blocked["error"]["code"] == "busy"
                output = manager.end_weight_update(EndWeightUpdateReqInput())
                assert output.success, output
                assert not service.weights.active and service.weights.version == "v2"
                assert len(flushes) == 1 and runner.calls[-1] == ("end", False)
                assert versions == ["v2"]

    def test_failed_parameter_result_closes_into_fresh_isolated_namespace(self):
        with service_owner(self, namespace_keys=True) as owner:
            with weight_system(owner) as system:
                failure = "partial"
                _, service, manager, runner, _, _ = system
                old = service.config.namespace
                runner.failure = failure
                output = manager.update_weights_from_disk(
                    weight_request(flush_cache=False)
                )
                assert not output.success and output.message == "injected " + (
                    "rollback" if failure == "rollback" else "partial update"
                )
                assert service.config.namespace != old and (not service.weights.active)
                assert service.weights.version is None
                assert service.config.namespace != weight_namespace(
                    service.base_namespace, "version", "v2"
                )
                assert runner.weight.item() == (1 if failure == "rollback" else 2)

    def test_unexpected_parameter_exception_keeps_inference_fenced(self):
        with service_owner(self, namespace_keys=True) as owner:
            with weight_system(owner) as system:
                _, service, manager, runner, _, _ = system
                runner.failure = "exception"
                output = manager.update_weights_from_disk(weight_request())
                assert not output.success and "remains fenced" in output.message
                assert service.weights.active and service.weights.fence.failed


if __name__ == "__main__":
    unittest.main()
