# SPDX-License-Identifier: Apache-2.0
"""Serving control, request completion and weight namespace isolation."""

import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import test_kv_control_cache as cache_fixtures
import torch

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
)
from sglang.srt.managers.kv_control.contracts import TokensTarget
from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt
from sglang.srt.managers.kv_control.service import (
    KVControlService,
    ServiceConfig,
    dispatch,
    runtime_config,
)
from sglang.srt.managers.kv_control.weight_lifecycle import weight_namespace
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


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
            cache_fixtures.insert(cache, [1, 2, 3, 4, 5], salt=salt)
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
            snapshot = service.control.executor.residency(
                service.control.index.resolve(namespace_target(service, salt))
            )
            assert (
                sum(
                    item.descriptor.end_token - item.descriptor.start_token
                    for item in snapshot.items
                )
                == 5
            )
            assert not snapshot.missing
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
