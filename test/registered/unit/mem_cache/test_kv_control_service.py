# SPDX-License-Identifier: Apache-2.0
"""Serving control, request completion and static model namespaces."""

import unittest
from array import array
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import test_kv_control_cache as cache_fixtures

from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightVersionReqInput,
)
from sglang.srt.managers.kv_control.contracts import TokensTarget, resolve_prefix
from sglang.srt.managers.kv_control.key_namespace import encode_cache_salt
from sglang.srt.managers.kv_control.service import (
    KVControlService,
    ServiceConfig,
    runtime_config,
)
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
                resolve_prefix(
                    namespace_target(service, salt),
                    service.control.context("test"),
                    service.config.limits,
                )
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
                resolve_prefix(
                    namespace_target(service, req.cache_salt),
                    service.control.context("test"),
                    service.config.limits,
                )
            )
            assert not encoded_as_logical.items and encoded_as_logical.missing
            cache.sanity_check()


class TestKVControlWeights(cache_fixtures.KVCacheTestCase):
    def test_mutations_rejected_before_touching_runner_or_cache(self):
        from sglang.srt.managers.scheduler import Scheduler

        with service_owner(self, namespace_keys=True) as (_, service):
            scheduler = Mock(kv_control_service=service)
            worker = Mock()
            flush = Mock()
            with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
                manager = SchedulerWeightUpdaterManager(
                    tp_worker=worker,
                    draft_worker=None,
                    memory_saver_adapter=None,
                    flush_cache=flush,
                    is_fully_idle=lambda: True,
                    scheduler=scheduler,
                )
            for method in (
                manager.begin_weight_update,
                manager.end_weight_update,
                manager.update_weights_from_disk,
                manager.update_weights_from_tensor,
                manager.update_weights_from_distributed,
                manager.update_weights_from_ipc,
                manager.release_memory_occupation,
                manager.resume_memory_occupation,
            ):
                with self.subTest(method=method.__name__):
                    self.assertFalse(method(SimpleNamespace()).success)
            self.assertFalse(
                manager.check_weights(
                    CheckWeightsReqInput(action="reset_tensors")
                ).success
            )
            self.assertFalse(
                Scheduler.handle_update_weight_version(
                    scheduler, UpdateWeightVersionReqInput(new_version="v2")
                ).success
            )
            self.assertEqual(worker.mock_calls, [])
            scheduler.record_weight_version_change.assert_not_called()
            flush.assert_not_called()
            self.assertIsNone(manager._session)
            self.assertEqual(service.capabilities()["weight_updates"], "unsupported")

    def test_disk_update_uses_normal_runner_when_control_disabled(self):
        worker = Mock()
        runner = Mock()
        runner.weight_updater.update_weights_from_disk.return_value = (True, "ok")
        worker.weight_update_runners.return_value = [("target", runner)]
        scheduler = Mock(kv_control_service=None)
        flush = Mock(return_value=True)
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            manager = SchedulerWeightUpdaterManager(
                tp_worker=worker,
                draft_worker=None,
                memory_saver_adapter=None,
                flush_cache=flush,
                is_fully_idle=lambda: True,
                scheduler=scheduler,
            )
        result = manager.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path="local", weight_version="v2")
        )
        self.assertTrue(result.success)
        runner.weight_updater.update_weights_from_disk.assert_called_once_with(
            "local", None, recapture_cuda_graph=False
        )
        scheduler.record_weight_version_change.assert_called_once_with(new_version="v2")
        flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
