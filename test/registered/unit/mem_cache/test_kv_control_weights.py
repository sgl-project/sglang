# SPDX-License-Identifier: Apache-2.0
"""Real cache generations and CPU parameters across the existing weight APIs."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import pytest
import test_kv_control_namespaces as namespaces
import torch
from test_kv_control_namespaces import owner as owner

from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.kv_control.service import dispatch
from sglang.srt.managers.kv_control.weight_lifecycle import weight_namespace
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class Runner:
    def __init__(self, service):
        self.service = service
        self.initial_namespace = service.config.namespace
        self.weight = torch.tensor([1.0])
        self.weight_updater = self
        self.calls = []
        self.failure = None
        self.received = [("weight", torch.tensor([2.0]))]

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
        old = self.weight.clone()
        self.weight.copy_(value)
        if self.failure == "rollback":
            self.weight.copy_(old)
            return False, "injected rollback"
        if self.failure == "partial":
            return False, "injected partial update"
        if self.failure == "exception":
            raise RuntimeError("injected parameter exception")
        return True, "loaded"

    def update_weights_from_disk(self, path, load_format, *, recapture_cuda_graph):
        return self.load(torch.tensor([2.0]))

    def update_weights_from_tensor(self, *, named_tensors, load_format):
        return self.load(named_tensors[0][1])

    def update_weights_from_ipc(self, request):
        return self.load(torch.tensor([2.0]))

    def receive_weights_from_distributed(self, **kwargs):
        return self.received

    def load_weights_from_distributed(self, weights):
        return self.load(weights[0][1])

    def check_weights(self, *, action, **kwargs):
        if action == "reset_tensors":
            self.load(torch.tensor([3.0]))
        return None


@pytest.fixture
def system(owner):
    cache, service = owner
    service.weights.version = "v1"
    runner = Runner(service)
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
        yield cache, service, manager, runner, versions, flushes
    # Unknown model failures remain fenced in production. This controlled CPU
    # model fixture restores its checkpoint and certifies quiescence for cleanup.
    if service.weights.active:
        runner.weight.fill_(1)
        service.weights.request_finish(False)
        service.weights.mark_quiescent()
        service.weights.finish(False, synchronized=True)


def request(**changes):
    fields = dict(
        weight_version="v2",
        flush_cache=True,
        torch_empty_cache=False,
        model_path="local",
        load_format=None,
        recapture_cuda_graph=False,
        selector="target",
        serialized_named_tensors=[("weight", torch.tensor([2.0]))],
        names=["weight"],
        dtypes=["float32"],
        shapes=[[1]],
        group_name="test",
    )
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "method", ["update_weights_from_disk", "update_weights_from_ipc"]
)
@pytest.mark.parametrize("flush_cache", [True, False])
def test_standalone_mutates_parameters_after_isolation_and_never_reuses_old_keys(
    system, method, flush_cache
):
    cache, service, manager, runner, versions, flushes = system
    req = namespaces.request("tenant")
    namespaces.complete(cache, service, req)
    old_context = service.control.context("router")
    old_namespace = service.config.namespace
    old_nodes = {n.id for n in cache.tree_core._node_arena.values()}
    output = getattr(manager, method)(request(flush_cache=flush_cache))
    assert output.success, output
    assert runner.weight.item() == 2.0 and versions == ["v2"]
    assert service.config.namespace == weight_namespace(
        service.base_namespace, "version", "v2"
    )
    assert service.config.namespace != old_namespace and not service.weights.active
    assert service.weights.version == "v2"
    target = namespaces.target(service, "tenant")
    snapshot = service.control.executor.residency(service.control.index.resolve(target))
    assert not snapshot.items and snapshot.missing
    assert bool(flushes) == flush_cache
    if not flush_cache:
        assert old_nodes.issubset(cache.tree_core._node_arena)
    with pytest.raises(ValueError):
        service.control.registry.validate_context(old_context)


@pytest.mark.parametrize(
    "method", ["update_weights_from_tensor", "update_weights_from_distributed"]
)
def test_multipart_holds_fence_and_defers_flush_until_end(system, method):
    cache, service, manager, runner, versions, flushes = system
    assert manager.begin_weight_update(
        BeginWeightUpdateReqInput(selector="target")
    ).success
    assert service.weights.active
    assert getattr(manager, method)(request()).success
    assert runner.weight.item() == 2 and not flushes
    assert service.weights.active and service.capabilities()["updating_weights"]
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


def test_incoming_generation_is_aborted_without_constructing_or_forwarding_during_update(
    system,
):
    _, service, manager, _, _, _ = system
    assert manager.begin_weight_update(
        BeginWeightUpdateReqInput(selector="target")
    ).success
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_control_service = service
    sent = []
    scheduler.ipc_channels = SimpleNamespace(
        send_to_tokenizer=SimpleNamespace(
            send_output=lambda output, req: sent.append((output, req))
        )
    )
    scheduler._handle_generate_request = Mock(
        side_effect=AssertionError("must not construct or forward")
    )
    recv = TokenizedGenerateReqInput(
        rid="new",
        input_text=None,
        input_ids=namespaces.array("q", [1]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )
    scheduler.handle_generate_request(recv)
    assert len(sent) == 1 and sent[0][0].finished_reason["status_code"] == 503
    scheduler._handle_generate_request.assert_not_called()
    assert manager.end_weight_update(EndWeightUpdateReqInput()).success


def test_active_request_or_same_version_refuses_before_mutating_namespace_or_parameters(
    system,
):
    cache, service, manager, runner, _, _ = system
    original = service.capabilities()
    row = namespaces.request()
    cache.req_to_token_pool.alloc([row])
    output = manager.update_weights_from_disk(request())
    assert (
        not output.success and not runner.calls and service.capabilities() == original
    )
    cache.req_to_token_pool.free(row)
    output = manager.update_weights_from_disk(request(weight_version="v1"))
    assert (
        not output.success and not runner.calls and service.capabilities() == original
    )


@pytest.mark.parametrize("failure", ["rollback", "partial"])
def test_failed_parameter_result_closes_into_fresh_isolated_namespace(system, failure):
    _, service, manager, runner, _, _ = system
    old = service.config.namespace
    runner.failure = failure
    output = manager.update_weights_from_disk(request(flush_cache=False))
    assert not output.success and output.message == "injected " + (
        "rollback" if failure == "rollback" else "partial update"
    )
    assert service.config.namespace != old and not service.weights.active
    assert service.weights.version is None
    assert service.config.namespace != weight_namespace(
        service.base_namespace, "version", "v2"
    )
    assert runner.weight.item() == (1 if failure == "rollback" else 2)


def test_failed_multipart_result_is_not_converted_to_success_by_end(system):
    _, service, manager, runner, _, _ = system
    assert manager.begin_weight_update(
        BeginWeightUpdateReqInput(selector="target")
    ).success
    runner.failure = "rollback"
    assert not manager.update_weights_from_tensor(request()).success
    output = manager.end_weight_update(EndWeightUpdateReqInput())
    assert not output.success and not service.weights.active
    assert runner.weight.item() == 1 and service.weights.version is None


def test_unexpected_parameter_exception_keeps_inference_fenced(system):
    _, service, manager, runner, _, _ = system
    runner.failure = "exception"
    output = manager.update_weights_from_disk(request())
    assert not output.success and "remains fenced" in output.message
    assert service.weights.active and service.weights.fence.failed


def test_unknown_versions_always_get_fresh_lookup_generations(system):
    _, service, manager, _, _, _ = system
    assert manager.update_weights_from_disk(
        request(weight_version=None, flush_cache=False)
    ).success
    first = service.config.namespace
    assert manager.update_weights_from_disk(
        request(weight_version=None, flush_cache=False)
    ).success
    assert service.config.namespace != first and service.weights.version is None


def test_memory_release_and_resume_cannot_race_parameter_update(system):
    from sglang.srt.managers.io_struct import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
    )

    _, service, manager, _, _, _ = system
    assert manager.begin_weight_update(BeginWeightUpdateReqInput()).success
    released = manager.release_memory_occupation(
        ReleaseMemoryOccupationReqInput(tags=["weights"])
    )
    resumed = manager.resume_memory_occupation(
        ResumeMemoryOccupationReqInput(tags=["weights"])
    )
    assert not released.success and not resumed.success
    assert service.weights.active
    assert manager.end_weight_update(EndWeightUpdateReqInput()).success


def test_offloaded_resources_refuse_parameter_mutation_before_namespace_change(system):
    _, service, manager, runner, _, _ = system
    previous = service.capabilities()
    manager.offload_tags.add("weights")
    result = manager.update_weights_from_disk(request())
    assert not result.success and not runner.calls
    assert service.capabilities() == previous
    manager.offload_tags.clear()


def test_synchronization_failure_retains_fence_until_explicit_finalization(system):
    _, service, manager, _, _, _ = system
    with patch("torch.cpu.synchronize", side_effect=RuntimeError("worker not retired")):
        result = manager.update_weights_from_disk(request())
    assert not result.success and service.weights.active
    assert not service.weights.fence.quiescent
    assert service.weights.fence.ready_to_finish
    finalized = manager.end_weight_update(EndWeightUpdateReqInput())
    assert not finalized.success  # Preserve the failed synchronization result.
    assert not service.weights.active and service.weights.version is None


def test_cache_flush_failure_can_finalize_without_repeating_parameter_mutation(system):
    _, service, manager, runner, _, _ = system
    original = manager.flush_cache
    manager.flush_cache = lambda **kwargs: False
    result = manager.update_weights_from_disk(request())
    assert not result.success and service.weights.active
    assert service.weights.fence.quiescent
    assert runner.calls == ["load"]
    manager.flush_cache = original
    finalized = manager.end_weight_update(EndWeightUpdateReqInput())
    assert finalized.success, finalized
    assert not service.weights.active and runner.calls == ["load"]
    assert service.weights.version == "v2"


def test_debug_parameter_reset_uses_same_namespace_boundary(system):
    from sglang.srt.managers.io_struct import CheckWeightsReqInput

    _, service, manager, runner, _, _ = system
    old = service.config.namespace
    with patch("torch.distributed.get_world_size", return_value=1):
        result = manager.check_weights(
            CheckWeightsReqInput(action="reset_tensors", selector="target")
        )
    assert result.success, result
    assert runner.weight.item() == 3
    assert service.config.namespace != old and service.weights.version is None
    assert not service.weights.active


def test_wrong_thread_end_cannot_touch_runner_before_rejection(system):
    from concurrent.futures import ThreadPoolExecutor

    _, service, manager, runner, _, _ = system
    assert manager.begin_weight_update(BeginWeightUpdateReqInput()).success
    before = list(runner.calls)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            manager.end_weight_update, EndWeightUpdateReqInput()
        ).result()
    assert not result.success and "owner" in result.message
    assert runner.calls == before and service.weights.active
    assert manager.end_weight_update(EndWeightUpdateReqInput()).success


def test_python_engine_version_reaches_owner_through_real_serialized_tensor_request(
    system,
):
    import asyncio
    from types import MethodType

    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import msgpack_decode, msgpack_encode
    from sglang.srt.utils.common import MultiprocessingSerializer

    _, service, manager, runner, _, _ = system
    manager.tp_worker.deserialize_own_rank = lambda payloads: (
        MultiprocessingSerializer.deserialize(payloads[0])
    )

    class Loop:
        def run_until_complete(self, coro):
            return asyncio.run(coro)

    class Bridge:
        async def begin_weight_update(self, obj, _):
            result = manager.begin_weight_update(msgpack_decode(msgpack_encode(obj)))
            return result.success, result.message

        async def update_weights_from_tensor(self, obj, _):
            result = manager.update_weights_from_tensor(
                msgpack_decode(msgpack_encode(obj))
            )
            return result.success, result.message

        async def end_weight_update(self, obj, _):
            result = manager.end_weight_update(msgpack_decode(msgpack_encode(obj)))
            return result.success, result.message

    engine = SimpleNamespace(loop=Loop(), tokenizer_manager=Bridge())
    engine._serialize_tensors_per_rank = MethodType(
        Engine._serialize_tensors_per_rank, engine
    )
    assert Engine.begin_weight_update(engine, selector="target")[0]
    loaded = Engine.update_weights_from_tensor(
        engine,
        [("weight", torch.tensor([4.0]))],
        flush_cache=False,
        weight_version="v2",
        selector="target",
    )
    assert loaded[0], loaded
    assert runner.weight.item() == 4.0 and service.weights.active
    assert Engine.end_weight_update(engine)[0]
    assert service.weights.version == "v2" and not service.weights.active


def test_explicit_version_assertion_is_a_guarded_namespace_boundary(system):
    from sglang.srt.managers.io_struct import UpdateWeightVersionReqInput

    _, service, manager, runner, versions, _ = system
    old = service.config.namespace
    result = manager.update_weight_version(
        UpdateWeightVersionReqInput(new_version="v2")
    )
    assert result.success, result
    assert service.config.namespace != old and service.weights.version == "v2"
    assert not runner.calls and versions == ["v2"]
    stable = service.capabilities()
    assert manager.update_weight_version(
        UpdateWeightVersionReqInput(new_version="v2")
    ).success
    assert service.capabilities() == stable


def test_external_transfer_version_is_not_published_until_end(system):
    from sglang.srt.managers.io_struct import UpdateWeightVersionReqInput

    _, service, manager, runner, _, _ = system
    assert manager.begin_weight_update(BeginWeightUpdateReqInput()).success
    provisional = service.config.namespace
    runner.weight.fill_(2)  # The existing P2P/RDMA begin/end protocol.
    assert manager.update_weight_version(
        UpdateWeightVersionReqInput(new_version="v2")
    ).success
    assert service.weights.active and service.config.namespace == provisional
    assert service.weights.version == "v1"
    assert manager.end_weight_update(EndWeightUpdateReqInput()).success
    assert service.weights.version == "v2" and service.config.namespace != provisional


def test_selector_refusal_precedes_namespace_change(system):
    _, service, manager, runner, _, _ = system
    previous = service.capabilities()
    result = manager.begin_weight_update(BeginWeightUpdateReqInput(selector="draft"))
    assert (
        not result.success and not runner.calls and service.capabilities() == previous
    )


def test_offloaded_cache_refuses_native_execution_without_touching_copy(system):
    from sglang.srt.managers.kv_control.contracts import Status

    _, service, _, _, _, _ = system
    service.cache_available = lambda: False
    ctx = msgspec.to_builtins(service.control.context("router"))
    result = dispatch(
        service,
        {
            "kind": "submit",
            "context": ctx,
            "envelope": {
                "protocol_version": "0.1",
                "message_id": "sleeping",
                "actions": [
                    {
                        "action_id": "a",
                        "action_type": "sglang.set_priority",
                        "action_version": "1.0",
                        "payload": {
                            "target": msgspec.json.decode(
                                msgspec.json.encode(namespaces.target(service))
                            ),
                            "location": {"tier": "gpu"},
                            "claim_id": "c",
                            "revision": 1,
                            "value": 90,
                            "ttl_seconds": 10,
                        },
                    }
                ],
            },
        },
    )
    assert result["success"]
    assert result["data"]["results"][0]["update"]["status"] == Status.DEFERRED.value
    assert result["data"]["results"][0]["update"]["code"] == "unavailable"
    assert not service.has_pending
    assert not service.control.executor.capabilities()["available"]
    service.cache_available = lambda: True
