# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from types import MethodType, ModuleType, SimpleNamespace
from unittest.mock import ANY, AsyncMock, MagicMock, call

import pytest
import torch

from sglang.srt.arg_groups.overrides import (
    modelexpress_transport_of,
    modelexpress_url_of,
)
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    ModelExpressWeightUpdateReqOutput,
    UpdateWeightsFromModelExpressReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.model_executor.model_runner_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def modelexpress_modules(monkeypatch):
    class WeightVersionRef:
        def __init__(self, version_id):
            self.version_id = version_id

    modelexpress_rl = ModuleType("modelexpress_rl")
    modelexpress_rl.__path__ = []
    modelexpress_rl.WeightVersionRef = WeightVersionRef
    monkeypatch.setitem(sys.modules, "modelexpress_rl", modelexpress_rl)
    integration = ModuleType("modelexpress_rl.inference.engines.sglang")
    integration.get_modelexpress_generator = MagicMock(
        side_effect=lambda runner: runner.modelexpress_generator
    )
    monkeypatch.setitem(sys.modules, integration.__name__, integration)
    monkeypatch.setattr(
        weight_updater_module,
        "get_model",
        lambda: SimpleNamespace(weight_cache_mode="off"),
    )
    monkeypatch.setattr(
        weight_updater_module,
        "_unsupported_derived_weight_cache_error",
        lambda model: None,
    )
    monkeypatch.setattr(
        weight_updater_module.DefaultModelLoader,
        "restore_weights_before_loading",
        MagicMock(),
    )
    parallel = SimpleNamespace(tp_group=SimpleNamespace(cpu_group=object()))
    monkeypatch.setattr(weight_updater_module, "get_parallel", lambda: parallel)
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.weight_updater.get_parallel",
        lambda: SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None), pp_size=1),
    )
    return integration


class Staged:
    def __init__(self, version_id):
        self.version_id = version_id
        self.applied = False
        self.released = 0
        self.release_error = None
        self.metrics = {}

    def release(self):
        self.released += 1
        if self.release_error is not None:
            raise self.release_error


class Generator:
    def __init__(self):
        self.worker_id = "generator-a"
        self.staged = []
        self.applied = []
        self.closed = 0
        self.stage_error = None
        self.apply_error = None
        self.release_error = None
        self.serving_version = None

    def stage_weight(self, *, version):
        if self.stage_error is not None:
            raise self.stage_error
        if self.staged and not self.staged[-1].released:
            staged = self.staged[-1]
            if staged.version_id != version.version_id:
                raise RuntimeError("another generator update is still active")
            return staged
        staged = Staged(version.version_id)
        # MX returns an already-applied handle for the version it serves.
        staged.applied = version.version_id == self.serving_version
        staged.release_error = self.release_error
        self.staged.append(staged)
        return staged

    def apply_weight(self, staged):
        if self.apply_error is not None:
            raise self.apply_error
        self.applied.append(staged)

    def close(self):
        self.closed += 1


def model_updater(generator):
    runner = SimpleNamespace(
        model=object(),
        loader=SimpleNamespace(
            _prepare_weights=lambda *_args: ("/models/launch", None, None)
        ),
        model_config=SimpleNamespace(model_path="model", revision=None),
        server_args=SimpleNamespace(
            modelexpress_config={
                "model_name": "model",
                "server_url": "mx:8001",
                "initial_base_version_id": "base-a",
                "refit_checkpoint_dir": "/tmp/mx-cache",
                "object_storage_endpoint_url": "http://minio:9000",
            },
        ),
    )
    if generator is not None:
        runner.modelexpress_generator = generator
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=runner.model_config,
        custom_weight_loaders={},
        get_model=lambda: runner.model,
        update_model_fields=MagicMock(),
        recapture_cuda_graph=MagicMock(),
        get_model_runner=lambda: runner,
    )


def test_update_stages_applies_and_releases_without_a_prepare_request(
    modelexpress_modules,
):
    generator = Generator()
    updater = model_updater(generator)

    installed, _ = updater.update_weights_from_modelexpress("2")

    modelexpress_modules.get_modelexpress_generator.assert_called_once_with(
        updater.get_model_runner()
    )
    assert [item.version_id for item in generator.staged] == ["2"]
    assert generator.applied == generator.staged
    assert generator.staged[0].released == 1
    assert installed is True


@pytest.mark.parametrize("serving", [False, True])
def test_packed_weights_are_restored_only_right_before_a_reload(serving):
    generator = Generator()
    generator.serving_version = "2" if serving else None
    updater = model_updater(generator)
    restore = weight_updater_module.DefaultModelLoader.restore_weights_before_loading
    apply_weight = generator.apply_weight

    def apply_after_restore(staged):
        assert restore.call_count == int(not serving)
        apply_weight(staged)

    generator.apply_weight = apply_after_restore

    assert updater.update_weights_from_modelexpress("2") == (True, "")
    assert generator.applied == generator.staged
    if not serving:
        restore.assert_called_once_with(updater.get_model(), torch.device("cpu"))


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize(
    "flush_options", [{}, {"flush_cache": False}, {"torch_empty_cache": True}]
)
def test_scheduler_records_version_before_flushing(success, flush_options):
    events = MagicMock()
    events.update.return_value = (success, "updated" if success else "stage failed")
    events.flush.return_value = True
    scheduler = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(update_weights_from_modelexpress=events.update),
        draft_worker=None,
        memory_saver_adapter=None,
        flush_cache=events.flush,
        is_fully_idle=MagicMock(),
        scheduler=SimpleNamespace(record_weight_version_change=events.record),
        metrics_collector=events.metrics,
    )
    request = UpdateWeightsFromModelExpressReqInput(weight_version="2", **flush_options)

    result = scheduler.update_weights_from_modelexpress(request)

    assert result.success is success
    assert result.message == ("updated" if success else "stage failed")
    expected = [call.update(request)]
    if success:
        expected.append(call.record(new_version="2"))
        if request.flush_cache:
            expected.append(call.flush(empty_cache=request.torch_empty_cache))
    expected.append(call.metrics.observe_weight_load(ANY, "modelexpress"))
    assert events.mock_calls == expected


def test_failed_post_update_flush_still_records_version():
    record = MagicMock()
    scheduler = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(
            update_weights_from_modelexpress=lambda request: (True, "")
        ),
        draft_worker=None,
        memory_saver_adapter=None,
        flush_cache=lambda **kwargs: False,
        is_fully_idle=MagicMock(),
        scheduler=SimpleNamespace(record_weight_version_change=record),
    )

    result = scheduler.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(weight_version="2")
    )

    assert result.success is False
    assert "cache was not flushed" in result.message
    record.assert_called_once_with(new_version="2")


def test_scheduler_rejects_draft_models_before_updating():
    update = MagicMock()
    scheduler = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(update_weights_from_modelexpress=update),
        draft_worker=object(),
        memory_saver_adapter=None,
        flush_cache=MagicMock(),
        is_fully_idle=MagicMock(),
    )

    result = scheduler.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(weight_version="2")
    )

    assert result.success is False
    assert "draft models" in result.message
    update.assert_not_called()


def test_scheduler_rejects_pipeline_parallelism_before_updating(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.managers.scheduler_components.weight_updater.get_parallel",
        lambda: SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None), pp_size=2),
    )
    update = MagicMock()
    scheduler = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(update_weights_from_modelexpress=update),
        draft_worker=None,
        memory_saver_adapter=None,
        flush_cache=MagicMock(),
        is_fully_idle=MagicMock(),
    )

    result = scheduler.update_weights_from_modelexpress(
        UpdateWeightsFromModelExpressReqInput(weight_version="2")
    )

    assert result.success is False
    assert "pipeline parallelism" in result.message
    update.assert_not_called()


@pytest.mark.parametrize("cache", ["shared", "derived"])
def test_model_updater_honors_native_weight_cache_restrictions(monkeypatch, cache):
    generator = Generator()
    updater = model_updater(generator)
    if cache == "shared":
        monkeypatch.setattr(
            weight_updater_module,
            "get_model",
            lambda: SimpleNamespace(weight_cache_mode="client"),
        )
    else:
        monkeypatch.setattr(
            weight_updater_module,
            "_unsupported_derived_weight_cache_error",
            lambda model: (
                "derived weight cache is active"
                if model is updater.get_model()
                else None
            ),
        )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    reports = []

    def gather(results, local, group):
        reports.append(local)
        results[:] = [local, local]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    # Raising here would kill the server instead of failing the request.
    success, message = updater.update_weights_from_modelexpress("2")

    assert success is False
    assert "cache" in message
    # The rejecting rank still joins the gather; skipping it would hang its peers.
    assert [ok for ok, _ in reports] == [False]
    assert generator.staged == []
    assert generator.applied == []


@pytest.mark.parametrize("refitted", [False, True])
def test_shutdown_closes_only_a_client_that_a_refit_created(
    modelexpress_modules, refitted
):
    generator = Generator()
    updater = model_updater(generator)
    if refitted:
        assert updater.update_weights_from_modelexpress("2") == (True, "")
        assert updater.update_weights_from_modelexpress("3") == (True, "")

    updater.close_modelexpress()
    updater.close_modelexpress()

    assert modelexpress_modules.get_modelexpress_generator.call_count == int(refitted)
    assert generator.closed == int(refitted)


def test_scheduler_shutdown_closes_the_modelexpress_client(monkeypatch):
    scheduler = MagicMock()
    monkeypatch.setattr(
        scheduler_module, "destroy_global_experts_capturer", lambda: None
    )
    monkeypatch.setattr(
        scheduler_module, "destroy_global_indexer_capturer", lambda: None
    )
    monkeypatch.setattr(
        scheduler_module.rank_consensus_checker, "shutdown", lambda: None
    )

    scheduler_module.Scheduler.release_host_resources(scheduler)

    weight_updater = scheduler.tp_worker.model_runner.weight_updater
    weight_updater.close_modelexpress.assert_called_once_with()


@pytest.mark.parametrize("endpoint_key", ["url", "server_url"])
def test_shared_config_preserves_p2p_settings(endpoint_key):
    args = SimpleNamespace(
        modelexpress_config=json.dumps(
            {
                endpoint_key: "mx:8001",
                "transport": "transfer_engine",
                "model_name": "model",
            }
        )
    )

    assert modelexpress_url_of(args) == "mx:8001"
    assert modelexpress_transport_of(args) == "transfer_engine"


def test_staging_error_is_reported_without_installing():
    generator = Generator()
    generator.stage_error = RuntimeError("prepare failed")
    success, message = model_updater(generator).update_weights_from_modelexpress("2")

    assert success is False
    assert message == "prepare failed"
    assert generator.applied == []


def test_apply_error_is_reported_and_releases_staging():
    generator = Generator()
    generator.apply_error = RuntimeError("apply failed")
    updater = model_updater(generator)
    success, message = updater.update_weights_from_modelexpress("2")

    assert success is False
    assert message == "apply failed"
    assert generator.staged[0].released == 1


def test_update_can_retry_after_staging_failure():
    generator = Generator()
    generator.stage_error = RuntimeError("stage failed")
    updater = model_updater(generator)

    failed, _ = updater.update_weights_from_modelexpress("2")
    generator.stage_error = None
    installed, _ = updater.update_weights_from_modelexpress("2")

    assert failed is False
    assert installed is True
    assert generator.applied == generator.staged
    assert generator.staged[0].released == 1


def test_staged_cleanup_failure_is_reported(monkeypatch):
    generator = Generator()
    updater = model_updater(generator)
    staged = Staged("2")
    staged.release_error = RuntimeError("lease unavailable")
    monkeypatch.setattr(generator, "stage_weight", lambda **_kwargs: staged)

    success, message = updater.update_weights_from_modelexpress("2")

    assert success is False
    assert message == "lease unavailable"
    assert staged.released == 1


def test_generator_initialization_failure_is_reported(modelexpress_modules):
    modelexpress_modules.get_modelexpress_generator.side_effect = RuntimeError(
        "initialization failed"
    )

    success, message = model_updater(None).update_weights_from_modelexpress("2")

    assert success is False
    assert message == "initialization failed"


def test_tp_results_include_remote_failure(monkeypatch):
    updater = model_updater(Generator())
    group = weight_updater_module.get_parallel().tp_group.cpu_group
    local = (True, "")
    remote = (False, "rank 1 load failed")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    world_size = MagicMock(return_value=2)
    monkeypatch.setattr(torch.distributed, "get_world_size", world_size)

    def gather(results, _local, *, group):
        assert group is weight_updater_module.get_parallel().tp_group.cpu_group
        results[:] = [local, remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    success, message = updater.update_weights_from_modelexpress("2")

    assert success is False
    assert "rank 1 load failed" in message
    world_size.assert_called_once_with(group=group)


def test_tp_reports_remote_failure_after_local_update_and_cleanup(monkeypatch):
    generator = Generator()
    updater = model_updater(generator)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    reports = []

    def gather(results, local, group):
        reports.append(local)
        if len(reports) == 1:
            # Nothing is installed until every rank has staged.
            assert generator.applied == []
            results[:] = [local, (True, "")]
            return
        assert generator.applied == generator.staged
        assert generator.staged[0].released == 1
        results[:] = [
            local,
            (False, "rank 1 apply failed"),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    success, message = updater.update_weights_from_modelexpress("2")

    assert success is False
    assert "rank 1 apply failed" in message
    assert reports == [(True, ""), (True, "")]


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_staging_failure_on_any_rank_installs_nothing(monkeypatch, failing_rank):
    generator = Generator()
    # A failed release must not escape once the ranks disagreed.
    generator.release_error = RuntimeError("lease unavailable")
    if failing_rank == 0:
        generator.stage_error = RuntimeError("rank 0 stage failed")
    peer = (False, "rank 1 stage failed") if failing_rank == 1 else (True, "")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    reports = []

    def gather(results, local, group):
        reports.append(local)
        results[:] = [local, peer]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    success, message = model_updater(generator).update_weights_from_modelexpress("2")

    assert success is False
    assert f"rank {failing_rank} stage failed" in message
    restore = weight_updater_module.DefaultModelLoader.restore_weights_before_loading
    restore.assert_not_called()
    assert generator.applied == []
    assert [s.released for s in generator.staged] == ([1] if failing_rank else [])
    # The rank that failed to stage still joins the agreement.
    assert [ok for ok, _ in reports] == [failing_rank == 1]


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("remote_success", [False, True])
@pytest.mark.parametrize("paused", [False, True])
def test_tokenizer_reports_all_ranks_without_extra_fanout(
    remote_success, paused, multimodal
):
    results = [
        ModelExpressWeightUpdateReqOutput(success=True),
        ModelExpressWeightUpdateReqOutput(
            success=remote_success,
            message="" if remote_success else "rank 1 load failed",
        ),
    ]
    tokenizer = SimpleNamespace(
        model_update_lock=SimpleNamespace(writer_lock=asyncio.Lock()),
        is_pause_cond=asyncio.Condition(),
        is_pause=paused,
        auto_create_handle_loop=MagicMock(),
        modelexpress_communicator=AsyncMock(return_value=results),
        mm_processor=MagicMock() if multimodal else None,
        _update_weight_version_if_provided=MagicMock(),
    )
    tokenizer._weight_update_session_call = MethodType(
        TokenizerControlMixin._weight_update_session_call, tokenizer
    )
    request = UpdateWeightsFromModelExpressReqInput(weight_version="2")

    async def update():
        # Pausing already owns the writer lock; refitting must not acquire it twice.
        writer = tokenizer.model_update_lock.writer_lock
        if paused:
            await writer.acquire()
        try:
            return await asyncio.wait_for(
                TokenizerControlMixin.update_weights_from_modelexpress(
                    tokenizer, request
                ),
                timeout=1,
            )
        finally:
            if paused:
                writer.release()

    result = asyncio.run(update())

    assert result.success is remote_success
    tokenizer.modelexpress_communicator.assert_awaited_once_with(request)
    if multimodal:
        clear = tokenizer.mm_processor.clear_preprocess_cache
        assert clear.call_count == int(remote_success)
    if remote_success:
        tokenizer._update_weight_version_if_provided.assert_called_once_with("2")
    else:
        assert "rank 1 load failed" in result.message
        tokenizer._update_weight_version_if_provided.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
