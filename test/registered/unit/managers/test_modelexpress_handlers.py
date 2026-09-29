# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
import torch

sys.modules.setdefault("sgl_kernel", MagicMock())
for name in ("quantization", "scalar_type", "flash_attn", "flash_mla", "mamba"):
    sys.modules.setdefault(f"sgl_kernel.{name}", MagicMock())

from sglang.srt.arg_groups.overrides import (  # noqa: E402
    modelexpress_transport_of,
    modelexpress_url_of,
)
from sglang.srt.managers import scheduler as scheduler_module  # noqa: E402
from sglang.srt.managers.io_struct import (  # noqa: E402
    ModelExpressWeightUpdateReqOutput,
    UpdateWeightsFromModelExpressReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
)
from sglang.srt.managers.tokenizer_control_mixin import (  # noqa: E402
    TokenizerControlMixin,
)
from sglang.srt.model_executor.model_runner_components import (  # noqa: E402
    weight_updater as weight_updater_module,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (  # noqa: E402
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

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
        weight_updater_module, "_unsupported_derived_weight_cache_error", lambda: None
    )
    return integration


class Staged:
    def __init__(self, version_id):
        self.version_id = version_id
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

    def stage_weight(self, *, version):
        if self.stage_error is not None:
            raise self.stage_error
        if self.staged and not self.staged[-1].released:
            staged = self.staged[-1]
            if staged.version_id != version.version_id:
                raise RuntimeError("another generator update is still active")
            return staged
        staged = Staged(version.version_id)
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
        tp_group=SimpleNamespace(cpu_group=object()),
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


@pytest.mark.parametrize("success", [True, False])
@pytest.mark.parametrize(
    "flush_options", [{}, {"flush_cache": False}, {"torch_empty_cache": True}]
)
def test_scheduler_flushes_before_recording_success(success, flush_options):
    events = MagicMock()
    events.update.return_value = (success, "updated" if success else "stage failed")
    events.flush.return_value = True
    scheduler = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(update_weights_from_modelexpress=events.update),
        draft_worker=None,
        tp_cpu_group=None,
        memory_saver_adapter=None,
        flush_cache=events.flush,
        is_fully_idle=MagicMock(),
        scheduler=SimpleNamespace(record_weight_version_change=events.record),
    )
    request = UpdateWeightsFromModelExpressReqInput(weight_version="2", **flush_options)

    result = scheduler.update_weights_from_modelexpress(request)

    assert result.success is success
    assert result.message == ("updated" if success else "stage failed")
    expected = [call.update(request)]
    if success:
        if request.flush_cache:
            expected.append(call.flush(empty_cache=request.torch_empty_cache))
        expected.append(call.record(new_version="2"))
    assert events.mock_calls == expected


def test_failed_post_update_flush_does_not_record_version():
    record = MagicMock()
    scheduler = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(
            update_weights_from_modelexpress=lambda request: (True, "")
        ),
        draft_worker=None,
        tp_cpu_group=None,
        memory_saver_adapter=None,
        flush_cache=lambda **kwargs: False,
        is_fully_idle=MagicMock(),
        scheduler=SimpleNamespace(record_weight_version_change=record),
    )

    with pytest.raises(AssertionError, match="Cache flush failed"):
        scheduler.update_weights_from_modelexpress(
            UpdateWeightsFromModelExpressReqInput(weight_version="2")
        )

    record.assert_not_called()


def test_scheduler_rejects_draft_models_before_updating():
    update = MagicMock()
    scheduler = SchedulerWeightUpdaterManager(
        tp_worker=SimpleNamespace(update_weights_from_modelexpress=update),
        draft_worker=object(),
        tp_cpu_group=None,
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
            lambda: "derived weight cache is active",
        )

    with pytest.raises(RuntimeError, match="cache"):
        updater.update_weights_from_modelexpress("2")

    assert generator.staged == []
    assert generator.applied == []


@pytest.mark.parametrize("initialized", [False, True])
def test_scheduler_shutdown_closes_only_initialized_client(monkeypatch, initialized):
    generator = Generator()
    runner = model_updater(generator if initialized else None).get_model_runner()
    scheduler = SimpleNamespace(
        hisparse_coordinator=None,
        tree_cache=MagicMock(),
        decode_offload_manager=None,
        tp_worker=SimpleNamespace(model_runner=runner),
    )
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

    assert generator.closed == int(initialized)
    assert hasattr(runner, "modelexpress_generator") is initialized


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
    local = (True, "")
    remote = (False, "rank 1 load failed")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    def gather(results, _local, group):
        results[:] = [local, remote]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    success, message = updater.update_weights_from_modelexpress("2")

    assert success is False
    assert "rank 1 load failed" in message


def test_tp_reports_remote_failure_after_local_update_and_cleanup(monkeypatch):
    generator = Generator()
    updater = model_updater(generator)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)

    reports = []

    def gather(results, local, group):
        reports.append(local)
        assert generator.applied == generator.staged
        assert generator.staged[0].released == 1
        results[:] = [
            local,
            (False, "rank 1 stage failed"),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    success, message = updater.update_weights_from_modelexpress("2")

    assert success is False
    assert "rank 1 stage failed" in message
    assert reports == [(True, "")]


@pytest.mark.parametrize("remote_success", [False, True])
def test_tokenizer_reports_all_ranks_without_extra_fanout(remote_success):
    results = [
        ModelExpressWeightUpdateReqOutput(success=True),
        ModelExpressWeightUpdateReqOutput(
            success=remote_success,
            message="" if remote_success else "rank 1 load failed",
        ),
    ]
    tokenizer = SimpleNamespace(
        model_update_lock=SimpleNamespace(writer_lock=MagicMock()),
        auto_create_handle_loop=MagicMock(),
        modelexpress_communicator=AsyncMock(return_value=results),
        _update_weight_version_if_provided=MagicMock(),
    )
    request = UpdateWeightsFromModelExpressReqInput(weight_version="2")
    result = asyncio.run(
        TokenizerControlMixin.update_weights_from_modelexpress(tokenizer, request)
    )

    assert result.success is remote_success
    tokenizer.modelexpress_communicator.assert_awaited_once_with(request)
    if remote_success:
        tokenizer._update_weight_version_if_provided.assert_called_once_with("2")
    else:
        assert "rank 1 load failed" in result.message
        tokenizer._update_weight_version_if_provided.assert_not_called()
