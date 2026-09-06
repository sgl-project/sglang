"""Unit tests for the post-capture startup weight-loading component."""

import dataclasses
import re
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelImpl
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    ModelStorageManifest,
    StartupWeightLoadManager,
    StartupWeightLoadOptions,
    StartupWeightLoadState,
)
from sglang.srt.model_loader.loader import DefaultModelLoader, ModelOptModelLoader
from sglang.srt.model_loader.weight_utils import initialize_capture_safe_weights
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_STARTUP_MODULE = (
    "sglang.srt.model_executor.model_runner_components.startup_weight_load"
)


def _make_options(**overrides):
    options = StartupWeightLoadOptions(
        device="cuda",
        is_cuda_platform=True,
        cuda_graph_enabled=True,
        prefill_cuda_graph_backend=Backend.FULL,
        is_draft_worker=False,
        speculative_algorithm=None,
        cpu_offload_gb=0,
        offload_group_size=-1,
        enable_weights_cpu_backup=False,
        enable_lora=False,
        has_lora_paths=False,
        weight_loader_disable_mmap=False,
        weight_loader_drop_cache_after_load=False,
        has_custom_weight_loader=False,
        enable_torch_compile=False,
        prefetch_num_threads=4,
    )
    return dataclasses.replace(options, **overrides)


def _make_model_config(**overrides):
    values = dict(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        dtype=torch.bfloat16,
        quantization=None,
        modelopt_quant=None,
        is_multimodal=False,
        is_generation=True,
        model_impl=ModelImpl.SGLANG,
        _resolved_model_impl=ModelImpl.SGLANG,
        _is_already_quantized=lambda: False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _RecordingPrefetchHandle:
    def __init__(self, trace, *, done=False, errors=()):
        self._trace = trace
        self.done = done
        self.errors = errors

    @property
    def failed(self):
        return bool(self.errors)

    def wait(self, timeout=None):
        self._trace.append("wait_prefetch")

    def stop(self, timeout=None):
        self._trace.append("stop_prefetch")
        self.wait()
        self.done = True


class _RecordingLoader:
    def __init__(self, model, trace):
        self._model = model
        self._trace = trace
        self.prefetch_handle = _RecordingPrefetchHandle(trace)

    def initialize_model_for_startup(self, *, model_config, device_config):
        self._trace.append("initialize")
        return self._model

    def resolve_model_weights(self, model_config, model):
        self._trace.append("resolve")
        return (object(),)

    def start_checkpoint_prefetch(self, resolved_sources, *, num_threads):
        self._trace.append("start_prefetch")
        return self.prefetch_handle

    def prepare_model_for_capture(self, *, model, model_config):
        self._trace.append("prepare_capture")
        return model

    def commit_model_weights(
        self,
        *,
        model,
        model_config,
        resolved_sources,
        target_device,
        startup_prefetch_active,
    ):
        self._trace.append("commit")
        self.startup_prefetch_active = startup_prefetch_active
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(3)


class _TiedWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.tied_weight = self.weight
        self.register_buffer("scale", torch.ones(2))


class _RuntimeStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.scratch = {"indices": torch.ones(1)}
        self.register_buffer("graph_scale", torch.ones(1), persistent=False)


class _PackedWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.optional_scale = nn.Parameter(torch.full((1,), -1.0))
        self.optional_scale._skip_weight_check = True
        self.packed_weight = nn.Parameter(
            torch.full((2, 2), 7, dtype=torch.uint8), requires_grad=False
        )
        self.packed_weight.weight_loader = lambda param, loaded: param.copy_(loaded)
        self.routing_table = nn.Parameter(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32), requires_grad=False
        )


class _UnsupportedDefaultLoaderSubclass(DefaultModelLoader):
    pass


class TestStartupWeightLoadSelector(CustomTestCase):
    def setUp(self):
        self.load_config = LoadConfig(load_format=LoadFormat.SAFETENSORS)
        self.loader = DefaultModelLoader(self.load_config)
        self.device_config = DeviceConfig("cuda", 0)

    def _create(
        self,
        *,
        options=None,
        model_config=None,
        load_config=None,
        loader=None,
    ):
        model_config = _make_model_config() if model_config is None else model_config
        return StartupWeightLoadManager.create(
            loader=self.loader if loader is None else loader,
            model_config=model_config,
            load_config=self.load_config if load_config is None else load_config,
            device_config=self.device_config,
            options=_make_options() if options is None else options,
        )

    def test_supported_overlap_creates_a_manager(self):
        supported_cases = (
            dict(model_config=_make_model_config(quantization="mxfp8")),
            dict(
                model_config=_make_model_config(
                    hf_config=SimpleNamespace(architectures=["OtherForCausalLM"])
                )
            ),
            dict(
                model_config=_make_model_config(
                    model_impl=ModelImpl.TRANSFORMERS,
                    _resolved_model_impl=ModelImpl.TRANSFORMERS,
                )
            ),
        )
        self.assertIsInstance(self._create(), StartupWeightLoadManager)
        for kwargs in supported_cases:
            with self.subTest(kwargs=kwargs):
                self.assertIsInstance(
                    self._create(**kwargs),
                    StartupWeightLoadManager,
                )

    def test_prequantized_modelopt_loader_supports_overlap(self):
        model_config = _make_model_config(
            quantization="modelopt_fp4",
            _is_already_quantized=lambda: True,
        )
        loader = ModelOptModelLoader(self.load_config)

        self.assertIsInstance(
            self._create(model_config=model_config, loader=loader),
            StartupWeightLoadManager,
        )

    def test_options_accept_current_server_args_schema(self):
        """Removed server options must not break overlap startup initialization."""
        server_args = ServerArgs(
            model_path="dummy", cuda_graph_config=CudaGraphConfig()
        )
        # The parallel sizes come from the bags, so the config has to be published.
        publish(server_args, role="test")
        self.addCleanup(reset_context)
        options = StartupWeightLoadOptions.from_published_config(
            is_draft_worker=False,
        )

        self.assertIsInstance(options, StartupWeightLoadOptions)

    def test_unsupported_overlap_is_rejected_instead_of_falling_back(self):
        cases = (
            (
                "non_cuda",
                dict(options=_make_options(device="cpu", is_cuda_platform=False)),
                "CUDA only",
            ),
            (
                "graphs_disabled",
                dict(options=_make_options(cuda_graph_enabled=False)),
                "CUDA graph capture is disabled",
            ),
            (
                "tc_piecewise_prefill",
                dict(
                    options=_make_options(
                        prefill_cuda_graph_backend=Backend.TC_PIECEWISE
                    )
                ),
                "tc_piecewise prefill CUDA graphs are not supported",
            ),
            (
                "pt_checkpoint",
                dict(load_config=LoadConfig(load_format=LoadFormat.PT)),
                "load format must be auto or safetensors",
            ),
            (
                "draft_worker",
                dict(options=_make_options(is_draft_worker=True)),
                "draft workers are not supported",
            ),
            (
                "draft_model_checkpoint",
                dict(
                    load_config=LoadConfig(
                        load_format=LoadFormat.SAFETENSORS,
                        draft_model_idx=0,
                    )
                ),
                "draft model loading is unsupported",
            ),
            (
                "speculative_decoding",
                dict(options=_make_options(speculative_algorithm="EAGLE")),
                "speculative decoding is not supported",
            ),
            (
                "loader_without_split_contract",
                dict(loader=_UnsupportedDefaultLoaderSubclass(self.load_config)),
                "does not support split startup loading",
            ),
            (
                "online_modelopt",
                dict(
                    loader=ModelOptModelLoader(self.load_config),
                    model_config=_make_model_config(quantization="modelopt_fp4"),
                ),
                "does not support split startup loading",
            ),
            (
                "layer_group_offload",
                dict(options=_make_options(offload_group_size=1)),
                "layer-group offloading is not supported",
            ),
            (
                "torch_compile",
                dict(options=_make_options(enable_torch_compile=True)),
                "torch.compile is not supported",
            ),
        )
        for name, kwargs, reason in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, re.escape(reason)):
                    self._create(**kwargs)


class TestStartupWeightLoadManager(CustomTestCase):
    def _manager(self, loader):
        return StartupWeightLoadManager(
            loader=loader,
            model_config=_make_model_config(),
            device_config=DeviceConfig("cpu", 0),
            options=_make_options(),
        )

    def test_prepare_capture_finalize_state_and_order(self):
        trace = []
        model = _TiedWeightModel()
        manager = self._manager(_RecordingLoader(model, trace))

        self.assertEqual(manager.state, StartupWeightLoadState.CREATED)
        self.assertIs(manager.prepare(), model)
        self.assertEqual(manager.state, StartupWeightLoadState.CAPTURE_READY)
        manager.start_prefetch()
        self.assertEqual(manager.state, StartupWeightLoadState.PREFETCHING)

        # CUDA graph capture is owned by Scheduler and occurs between these calls.
        trace.append("capture")
        with (
            patch(
                f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"
            ) as parallel_state_patch,
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            patch(f"{_STARTUP_MODULE}.logger.info") as log_info,
        ):
            manager.finalize()

        self.assertEqual(manager.state, StartupWeightLoadState.READY)
        self.assertEqual(
            trace,
            [
                "initialize",
                "resolve",
                "prepare_capture",
                "start_prefetch",
                "capture",
                "commit",
                "stop_prefetch",
                "wait_prefetch",
            ],
        )

        # Finalization is idempotent after a successful commit.
        manager.finalize()
        self.assertEqual(trace.count("commit"), 1)
        self.assertIs(model.weight, model.tied_weight)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3))
        self.assertTrue(log_info.call_args.args[0].startswith("Load weight end."))
        self.assertTrue(manager._loader.startup_prefetch_active)
        self.assertEqual(
            parallel_state_patch.call_args_list,
            [call(), call(reverse=True)],
        )

    def test_finalize_rejects_graph_visible_storage_rebind(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)

        def rebind_tied_weight(**kwargs):
            trace.append("commit")
            model.tied_weight = nn.Parameter(model.tied_weight.detach().clone())

        loader.commit_model_weights = rebind_tied_weight
        manager = self._manager(loader)
        manager.prepare()
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            self.assertRaisesRegex(
                RuntimeError,
                "changed graph-visible tensor storage: parameter:tied_weight",
            ),
        ):
            manager.finalize()

    def test_finalize_rejects_parameter_left_at_capture_sentinel(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)

        def skip_commit(**kwargs):
            trace.append("commit")

        loader.commit_model_weights = skip_commit
        manager = self._manager(loader)
        manager.prepare()
        with torch.no_grad():
            model.weight.fill_(1e-3)
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            self.assertRaisesRegex(
                RuntimeError,
                "did not populate capture-safe parameters: parameter:tied_weight",
            ),
        ):
            manager.finalize()

    def test_completed_prefetch_restores_normal_loader(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)
        loader.prefetch_handle.done = True
        manager = self._manager(loader)
        manager.prepare()
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
        ):
            manager.finalize()

        self.assertFalse(loader.startup_prefetch_active)
        self.assertIn("wait_prefetch", trace)
        self.assertNotIn("stop_prefetch", trace)

    def test_failed_prefetch_falls_back_and_logs_summary(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)
        loader.prefetch_handle.errors = (("bad.safetensors", OSError("failed")),)
        manager = self._manager(loader)
        manager.prepare()
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            patch(f"{_STARTUP_MODULE}.logger.warning") as warning,
        ):
            manager.finalize()

        self.assertFalse(loader.startup_prefetch_active)
        warning.assert_called_once()
        self.assertIn("falling back", warning.call_args.args[2])

    def test_stop_timeout_after_commit_does_not_fail_startup(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)

        def _stop_times_out(timeout=None):
            trace.append("stop_prefetch")
            raise TimeoutError("Timed out waiting for checkpoint prefetching")

        loader.prefetch_handle.stop = _stop_times_out
        manager = self._manager(loader)
        manager.prepare()
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            patch(f"{_STARTUP_MODULE}.logger.warning") as warning,
        ):
            manager.finalize()

        self.assertEqual(manager.state, StartupWeightLoadState.READY)
        self.assertIn("stop_prefetch", trace)
        warning.assert_called_once()
        self.assertIn("did not stop within its timeout", warning.call_args.args[0])

    def test_start_prefetch_requires_capture_ready_and_starts_once(self):
        trace = []
        manager = self._manager(_RecordingLoader(nn.Linear(2, 2), trace))

        with self.assertRaisesRegex(RuntimeError, "from state"):
            manager.start_prefetch()

        manager.prepare()
        manager.start_prefetch()
        self.assertEqual(manager.state, StartupWeightLoadState.PREFETCHING)

        with self.assertRaisesRegex(RuntimeError, "from state"):
            manager.start_prefetch()
        self.assertEqual(trace.count("start_prefetch"), 1)


class TestModelStorageManifest(CustomTestCase):
    def test_in_place_updates_preserve_the_manifest(self):
        model = _TiedWeightModel()
        manifest = ModelStorageManifest.capture(model)

        with torch.no_grad():
            model.weight.fill_(2)
            model.scale.fill_(3)

        self.assertEqual(manifest.changed_names(model), ())

    def test_manifest_keeps_strong_tensor_references(self):
        model = _TiedWeightModel()
        manifest = ModelStorageManifest.capture(model)

        metadata = dict(manifest.tensors)["parameter:weight"]
        self.assertIs(metadata.tensor, model.weight)

    def test_capture_sentinel_check_ignores_buffers(self):
        model = _TiedWeightModel()
        with torch.no_grad():
            model.weight.fill_(1e-3)
            model.scale.fill_(1e-3)
        manifest = ModelStorageManifest.capture(model)

        self.assertEqual(
            manifest.unchanged_parameter_names(1e-3),
            ("parameter:tied_weight",),
        )

    def test_parameter_rebind_and_alias_break_are_detected(self):
        model = _TiedWeightModel()
        manifest = ModelStorageManifest.capture(model)

        model.tied_weight = nn.Parameter(model.tied_weight.detach().clone())

        self.assertEqual(
            manifest.changed_names(model),
            ("parameter:tied_weight",),
        )

    def test_nonpersistent_buffers_are_manifested_but_plain_state_is_not(self):
        model = _RuntimeStateModel()
        manifest = ModelStorageManifest.capture(model)

        model.scratch["indices"] = model.scratch["indices"].clone()
        with torch.no_grad():
            model.graph_scale.fill_(2)

        self.assertNotIn("graph_scale", model.state_dict())
        self.assertEqual(manifest.changed_names(model), ())

        model.graph_scale = model.graph_scale.clone()

        self.assertEqual(manifest.changed_names(model), ("buffer:graph_scale",))

    def test_integral_weights_are_excluded_from_sentinel_check(self):
        model = _PackedWeightModel()
        initialize_capture_safe_weights(model)
        manifest = ModelStorageManifest.capture(model)

        self.assertEqual(
            manifest.unchanged_parameter_names(1e-3),
            ("parameter:weight",),
        )


class TestCaptureSafeWeightInitialization(CustomTestCase):
    def test_parameters_are_filled_with_dtype_safe_values(self):
        model = _PackedWeightModel()
        model.register_buffer("scale", torch.ones(2))

        initialize_capture_safe_weights(model, value=0.125)

        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 0.125))
        torch.testing.assert_close(
            model.packed_weight, torch.zeros_like(model.packed_weight)
        )
        torch.testing.assert_close(model.optional_scale, torch.full((1,), -1.0))
        torch.testing.assert_close(
            model.routing_table,
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        )
        torch.testing.assert_close(model.scale, torch.ones_like(model.scale))

    def test_omitted_kv_cache_scales_keep_their_serial_default(self):
        layer = nn.Module()
        method = BaseKVCacheMethod(SimpleNamespace())
        method.create_weights(layer)

        initialize_capture_safe_weights(layer)
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.k_scale.item(), 1.0)
        self.assertEqual(layer.v_scale.item(), 1.0)
        self.assertEqual(layer.k_scale_float, 1.0)
        self.assertEqual(layer.v_scale_float, 1.0)


class _LifecycleRunner:
    def __init__(self, name, trace):
        self._name = name
        self._trace = trace

    def start_startup_weight_load(self):
        self._trace.append(f"start:{self._name}")

    def finalize_startup_weight_load(self):
        self._trace.append(f"finalize:{self._name}")


class TestStartupWeightLoadFanout(CustomTestCase):
    def test_primary_and_multi_runner_extras_are_started_once(self):
        trace = []
        primary = _LifecycleRunner("primary", trace)
        extra_1 = _LifecycleRunner("extra_1", trace)
        extra_2 = _LifecycleRunner("extra_2", trace)
        worker = TpModelWorker.__new__(TpModelWorker)
        worker._model_runner = primary
        worker.model_runner_list = [primary, extra_1, extra_2]

        worker.start_startup_weight_load()

        self.assertEqual(
            trace,
            ["start:primary", "start:extra_1", "start:extra_2"],
        )

    def test_primary_and_multi_runner_extras_are_finalized_once(self):
        for multi_runner in (False, True):
            with self.subTest(multi_runner=multi_runner):
                trace = []
                primary = _LifecycleRunner("primary", trace)
                extra_1 = _LifecycleRunner("extra_1", trace)
                extra_2 = _LifecycleRunner("extra_2", trace)
                worker = TpModelWorker.__new__(TpModelWorker)
                worker._model_runner = primary
                worker.model_runner_list = (
                    [primary, extra_1, extra_2] if multi_runner else []
                )

                worker.finalize_startup_weight_load()

                self.assertEqual(
                    trace,
                    (
                        ["finalize:primary", "finalize:extra_1", "finalize:extra_2"]
                        if multi_runner
                        else ["finalize:primary"]
                    ),
                )


class _RunnerStartupManager:
    def __init__(self, trace):
        self._trace = trace

    def start_prefetch(self):
        self._trace.append("start_prefetch")

    def finalize(self):
        self._trace.append("finalize")


class TestModelRunnerStartupWeightLoadOwnership(CustomTestCase):
    @staticmethod
    def _runner(manager):
        runner = ModelRunner.__new__(ModelRunner)
        runner.startup_weight_load = manager
        runner.server_args = SimpleNamespace(
            elastic_ep_backend=None,
            is_ep_joiner=False,
        )
        runner.ps = SimpleNamespace(tp_rank=0)
        return runner

    def test_start_delegates_to_the_manager(self):
        trace = []
        runner = self._runner(_RunnerStartupManager(trace))

        runner.start_startup_weight_load()

        self.assertEqual(trace, ["start_prefetch"])

    def test_success_releases_ownership_after_the_barrier(self):
        trace = []
        manager = _RunnerStartupManager(trace)
        runner = self._runner(manager)

        def barrier(**kwargs):
            self.assertIs(runner.startup_weight_load, manager)
            trace.append("barrier")

        with (
            patch(
                "sglang.srt.model_executor.model_runner.dist_barrier_after_load",
                side_effect=barrier,
            ),
            get_context().override_server_args(),
        ):
            runner.finalize_startup_weight_load()

        self.assertEqual(trace, ["finalize", "barrier"])
        self.assertIsNone(runner.startup_weight_load)


class _SchedulerWorker:
    def __init__(self, trace, *, post_capture_active=False):
        self._trace = trace
        self.model_runner = SimpleNamespace(
            device="cuda",
            forward_stream=object(),
            prewarm_sampling=lambda: trace.append("prewarm"),
            token_to_kv_pool=SimpleNamespace(post_capture_active=post_capture_active),
            post_capture_resize_kv_pool=lambda: trace.append("resize"),
        )

    def start_startup_weight_load(self):
        self._trace.append("start")

    def finalize_startup_weight_load(self):
        self._trace.append("finalize")


class TestStartupWeightLoadSchedulerRouting(CustomTestCase):
    @staticmethod
    def _scheduler(worker, trace, *, mode, draft_worker=None):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(
            is_startup_weight_load_overlap=mode == "overlap"
        )
        scheduler.init_tp_model_worker = lambda: setattr(scheduler, "tp_worker", worker)
        scheduler.maybe_init_draft_worker = lambda: setattr(
            scheduler, "draft_worker", draft_worker
        )
        scheduler.init_memory_pools = lambda: trace.append("memory_pool")
        scheduler.init_all_attention_backends = lambda: trace.append("attention")
        scheduler.init_all_cuda_graphs = lambda: trace.append("capture")
        return scheduler

    def _run_startup(self, mode, *, use_draft_worker=False):
        trace = []
        worker = _SchedulerWorker(trace, post_capture_active=True)
        draft_worker = (
            SimpleNamespace(prewarm_sampling=lambda: trace.append("draft_prewarm"))
            if use_draft_worker
            else None
        )
        scheduler = self._scheduler(
            worker,
            trace,
            mode=mode,
            draft_worker=draft_worker,
        )

        class StreamContext:
            def __enter__(self):
                trace.append("stream_enter")

            def __exit__(self, *_args):
                trace.append("stream_exit")

        def stream_context(stream):
            self.assertIs(stream, worker.model_runner.forward_stream)
            return StreamContext()

        def stop_after_startup():
            raise RuntimeError("stop after startup")

        scheduler.spec_algorithm = SimpleNamespace(is_none=stop_after_startup)

        with (
            patch(
                "sglang.srt.managers.scheduler.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(
                        elastic_ep_backend=None,
                        ep_join_mode=None,
                    )
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler.torch.get_device_module",
                return_value=SimpleNamespace(stream=stream_context),
            ),
            self.assertRaisesRegex(RuntimeError, "stop after startup"),
        ):
            scheduler.init_model_worker()

        return trace

    def test_serial_path_skips_overlap_hooks(self):
        self.assertEqual(
            self._run_startup("serial"),
            [
                "memory_pool",
                "attention",
                "capture",
                "stream_enter",
                "prewarm",
                "stream_exit",
                "resize",
            ],
        )

    def test_overlap_starts_before_capture_and_finalizes_after(self):
        self.assertEqual(
            self._run_startup("overlap"),
            [
                "start",
                "memory_pool",
                "attention",
                "capture",
                "stream_enter",
                "prewarm",
                "stream_exit",
                "resize",
                "finalize",
            ],
        )

    def test_draft_worker_prewarm_uses_target_forward_stream(self):
        self.assertEqual(
            self._run_startup("serial", use_draft_worker=True),
            [
                "memory_pool",
                "attention",
                "capture",
                "stream_enter",
                "draft_prewarm",
                "stream_exit",
                "resize",
            ],
        )


if __name__ == "__main__":
    unittest.main()
