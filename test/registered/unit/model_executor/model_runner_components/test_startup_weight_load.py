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
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    ModelStorageManifest,
    StartupWeightLoadManager,
    StartupWeightLoadOptions,
    StartupWeightLoadState,
)
from sglang.srt.model_loader.loader import DefaultModelLoader

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_STARTUP_MODULE = (
    "sglang.srt.model_executor.model_runner_components.startup_weight_load"
)


class _CanonicalModel:
    pass


class _ExternalModel:
    pass


def _make_options(**overrides):
    options = StartupWeightLoadOptions(
        device="cuda",
        is_cuda_platform=True,
        cuda_graph_enabled=True,
        prefill_cuda_graph_backend=Backend.FULL,
        is_draft_worker=False,
        speculative_algorithm=None,
        tp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        pp_size=1,
        dp_size=1,
        ep_size=1,
        cpu_offload_gb=0,
        offload_group_size=-1,
        enable_memory_saver=False,
        enable_weights_cpu_backup=False,
        torchao_config="",
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
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _RecordingPrefetchHandle:
    def __init__(self, trace):
        self._trace = trace

    def wait(self, timeout=None):
        self._trace.append("wait_prefetch")

    def stop(self):
        self._trace.append("stop_prefetch")
        self.wait()


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
    ):
        self._trace.append("commit")
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(3)


class _TiedWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.tied_weight = self.weight
        self.register_buffer("scale", torch.ones(2))


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
        resolved_model_class=None,
    ):
        model_config = _make_model_config() if model_config is None else model_config
        architecture = model_config.hf_config.architectures[0]
        with (
            patch(
                f"{_STARTUP_MODULE}.get_model_architecture",
                return_value=(
                    resolved_model_class or _CanonicalModel,
                    architecture,
                ),
            ),
            patch(
                f"{_STARTUP_MODULE}._get_canonical_model_class",
                return_value=_CanonicalModel,
            ),
        ):
            return StartupWeightLoadManager.create(
                loader=self.loader if loader is None else loader,
                model_config=model_config,
                load_config=self.load_config if load_config is None else load_config,
                device_config=self.device_config,
                options=_make_options() if options is None else options,
            )

    def test_supported_overlap_creates_a_manager(self):
        self.assertIsInstance(self._create(), StartupWeightLoadManager)
        self.assertIsInstance(
            self._create(options=_make_options(tp_size=2)),
            StartupWeightLoadManager,
        )

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
                "tp3",
                dict(options=_make_options(tp_size=3)),
                "only TP1 and TP2 are supported",
            ),
            (
                "attention_context_parallel",
                dict(options=_make_options(tp_size=2, attn_cp_size=2)),
                "attention context parallelism is not supported",
            ),
            (
                "decode_context_parallel",
                dict(options=_make_options(tp_size=2, dcp_size=2)),
                "decode context parallelism is not supported",
            ),
            (
                "quantized_model",
                dict(model_config=_make_model_config(quantization="fp8")),
                "quantization is not supported",
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
            (
                "transformers_model_impl",
                dict(
                    model_config=_make_model_config(
                        model_impl=ModelImpl.TRANSFORMERS,
                        _resolved_model_impl=ModelImpl.TRANSFORMERS,
                    ),
                    resolved_model_class=_ExternalModel,
                ),
                "the native SGLang model implementation is required",
            ),
            (
                "external_model_implementation",
                dict(resolved_model_class=_ExternalModel),
                "the native SGLang model implementation is required",
            ),
            (
                "unknown_architecture",
                dict(
                    model_config=_make_model_config(
                        hf_config=SimpleNamespace(architectures=["OtherForCausalLM"])
                    )
                ),
                "model architecture is not in the startup-overlap allowlist",
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

    def test_parameter_rebind_and_alias_break_are_detected(self):
        model = _TiedWeightModel()
        manifest = ModelStorageManifest.capture(model)

        model.tied_weight = nn.Parameter(model.tied_weight.detach().clone())

        self.assertEqual(
            manifest.changed_names(model),
            ("parameter:tied_weight",),
        )


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
            is_ep_scale_joiner=False,
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

        with patch(
            "sglang.srt.model_executor.model_runner.dist_barrier_after_load",
            side_effect=barrier,
        ):
            runner.finalize_startup_weight_load()

        self.assertEqual(trace, ["finalize", "barrier"])
        self.assertIsNone(runner.startup_weight_load)


class _SchedulerWorker:
    def __init__(self, trace, *, post_capture_active=False):
        self._trace = trace
        self.model_runner = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(post_capture_active=post_capture_active),
            post_capture_resize_kv_pool=lambda: trace.append("resize"),
        )

    def start_startup_weight_load(self):
        self._trace.append("start")

    def finalize_startup_weight_load(self):
        self._trace.append("finalize")


class TestStartupWeightLoadSchedulerRouting(CustomTestCase):
    @staticmethod
    def _scheduler(worker, trace, *, mode):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(startup_weight_load_mode=mode)
        scheduler.init_tp_model_worker = lambda: setattr(scheduler, "tp_worker", worker)
        scheduler.maybe_init_draft_worker = lambda: setattr(
            scheduler, "draft_worker", None
        )
        scheduler.init_memory_pools = lambda: trace.append("memory_pool")
        scheduler.init_all_attention_backends = lambda: trace.append("attention")
        scheduler.init_all_cuda_graphs = lambda: trace.append("capture")
        return scheduler

    def _run_startup(self, mode):
        trace = []
        worker = _SchedulerWorker(trace, post_capture_active=True)
        scheduler = self._scheduler(worker, trace, mode=mode)

        def stop_after_startup():
            raise RuntimeError("stop after startup")

        scheduler.spec_algorithm = SimpleNamespace(is_none=stop_after_startup)

        with self.assertRaisesRegex(RuntimeError, "stop after startup"):
            scheduler.init_model_worker()

        return trace

    def test_serial_path_skips_overlap_hooks(self):
        self.assertEqual(
            self._run_startup("serial"),
            ["memory_pool", "attention", "capture", "resize"],
        )

    def test_overlap_starts_before_capture_and_finalizes_after(self):
        self.assertEqual(
            self._run_startup("overlap"),
            ["start", "memory_pool", "attention", "capture", "resize", "finalize"],
        )


if __name__ == "__main__":
    unittest.main()
