"""Unit tests for the post-capture startup weight-loading component."""

import dataclasses
import unittest
from contextlib import contextmanager, nullcontext
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
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    load_model_with_memory_saver,
)
from sglang.srt.model_executor.model_runner_components.startup_weight_load import (
    _STARTUP_WEIGHT_LOAD_PROFILE_SPECS,
    ModelStorageManifest,
    StartupWeightLoadManager,
    StartupWeightLoadOptions,
    StartupWeightLoadPlan,
    StartupWeightLoadProfile,
    StartupWeightLoadState,
    StartupWeightLoadTimings,
    _get_native_model_class,
    _get_startup_weight_load_profile,
    evaluate_startup_weight_load_admission,
    refresh_attention_weight_copies,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import (
    initialize_capture_safe_weights,
    restore_optional_checkpoint_parameter_values,
)
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    publish,
    reset_context,
)
from sglang.srt.server_args import ServerArgs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_STARTUP_MODULE = (
    "sglang.srt.model_executor.model_runner_components.startup_weight_load"
)
_LOAD_MODEL_UTILS_MODULE = (
    "sglang.srt.model_executor.model_runner_components.load_model_utils"
)


class _CanonicalModel:
    pass


class _ExternalModel:
    pass


_PROFILES_BY_ARCHITECTURE = {
    "LlamaForCausalLM": StartupWeightLoadProfile.NATIVE_DENSE,
    "Qwen2ForCausalLM": StartupWeightLoadProfile.NATIVE_DENSE,
    "Qwen3ForCausalLM": StartupWeightLoadProfile.NATIVE_DENSE,
    "Qwen3_5ForConditionalGeneration": StartupWeightLoadProfile.QWEN3_5_HYBRID_VLM,
    "Qwen3_5MoeForConditionalGeneration": (
        StartupWeightLoadProfile.QWEN3_5_MOE_HYBRID_VLM
    ),
    "Qwen3MoeForCausalLM": StartupWeightLoadProfile.QWEN3_MOE_EP,
    "GlmMoeDsaForCausalLM": StartupWeightLoadProfile.GLM_MOE_DSA,
}


def _make_options(**overrides):
    options = StartupWeightLoadOptions(
        device="cuda",
        is_cuda_platform=True,
        cuda_graph_enabled=True,
        moe_a2a_backend="none",
        moe_runner_backend="triton",
        fp8_gemm_runner_backend="triton",
        offload_group_size=-1,
        has_lora=False,
        prefetch_num_threads=4,
    )
    return dataclasses.replace(options, **overrides)


def _make_plan(**overrides):
    plan = StartupWeightLoadPlan(
        profile=StartupWeightLoadProfile.NATIVE_DENSE,
        prefetch_num_threads=4,
    )
    return dataclasses.replace(plan, **overrides)


def _make_model_config(**overrides):
    values = dict(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        dtype=torch.bfloat16,
        quantization=None,
        is_multimodal=False,
        is_generation=True,
        model_impl=ModelImpl.SGLANG,
        _resolved_model_impl=ModelImpl.SGLANG,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_qwen35_hybrid_vlm_model_config(**overrides):
    values = dict(
        hf_config=SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            encoder_only=False,
            language_only=False,
            language_model_only=False,
        ),
        is_multimodal=True,
    )
    values.update(overrides)
    return _make_model_config(**values)


def _make_qwen35_moe_hybrid_vlm_model_config(**overrides):
    values = dict(
        hf_config=SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"],
            encoder_only=False,
            language_only=False,
            language_model_only=False,
        ),
        is_multimodal=True,
    )
    values.update(overrides)
    return _make_model_config(**values)


def _make_qwen3_moe_model_config(**overrides):
    values = dict(
        hf_config=SimpleNamespace(architectures=["Qwen3MoeForCausalLM"]),
    )
    values.update(overrides)
    return _make_model_config(**values)


def _make_glm_moe_dsa_fp8_model_config(**overrides):
    hf_config_overrides = overrides.pop("hf_config_overrides", {})
    hf_config_values = dict(
        architectures=["GlmMoeDsaForCausalLM"],
        cli_factor=1,
        index_topk_pattern=None,
        index_skip_topk_offset=3,
        index_topk_freq=4,
        quantization_config={
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    hf_config_values.update(hf_config_overrides)
    values = dict(
        hf_config=SimpleNamespace(**hf_config_values),
        quantization="fp8",
    )
    values.update(overrides)
    return _make_model_config(**values)


class _RecordingPrefetchHandle:
    def __init__(self, trace, *, done=False, succeeded=True, errors=()):
        self._trace = trace
        self.done = done
        self.succeeded = succeeded
        self.errors = errors

    @property
    def failed(self):
        return bool(self.errors) or (self.done and not self.succeeded)

    def wait(self, timeout=None):
        self._trace.append("wait_prefetch")

    def cancel(self):
        self._trace.append("cancel_prefetch")

    def stop(self, timeout=None):
        self._trace.append("stop_prefetch")
        self.cancel()
        self.wait()
        self.done = True


class _RecordingLoader:
    def __init__(self, model, trace):
        self._model = model
        self._trace = trace
        self.use_safetensors = True
        self.hf_folder = "/dummy"
        self.prefetch_handle = _RecordingPrefetchHandle(trace)

    def initialize_model_for_startup(self, *, model_config, device_config):
        self._trace.append("initialize")
        return self._model

    def resolve_model_weights(self, model_config, model):
        self._trace.append("resolve")
        return tuple(
            SimpleNamespace(
                use_safetensors=self.use_safetensors,
                source=object(),
                hf_folder=self.hf_folder,
            )
            for _ in range(getattr(self, "num_resolved_sources", 1))
        )

    def load_initialized_model_from_resolved_sources(
        self,
        *,
        model,
        model_config,
        resolved_sources,
        target_device,
    ):
        self._trace.append("serial_load")
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(3)
        return model.eval()

    def start_checkpoint_prefetch(self, resolved_sources, *, num_threads):
        self._trace.append("start_prefetch")
        self.prefetch_num_threads = num_threads
        return self.prefetch_handle

    def prepare_model_for_capture(self, *, model, model_config, target_device):
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


class _DerivedTensorModel(_TiedWeightModel):
    def __init__(self):
        super().__init__()
        self.graph_weight = torch.ones(2, 2)

    def named_startup_weight_load_derived_tensors(self):
        return (("graph_weight", self.graph_weight),)


class _NestedDerivedTensorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = _DerivedTensorModel()


class TestStartupWeightLoadSelector(CustomTestCase):
    def setUp(self):
        self.load_config = LoadConfig(load_format=LoadFormat.SAFETENSORS)
        self.loader = DefaultModelLoader(self.load_config)
        self.device_config = DeviceConfig("cuda", 0)
        publish(
            ServerArgs(model_path="dummy", cuda_graph_config=CudaGraphConfig()),
            role="test",
        )
        self.addCleanup(reset_context)

    def _create(
        self,
        *,
        options=None,
        model_config=None,
        load_config=None,
        loader=None,
        resolved_model_class=None,
        native_model_class=None,
    ):
        model_config = _make_model_config() if model_config is None else model_config
        architecture = model_config.hf_config.architectures[0]
        native_model_class = native_model_class or _CanonicalModel
        profile = _PROFILES_BY_ARCHITECTURE.get(
            architecture, StartupWeightLoadProfile.NATIVE_DENSE
        )
        profile_spec = next(
            spec
            for spec in _STARTUP_WEIGHT_LOAD_PROFILE_SPECS
            if spec.profile == profile
        )
        with (
            patch(
                f"{_STARTUP_MODULE}.get_model_architecture",
                return_value=(
                    resolved_model_class or native_model_class,
                    architecture,
                ),
            ),
            patch(
                f"{_STARTUP_MODULE}._get_native_model_class",
                return_value=native_model_class,
            ),
            patch(
                f"{_STARTUP_MODULE}._STARTUP_WEIGHT_LOAD_PROFILE_SPECS",
                (
                    dataclasses.replace(
                        profile_spec,
                        model_classes={"_CanonicalModel": lambda: _CanonicalModel},
                    ),
                ),
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
        manager = self._create()
        self.assertIsInstance(manager, StartupWeightLoadManager)
        self.assertFalse(manager._fallback_to_serial)
        self.assertEqual(
            manager._plan,
            StartupWeightLoadPlan(
                profile=StartupWeightLoadProfile.NATIVE_DENSE,
                prefetch_num_threads=4,
            ),
        )

    def test_profile_registry_is_complete_and_unambiguous(self):
        resolvers = [
            resolve
            for spec in _STARTUP_WEIGHT_LOAD_PROFILE_SPECS
            for resolve in spec.model_classes.values()
        ]
        self.assertEqual(
            {spec.profile for spec in _STARTUP_WEIGHT_LOAD_PROFILE_SPECS},
            set(StartupWeightLoadProfile),
        )
        self.assertEqual(len(resolvers), len(set(resolvers)))

    def test_empty_native_aliases_reuse_the_base_loading_path(self):
        class Alias(_CanonicalModel):
            pass

        class NestedAlias(Alias):
            """Only metadata differs from the native implementation."""

        for model_class in (Alias, NestedAlias):
            with self.subTest(model_class=model_class):
                manager = self._create(
                    native_model_class=model_class,
                    model_config=_make_model_config(
                        hf_config=SimpleNamespace(architectures=[model_class.__name__])
                    ),
                )
                self.assertEqual(
                    manager._plan.profile, StartupWeightLoadProfile.NATIVE_DENSE
                )

    def test_profile_selection_does_not_import_unrelated_models(self):
        def unavailable():
            raise ImportError("optional model dependency is unavailable")

        spec = dataclasses.replace(
            _STARTUP_WEIGHT_LOAD_PROFILE_SPECS[0],
            model_classes={
                "UnrelatedModel": unavailable,
                "_CanonicalModel": lambda: _CanonicalModel,
            },
        )
        with patch(f"{_STARTUP_MODULE}._STARTUP_WEIGHT_LOAD_PROFILE_SPECS", (spec,)):
            self.assertEqual(
                _get_startup_weight_load_profile(_CanonicalModel),
                StartupWeightLoadProfile.NATIVE_DENSE,
            )

    def test_subclass_behavior_changes_require_a_loading_path_review(self):
        for member in (
            "__init__",
            "load_weights",
            "post_load_weights",
            "forward",
            "setting",
        ):
            with self.subTest(member=member):
                model_class = type("Changed", (_CanonicalModel,), {member: object()})
                with self.assertRaisesRegex(
                    ValueError, "no startup overlap loading path"
                ):
                    self._create(native_model_class=model_class)

        class Mixin:
            pass

        class MultipleBases(_CanonicalModel, Mixin):
            pass

        class Meta(type):
            pass

        class CustomMetaclass(_CanonicalModel, metaclass=Meta):
            pass

        for model_class in (MultipleBases, CustomMetaclass):
            with self.subTest(model_class=model_class):
                with self.assertRaisesRegex(
                    ValueError, "no startup overlap loading path"
                ):
                    self._create(native_model_class=model_class)

    def test_native_resolution_preserves_priority_and_rejects_external_overrides(self):
        registry = SimpleNamespace(
            ModelRegistry=SimpleNamespace(
                models={"Native": _CanonicalModel, "External": _ExternalModel}
            ),
            import_model_classes=lambda _, strict: {"Native": _CanonicalModel},
        )
        with patch.dict("sys.modules", {"sglang.srt.models.registry": registry}):
            self.assertIs(
                _get_native_model_class(["Unknown", "Native"]), _CanonicalModel
            )
            self.assertIsNone(_get_native_model_class(["External", "Native"]))
            self.assertIsNone(_get_native_model_class(["Unknown"]))
            registry.ModelRegistry.models["Native"] = _ExternalModel
            self.assertIsNone(_get_native_model_class(["Native"]))

    def test_multiple_architecture_names_use_the_selected_implementation(self):
        manager = self._create(
            model_config=_make_model_config(
                hf_config=SimpleNamespace(
                    architectures=["Unknown", "LlamaForCausalLM", "Qwen2ForCausalLM"]
                )
            )
        )
        self.assertEqual(manager._plan.profile, StartupWeightLoadProfile.NATIVE_DENSE)

    def test_auto_mode_falls_back_for_config_rejection(self):
        with (
            patch.object(
                StartupWeightLoadOptions,
                "from_published_config",
                return_value=_make_options(device="cpu", is_cuda_platform=False),
            ),
            patch(
                f"{_STARTUP_MODULE}.get_model",
                return_value=SimpleNamespace(startup_weight_load_mode="auto"),
            ),
            patch(f"{_STARTUP_MODULE}.logger.info") as log_info,
        ):
            manager = StartupWeightLoadManager.create_from_published_config(
                loader=self.loader,
                model_config=_make_model_config(),
                load_config=self.load_config,
                device_config=self.device_config,
                is_draft_worker=False,
            )

        self.assertIsNone(manager)
        self.assertIn("non_cuda: CUDA only", log_info.call_args.args[1])

    def test_auto_mode_creates_a_manager_when_admitted(self):
        with (
            patch.object(
                StartupWeightLoadOptions,
                "from_published_config",
                return_value=_make_options(),
            ),
            patch(
                f"{_STARTUP_MODULE}.get_model",
                return_value=SimpleNamespace(startup_weight_load_mode="auto"),
            ),
            patch(
                f"{_STARTUP_MODULE}.get_model_architecture",
                return_value=(_CanonicalModel, "LlamaForCausalLM"),
            ),
            patch(
                f"{_STARTUP_MODULE}._get_native_model_class",
                return_value=_CanonicalModel,
            ),
            patch(
                f"{_STARTUP_MODULE}._get_startup_weight_load_profile",
                return_value=StartupWeightLoadProfile.NATIVE_DENSE,
            ),
        ):
            manager = StartupWeightLoadManager.create_from_published_config(
                loader=self.loader,
                model_config=_make_model_config(),
                load_config=self.load_config,
                device_config=self.device_config,
                is_draft_worker=False,
            )

        self.assertIsInstance(manager, StartupWeightLoadManager)
        self.assertTrue(manager._fallback_to_serial)

    def test_admission_collects_preflight_rejections_before_importing_models(self):
        model_config = _make_model_config(quantization="modelopt_fp8")
        with (
            patch(
                f"{_STARTUP_MODULE}.get_model_architecture",
                return_value=(_CanonicalModel, "LlamaForCausalLM"),
            ) as resolve_architecture,
            patch(
                f"{_STARTUP_MODULE}._get_native_model_class",
                return_value=_CanonicalModel,
            ) as get_native_model_class,
        ):
            admission = evaluate_startup_weight_load_admission(
                loader=self.loader,
                model_config=model_config,
                load_config=self.load_config,
                options=_make_options(
                    device="cpu",
                    is_cuda_platform=False,
                    cuda_graph_enabled=False,
                    prefetch_num_threads=0,
                ),
            )

        self.assertFalse(admission.supported)
        self.assertIsNone(admission.plan)
        self.assertEqual(
            tuple(rejection.code for rejection in admission.rejections),
            ("non_cuda", "cuda_graph_disabled", "prefetch_threads"),
        )
        resolve_architecture.assert_not_called()
        get_native_model_class.assert_not_called()

    def test_create_formats_rejection_codes_and_messages(self):
        with self.assertRaisesRegex(
            ValueError,
            "non_cuda: CUDA only; cuda_graph_disabled: .*",
        ):
            self._create(
                options=_make_options(
                    device="cpu",
                    is_cuda_platform=False,
                    cuda_graph_enabled=False,
                ),
                model_config=_make_model_config(quantization="modelopt_fp8"),
            )

    def test_non_native_paths_fail_without_remote_model_resolution(self):
        for architectures, model_impl in (
            ([], ModelImpl.AUTO),
            (["Unknown"], ModelImpl.AUTO),
            (["LlamaForCausalLM"], ModelImpl.TRANSFORMERS),
            (["LlamaForCausalLM"], ModelImpl.MINDSPORE),
        ):
            with (
                self.subTest(architectures=architectures, model_impl=model_impl),
                patch(
                    f"{_STARTUP_MODULE}.get_model_architecture"
                ) as resolve_architecture,
                patch(
                    f"{_STARTUP_MODULE}._get_native_model_class", return_value=None
                ) as get_native_model_class,
            ):
                admission = evaluate_startup_weight_load_admission(
                    loader=self.loader,
                    model_config=_make_model_config(
                        hf_config=SimpleNamespace(architectures=architectures),
                        model_impl=model_impl,
                    ),
                    load_config=self.load_config,
                    options=_make_options(),
                )

                self.assertEqual(
                    tuple(rejection.code for rejection in admission.rejections),
                    ("model_implementation",),
                )
                resolve_architecture.assert_not_called()
                if not architectures or model_impl != ModelImpl.AUTO:
                    get_native_model_class.assert_not_called()

    def _published_options(self, **changes):
        graph_config = CudaGraphConfig()
        graph_config.decode.backend = Backend.FULL
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            cuda_graph_config=graph_config,
            **changes,
        )
        publish(args, role="test")
        with patch(f"{_STARTUP_MODULE}.current_platform.is_cuda", return_value=True):
            return StartupWeightLoadOptions.from_published_config()

    def test_ordinary_runtime_settings_are_not_overlap_admission_inputs(self):
        cases = (
            {"tp_size": 4, "pp_size": 2, "dp_size": 2},
            {"tp_size": 4, "ep_size": 2, "moe_dp_size": 2},
            {"tp_size": 4, "attn_cp_size": 2, "dcp_size": 2},
            {"enable_dp_attention": True, "enable_two_batch_overlap": True},
            {"enable_eplb": True, "ep_num_redundant_experts": 2},
            {"init_expert_location": "random", "enable_elastic_expert_backup": True},
            {"enable_memory_saver": True, "enable_weights_cpu_backup": True},
            {"custom_weight_loader": ["example.loader"], "enable_torch_compile": True},
            {"attention_backend": "flashinfer", "kv_cache_dtype": "fp8_e4m3"},
            {"linear_attn_prefill_backend": "flashinfer"},
            {"speculative_algorithm": "EAGLE3"},
        )
        expected = self._published_options()
        for changes in cases:
            with self.subTest(changes=changes):
                options = self._published_options(**changes)
                self.assertEqual(options, expected)
                self.assertIsInstance(
                    self._create(options=options), StartupWeightLoadManager
                )

    def test_unquantized_profiles_delegate_dtype_and_modality_validation(self):
        profiles = (
            (_make_model_config, StartupWeightLoadProfile.NATIVE_DENSE),
            (_make_qwen3_moe_model_config, StartupWeightLoadProfile.QWEN3_MOE_EP),
            (
                _make_qwen35_hybrid_vlm_model_config,
                StartupWeightLoadProfile.QWEN3_5_HYBRID_VLM,
            ),
            (
                _make_qwen35_moe_hybrid_vlm_model_config,
                StartupWeightLoadProfile.QWEN3_5_MOE_HYBRID_VLM,
            ),
            (_make_glm_moe_dsa_fp8_model_config, StartupWeightLoadProfile.GLM_MOE_DSA),
        )
        for make_config, profile in profiles:
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(profile=profile, dtype=dtype):
                    config = make_config(
                        dtype=dtype,
                        quantization=None,
                        is_multimodal=False,
                        is_generation=False,
                    )
                    config.hf_config.encoder_only = True
                    config.hf_config.language_only = True
                    config.hf_config.language_model_only = True
                    self.assertEqual(
                        self._create(model_config=config)._plan.profile, profile
                    )

    def test_graph_capture_backend_does_not_change_admission(self):
        for backend in (Backend.FULL, Backend.BREAKABLE, Backend.TC_PIECEWISE):
            with self.subTest(backend=backend):
                self._published_options()
                with get_exec().graph.override(enable_torch_compile=True):
                    get_exec().graph.cuda_graph_config.prefill.backend = backend
                    with patch(
                        f"{_STARTUP_MODULE}.current_platform.is_cuda", return_value=True
                    ):
                        options = StartupWeightLoadOptions.from_published_config()
                manager = self._create(
                    options=options, model_config=_make_qwen35_hybrid_vlm_model_config()
                )
                self.assertEqual(
                    manager._plan.profile, StartupWeightLoadProfile.QWEN3_5_HYBRID_VLM
                )

    def test_hybrid_attention_backends_do_not_restrict_admission(self):
        for make_config in (
            _make_qwen35_hybrid_vlm_model_config,
            _make_qwen35_moe_hybrid_vlm_model_config,
        ):
            for base, decode, prefill, verify, spec in (
                ("triton", None, None, None, None),
                ("cutedsl", None, None, None, None),
                ("flashinfer", None, None, None, None),
                ("triton", "flashinfer", None, "triton", None),
                ("triton", None, "flashinfer", "flashinfer", "EAGLE3"),
            ):
                with self.subTest(
                    model=make_config.__name__,
                    base=base,
                    decode=decode,
                    prefill=prefill,
                    verify=verify,
                    spec=spec,
                ):
                    options = self._published_options(
                        linear_attn_backend=base,
                        linear_attn_decode_backend=decode,
                        linear_attn_prefill_backend=prefill,
                        linear_attn_verify_backend=verify,
                        speculative_algorithm=spec,
                    )
                    self._create(options=options, model_config=make_config())

    def test_refreshed_speculative_and_dcp_copies_do_not_restrict_admission(self):
        cases = (
            {"dcp_size": 2, "dcp_replicate_q_proj": True},
            {"speculative_algorithm": "EAGLE", "speculative_token_map": "/dummy/map"},
            {"speculative_algorithm": "EAGLE3", "speculative_token_map": "/dummy/map"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                options = self._published_options(**changes)
                self._create(options=options)

    def test_lora_detection_includes_uno_before_lora_initialization(self):
        for changes in (
            {"enable_lora": True},
            {"lora_paths": ["adapter=/dummy/adapter"]},
            {"speculative_algorithm": "UNO"},
        ):
            with self.subTest(changes=changes):
                options = self._published_options(**changes)
                self.assertTrue(options.has_lora)
                self._create(options=options)
                with self.assertRaisesRegex(ValueError, "lora"):
                    self._create(
                        options=options, model_config=_make_qwen3_moe_model_config()
                    )

    def test_unquantized_moe_rejects_only_non_reloadable_weight_layouts(self):
        for make_config in (
            _make_qwen3_moe_model_config,
            _make_qwen35_moe_hybrid_vlm_model_config,
            lambda: _make_glm_moe_dsa_fp8_model_config(quantization=None),
        ):
            for runner in (
                "auto",
                "triton",
                "deep_gemm",
                "flashinfer_cutlass",
                "flashinfer_trtllm_routed",
                "flashinfer_trtllm",
                "experimental_sgl_trtllm",
            ):
                with self.subTest(model=make_config, runner=runner):
                    options = _make_options(
                        moe_runner_backend=runner, moe_a2a_backend="deepep"
                    )
                    self._create(model_config=make_config(), options=options)

    def test_glm_fp8_checks_layout_not_benchmark_configuration(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                config = _make_glm_moe_dsa_fp8_model_config(
                    dtype=dtype,
                    hf_config_overrides={
                        "cli_factor": 3,
                        "index_topk_pattern": "different",
                    },
                )
                manager = self._create(model_config=config)
                self.assertEqual(
                    manager._plan.profile, StartupWeightLoadProfile.GLM_MOE_DSA
                )
        for quantization_config, rejection in (
            (None, "quantization"),
            (
                {"quant_method": "modelopt", "weight_block_size": [128, 128]},
                "quantization",
            ),
            ({"quant_method": "fp8"}, "quantization"),
        ):
            with self.subTest(quantization_config=quantization_config):
                config = _make_glm_moe_dsa_fp8_model_config(
                    hf_config_overrides={"quantization_config": quantization_config}
                )
                with self.assertRaisesRegex(ValueError, rejection):
                    self._create(model_config=config)

    def test_profiles_share_serialized_block_fp8_loading_rules(self):
        factories = (
            _make_model_config,
            _make_qwen3_moe_model_config,
            _make_qwen35_hybrid_vlm_model_config,
            _make_qwen35_moe_hybrid_vlm_model_config,
            _make_glm_moe_dsa_fp8_model_config,
        )
        for make_config in factories:
            config = make_config(quantization="fp8")
            for quantization_config, rejected in (
                ({"quant_method": "fp8", "weight_block_size": [128, 128]}, False),
                (None, True),
                ({"weight_block_size": [128, 128]}, True),
                ({"quant_method": "fp8"}, True),
                ({"quant_method": "fp8", "weight_block_size": None}, True),
                ({"quant_method": "fp8", "weight_block_size": [128]}, True),
                ({"quant_method": "modelopt", "weight_block_size": [128, 128]}, True),
            ):
                with self.subTest(
                    model=make_config.__name__, quantization_config=quantization_config
                ):
                    config.hf_config.quantization_config = quantization_config
                    if rejected:
                        with self.assertRaisesRegex(ValueError, "quantization"):
                            self._create(model_config=config)
                    else:
                        self._create(model_config=config)

    def test_mla_and_dense_share_block_fp8_admission(self):
        for make_config in (
            _make_model_config,
            _make_qwen3_moe_model_config,
            _make_qwen35_hybrid_vlm_model_config,
            _make_qwen35_moe_hybrid_vlm_model_config,
            _make_glm_moe_dsa_fp8_model_config,
        ):
            with self.subTest(model=make_config.__name__):
                config = make_config(quantization="fp8")
                config.hf_config.quantization_config = {
                    "quant_method": "fp8",
                    "weight_block_size": [64, 128],
                }
                self._create(model_config=config)

    def test_fp8_linear_scale_conversion_guard_is_shared(self):
        for make_config in (
            _make_model_config,
            _make_qwen3_moe_model_config,
            _make_qwen35_hybrid_vlm_model_config,
            _make_qwen35_moe_hybrid_vlm_model_config,
            _make_glm_moe_dsa_fp8_model_config,
        ):
            config = make_config(quantization="fp8")
            config.hf_config.quantization_config = {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            for requantize, backend, dtype, rejected in (
                (True, "deep_gemm", torch.bfloat16, True),
                (False, "deep_gemm", torch.bfloat16, False),
                (True, "triton", torch.bfloat16, False),
                (True, "deep_gemm", torch.float16, False),
            ):
                with (
                    self.subTest(
                        model=make_config.__name__,
                        requantize=requantize,
                        backend=backend,
                        dtype=dtype,
                    ),
                    patch(
                        "sglang.srt.model_loader.utils.should_deepgemm_weight_requant_ue8m0",
                        return_value=requantize,
                    ),
                ):
                    config.dtype = dtype
                    options = _make_options(fp8_gemm_runner_backend=backend)
                    if rejected:
                        with self.assertRaisesRegex(ValueError, "fp8_gemm_backend"):
                            self._create(model_config=config, options=options)
                    else:
                        self._create(model_config=config, options=options)

    def test_moe_fp8_rejects_scale_conversion_and_packed_caches(self):
        cases = (
            (False, "deep_gemm", "triton", "none", None),
            (True, "deep_gemm", "triton", "none", "fp8_gemm_backend"),
            (True, "triton", "deep_gemm", "none", "moe_runner_backend"),
            (False, "triton", "hpc_ops", "none", None),
            (False, "triton", "triton", "megamoe", "moe_a2a_backend"),
            (False, "triton", "flashinfer_trtllm", "none", None),
        )
        for make_config in (
            _make_qwen3_moe_model_config,
            _make_qwen35_moe_hybrid_vlm_model_config,
            _make_glm_moe_dsa_fp8_model_config,
        ):
            config = make_config(quantization="fp8")
            config.hf_config.quantization_config = {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            }
            for requantize, linear, moe, a2a, rejection in cases:
                with (
                    self.subTest(
                        requantize=requantize, linear=linear, moe=moe, a2a=a2a
                    ),
                    patch(
                        "sglang.srt.model_loader.utils.should_deepgemm_weight_requant_ue8m0",
                        return_value=requantize,
                    ),
                ):
                    options = _make_options(
                        fp8_gemm_runner_backend=linear,
                        moe_runner_backend=moe,
                        moe_a2a_backend=a2a,
                    )
                    if rejection:
                        with self.assertRaisesRegex(ValueError, rejection):
                            self._create(
                                model_config=config,
                                options=options,
                            )
                    else:
                        self._create(
                            model_config=config,
                            options=options,
                        )

    def test_glm_fp8_auto_linear_checks_the_resolved_backend(self):
        from sglang.srt.layers.quantization.fp8_utils import (
            deepgemm_w8a8_block_fp8_linear_with_fallback,
        )

        for backend, rejected in (
            (deepgemm_w8a8_block_fp8_linear_with_fallback, True),
            (object(), False),
        ):
            with (
                self.subTest(rejected=rejected),
                patch(
                    "sglang.srt.model_loader.utils.should_deepgemm_weight_requant_ue8m0",
                    return_value=True,
                ),
                patch(
                    "sglang.srt.layers.quantization.fp8_utils._dispatch_auto_backend",
                    return_value=backend,
                ),
            ):
                options = _make_options(fp8_gemm_runner_backend="auto")
                if rejected:
                    with self.assertRaisesRegex(ValueError, "fp8_gemm_backend"):
                        self._create(
                            options=options,
                            model_config=_make_glm_moe_dsa_fp8_model_config(),
                        )
                else:
                    self._create(
                        options=options,
                        model_config=_make_glm_moe_dsa_fp8_model_config(),
                    )

    def test_modelopt_quantization_is_rejected_before_model_resolution(self):
        for make_config in (
            _make_model_config,
            _make_qwen35_hybrid_vlm_model_config,
            _make_qwen35_moe_hybrid_vlm_model_config,
            _make_qwen3_moe_model_config,
            _make_glm_moe_dsa_fp8_model_config,
        ):
            for quantization in ("modelopt_fp8", "modelopt_fp4", "modelopt_mixed"):
                with self.subTest(
                    model=make_config.__name__, quantization=quantization
                ):
                    with self.assertRaisesRegex(ValueError, "quantization"):
                        self._create(
                            model_config=make_config(quantization=quantization)
                        )

    def test_draft_workers_remain_serial_without_admission_or_allocation(self):
        for mode in ("auto", "overlap"):
            with (
                self.subTest(mode=mode),
                patch(
                    f"{_STARTUP_MODULE}.get_model",
                    return_value=SimpleNamespace(startup_weight_load_mode=mode),
                ),
                patch.object(
                    StartupWeightLoadOptions, "from_published_config"
                ) as snapshot,
                patch.object(
                    StartupWeightLoadManager, "__init__", return_value=None
                ) as allocate,
            ):
                manager = StartupWeightLoadManager.create_from_published_config(
                    loader=self.loader,
                    model_config=_make_model_config(),
                    load_config=self.load_config,
                    device_config=self.device_config,
                    is_draft_worker=True,
                )
                self.assertIsNone(manager)
                snapshot.assert_not_called()
                allocate.assert_not_called()

    def test_options_accept_current_server_args_schema(self):
        options = self._published_options(linear_attn_prefill_backend="flashinfer")
        self.assertEqual(options.moe_a2a_backend, "none")
        self.assertEqual(options.moe_runner_backend, "auto")
        self.assertFalse(options.has_lora)
        for mode, expected_overlap, expected_attempt in (
            ("serial", False, False),
            ("overlap", True, True),
            ("auto", False, True),
        ):
            with self.subTest(mode=mode):
                with get_context().override_server_args(
                    startup_weight_load_mode=mode
                ) as mode_args:
                    self.assertEqual(
                        mode_args.is_startup_weight_load_overlap, expected_overlap
                    )
                    self.assertEqual(
                        mode_args.should_attempt_startup_weight_load_overlap,
                        expected_attempt,
                    )

    def test_non_deferred_loading_paths_are_rejected_in_explicit_mode(self):
        cases = (
            (
                {"options": _make_options(device="cpu", is_cuda_platform=False)},
                "non_cuda",
            ),
            (
                {"options": _make_options(cuda_graph_enabled=False)},
                "cuda_graph_disabled",
            ),
            ({"loader": object()}, "loader"),
            ({"load_config": LoadConfig(load_format=LoadFormat.PT)}, "load_format"),
            ({"options": _make_options(offload_group_size=1)}, "layer_group_offload"),
            ({"options": _make_options(prefetch_num_threads=0)}, "prefetch_threads"),
            ({"resolved_model_class": _ExternalModel}, "model_implementation"),
        )
        for kwargs, rejection in cases:
            with self.subTest(rejection=rejection):
                with self.assertRaisesRegex(ValueError, rejection):
                    self._create(**kwargs)


class TestStartupWeightLoadManager(CustomTestCase):
    def _manager(self, loader, *, plan=None, fallback_to_serial=False):
        return StartupWeightLoadManager(
            loader=loader,
            model_config=_make_model_config(),
            device_config=DeviceConfig("cpu", 0),
            plan=_make_plan() if plan is None else plan,
            fallback_to_serial=fallback_to_serial,
        )

    def test_prepare_rejects_resolved_pt_before_capture_mutation(self):
        trace = []
        loader = _RecordingLoader(_TiedWeightModel(), trace)
        loader.use_safetensors = False
        manager = self._manager(loader)

        with self.assertRaisesRegex(ValueError, "requires safetensors checkpoints"):
            manager.prepare()

        self.assertEqual(trace, ["initialize", "resolve"])

    def test_auto_resolved_pt_fallback_reuses_initialized_model(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)
        loader.use_safetensors = False
        manager = self._manager(loader, fallback_to_serial=True)

        self.assertIs(manager.prepare(), model)

        self.assertEqual(trace, ["initialize", "resolve", "serial_load"])
        self.assertEqual(manager.state, StartupWeightLoadState.READY)
        self.assertFalse(manager.is_deferred)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3))
        self.assertIsNone(manager.finalize())
        self.assertIsNone(manager.finalize())
        self.assertEqual(trace, ["initialize", "resolve", "serial_load"])

    def test_multiple_safetensors_sources_share_the_overlap_lifecycle(self):
        trace = []
        loader = _RecordingLoader(_TiedWeightModel(), trace)
        loader.num_resolved_sources = 2
        manager = self._manager(loader, fallback_to_serial=True)

        manager.prepare()
        self.assertEqual(len(manager._resolved_sources), 2)
        manager.start_prefetch()
        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
        ):
            manager.finalize()

        self.assertEqual(trace[:3], ["initialize", "resolve", "prepare_capture"])
        self.assertNotIn("serial_load", trace)
        self.assertEqual(trace.count("commit"), 1)
        self.assertEqual(manager.state, StartupWeightLoadState.READY)

    def test_mixed_source_formats_fall_back_before_capture_mutation(self):
        for auto in (False, True):
            with self.subTest(auto=auto):
                trace = []
                model = _TiedWeightModel()
                loader = _RecordingLoader(model, trace)
                loader.num_resolved_sources = 2
                resolve = loader.resolve_model_weights

                def mixed_sources(model_config, model):
                    sources = resolve(model_config, model)
                    sources[1].use_safetensors = False
                    return sources

                loader.resolve_model_weights = mixed_sources
                manager = self._manager(loader, fallback_to_serial=auto)
                if auto:
                    self.assertIs(manager.prepare(), model)
                    self.assertEqual(trace, ["initialize", "resolve", "serial_load"])
                else:
                    with self.assertRaisesRegex(ValueError, "requires safetensors"):
                        manager.prepare()
                    self.assertEqual(trace, ["initialize", "resolve"])

    def test_auto_supported_safetensors_source_overlaps_regardless_of_path(self):
        for hf_folder in ("/local/model", "/remote/model"):
            with self.subTest(hf_folder=hf_folder):
                trace = []
                loader = _RecordingLoader(_TiedWeightModel(), trace)
                loader.hf_folder = hf_folder
                manager = self._manager(loader, fallback_to_serial=True)

                manager.prepare()

                self.assertEqual(
                    trace,
                    ["initialize", "resolve", "prepare_capture"],
                )
                self.assertEqual(manager.state, StartupWeightLoadState.CAPTURE_READY)
                self.assertTrue(manager.is_deferred)

    def test_plan_prefetch_thread_count_reaches_the_loader(self):
        trace = []
        loader = _RecordingLoader(_TiedWeightModel(), trace)
        manager = self._manager(
            loader,
            plan=_make_plan(prefetch_num_threads=7),
        )

        manager.prepare()
        manager.start_prefetch()

        self.assertEqual(loader.prefetch_num_threads, 7)

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
                "stop_prefetch",
                "cancel_prefetch",
                "wait_prefetch",
                "commit",
            ],
        )

        # Finalization is idempotent after a successful commit.
        self.assertIs(manager.finalize(), manager._timings)
        self.assertEqual(trace.count("commit"), 1)
        self.assertIs(model.weight, model.tied_weight)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3))
        self.assertTrue(log_info.call_args.args[0].startswith("Load weight end."))
        self.assertFalse(manager._loader.startup_prefetch_active)
        self.assertEqual(
            parallel_state_patch.call_args_list,
            [call(), call(reverse=True)],
        )

    def test_finalize_returns_complete_phase_timings(self):
        trace = []
        model = _TiedWeightModel()
        with patch(
            f"{_STARTUP_MODULE}.time.perf_counter",
            side_effect=(0.0, 2.0, 3.0, 7.0, 8.0, 11.0, 13.0),
        ):
            manager = self._manager(_RecordingLoader(model, trace))
            manager.prepare()
            manager.start_prefetch()
            with (
                patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
                patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            ):
                timings = manager.finalize()

        self.assertEqual(
            timings,
            StartupWeightLoadTimings(
                prepare_seconds=2.0,
                prefetch_start_delay_seconds=1.0,
                prefetch_window_seconds=4.0,
                commit_seconds=3.0,
                prefetch_cleanup_seconds=3.0,
                total_seconds=13.0,
            ),
        )

        self.assertEqual(timings.weight_load_seconds, 8.0)
        self.assertEqual(
            timings.total_seconds,
            timings.prepare_seconds
            + timings.prefetch_start_delay_seconds
            + timings.prefetch_window_seconds
            + timings.commit_seconds
            + timings.prefetch_cleanup_seconds,
        )
        self.assertIs(manager.finalize(), timings)

    def test_weight_copies_refresh_before_synchronization_and_validation(self):
        trace = []
        model = _TiedWeightModel()
        model.register_buffer("weight_copy", model.weight.detach().clone())
        manifest = ModelStorageManifest.capture(model)
        manager = self._manager(_RecordingLoader(model, trace))
        manager.prepare()
        manager.start_prefetch()

        def refresh():
            trace.append("refresh")
            model.weight_copy.copy_(model.weight)

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(
                f"{_STARTUP_MODULE}.torch.cuda.synchronize",
                side_effect=lambda: trace.append("synchronize"),
            ),
        ):
            manager.finalize(after_weight_load=refresh)
            manager.finalize(after_weight_load=refresh)

        self.assertEqual(trace[-3:], ["commit", "refresh", "synchronize"])
        self.assertEqual(manifest.changed_names(model), ())
        torch.testing.assert_close(model.weight_copy, torch.full_like(model.weight, 3))

    def test_refresh_cannot_replace_graph_visible_storage(self):
        model = _TiedWeightModel()
        model.register_buffer("weight_copy", model.weight.detach().clone())
        manager = self._manager(_RecordingLoader(model, []))
        manager.prepare()
        manager.start_prefetch()
        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            self.assertRaisesRegex(RuntimeError, "buffer:weight_copy"),
        ):
            manager.finalize(
                after_weight_load=lambda: setattr(
                    model, "weight_copy", model.weight.detach().clone()
                )
            )
        self.assertEqual(manager.state, StartupWeightLoadState.COMMITTING)

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
                "changed graph-visible storage or constants: parameter:tied_weight",
            ),
        ):
            manager.finalize()

    def test_finalize_rejects_derived_tensor_storage_rebind(self):
        trace = []
        model = _DerivedTensorModel()
        loader = _RecordingLoader(model, trace)
        original_commit = loader.commit_model_weights

        def rebind_derived_tensor(**kwargs):
            original_commit(**kwargs)
            model.graph_weight = model.graph_weight.clone()

        loader.commit_model_weights = rebind_derived_tensor
        manager = self._manager(loader)
        manager.prepare()
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            self.assertRaisesRegex(
                RuntimeError,
                "changed graph-visible storage or constants: derived:graph_weight",
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
                "did not replace capture-safe dummy values: parameter:tied_weight",
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
        self.assertLess(trace.index("stop_prefetch"), trace.index("commit"))
        self.assertLess(trace.index("cancel_prefetch"), trace.index("commit"))

    def test_failed_completed_prefetch_does_not_disable_multithread_loading(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)
        loader.prefetch_handle.done = True
        loader.prefetch_handle.errors = (("bad.safetensors", OSError("failed")),)
        manager = self._manager(loader)
        manager.prepare()
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            patch(f"{_STARTUP_MODULE}.logger.warning"),
        ):
            manager.finalize()

        self.assertFalse(loader.startup_prefetch_active)
        self.assertLess(trace.index("wait_prefetch"), trace.index("commit"))
        self.assertNotIn("cancel_prefetch", trace)
        self.assertIn("wait_prefetch", trace)
        self.assertNotIn("stop_prefetch", trace)

    def test_terminated_prefetch_worker_without_errors_falls_back(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)
        loader.prefetch_handle.done = True
        loader.prefetch_handle.succeeded = False
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
        self.assertIn("terminated before completion", warning.call_args.args[1])
        self.assertIn("falling back", warning.call_args.args[2])

    def test_prefetch_failure_during_commit_after_stop_timeout_is_reported(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)
        original_commit = loader.commit_model_weights

        def stop_times_out(timeout=None):
            trace.append("stop_prefetch")
            loader.prefetch_handle.cancel()
            raise TimeoutError("Timed out waiting for checkpoint prefetching")

        def fail_prefetch_during_commit(**kwargs):
            original_commit(**kwargs)
            loader.prefetch_handle.errors = (("late.safetensors", OSError("failed")),)

        loader.prefetch_handle.stop = stop_times_out
        loader.commit_model_weights = fail_prefetch_during_commit
        manager = self._manager(loader)
        manager.prepare()
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
            patch(f"{_STARTUP_MODULE}.logger.warning") as warning,
        ):
            manager.finalize()

        self.assertIn(
            "completed despite",
            warning.call_args_list[-1].args[2],
        )
        self.assertLess(trace.index("stop_prefetch"), trace.index("commit"))
        self.assertEqual(trace.count("stop_prefetch"), 1)

    def test_stop_timeout_before_commit_falls_back_without_second_wait(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)

        def stop_times_out(timeout=None):
            trace.append("stop_prefetch")
            loader.prefetch_handle.cancel()
            raise TimeoutError("Timed out waiting for checkpoint prefetching")

        loader.prefetch_handle.stop = stop_times_out
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
        self.assertTrue(loader.startup_prefetch_active)
        self.assertEqual(trace.count("stop_prefetch"), 1)
        self.assertLess(trace.index("stop_prefetch"), trace.index("commit"))
        self.assertEqual(warning.call_count, 2)
        self.assertIn("did not stop before", warning.call_args_list[0].args[0])
        self.assertIn("still exiting after", warning.call_args_list[1].args[0])

    def test_worker_finishing_at_stop_timeout_restores_normal_loader(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)

        def finish_at_timeout(timeout=None):
            trace.append("stop_prefetch")
            loader.prefetch_handle.cancel()
            loader.prefetch_handle.done = True
            raise TimeoutError("Timed out waiting for checkpoint prefetching")

        loader.prefetch_handle.stop = finish_at_timeout
        manager = self._manager(loader)
        manager.prepare()
        manager.start_prefetch()

        with (
            patch(f"{_STARTUP_MODULE}.monkey_patch_vllm_parallel_state"),
            patch(f"{_STARTUP_MODULE}.torch.cuda.synchronize"),
        ):
            manager.finalize()

        self.assertFalse(loader.startup_prefetch_active)
        self.assertLess(trace.index("wait_prefetch"), trace.index("commit"))
        self.assertEqual(trace.count("wait_prefetch"), 1)

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


class _PolicyLoader:
    def __init__(self, trace, serial_model):
        self._trace = trace
        self._serial_model = serial_model

    def load_model(self, *, model_config, device_config):
        self._trace.append("serial_load")
        return self._serial_model


class _PolicyManager:
    def __init__(self, trace, model, *, is_deferred):
        self._trace = trace
        self._model = model
        self.is_deferred = is_deferred

    def prepare(self):
        self._trace.append("prepare")
        return self._model


class TestStartupWeightLoadPolicyRouting(CustomTestCase):
    @staticmethod
    def _load(*, manager):
        trace = []
        serial_model = nn.Linear(1, 1)
        loader = _PolicyLoader(trace, serial_model)
        server_args = SimpleNamespace(
            enable_weights_cpu_backup=False,
            enable_draft_weights_cpu_backup=False,
            weight_cache_mode="off",
            should_attempt_startup_weight_load_overlap=True,
            startup_weight_load_mode="auto",
        )
        memory_saver_adapter = SimpleNamespace(
            region=lambda *args, **kwargs: nullcontext()
        )
        with (
            get_context().override_server_args(
                enable_weights_cpu_backup=False,
                enable_draft_weights_cpu_backup=False,
                weight_cache_mode="off",
            ),
            patch(f"{_LOAD_MODEL_UTILS_MODULE}.get_model_loader", return_value=loader),
            patch(f"{_LOAD_MODEL_UTILS_MODULE}.monkey_patch_vllm_parallel_state"),
            patch.object(
                StartupWeightLoadManager,
                "create_from_published_config",
                return_value=manager,
            ),
        ):
            loaded = load_model_with_memory_saver(
                server_args=server_args,
                model_config=SimpleNamespace(),
                load_config=LoadConfig(load_format=LoadFormat.SAFETENSORS),
                device="cuda",
                gpu_id=0,
                memory_saver_adapter=memory_saver_adapter,
                is_draft_worker=False,
            )
        return trace, serial_model, loaded

    def test_config_rejection_uses_the_original_serial_loader_once(self):
        trace, serial_model, loaded = self._load(manager=None)

        self.assertEqual(trace, ["serial_load"])
        self.assertIs(loaded.model, serial_model)
        self.assertIsNone(loaded.startup_weight_load)

    def test_source_rejection_does_not_allocate_a_second_model(self):
        manager_trace = []
        fallback_model = nn.Linear(1, 1)
        manager = _PolicyManager(
            manager_trace,
            fallback_model,
            is_deferred=False,
        )

        trace, _, loaded = self._load(manager=manager)

        self.assertEqual(manager_trace, ["prepare"])
        self.assertEqual(trace, [])
        self.assertIs(loaded.model, fallback_model)
        self.assertIsNone(loaded.startup_weight_load)

    def test_admitted_auto_mode_retains_the_manager(self):
        manager_trace = []
        deferred_model = nn.Linear(1, 1)
        manager = _PolicyManager(
            manager_trace,
            deferred_model,
            is_deferred=True,
        )

        trace, _, loaded = self._load(manager=manager)

        self.assertEqual(manager_trace, ["prepare"])
        self.assertEqual(trace, [])
        self.assertIs(loaded.model, deferred_model)
        self.assertIs(loaded.startup_weight_load, manager)


class TestModelStorageManifest(CustomTestCase):
    def test_capture_postprocess_stages_each_quantized_module(self):
        from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

        model = nn.Module()
        model.attn = nn.Module()
        method = BaseKVCacheMethod(quant_config=None)
        method.create_weights(model.attn)
        model.attn.quant_method = method
        loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.SAFETENSORS))
        target_device = torch.device("cpu")
        trace = []

        @contextmanager
        def stage(module, device):
            self.assertIs(module, model.attn)
            self.assertEqual(device, target_device)
            trace.append("stage")
            try:
                yield module
            finally:
                trace.append("restore")

        original = method.process_weights_after_loading

        def process(module):
            self.assertEqual(trace, ["stage"])
            trace.append("process")
            original(module)

        with (
            patch("sglang.srt.model_loader.loader.device_loading_context", stage),
            patch.object(method, "process_weights_after_loading", side_effect=process),
        ):
            loader.prepare_model_for_capture(
                model=model,
                model_config=SimpleNamespace(dtype=torch.float32),
                target_device=target_device,
            )
        self.assertEqual(trace, ["stage", "process", "restore"])

    def test_kv_cache_post_load_detects_changed_capture_constants(self):
        from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
        from sglang.srt.layers.radix_attention import RadixAttention

        for checkpoint_scale in (None, 0.5):
            with self.subTest(checkpoint_scale=checkpoint_scale):
                model = nn.Module()
                model.attn = RadixAttention.__new__(RadixAttention)
                nn.Module.__init__(model.attn)
                method = BaseKVCacheMethod(quant_config=None)
                method.create_weights(model.attn)
                optional_values = initialize_capture_safe_weights(model)
                method.process_weights_after_loading(model.attn)
                manifest = ModelStorageManifest.capture(model)
                pointers = (
                    model.attn.k_scale.data_ptr(),
                    model.attn.v_scale.data_ptr(),
                )

                restore_optional_checkpoint_parameter_values(optional_values)
                if checkpoint_scale is not None:
                    model.attn.k_scale.copy_(checkpoint_scale)
                method.process_weights_after_loading(model.attn)

                self.assertEqual(
                    pointers,
                    (model.attn.k_scale.data_ptr(), model.attn.v_scale.data_ptr()),
                )
                self.assertEqual(
                    manifest.changed_names(model),
                    ()
                    if checkpoint_scale is None
                    else (
                        "constant:attn.k_scale_float",
                        "constant:attn.v_scale_float",
                    ),
                )

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

    def test_explicit_derived_tensor_is_checked_for_storage_stability(self):
        model = _DerivedTensorModel()
        manifest = ModelStorageManifest.capture(model)

        model.graph_weight.copy_(torch.full_like(model.graph_weight, 2))
        self.assertEqual(manifest.changed_names(model), ())

        model.graph_weight = model.graph_weight.clone()
        self.assertEqual(
            manifest.changed_names(model),
            ("derived:graph_weight",),
        )

    def test_duplicate_derived_tensor_names_are_rejected(self):
        model = _DerivedTensorModel()
        model.named_startup_weight_load_derived_tensors = lambda: (
            ("graph_weight", model.graph_weight),
            ("graph_weight", model.graph_weight),
        )

        with self.assertRaisesRegex(
            ValueError, "Duplicate startup weight-load tensor name"
        ):
            ModelStorageManifest.capture(model)

    def test_nested_derived_tensor_names_include_the_module_path(self):
        model = _NestedDerivedTensorModel()
        manifest = ModelStorageManifest.capture(model)

        self.assertIn("derived:child.graph_weight", dict(manifest.tensors))

    def test_invalid_derived_tensor_entries_are_rejected(self):
        cases = (
            ("empty_name", "", torch.ones(1), ValueError, "non-empty strings"),
            ("non_string_name", 1, torch.ones(1), ValueError, "non-empty strings"),
            ("non_tensor", "graph_weight", object(), TypeError, "not a torch.Tensor"),
        )
        for case, name, value, error_type, message in cases:
            with self.subTest(case=case):
                model = _DerivedTensorModel()
                model.named_startup_weight_load_derived_tensors = (
                    lambda name=name, value=value: ((name, value),)
                )

                with self.assertRaisesRegex(error_type, message):
                    ModelStorageManifest.capture(model)

    def test_deepseek_mla_exposes_graph_visible_post_load_operands(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        w_kc = torch.ones(2, 3)
        w_vc = torch.ones(3, 2)
        w_scale = torch.ones(1)
        w_scale_k = torch.ones(1)
        w_scale_v = torch.ones(1)
        attention = SimpleNamespace(
            w_kc=w_kc,
            w_vc=w_vc,
            w_scale=w_scale,
            w_scale_k=w_scale_k,
            w_scale_v=w_scale_v,
        )

        derived = dict(
            DeepseekV2AttentionMLA.named_startup_weight_load_derived_tensors(attention)
        )
        self.assertEqual(
            tuple(derived),
            ("w_kc", "w_vc", "w_scale", "w_scale_k", "w_scale_v"),
        )
        self.assertIs(derived["w_kc"], w_kc)
        self.assertIs(derived["w_vc"], w_vc)
        self.assertIs(derived["w_scale"], w_scale)
        self.assertIs(derived["w_scale_k"], w_scale_k)
        self.assertIs(derived["w_scale_v"], w_scale_v)


class TestCaptureSafeWeightInitialization(CustomTestCase):
    def test_only_parameters_are_filled(self):
        model = _TiedWeightModel()

        initialize_capture_safe_weights(model, value=0.125)

        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 0.125))
        torch.testing.assert_close(model.scale, torch.ones_like(model.scale))

    def test_optional_checkpoint_parameters_keep_post_load_defaults(self):
        from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

        layer = nn.Module()
        method = BaseKVCacheMethod(quant_config=None)
        method.create_weights(layer)

        optional_values = initialize_capture_safe_weights(layer, value=0.125)

        self.assertEqual(layer.k_scale.item(), -1.0)
        self.assertEqual(layer.v_scale.item(), -1.0)
        method.process_weights_after_loading(layer)
        self.assertEqual(layer.k_scale.item(), 1.0)
        self.assertEqual(layer.v_scale.item(), 1.0)

        # Restore the constructor values before the real checkpoint commit so
        # optional-weight post-load logic observes the same inputs as serial.
        k_scale_ptr = layer.k_scale.data_ptr()
        v_scale_ptr = layer.v_scale.data_ptr()
        restore_optional_checkpoint_parameter_values(optional_values)
        method.process_weights_after_loading(layer)
        self.assertEqual(layer.k_scale.data_ptr(), k_scale_ptr)
        self.assertEqual(layer.v_scale.data_ptr(), v_scale_ptr)
        self.assertEqual(layer.k_scale.item(), 1.0)
        self.assertEqual(layer.v_scale.item(), 1.0)

    def test_single_checkpoint_kv_scale_keeps_serial_post_load_semantics(self):
        from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

        layer = nn.Module()
        method = BaseKVCacheMethod(quant_config=None)
        method.create_weights(layer)
        k_scale_ptr = layer.k_scale.data_ptr()
        v_scale_ptr = layer.v_scale.data_ptr()
        optional_values = initialize_capture_safe_weights(layer)
        method.process_weights_after_loading(layer)

        restore_optional_checkpoint_parameter_values(optional_values)
        layer.k_scale.copy_(0.5)
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.k_scale.item(), 0.5)
        self.assertEqual(layer.v_scale.item(), 0.5)
        self.assertEqual(layer.k_scale.data_ptr(), k_scale_ptr)
        self.assertEqual(layer.v_scale.data_ptr(), v_scale_ptr)

    def test_separate_checkpoint_kv_scales_survive_capture_post_load(self):
        from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

        layer = nn.Module()
        method = BaseKVCacheMethod(quant_config=None)
        method.create_weights(layer)
        optional_values = initialize_capture_safe_weights(layer)
        method.process_weights_after_loading(layer)

        restore_optional_checkpoint_parameter_values(optional_values)
        layer.k_scale.copy_(0.5)
        layer.v_scale.copy_(0.25)
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.k_scale.item(), 0.5)
        self.assertEqual(layer.v_scale.item(), 0.25)

    def test_default_loader_restores_optional_values_before_real_load(self):
        from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

        model = nn.Module()
        model.attn = nn.Module()
        method = BaseKVCacheMethod(quant_config=None)
        method.create_weights(model.attn)
        model.attn.quant_method = method
        model_config = SimpleNamespace(dtype=torch.float32)
        loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.SAFETENSORS))

        loader.prepare_model_for_capture(
            model=model,
            model_config=model_config,
            target_device=torch.device("cpu"),
        )
        k_scale_ptr = model.attn.k_scale.data_ptr()
        v_scale_ptr = model.attn.v_scale.data_ptr()
        self.assertEqual(model.attn.k_scale.item(), 1.0)
        self.assertEqual(model.attn.v_scale.item(), 1.0)

        def load_real_scale(model, weights, target_device):
            self.assertEqual(model.attn.k_scale.item(), -1.0)
            self.assertEqual(model.attn.v_scale.item(), -1.0)
            model.attn.k_scale.copy_(0.5)
            method.process_weights_after_loading(model.attn)

        with patch.object(
            loader,
            "load_weights_and_postprocess",
            side_effect=load_real_scale,
        ):
            loader.commit_model_weights(
                model=model,
                model_config=model_config,
                resolved_sources=(),
                target_device=torch.device("cpu"),
                startup_prefetch_active=False,
            )

        self.assertEqual(model.attn.k_scale.data_ptr(), k_scale_ptr)
        self.assertEqual(model.attn.v_scale.data_ptr(), v_scale_ptr)
        self.assertEqual(model.attn.k_scale.item(), 0.5)
        self.assertEqual(model.attn.v_scale.item(), 0.5)
        self.assertEqual(model.attn.k_scale_float, 0.5)
        self.assertEqual(model.attn.v_scale_float, 0.5)
        self.assertIsNone(loader._startup_optional_parameter_values)

    def test_manifest_sentinel_check_omits_optional_parameters(self):
        model = _TiedWeightModel()
        model.optional_weight = nn.Parameter(torch.full((1,), 0.125))
        model.optional_weight._skip_weight_check = True
        with torch.no_grad():
            model.weight.fill_(0.125)
        manifest = ModelStorageManifest.capture(model)

        self.assertEqual(
            manifest.unchanged_parameter_names(0.125),
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

    def finalize(self, *, after_weight_load=None):
        self._trace.append("finalize")
        if after_weight_load is not None:
            after_weight_load()
        return StartupWeightLoadTimings(
            prepare_seconds=1.0,
            prefetch_start_delay_seconds=0.0,
            prefetch_window_seconds=2.0,
            commit_seconds=3.0,
            prefetch_cleanup_seconds=4.0,
            total_seconds=10.0,
        )


class TestModelRunnerStartupWeightLoadOwnership(CustomTestCase):
    def test_shared_attention_roots_refresh_once(self):
        trace = []
        shared = SimpleNamespace(on_after_weight_load=lambda: trace.append("shared"))
        extra = SimpleNamespace(on_after_weight_load=lambda: trace.append("extra"))
        refresh_attention_weight_copies(shared, shared, [shared, None, extra])
        self.assertEqual(trace, ["shared", "extra"])

    def test_retained_serial_fallback_does_not_overwrite_timing_or_repeat_barrier(self):
        trace = []
        loader = _RecordingLoader(_TiedWeightModel(), trace)
        loader.use_safetensors = False
        manager = StartupWeightLoadManager(
            loader=loader,
            model_config=_make_model_config(),
            device_config=DeviceConfig("cpu", 0),
            plan=_make_plan(),
            fallback_to_serial=True,
        )
        manager.prepare()
        runner = self._runner(manager)
        runner.weight_load_time = 11.0
        with patch(
            "sglang.srt.model_executor.model_runner.dist_barrier_after_load"
        ) as barrier:
            runner.finalize_startup_weight_load()
        barrier.assert_not_called()
        self.assertEqual(runner.weight_load_time, 11.0)
        self.assertIsNone(runner.startup_weight_load)
        self.assertEqual(trace, ["initialize", "resolve", "serial_load"])

    @staticmethod
    def _runner(manager):
        runner = ModelRunner.__new__(ModelRunner)
        runner.startup_weight_load = manager
        runner.server_args = SimpleNamespace(
            elastic_ep_backend=None,
            is_ep_joiner=False,
        )
        runner.ps = SimpleNamespace(tp_rank=0)
        runner.weight_load_time = 0.0
        runner.attn_backend = None
        runner.decode_attn_backend = None
        runner.decode_attn_backend_group = []
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
            self.assertEqual(runner.weight_load_time, 8.0)
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
        self.assertEqual(runner.weight_load_time, 8.0)
        self.assertIsNone(runner.startup_weight_load)


class _SchedulerWorker:
    def __init__(self, trace, *, startup_weight_load_active, post_capture_active=False):
        self._trace = trace
        self._startup_weight_load_active = startup_weight_load_active
        self.model_runner = SimpleNamespace(
            device="cuda",
            forward_stream=object(),
            prewarm_sampling=lambda: trace.append("prewarm"),
            token_to_kv_pool=SimpleNamespace(post_capture_active=post_capture_active),
            post_capture_resize_kv_pool=lambda: trace.append("resize"),
            post_capture_elastic_ep_recover=lambda: trace.append("recover"),
        )

    def start_startup_weight_load(self):
        self._trace.append("start")

    def finalize_startup_weight_load(self):
        self._trace.append("finalize")

    def has_startup_weight_load(self):
        return self._startup_weight_load_active


class TestStartupWeightLoadSchedulerRouting(CustomTestCase):
    @staticmethod
    def _scheduler(worker, trace, *, mode, draft_worker=None):
        from sglang.srt.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = SimpleNamespace(startup_weight_load_mode=mode)
        scheduler.init_tp_model_worker = lambda: setattr(scheduler, "tp_worker", worker)
        scheduler.maybe_init_draft_worker = lambda: setattr(
            scheduler, "draft_worker", draft_worker
        )
        scheduler.init_memory_pools = lambda: trace.append("memory_pool")
        scheduler.init_all_attention_backends = lambda: trace.append("attention")
        scheduler.init_all_cuda_graphs = lambda: trace.append("capture")
        return scheduler

    def _run_startup(
        self, mode, *, use_draft_worker=False, recovering=False, commit_fails=False
    ):
        trace = []
        worker = _SchedulerWorker(
            trace,
            startup_weight_load_active=mode in ("overlap", "auto_overlap"),
            post_capture_active=True,
        )
        if commit_fails:

            def fail_commit():
                trace.append("finalize")
                raise RuntimeError("weight commit failed")

            worker.finalize_startup_weight_load = fail_commit
        draft_worker = (
            SimpleNamespace(
                prewarm_sampling=lambda: trace.append("draft_prewarm"),
                refresh_startup_weight_load=lambda: trace.append("draft_refresh"),
            )
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
                        elastic_ep_backend="mooncake" if recovering else None,
                        ep_join_mode="recover" if recovering else None,
                    )
                ),
            ),
            patch(
                "sglang.srt.managers.scheduler.torch.get_device_module",
                return_value=SimpleNamespace(
                    stream=stream_context,
                    synchronize=lambda: trace.append("synchronize"),
                ),
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "weight commit failed" if commit_fails else "stop after startup",
            ),
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

    def test_mapped_draft_weights_refresh_only_after_real_target_commit(self):
        self.assertEqual(
            self._run_startup("overlap", use_draft_worker=True)[-3:],
            ["finalize", "draft_refresh", "synchronize"],
        )

    def test_elastic_recovery_rejoins_after_real_weight_commit(self):
        self.assertEqual(
            self._run_startup("overlap", recovering=True)[-2:],
            ["finalize", "recover"],
        )
        self.assertEqual(self._run_startup("serial", recovering=True)[-1], "recover")

    def test_failed_commit_never_refreshes_draft_or_rejoins(self):
        trace = self._run_startup(
            "overlap", use_draft_worker=True, recovering=True, commit_fails=True
        )
        self.assertEqual(trace[-1], "finalize")
        self.assertNotIn("recover", trace)
        self.assertNotIn("draft_refresh", trace)

    def test_auto_routes_from_the_actual_admission_result(self):
        self.assertEqual(
            self._run_startup("auto"),
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
        self.assertEqual(
            self._run_startup("auto_overlap"),
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


if __name__ == "__main__":
    unittest.main()
