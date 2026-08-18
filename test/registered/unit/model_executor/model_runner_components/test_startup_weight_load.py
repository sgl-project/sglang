"""Unit tests for the post-capture startup weight-loading component."""

import dataclasses
import re
import unittest
from contextlib import nullcontext
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
    ModelStorageManifest,
    StartupWeightLoadManager,
    StartupWeightLoadOptions,
    StartupWeightLoadPlan,
    StartupWeightLoadProfile,
    StartupWeightLoadState,
    StartupWeightLoadTimings,
    evaluate_startup_weight_load_admission,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.weight_utils import initialize_capture_safe_weights
from sglang.srt.runtime_context import get_context, publish, reset_context
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
        moe_dp_size=1,
        moe_a2a_backend="none",
        moe_runner_backend="triton",
        enable_dp_attention=False,
        enable_two_batch_overlap=False,
        enable_eplb=False,
        ep_num_redundant_experts=0,
        init_expert_location="trivial",
        elastic_ep_backend=None,
        enable_elastic_expert_backup=False,
        ep_join_mode=None,
        max_ep_size=None,
        linear_attn_backend="triton",
        linear_attn_decode_backend=None,
        linear_attn_prefill_backend=None,
        cpu_offload_gb=0,
        offload_group_size=-1,
        enable_memory_saver=False,
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
        modelopt_quant=None,
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
        self.prefetch_handle = _RecordingPrefetchHandle(trace)

    def initialize_model_for_startup(self, *, model_config, device_config):
        self._trace.append("initialize")
        return self._model

    def resolve_model_weights(self, model_config, model):
        self._trace.append("resolve")
        return tuple(
            SimpleNamespace(use_safetensors=self.use_safetensors, source=object())
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
        self.assertIsInstance(
            self._create(options=_make_options(tp_size=2)),
            StartupWeightLoadManager,
        )

    def test_auto_mode_falls_back_for_config_rejection(self):
        server_args = SimpleNamespace(startup_weight_load_mode="auto")
        with (
            patch.object(
                StartupWeightLoadOptions,
                "from_server_args",
                return_value=_make_options(device="cpu", is_cuda_platform=False),
            ),
            patch(f"{_STARTUP_MODULE}.logger.info") as log_info,
        ):
            manager = StartupWeightLoadManager.create_from_server_args(
                loader=self.loader,
                model_config=_make_model_config(),
                load_config=self.load_config,
                device_config=self.device_config,
                server_args=server_args,
                is_draft_worker=False,
            )

        self.assertIsNone(manager)
        self.assertIn("non_cuda: CUDA only", log_info.call_args.args[1])

    def test_auto_mode_creates_a_manager_when_admitted(self):
        server_args = SimpleNamespace(startup_weight_load_mode="auto")
        with (
            patch.object(
                StartupWeightLoadOptions,
                "from_server_args",
                return_value=_make_options(),
            ),
            patch(
                f"{_STARTUP_MODULE}.get_model_architecture",
                return_value=(_CanonicalModel, "LlamaForCausalLM"),
            ),
            patch(
                f"{_STARTUP_MODULE}._get_canonical_model_class",
                return_value=_CanonicalModel,
            ),
        ):
            manager = StartupWeightLoadManager.create_from_server_args(
                loader=self.loader,
                model_config=_make_model_config(),
                load_config=self.load_config,
                device_config=self.device_config,
                server_args=server_args,
                is_draft_worker=False,
            )

        self.assertIsInstance(manager, StartupWeightLoadManager)
        self.assertTrue(manager._fallback_to_serial)

    def test_qwen35_family_hybrid_vlm_profile_is_admitted(self):
        for tp_size in (2, 4):
            for backend in (Backend.BREAKABLE, Backend.DISABLED):
                with self.subTest(
                    tp_size=tp_size,
                    prefill_cuda_graph_backend=backend,
                ):
                    manager = self._create(
                        options=_make_options(
                            tp_size=tp_size,
                            prefill_cuda_graph_backend=backend,
                        ),
                        model_config=_make_qwen35_hybrid_vlm_model_config(),
                    )

                    self.assertEqual(
                        manager._plan.profile,
                        StartupWeightLoadProfile.QWEN3_5_HYBRID_VLM,
                    )

    def test_qwen35_family_hybrid_vlm_profile_rejects_near_misses(self):
        cases = (
            (
                "tp1",
                _make_options(tp_size=1),
                _make_qwen35_hybrid_vlm_model_config(),
                "requires TP2 or TP4",
            ),
            (
                "fp16",
                _make_options(tp_size=2),
                _make_qwen35_hybrid_vlm_model_config(dtype=torch.float16),
                "requires BF16",
            ),
            (
                "quantized",
                _make_options(tp_size=2),
                _make_qwen35_hybrid_vlm_model_config(quantization="fp8"),
                "quantization is not supported",
            ),
            (
                "modelopt",
                _make_options(tp_size=2),
                _make_qwen35_hybrid_vlm_model_config(modelopt_quant="fp4"),
                "ModelOpt is not supported",
            ),
            (
                "expert_parallel",
                _make_options(tp_size=2, ep_size=2),
                _make_qwen35_hybrid_vlm_model_config(),
                "expert parallelism is not supported",
            ),
            (
                "multimodal_disabled",
                _make_options(tp_size=2),
                _make_qwen35_hybrid_vlm_model_config(is_multimodal=False),
                "requires multimodal execution",
            ),
            (
                "encoder_only",
                _make_options(tp_size=2),
                _make_qwen35_hybrid_vlm_model_config(
                    hf_config=SimpleNamespace(
                        architectures=["Qwen3_5ForConditionalGeneration"],
                        encoder_only=True,
                        language_only=False,
                        language_model_only=False,
                    )
                ),
                "encoder-only execution is not supported",
            ),
            (
                "language_only",
                _make_options(tp_size=2),
                _make_qwen35_hybrid_vlm_model_config(
                    hf_config=SimpleNamespace(
                        architectures=["Qwen3_5ForConditionalGeneration"],
                        encoder_only=False,
                        language_only=True,
                        language_model_only=False,
                    )
                ),
                "language-only encoder disaggregation is not supported",
            ),
            (
                "language_model_only",
                _make_options(tp_size=2),
                _make_qwen35_hybrid_vlm_model_config(
                    hf_config=SimpleNamespace(
                        architectures=["Qwen3_5ForConditionalGeneration"],
                        encoder_only=False,
                        language_only=False,
                        language_model_only=True,
                    )
                ),
                "language-model-only execution is not supported",
            ),
            (
                "flashinfer_linear_attention_base",
                _make_options(
                    tp_size=2,
                    linear_attn_backend="flashinfer",
                ),
                _make_qwen35_hybrid_vlm_model_config(),
                "requires Triton linear attention",
            ),
            (
                "flashinfer_linear_attention_decode",
                _make_options(
                    tp_size=2,
                    linear_attn_decode_backend="flashinfer",
                ),
                _make_qwen35_hybrid_vlm_model_config(),
                "requires Triton linear attention",
            ),
            (
                "flashinfer_linear_attention_prefill",
                _make_options(
                    tp_size=2,
                    linear_attn_prefill_backend="flashinfer",
                ),
                _make_qwen35_hybrid_vlm_model_config(),
                "requires Triton linear attention",
            ),
            (
                "full_prefill_cuda_graph",
                _make_options(
                    tp_size=2,
                    prefill_cuda_graph_backend=Backend.FULL,
                ),
                _make_qwen35_hybrid_vlm_model_config(),
                "does not support full prefill CUDA graphs",
            ),
        )
        for name, options, model_config, reason in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, re.escape(reason)):
                    self._create(options=options, model_config=model_config)

    def test_qwen35_moe_hybrid_vlm_profile_is_admitted(self):
        manager = self._create(
            options=_make_options(
                tp_size=2,
                ep_size=2,
                prefill_cuda_graph_backend=Backend.BREAKABLE,
            ),
            model_config=_make_qwen35_moe_hybrid_vlm_model_config(),
        )

        self.assertEqual(
            manager._plan.profile,
            StartupWeightLoadProfile.QWEN3_5_MOE_HYBRID_VLM,
        )

    def test_qwen35_moe_hybrid_vlm_profile_rejects_near_misses(self):
        cases = (
            (
                "tp4",
                _make_options(
                    tp_size=4,
                    ep_size=2,
                    prefill_cuda_graph_backend=Backend.BREAKABLE,
                ),
                _make_qwen35_moe_hybrid_vlm_model_config(),
                "requires TP2",
            ),
            (
                "tp_only",
                _make_options(
                    tp_size=2,
                    prefill_cuda_graph_backend=Backend.BREAKABLE,
                ),
                _make_qwen35_moe_hybrid_vlm_model_config(),
                "requires EP2",
            ),
            (
                "quantized",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    prefill_cuda_graph_backend=Backend.BREAKABLE,
                ),
                _make_qwen35_moe_hybrid_vlm_model_config(quantization="fp8"),
                "quantization is not supported",
            ),
            (
                "text_only",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    prefill_cuda_graph_backend=Backend.BREAKABLE,
                ),
                _make_qwen35_moe_hybrid_vlm_model_config(is_multimodal=False),
                "requires multimodal execution",
            ),
            (
                "flashinfer_linear_attention",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    linear_attn_backend="flashinfer",
                    prefill_cuda_graph_backend=Backend.BREAKABLE,
                ),
                _make_qwen35_moe_hybrid_vlm_model_config(),
                "requires Triton linear attention",
            ),
            (
                "full_prefill_cuda_graph",
                _make_options(tp_size=2, ep_size=2),
                _make_qwen35_moe_hybrid_vlm_model_config(),
                "does not support full prefill CUDA graphs",
            ),
        )
        for name, options, model_config, reason in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, re.escape(reason)):
                    self._create(options=options, model_config=model_config)

    def test_qwen3_moe_ep_profile_is_admitted(self):
        manager = self._create(
            options=_make_options(tp_size=2, ep_size=2),
            model_config=_make_qwen3_moe_model_config(),
        )

        self.assertEqual(
            manager._plan.profile,
            StartupWeightLoadProfile.QWEN3_MOE_EP,
        )

    def test_qwen3_moe_ep_profile_rejects_near_misses(self):
        cases = (
            ("tp1", _make_options(ep_size=2), "requires TP2"),
            (
                "fp16",
                _make_options(tp_size=2, ep_size=2),
                "requires BF16",
                _make_qwen3_moe_model_config(dtype=torch.float16),
            ),
            (
                "quantized",
                _make_options(tp_size=2, ep_size=2),
                "quantization is not supported",
                _make_qwen3_moe_model_config(quantization="fp8"),
            ),
            (
                "tp_only",
                _make_options(tp_size=2),
                "requires EP2",
            ),
            (
                "moe_dp",
                _make_options(tp_size=2, ep_size=2, moe_dp_size=2),
                "MoE data parallelism is not supported",
            ),
            (
                "deepep",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    moe_a2a_backend="deepep",
                ),
                "requires the standard EP path",
            ),
            (
                "flashinfer_runner",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    moe_runner_backend="flashinfer_trtllm",
                ),
                "requires the Triton MoE runner",
            ),
            (
                "dp_attention",
                _make_options(tp_size=2, ep_size=2, enable_dp_attention=True),
                "DP attention is not supported",
            ),
            (
                "two_batch_overlap",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    enable_two_batch_overlap=True,
                ),
                "two-batch overlap is not supported",
            ),
            (
                "eplb",
                _make_options(tp_size=2, ep_size=2, enable_eplb=True),
                "EPLB is not supported",
            ),
            (
                "redundant_experts",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    ep_num_redundant_experts=1,
                ),
                "redundant experts are not supported",
            ),
            (
                "expert_placement",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    init_expert_location="random",
                ),
                "non-trivial expert placement is not supported",
            ),
            (
                "elastic_ep_backend",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    elastic_ep_backend="nixl",
                ),
                "elastic expert parallelism is not supported",
            ),
            (
                "elastic_expert_backup",
                _make_options(
                    tp_size=2,
                    ep_size=2,
                    enable_elastic_expert_backup=True,
                ),
                "elastic expert parallelism is not supported",
            ),
            (
                "ep_join",
                _make_options(tp_size=2, ep_size=2, ep_join_mode="scale"),
                "elastic expert parallelism is not supported",
            ),
            (
                "max_ep_size",
                _make_options(tp_size=2, ep_size=2, max_ep_size=4),
                "elastic expert parallelism is not supported",
            ),
        )
        default_model_config = _make_qwen3_moe_model_config()
        for case in cases:
            name, options, reason, *model_configs = case
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, re.escape(reason)):
                    self._create(
                        options=options,
                        model_config=(
                            model_configs[0] if model_configs else default_model_config
                        ),
                    )

    def test_admission_collects_all_rejections_in_rule_order(self):
        model_config = _make_model_config(quantization="fp8")
        with (
            patch(
                f"{_STARTUP_MODULE}.get_model_architecture",
                return_value=(_CanonicalModel, "LlamaForCausalLM"),
            ) as resolve_architecture,
            patch(
                f"{_STARTUP_MODULE}._get_canonical_model_class",
                return_value=_CanonicalModel,
            ) as get_canonical_model_class,
        ):
            admission = evaluate_startup_weight_load_admission(
                loader=self.loader,
                model_config=model_config,
                load_config=self.load_config,
                options=_make_options(
                    device="cpu",
                    is_cuda_platform=False,
                    tp_size=3,
                ),
            )

        self.assertFalse(admission.supported)
        self.assertIsNone(admission.plan)
        self.assertEqual(
            tuple(rejection.code for rejection in admission.rejections),
            ("non_cuda", "tensor_parallelism", "quantization"),
        )
        resolve_architecture.assert_not_called()
        get_canonical_model_class.assert_not_called()

    def test_create_formats_rejection_codes_and_messages(self):
        with self.assertRaisesRegex(
            ValueError,
            "non_cuda: CUDA only; tensor_parallelism: only TP1 and TP2",
        ):
            self._create(
                options=_make_options(
                    device="cpu",
                    is_cuda_platform=False,
                    tp_size=3,
                )
            )

    def test_empty_or_multiple_architectures_fail_without_resolution(self):
        for name, architectures in (
            ("empty", []),
            ("multiple", ["LlamaForCausalLM", "Qwen2ForCausalLM"]),
        ):
            with (
                self.subTest(name=name),
                patch(
                    f"{_STARTUP_MODULE}.get_model_architecture"
                ) as resolve_architecture,
                patch(
                    f"{_STARTUP_MODULE}._get_canonical_model_class"
                ) as get_canonical_model_class,
            ):
                admission = evaluate_startup_weight_load_admission(
                    loader=self.loader,
                    model_config=_make_model_config(
                        hf_config=SimpleNamespace(architectures=architectures)
                    ),
                    load_config=self.load_config,
                    options=_make_options(),
                )

                self.assertEqual(
                    tuple(rejection.code for rejection in admission.rejections),
                    ("architecture",),
                )
                self.assertIn(
                    "exactly one supported model architecture",
                    admission.rejections[0].message,
                )
                resolve_architecture.assert_not_called()
                get_canonical_model_class.assert_not_called()

    def test_options_accept_current_server_args_schema(self):
        """Removed server options must not break overlap startup initialization."""
        server_args = ServerArgs(
            model_path="dummy", cuda_graph_config=CudaGraphConfig()
        )
        # The parallel sizes come from the bags, so the config has to be published.
        publish(server_args, role="test")
        self.addCleanup(reset_context)
        options = StartupWeightLoadOptions.from_server_args(
            server_args=server_args,
            is_draft_worker=False,
        )

        self.assertIsInstance(options, StartupWeightLoadOptions)
        self.assertEqual(options.linear_attn_backend, "triton")
        self.assertIsNone(options.linear_attn_decode_backend)
        self.assertIsNone(options.linear_attn_prefill_backend)
        self.assertEqual(options.moe_a2a_backend, "none")
        self.assertEqual(options.moe_runner_backend, "auto")

        for mode, expected_overlap, expected_attempt in (
            ("serial", False, False),
            ("overlap", True, True),
            ("auto", False, True),
        ):
            with self.subTest(mode=mode):
                server_args.startup_weight_load_mode = mode
                self.assertEqual(
                    server_args.is_startup_weight_load_overlap,
                    expected_overlap,
                )
                self.assertEqual(
                    server_args.should_attempt_startup_weight_load_overlap,
                    expected_attempt,
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
                "zero_prefetch_threads",
                dict(options=_make_options(prefetch_num_threads=0)),
                "checkpoint prefetch requires at least one thread",
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
                "exactly one supported model architecture is required",
            ),
        )
        for name, kwargs, reason in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, re.escape(reason)):
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

    def test_auto_secondary_weights_fallback_reuses_initialized_model(self):
        trace = []
        loader = _RecordingLoader(_TiedWeightModel(), trace)
        loader.num_resolved_sources = 2
        manager = self._manager(loader, fallback_to_serial=True)

        manager.prepare()

        self.assertEqual(trace, ["initialize", "resolve", "serial_load"])
        self.assertEqual(manager.state, StartupWeightLoadState.READY)
        self.assertFalse(manager.is_deferred)

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
                "commit",
                "stop_prefetch",
                "cancel_prefetch",
                "wait_prefetch",
            ],
        )

        # Finalization is idempotent after a successful commit.
        self.assertIs(manager.finalize(), manager._timings)
        self.assertEqual(trace.count("commit"), 1)
        self.assertIs(model.weight, model.tied_weight)
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 3))
        self.assertTrue(log_info.call_args.args[0].startswith("Load weight end."))
        self.assertTrue(manager._loader.startup_prefetch_active)
        self.assertEqual(
            parallel_state_patch.call_args_list,
            [call(), call(reverse=True)],
        )

    def test_finalize_returns_complete_phase_timings(self):
        trace = []
        model = _TiedWeightModel()
        with patch(
            f"{_STARTUP_MODULE}.time.perf_counter",
            side_effect=(0.0, 2.0, 3.0, 7.0, 11.0, 13.0),
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
                commit_seconds=4.0,
                prefetch_cleanup_seconds=2.0,
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
                "changed graph-visible tensor storage: derived:graph_weight",
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

        self.assertTrue(loader.startup_prefetch_active)
        warning.assert_called_once()
        self.assertIn("falling back", warning.call_args.args[2])
        self.assertLess(trace.index("cancel_prefetch"), trace.index("commit"))
        self.assertLess(trace.index("commit"), trace.index("stop_prefetch"))

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
        self.assertLess(trace.index("cancel_prefetch"), trace.index("commit"))
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

    def test_prefetch_failure_during_commit_is_reported_after_real_load(self):
        trace = []
        model = _TiedWeightModel()
        loader = _RecordingLoader(model, trace)
        original_commit = loader.commit_model_weights

        def fail_prefetch_during_commit(**kwargs):
            original_commit(**kwargs)
            loader.prefetch_handle.errors = (("late.safetensors", OSError("failed")),)

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

        warning.assert_called_once()
        self.assertIn("completed despite", warning.call_args.args[2])
        self.assertLess(trace.index("commit"), trace.index("stop_prefetch"))

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
            patch(f"{_LOAD_MODEL_UTILS_MODULE}.get_model_loader", return_value=loader),
            patch(f"{_LOAD_MODEL_UTILS_MODULE}.monkey_patch_vllm_parallel_state"),
            patch.object(
                StartupWeightLoadManager,
                "create_from_server_args",
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


class TestCaptureSafeWeightInitialization(CustomTestCase):
    def test_only_parameters_are_filled(self):
        model = _TiedWeightModel()

        initialize_capture_safe_weights(model, value=0.125)

        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 0.125))
        torch.testing.assert_close(model.scale, torch.ones_like(model.scale))


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
        return StartupWeightLoadTimings(
            prepare_seconds=1.0,
            prefetch_start_delay_seconds=0.0,
            prefetch_window_seconds=2.0,
            commit_seconds=3.0,
            prefetch_cleanup_seconds=4.0,
            total_seconds=10.0,
        )


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
        runner.weight_load_time = 0.0
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
            token_to_kv_pool=SimpleNamespace(post_capture_active=post_capture_active),
            post_capture_resize_kv_pool=lambda: trace.append("resize"),
        )

    def start_startup_weight_load(self):
        self._trace.append("start")

    def finalize_startup_weight_load(self):
        self._trace.append("finalize")

    def has_startup_weight_load(self):
        return self._startup_weight_load_active


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
        worker = _SchedulerWorker(
            trace,
            startup_weight_load_active=mode in ("overlap", "auto_overlap"),
            post_capture_active=True,
        )
        scheduler = self._scheduler(worker, trace, mode=mode)

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
            self.assertRaisesRegex(RuntimeError, "stop after startup"),
        ):
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

    def test_auto_routes_from_the_actual_admission_result(self):
        self.assertEqual(
            self._run_startup("auto"),
            ["memory_pool", "attention", "capture", "resize"],
        )
        self.assertEqual(
            self._run_startup("auto_overlap"),
            ["start", "memory_pool", "attention", "capture", "resize", "finalize"],
        )


if __name__ == "__main__":
    unittest.main()
