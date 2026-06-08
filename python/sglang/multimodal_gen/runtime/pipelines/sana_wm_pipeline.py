# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import get_memory_usage_of_component
from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import (
    VanillaD2HStrategy,
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    LAYERWISE_OFFLOAD_ALL_COMPONENTS,
    LAYERWISE_OFFLOAD_DIT_GROUP,
    normalize_layerwise_offload_components,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.stages import (
    SanaWMBeforeDenoisingStage,
    SanaWMDecodingStage,
    SanaWMDenoisingStage,
    SanaWMTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.refiner import (
    SanaWMLTX2RefinerStage,
    SanaWMRefinerDecodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)

_SANA_WM_NATIVE_REFINER_CLASS = "SanaWMLTX2VideoRefiner"
_SANA_WM_REFINER_BACKENDS: tuple[str, ...] = ("auto", "official", "native")
_SANA_WM_TWO_STAGE_RESIDENCY_MODES: tuple[str, ...] = (
    "auto",
    "resident",
    "sequential",
)


@dataclass(frozen=True)
class _SanaWMRefinerSubmodule:
    component_name: str
    subpath: str


_SANA_WM_REFINER_SUB_MODULES: tuple[_SanaWMRefinerSubmodule, ...] = (
    _SanaWMRefinerSubmodule("transformer_2", "refiner/transformer"),
    _SanaWMRefinerSubmodule("connectors", "refiner/connectors"),
    _SanaWMRefinerSubmodule("text_encoder_2", "refiner/text_encoder"),
    _SanaWMRefinerSubmodule("tokenizer_2", "refiner/text_encoder"),
)

_SANA_WM_SEQUENTIAL_RESIDENCY_COMPONENTS: tuple[str, ...] = (
    "text_encoder",
    "transformer",
    "text_encoder_2",
    "connectors",
    "transformer_2",
)


class _OfficialLayerwiseModule(nn.Module, LayerwiseOffloadableModuleMixin):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.layerwise_offload_managers = []

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError as exc:
            wrapped = self.__dict__.get("module")
            if wrapped is not None:
                return getattr(wrapped, name)
            modules = self.__dict__.get("_modules", {})
            wrapped = modules.get("module")
            if wrapped is not None:
                return getattr(wrapped, name)
            raise exc


class _OfficialDiffusersLTX2RefinerModule(_OfficialLayerwiseModule):
    """Thin offload wrapper around Diffusers' official LTX-2 refiner module."""

    layer_names = ["module.transformer_blocks"]


class _OfficialGemma3TextEncoderModule(_OfficialLayerwiseModule):
    layer_names = [
        "module.language_model.layers",
        "module.model.language_model.layers",
    ]


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _sana_wm_skip_refiner_enabled(
    batch: Any | None = None,
    pipeline_config: Any | None = None,
) -> bool:
    if _truthy_flag(getattr(pipeline_config, "sana_wm_skip_refiner", False)):
        return True
    if batch is None:
        return False
    extra = getattr(batch, "extra", None) or {}
    return any(
        _truthy_flag(value)
        for value in (
            extra.get("skip_refiner"),
            extra.get("sana_wm_skip_refiner"),
        )
    )


def _default_sana_wm_refiner_dtype(server_args: ServerArgs) -> torch.dtype:
    precision = getattr(server_args.pipeline_config, "dit_precision", "bf16")
    return PRECISION_TO_TYPE.get(precision, torch.bfloat16)


def _normalize_sana_wm_choice(
    value,
    *,
    default: str,
    valid_values: tuple[str, ...],
    name: str,
    strict: bool,
) -> str:
    mode = default if value is None else str(value).strip().lower()
    if not mode:
        mode = default
    if mode in valid_values:
        return mode

    if strict:
        raise ValueError(
            f"{name} must be one of {sorted(valid_values)}, got {value!r}."
        )

    logger.warning(
        "Ignoring invalid %s=%r. Expected one of %s; using %r.",
        name,
        value,
        sorted(valid_values),
        default,
    )
    return default


def _normalize_sana_wm_refiner_backend(
    value,
    *,
    strict: bool = True,
    name: str = "sana_wm_refiner_backend",
) -> str:
    return _normalize_sana_wm_choice(
        value,
        default="auto",
        valid_values=_SANA_WM_REFINER_BACKENDS,
        name=name,
        strict=strict,
    )


def _normalize_sana_wm_two_stage_residency(
    value,
    *,
    strict: bool = True,
    name: str = "sana_wm_two_stage_residency",
) -> str:
    return _normalize_sana_wm_choice(
        value,
        default="auto",
        valid_values=_SANA_WM_TWO_STAGE_RESIDENCY_MODES,
        name=name,
        strict=strict,
    )


def _nonempty_config_string(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _resolve_sana_wm_startup_choice(
    server_args: ServerArgs,
    *,
    field_name: str,
    normalizer: Callable[..., str],
) -> str:
    mode = _nonempty_config_string(getattr(server_args, field_name, None))
    if mode is not None:
        return normalizer(mode, strict=True, name=f"server_args.{field_name}")

    pipeline_config = getattr(server_args, "pipeline_config", None)
    pipeline_mode = _nonempty_config_string(getattr(pipeline_config, field_name, None))
    if pipeline_mode is not None and pipeline_mode.lower() != "auto":
        return normalizer(
            pipeline_mode,
            strict=True,
            name=f"pipeline_config.{field_name}",
        )

    if pipeline_mode is not None:
        return normalizer(
            pipeline_mode,
            strict=True,
            name=f"pipeline_config.{field_name}",
        )

    return normalizer("auto", strict=True, name=field_name)


def _configured_sana_wm_refiner_backend(server_args: ServerArgs) -> str:
    mode = _resolve_sana_wm_startup_choice(
        server_args,
        field_name="sana_wm_refiner_backend",
        normalizer=_normalize_sana_wm_refiner_backend,
    )
    if mode != "auto":
        return mode

    try:
        tp_size = max(int(getattr(server_args, "tp_size", 1) or 1), 1)
    except (TypeError, ValueError):
        tp_size = 1
    if tp_size > 1 and getattr(server_args, "enable_cfg_parallel", False):
        return "official"
    return "native"


def _configured_sana_wm_two_stage_residency(server_args: ServerArgs) -> str:
    return _resolve_sana_wm_startup_choice(
        server_args,
        field_name="sana_wm_two_stage_residency",
        normalizer=_normalize_sana_wm_two_stage_residency,
    )


def _sana_wm_dit_num_attention_heads(dit_config: Any) -> int | None:
    arch = getattr(dit_config, "arch_config", None)
    num_heads = getattr(arch, "num_attention_heads", None)
    if num_heads is None:
        return None
    return int(num_heads)


def _sana_wm_common_tp_sizes(*num_heads_values: int) -> list[int]:
    if not num_heads_values:
        return []
    max_candidate = min(num_heads_values)
    return [
        candidate
        for candidate in range(1, max_candidate + 1)
        if all(num_heads % candidate == 0 for num_heads in num_heads_values)
    ]


def _configured_sana_wm_cfg_parallel_degree(server_args: ServerArgs) -> int:
    if not getattr(server_args, "enable_cfg_parallel", False):
        return 1
    try:
        value = getattr(server_args, "cfg_parallel_degree", 2)
        if value is None:
            return 2
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 2


def _validate_sana_wm_stage1_parallelism(server_args: ServerArgs) -> None:
    tp_size = getattr(server_args, "tp_size", 1) or 1
    if tp_size < 1:
        raise ValueError(f"Invalid SANA-WM tensor parallelism size: {tp_size}.")
    if tp_size > 1:
        pipeline_config = getattr(server_args, "pipeline_config", None)
        dit_config = getattr(pipeline_config, "dit_config", None)
        num_heads = _sana_wm_dit_num_attention_heads(dit_config)
        if num_heads is not None and num_heads % tp_size != 0:
            raise ValueError(
                "SANA-WM tensor parallelism requires num_attention_heads "
                f"({num_heads}) to be divisible by tp_size ({tp_size})."
            )
        logger.info(
            "SANA-WM tensor parallelism enabled with tp_size=%d. "
            "Attention/GDN heads are sharded across TP ranks; sequence "
            "parallelism remains disabled.",
            tp_size,
        )

    if getattr(server_args, "enable_cfg_parallel", False):
        cfg_parallel_degree = _configured_sana_wm_cfg_parallel_degree(server_args)
        if cfg_parallel_degree != 2:
            raise ValueError(
                "SANA-WM CFG parallelism requires exactly two branches "
                "(positive and negative prompt). "
                f"Expected cfg_parallel_degree == 2, got {cfg_parallel_degree}."
            )

    sp_degree = getattr(server_args, "sp_degree", 1) or 1
    if sp_degree != 1:
        raise ValueError(
            "SANA-WM does not support sequence parallelism, so --sp-degree "
            f"must be 1; got sp_degree={sp_degree}. The stage-1 DiT contains "
            "layout-dependent cross-frame operators (bidirectional GDN scan, "
            "GLUMBConvTemp temporal convolution, camera UCPE, and Plucker "
            "conditioning). Sharding the time/sequence dimension would require "
            "operator-aware state or halo exchange and can otherwise be slower "
            "or incorrect. Use tensor parallelism instead, e.g. tp_size=2 or "
            "tp_size=4."
        )


def _validate_sana_wm_native_refiner_tp_args(server_args: ServerArgs) -> None:
    tp_size = getattr(server_args, "tp_size", 1) or 1
    if tp_size <= 1:
        return

    pipeline_config = getattr(server_args, "pipeline_config", None)
    if _sana_wm_skip_refiner_enabled(pipeline_config=pipeline_config):
        return
    refiner_backend = _configured_sana_wm_refiner_backend(server_args)
    if refiner_backend != "native":
        return

    if getattr(server_args, "enable_cfg_parallel", False):
        raise ValueError(
            "SANA-WM two-stage native refiner does not support "
            "enable_cfg_parallel with tp_size > 1. The native refiner is "
            "tensor-parallel and must run on all TP ranks, while CFG "
            "parallel refiner execution currently runs only one branch/rank. "
            "Use sana_wm_refiner_backend=official, disable CFG parallelism, "
            "or set sana_wm_skip_refiner=true."
        )

    stage1_heads = _sana_wm_dit_num_attention_heads(
        getattr(pipeline_config, "dit_config", None)
    )
    refiner_heads = _sana_wm_dit_num_attention_heads(
        getattr(pipeline_config, "refiner_dit_config", None)
    )
    if stage1_heads is None or refiner_heads is None:
        return
    if refiner_heads % tp_size == 0:
        return

    valid_tp_sizes = _sana_wm_common_tp_sizes(stage1_heads, refiner_heads)
    raise ValueError(
        "SANA-WM two-stage native refiner tensor parallelism requires "
        "tp_size to divide both stage-1 num_attention_heads "
        f"({stage1_heads}) and refiner num_attention_heads ({refiner_heads}). "
        f"Valid common tp_size values: {valid_tp_sizes}; got tp_size={tp_size}. "
        "Use tp_size=2 or tp_size=4 for the two-stage native refiner path, "
        "or set sana_wm_refiner_backend=official / sana_wm_skip_refiner=true."
    )


def _validate_sana_wm_two_stage_parallelism(server_args: ServerArgs) -> None:
    _validate_sana_wm_stage1_parallelism(server_args)
    _validate_sana_wm_native_refiner_tp_args(server_args)


def _resolve_sana_wm_refiner_component_paths(
    model_path: str,
    component_paths: dict[str, str] | None,
) -> dict[str, str]:
    resolved = dict(component_paths or {})
    refiner_root = resolved.get("refiner", os.path.join(model_path, "refiner"))
    refiner_text_encoder_root = resolved.get(
        "refiner_text_encoder",
        resolved.get("text_encoder_2", os.path.join(refiner_root, "text_encoder")),
    )

    defaults = {
        "transformer_2": os.path.join(refiner_root, "transformer"),
        "connectors": os.path.join(refiner_root, "connectors"),
        "text_encoder_2": refiner_text_encoder_root,
        "tokenizer_2": refiner_text_encoder_root,
    }
    auto_resolved = []
    for component_name, path in defaults.items():
        if component_name not in resolved:
            resolved[component_name] = path
            auto_resolved.append(f"{component_name}={path}")

    if auto_resolved:
        logger.info(
            "Auto-resolved SANA-WM refiner components: %s",
            ", ".join(auto_resolved),
        )
    return resolved


class _SanaWMRefinerModuleLoader:
    def load_modules(
        self,
        server_args: ServerArgs,
        *,
        native_loader: Callable[[str, ServerArgs], tuple[Any, float]],
        official_loader: Callable[[str, str, ServerArgs], tuple[Any, float]],
    ) -> dict[str, tuple[Any, float]]:
        backend = _configured_sana_wm_refiner_backend(server_args)
        logger.info("SANA-WM refiner backend resolved to %s.", backend)

        loaded: dict[str, tuple[Any, float]] = {}
        component_paths = getattr(server_args, "component_paths", {}) or {}
        for spec in _SANA_WM_REFINER_SUB_MODULES:
            component_path = component_paths.get(spec.component_name)
            if component_path is None:
                raise ValueError(
                    "SANA-WM refiner component path was not resolved for "
                    f"{spec.component_name}. Call "
                    "_resolve_sana_wm_refiner_component_paths before loading "
                    "refiner modules."
                )
            logger.info(
                "SANA-WM loading refiner component %s from %s",
                spec.component_name,
                component_path,
            )
            if spec.component_name == "transformer_2" and backend == "native":
                loaded[spec.component_name] = native_loader(
                    component_path,
                    server_args,
                )
            else:
                loaded[spec.component_name] = official_loader(
                    spec.component_name,
                    component_path,
                    server_args,
                )
        return loaded

    @staticmethod
    def load_native_refiner_transformer(
        component_path: str,
        server_args: ServerArgs,
    ) -> tuple[Any, float]:
        module, memory_usage = PipelineComponentLoader.load_component(
            component_name="transformer_2",
            component_model_path=component_path,
            transformers_or_diffusers="diffusers",
            server_args=server_args,
            component_architecture=_SANA_WM_NATIVE_REFINER_CLASS,
        )
        if module.__class__.__name__ != _SANA_WM_NATIVE_REFINER_CLASS:
            raise RuntimeError(
                "SANA-WM native refiner backend expected "
                f"{_SANA_WM_NATIVE_REFINER_CLASS}, got {module.__class__.__name__}."
            )
        return module, memory_usage

    @staticmethod
    def load_official_refiner_component(
        module_name: str,
        component_path: str,
        server_args: ServerArgs,
    ) -> tuple[Any, float]:
        dtype = _default_sana_wm_refiner_dtype(server_args)
        if module_name == "transformer_2":
            from diffusers.models.transformers.transformer_ltx2 import (
                LTX2VideoTransformer3DModel,
            )

            module = LTX2VideoTransformer3DModel.from_pretrained(
                component_path,
                torch_dtype=dtype,
            ).eval()
            module = _OfficialDiffusersLTX2RefinerModule(module)
        elif module_name == "connectors":
            from diffusers.pipelines.ltx2 import LTX2TextConnectors

            module = LTX2TextConnectors.from_pretrained(
                component_path,
                torch_dtype=dtype,
            ).eval()
        elif module_name == "text_encoder_2":
            from transformers import Gemma3ForConditionalGeneration

            module = Gemma3ForConditionalGeneration.from_pretrained(
                component_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).eval()
            module = _OfficialGemma3TextEncoderModule(module)
        elif module_name == "tokenizer_2":
            from transformers import AutoTokenizer

            module = AutoTokenizer.from_pretrained(component_path)
        else:
            raise ValueError(f"Unsupported SANA-WM refiner component: {module_name}")

        memory_usage = get_memory_usage_of_component(module)
        logger.info(
            "Loaded %s: %s (official diffusers version). model size: %s GB",
            module_name,
            module.__class__.__name__,
            memory_usage if memory_usage is not None else "NA",
        )
        return module, memory_usage or 0.0


class _SanaWMTwoStageResidencyPlanner:
    def __init__(
        self,
        *,
        modules: dict[str, Any],
        component_residency_strategies: dict[str, Any],
        component_names: tuple[str, ...] = _SANA_WM_SEQUENTIAL_RESIDENCY_COMPONENTS,
    ) -> None:
        self.modules = modules
        self.component_residency_strategies = component_residency_strategies
        self.component_names = component_names

    @staticmethod
    def residency_mode(server_args: ServerArgs | None = None) -> str:
        if server_args is not None:
            return _configured_sana_wm_two_stage_residency(server_args)
        return _normalize_sana_wm_two_stage_residency(
            "auto",
            strict=False,
            name="sana_wm_two_stage_residency",
        )

    @staticmethod
    def has_conflicting_memory_policy(server_args: ServerArgs) -> bool:
        if getattr(server_args, "use_fsdp_inference", False):
            return True
        if getattr(server_args, "dit_layerwise_offload", False):
            return True

        layerwise_components = normalize_layerwise_offload_components(
            getattr(server_args, "layerwise_offload_components", None)
        )
        if layerwise_components is None:
            return False

        return bool(
            {
                LAYERWISE_OFFLOAD_ALL_COMPONENTS,
                LAYERWISE_OFFLOAD_DIT_GROUP,
                "transformer",
                "transformer_2",
            }
            & set(layerwise_components)
        )

    def should_use_sequential(self, server_args: ServerArgs) -> tuple[bool, str]:
        mode = self.residency_mode(server_args)
        if mode == "resident":
            return False, "sana_wm_two_stage_residency=resident"
        if mode == "sequential":
            return True, "sana_wm_two_stage_residency=sequential"

        if self.has_conflicting_memory_policy(server_args):
            return False, "FSDP or DiT layerwise offload is active"

        performance_mode = str(
            getattr(server_args, "performance_mode", "auto") or "auto"
        ).lower()
        tp_size = getattr(server_args, "tp_size", 1) or 1
        cfg_parallel = bool(getattr(server_args, "enable_cfg_parallel", False))
        is_manual_memory_path = performance_mode in ("manual", "memory")
        is_tp_without_cfg_path = tp_size > 1 and not cfg_parallel
        is_single_gpu = tp_size <= 1

        if is_manual_memory_path or is_tp_without_cfg_path or is_single_gpu:
            return (
                True,
                "auto detected no FSDP/layerwise policy with "
                f"performance_mode={performance_mode}, tp_size={tp_size}, "
                f"enable_cfg_parallel={cfg_parallel}",
            )

        return False, "auto did not detect a high-risk residency combination"

    def apply(self, server_args: ServerArgs) -> list[str]:
        should_configure, reason = self.should_use_sequential(server_args)
        if not should_configure:
            logger.info(
                "SANA-WM two-stage sequential residency disabled: %s. "
                "Set sana_wm_two_stage_residency=sequential to force the "
                "memory-safe path.",
                reason,
            )
            return []

        configured: list[str] = []
        for component_name in self.component_names:
            if component_name in self.component_residency_strategies:
                continue
            module = self.modules.get(component_name)
            if not isinstance(module, nn.Module):
                continue
            if is_fsdp_managed_module(module) or is_layerwise_offloaded_module(module):
                continue
            self.component_residency_strategies[component_name] = VanillaD2HStrategy()
            configured.append(component_name)

        if configured:
            logger.info(
                "SANA-WM two-stage sequential residency enabled (%s) for "
                "components: %s. Set sana_wm_two_stage_residency=resident to "
                "force the GPU-resident path.",
                reason,
                configured,
            )
        return configured


class SanaWMPipeline(LoRAPipeline, ComposedPipelineBase):
    """SANA-WM TI2V pipeline (single-stage)."""

    pipeline_name = "SanaWMPipeline"
    pipeline_config_cls = SanaWMPipelineConfig
    sampling_params_cls = SanaWMSamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    @staticmethod
    def _validate_parallelism_args(server_args: ServerArgs) -> None:
        _validate_sana_wm_stage1_parallelism(server_args)

    def create_pipeline_stages(self, server_args: ServerArgs):
        self._validate_parallelism_args(server_args)
        self.add_stage(InputValidationStage())

        self.add_stage(
            SanaWMTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
            "prompt_encoding_stage",
        )

        self.add_stage(
            SanaWMBeforeDenoisingStage(
                vae=self.get_module("vae"),
                scheduler=self.get_module("scheduler"),
                pipeline_config=server_args.pipeline_config,
            ),
            "sana_wm_before_denoising",
        )

        self.add_stage(
            SanaWMDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # Subclasses (e.g. SanaWMTwoStagePipeline) insert latent-domain stages
        # between denoising and VAE decoding.
        self._maybe_add_refiner_stage(server_args)

        self._add_decoding_stage(server_args)

    def _add_decoding_stage(self, server_args: ServerArgs | None = None) -> None:
        self.add_stage(
            SanaWMDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
                component_name="vae",
            ),
            "decoding_stage",
        )

    def _maybe_add_refiner_stage(self, server_args: ServerArgs) -> None:
        """Hook for subclasses; single-stage pipeline is a no-op."""
        return None


class SanaWMTwoStagePipeline(SanaWMPipeline):
    """SANA-WM two-stage pipeline: SANA-WM DiT + LTX-2 latent refiner.

    Stage-1 generates a coarse 720p latent; the LTX-2 video refiner then runs
    3 Euler steps on that latent before VAE decode, matching the NVlabs
    ``inference_sana_wm.py`` default.
    """

    pipeline_name = "SanaWMTwoStagePipeline"

    # Stage-2 refiner sub-modules and their on-disk layout. These are loaded
    # through the official Diffusers/Transformers classes because NVlabs'
    # reference refiner is a narrow video-only wrapper around those modules.
    _REFINER_SUB_MODULES: tuple[tuple[str, str], ...] = tuple(
        (spec.component_name, spec.subpath) for spec in _SANA_WM_REFINER_SUB_MODULES
    )

    @staticmethod
    def _validate_parallelism_args(server_args: ServerArgs) -> None:
        _validate_sana_wm_two_stage_parallelism(server_args)

    @staticmethod
    def _validate_native_refiner_tp_args(server_args: ServerArgs) -> None:
        _validate_sana_wm_native_refiner_tp_args(server_args)

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self._validate_native_refiner_tp_args(server_args)
        super().initialize_pipeline(server_args)
        if _sana_wm_skip_refiner_enabled(
            pipeline_config=getattr(server_args, "pipeline_config", None)
        ):
            logger.info(
                "SANA-WM refiner component loading skipped by "
                "pipeline_config.sana_wm_skip_refiner."
            )
            return
        self._load_refiner_modules(server_args)
        self._configure_two_stage_component_residency(server_args)

    def _refiner_loader(self) -> _SanaWMRefinerModuleLoader:
        return _SanaWMRefinerModuleLoader()

    def _load_refiner_modules(self, server_args: ServerArgs) -> None:
        server_args.component_paths = _resolve_sana_wm_refiner_component_paths(
            self.model_path,
            getattr(server_args, "component_paths", {}) or {},
        )
        loaded = self._refiner_loader().load_modules(
            server_args,
            native_loader=_SanaWMRefinerModuleLoader.load_native_refiner_transformer,
            official_loader=_SanaWMRefinerModuleLoader.load_official_refiner_component,
        )
        for module_name, (module, memory_usage) in loaded.items():
            self.modules[module_name] = module
            self.memory_usages[module_name] = memory_usage

    def _configure_two_stage_component_residency(self, server_args: ServerArgs) -> None:
        """Avoid high-risk stage-1/refiner GPU residency overlap.

        SANA-WM's stage-2 refiner includes a large Gemma-3 text encoder and an
        LTX-2 transformer.  In manual TP runs without FSDP or layerwise
        offload, the default resident strategy can leave stage-1 text/DiT
        weights on GPU while the refiner starts loading, which exhausts H100
        memory even for small smoke inputs.  Auto mode applies this conservative
        residency only for those high-risk combinations; users can force either
        path with sana_wm_two_stage_residency=resident|sequential.
        """
        _SanaWMTwoStageResidencyPlanner(
            modules=self.modules,
            component_residency_strategies=self.component_residency_strategies,
        ).apply(server_args)

    def _maybe_add_refiner_stage(self, server_args: ServerArgs) -> None:
        if _sana_wm_skip_refiner_enabled(
            pipeline_config=getattr(server_args, "pipeline_config", None)
        ):
            return
        self.add_stage(
            SanaWMLTX2RefinerStage(
                transformer=self.get_module("transformer_2"),
                connectors=self.get_module("connectors"),
                text_encoder=self.get_module("text_encoder_2"),
                tokenizer=self.get_module("tokenizer_2"),
                dtype=_default_sana_wm_refiner_dtype(server_args),
            ),
            "sana_wm_refiner",
        )

    def _add_decoding_stage(self, server_args: ServerArgs | None = None) -> None:
        if _sana_wm_skip_refiner_enabled(
            pipeline_config=getattr(server_args, "pipeline_config", None)
        ):
            return super()._add_decoding_stage(server_args)
        self.add_stage(
            SanaWMRefinerDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
                component_name="vae",
            ),
            "decoding_stage",
        )


EntryClass = [SanaWMPipeline, SanaWMTwoStagePipeline]
