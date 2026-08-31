"""Helpers and adapters for transformer quantized checkpoint loading.

This module keeps format-specific loading quirks out of `TransformerLoader`.
The loader should stay focused on the generic load flow, while special cases
such as Nunchaku validation, NVFP4 fallback adjustments, and post-load patching
are handled here behind a small helper/adapter layer.
"""

import json
import os
import re
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import torch
from diffusers.utils import SAFE_WEIGHTS_INDEX_NAME
from safetensors import safe_open
from torch import nn

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a4_config import (
    KitchenW4A4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a8_config import (
    KitchenW4A8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
    _patch_nunchaku_scales,
)
from sglang.multimodal_gen.runtime.loader.gguf_weights import (
    names_gguf_checkpoint,
    read_gguf_tensor_meta,
)
from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    filter_duplicate_safetensors_files,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
    ComponentResidencyError,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    build_nvfp4_config_from_safetensors_list,
    get_metadata_from_safetensors_file,
    get_quant_config,
    get_quant_config_from_safetensors_metadata,
)
from sglang.multimodal_gen.runtime.weights.source import (
    materialize_weight_set,
    materialize_weight_set_config,
    resolve_safetensors_weight_set,
)
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)
from sglang.srt.utils.hf_transformers import (
    check_gguf_file,
    resolve_hf_gguf_reference,
)

logger = init_logger(__name__)

PostLoadHook = Callable[[nn.Module], None]

_PRECISION_VARIANT_SUFFIX_RE = re.compile(
    r"^(?P<stem>.+?)(?P<precision>\.(?:fp16|bf16|fp32))(?P<shard>-\d+-of-\d+)?(?P<ext>\.safetensors)$"
)
_MIXED_SAFETENSORS_RE = re.compile(r".*-mixed(?:-\d+-of-\d+)?\.safetensors$")


def _get_quant_config_name(config: Optional[QuantizationConfig]) -> Optional[str]:
    if config is None:
        return None
    return config.get_name()


def _merge_modelopt_fp4_configs(
    existing_config: Optional[QuantizationConfig],
    inferred_config: Optional[QuantizationConfig],
) -> Optional[QuantizationConfig]:
    """Prefer safetensors-inferred NVFP4 layout over stale config.json ignores.

    Some ModelOpt NVFP4 transformer repos ship a flat `quantization_config` in
    `config.json`, but its `ignore` list can lag behind the actual checkpoint
    contents. The safetensors shards are the source of truth for which modules
    remain BF16 fallbacks, so when we can infer an NVFP4 config from the shards
    we should use its exclude list while preserving explicit repo-level knobs
    such as `swap_weight_nibbles`.
    """
    if inferred_config is None:
        return existing_config

    if _get_quant_config_name(inferred_config) != "modelopt_fp4":
        return existing_config or inferred_config

    if existing_config is None:
        return inferred_config

    if _get_quant_config_name(existing_config) != "modelopt_fp4":
        return existing_config

    existing_excludes = getattr(existing_config, "exclude_modules", []) or []
    inferred_excludes = getattr(inferred_config, "exclude_modules", []) or []
    if inferred_excludes != existing_excludes:
        logger.warning(
            "Overriding ModelOpt NVFP4 exclude_modules from config.json with "
            "safetensors-inferred layout (%d -> %d entries).",
            len(existing_excludes),
            len(inferred_excludes),
        )

    inferred_config.packed_modules_mapping = getattr(
        existing_config, "packed_modules_mapping", {}
    )
    inferred_config.checkpoint_uses_packed_qkv = getattr(
        inferred_config, "checkpoint_uses_packed_qkv", False
    ) or getattr(existing_config, "checkpoint_uses_packed_qkv", False)
    inferred_config.swap_weight_nibbles = getattr(
        inferred_config, "swap_weight_nibbles", False
    ) or getattr(existing_config, "swap_weight_nibbles", False)
    existing_scale_layout = getattr(
        existing_config, "checkpoint_weight_scale_layout", "linear"
    )
    inferred_scale_layout = getattr(
        inferred_config, "checkpoint_weight_scale_layout", "linear"
    )
    inferred_config.checkpoint_weight_scale_layout = (
        existing_scale_layout
        if inferred_scale_layout == "linear" and existing_scale_layout != "linear"
        else inferred_scale_layout
    )
    if getattr(inferred_config, "group_size", None) is None:
        inferred_config.group_size = getattr(existing_config, "group_size", None)
    inferred_config.checkpoint_uses_comfy_quantization = (
        inferred_config.checkpoint_uses_comfy_quantization
        or existing_config.checkpoint_uses_comfy_quantization
    )
    inferred_config.checkpoint_uses_native_qkv_layout = (
        inferred_config.checkpoint_uses_native_qkv_layout
        or existing_config.checkpoint_uses_native_qkv_layout
    )

    return inferred_config


@dataclass
class TransformerQuantLoadSpec:
    """Resolved loading plan for a transformer checkpoint."""

    safetensors_list: list[str]
    quant_config: Optional[QuantizationConfig]
    nunchaku_config: Optional[NunchakuConfig]
    param_dtype: Optional[torch.dtype]
    needs_device_weight_postprocess: bool = False
    post_load_hooks: list[PostLoadHook] = field(default_factory=list)
    # Set instead of ``safetensors_list`` when the transformer comes from GGUF.
    gguf_file: Optional[str] = None

    @property
    def runtime_quant_config(self) -> Optional[object]:
        if self.quant_config is not None:
            return self.quant_config
        return self.nunchaku_config

    @property
    def is_modelopt_fp4(self) -> bool:
        return _get_quant_config_name(self.quant_config) == "modelopt_fp4"

    @property
    def is_comfy_fp8(self) -> bool:
        return _get_quant_config_name(self.quant_config) == "comfy_fp8"

    @property
    def is_serialized_kitchen_int8(self) -> bool:
        return (
            isinstance(self.quant_config, KitchenInt8Config)
            and self.quant_config.is_checkpoint_int8_serialized
        )

    @property
    def is_serialized_kitchen_w4a8(self) -> bool:
        return isinstance(self.quant_config, KitchenW4A8Config)

    @property
    def is_serialized_kitchen_w4a4(self) -> bool:
        return isinstance(self.quant_config, KitchenW4A4Config)

    @property
    def uses_comfy_layer_markers(self) -> bool:
        return (
            self.is_comfy_fp8
            or self.is_serialized_kitchen_int8
            or self.is_serialized_kitchen_w4a4
            or self.is_serialized_kitchen_w4a8
            or (
                self.quant_config is not None
                and self.quant_config.checkpoint_uses_comfy_quantization
            )
            or (
                _get_quant_config_name(self.quant_config) == "mxfp8"
                and self.quant_config.layer_markers is not None
            )
        )


@dataclass(frozen=True)
class TransformerCheckpointFiles:
    """Files from one checkpoint revision needed during transformer loading."""

    safetensors: tuple[str, ...]
    config_path: str | None


class _TransformerQuantAdapter:
    def prepare(self) -> None:
        """initialize"""
        pass

    def get_post_load_hooks(self) -> list[PostLoadHook]:
        """post - fsdp load - hook"""
        return []


def _uses_component_offload(
    server_args: ServerArgs,
    component_name: str | None,
    *,
    legacy_enabled: bool,
) -> bool:
    if component_name is None:
        return legacy_enabled
    return server_args.residency_mode(component_name) == COMPONENT_OFFLOAD


def _reject_explicit_component_selector(
    server_args: ServerArgs,
    component_name: str | None,
    *,
    feature_name: str,
) -> None:
    if component_name is None:
        return
    selected_by_component_residency = (
        server_args.canonical_residency_mode(component_name) == COMPONENT_OFFLOAD
    )
    if selected_by_component_residency:
        raise ComponentResidencyError(
            f"{feature_name} does not support component-offload for "
            f"{component_name!r}; select resident or layerwise-offload"
        )


class _NunchakuQuantAdapter(_TransformerQuantAdapter):
    """Adapter for Nunchaku checkpoints"""

    def __init__(
        self,
        *,
        nunchaku_config: NunchakuConfig,
        model_cls: type[nn.Module],
        safetensors_list: list[str],
    ) -> None:
        self.nunchaku_config = nunchaku_config
        self.model_cls = model_cls
        self.safetensors_list = safetensors_list

    @staticmethod
    def _validate_nunchaku_checkpoint_matches_model(
        nunchaku_config: NunchakuConfig, model_cls: type[nn.Module]
    ) -> None:
        metadata = get_metadata_from_safetensors_file(
            nunchaku_config.transformer_weights_path
        )
        original_dit_cls_name = json.loads(metadata.get("config"))["_class_name"]
        specified_dit_cls_name = str(model_cls.__name__)
        if original_dit_cls_name != specified_dit_cls_name:
            raise Exception(
                f"Class name of DiT specified in nunchaku transformer_weights_path: "
                f"{original_dit_cls_name} does not match that of specified DiT name: "
                f"{specified_dit_cls_name}"
            )

    def prepare(self) -> None:
        self.nunchaku_config.model_cls = self.model_cls
        _NunchakuQuantAdapter._validate_nunchaku_checkpoint_matches_model(
            nunchaku_config=self.nunchaku_config,
            model_cls=self.model_cls,
        )

    def get_post_load_hooks(self) -> list[PostLoadHook]:
        return [partial(_patch_nunchaku_scales, safetensors_list=self.safetensors_list)]


class _Flux2Nvfp4FallbackAdapter(_TransformerQuantAdapter):
    """Adapter for black-forest-labs/FLUX.2-dev-NVFP4"""

    def __init__(
        self,
        *,
        cls_name: str,
        server_args: ServerArgs,
        quant_config: Optional[QuantizationConfig],
        component_name: str | None,
    ) -> None:
        self.cls_name = cls_name
        self.server_args = server_args
        self.quant_config = quant_config
        self.component_name = component_name

    @staticmethod
    def _maybe_adjust_flux2_nvfp4_fallback_defaults(
        cls_name: str,
        server_args: ServerArgs,
        quant_config: Optional[QuantizationConfig],
        component_name: str | None = None,
    ) -> None:
        if cls_name != "Flux2Transformer2DModel" or quant_config is None:
            return

        if _get_quant_config_name(quant_config) != "modelopt_fp4":
            return

        weights_path = os.path.basename(server_args.transformer_weights_path or "")
        if not weights_path.endswith("-mixed.safetensors") or server_args.tp_size <= 1:
            return

        dit_component_offload = _uses_component_offload(
            server_args,
            component_name,
            legacy_enabled=bool(server_args.dit_cpu_offload),
        )
        text_encoder_component_offload = _uses_component_offload(
            server_args,
            "text_encoder" if component_name is not None else None,
            legacy_enabled=bool(server_args.text_encoder_cpu_offload),
        )
        if dit_component_offload:
            _reject_explicit_component_selector(
                server_args,
                component_name,
                feature_name="FLUX.2 mixed NVFP4 with tensor parallelism",
            )
        if text_encoder_component_offload:
            _reject_explicit_component_selector(
                server_args,
                "text_encoder" if component_name is not None else None,
                feature_name="FLUX.2 mixed NVFP4 with tensor parallelism",
            )
        if dit_component_offload or text_encoder_component_offload:
            if component_name is None:
                server_args.dit_cpu_offload = False
                server_args.text_encoder_cpu_offload = False
            else:
                if dit_component_offload:
                    server_args.require_component_resident(
                        component_name,
                        feature_name="FLUX.2 mixed NVFP4 with tensor parallelism",
                    )
                if text_encoder_component_offload:
                    server_args.require_component_resident(
                        "text_encoder",
                        feature_name="FLUX.2 mixed NVFP4 with tensor parallelism",
                    )
            logger.warning(
                "FLUX.2 mixed NVFP4 is using the ModelOpt FP4 path with tp_size=%d; "
                "keeping the DiT and text encoder resident to avoid TP all-gather "
                "launch failures.",
                server_args.tp_size,
            )

    def prepare(self) -> None:
        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name=self.cls_name,
            server_args=self.server_args,
            quant_config=self.quant_config,
            component_name=self.component_name,
        )


class _ModelOptFp8OffloadAdapter(_TransformerQuantAdapter):
    """Adapter for diffusion ModelOpt FP8 checkpoints."""

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        quant_config: Optional[QuantizationConfig],
        component_name: str | None,
    ) -> None:
        self.server_args = server_args
        self.quant_config = quant_config
        self.component_name = component_name

    @staticmethod
    def _maybe_disable_incompatible_dit_offload_modes(
        server_args: ServerArgs,
        quant_config: Optional[QuantizationConfig],
        component_name: str | None = None,
    ) -> None:
        if quant_config is None:
            return

        if _get_quant_config_name(quant_config) != "modelopt_fp8":
            return

        component_offload = _uses_component_offload(
            server_args,
            component_name,
            legacy_enabled=bool(server_args.dit_cpu_offload),
        )
        if component_offload:
            _reject_explicit_component_selector(
                server_args,
                component_name,
                feature_name="ModelOpt FP8 diffusion checkpoints",
            )
            if component_name is None:
                server_args.dit_cpu_offload = False
            else:
                server_args.require_component_resident(
                    component_name,
                    feature_name="ModelOpt FP8 diffusion checkpoints",
                )
            logger.warning(
                "ModelOpt FP8 diffusion checkpoints keep the DiT resident instead "
                "of using component offload. Layerwise offload remains supported.",
            )

    def prepare(self) -> None:
        _ModelOptFp8OffloadAdapter._maybe_disable_incompatible_dit_offload_modes(
            server_args=self.server_args,
            quant_config=self.quant_config,
            component_name=self.component_name,
        )


class _BitsAndBytes4BitAdapter(_TransformerQuantAdapter):
    """Adapter for pre-quantized bitsandbytes 4-bit transformer checkpoints."""

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        quant_config: Optional[QuantizationConfig],
        component_name: str | None,
    ) -> None:
        self.server_args = server_args
        self.quant_config = quant_config
        self.component_name = component_name

    @staticmethod
    def _maybe_disable_incompatible_offload_modes(
        server_args: ServerArgs,
        quant_config: Optional[QuantizationConfig],
        component_name: str | None = None,
    ) -> None:
        if _get_quant_config_name(quant_config) != "bitsandbytes":
            return

        changed = []
        component_offload = _uses_component_offload(
            server_args,
            component_name,
            legacy_enabled=bool(server_args.dit_cpu_offload),
        )
        if component_offload:
            _reject_explicit_component_selector(
                server_args,
                component_name,
                feature_name="bitsandbytes 4-bit transformer checkpoints",
            )
            if component_name is None:
                server_args.dit_cpu_offload = False
            else:
                server_args.require_component_resident(
                    component_name,
                    feature_name="bitsandbytes 4-bit transformer checkpoints",
                )
            changed.append(
                "dit_cpu_offload=False"
                if component_name is None
                else f"{component_name}=resident"
            )
        if component_name is None:
            if server_args.use_fsdp_inference:
                server_args.use_fsdp_inference = False
                changed.append("use_fsdp_inference=False")
        elif server_args.should_use_fsdp_for_component(component_name):
            server_args.disable_fsdp_for_component(component_name)
            changed.append(f"{component_name}.fsdp=False")
        if changed:
            logger.warning(
                "Keeping bitsandbytes 4-bit transformer GPU-resident: %s",
                ", ".join(changed),
            )

    def prepare(self) -> None:
        _BitsAndBytes4BitAdapter._maybe_disable_incompatible_offload_modes(
            server_args=self.server_args,
            quant_config=self.quant_config,
            component_name=self.component_name,
        )


def _validate_gguf_runtime_support(
    server_args: ServerArgs, component_name: str | None = None
) -> None:
    """Reject configurations a GGUF transformer cannot serve.

    Called before the checkpoint is downloaded or read, so an unsupported
    combination costs a second rather than a multi-gigabyte fetch.

    ``component_name`` selects the FSDP decision to check. FSDP is resolved per
    component, so a globally enabled ``--use-fsdp-inference`` does not shard a
    transformer that is offloaded; only the component actually holding the
    packed weights matters.
    """
    # The quantization comes from the file, so an explicit --quantization is
    # either redundant (gguf) or a conflicting request that would otherwise be
    # dropped without a word.
    if server_args.quantization == "gguf":
        raise ValueError(
            "GGUF is selected by passing the checkpoint itself, not "
            "`--quantization gguf`. Drop the flag; "
            "`--transformer-weights-path <file.gguf>` is what enables it."
        )
    if server_args.quantization is not None:
        raise ValueError(
            f"--quantization {server_args.quantization} cannot be combined with "
            "a GGUF transformer, whose quantization is fixed by the checkpoint. "
            "Drop the flag, or use an unquantized checkpoint to quantize online."
        )
    # Nunchaku shares --transformer-weights-path with GGUF, and the GGUF plan is
    # resolved first, so without this the SVDQuant request would be dropped in
    # silence rather than refused.
    if server_args.nunchaku_config is not None:
        raise ValueError(
            "--enable-svdquant cannot be combined with a GGUF transformer: both "
            "supply the transformer weights. Point "
            "--transformer-weights-path at either an SVDQuant checkpoint or a "
            ".gguf, not one while requesting the other."
        )
    if not current_platform.is_cuda():
        raise ValueError(
            "GGUF diffusion checkpoints require CUDA; the GGML kernels have no "
            f"{current_platform.device_type} implementation."
        )
    uses_fsdp = (
        server_args.should_use_fsdp_for_component(component_name)
        if component_name is not None
        else server_args.use_fsdp_inference
    )
    if uses_fsdp:
        raise ValueError(
            "GGUF diffusion checkpoints are incompatible with FSDP inference. "
            "Run without --use-fsdp-inference, or keep this component offloaded "
            "so FSDP does not manage it."
        )
    if server_args.lora_path is not None:
        raise ValueError(
            "LoRA is not supported on a GGUF transformer: an adapter cannot be "
            "merged into packed GGML blocks. Use the unquantized checkpoint to "
            "serve LoRA."
        )
    # H3's AdaLN paths read the transformer's safetensors directly -- the cache
    # builder needs unquantized weights, and the online rebuild is handed the
    # safetensors file list, which is empty for a GGUF load.
    if server_args.minimax_h3_adaln_online:
        raise ValueError(
            "--minimax-h3-adaln-online rebuilds AdaLN outputs from the "
            "safetensors checkpoint and cannot read a GGUF transformer."
        )
    if server_args.minimax_h3_adaln_cache_path is not None:
        raise ValueError(
            "--minimax-h3-adaln-cache-path requires the unquantized "
            "transformer and cannot be combined with a GGUF checkpoint."
        )


def resolve_transformer_gguf_to_load(
    server_args: ServerArgs, component_name: str | None = None
) -> Optional[str]:
    """Resolve ``--transformer-weights-path`` to a local ``.gguf``, if it is one.

    Returns ``None`` when the override is absent or is not GGUF, so the caller
    falls through to the safetensors path.
    """
    override = server_args.transformer_weights_path
    if not override:
        return None
    # A `~` can reach us unexpanded from a config file or a quoted argument.
    override = os.path.expanduser(override)
    if not names_gguf_checkpoint(override):
        return None

    # Before any download: a Hub reference would otherwise fetch gigabytes and
    # only then hit an unsupported-configuration error.
    _validate_gguf_runtime_support(server_args, component_name)

    is_local_reference = os.path.isabs(override) or override.startswith(".")
    resolved = (
        override
        if is_local_reference
        else resolve_hf_gguf_reference(override, revision=server_args.revision)
        or override
    )
    if not check_gguf_file(resolved):
        raise ValueError(f"Resolved GGUF path is not a GGUF file: {resolved}")
    logger.info("using GGUF transformer weights from: %s", resolved)
    return resolved


def resolve_transformer_checkpoint_files(
    server_args: ServerArgs, component_model_path: str
) -> TransformerCheckpointFiles:
    """Resolve transformer weights from the base component path or an override."""
    quantized_path = server_args.transformer_weights_path

    if quantized_path:
        resolved_set = resolve_safetensors_weight_set(
            quantized_path,
            revision=server_args.revision,
            select_unindexed_weight=_select_single_mixed_safetensors_file,
        )
        safetensors_list = materialize_weight_set(resolved_set)
        logger.info(
            "using transformer weight set from %s: %s",
            quantized_path,
            safetensors_list,
        )
        return TransformerCheckpointFiles(
            safetensors=safetensors_list,
            config_path=materialize_weight_set_config(resolved_set),
        )

    safetensors_list = _list_safetensors_files(component_model_path)
    if safetensors_list:
        # Preserve legacy cleanup for the base component. Explicit overrides
        # are resolved above, where an index is already the final authority.
        safetensors_list = filter_duplicate_safetensors_files(
            safetensors_list,
            os.path.dirname(safetensors_list[0]),
            SAFE_WEIGHTS_INDEX_NAME,
        )
        safetensors_list = _prefer_mixed_safetensors_files(safetensors_list)
        safetensors_list = _filter_duplicate_precision_variant_safetensors(
            safetensors_list
        )

    if not safetensors_list:
        raise ValueError(f"no safetensors files found in {component_model_path}")

    return TransformerCheckpointFiles(tuple(safetensors_list), None)


def _select_single_mixed_safetensors_file(
    candidates: tuple[str, ...],
) -> str | None:
    """Preserve the transformer's established mixed-export preference."""
    mixed = tuple(path for path in candidates if _MIXED_SAFETENSORS_RE.fullmatch(path))
    return mixed[0] if len(mixed) == 1 else None


def _prefer_mixed_safetensors_files(safetensors_list: list[str]) -> list[str]:
    """Prefer mixed-precision transformer exports over sibling full exports.

    Some raw ModelOpt NVFP4 repos ship both `foo-mixed.safetensors` and
    `foo.safetensors`. They are alternative full transformer exports, not
    shards, so loading both trips duplicate tensor-name validation.
    """
    mixed_files = [
        path
        for path in safetensors_list
        if _MIXED_SAFETENSORS_RE.match(os.path.basename(path))
    ]
    if not mixed_files or len(mixed_files) == len(safetensors_list):
        return safetensors_list

    logger.info(
        "Using %d mixed transformer safetensors file(s) and ignoring %d sibling "
        "non-mixed file(s): %s",
        len(mixed_files),
        len(safetensors_list) - len(mixed_files),
        mixed_files,
    )
    return mixed_files


def _filter_duplicate_precision_variant_safetensors(
    safetensors_list: list[str],
) -> list[str]:
    """Drop precision-specific duplicates when a canonical file is present.

    Diffusers checkpoints sometimes ship both `foo.safetensors` and
    `foo.fp16.safetensors` (and their sharded variants) in the same directory.
    Loading both is unsafe because duplicate parameter names race and whichever
    tensor arrives last wins, leading to non-deterministic behavior

    If a canonical unsuffixed (non bf16|fp32) file exists, prefer it and drop the precision
    variant from the same family. Precision-only families are left untouched.
    """
    canonical_paths = set(safetensors_list)
    filtered: list[str] = []
    removed: list[str] = []

    for path in safetensors_list:
        match = _PRECISION_VARIANT_SUFFIX_RE.match(path)
        if match is None:
            filtered.append(path)
            continue

        canonical_path = (
            f"{match.group('stem')}{match.group('shard') or ''}{match.group('ext')}"
        )
        if canonical_path in canonical_paths:
            removed.append(path)
            continue

        filtered.append(path)

    if removed:
        logger.info(
            "Filtered %d duplicate transformer precision variant file(s): %s",
            len(removed),
            removed,
        )

    return filtered


def resolve_transformer_quant_load_spec(
    *,
    hf_config: dict,
    server_args: ServerArgs,
    safetensors_list: list[str],
    component_model_path: str,
    model_cls: type[nn.Module],
    cls_name: str,
    component_name: str | None = None,
    gguf_file: str | None = None,
    checkpoint_quant_config: QuantizationConfig | None = None,
    transformer_override_config_path: str | None = None,
    arch_config: DiTArchConfig | None = None,
) -> TransformerQuantLoadSpec:
    if gguf_file is not None:
        if checkpoint_quant_config is not None:
            raise ValueError("GGUF and safetensors quantization metadata conflict")
        return _resolve_gguf_quant_load_spec(
            gguf_file=gguf_file,
            server_args=server_args,
            model_cls=model_cls,
            component_name=component_name,
        )

    if checkpoint_quant_config is not None:
        if server_args.quantization is not None:
            raise ValueError(
                "Checkpoint quantization is encoded in per-layer metadata; do not "
                "also set --quantization"
            )
        if server_args.nunchaku_config is not None:
            raise ValueError(
                "Per-layer checkpoint quantization and Nunchaku are mutually "
                "exclusive"
            )
        quant_config = checkpoint_quant_config
    elif getattr(model_cls, "handles_checkpoint_quantization", False):
        quant_config = None
    else:
        quant_config = _resolve_quant_config(
            hf_config=hf_config,
            server_args=server_args,
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            transformer_override_config_path=transformer_override_config_path,
            arch_config=arch_config,
        )

    if quant_config is not None:
        packed = getattr(model_cls, "packed_modules_mapping", None)
        if packed and hasattr(quant_config, "packed_modules_mapping"):
            quant_config.packed_modules_mapping = packed
        quant_config.remap_checkpoint_prefixes(
            vars(model_cls).get("param_names_mapping", {})
        )

    nunchaku_config = server_args.nunchaku_config
    if quant_config is not None and nunchaku_config is not None:
        raise ValueError(
            "Replacement checkpoint quantization and Nunchaku are mutually exclusive"
        )

    # resolve target param dtype
    param_dtype = _resolve_target_param_dtype(
        quant_config=quant_config,
        nunchaku_config=nunchaku_config,
        server_args=server_args,
    )

    adapters = _build_transformer_quant_adapters(
        cls_name=cls_name,
        server_args=server_args,
        quant_config=quant_config,
        nunchaku_config=nunchaku_config,
        model_cls=model_cls,
        safetensors_list=safetensors_list,
        component_name=component_name,
    )
    for adapter in adapters:
        adapter.prepare()

    # collect post-load hooks from built adapters
    post_load_hooks: list[PostLoadHook] = []
    for adapter in adapters:
        post_load_hooks.extend(adapter.get_post_load_hooks())

    return TransformerQuantLoadSpec(
        safetensors_list=safetensors_list,
        quant_config=quant_config,
        nunchaku_config=nunchaku_config,
        param_dtype=param_dtype,
        needs_device_weight_postprocess=_needs_device_weight_postprocess(quant_config),
        post_load_hooks=post_load_hooks,
    )


def _resolve_gguf_quant_load_spec(
    *,
    gguf_file: str,
    server_args: ServerArgs,
    model_cls: type[nn.Module],
    component_name: str | None = None,
) -> TransformerQuantLoadSpec:
    """Build the load plan for a GGUF transformer checkpoint."""
    from sglang.multimodal_gen.runtime.layers.quantization.gguf import GGUFConfig

    _validate_gguf_runtime_support(server_args, component_name)

    quant_config = GGUFConfig(
        gguf_file=gguf_file,
        tensor_meta=read_gguf_tensor_meta(gguf_file),
    )
    packed = getattr(model_cls, "packed_modules_mapping", None)
    if packed:
        quant_config.packed_modules_mapping = packed

    return TransformerQuantLoadSpec(
        safetensors_list=[],
        quant_config=quant_config,
        nunchaku_config=None,
        # No single dtype for the load: each parameter keeps the dtype the model
        # declared for it, which the generic loader casts to. Packed weights are
        # registered uint8, so that cast is a no-op for them. Note this matches
        # every other quant path -- _resolve_target_param_dtype returns None
        # whenever a quant_config is present.
        param_dtype=None,
        gguf_file=gguf_file,
    )


def _needs_device_weight_postprocess(
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Return whether post-load weight processing needs CUDA/NPU tensors."""
    quant_name = _get_quant_config_name(quant_config)
    if quant_name in ("modelopt_fp8", "comfy_fp8", "auto-round", "mxfp8"):
        return True
    if quant_name == "kitchen_int8":
        assert isinstance(quant_config, KitchenInt8Config)
        return not quant_config.is_checkpoint_int8_serialized

    serialized_flag_by_quant_name = {
        "fp8": "is_checkpoint_fp8_serialized",
        "mxfp4": "is_checkpoint_mxfp4_serialized",
        "mxfp4_npu": "is_checkpoint_mxfp4_npu_serialized",
    }
    serialized_flag = serialized_flag_by_quant_name.get(quant_name)
    if serialized_flag is None:
        return False
    return not getattr(quant_config, serialized_flag, False)


def _build_transformer_quant_adapters(
    *,
    cls_name: str,
    server_args: ServerArgs,
    quant_config: Optional[QuantizationConfig],
    nunchaku_config: Optional[NunchakuConfig],
    model_cls: type[nn.Module],
    safetensors_list: list[str],
    component_name: str | None,
) -> list[_TransformerQuantAdapter]:
    adapters: list[_TransformerQuantAdapter] = [
        _Flux2Nvfp4FallbackAdapter(
            cls_name=cls_name,
            server_args=server_args,
            quant_config=quant_config,
            component_name=component_name,
        ),
        _ModelOptFp8OffloadAdapter(
            server_args=server_args,
            quant_config=quant_config,
            component_name=component_name,
        ),
        _BitsAndBytes4BitAdapter(
            server_args=server_args,
            quant_config=quant_config,
            component_name=component_name,
        ),
    ]
    if nunchaku_config is not None:
        adapters.append(
            _NunchakuQuantAdapter(
                nunchaku_config=nunchaku_config,
                model_cls=model_cls,
                safetensors_list=safetensors_list,
            )
        )
    return adapters


def _merge_quant_declaration(base: dict, incoming: dict) -> dict:
    """Merge compatible checkpoint declarations and reject conflicts."""
    merged = dict(base)
    for key, value in incoming.items():
        previous = merged.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            merged[key] = _merge_quant_declaration(previous, value)
        elif key in merged and previous != value:
            raise ValueError(f"Conflicting checkpoint quantization field {key!r}")
        else:
            merged[key] = value
    return merged


def _resolve_weight_override_quantization(
    safetensors_list: list[str],
    reverse_param_names_mapping: dict,
    quant_ignore_remap: dict,
) -> tuple[Optional[QuantizationConfig], bool]:
    """Resolve declarations carried by the materialized replacement weight set."""
    component_model_path = os.path.dirname(safetensors_list[0])
    component_config = {}
    component_config_path = os.path.join(component_model_path, "config.json")
    if os.path.isfile(component_config_path):
        with open(component_config_path, encoding="utf-8") as config_stream:
            component_config = json.load(config_stream)

    config_spec = resolve_checkpoint_quant_spec(component_config)
    declaration = config_spec.config if config_spec is not None else None
    header_quant_config = None
    detected_quantized_tensors = False

    for safetensors_file in safetensors_list:
        metadata = get_metadata_from_safetensors_file(safetensors_file) or {}
        file_quant_config = get_quant_config_from_safetensors_metadata(safetensors_file)
        if file_quant_config is not None:
            if header_quant_config is not None and _get_quant_config_name(
                header_quant_config
            ) != _get_quant_config_name(file_quant_config):
                raise ValueError("Conflicting safetensors quantization declarations")
            header_quant_config = file_quant_config

        for metadata_key in ("_quantization_metadata", "quantization_config"):
            serialized = metadata.get(metadata_key)
            if serialized is None:
                continue
            try:
                metadata_config = json.loads(serialized)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid {metadata_key} in {safetensors_file}"
                ) from error
            if not isinstance(metadata_config, dict):
                raise ValueError(
                    f"Invalid {metadata_key} in {safetensors_file}: expected an object"
                )
            metadata_spec = resolve_checkpoint_quant_spec(
                {"quantization_config": metadata_config}
            )
            assert metadata_spec is not None
            declaration = (
                metadata_spec.config
                if declaration is None
                else _merge_quant_declaration(declaration, metadata_spec.config)
            )

        with safe_open(safetensors_file, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key.endswith((".weight_scale", ".input_scale", ".comfy_quant")):
                    detected_quantized_tensors = True
                    break
                if key.endswith(".weight") and checkpoint.get_slice(
                    key
                ).get_dtype() in ("F8_E4M3", "I8", "U8"):
                    detected_quantized_tensors = True
                    break

    if declaration is not None:
        if "quant_method" not in declaration:
            return header_quant_config, True
        return (
            get_quant_config(
                {"quantization_config": declaration},
                component_model_path,
                reverse_param_names_mapping=reverse_param_names_mapping,
                quant_ignore_remap=quant_ignore_remap,
            ),
            True,
        )
    if header_quant_config is not None:
        return header_quant_config, True

    description_config = get_quant_config(
        component_config,
        component_model_path,
        reverse_param_names_mapping=reverse_param_names_mapping,
        quant_ignore_remap=quant_ignore_remap,
    )
    return (
        description_config,
        detected_quantized_tensors or description_config is not None,
    )


def _resolve_quant_config_from_transformer_override(
    override_config_path: str,
) -> Optional[QuantizationConfig]:
    """Resolve quant config from an override transformer repo or directory."""
    with open(override_config_path, encoding="utf-8") as f:
        override_hf_config = json.load(f)

    return get_quant_config(
        override_hf_config,
        os.path.dirname(override_config_path),
    )


def _resolve_quant_config(
    *,
    hf_config: dict,
    server_args: ServerArgs,
    safetensors_list: list[str],
    component_model_path: str,
    transformer_override_config_path: str | None = None,
    arch_config: DiTArchConfig | None = None,
) -> Optional[QuantizationConfig]:
    """
    resolve quant config from checkpoints' metadata
    priority: explicit --quantization flag -> model config.json -> safetensors metadata -> format-specific fallback
    """
    if arch_config is None:
        arch_config = server_args.pipeline_config.dit_config.arch_config
    param_names_mapping_dict = arch_config.param_names_mapping
    reverse_param_names_mapping_dict = arch_config.reverse_param_names_mapping
    quant_ignore_remap_dict = arch_config.quant_ignore_remap

    override_quant_config = None
    override_declares_quantization = False
    if server_args.transformer_weights_path:
        (
            override_quant_config,
            override_declares_quantization,
        ) = _resolve_weight_override_quantization(
            safetensors_list,
            reverse_param_names_mapping_dict,
            quant_ignore_remap_dict,
        )

    # priority: explicit --quantization flag (e.g. mxfp8, mxfp4_npu, modelslim)
    if server_args.quantization is not None:
        if override_declares_quantization:
            raise ValueError(
                "The replacement checkpoint already contains or declares "
                "quantization; do not also set an online --quantization override"
            )
        from sglang.multimodal_gen.runtime.layers.quantization import (
            get_quantization_config,
        )

        # modelslim requires a per-layer quant description file; load it from
        # the component directory rather than constructing an empty config.
        if server_args.quantization == "modelslim":
            return get_quant_config(hf_config, component_model_path)

        # GGUF is selected by pointing at the file, not by this flag: the config
        # has to be built from that file's header.
        if server_args.quantization == "gguf":
            raise ValueError(
                "GGUF is selected by passing the checkpoint itself, not "
                "`--quantization gguf`. Use "
                "`--transformer-weights-path <file.gguf>` (or a Hub reference "
                "such as owner/repo:Q4_K_M)."
            )

        # Online-quant convention: for `fp8`, `mxfp4` and `kitchen_int8`, a
        # no-arg QuantizationConfig() selects the post-load path -- weights
        # load in source dtype and are quantized in
        # process_weights_after_loading.
        quant_cls = get_quantization_config(server_args.quantization)
        quant_kwargs = {}
        if server_args.quantization in {"fp8", "mxfp4", "kitchen_int8"}:
            quant_kwargs["ignored_layers"] = getattr(
                server_args, "quantization_ignored_layers", None
            )
        return quant_cls(**quant_kwargs)

    quant_config = (
        override_quant_config
        if server_args.transformer_weights_path
        else get_quant_config(
            hf_config,
            component_model_path,
            reverse_param_names_mapping=reverse_param_names_mapping_dict,
            quant_ignore_remap=quant_ignore_remap_dict,
        )
    )
    quant_config_name = _get_quant_config_name(quant_config)
    inferred_nvfp4_config = None
    if quant_config is None or quant_config_name == "modelopt_fp4":
        fallback_group_size = None
        if quant_config_name == "modelopt_fp4":
            fallback_group_size = getattr(quant_config, "group_size", None)
        inferred_nvfp4_config = build_nvfp4_config_from_safetensors_list(
            safetensors_list,
            param_names_mapping_dict,
            reverse_param_names_mapping_dict,
            fallback_group_size,
        )
    if override_declares_quantization and override_quant_config is None:
        if inferred_nvfp4_config is None:
            raise ValueError(
                "Replacement checkpoint contains quantized tensors but no supported "
                "native quantization declaration"
            )
        quant_config = inferred_nvfp4_config
    else:
        quant_config = _merge_modelopt_fp4_configs(quant_config, inferred_nvfp4_config)
    if quant_config is not None or transformer_override_config_path is None:
        return quant_config

    quant_config = _resolve_quant_config_from_transformer_override(
        transformer_override_config_path,
    )
    quant_config = _merge_modelopt_fp4_configs(quant_config, inferred_nvfp4_config)
    if quant_config is not None:
        return quant_config

    for safetensors_file in safetensors_list:
        quant_config = get_quant_config_from_safetensors_metadata(safetensors_file)
        if quant_config is not None:
            return quant_config

    return inferred_nvfp4_config


def _resolve_target_param_dtype(
    *,
    quant_config: Optional[QuantizationConfig],
    nunchaku_config: Optional[NunchakuConfig],
    server_args: ServerArgs,
) -> Optional[torch.dtype]:
    if quant_config is not None or nunchaku_config is not None:
        return None
    return resolve_precision(server_args, "dit", precision_attr="dit_precision")
