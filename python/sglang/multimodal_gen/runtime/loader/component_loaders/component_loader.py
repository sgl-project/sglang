# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import pkgutil
import traceback
from abc import ABC
from collections.abc import Callable, Iterator
from typing import Any, Type

import torch
import transformers
from diffusers import AutoModel
from torch import nn
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.quantizers import AutoHfQuantizer

from sglang.multimodal_gen.configs.models.base import ModelConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    ComponentAttentionBackendNotAppliedError,
    component_attn_backend_context_manager,
    get_component_attn_backend_context,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    _normalize_component_type,
    component_name_to_loader_cls,
    finalize_loaded_model,
    format_component_residency,
    get_memory_usage_of_component,
    get_param_names_mapping,
    hf_to_custom_state_dict,
    initialize_model,
    load_model_state_dict,
)
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    checkpoint_weights_iterator,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    RESIDENT,
    ComponentResidencyError,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
    get_hf_config,
    prepare_diffusers_component_path_for_loading,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    resolve_component_precision,
    resolve_precision,
)
from sglang.multimodal_gen.runtime.weights.source import (
    materialize_weight,
    resolve_weight,
)
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)

logger = init_logger(__name__)


class ComponentCheckpointUnsupportedError(ValueError):
    """A component checkpoint is unsupported and must not use native fallback."""


class NativeComponentLoaderRequired(RuntimeError):
    """The customized loader must defer to the native library loader."""


def uses_native_transformers_quantization(config: object, component_name: str) -> bool:
    """Validate quantization metadata that Transformers can restore itself."""
    try:
        quant_spec = resolve_checkpoint_quant_spec(config)
    except (TypeError, ValueError) as error:
        raise ComponentCheckpointUnsupportedError(
            f"Cannot parse checkpoint quantization for {component_name!r}: {error}"
        ) from error
    if quant_spec is None:
        return False
    if quant_spec.source != "quantization_config":
        raise ComponentCheckpointUnsupportedError(
            f"Transformers-managed {component_name!r} quantization requires "
            "a top-level quantization_config; "
            f"got metadata from {quant_spec.source!r}"
        )

    try:
        supported = AutoHfQuantizer.supports_quant_method(dict(quant_spec.config))
    except (TypeError, ValueError) as error:
        raise ComponentCheckpointUnsupportedError(
            f"Cannot configure Transformers-managed quantization for "
            f"{component_name!r}: {error}"
        ) from error
    if not supported:
        method = quant_spec.declared_method or "unspecified"
        raise ComponentCheckpointUnsupportedError(
            f"Transformers does not support quant_method={method!r} declared by "
            f"{component_name!r}"
        )
    return True


def _load_auto_tokenizer_with_roberta_processing_compat(*args, **kwargs):
    from tokenizers import processors

    roberta_processing = processors.RobertaProcessing

    def roberta_processing_compat(*processor_args, **processor_kwargs):
        if "sep" in processor_kwargs and "cls" in processor_kwargs:
            sep = processor_kwargs.pop("sep")
            cls_token = processor_kwargs.pop("cls")
            return roberta_processing(
                sep, cls_token, *processor_args, **processor_kwargs
            )
        return roberta_processing(*processor_args, **processor_kwargs)

    processors.RobertaProcessing = roberta_processing_compat
    try:
        return AutoTokenizer.from_pretrained(*args, **kwargs)
    finally:
        processors.RobertaProcessing = roberta_processing


class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""

    # the list of possible name of the component in model_index.json, e.g., scheduler
    component_names: list[str] = []

    # diffusers or transformers
    expected_library: str = ""

    _loaders_registered = False

    def __init_subclass__(cls, **kwargs):
        """
        register loaders, called when subclass is imported
        """
        super().__init_subclass__(**kwargs)
        for component_name in cls.component_names:
            component_name_to_loader_cls[component_name] = cls

    def __init__(self, device=None) -> None:
        self.device = device
        self.component_architecture: str | None = None
        self.component_type: str | None = None
        self._native_load_manages_placement = False

    def structural_component_name(self, component_name: str) -> str:
        """Return the config slot without changing the exact policy key."""
        return self.component_type or component_name

    def structural_component_type(self, component_name: str) -> str:
        """Return the normalized loader role for an exact component key."""
        return _normalize_component_type(self.structural_component_name(component_name))

    @staticmethod
    def target_device(component_starts_on_cpu: bool) -> torch.device:
        if component_starts_on_cpu:
            return (
                torch.device("mps")
                if current_platform.is_mps()
                else torch.device("cpu")
            )
        return get_local_torch_device()

    def customized_load_kwargs_for_component(
        self, _server_args: ServerArgs, _component_name: str
    ) -> dict[str, Any]:
        return {}

    def component_load_precision(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        """Return an exact precision override or reject an unsupported one."""
        precision = server_args.component_precisions.get(component_name)
        if precision is not None:
            raise ComponentCheckpointUnsupportedError(
                f"{component_name!r} does not support an exact component precision "
                "override"
            )
        return None

    def resolve_component_weight_override(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        """Return the consumed weights-only override or reject it."""
        override = server_args.component_weights_paths.get(component_name)
        if override is not None:
            raise ComponentCheckpointUnsupportedError(
                f"{component_name!r} does not support a weights-only override; "
                f"use --component-paths.{component_name} to replace its config "
                "and weights together"
            )
        return None

    def resolve_component_quantization_override(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        """Return the consumed online quantization override or reject it."""
        quantization = server_args.component_quantizations.get(component_name)
        if quantization is not None:
            raise ComponentCheckpointUnsupportedError(
                f"{component_name!r} does not support an explicit quantization "
                "override; use a self-describing quantized component checkpoint "
                "when supported"
            )
        return None

    def resolve_component_direct_gpu_loading(
        self, server_args: ServerArgs, component_name: str
    ) -> bool:
        """Return whether this load consumes the exact direct-GPU request."""
        requested = server_args.should_direct_gpu_weight_load_component(component_name)
        if requested:
            raise ComponentCheckpointUnsupportedError(
                f"{component_name!r} does not support direct GPU weight loading"
            )
        return False

    def component_attention_backend_context(
        self,
        attn_backend: Any,
        component_attn_name: str | None,
        require_backend_selection: bool,
    ):
        """Build the attention-selection context used by this loader."""
        return component_attn_backend_context_manager(
            attn_backend,
            component_name=component_attn_name,
            allow_global_backend_fallback=True,
            require_backend_selection=require_backend_selection,
        )

    def is_native_only_component(
        self, server_args: ServerArgs, component_name: str
    ) -> bool:
        native_only_components = server_args.pipeline_config.native_only_components
        return any(
            name in native_only_components
            for name in (
                component_name,
                self.structural_component_name(component_name),
                self.structural_component_type(component_name),
            )
        )

    def should_raise_customized_load_error(
        self, server_args: ServerArgs, component_name: str
    ) -> bool:
        return self.is_native_only_component(server_args, component_name)

    def validate_native_fallback(
        self, _server_args: ServerArgs, _component_name: str
    ) -> None:
        """Validate that fallback preserves the exact component's runtime contract."""
        pass

    def _load_customized_with_context(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        attn_backend: Any,
        component_attn_name: str | None,
        require_backend_selection: bool,
    ) -> AutoModel:
        with self.component_attention_backend_context(
            attn_backend,
            component_attn_name,
            require_backend_selection,
        ):
            load_kwargs = self.customized_load_kwargs_for_component(
                server_args, component_name
            )
            return self.load_customized(
                component_model_path, server_args, component_name, **load_kwargs
            )

    def _load_native_with_context(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
        attn_backend: Any,
        component_attn_name: str | None,
        require_backend_selection: bool,
    ) -> AutoModel:
        with self.component_attention_backend_context(
            attn_backend,
            component_attn_name,
            require_backend_selection,
        ):
            component = self.load_native(
                component_model_path,
                server_args,
                transformers_or_diffusers,
                component_name,
            )
        return component

    def load(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str,
        *,
        component_attn_backend: Any = None,
        component_attn_name: str | None = None,
    ) -> tuple[AutoModel, float]:
        """
        Template method that standardizes logging around the core load implementation.
        The priority of loading method is:
            1. load customized component
            2. load native diffusers/transformers component
        If all of the above methods failed, an error will be thrown

        """
        self._native_load_manages_placement = False
        self.component_load_precision(server_args, component_name)
        component_weight_override = self.resolve_component_weight_override(
            server_args, component_name
        )
        self.resolve_component_quantization_override(server_args, component_name)
        self.resolve_component_direct_gpu_loading(server_args, component_name)
        fsdp_requested = server_args.should_use_fsdp_for_component(component_name)

        gpu_mem_before_loading = current_platform.get_available_gpu_memory()
        logger.info(
            "Loading %s from %s. avail mem: %.2f GB",
            component_name,
            component_model_path,
            gpu_mem_before_loading,
        )
        if (
            component_attn_backend is None
            and component_attn_name is None
            and get_component_attn_backend_context() is None
        ):
            component_attn_backend, matched_backend_key = (
                server_args.resolve_component_attention_backend(component_name)
            )
            component_attn_name = matched_backend_key or component_name
            if component_attn_backend is not None:
                logger.info(
                    "Using %s backend for component: %s",
                    component_attn_backend.name.lower(),
                    matched_backend_key,
                )
        requested_backend = (
            server_args.requested_component_attention_backend(component_attn_name)
            if component_attn_name is not None
            else None
        )
        require_backend_selection = requested_backend is not None
        if require_backend_selection and (
            component_attn_backend is None
            or component_attn_backend.name.lower() != requested_backend
        ):
            raise ValueError(
                f"Component attention backend for {component_attn_name!r} no longer "
                f"matches the explicit request {requested_backend!r}"
            )
        try:
            component = self._load_customized_with_context(
                component_model_path,
                server_args,
                component_name,
                component_attn_backend,
                component_attn_name,
                require_backend_selection,
            )
            source = "sgl-diffusion"
        except (
            ComponentAttentionBackendNotAppliedError,
            ComponentCheckpointUnsupportedError,
            ComponentResidencyError,
        ):
            raise
        except Exception as e:
            if require_backend_selection:
                raise
            native_loader_required = isinstance(e, NativeComponentLoaderRequired)
            if native_loader_required and component_weight_override is not None:
                raise ComponentCheckpointUnsupportedError(
                    f"{component_name!r} requires its library loader, which cannot "
                    "consume a weights-only override; use "
                    f"--component-paths.{component_name} to replace its config "
                    "and weights together"
                ) from e
            if (
                component_weight_override is not None
                or self.should_raise_customized_load_error(server_args, component_name)
            ):
                if native_loader_required:
                    raise
                if component_weight_override is not None:
                    raise RuntimeError(
                        f"Failed to load the weights-only override for "
                        f"{component_name!r}; fallback would ignore it. Use "
                        f"--component-paths.{component_name} when the checkpoint "
                        "also requires a different config or library loader."
                    ) from e
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to load customized {component_name}; native fallback "
                    "is disabled for this component configuration."
                ) from e
            self.validate_native_fallback(server_args, component_name)
            if native_loader_required:
                logger.info("%s", e)
            elif "Unsupported model architecture" in str(e):
                logger.info(
                    f"Component: {component_name} doesn't have a customized version yet, using native version"
                )
            else:
                traceback.print_exc()
                logger.error(
                    f"Error while loading customized {component_name}, falling back to native version"
                )
            # fallback to native version
            component = self._load_native_with_context(
                component_model_path,
                server_args,
                component_name,
                transformers_or_diffusers,
                component_attn_backend,
                component_attn_name,
                require_backend_selection,
            )
            source = "native"
            logger.warning(
                "Native component %s: %s is loaded, performance may be sub-optimal",
                component_name,
                component.__class__.__name__,
            )

        if component is None:
            logger.error("Load %s failed", component_name)
            consumed = 0.0
        else:
            if fsdp_requested and (
                not isinstance(component, nn.Module)
                or not is_fsdp_managed_module(component)
            ):
                # The returned module is the source of truth. Loaders do not need
                # a parallel capability declaration for FSDP support.
                server_args.disable_fsdp_for_component(component_name)
            if isinstance(component, nn.Module):
                component = finalize_loaded_model(component)
                if (
                    not is_fsdp_managed_module(component)
                    and not self._native_load_manages_placement
                ):
                    component = component.to(
                        self.target_device(
                            server_args.should_start_component_on_cpu(component_name)
                        )
                    )
            current_gpu_mem = current_platform.get_available_gpu_memory()
            model_size = get_memory_usage_of_component(component) or "NA"
            consumed = gpu_mem_before_loading - current_gpu_mem
            logger.info(
                f"Loaded %s: %s ({source} version). model size: %s GB, %s. avail GPU mem: %.2f GB",
                component_name,
                component.__class__.__name__,
                model_size,
                format_component_residency(component),
                current_gpu_mem,
            )
        return component, consumed

    def load_native(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str,
        component_name: str | None = None,
    ) -> AutoModel:
        """
        Load the component using the native library (transformers/diffusers).
        """
        precision = None
        if component_name is not None:
            precision_names = dict.fromkeys(
                (
                    component_name,
                    self.structural_component_name(component_name),
                    self.structural_component_type(component_name),
                )
            )
            for precision_name in precision_names:
                precision = resolve_component_precision(server_args, precision_name)
                if precision is not None:
                    break
        load_kwargs = {}
        if precision is not None:
            load_kwargs["torch_dtype"] = precision

        if transformers_or_diffusers == "transformers":
            self._native_load_manages_placement = False
            config = get_hf_config(
                component_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            if uses_native_transformers_quantization(
                config, component_name or "component"
            ):
                resolved_component_name = component_name or "component"
                explicit_residency = server_args.explicit_residency_mode(
                    resolved_component_name
                )
                if explicit_residency is not None and explicit_residency != RESIDENT:
                    raise ComponentCheckpointUnsupportedError(
                        "Transformers-managed quantized component "
                        f"{resolved_component_name!r} requires resident placement; "
                        f"got explicit mode {explicit_residency!r}"
                    )
                server_args.require_component_resident(
                    resolved_component_name,
                    feature_name="Transformers quantized component",
                )
                load_kwargs["device_map"] = {
                    "": self.target_device(component_starts_on_cpu=False)
                }
                self._native_load_manages_placement = True
            model_class = self.resolve_native_transformers_model_class(config)
            return model_class.from_pretrained(
                component_model_path,
                config=config,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                **load_kwargs,
            )
        elif transformers_or_diffusers == "diffusers":
            from diffusers import AutoModel

            component_model_path = prepare_diffusers_component_path_for_loading(
                component_model_path
            )
            return AutoModel.from_pretrained(
                component_model_path,
                revision=server_args.revision,
                trust_remote_code=server_args.trust_remote_code,
                **load_kwargs,
            )
        else:
            raise ValueError(f"Unsupported library: {transformers_or_diffusers}")

    def resolve_native_transformers_model_class(self, config: PretrainedConfig) -> type:
        return transformers.AutoModel

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """
        Load the customized version component, implemented and optimized in SGL-diffusion
        """
        raise NotImplementedError(
            f"load_customized not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def _ensure_loaders_registered(cls):
        """
        avoid multiple registration
        """
        if cls._loaders_registered:
            return

        package_dir = os.path.dirname(__file__)
        package_name = (
            __package__
            or "sglang.multimodal_gen.runtime.loader.component_loaders.component_loaders"
        )

        for _, name, _ in pkgutil.iter_modules([package_dir]):
            # skip importing self to avoid circular dependency issues
            if name == "component_loader":
                continue
            try:
                importlib.import_module(f".{name}", package=package_name)
            except ImportError as e:
                logger.warning(f"Failed to import loader component {name}: {e}")

        cls._loaders_registered = True

    @classmethod
    def resolve_transformers_or_diffusers(
        self, transformers_or_diffusers: str, component_name: str
    ) -> str:
        # NOTE(FlamingoPg): special for LTX-2 models
        # `model_index.json` records these under an `ltx2` library that is not a
        # real importable package; SGLang implements them natively.
        if component_name in (
            "vocoder",
            "connectors",
            "duration_head",
            "diffusion_decoder",
        ):
            transformers_or_diffusers = "diffusers"

        # NOTE(CloudRipple): special for MOVA models
        # TODO(CloudRipple): remove most of these special cases after unifying the loading logic
        if component_name in [
            "audio_vae",
            "audio_dit",
            "dual_tower_bridge",
            "video_dit",
        ]:
            transformers_or_diffusers = "diffusers"

        if (
            component_name == "scheduler"
            and transformers_or_diffusers == "mova.diffusion.schedulers.flow_match_pair"
        ):
            transformers_or_diffusers = "diffusers"

        if transformers_or_diffusers.startswith("lingbot_video"):
            transformers_or_diffusers = "diffusers"

        return transformers_or_diffusers

    @classmethod
    def for_component_type(
        cls,
        component_type: str,
        transformers_or_diffusers: str,
        component_architecture: str | None = None,
    ) -> "ComponentLoader":
        """
        Factory method to create a component loader for a specific component type.

        Args:
            component_type: Structural role (e.g. "vae" or "text_encoder")
            transformers_or_diffusers: Whether the component is from transformers or diffusers
        """
        cls._ensure_loaders_registered()

        # Map of component types to their loader classes and expected library
        structural_component_name = component_type
        loader_type = _normalize_component_type(component_type)

        transformers_or_diffusers = cls.resolve_transformers_or_diffusers(
            transformers_or_diffusers, loader_type
        )

        if loader_type in component_name_to_loader_cls:
            loader_cls: Type[ComponentLoader] = component_name_to_loader_cls[
                loader_type
            ]
            expected_library = loader_cls.expected_library
            # Assert that the library matches what's expected for this component type
            assert transformers_or_diffusers == expected_library, (
                f"{loader_type} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            )
            loader = loader_cls()
            loader.component_type = structural_component_name
            loader.component_architecture = component_architecture
            return loader

        # For unknown component types, use a generic loader
        logger.warning(
            "No specific loader found for component type: %s. Using generic loader.",
            loader_type,
        )
        loader = GenericComponentLoader(
            transformers_or_diffusers, component_architecture
        )
        loader.component_type = structural_component_name
        return loader


class WeightOverrideComponentLoader(ComponentLoader):
    """Base for loaders that consume an exact weights-only override."""

    ignored_checkpoint_prefixes: tuple[str, ...] = ()

    def load_state_dict_model(
        self,
        model_cls: type[nn.Module],
        init_params: dict[str, Any],
        weight_files: list[str],
        server_args: ServerArgs,
        component_name: str,
        dtype: torch.dtype,
        *,
        component_starts_on_cpu: bool,
        weight_load_plan: WeightLoadPlan | None = None,
        checkpoint_key_filter: Callable[[str], bool] | None = None,
        weights_iterator: Iterator[tuple[str, torch.Tensor]] | None = None,
    ) -> nn.Module:
        """Restore mapped model state with optional TP/FSDP materialization."""
        return maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params=init_params,
            weight_dir_list=weight_files,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            component_starts_on_cpu=component_starts_on_cpu,
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.should_use_fsdp_for_component(component_name),
            param_dtype=dtype,
            reduce_dtype=torch.float32,
            strict=False,
            weight_load_plan=weight_load_plan,
            checkpoint_key_filter=checkpoint_key_filter,
            weights_iterator=weights_iterator,
        )

    def validate_checkpoint_keys(
        self, missing: list[str] | set[str], unexpected: list[str], component_name: str
    ) -> None:
        unexpected = [
            name
            for name in unexpected
            if not name.startswith(self.ignored_checkpoint_prefixes)
        ]
        if missing or unexpected:
            raise ComponentCheckpointUnsupportedError(
                f"Checkpoint weights do not match {component_name!r}. "
                f"Missing: {sorted(missing)}. Unexpected: {sorted(unexpected)}."
            )

    def resolve_component_weight_override(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        return server_args.component_weights_paths.get(component_name)

    def validate_component_weight_override(self, _override: str) -> None:
        pass

    def resolve_component_weights_path(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
    ) -> str:
        override = self.resolve_component_weight_override(server_args, component_name)
        if override is None:
            return component_model_path
        self.validate_component_weight_override(override)
        weights_path = materialize_weight(resolve_weight(override))
        logger.info("Using weight override for %s: %s", component_name, weights_path)
        return weights_path


class OnlineQuantizationComponentLoader(WeightOverrideComponentLoader):
    """Base for loaders that also consume an online quantization override."""

    def resolve_component_quantization_override(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        return server_args.component_quantizations.get(component_name)


class PlainStateDictComponentLoader(WeightOverrideComponentLoader):
    """Construct registered modules and restore a complete plain state dict."""

    expected_library = "diffusers"
    config_classes: dict[str, type[ModelConfig]] = {}
    default_precision_attr = "dit_precision"
    default_dtype = torch.bfloat16

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> nn.Module:
        config = self.load_component_config(component_model_path, component_name)
        class_name = config.pop("_class_name", None) or self.component_architecture
        if class_name is None:
            raise ComponentCheckpointUnsupportedError(
                f"{component_name!r} must declare _class_name in config.json "
                "or its architecture in model_index.json"
            )
        weights_path = self.resolve_component_weights_path(
            component_model_path, server_args, component_name
        )
        model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
        model_config = self.build_model_config(config, component_name)
        dtype = self.resolve_dtype(server_args, component_name)
        component_starts_on_cpu = server_args.should_start_component_on_cpu(
            component_name
        )
        server_args.model_paths[component_name] = component_model_path
        if issubclass(model_cls, BaseDiT):
            weight_files = _list_safetensors_files(weights_path)
            return self.load_state_dict_model(
                model_cls,
                {"config": model_config, "hf_config": config},
                weight_files,
                server_args,
                component_name,
                dtype,
                component_starts_on_cpu=component_starts_on_cpu,
                weights_iterator=(
                    None if weight_files else checkpoint_weights_iterator(weights_path)
                ),
            )

        target_device = self.target_device(component_starts_on_cpu)
        model = initialize_model(
            model_cls,
            model_config
            if isinstance(model_config, dict)
            else {"config": model_config},
            dtype,
        ).to(target_device)

        try:
            state_dict, _ = hf_to_custom_state_dict(
                checkpoint_weights_iterator(weights_path),
                get_param_names_mapping(
                    model_config.arch_config.param_names_mapping
                    if isinstance(model_config, ModelConfig)
                    else {}
                ),
                valid_target_names=set(model.state_dict()),
                strict=True,
            )
            missing, unexpected = load_model_state_dict(model, state_dict, strict=False)
        except (RuntimeError, ValueError) as error:
            raise ComponentCheckpointUnsupportedError(
                f"Cannot restore checkpoint for {component_name!r}: {error}"
            ) from error
        self.validate_checkpoint_keys(missing, unexpected, component_name)
        return model

    def build_model_config(
        self, config: dict[str, Any], component_name: str
    ) -> ModelConfig | dict[str, Any]:
        config_cls = self.config_classes.get(
            self.structural_component_type(component_name)
        )
        if config_cls is not None:
            model_config = config_cls()
            model_config.update_model_arch(config)
            return model_config
        return {key: value for key, value in config.items() if not key.startswith("_")}

    def resolve_dtype(
        self, server_args: ServerArgs, component_name: str
    ) -> torch.dtype:
        try:
            return resolve_precision(
                server_args, component_name, precision_attr=self.default_precision_attr
            )
        except AttributeError:
            return self.default_dtype

    def component_load_precision(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        return server_args.component_precisions.get(component_name)

    @staticmethod
    def ensure_plain_state_dict_checkpoint(config: object, component_name: str) -> None:
        try:
            quant_spec = resolve_checkpoint_quant_spec(config)
        except (TypeError, ValueError) as error:
            raise ComponentCheckpointUnsupportedError(
                f"Cannot parse checkpoint quantization metadata for "
                f"{component_name!r}: {error}"
            ) from error
        if quant_spec is None:
            return

        method = quant_spec.declared_method or "unspecified"
        raise ComponentCheckpointUnsupportedError(
            f"{component_name!r} checkpoint declares quantization metadata in "
            f"{quant_spec.source} (quant_method={method!r}), which its current "
            "plain state-dict materializer cannot restore."
        )

    def load_component_config(
        self, component_model_path: str, component_name: str
    ) -> dict[str, Any]:
        config = get_diffusers_component_config(component_path=component_model_path)
        self.ensure_plain_state_dict_checkpoint(config, component_name)
        return config


class ImageProcessorLoader(ComponentLoader):
    """Loader for image processor."""

    component_names = ["image_processor"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        return AutoImageProcessor.from_pretrained(
            component_model_path, backend="torchvision"
        )


class AutoProcessorLoader(ComponentLoader):
    """Loader for auto processor."""

    component_names = ["processor", "text_processor"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        return AutoProcessor.from_pretrained(component_model_path)


class TokenizerLoader(ComponentLoader):
    """Loader for tokenizers."""

    component_names = ["tokenizer", "text_tokenizer"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        # Some pipelines keep the slot name `tokenizer` in model_index.json even
        # when the declared class is a processor. e.g. FLUX.2:
        # `tokenizer: ["transformers", "PixtralProcessor"]`.
        # Honor the declared component class instead of guessing from the slot name.
        if (
            self.component_architecture is not None
            and self.component_architecture.endswith("Processor")
        ):
            return AutoProcessor.from_pretrained(component_model_path)

        # Qwen-Image's model_index declares Qwen2Tokenizer; using the fast class
        # changes text preprocessing and shifts official GT comparisons.
        use_fast = self.component_architecture != "Qwen2Tokenizer"
        try:
            return AutoTokenizer.from_pretrained(
                component_model_path,
                padding_side="right",
                use_fast=use_fast,
            )
        except TypeError as e:
            # tokenizers>=0.21 removed the `cls` kwarg from RobertaProcessing,
            # but some transformers CLIPTokenizer builds still pass it. Fall back
            # to the pure-Python (slow) tokenizer which avoids the rust path.
            if "RobertaProcessing" in str(e) and use_fast:
                logger.warning(
                    "Fast tokenizer failed (%s), retrying with use_fast=False", e
                )
                return _load_auto_tokenizer_with_roberta_processing_compat(
                    component_model_path,
                    padding_side="right",
                    use_fast=False,
                )
            raise


class GenericComponentLoader(ComponentLoader):
    """Generic loader for components that don't have a specific loader."""

    def __init__(
        self, library="transformers", component_architecture: str | None = None
    ) -> None:
        super().__init__()
        self.library = library
        self.component_architecture = component_architecture

    def component_attention_backend_context(
        self,
        attn_backend: Any,
        component_attn_name: str | None,
        require_backend_selection: bool,
    ):
        # An unknown out-of-tree component may itself be the primary transformer.
        # Require it to opt into fallback through a registered component loader.
        return component_attn_backend_context_manager(
            attn_backend,
            component_name=component_attn_name,
            allow_global_backend_fallback=False,
            require_backend_selection=require_backend_selection,
        )


class PipelineComponentLoader:
    """
    Utility class for loading the components in a pipeline.
    """

    @staticmethod
    def load_component(
        component_name: str,
        component_model_path: str,
        transformers_or_diffusers: str,
        server_args: ServerArgs,
        component_architecture: str | None = None,
        component_attn_backend: Any = None,
        component_attn_name: str | None = None,
        component_type: str | None = None,
    ):
        """
        Load a pipeline component.

        Args:
            component_name: Name of the component (e.g., "vae", "text_encoder", "transformer", "scheduler")
            component_model_path: Path to the component model
            transformers_or_diffusers: Whether the component is from transformers or diffusers
            component_architecture: the class name of the module
            component_type: structural config slot when it differs from the exact key
        """

        # Get the appropriate loader for this component type
        loader = ComponentLoader.for_component_type(
            component_type or component_name,
            transformers_or_diffusers,
            component_architecture,
        )

        try:
            return loader.load(
                component_model_path,
                server_args,
                component_name,
                transformers_or_diffusers,
                component_attn_backend=component_attn_backend,
                component_attn_name=component_attn_name,
            )
        except Exception:
            logger.error(
                f"Error while loading component: {component_name}, {component_model_path=}"
            )
            raise
