# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import pkgutil
import traceback
from abc import ABC
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

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    component_attn_backend_context_manager,
    get_component_attn_backend_context,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _normalize_component_type,
    component_name_to_loader_cls,
    format_component_residency,
    get_memory_usage_of_component,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    ComponentResidencyError,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    is_fsdp_managed_module,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
    get_hf_config,
    prepare_diffusers_component_path_for_loading,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_component_precision
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


def uses_native_transformers_bnb4(config: object, component_name: str) -> bool:
    """Validate a serialized BnB4 checkpoint owned by Transformers."""
    try:
        quant_spec = resolve_checkpoint_quant_spec(config)
    except (TypeError, ValueError) as error:
        raise ComponentCheckpointUnsupportedError(
            f"Cannot parse checkpoint quantization for {component_name!r}: {error}"
        ) from error
    if quant_spec is None or quant_spec.declared_method != "bitsandbytes":
        return False
    if quant_spec.source != "quantization_config":
        raise ComponentCheckpointUnsupportedError(
            f"Transformers-managed {component_name!r} quantization requires "
            "a top-level quantization_config; "
            f"got metadata from {quant_spec.source!r}"
        )

    load_in_4bit = quant_spec.config.get(
        "load_in_4bit", quant_spec.config.get("_load_in_4bit")
    )
    load_in_8bit = quant_spec.config.get(
        "load_in_8bit", quant_spec.config.get("_load_in_8bit", False)
    )
    if load_in_4bit is not True or load_in_8bit is True:
        raise ComponentCheckpointUnsupportedError(
            f"Transformers-managed {component_name!r} quantization supports only "
            "serialized BitsAndBytes 4-bit checkpoints"
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

    # --attention-backend primarily selects the DiT backend. Auxiliary
    # components may fall back when that global choice is incompatible; an
    # explicit --component-attention-backends entry remains strict.
    allow_global_attention_backend_fallback = True
    # Gates only --component-quantizations.<name>. Quantization declared by a
    # checkpoint is discovered and admitted by the component's normal loader.
    supports_online_quantization_override = False

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

    def should_raise_customized_load_error(
        self, server_args: ServerArgs, component_name: str
    ) -> bool:
        native_only_components = getattr(
            server_args.pipeline_config, "native_only_components", ()
        )
        return component_name in native_only_components

    def _load_customized_with_context(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        attn_backend: Any,
        component_attn_name: str | None,
        allow_global_backend_fallback: bool,
    ) -> AutoModel:
        with component_attn_backend_context_manager(
            attn_backend,
            component_name=component_attn_name,
            allow_global_backend_fallback=allow_global_backend_fallback,
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
        allow_global_backend_fallback: bool,
    ) -> AutoModel:
        with component_attn_backend_context_manager(
            attn_backend,
            component_name=component_attn_name,
            allow_global_backend_fallback=allow_global_backend_fallback,
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
    ) -> tuple[AutoModel, float]:
        """
        Template method that standardizes logging around the core load implementation.
        The priority of loading method is:
            1. load customized component
            2. load native diffusers/transformers component
        If all of the above methods failed, an error will be thrown

        """
        component_quantization = server_args.component_quantizations.get(component_name)
        if (
            component_quantization is not None
            and not self.supports_online_quantization_override
        ):
            raise ValueError(
                f"{component_name!r} does not support an explicit quantization "
                "override; "
                "use a self-describing quantized component checkpoint when supported"
            )

        gpu_mem_before_loading = current_platform.get_available_gpu_memory()
        logger.info(
            "Loading %s from %s. avail mem: %.2f GB",
            component_name,
            component_model_path,
            gpu_mem_before_loading,
        )
        attn_backend = None
        component_attn_name = None
        if get_component_attn_backend_context() is None:
            attn_backend, matched_backend_key = (
                server_args.resolve_component_attention_backend(component_name)
            )
            component_attn_name = matched_backend_key or component_name
            if attn_backend is not None:
                logger.info(
                    "Using %s backend for component: %s",
                    attn_backend.name.lower(),
                    matched_backend_key,
                )
        try:
            component = self._load_customized_with_context(
                component_model_path,
                server_args,
                component_name,
                attn_backend,
                component_attn_name,
                self.allow_global_attention_backend_fallback,
            )
            source = "sgl-diffusion"
        except (ComponentCheckpointUnsupportedError, ComponentResidencyError):
            raise
        except Exception as e:
            native_loader_required = isinstance(e, NativeComponentLoaderRequired)
            if self.should_raise_customized_load_error(server_args, component_name):
                if native_loader_required:
                    raise
                traceback.print_exc()
                raise RuntimeError(
                    f"Failed to load customized {component_name}; native fallback "
                    "is disabled for this component configuration."
                ) from e
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
                attn_backend,
                component_attn_name,
                self.allow_global_attention_backend_fallback,
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
            if isinstance(component, nn.Module):
                component = component.eval()
                if not is_fsdp_managed_module(component):
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
        precision = (
            resolve_component_precision(server_args, component_name)
            if component_name is not None
            else None
        )
        load_kwargs = {}
        if precision is not None:
            load_kwargs["torch_dtype"] = precision

        if transformers_or_diffusers == "transformers":
            config = get_hf_config(
                component_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            if uses_native_transformers_bnb4(config, component_name or "component"):
                server_args.require_component_resident(
                    component_name or "component",
                    feature_name="Transformers bitsandbytes component",
                )
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
        component_name: str,
        transformers_or_diffusers: str,
        component_architecture: str | None = None,
    ) -> "ComponentLoader":
        """
        Factory method to create a component loader for a specific component type.

        Args:
            component_name: Type of component (e.g., "vae", "text_encoder", "transformer", "scheduler")
            transformers_or_diffusers: Whether the component is from transformers or diffusers
        """
        cls._ensure_loaders_registered()

        # Map of component types to their loader classes and expected library
        component_name = _normalize_component_type(component_name)

        transformers_or_diffusers = cls.resolve_transformers_or_diffusers(
            transformers_or_diffusers, component_name
        )

        if component_name in component_name_to_loader_cls:
            loader_cls: Type[ComponentLoader] = component_name_to_loader_cls[
                component_name
            ]
            expected_library = loader_cls.expected_library
            # Assert that the library matches what's expected for this component type
            assert (
                transformers_or_diffusers == expected_library
            ), f"{component_name} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            loader = loader_cls()
            loader.component_architecture = component_architecture
            return loader

        # For unknown component types, use a generic loader
        logger.warning(
            "No specific loader found for component type: %s. Using generic loader.",
            component_name,
        )
        return GenericComponentLoader(transformers_or_diffusers, component_architecture)


class PlainStateDictComponentLoader(ComponentLoader):
    """Base for native loaders whose current materializer expects plain weights."""

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

    def resolve_component_weights_path(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
    ) -> str:
        override = server_args.component_weights_paths.get(component_name)
        if override is None:
            return component_model_path
        weights_path = materialize_weight(resolve_weight(override))
        logger.info("Using weight override for %s: %s", component_name, weights_path)
        return weights_path


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

    # An unknown out-of-tree component may itself be the primary transformer.
    # Require it to opt into fallback through a registered component loader.
    allow_global_attention_backend_fallback = False

    def __init__(
        self, library="transformers", component_architecture: str | None = None
    ) -> None:
        super().__init__()
        self.library = library
        self.component_architecture = component_architecture


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
    ):
        """
        Load a pipeline component.

        Args:
            component_name: Name of the component (e.g., "vae", "text_encoder", "transformer", "scheduler")
            component_model_path: Path to the component model
            transformers_or_diffusers: Whether the component is from transformers or diffusers
            component_architecture: the class name of the module
        """

        # Get the appropriate loader for this component type
        loader = ComponentLoader.for_component_type(
            component_name, transformers_or_diffusers, component_architecture
        )

        try:
            with component_attn_backend_context_manager(
                component_attn_backend,
                component_name=component_attn_name,
                allow_global_backend_fallback=(
                    loader.allow_global_attention_backend_fallback
                ),
            ):
                return loader.load(
                    component_model_path,
                    server_args,
                    component_name,
                    transformers_or_diffusers,
                )
        except Exception:
            logger.error(
                f"Error while loading component: {component_name}, {component_model_path=}"
            )
            raise
