# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import importlib
import os
import pkgutil
import traceback
from abc import ABC
from typing import Any, Type

import torch
from diffusers import AutoModel
from torch import nn
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.utils import (
    _normalize_component_type,
    component_name_to_loader_cls,
    get_memory_usage_of_component,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import get_hf_config
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ):
        # not offload by default
        return False

    def target_device(self, should_offload):
        if should_offload:
            return (
                torch.device("mps")
                if current_platform.is_mps()
                else torch.device("cpu")
            )
        else:
            return get_local_torch_device()

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
        gpu_mem_before_loading = current_platform.get_available_gpu_memory()
        logger.info(
            "Loading %s from %s. avail mem: %.2f GB",
            component_name,
            component_model_path,
            gpu_mem_before_loading,
        )
        try:
            component = self.load_customized(
                component_model_path, server_args, component_name
            )
            source = "sgl-diffusion"
        except Exception as e:
            if "Unsupported model architecture" in str(e):
                logger.info(
                    f"Component: {component_name} doesn't have a customized version yet, using native version"
                )
            else:
                traceback.print_exc()
                logger.error(
                    f"Error while loading customized {component_name}, falling back to native version"
                )
            # fallback to native version
            component = self.load_native(
                component_model_path, server_args, transformers_or_diffusers
            )
            should_offload = self.should_offload(server_args)
            target_device = self.target_device(should_offload)
            component = component.to(device=target_device)
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
            current_gpu_mem = current_platform.get_available_gpu_memory()
            model_size = get_memory_usage_of_component(component) or "NA"
            consumed = gpu_mem_before_loading - current_gpu_mem
            logger.info(
                f"Loaded %s: %s ({source} version). model size: %s GB, consumed GPU mem: %.2f GB, avail GPU mem: %.2f GB",
                component_name,
                component.__class__.__name__,
                model_size,
                consumed,
                current_gpu_mem,
            )
        return component, consumed

    def load_native(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str,
    ) -> AutoModel:
        """
        Load the component using the native library (transformers/diffusers).
        """
        if transformers_or_diffusers == "transformers":
            from transformers import AutoModel

            config = get_hf_config(
                component_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            return AutoModel.from_pretrained(
                component_model_path,
                config=config,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
        elif transformers_or_diffusers == "diffusers":
            from diffusers import AutoModel

            return AutoModel.from_pretrained(
                component_model_path,
                revision=server_args.revision,
                trust_remote_code=server_args.trust_remote_code,
            )
        else:
            raise ValueError(f"Unsupported library: {transformers_or_diffusers}")

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
    def for_component_type(
        cls, component_name: str, transformers_or_diffusers: str
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

        # NOTE(FlamingoPg): special for LTX-2 models
        if component_name == "vocoder" or component_name == "connectors":
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

        if component_name in component_name_to_loader_cls:
            loader_cls: Type[ComponentLoader] = component_name_to_loader_cls[
                component_name
            ]
            expected_library = loader_cls.expected_library
            # Assert that the library matches what's expected for this component type
            assert (
                transformers_or_diffusers == expected_library
            ), f"{component_name} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            return loader_cls()

        # For unknown component types, use a generic loader
        logger.warning(
            "No specific loader found for component type: %s. Using generic loader.",
            component_name,
        )
        return GenericComponentLoader(transformers_or_diffusers)


class ImageProcessorLoader(ComponentLoader):
    """Loader for image processor."""

    component_names = ["image_processor"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        return AutoImageProcessor.from_pretrained(component_model_path, use_fast=True)


class AutoProcessorLoader(ComponentLoader):
    """Loader for auto processor."""

    component_names = ["processor"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        return AutoProcessor.from_pretrained(component_model_path)


class TokenizerLoader(ComponentLoader):
    """Loader for tokenizers."""

    component_names = ["tokenizer"]
    expected_library = "transformers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ) -> Any:
        return AutoTokenizer.from_pretrained(
            component_model_path,
            padding_size="right",
        )


class GenericComponentLoader(ComponentLoader):
    """Generic loader for components that don't have a specific loader."""

    def __init__(self, library="transformers") -> None:
        super().__init__()
        self.library = library


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
    ):
        """
        Load a pipeline component.

        Args:
            component_name: Name of the component (e.g., "vae", "text_encoder", "transformer", "scheduler")
            component_model_path: Path to the component model
            transformers_or_diffusers: Whether the component is from transformers or diffusers

        """

        # Get the appropriate loader for this component type
        loader = ComponentLoader.for_component_type(
            component_name, transformers_or_diffusers
        )

        try:
            # Load the component
            return loader.load(
                component_model_path,
                server_args,
                component_name,
                transformers_or_diffusers,
            )
        except Exception as e:
            logger.error(
                f"Error while loading component: {component_name}, {component_model_path=}"
            )
            raise e
