# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import traceback
from abc import ABC
from typing import Any

import torch
from diffusers import AutoModel
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer

from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.adapter_loader import AdapterLoader
from sglang.multimodal_gen.runtime.loader.bridge_loader import BridgeLoader
from sglang.multimodal_gen.runtime.loader.image_encoder_loader import ImageEncoderLoader
from sglang.multimodal_gen.runtime.loader.scheduler_loader import SchedulerLoader
from sglang.multimodal_gen.runtime.loader.text_encoder_loader import TextEncoderLoader
from sglang.multimodal_gen.runtime.loader.transformer_loader import TransformerLoader
from sglang.multimodal_gen.runtime.loader.utils import (
    _normalize_module_type,
    get_memory_usage_of_component,
)
from sglang.multimodal_gen.runtime.loader.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.loader.vl_encoder_loader import (
    VisionLanguageEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.vocoder_loader import VocoderLoader
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import get_hf_config
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""

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
        module_name: str,
        transformers_or_diffusers: str,
    ) -> tuple[AutoModel, float]:
        """
        Template method that standardizes logging around the core load implementation.
        The priority of loading method is:
            1. load customized module
            2. load native diffusers/transformers module
        If all of the above methods failed, an error will be thrown

        """
        gpu_mem_before_loading = current_platform.get_available_gpu_memory()
        logger.info(
            "Loading %s from %s. avail mem: %.2f GB",
            module_name,
            component_model_path,
            gpu_mem_before_loading,
        )
        try:
            component = self.load_customized(
                component_model_path, server_args, module_name
            )
            source = "sgl-diffusion"
        except Exception as e:
            if "Unsupported model architecture" in str(e):
                logger.info(
                    f"Module: {module_name} doesn't have a customized version yet, using native version"
                )
            else:
                traceback.print_exc()
                logger.error(
                    f"Error while loading customized {module_name}, falling back to native version"
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
                "Native module %s: %s is loaded, performance may be sub-optimal",
                module_name,
                component.__class__.__name__,
            )

        if component is None:
            logger.warning("Loaded %s returned None", module_name)
            consumed = 0.0
        else:
            current_gpu_mem = current_platform.get_available_gpu_memory()
            consumed = get_memory_usage_of_component(component)
            if consumed is None or consumed == 0.0:
                consumed = gpu_mem_before_loading - current_gpu_mem
            logger.info(
                f"Loaded %s: %s ({source} version). model size: %.2f GB, avail mem: %.2f GB",
                module_name,
                component.__class__.__name__,
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
        self, component_model_path: str, server_args: ServerArgs, module_name: str
    ):
        """
        Load the customized version component, implemented and optimized in SGL-diffusion
        """
        raise NotImplementedError(
            f"load_customized not implemented for {self.__class__.__name__}"
        )

    @classmethod
    def for_module_type(
        cls, module_type: str, transformers_or_diffusers: str
    ) -> "ComponentLoader":
        """
        Factory method to create a component loader for a specific module type.

        Args:
            module_type: Type of module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            transformers_or_diffusers: Whether the module is from transformers or diffusers
        """
        # Map of module types to their loader classes and expected library
        module_type = _normalize_module_type(module_type)
        module_loaders = {
            "scheduler": (SchedulerLoader, "diffusers"),
            "transformer": (TransformerLoader, "diffusers"),
            "vae": (VAELoader, "diffusers"),
            "text_encoder": (TextEncoderLoader, "transformers"),
            "tokenizer": (TokenizerLoader, "transformers"),
            "image_processor": (ImageProcessorLoader, "transformers"),
            "image_encoder": (ImageEncoderLoader, "transformers"),
            "processor": (AutoProcessorLoader, "transformers"),
            "vision_language_encoder": (VisionLanguageEncoderLoader, "transformers"),
        }
        # Loaders for audio/video specific components that might vary
        av_module_loaders = {
            "audio_dit": (TransformerLoader, "diffusers"),
            "audio_vae": (VAELoader, "diffusers"),
            "connectors": (AdapterLoader, "diffusers"),
            "dual_tower_bridge": (BridgeLoader, "diffusers"),
            "video_dit": (TransformerLoader, "diffusers"),
            "video_vae": (VAELoader, "diffusers"),
            "vocoder": (VocoderLoader, "diffusers"),
        }

        # NOTE(FlamingoPg): special for LTX-2 models
        if module_type == "vocoder" or module_type == "connectors":
            transformers_or_diffusers = "diffusers"

        # NOTE(CloudRipple): special for MOVA models
        # TODO(CloudRipple): remove most of these special cases after unifying the loading logic
        if module_type in [
            "audio_vae",
            "audio_dit",
            "dual_tower_bridge",
            "video_dit",
        ]:
            transformers_or_diffusers = "diffusers"

        if module_type in module_loaders:
            loader_cls, expected_library = module_loaders[module_type]
            # Assert that the library matches what's expected for this module type
            assert (
                transformers_or_diffusers == expected_library
            ), f"{module_type} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            return loader_cls()

        if module_type in av_module_loaders:
            loader_cls, expected_library = av_module_loaders[module_type]
            if transformers_or_diffusers == expected_library:
                return loader_cls()

        # For unknown module types, use a generic loader
        logger.warning(
            "No specific loader found for module type: %s. Using generic loader.",
            module_type,
        )
        return GenericComponentLoader(transformers_or_diffusers)


class ImageProcessorLoader(ComponentLoader):
    """Loader for image processor."""

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, module_name: str
    ) -> Any:
        return AutoImageProcessor.from_pretrained(component_model_path, use_fast=True)


class AutoProcessorLoader(ComponentLoader):
    """Loader for auto processor."""

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, module_name: str
    ) -> Any:
        return AutoProcessor.from_pretrained(component_model_path)


class TokenizerLoader(ComponentLoader):
    """Loader for tokenizers."""

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, module_name: str
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
    Utility class for loading pipeline components.
    This replaces the chain of if-else statements in load_pipeline_module.
    """

    @staticmethod
    def load_module(
        module_name: str,
        component_model_path: str,
        transformers_or_diffusers: str,
        server_args: ServerArgs,
    ):
        """
        Load a pipeline module.

        Args:
            module_name: Name of the module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            component_model_path: Path to the component model
            transformers_or_diffusers: Whether the module is from transformers or diffusers

        """

        # Get the appropriate loader for this module type
        loader = ComponentLoader.for_module_type(module_name, transformers_or_diffusers)

        try:
            # Load the module
            return loader.load(
                component_model_path,
                server_args,
                module_name,
                transformers_or_diffusers,
            )
        except Exception as e:
            logger.error(
                f"Error while loading component: {module_name}, {component_model_path=}"
            )
            raise e
