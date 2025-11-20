# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import glob
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from copy import deepcopy
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from torch.distributed import init_device_mesh
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from sglang.multimodal_gen.configs.models import EncoderConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    maybe_load_fsdp_model,
    shard_model,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_config,
    get_diffusers_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""

    def __init__(self, device=None) -> None:
        self.device = device

    @abstractmethod
    def load(self, model_path: str, server_args: ServerArgs, module_name: str):
        """
        Load the component based on the model path, architecture, and inference args.

        Args:
            model_path: Path to the component model
            server_args: ServerArgs

        Returns:
            The loaded component
        """
        raise NotImplementedError

    @classmethod
    def for_module_type(
        cls, module_type: str, transformers_or_diffusers: str
    ) -> "ComponentLoader":
        """
        Factory method to create a component loader for a specific module type.

        Args:
            module_type: Type of module (e.g., "vae", "text_encoder", "transformer", "scheduler")
            transformers_or_diffusers: Whether the module is from transformers or diffusers

        Returns:
            A component loader for the specified module type
        """
        # Map of module types to their loader classes and expected library
        module_loaders = {
            "scheduler": (SchedulerLoader, "diffusers"),
            "transformer": (TransformerLoader, "diffusers"),
            "transformer_2": (TransformerLoader, "diffusers"),
            "vae": (VAELoader, "diffusers"),
            "text_encoder": (TextEncoderLoader, "transformers"),
            "text_encoder_2": (TextEncoderLoader, "transformers"),
            "tokenizer": (TokenizerLoader, "transformers"),
            "tokenizer_2": (TokenizerLoader, "transformers"),
            "image_processor": (ImageProcessorLoader, "transformers"),
            "image_encoder": (ImageEncoderLoader, "transformers"),
            "processor": (AutoProcessorLoader, "transformers"),
        }

        if module_type in module_loaders:
            loader_cls, expected_library = module_loaders[module_type]
            # Assert that the library matches what's expected for this module type
            assert (
                transformers_or_diffusers == expected_library
            ), f"{module_type} must be loaded from {expected_library}, got {transformers_or_diffusers}"
            return loader_cls()

        # For unknown module types, use a generic loader
        logger.warning(
            "No specific loader found for module type: %s. Using generic loader.",
            module_type,
        )
        return GenericComponentLoader(transformers_or_diffusers)


class TextEncoderLoader(ComponentLoader):
    """Loader for text encoders."""

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    counter_before_loading_weights: float = 0.0
    counter_after_loading_weights: float = 0.0

    def _prepare_weights(
        self,
        model_name_or_path: str,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        # model_name_or_path = (self._maybe_download_from_modelscope(
        #     model_name_or_path, revision) or model_name_or_path)

        is_local = os.path.isdir(model_name_or_path)
        assert is_local, "Model path must be a local directory"

        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        allow_patterns = ["*.safetensors", "*.bin"]

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder, index_file
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, source: "Source", to_cpu: bool
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
        )
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files, to_cpu=to_cpu
            )
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files, to_cpu=to_cpu)

        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        # Apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model: nn.Module,
        model_path: str,
        to_cpu: bool,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        primary_weights = TextEncoderLoader.Source(
            model_path,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(primary_weights, to_cpu)

        secondary_weights = cast(
            Iterable[TextEncoderLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(source, to_cpu)

    def load(self, model_path: str, server_args: ServerArgs, module_name: str):
        """Load the text encoders based on the model path, and inference args."""
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=server_args.trust_remote_code,
        #     revision=server_args.revision,
        #     model_override_args=None,
        # )
        diffusers_pretrained_config = get_config(model_path, trust_remote_code=True)
        model_config = get_diffusers_config(model=model_path)
        model_config.pop("_name_or_path", None)
        model_config.pop("transformers_version", None)
        model_config.pop("model_type", None)
        model_config.pop("tokenizer_class", None)
        model_config.pop("torch_dtype", None)
        logger.info("HF model config: %s", model_config)

        def is_not_first_encoder(module_name):
            return "2" in module_name

        # TODO(mick): had to throw an exception for different text-encoder arch
        if not is_not_first_encoder(module_name):
            encoder_config = server_args.pipeline_config.text_encoder_configs[0]
            encoder_config.update_model_arch(model_config)
            for key, value in diffusers_pretrained_config.__dict__.items():
                setattr(encoder_config.arch_config, key, value)
            encoder_dtype = server_args.pipeline_config.text_encoder_precisions[0]
        else:
            assert len(server_args.pipeline_config.text_encoder_configs) == 2
            encoder_config = server_args.pipeline_config.text_encoder_configs[1]
            encoder_config.update_model_arch(model_config)
            encoder_dtype = server_args.pipeline_config.text_encoder_precisions[1]
        target_device = get_local_torch_device()
        # TODO(will): add support for other dtypes
        return self.load_model(
            model_path,
            encoder_config,
            target_device,
            server_args,
            encoder_dtype,
        )

    def load_model(
        self,
        model_path: str,
        model_config: EncoderConfig,
        target_device: torch.device,
        server_args: ServerArgs,
        dtype: str = "fp16",
    ):
        use_cpu_offload = (
            server_args.text_encoder_cpu_offload
            and len(getattr(model_config, "_fsdp_shard_conditions", [])) > 0
        )

        if server_args.text_encoder_cpu_offload:
            target_device = (
                torch.device("mps")
                if current_platform.is_mps()
                else torch.device("cpu")
            )

        with set_default_torch_dtype(PRECISION_TO_TYPE[dtype]):
            with target_device:
                architectures = getattr(model_config, "architectures", [])
                model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
                model = model_cls(model_config)

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self._get_all_weights(model, model_path, to_cpu=use_cpu_offload)
            )
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights
                - self.counter_before_loading_weights,
            )

            # Explicitly move model to target device after loading weights
            model = model.to(target_device)

            if use_cpu_offload:
                # Disable FSDP for MPS as it's not compatible
                if current_platform.is_mps():
                    logger.info(
                        "Disabling FSDP sharding for MPS platform as it's not compatible"
                    )
                else:
                    mesh = init_device_mesh(
                        "cuda",
                        mesh_shape=(1, dist.get_world_size()),
                        mesh_dim_names=("offload", "replicate"),
                    )
                    shard_model(
                        model,
                        cpu_offload=True,
                        reshard_after_forward=True,
                        mesh=mesh["offload"],
                        fsdp_shard_conditions=model._fsdp_shard_conditions,
                        pin_cpu_memory=server_args.pin_cpu_memory,
                    )
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            # if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )

        return model.eval()


class ImageEncoderLoader(TextEncoderLoader):

    def load(self, model_path: str, server_args: ServerArgs, *args):
        """Load the text encoders based on the model path, and inference args."""
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=server_args.trust_remote_code,
        #     revision=server_args.revision,
        #     model_override_args=None,
        # )
        with open(os.path.join(model_path, "config.json")) as f:
            model_config = json.load(f)
        model_config.pop("_name_or_path", None)
        model_config.pop("transformers_version", None)
        model_config.pop("torch_dtype", None)
        model_config.pop("model_type", None)
        logger.info("HF model config: %s", model_config)

        encoder_config = server_args.pipeline_config.image_encoder_config
        encoder_config.update_model_arch(model_config)

        if server_args.image_encoder_cpu_offload:
            target_device = (
                torch.device("mps")
                if current_platform.is_mps()
                else torch.device("cpu")
            )
        else:
            target_device = get_local_torch_device()
        # TODO(will): add support for other dtypes
        return self.load_model(
            model_path,
            encoder_config,
            target_device,
            server_args,
            server_args.pipeline_config.image_encoder_precision,
        )


class ImageProcessorLoader(ComponentLoader):
    """Loader for image processor."""

    def load(self, model_path: str, server_args: ServerArgs, *args):
        """Load the image processor based on the model path, and inference args."""
        logger.info("Loading image processor from %s", model_path)

        image_processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        logger.info("Loaded image processor: %s", image_processor.__class__.__name__)
        return image_processor


class AutoProcessorLoader(ComponentLoader):
    """Loader for auto processor."""

    def load(self, model_path: str, server_args: ServerArgs, *args):
        """Load the image processor based on the model path, and inference args."""
        logger.info("Loading auto processor from %s", model_path)

        processor = AutoProcessor.from_pretrained(
            model_path,
        )
        logger.info("Loaded auto processor: %s", processor.__class__.__name__)
        return processor


class TokenizerLoader(ComponentLoader):
    """Loader for tokenizers."""

    def load(self, model_path: str, server_args: ServerArgs, *args):
        """Load the tokenizer based on the model path, and inference args."""
        logger.info("Loading tokenizer from %s", model_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,  # "<path to model>/tokenizer"
            # in v0, this was same string as encoder_name "ClipTextModel"
            # TODO(will): pass these tokenizer kwargs from inference args? Maybe
            # other method of config?
            padding_size="right",
        )
        logger.info("Loaded tokenizer: %s", tokenizer.__class__.__name__)
        return tokenizer


class VAELoader(ComponentLoader):
    """Loader for VAE."""

    def load(self, model_path: str, server_args: ServerArgs, *args):
        """Load the VAE based on the model path, and inference args."""
        config = get_diffusers_config(model=model_path)
        class_name = config.pop("_class_name")
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        server_args.model_paths["vae"] = model_path

        # TODO: abstract these logics
        logger.info("HF model config: %s", config)
        vae_config = server_args.pipeline_config.vae_config
        vae_config.update_model_arch(config)

        # NOTE: some post init logics are only available after updated with config
        vae_config.post_init()

        if server_args.vae_cpu_offload:
            target_device = (
                torch.device("mps")
                if current_platform.is_mps()
                else torch.device("cpu")
            )
        else:
            target_device = get_local_torch_device()

        with set_default_torch_dtype(
            PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        ):
            vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vae = vae_cls(vae_config).to(target_device)

        # Find all safetensors files
        safetensors_list = glob.glob(os.path.join(str(model_path), "*.safetensors"))
        # TODO(PY)
        assert (
            len(safetensors_list) == 1
        ), f"Found {len(safetensors_list)} safetensors files in {model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(
            loaded, strict=False
        )  # We might only load encoder or decoder

        return vae.eval()


class TransformerLoader(ComponentLoader):
    """Loader for transformer."""

    def load(self, model_path: str, server_args: ServerArgs, *args):
        """Load the transformer based on the model path, and inference args."""
        config = get_diffusers_config(model=model_path)
        hf_config = deepcopy(config)
        cls_name = config.pop("_class_name")
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        logger.info("transformer cls_name: %s", cls_name)
        if server_args.override_transformer_cls_name is not None:
            cls_name = server_args.override_transformer_cls_name
            logger.info("Overriding transformer cls_name to %s", cls_name)

        server_args.model_paths["transformer"] = model_path

        # Config from Diffusers supersedes sgl_diffusion's model config
        dit_config = server_args.pipeline_config.dit_config
        dit_config.update_model_arch(config)

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Find all safetensors files
        safetensors_list = glob.glob(os.path.join(str(model_path), "*.safetensors"))
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {model_path}")

        # Check if we should use custom initialization weights
        custom_weights_path = getattr(
            server_args, "init_weights_from_safetensors", None
        )
        use_custom_weights = False

        if use_custom_weights:
            logger.info(
                "Using custom initialization weights from: %s", custom_weights_path
            )
            assert (
                custom_weights_path is not None
            ), "Custom initialization weights must be provided"
            if os.path.isdir(custom_weights_path):
                safetensors_list = glob.glob(
                    os.path.join(str(custom_weights_path), "*.safetensors")
                )
            else:
                assert custom_weights_path.endswith(
                    ".safetensors"
                ), "Custom initialization weights must be a safetensors file"
                safetensors_list = [custom_weights_path]

        logger.info(
            "Loading model from %s safetensors files: %s",
            len(safetensors_list),
            safetensors_list,
        )

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        # Load the model using FSDP loader
        logger.info("Loading %s, default_dtype: %s", cls_name, default_dtype)
        assert server_args.hsdp_shard_dim is not None
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={"config": dit_config, "hf_config": hf_config},
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=server_args.dit_cpu_offload,
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.use_fsdp_inference,
            # TODO(will): make these configurable
            default_dtype=default_dtype,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        assert (
            next(model.parameters()).dtype == default_dtype
        ), "Model dtype does not match default dtype"

        model = model.eval()
        return model


class SchedulerLoader(ComponentLoader):
    """Loader for scheduler."""

    def load(self, model_path: str, server_args: ServerArgs, *args):
        """Load the scheduler based on the model path, and inference args."""
        config = get_diffusers_config(model=model_path)

        class_name = config.pop("_class_name")
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        scheduler_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        scheduler = scheduler_cls(**config)
        if server_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(server_args.pipeline_config.flow_shift)
        if server_args.pipeline_config.timesteps_scale is not None:
            scheduler.set_timesteps_scale(server_args.pipeline_config.timesteps_scale)
        return scheduler


class GenericComponentLoader(ComponentLoader):
    """Generic loader for components that don't have a specific loader."""

    def __init__(self, library="transformers") -> None:
        super().__init__()
        self.library = library

    def load(self, model_path: str, server_args: ServerArgs, *args):
        """Load a generic component based on the model path, and inference args."""
        logger.warning(
            "Using generic loader for %s with library %s", model_path, self.library
        )

        if self.library == "transformers":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            logger.info(
                "Loaded generic transformers model: %s", model.__class__.__name__
            )
            return model
        elif self.library == "diffusers":
            logger.warning(
                "Generic loading for diffusers components is not fully implemented"
            )

            model_config = get_diffusers_config(model=model_path)
            logger.info("Diffusers Model config: %s", model_config)
            # This is a placeholder - in a real implementation, you'd need to handle this properly
            return None
        else:
            raise ValueError(f"Unsupported library: {self.library}")


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

        Returns:
            The loaded module
        """
        logger.info(
            "Loading %s using %s from %s",
            module_name,
            transformers_or_diffusers,
            component_model_path,
        )

        # Get the appropriate loader for this module type
        loader = ComponentLoader.for_module_type(module_name, transformers_or_diffusers)

        try:
            # Load the module
            return loader.load(component_model_path, server_args, module_name)
        except Exception as e:
            logger.error(
                f"Error while loading component: {module_name}, {component_model_path=}"
            )
            raise e
