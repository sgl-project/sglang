# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import dataclasses
import glob
import importlib.util
import json
import os
import time
import traceback
from abc import ABC
from collections.abc import Generator, Iterable
from copy import deepcopy
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from torch.distributed import init_device_mesh
from transformers import AutoImageProcessor, AutoProcessor, AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from sglang.multimodal_gen.configs.models import EncoderConfig, ModelConfig
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
    get_diffusers_component_config,
    get_hf_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class skip_init_modules:
    def __enter__(self):
        # Save originals
        self._orig_reset = {}
        for cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
            self._orig_reset[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self: None  # skip init

    def __exit__(self, exc_type, exc_value, traceback):
        # restore originals
        for cls, orig in self._orig_reset.items():
            cls.reset_parameters = orig


def _normalize_module_type(module_type: str) -> str:
    """Normalize module types like 'text_encoder_2' -> 'text_encoder'."""
    if module_type.endswith("_2"):
        return module_type[:-2]
    return module_type


def _clean_hf_config_inplace(model_config: dict) -> None:
    """Remove common extraneous HF fields if present."""
    for key in (
        "_name_or_path",
        "transformers_version",
        "model_type",
        "tokenizer_class",
        "torch_dtype",
    ):
        model_config.pop(key, None)


def _list_safetensors_files(model_path: str) -> list[str]:
    """List all .safetensors files under a directory."""
    return sorted(glob.glob(os.path.join(str(model_path), "*.safetensors")))


def load_native(library, component_module_path: str, server_args: ServerArgs):
    if library == "transformers":
        from transformers import AutoModel

        config = get_hf_config(
            component_module_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
        return AutoModel.from_pretrained(
            component_module_path,
            config=config,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
        )
    elif library == "diffusers":
        from diffusers import AutoModel

        return AutoModel.from_pretrained(
            component_module_path,
            revision=server_args.revision,
            trust_remote_code=server_args.trust_remote_code,
        )
    else:
        raise ValueError(f"Unsupported library: {library}")


class ComponentLoader(ABC):
    """Base class for loading a specific type of model component."""

    def __init__(self, device=None) -> None:
        self.device = device

    def should_offload(self, server_args, model_config: ModelConfig | None = None):
        # offload by default
        return True

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
    ):
        """
        Template method that standardizes logging around the core load implementation.
        The priority of loading method is:
            1. load customized module
            2. load native diffusers/transformers module
        If all of the above methods failed, an error will be thrown

        """
        logger.info("Loading %s from %s", module_name, component_model_path)
        try:
            component = self.load_customized(
                component_model_path, server_args, module_name
            )
            source = "customized"
        except Exception as _e:
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
        else:
            logger.info(
                f"Loaded %s: %s from: {source}",
                module_name,
                component.__class__.__name__,
            )
        return component

    def load_native(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str,
    ):
        """
        Load the component using the native library (transformers/diffusers).
        """
        return load_native(transformers_or_diffusers, component_model_path, server_args)

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

        Returns:
            A component loader for the specified module type
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

    def should_offload(self, server_args, model_config: ModelConfig | None = None):
        should_offload = server_args.text_encoder_cpu_offload
        fsdp_shard_conditions = getattr(model_config, "_fsdp_shard_conditions", [])
        use_cpu_offload = should_offload and len(fsdp_shard_conditions) > 0
        return use_cpu_offload

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
        """get an iterator for the model weights based on the load format."""
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
        # apply the prefix.
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

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, module_name: str
    ):
        """Load the text encoders based on the model path, and inference args."""
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=server_args.trust_remote_code,
        #     revision=server_args.revision,
        #     model_override_args=None,
        # )
        diffusers_pretrained_config = get_config(
            component_model_path, trust_remote_code=True
        )
        model_config = get_diffusers_component_config(model_path=component_model_path)
        _clean_hf_config_inplace(model_config)
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
        # TODO(will): add support for other dtypes
        return self.load_model(
            component_model_path,
            encoder_config,
            server_args,
            encoder_dtype,
        )

    def load_model(
        self,
        model_path: str,
        model_config: EncoderConfig,
        server_args: ServerArgs,
        dtype: str = "fp16",
        cpu_offload_flag: bool | None = None,
    ):
        # Determine CPU offload behavior and target device

        local_torch_device = get_local_torch_device()
        should_offload = self.should_offload(server_args, model_config)
        with set_default_torch_dtype(PRECISION_TO_TYPE[dtype]):
            with local_torch_device, skip_init_modules():
                architectures = getattr(model_config, "architectures", [])
                model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
                model = model_cls(model_config)

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self._get_all_weights(model, model_path, to_cpu=should_offload)
            )
            self.counter_after_loading_weights = time.perf_counter()
            logger.info(
                "Loading weights took %.2f seconds",
                self.counter_after_loading_weights
                - self.counter_before_loading_weights,
            )

            # Explicitly move model to target device after loading weights
            model = model.to(local_torch_device)

            if should_offload:
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
                        fsdp_shard_conditions=model_config.arch_config._fsdp_shard_conditions
                        or getattr(model, "_fsdp_shard_conditions", None),
                        pin_cpu_memory=server_args.pin_cpu_memory,
                    )
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            # if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following model weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )

        return model.eval()


class ImageEncoderLoader(TextEncoderLoader):
    def should_offload(self, server_args, model_config: ModelConfig | None = None):
        should_offload = server_args.image_encoder_cpu_offload
        fsdp_shard_conditions = getattr(model_config, "_fsdp_shard_conditions", [])
        use_cpu_offload = should_offload and len(fsdp_shard_conditions) > 0
        return use_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the text encoders based on the model path, and inference args."""
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=server_args.trust_remote_code,
        #     revision=server_args.revision,
        #     model_override_args=None,
        # )
        with open(os.path.join(component_model_path, "config.json")) as f:
            model_config = json.load(f)
        _clean_hf_config_inplace(model_config)
        logger.info("HF model config: %s", model_config)

        encoder_config = server_args.pipeline_config.image_encoder_config
        encoder_config.update_model_arch(model_config)

        # Always start with local device; load_model will adjust for offload if needed
        # TODO(will): add support for other dtypes
        return self.load_model(
            component_model_path,
            encoder_config,
            server_args,
            server_args.pipeline_config.image_encoder_precision,
            cpu_offload_flag=server_args.image_encoder_cpu_offload,
        )


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


class VAELoader(ComponentLoader):
    """Loader for VAE."""

    def should_offload(self, server_args, cpu_offload_flag, model_config):
        return True

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the VAE based on the model path, and inference args."""
        config = get_diffusers_component_config(model_path=component_model_path)
        class_name = config.pop("_class_name", None)
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        server_args.model_paths["vae"] = component_model_path

        logger.info("HF model config: %s", config)
        vae_config = server_args.pipeline_config.vae_config
        vae_config.update_model_arch(config)

        # NOTE: some post init logics are only available after updated with config
        vae_config.post_init()

        target_device = self.target_device(server_args.vae_cpu_offload)

        # Check for auto_map first (custom VAE classes)
        auto_map = config.get("auto_map", {})
        auto_model_map = auto_map.get("AutoModel")
        if auto_model_map:
            module_path, cls_name = auto_model_map.rsplit(".", 1)
            custom_module_file = os.path.join(component_model_path, f"{module_path}.py")
            spec = importlib.util.spec_from_file_location("_custom", custom_module_file)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            vae_cls = getattr(custom_module, cls_name)
            vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
            with set_default_torch_dtype(vae_dtype):
                vae = vae_cls.from_pretrained(
                    component_model_path,
                    revision=server_args.revision,
                    trust_remote_code=server_args.trust_remote_code,
                )
            vae = vae.to(device=target_device, dtype=vae_dtype)
            return vae.eval()

        # Load from ModelRegistry (standard VAE classes)
        with (
            set_default_torch_dtype(
                PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
            ),
            skip_init_modules(),
        ):
            vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vae = vae_cls(vae_config).to(target_device)

        safetensors_list = _list_safetensors_files(component_model_path)
        assert (
            len(safetensors_list) == 1
        ), f"Found {len(safetensors_list)} safetensors files in {component_model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(loaded, strict=False)
        return vae.eval()


class TransformerLoader(ComponentLoader):
    """Loader for transformer."""

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the transformer based on the model path, and inference args."""
        config = get_diffusers_component_config(model_path=component_model_path)
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

        server_args.model_paths["transformer"] = component_model_path

        # Config from Diffusers supersedes sgl_diffusion's model config
        dit_config = server_args.pipeline_config.dit_config
        dit_config.update_model_arch(config)

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Find all safetensors files
        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")

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
                safetensors_list = _list_safetensors_files(custom_weights_path)
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

        if hasattr(model, "fuse_qkv_projections"):
            logger.info("Fusing QKV projections for better performance")
            model.fuse_qkv_projections()

        return model


class SchedulerLoader(ComponentLoader):
    """Loader for scheduler."""

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the scheduler based on the model path, and inference args."""
        config = get_diffusers_component_config(model_path=component_model_path)

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
