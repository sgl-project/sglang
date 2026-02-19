import dataclasses
import glob
import os
from collections.abc import Generator, Iterable
from typing import Generator, Iterable, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import nn
from torch.distributed import init_device_mesh
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from sglang.multimodal_gen.configs.models import EncoderConfig, ModelConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import shard_model
from sglang.multimodal_gen.runtime.loader.utils import (
    _clean_hf_config_inplace,
    set_default_torch_dtype,
    skip_init_modules,
)
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
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class TextEncoderLoader(ComponentLoader):
    """Loader for text encoders."""

    component_names = ["text_encoder"]
    expected_library = "transformers"

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

    def should_offload(self, server_args, model_config: ModelConfig | None = None):
        should_offload = server_args.text_encoder_cpu_offload
        if not should_offload:
            return False
        # _fsdp_shard_conditions is in arch_config, not directly on model_config
        arch_config = (
            getattr(model_config, "arch_config", model_config) if model_config else None
        )
        fsdp_shard_conditions = (
            getattr(arch_config, "_fsdp_shard_conditions", []) if arch_config else []
        )
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
        self, component_model_path: str, server_args: ServerArgs, component_name: str
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
        logger.debug("HF model config: %s", model_config)

        def is_not_first_encoder(module_name):
            return "2" in module_name

        # TODO(mick): had to throw an exception for different text-encoder arch
        if not is_not_first_encoder(component_name):
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

        if should_offload and not current_platform.is_mps():
            model_device = torch.device("cpu")
        else:
            model_device = local_torch_device

        with set_default_torch_dtype(PRECISION_TO_TYPE[dtype]):
            with model_device, skip_init_modules():
                architectures = getattr(model_config, "architectures", [])
                model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
                enable_image_understanding = (
                    True
                    if isinstance(
                        server_args.pipeline_config, QwenImageEditPipelineConfig
                    )
                    else False
                )
                model_config.enable_image_understanding = enable_image_understanding
                model = model_cls(model_config)

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self._get_all_weights(model, model_path, to_cpu=should_offload)
            )

            # Explicitly move model to target device after loading weights
            if not should_offload:
                model = model.to(local_torch_device)

            if should_offload:
                # Disable FSDP for MPS as it's not compatible
                if current_platform.is_mps():
                    logger.info(
                        "Disabling FSDP sharding for MPS platform as it's not compatible"
                    )
                    model = model.to(local_torch_device)
                else:
                    mesh = init_device_mesh(
                        current_platform.device_type,
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
            else:
                model = model.to(local_torch_device)
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            # if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                # NOTE:
                # If we silently continue with uninitialized weights, the text encoder can
                # produce NaNs/garbage embeddings that later fail stage verification in a
                # hard-to-debug way (e.g., `prompt_embeds` fails the NaN check).
                #
                # We allow a small set of known-optional parameters to be missing, but
                # default to strict behavior for the rest.
                allowed_missing_patterns = (
                    getattr(model, "_allowed_missing_weights_patterns", []) or []
                )
                unexpected_missing = {
                    n
                    for n in weights_not_loaded
                    if not any(pat in n for pat in allowed_missing_patterns)
                }
                if unexpected_missing:
                    raise ValueError(
                        "Following text encoder weights were not initialized from checkpoint: "
                        f"{sorted(unexpected_missing)}. "
                        "This usually indicates a checkpoint/model-arch mismatch or a broken "
                        "weight-name mapping. If these are truly optional, set "
                        "`model._allowed_missing_weights_patterns` to whitelist patterns."
                    )
                logger.warning(
                    "Following (allowed) text encoder weights were not initialized from "
                    "checkpoint: %s (allowed patterns: %s)",
                    sorted(weights_not_loaded),
                    allowed_missing_patterns,
                )

        return model
