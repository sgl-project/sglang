import os
from copy import deepcopy

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    _normalize_component_type,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    component_names = ["transformer", "audio_dit", "video_dit"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
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

        component_name = _normalize_component_type(component_name)
        server_args.model_paths[component_name] = component_model_path

        if component_name in ("transformer", "video_dit"):
            pipeline_dit_config_attr = "dit_config"
        elif component_name in ("audio_dit",):
            pipeline_dit_config_attr = "audio_dit_config"
        else:
            raise ValueError(f"Invalid module name: {component_name}")
        # Config from Diffusers supersedes sgl_diffusion's model config
        dit_config = getattr(server_args.pipeline_config, pipeline_dit_config_attr)
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

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        logger.info(
            "Loading %s from %s safetensors files, default_dtype: %s",
            cls_name,
            len(safetensors_list),
            default_dtype,
        )

        # Load the model using FSDP loader
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
            strict=False,
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        assert (
            next(model.parameters()).dtype == default_dtype
        ), "Model dtype does not match default dtype"

        return model
