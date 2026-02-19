from copy import deepcopy

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class BridgeLoader(ComponentLoader):
    """Loader for MOVA dual tower bridge with FSDP support."""

    pipeline_bridge_config_attr: str = "bridge_config"

    component_names = ["dual_tower_bridge"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        config = get_diffusers_component_config(model_path=component_model_path)
        hf_config = deepcopy(config)
        class_name = config.pop("_class_name", None)
        if class_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )
        server_args.model_paths[component_name] = component_model_path

        # Try to get bridge config from pipeline config, fallback to creating one
        bridge_config = getattr(
            server_args.pipeline_config, self.pipeline_bridge_config_attr, None
        )
        if bridge_config is not None:
            bridge_config.update_model_arch(config)
        else:
            # Create a minimal config from hf_config
            from sglang.multimodal_gen.configs.models.bridges.mova_dual_tower import (
                MOVADualTowerConfig,
            )

            bridge_config = MOVADualTowerConfig()
            bridge_config.update_model_arch(config)

        model_cls, _ = ModelRegistry.resolve_model_cls(class_name)

        # Find all safetensors files
        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        logger.info(
            "Loading %s from %s safetensors files, default_dtype: %s",
            class_name,
            len(safetensors_list),
            default_dtype,
        )

        # Check if FSDP loading is available
        if (
            server_args.hsdp_shard_dim is not None
            and hasattr(model_cls, "_fsdp_shard_conditions")
            and model_cls._fsdp_shard_conditions
        ):
            # Load with FSDP support
            model = maybe_load_fsdp_model(
                model_cls=model_cls,
                init_params={"config": bridge_config, "hf_config": hf_config},
                weight_dir_list=safetensors_list,
                device=get_local_torch_device(),
                hsdp_replicate_dim=server_args.hsdp_replicate_dim,
                hsdp_shard_dim=server_args.hsdp_shard_dim,
                cpu_offload=server_args.dit_cpu_offload,
                pin_cpu_memory=server_args.pin_cpu_memory,
                fsdp_inference=server_args.use_fsdp_inference,
                param_dtype=default_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                strict=False,
            )
        else:
            # Fallback to simple loading (for non-FSDP or legacy models)
            model = model_cls.from_pretrained(
                component_model_path, torch_dtype=default_dtype
            )
            model = model.to(device=get_local_torch_device(), dtype=default_dtype)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded bridge model with %.2fM parameters", total_params / 1e6)

        return model
