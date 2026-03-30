import logging
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.loader.utils import _normalize_component_type
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import get_log_level, init_logger
from sglang.srt.utils import is_npu

_is_npu = is_npu()

logger = init_logger(__name__)


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    component_names = ["transformer", "audio_dit", "video_dit"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the transformer based on the model path, and inference args."""
        # 1. hf config
        config = get_diffusers_component_config(component_path=component_model_path)

        safetensors_list = resolve_transformer_safetensors_to_load(
            server_args, component_model_path
        )

        # 2. dit config
        # Config from Diffusers supersedes sgl_diffusion's model config
        component_name = _normalize_component_type(component_name)
        server_args.model_paths[component_name] = component_model_path
        if component_name in ("transformer", "video_dit"):
            pipeline_dit_config_attr = "dit_config"
        elif component_name in ("audio_dit",):
            pipeline_dit_config_attr = "audio_dit_config"
        else:
            raise ValueError(f"Invalid module name: {component_name}")
        dit_config = getattr(server_args.pipeline_config, pipeline_dit_config_attr)
        dit_config.update_model_arch(config)

        cls_name = config.pop("_class_name")
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        quant_spec = resolve_transformer_quant_load_spec(
            hf_config=config,
            server_args=server_args,
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            model_cls=model_cls,
            cls_name=cls_name,
        )

        logger.info(
            "Loading %s from %s safetensors file(s) %s, param_dtype: %s",
            cls_name,
            len(safetensors_list),
            f": {safetensors_list}" if get_log_level() == logging.DEBUG else "",
            quant_spec.param_dtype,
        )
        # prepare init_param
        init_params: dict[str, Any] = {
            "config": dit_config,
            "hf_config": config,
            "quant_config": quant_spec.runtime_quant_config,
        }
        if (
            init_params["quant_config"] is None
            and server_args.transformer_weights_path is not None
        ):
            logger.warning(
                f"transformer_weights_path provided, but quantization config not resolved, which is unexpected and likely to cause errors"
            )
        else:
            logger.debug("quantization config: %s", init_params["quant_config"])

        # Load the model using FSDP loader
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params=init_params,
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=server_args.dit_cpu_offload,
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.use_fsdp_inference,
            param_dtype=quant_spec.param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )

        # post-hooks (e.g., patch scales (nunchaku))
        for post_load_hook in quant_spec.post_load_hooks:
            post_load_hook(model)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        # considering the existent of mixed-precision models (e.g., nunchaku)
        if (
            next(model.parameters()).dtype != quant_spec.param_dtype
            and quant_spec.param_dtype
        ):
            logger.warning(
                "Model dtype does not match expected param dtype, %s vs %s",
                next(model.parameters()).dtype,
                quant_spec.param_dtype,
            )

        return model
