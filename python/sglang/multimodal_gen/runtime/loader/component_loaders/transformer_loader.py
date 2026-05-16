import copy
import logging
import os
import time
from typing import Any

import torch

from sglang.multimodal_gen import envs
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
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    can_use_runai_distributed_streamer,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import get_log_level, init_logger
from sglang.srt.utils import is_npu

_is_npu = is_npu()

logger = init_logger(__name__)


def _server_args_for_transformer_component(
    server_args: ServerArgs, component_name: str
) -> ServerArgs:
    """Mask global quantized override flags for secondary transformer components."""
    if component_name != "transformer_2":
        return server_args

    if (
        server_args.transformer_weights_path is None
        and server_args.nunchaku_config is None
    ):
        return server_args

    component_server_args = copy.copy(server_args)
    component_server_args.transformer_weights_path = None
    component_server_args.nunchaku_config = None
    logger.info(
        "Ignoring global transformer_weights_path for %s; keep it on the base "
        "checkpoint unless a per-component override path is provided.",
        component_name,
    )
    return component_server_args


def _has_merged_param_mapping(model_cls: type[torch.nn.Module]) -> bool:
    param_names_mapping = getattr(model_cls, "param_names_mapping", {})
    return any(
        isinstance(replacement, tuple) for replacement in param_names_mapping.values()
    )


def _checkpoint_size_gib(safetensors_list: list[str]) -> float | None:
    if not safetensors_list:
        return None

    total_bytes = 0
    for path in safetensors_list:
        try:
            total_bytes += os.path.getsize(path)
        except OSError:
            return None
    return total_bytes / (1024**3)


def _should_use_runai_distributed_streaming(
    server_args: ServerArgs,
    component_server_args: ServerArgs,
    model_cls: type[torch.nn.Module],
    quant_spec,
    safetensors_list: list[str],
) -> tuple[bool, str]:
    if not can_use_runai_distributed_streamer():
        return False, "runai distributed streamer is not available"
    if component_server_args.dit_cpu_offload:
        return False, "dit_cpu_offload is enabled"
    if server_args.use_fsdp_inference:
        return False, "FSDP inference is enabled"
    if quant_spec.runtime_quant_config is not None:
        return False, "quantized transformer load is enabled"
    if quant_spec.post_load_hooks:
        return False, "post-load hooks are required"
    if _has_merged_param_mapping(model_cls):
        return False, "merged parameter mapping is required"

    min_weight_gib = envs.SGLANG_RUNAI_DISTRIBUTED_MODEL_STREAMER_MIN_WEIGHT_GB
    checkpoint_size_gib = _checkpoint_size_gib(safetensors_list)
    if checkpoint_size_gib is not None and checkpoint_size_gib < min_weight_gib:
        return (
            False,
            "checkpoint is too small for distributed streaming "
            f"({checkpoint_size_gib:.2f} GiB < {min_weight_gib:.2f} GiB)",
        )
    return True, ""


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    component_names = ["transformer", "audio_dit", "video_dit"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the transformer based on the model path, and inference args."""
        requested_component_name = component_name
        component_server_args = _server_args_for_transformer_component(
            server_args, component_name
        )

        # 1. hf config
        config = get_diffusers_component_config(component_path=component_model_path)

        safetensors_list = resolve_transformer_safetensors_to_load(
            component_server_args, component_model_path
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
            server_args=component_server_args,
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
            and component_server_args.transformer_weights_path is not None
        ):
            logger.warning(
                f"transformer_weights_path provided, but quantization config not resolved, which is unexpected and likely to cause errors"
            )
        else:
            logger.debug("quantization config: %s", init_params["quant_config"])

        # Load the model using FSDP loader
        use_runai_distributed_streaming, disabled_reason = (
            _should_use_runai_distributed_streaming(
                server_args,
                component_server_args,
                model_cls,
                quant_spec,
                safetensors_list,
            )
        )
        if use_runai_distributed_streaming:
            logger.info(
                "Using RunAI distributed GPU streaming for %s",
                requested_component_name,
            )
        elif can_use_runai_distributed_streamer():
            logger.info(
                "RunAI distributed GPU streaming disabled for %s: %s",
                requested_component_name,
                disabled_reason,
            )

        load_start = time.perf_counter()
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params=init_params,
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=component_server_args.dit_cpu_offload,
            pin_cpu_memory=component_server_args.pin_cpu_memory,
            fsdp_inference=component_server_args.use_fsdp_inference,
            param_dtype=quant_spec.param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
            use_runai_distributed_streaming=use_runai_distributed_streaming,
        )
        logger.info(
            "Loaded %s weights in %.2fs",
            requested_component_name,
            time.perf_counter() - load_start,
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
