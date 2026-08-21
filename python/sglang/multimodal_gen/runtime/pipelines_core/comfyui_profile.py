# SPDX-License-Identifier: Apache-2.0
"""ComfyUI profile: run a native pipeline as a DiT-only forward service.

When ``--comfyui-mode`` is set, ComfyUI owns the sampler loop and calls into
SGLang once per step for the DiT forward. The pipeline therefore drops every
module except the transformer, swaps in a pass-through scheduler so nothing
touches the latents, and keeps only the two stages needed to run one forward.

This is a profile of the regular pipelines, not a separate family: a model
opts in by declaring a ``ComfyUICheckpointSpec``. Non-ComfyUI paths never reach
this module.
"""

import os
from typing import TYPE_CHECKING, Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoint import (
    get_comfyui_checkpoint_spec,
    get_registered_comfyui_pipeline_names,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_comfyui_passthrough import (
    ComfyUIPassThroughScheduler,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )

logger = init_logger(__name__)

COMFYUI_REQUIRED_MODULES = ["transformer", "scheduler"]


def is_comfyui_mode(server_args: ServerArgs) -> bool:
    return bool(server_args.comfyui_mode)


def is_comfyui_single_file(model_path: str) -> bool:
    """ComfyUI ships DiTs as one safetensors file with no model_index.json."""
    return os.path.isfile(model_path) and model_path.endswith(".safetensors")


def initialize_comfyui_pipeline(
    pipeline: "ComposedPipelineBase", server_args: ServerArgs
) -> None:
    """Install the pass-through scheduler and finish deriving VAE geometry.

    The VAE model itself is never loaded, but its config still carries the
    compression ratios that RoPE frequency construction reads.
    """
    pipeline.modules["scheduler"] = ComfyUIPassThroughScheduler(num_train_timesteps=1000)

    vae_config = getattr(server_args.pipeline_config, "vae_config", None)
    if (
        vae_config is not None
        and hasattr(vae_config, "post_init")
        and not hasattr(vae_config, "_post_init_called")
    ):
        vae_config.post_init()


def create_comfyui_pipeline_stages(
    pipeline: "ComposedPipelineBase", server_args: ServerArgs
) -> None:
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        ComfyUILatentPreparationStage,
        DenoisingStage,
    )

    transformer = pipeline.get_module("transformer")
    scheduler = pipeline.get_module("scheduler")
    pipeline.add_stages(
        [
            ComfyUILatentPreparationStage(scheduler=scheduler, transformer=transformer),
            DenoisingStage(transformer=transformer, scheduler=scheduler),
        ]
    )


def load_comfyui_transformer(
    pipeline: "ComposedPipelineBase",
    server_args: ServerArgs,
    loaded_modules: dict[str, torch.nn.Module] | None = None,
) -> dict[str, Any]:
    """Load the DiT from a single ComfyUI safetensors file.

    Reuses the shared FSDP-aware transformer load path; the spec only supplies
    what the file itself cannot describe.
    """
    if loaded_modules is not None and "transformer" in loaded_modules:
        return {
            "transformer": loaded_modules["transformer"],
            "scheduler": pipeline.get_module("scheduler"),
        }

    spec = get_comfyui_checkpoint_spec(pipeline.pipeline_name)
    if spec is None:
        raise ValueError(
            f"{pipeline.pipeline_name} has no ComfyUI checkpoint spec, so it cannot "
            f"load a single safetensors file. Pipelines with a spec: "
            f"{get_registered_comfyui_pipeline_names()}"
        )

    model_path = pipeline.model_path
    dit_config = spec.build_dit_config(server_args)
    mapping = dict(spec.param_names_mapping)
    if spec.inherit_config_mapping:
        mapping = {
            **(dit_config.arch_config.param_names_mapping or {}),
            **mapping,
        }
    dit_config.arch_config.param_names_mapping = mapping

    model_cls, _ = ModelRegistry.resolve_model_cls(spec.dit_cls_name)
    param_dtype = resolve_precision(server_args, "dit", precision_attr="dit_precision")
    server_args.model_paths["transformer"] = os.path.dirname(model_path) or "."

    # Only override the iterator when tensors need reshaping; leaving it None
    # keeps the rank-local checkpoint fast path available.
    weights_iterator = None
    if spec.convert_weights is not None:
        weights_iterator = spec.convert_weights(
            safetensors_weights_iterator([model_path]), dit_config
        )

    logger.info(
        "Loading %s from ComfyUI checkpoint %s, param_dtype: %s",
        spec.dit_cls_name,
        model_path,
        param_dtype,
    )

    # Weight loading reads param_names_mapping off the model, which inherits it
    # from the class, so the ComfyUI names have to be visible for the whole load.
    original_mapping = model_cls.param_names_mapping
    model_cls.param_names_mapping = mapping
    try:
        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params={"config": dit_config, "hf_config": {}},
            weight_dir_list=[model_path],
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            component_starts_on_cpu=server_args.should_start_component_on_cpu(
                "transformer"
            ),
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.should_use_fsdp_for_component("transformer"),
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=spec.strict,
            weights_iterator=weights_iterator,
        )
    finally:
        model_cls.param_names_mapping = original_mapping

    for param in model.parameters():
        param.requires_grad = False

    logger.info(
        "Loaded transformer with %.2fB parameters",
        sum(p.numel() for p in model.parameters()) / 1e9,
    )

    return {"transformer": model, "scheduler": pipeline.get_module("scheduler")}
