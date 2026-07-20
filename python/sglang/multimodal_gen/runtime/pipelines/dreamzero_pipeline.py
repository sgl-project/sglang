# SPDX-License-Identifier: Apache-2.0
"""DreamZero DROID one-shot action pipeline."""

from __future__ import annotations

import os
import types
from contextlib import contextmanager
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.encoders import CLIPVisionConfig
from sglang.multimodal_gen.configs.models.encoders.clip import CLIPVisionArchConfig
from sglang.multimodal_gen.configs.pipeline_configs.dreamzero import (
    DreamZeroPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.dreamzero import DreamZeroSamplingParams
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_sp_world_size,
    get_tp_world_size,
    get_world_group,
    init_parallel_group_coordinator,
    model_parallel_is_initialized,
    patch_tensor_parallel_group,
    world_group_is_initialized,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.session_cache import (
    DreamZeroCachePoolManager,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.denoising import (
    DreamZeroActionOutputStage,
    DreamZeroCausalDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.image_encoding import (
    DreamZeroObsPrepStage,
    DreamZeroVisualEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dreamzero.text_encoding import (
    DreamZeroTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    build_torch_compile_kwargs,
    maybe_enable_inductor_compute_comm_overlap,
)

logger = init_logger(__name__)


def _compile_dreamzero_dit_blocks(transformer: Any) -> int:
    blocks = transformer.blocks
    if not isinstance(blocks, torch.nn.ModuleList):
        raise TypeError("DreamZero transformer.blocks must be a ModuleList")

    maybe_enable_inductor_compute_comm_overlap()

    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit,
        128,
    )
    compile_kwargs = build_torch_compile_kwargs(mode="default")
    for index, block in enumerate(blocks):
        blocks[index] = torch.compile(block, **compile_kwargs)
    return len(blocks)


def _is_sglang_dreamzero_checkpoint(model_path: str) -> bool:
    return (
        os.path.isfile(os.path.join(model_path, "model_index.json"))
        and os.path.isdir(os.path.join(model_path, "transformer"))
        and os.path.isdir(os.path.join(model_path, "text_encoder"))
        and os.path.isdir(os.path.join(model_path, "image_encoder"))
        and os.path.isdir(os.path.join(model_path, "vae"))
    )


def _dreamzero_clip_vision_config() -> CLIPVisionConfig:
    return CLIPVisionConfig(
        prefix="dreamzero_image_encoder",
        num_hidden_layers_override=31,
        require_post_norm=False,
        arch_config=CLIPVisionArchConfig(
            hidden_size=1280,
            intermediate_size=5120,
            projection_dim=1024,
            num_hidden_layers=32,
            num_attention_heads=16,
            num_channels=3,
            image_size=224,
            patch_size=14,
            hidden_act="gelu",
            layer_norm_eps=1e-5,
            dropout=0.0,
            attention_dropout=0.0,
        ),
    )


def _dreamzero_non_causal_clip_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    qkv_states, _ = self.qkv_proj(hidden_states)
    query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
    query_states = query_states.reshape(
        query_states.shape[0],
        query_states.shape[1],
        self.num_heads_per_partition,
        self.head_dim,
    )
    key_states = key_states.reshape(
        key_states.shape[0],
        key_states.shape[1],
        self.num_heads_per_partition,
        self.head_dim,
    )
    value_states = value_states.reshape(
        value_states.shape[0],
        value_states.shape[1],
        self.num_heads_per_partition,
        self.head_dim,
    )

    if self.attn.backend == AttentionBackendEnum.TORCH_SDPA:
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :].to(
                    dtype=query_states.dtype
                )
                attn_mask = (1.0 - attn_mask) * torch.finfo(query_states.dtype).min
            else:
                attn_mask = attention_mask
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.scale,
        )
        attn_output = attn_output.transpose(1, 2)
    else:
        attn_output = self.attn(query_states, key_states, value_states)

    attn_output = attn_output.reshape(
        attn_output.shape[0],
        attn_output.shape[1],
        self.num_heads_per_partition * self.head_dim,
    )
    attn_output, _ = self.out_proj(attn_output)
    return attn_output, None


def _patch_dreamzero_clip_vision_attention(model: torch.nn.Module) -> None:
    for layer in model.vision_model.encoder.layers:
        attention = layer.self_attn
        attention.attn.attn_impl.causal = False
        attention.forward = types.MethodType(
            _dreamzero_non_causal_clip_attention_forward,
            attention,
        )


@contextmanager
def _replicated_tp_group_for_dreamzero_image_encoder():
    if (
        not world_group_is_initialized()
        or not model_parallel_is_initialized()
        or get_tp_world_size() == 1
    ):
        yield
        return

    world_group = get_world_group()
    world_size = world_group.world_size
    backend = torch.distributed.get_backend(world_group.device_group)
    replicated_tp_group = init_parallel_group_coordinator(
        group_ranks=[[r] for r in range(world_size)],
        local_rank=world_group.local_rank,
        backend=backend,
        parallel_mode="tensor",
    )
    with patch_tensor_parallel_group(replicated_tp_group):
        yield


def _load_dreamzero_image_encoder(
    server_args: ServerArgs,
    component_model_path: str,
) -> torch.nn.Module:
    from sglang.multimodal_gen.runtime.loader.component_loaders.image_encoder_loader import (
        ImageEncoderLoader,
    )
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        get_diffusers_component_config,
    )

    image_config = _dreamzero_clip_vision_config()
    image_config.update_model_arch(
        get_diffusers_component_config(component_model_path)
    )
    with _replicated_tp_group_for_dreamzero_image_encoder():
        image_encoder = ImageEncoderLoader().load_model(
            component_model_path,
            image_config,
            server_args,
            server_args.pipeline_config.image_encoder_precision,
        )
    _patch_dreamzero_clip_vision_attention(image_encoder)
    return image_encoder


class DreamZeroPipeline(ComposedPipelineBase):
    """Pipeline that composes DreamZero obs prep, text encoding, DiT and action output."""

    pipeline_name = "DreamZeroPipeline"
    is_video_pipeline = False
    _required_config_modules = [
        "text_encoder",
        "vae",
        "transformer",
    ]
    pipeline_config_cls = DreamZeroPipelineConfig
    sampling_params_cls = DreamZeroSamplingParams

    def _build_scheduler(self, server_args: ServerArgs) -> FlowUniPCMultistepScheduler:
        return FlowUniPCMultistepScheduler(
            shift=server_args.pipeline_config.flow_shift,
        )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        modules = dict(loaded_modules or {})
        modules.setdefault("scheduler", self._build_scheduler(server_args))
        if loaded_modules is not None:
            return modules

        if not _is_sglang_dreamzero_checkpoint(self.model_path):
            raise RuntimeError(
                "DreamZeroPipeline requires a checkpoint in SGLang component layout "
                "with model_index.json plus transformer/text_encoder/image_encoder/vae "
                "component directories."
            )

        server_args.pipeline_config.dit_config.arch_config.use_tensor_parallel = (
            server_args.pipeline_config.dreamzero_tensor_parallel_size > 1
        )
        modules.update(super().load_modules(server_args, loaded_modules))

        modules["image_encoder"] = _load_dreamzero_image_encoder(
            server_args,
            self._resolve_component_path(server_args, "image_encoder", "image_encoder"),
        )
        if server_args.pipeline_config.dreamzero_compile_components:
            compiled_blocks = _compile_dreamzero_dit_blocks(modules["transformer"])
            logger.info("Compiled %d DreamZero DiT blocks", compiled_blocks)
        return modules

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self.modules.setdefault("scheduler", self._build_scheduler(server_args))
        configured_tp_size = int(
            server_args.pipeline_config.dreamzero_tensor_parallel_size
        )
        configured_sp_size = int(
            server_args.pipeline_config.dreamzero_sequence_parallel_size
        )
        if model_parallel_is_initialized():
            actual_tp_size = get_tp_world_size()
            if configured_tp_size != actual_tp_size:
                raise ValueError(
                    "DreamZero tensor parallel size must match the initialized TP "
                    f"group: configured={configured_tp_size}, actual={actual_tp_size}"
                )
            actual_sp_size = get_sp_world_size()
            if configured_sp_size != actual_sp_size:
                raise ValueError(
                    "DreamZero sequence parallel size must match the initialized SP "
                    f"group: configured={configured_sp_size}, actual={actual_sp_size}"
                )
        elif configured_tp_size > 1 or configured_sp_size > 1:
            raise RuntimeError(
                "DreamZero TP/SP requires initialized model-parallel process groups"
            )
        self.cache_manager = DreamZeroCachePoolManager(
            max_sessions=server_args.pipeline_config.dreamzero_max_sessions
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(DreamZeroObsPrepStage(), "dreamzero_obs_prep_stage")
        self.add_stage(
            DreamZeroTextEncodingStage(
                self.get_module("text_encoder"),
                cache_manager=self.cache_manager,
            ),
            "dreamzero_text_encoding_stage",
        )
        self.add_stage(
            DreamZeroVisualEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                vae=self.get_module("vae"),
                cache_manager=self.cache_manager,
            ),
            "dreamzero_visual_encoding_stage",
        )
        self.add_stage(
            DreamZeroCausalDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                cache_manager=self.cache_manager,
            ),
            "dreamzero_causal_denoising_stage",
        )
        self.add_stage(
            DreamZeroActionOutputStage(),
            "dreamzero_action_postproc_stage",
        )


EntryClass = DreamZeroPipeline
