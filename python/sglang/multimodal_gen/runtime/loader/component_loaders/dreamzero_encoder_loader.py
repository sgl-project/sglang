# SPDX-License-Identifier: Apache-2.0
"""DreamZero text/image encoder loaders.

DreamZero-DROID stores Wan text encoder and open-clip ViT-H weights under
action-head prefixes. These loaders reuse SGLang-native UMT5 and CLIP modules
and keep DreamZero-specific config, key remapping, and visual-attention policy
localized here.
"""

from __future__ import annotations

import os
import types
from contextlib import contextmanager

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders import CLIPVisionConfig, T5Config
from sglang.multimodal_gen.configs.models.encoders.clip import CLIPVisionArchConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_checkpoint_utils import (
    DreamZeroCheckpointLoadReport,
    iter_prefixed_safetensors,
    raise_for_strict_report,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_tp_world_size,
    get_world_group,
    init_parallel_group_coordinator,
    model_parallel_is_initialized,
    patch_tensor_parallel_group,
    world_group_is_initialized,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

_DROID_TEXT_ENCODER_PREFIX = "action_head.text_encoder."
_DROID_IMAGE_ENCODER_PREFIX = "action_head.image_encoder."


def _dreamzero_umt5_config() -> T5Config:
    return T5Config(
        prefix="dreamzero_text_encoder",
        arch_config=T5ArchConfig(
            vocab_size=256384,
            d_model=4096,
            d_kv=64,
            d_ff=10240,
            num_layers=24,
            num_heads=64,
            relative_attention_num_buckets=32,
            relative_attention_max_distance=128,
            dropout_rate=0.1,
            feed_forward_proj="gated-gelu",
            layer_norm_epsilon=1e-6,
            text_len=512,
        ),
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


def _patch_dreamzero_clip_vision_attention(model: nn.Module) -> None:
    for layer in model.vision_model.encoder.layers:
        attention = layer.self_attn
        attention.attn.attn_impl.causal = False
        attention.forward = types.MethodType(
            _dreamzero_non_causal_clip_attention_forward,
            attention,
        )


@contextmanager
def _replicated_tp_group_for_dreamzero_image_encoder():
    """DreamZero's CLIP vision checkpoint is replicated, not tensor-sharded."""
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


def remap_dreamzero_text_model_key(key: str) -> str | None:
    if key == "token_embedding.weight":
        return "shared.weight"
    if key == "norm.weight":
        return "encoder.final_layer_norm.weight"

    parts = key.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        return None

    layer = parts[1]
    tail = ".".join(parts[2:])
    prefix = f"encoder.block.{layer}"
    replacements = {
        "norm1.weight": f"{prefix}.layer.0.layer_norm.weight",
        "attn.q.weight": f"{prefix}.layer.0.SelfAttention.q.weight",
        "attn.k.weight": f"{prefix}.layer.0.SelfAttention.k.weight",
        "attn.v.weight": f"{prefix}.layer.0.SelfAttention.v.weight",
        "attn.o.weight": f"{prefix}.layer.0.SelfAttention.o.weight",
        "pos_embedding.embedding.weight": (
            f"{prefix}.layer.0.SelfAttention.relative_attention_bias.weight"
        ),
        "norm2.weight": f"{prefix}.layer.1.layer_norm.weight",
        "ffn.gate.0.weight": f"{prefix}.layer.1.DenseReluDense.wi_0.weight",
        "ffn.fc1.weight": f"{prefix}.layer.1.DenseReluDense.wi_1.weight",
        "ffn.fc2.weight": f"{prefix}.layer.1.DenseReluDense.wo.weight",
    }
    return replacements.get(tail)


def remap_dreamzero_text_checkpoint_key(checkpoint_key: str) -> str | None:
    if not checkpoint_key.startswith(_DROID_TEXT_ENCODER_PREFIX):
        return None
    return remap_dreamzero_text_model_key(
        checkpoint_key[len(_DROID_TEXT_ENCODER_PREFIX) :]
    )


def _loaded_text_param_name(key: str) -> str:
    for source in (".q.weight", ".k.weight", ".v.weight"):
        if source in key:
            return key.replace(source, ".qkv_proj.weight")
    return key


def _is_ignored_dreamzero_image_key(key: str) -> bool:
    return (
        key == "model.log_scale"
        or key == "model.visual.head"
        or key.startswith("model.visual.post_norm.")
        or key.startswith("model.visual.transformer.31.")
    )


def remap_dreamzero_image_model_key(key: str) -> str | None:
    if _is_ignored_dreamzero_image_key(key):
        return None
    if key == "model.visual.cls_embedding":
        return "vision_model.embeddings.class_embedding"
    if key == "model.visual.patch_embedding.weight":
        return "vision_model.embeddings.patch_embedding.weight"
    if key == "model.visual.pos_embedding":
        return "vision_model.embeddings.position_embedding.weight"
    if key == "model.visual.pre_norm.weight":
        return "vision_model.pre_layrnorm.weight"
    if key == "model.visual.pre_norm.bias":
        return "vision_model.pre_layrnorm.bias"

    parts = key.split(".")
    if len(parts) < 6 or parts[:3] != ["model", "visual", "transformer"]:
        return None

    layer = parts[3]
    tail = ".".join(parts[4:])
    prefix = f"vision_model.encoder.layers.{layer}"
    replacements = {
        "norm1.weight": f"{prefix}.layer_norm1.weight",
        "norm1.bias": f"{prefix}.layer_norm1.bias",
        "attn.to_qkv.weight": f"{prefix}.self_attn.qkv_proj.weight",
        "attn.to_qkv.bias": f"{prefix}.self_attn.qkv_proj.bias",
        "attn.proj.weight": f"{prefix}.self_attn.out_proj.weight",
        "attn.proj.bias": f"{prefix}.self_attn.out_proj.bias",
        "norm2.weight": f"{prefix}.layer_norm2.weight",
        "norm2.bias": f"{prefix}.layer_norm2.bias",
        "mlp.0.weight": f"{prefix}.mlp.fc1.weight",
        "mlp.0.bias": f"{prefix}.mlp.fc1.bias",
        "mlp.2.weight": f"{prefix}.mlp.fc2.weight",
        "mlp.2.bias": f"{prefix}.mlp.fc2.bias",
    }
    return replacements.get(tail)


def remap_dreamzero_image_checkpoint_key(checkpoint_key: str) -> str | None:
    if not checkpoint_key.startswith(_DROID_IMAGE_ENCODER_PREFIX):
        return None
    return remap_dreamzero_image_model_key(
        checkpoint_key[len(_DROID_IMAGE_ENCODER_PREFIX) :]
    )


def _convert_dreamzero_image_tensor(
    target_key: str, tensor: torch.Tensor
) -> torch.Tensor:
    if target_key == "vision_model.embeddings.class_embedding":
        return tensor.reshape(-1)
    if target_key == "vision_model.embeddings.position_embedding.weight":
        return tensor.squeeze(0)
    return tensor


def expected_dreamzero_text_state_keys(
    state_keys: set[str], loaded_keys: set[str]
) -> set[str]:
    expected_keys = set(state_keys)
    # UMT5 ties encoder.embed_tokens to shared. Loading shared.weight is enough;
    # state_dict still exposes the alias as a second key.
    if "shared.weight" in loaded_keys:
        expected_keys.discard("encoder.embed_tokens.weight")
    return expected_keys


def build_dreamzero_text_encoder(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> nn.Module:
    from sglang.multimodal_gen.runtime.models.encoders.t5 import UMT5EncoderModel

    if device is None:
        with torch.device("meta"):
            model = UMT5EncoderModel(_dreamzero_umt5_config()).to(dtype=dtype)
    else:
        with torch.device(device):
            model = UMT5EncoderModel(_dreamzero_umt5_config()).to(dtype=dtype)
    model.eval()
    return model


def build_dreamzero_image_encoder(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> nn.Module:
    from sglang.multimodal_gen.runtime.models.encoders.clip import CLIPVisionModel

    if device is None:
        with _replicated_tp_group_for_dreamzero_image_encoder(), torch.device("meta"):
            model = CLIPVisionModel(_dreamzero_clip_vision_config()).to(dtype=dtype)
    else:
        with _replicated_tp_group_for_dreamzero_image_encoder(), torch.device(device):
            model = CLIPVisionModel(_dreamzero_clip_vision_config()).to(dtype=dtype)
    _patch_dreamzero_clip_vision_attention(model)
    model.eval()
    return model


def load_dreamzero_text_encoder_checkpoint(
    model: nn.Module,
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    strict: bool = False,
) -> DreamZeroCheckpointLoadReport:
    if not hasattr(model, "load_weights"):
        raise TypeError("DreamZero text encoder must provide load_weights()")

    tensors: list[tuple[str, torch.Tensor]] = []
    unexpected_keys: list[str] = []
    for checkpoint_key, tensor in iter_prefixed_safetensors(
        model_path, _DROID_TEXT_ENCODER_PREFIX
    ):
        target_key = remap_dreamzero_text_checkpoint_key(checkpoint_key)
        if target_key is None:
            unexpected_keys.append(checkpoint_key)
            continue
        tensors.append((target_key, tensor.to(device=device)))

    loaded_keys = sorted(
        _loaded_text_param_name(key) for key in model.load_weights(tensors)
    )
    expected_keys = expected_dreamzero_text_state_keys(
        set(model.state_dict()), set(loaded_keys)
    )
    report = DreamZeroCheckpointLoadReport(
        loaded_keys=loaded_keys,
        missing_keys=sorted(expected_keys - set(loaded_keys)),
        unexpected_keys=unexpected_keys,
        shape_mismatches={},
    )
    raise_for_strict_report(
        report,
        strict=strict,
        error_prefix="DreamZero text encoder checkpoint load failed",
    )
    return report


def load_dreamzero_image_encoder_checkpoint(
    model: nn.Module,
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    strict: bool = False,
) -> DreamZeroCheckpointLoadReport:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    target_tensors = {**params, **buffers}
    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for checkpoint_key, tensor in iter_prefixed_safetensors(
        model_path, _DROID_IMAGE_ENCODER_PREFIX
    ):
        raw_key = checkpoint_key[len(_DROID_IMAGE_ENCODER_PREFIX) :]
        target_key = remap_dreamzero_image_model_key(raw_key)
        if target_key is None:
            if not _is_ignored_dreamzero_image_key(raw_key):
                unexpected_keys.append(checkpoint_key)
            continue
        target = target_tensors.get(target_key)
        if target is None:
            unexpected_keys.append(checkpoint_key)
            continue
        converted = _convert_dreamzero_image_tensor(target_key, tensor).to(
            device=device,
            dtype=target.dtype,
        )
        if tuple(converted.shape) != tuple(target.shape):
            shape_mismatches[target_key] = (
                tuple(target.shape),
                tuple(converted.shape),
            )
            continue
        weight_loader = getattr(target, "weight_loader", default_weight_loader)
        weight_loader(target, converted)
        loaded_keys.append(target_key)

    expected_keys = set(model.state_dict())
    report = DreamZeroCheckpointLoadReport(
        loaded_keys=sorted(loaded_keys),
        missing_keys=sorted(expected_keys - set(loaded_keys)),
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
    )
    raise_for_strict_report(
        report,
        strict=strict,
        error_prefix="DreamZero image encoder checkpoint load failed",
    )
    return report


def build_dreamzero_text_encoder_from_checkpoint(
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> tuple[nn.Module, DreamZeroCheckpointLoadReport]:
    model = build_dreamzero_text_encoder(dtype=dtype, device=device)
    report = load_dreamzero_text_encoder_checkpoint(
        model,
        model_path,
        device=device,
        strict=strict,
    )
    return model, report


def build_dreamzero_image_encoder_from_checkpoint(
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> tuple[nn.Module, DreamZeroCheckpointLoadReport]:
    model = build_dreamzero_image_encoder(dtype=dtype, device=device)
    report = load_dreamzero_image_encoder_checkpoint(
        model,
        model_path,
        device=device,
        strict=strict,
    )
    return model, report
