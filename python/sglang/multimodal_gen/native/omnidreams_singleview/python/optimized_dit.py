# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""native-accelerated drop-in for ``CosmosTransformer``.

Subclasses ``CosmosTransformer`` and overrides ``predict_flow`` to run
the per-step DiT forward through ``native_extension.optimized_dit_forward``
(C++/CUDA) instead of the upstream PyTorch network. The mask
injection (first-frame image latent at AR step 0) and the AR cache
state machine are inherited from the parent. The scheduler loop
(``FlowMatchScheduler.sample``) and noise/denoise math are owned by
the diffusion infra and stay unchanged.

Streaming-cache lifecycle. ``native_extension.optimized_dit_forward``
writes per-layer self-attn K/V in-place at ``self_attn_write_start``
but does NOT call the cache's ``before_update`` / ``after_update``
(the upstream ``net.forward(eager_mode=True)`` does). We wrap the C++
call so the ``BlockKVCache`` state machine evolves the same way:

  * before_update(chunk_idx) -> compute write_start -> C++ call ->
    after_update(chunk_idx)

For the multi-step scheduler loop (one call per denoising step plus
one finalize call) this is correct because
``BlockKVCache.before_update`` is a no-op when
``chunk_idx == _prev_chunk_idx``: only the first call per AR step
advances the cursor; subsequent calls overwrite the rightmost slot
with refined K/V from a less-noisy x0.

Memory. We snapshot ``self.network.state_dict()`` lazily on the first
``predict_flow`` call to hand to the C++ side. Lazy because
``nn.Module.to(device)`` (called by the framework AFTER ``__init__``)
replaces parameter storage; snapshotting earlier would capture stale
CPU references. The dict holds parameter references (not copies), so
memory cost is negligible above what the parent already holds.
``update_parameters_after_loading_checkpoint`` (the fuse ops -- bake
the padding-mask channel out of ``x_embedder`` and reshape
``final_layer.linear.weight`` into the post-shuffle layout) has
already run by then (parent's __init__ calls it before returning).
"""

from __future__ import annotations

import gc
import logging
import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

# === SGLang stub: replace flashdreams + omnidreams deps ===
# PEP 563 (from __future__ import annotations) makes type annotations
# lazy strings, so these stubs only need to exist as module-level names.
# SGLang's OmniDreamsFP8DiT.__call__ routes through _predict_flow_ext_impl
# directly and never touches predict_flow / _capture_network_cache_templates
# / _clone_network_cache / _make_cross_attn_cache.

class CosmosTransformer:
    pass

class CosmosTransformerCache:
    pass

class CosmosDiTNetworkCache:
    pass

class _FakeCUDAGraphWrapper:
    """Minimal stub so ``OptimizedDiTExecutor.__init__`` instantiates
    ``self._optimized_call`` without the FlashDreams graph wrapper.
    SGLang *never* calls this — it always routes through
    ``_predict_flow_ext_impl`` directly."""
    def __init__(self, fn, warmup_iters=0, capture_error_mode=None):
        self._fn = fn
    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

CUDAGraphWrapper = _FakeCUDAGraphWrapper

class BlockKVCache:
    @classmethod
    def from_tensor(cls, k, v, seq_dim):
        raise NotImplementedError("stub — SGLang cross-attn path is not stubbed")

def cat_outputs_cp(x, seq_dim, cp_group):
    return x

def split_inputs_cp(x, seq_dim, cp_group):
    return x

# === end SGLang stubs ===

_LOGGER = logging.getLogger(__name__)

_COSMOS_PREPARED_SUFFIXES = (
    "self_attn.qkv_proj.weight",
    "self_attn.output_proj.weight",
    "cross_attn.q_proj.weight",
    "cross_attn.output_proj.weight",
    "mlp.layer1.weight",
    "mlp.layer2.weight",
)


@dataclass
class _CosmosInvariantTensors:
    t_emb: Tensor
    t_emb_silu: Tensor
    adaln_lora: Tensor
    final_shift: Tensor
    final_scale: Tensor
    block_mods_sa: Tensor
    block_mods_ca: Tensor
    block_mods_mlp: Tensor


@dataclass
class _CosmosCacheTensorLists:
    k_cross: list[Tensor]
    v_cross: list[Tensor]
    k_self: list[Tensor]
    v_self: list[Tensor]


def _clone_block_kv_cache(cache: Any) -> Any:
    cloned = type(cache)(
        k_shape=tuple(cache.k_shape),
        v_shape=tuple(cache.v_shape),
        seq_dim=cache.seq_dim,
        chunk_size=cache.chunk_size,
        window_size=cache.window_size,
        sink_size=cache.sink_size,
        device=cache._k.device,
        dtype=cache._k.dtype,
    )
    cloned._prev_chunk_idx = cache._prev_chunk_idx
    cloned._curr_chunk_idx = cache._curr_chunk_idx
    cloned._n_cached = cache._n_cached
    cloned._k.copy_(cache._k)
    cloned._v.copy_(cache._v)
    return cloned


def _clone_network_cache(cache: CosmosDiTNetworkCache) -> CosmosDiTNetworkCache:
    return CosmosDiTNetworkCache(
        block_caches=[
            type(block)(
                self_attn=_clone_block_kv_cache(block.self_attn),
                cross_attn=_clone_block_kv_cache(block.cross_attn),
            )
            for block in cache.block_caches
        ]
    )


def _cosmos_sinusoidal_emb(ts: Tensor, num_channels: int) -> Tensor:
    half = num_channels // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=ts.device)
        / float(half)
    )
    angles = ts.to(torch.float32).reshape(-1, 1) * freqs.reshape(1, -1)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


def _make_cosmos_timestep_cache(
    timesteps: Tensor,
    weights: Mapping[str, Tensor],
    *,
    model_channels: int,
    timestep_scale: float,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    t_sin = _cosmos_sinusoidal_emb(
        timesteps.to(torch.float32) * timestep_scale,
        model_channels,
    ).to(dtype)
    w_t1 = weights["t_embedder.1.linear_1.weight"].to(
        device=timesteps.device, dtype=dtype
    )
    w_t2 = weights["t_embedder.1.linear_2.weight"].to(
        device=timesteps.device, dtype=dtype
    )
    t_h = F.silu(t_sin @ w_t1.t())
    adaln_lora = (t_h @ w_t2.t()).contiguous()

    gamma = weights["t_embedding_norm.weight"].to(
        device=timesteps.device, dtype=torch.float32
    )
    xf = t_sin.to(torch.float32)
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1.0e-6)
    t_emb = (xf * inv * gamma).to(dtype).contiguous()
    t_emb_silu = F.silu(t_emb).to(dtype).contiguous()
    return t_emb, t_emb_silu, adaln_lora


def _make_cosmos_final_mod_cache(
    t_emb: Tensor,
    adaln_lora: Tensor,
    weights: Mapping[str, Tensor],
    *,
    model_channels: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    fl_down = weights["final_layer.adaln_modulation.1.weight"].to(
        device=t_emb.device, dtype=dtype
    )
    fl_up = weights["final_layer.adaln_modulation.2.weight"].to(
        device=t_emb.device, dtype=dtype
    )
    fl_mods = (F.silu(t_emb).to(dtype) @ fl_down.t()) @ fl_up.t()
    fl_mods = fl_mods + adaln_lora[:, : 2 * model_channels]
    shift, scale = fl_mods.split(model_channels, dim=-1)
    return shift.contiguous(), scale.contiguous()


def _make_cosmos_block_mod_cache(
    t_emb_silu: Tensor,
    adaln_lora: Tensor,
    weights: Mapping[str, Tensor],
    *,
    num_blocks: int,
    model_channels: int,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    device = t_emb_silu.device
    t = t_emb_silu.to(dtype=dtype)
    lora = adaln_lora.to(device=device, dtype=dtype)

    def build(rel: str) -> Tensor:
        per_block: list[Tensor] = []
        for block_idx in range(num_blocks):
            prefix = f"blocks.{block_idx}.{rel}"
            down = weights[f"{prefix}.1.weight"].to(device=device, dtype=dtype)
            up = weights[f"{prefix}.2.weight"].to(device=device, dtype=dtype)
            per_block.append((((t @ down.t()) @ up.t()) + lora).contiguous())
        return torch.stack(per_block, dim=0).contiguous()

    return (
        build("adaln_modulation_self_attn"),
        build("adaln_modulation_cross_attn"),
        build("adaln_modulation_mlp"),
    )


def _make_cosmos_rope_cache(
    rope_emb: Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    rope_view = rope_emb.permute(1, 0, 2, 3).reshape(
        rope_emb.shape[0], rope_emb.shape[-1]
    )
    return (
        torch.cos(rope_view).to(dtype).contiguous(),
        torch.sin(rope_view).to(dtype).contiguous(),
    )


def _make_cosmos_hdmap_cache(
    hdmap_patched: Tensor,
    weights: Mapping[str, Tensor],
    *,
    model_channels: int,
    dtype: torch.dtype,
) -> Tensor:
    batch, views, frames, tokens, _ = hdmap_patched.shape
    hdmap_flat = (
        hdmap_patched.to(dtype=dtype)
        .reshape(
            batch,
            views * frames * tokens,
            -1,
        )
        .contiguous()
    )
    w_hd = weights["additional_patch_embedding.proj.1.weight"].to(
        device=hdmap_patched.device,
        dtype=dtype,
    )
    if w_hd.shape[0] != model_channels:
        raise RuntimeError(
            "additional_patch_embedding.proj.1.weight output dimension "
            f"{w_hd.shape[0]} does not match model_channels={model_channels}"
        )
    return (hdmap_flat @ w_hd.t()).contiguous()


def _make_cosmos_streaming_workspace(
    *,
    batch: int,
    tokens: int,
    max_attn_tokens: int,
    num_blocks: int,
    model_channels: int,
    heads: int,
    ff: int,
    lora_dim: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    use_sage3_fp8_attention: bool = False,
) -> dict[str, Tensor]:
    """Allocate the caller-owned scratch tensors expected by native_extension.

    Keep this local to the FlashDreams shim: the bundled extension path only
    guarantees the built `.pyd` and may not include optimized native's Python helpers.
    """
    if model_channels % heads != 0:
        raise ValueError("model_channels must be divisible by heads")
    if batch <= 0 or tokens <= 0 or max_attn_tokens <= 0 or num_blocks <= 0:
        raise ValueError(
            "batch, tokens, max_attn_tokens, and num_blocks must be positive"
        )

    head_dim = model_channels // heads
    linear_scratch_features = max(model_channels, ff, 3 * model_channels)
    singleton = (1,)

    bf16_specs: Mapping[str, tuple[int, ...]] = {
        "qkv_row": (tokens, 3 * model_channels),
        "q_row": (tokens, model_channels),
        "k_row": (tokens, model_channels),
        "v_row": (tokens, model_channels),
        "q_bmhk": (tokens, heads, head_dim),
        "k_bmhk": (tokens, heads, head_dim),
        "o_bmhk": (tokens, heads, head_dim),
        "normed": (tokens, model_channels),
        "ffn_intermediate": (tokens, ff),
        "lora_hidden_sa": (batch, lora_dim),
        "lora_hidden_ca": (batch, lora_dim),
        "lora_hidden_mlp": (batch, lora_dim),
        "lora_hidden_all": (num_blocks * 3, batch, lora_dim),
        "mods_sa": (batch, 3 * model_channels),
        "mods_ca": (batch, 3 * model_channels),
        "mods_mlp": (batch, 3 * model_channels),
        "mods_all": (num_blocks * 3, batch, 3 * model_channels),
        "attn_scores_bf16": singleton,
        "attn_score_c_bf16": singleton,
        "attn_o_bhmd_bf16": (batch, heads, tokens, head_dim),
        "attn_o_c_bf16": (batch, heads, tokens, head_dim),
    }
    u8_specs: Mapping[str, tuple[int, ...]] = {
        "linear_fp8_scratch": (tokens, linear_scratch_features),
        "attn_q_fp8": (batch, tokens, heads, head_dim),
        "attn_k_fp8": (batch, max_attn_tokens, heads, head_dim),
        "attn_v_fp8": (batch, max_attn_tokens, heads, head_dim),
        "attn_q_bhmd_fp8": (batch, heads, tokens, head_dim),
        "attn_k_bhmd_fp8": (batch, heads, max_attn_tokens, head_dim),
        "attn_v_bhmd_fp8": (batch, heads, max_attn_tokens, head_dim),
        "attn_v_bhdm_fp8": (batch, heads, head_dim, max_attn_tokens),
        "attn_probs_fp8": singleton,
    }
    fp8_specs: Mapping[str, tuple[int, ...]] = {}
    if use_sage3_fp8_attention:
        sage3_q_tokens = ((tokens + 127) // 128) * 128
        u8_specs = dict(u8_specs)
        u8_specs["attn_q_sage3_fp4"] = (batch, heads, sage3_q_tokens, head_dim // 2)
        fp8_specs = {
            "attn_q_sage3_sf": (batch, heads, sage3_q_tokens, head_dim // 16),
        }
    fp16_specs: Mapping[str, tuple[int, ...]] = {
        "linear_half_scratch": (tokens, linear_scratch_features),
        "attn_scores_half": singleton,
        "attn_o_bhmd_half": (batch, heads, tokens, head_dim),
        "attn_o_half": (batch, tokens, heads, head_dim),
    }

    workspace: dict[str, Tensor] = {}
    workspace.update(
        {
            name: torch.empty(shape, device=device, dtype=dtype)
            for name, shape in bf16_specs.items()
        }
    )
    workspace.update(
        {
            name: torch.empty(shape, device=device, dtype=torch.uint8)
            for name, shape in u8_specs.items()
        }
    )
    workspace.update(
        {
            name: torch.empty(shape, device=device, dtype=torch.float8_e4m3fn)
            for name, shape in fp8_specs.items()
        }
    )
    workspace.update(
        {
            name: torch.empty(shape, device=device, dtype=torch.float16)
            for name, shape in fp16_specs.items()
        }
    )
    workspace["attn_tc_scale"] = torch.ones((12,), device=device, dtype=torch.float32)
    return workspace


class _CosmosNetworkShapeOps(torch.nn.Module):
    """Lightweight replacement for CosmosDiTNetwork patch/cache helpers.

    FP8 native execution can drop the full PyTorch DiT after the first rollout,
    but scene switches still need fresh cross-attention K/V from the new text
    context. Keep only the small projection subset needed for cache init.
    """

    def __init__(
        self,
        config: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cache_templates: tuple[CosmosDiTNetworkCache, ...] = (),
        cross_cache_weights: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self._cache_templates = cache_templates
        self._cache_template_index = 0
        self._cross_cache_weights = dict(cross_cache_weights or {})
        self._device_anchor = torch.nn.Parameter(
            torch.empty(0, device=device, dtype=dtype),
            requires_grad=False,
        )

    def set_cache_templates(
        self,
        cache_templates: tuple[CosmosDiTNetworkCache, ...],
    ) -> None:
        self._cache_templates = cache_templates
        self._cache_template_index = 0

    def reset_cache_template_cursor(self) -> None:
        self._cache_template_index = 0

    def initialize_cache(
        self,
        chunk_size: int,
        window_size: int,
        sink_size: int,
        context: Tensor,
    ) -> CosmosDiTNetworkCache:
        """Clone the geometry template and rebuild prompt-dependent K/V."""
        if self._cache_template_index >= len(self._cache_templates):
            raise RuntimeError(
                "optimized native FP8 DiT released the upstream network before a "
                "cache template was captured. Start one rollout before "
                "resetting; keep at least one captured cache template per rollout."
            )
        template = self._cache_templates[self._cache_template_index]
        self._cache_template_index += 1
        first_self_attn = template.block_caches[0].self_attn
        if (
            first_self_attn.chunk_size != chunk_size
            or first_self_attn.window_size != window_size
            or first_self_attn.sink_size != sink_size
        ):
            raise RuntimeError(
                "optimized native FP8 DiT cache template geometry does not match "
                "the requested rollout reset geometry."
            )
        cache = _clone_network_cache(template)
        context = self._project_crossattn_context(context)
        for block_idx, block_cache in enumerate(cache.block_caches):
            block_cache.cross_attn = self._make_cross_attn_cache(block_idx, context)
        return cache

    def _weight(self, key: str, *, like: Tensor) -> Tensor:
        try:
            weight = self._cross_cache_weights[key]
        except KeyError as exc:
            raise RuntimeError(
                "optimized native FP8 DiT cannot rebuild cross-attention cache "
                f"after releasing the PyTorch network; missing weight {key!r}."
            ) from exc
        return weight.to(device=like.device, dtype=like.dtype)

    def _project_crossattn_context(self, context: Tensor) -> Tensor:
        if not getattr(self.config, "use_crossattn_projection", False):
            return context
        weight = self._weight("crossattn_proj.0.weight", like=context)
        bias = self._weight("crossattn_proj.0.bias", like=context)
        return F.gelu(F.linear(context, weight, bias))

    def _make_cross_attn_cache(self, block_idx: int, context: Tensor) -> BlockKVCache:
        prefix = f"blocks.{block_idx}.cross_attn."
        k_weight = self._weight(prefix + "k_proj.weight", like=context)
        v_weight = self._weight(prefix + "v_proj.weight", like=context)
        k_norm_weight = self._weight(prefix + "k_norm.weight", like=context)

        batch_shape = context.shape[:-2]
        batch_size = math.prod(batch_shape)
        token_count = int(context.shape[-2])
        inner_dim = int(k_weight.shape[0])
        num_heads = int(self.config.num_heads)
        if inner_dim % num_heads != 0:
            raise RuntimeError(
                f"cross-attention inner_dim={inner_dim} is not divisible by "
                f"num_heads={num_heads}"
            )
        head_dim = inner_dim // num_heads
        k = F.linear(context, k_weight).reshape(
            batch_size, token_count, num_heads, head_dim
        )
        k = F.rms_norm(k, (head_dim,), weight=k_norm_weight, eps=1e-6)
        v = F.linear(context, v_weight).reshape(
            batch_size, token_count, num_heads, head_dim
        )
        return BlockKVCache.from_tensor(k.contiguous(), v.contiguous(), seq_dim=-3)

    def patchify_and_maybe_split_cp(
        self,
        x: Tensor,
        process_groups: list[Any | None] | None = None,
        cp_dims: list[int | None] | None = None,
        flatten_thw: bool = False,
    ) -> Tensor:
        assert x.ndim == 6, f"x must be a 6D tensor, but got shape {x.shape}"
        pattern = (
            "... v (t kt) c (h kh) (w kw) -> ... v (t h w) (c kt kh kw)"
            if flatten_thw
            else "... v (t kt) c (h kh) (w kw) -> ... v t (h w) (c kt kh kw)"
        )
        x = rearrange(
            x,
            pattern,
            kt=self.config.patch_temporal,
            kh=self.config.patch_spatial,
            kw=self.config.patch_spatial,
        )
        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups)
            for cp_dim, process_group in zip(cp_dims, process_groups, strict=True):
                if process_group is not None:
                    assert cp_dim is not None
                    x = split_inputs_cp(x, seq_dim=cp_dim, cp_group=process_group)
        return x

    def unpatchify_and_maybe_gather_cp(
        self,
        pH: int,
        pW: int,
        x: Tensor,
        process_groups: list[Any | None] | None = None,
        cp_dims: list[int | None] | None = None,
        flatten_thw: bool = False,
    ) -> Tensor:
        if flatten_thw:
            pattern = "b v (t h w) (c kt kh kw) -> b v (t kt) c (h kh) (w kw)"
            assert x.ndim == 4, f"x must be a 4D tensor, but got shape {x.shape}"
        else:
            pattern = "b v t (h w) (c kt kh kw) -> b v (t kt) c (h kh) (w kw)"
            assert x.ndim == 5, f"x must be a 5D tensor, but got shape {x.shape}"
        if process_groups is not None:
            assert cp_dims is not None and len(cp_dims) == len(process_groups)
            for cp_dim, process_group in zip(cp_dims, process_groups, strict=True):
                if process_group is not None:
                    assert cp_dim is not None
                    x = cat_outputs_cp(x, seq_dim=cp_dim, cp_group=process_group)
        return rearrange(
            x,
            pattern,
            h=pH,
            w=pW,
            kt=self.config.patch_temporal,
            kh=self.config.patch_spatial,
            kw=self.config.patch_spatial,
        )


def prepare_cosmos_streaming_weights(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Add per-block fused self-attention QKV weights for native_extension.

    This lives alongside the public native helper so deployment only needs the
    built ``native_extension`` binary and this source tree.
    """
    weights = dict(state_dict)
    q_suffix = "self_attn.q_proj.weight"
    for q_key, q_weight in list(weights.items()):
        if not q_key.endswith(q_suffix):
            continue

        prefix = q_key[: -len(q_suffix)]
        fused_key = prefix + "self_attn.qkv_proj.weight"
        if fused_key in weights:
            weights[fused_key] = weights[fused_key].contiguous()
            continue

        k_key = prefix + "self_attn.k_proj.weight"
        v_key = prefix + "self_attn.v_proj.weight"
        if k_key not in weights or v_key not in weights:
            raise KeyError(f"Missing K/V weights for fused QKV key {fused_key!r}")

        weights[fused_key] = torch.cat(
            [q_weight, weights[k_key], weights[v_key]], dim=0
        ).contiguous()

    for key, weight in list(weights.items()):
        if key.endswith(".weight_prepared"):
            weights[key] = weight.contiguous()
            continue
        if key.endswith(_COSMOS_PREPARED_SUFFIXES):
            weights[f"{key}_prepared"] = weight.t().contiguous()

    return {k: v.contiguous() for k, v in weights.items()}


def _seq_slice(ndim: int, seq_dim: int, start: int, end: int) -> tuple[slice, ...]:
    idx = [slice(None)] * ndim
    idx[seq_dim] = slice(start, end)
    return tuple(idx)


def _roll_fp8_cache_left_like_block_cache(
    tensor: Tensor,
    block_cache: Any,
    *,
    seq_dim: int | None = None,
) -> None:
    actual_seq_dim = block_cache.seq_dim if seq_dim is None else seq_dim
    total_size = tensor.shape[actual_seq_dim]
    tokens_to_keep = int(block_cache.window_size - block_cache.chunk_size)
    if tokens_to_keep <= 0:
        return
    src_start = int(block_cache.sink_size + block_cache.chunk_size)
    src_end = total_size
    dst_start = int(block_cache.sink_size)
    dst_end = int(block_cache.sink_size + tokens_to_keep)
    tensor[_seq_slice(tensor.ndim, actual_seq_dim, dst_start, dst_end)].copy_(
        tensor[_seq_slice(tensor.ndim, actual_seq_dim, src_start, src_end)].clone()
    )


def compute_self_attn_write_start(self_attn_cache: Any) -> int:
    """The cursor optimized_dit_forward writes K/V at. Three cases
    (matching ``BlockKVCache.update``):

      * steady-state: write at the rightmost slot, after the left-roll;
      * filling, advancing chunk: append at ``_n_cached``;
      * filling, same chunk: overwrite the just-written rightmost slot.
        This last case fires when ``predict_flow`` runs multiple times
        per AR step (once per scheduler step).

    Must be called AFTER ``before_update(chunk_idx)`` so
    ``_curr_chunk_idx`` is set and any steady-state roll has already
    happened.
    """
    if self_attn_cache.is_steady_state():
        total_size = self_attn_cache._k.shape[self_attn_cache.seq_dim]
        return int(total_size - self_attn_cache.chunk_size)
    if self_attn_cache._curr_chunk_idx == self_attn_cache._prev_chunk_idx + 1:
        return int(self_attn_cache._n_cached)
    if self_attn_cache._curr_chunk_idx == self_attn_cache._prev_chunk_idx:
        return int(self_attn_cache._n_cached - self_attn_cache.chunk_size)
    raise RuntimeError(
        f"Unexpected cache state: _curr_chunk_idx={self_attn_cache._curr_chunk_idx} "
        f"vs _prev_chunk_idx={self_attn_cache._prev_chunk_idx}"
    )


class OptimizedDiTExecutor:
    """Runs the single-view DiT hot path through the native streaming extension.

    The public transformer keeps ownership of patch/unpatch, masks, CFG,
    scheduler state, and the standard cache lifecycle. This object owns the
    optimized native call, quantized runtime state, and precomputed invariants.
    """

    def __init__(
        self,
        transformer: CosmosTransformer,
        extension: Any,
        *,
        dit_backend: str = "fp8_kvcache_cudnn",
        attention_backend: str = "auto",
    ) -> None:
        self.transformer = transformer
        self.config = transformer.config
        self._native_extension = extension

        config = self.config
        if not hasattr(extension, "optimized_dit_forward"):
            raise RuntimeError(
                "native extension has no optimized_dit_forward entry point"
            )
        if config.num_views > 1:
            raise NotImplementedError(
                "optimized native DiT supports num_views=1 only "
                f"(got {config.num_views}); optimized_dit_forward has "
                "not been validated against the cross-view-attn path."
            )
        supports_block_mod_cache = getattr(
            extension,
            "optimized_dit_supports_block_mod_cache",
            None,
        )
        self._supports_block_mod_cache = (
            bool(supports_block_mod_cache())
            if supports_block_mod_cache is not None
            else False
        )
        supports_hdmap_cache = getattr(
            extension,
            "optimized_dit_supports_hdmap_cache",
            None,
        )
        self._supports_hdmap_cache = (
            bool(supports_hdmap_cache()) if supports_hdmap_cache is not None else False
        )
        self._dit_backend = dit_backend
        if self._dit_backend not in {"bf16", "fp8_kvcache_cudnn"}:
            raise ValueError(
                f"Unsupported optimized native DiT backend {self._dit_backend!r}; "
                "expected 'bf16' or 'fp8_kvcache_cudnn'."
            )
        self._uses_fp8_dit = self._dit_backend == "fp8_kvcache_cudnn"

        net_cfg = config.network
        self._optimized_streaming_config = {
            "num_blocks": net_cfg.num_blocks,
            "num_heads": net_cfg.num_heads,
            "model_channels": net_cfg.model_channels,
            "adaln_lora_dim": net_cfg.adaln_lora_dim,
            "timestep_scale": float(net_cfg.timestep_scale),
        }
        self._requested_attention_backend = (
            (attention_backend or "auto").strip().lower()
        )
        if self._requested_attention_backend not in {
            "auto",
            "cudnn",
            "sage3",
            "sage3_fp8",
        }:
            raise ValueError(
                "--native-attention-backend must be 'auto', 'cudnn', "
                f"'sage3', or 'sage3_fp8' (got {self._requested_attention_backend!r})"
            )
        self._attention_backend = "cudnn"
        self._attention_backend_device: torch.device | None = None

        self._optimized_weights: dict[str, Tensor] | None = None
        self._cross_cache_weights: dict[str, Tensor] | None = None
        self._bf16_runtime: dict[str, Any] | None = None
        self._fp8_runtime: dict[str, Any] | None = None
        self._bf16_runtime_device: torch.device | None = None
        self._fp8_runtime_device: torch.device | None = None
        self._optimized_invariant_cache: dict[
            tuple[Any, ...], _CosmosInvariantTensors
        ] = {}
        self._optimized_rope_cache: dict[tuple[Any, ...], tuple[Tensor, Tensor]] = {}
        self._optimized_rope_freqs_cache: dict[tuple[Any, ...], Tensor] = {}
        self._optimized_hdmap_cache: dict[tuple[Any, ...], Tensor] = {}
        self._optimized_empty_hdmap_cache: dict[tuple[Any, ...], Tensor] = {}
        self._optimized_kv_tensor_lists: dict[int, _CosmosCacheTensorLists] = {}
        self._optimized_runtime_config_id: int | None = None
        self._optimized_last_ar_idx: int | None = None
        self._optimized_scheduler_call_idx = 0
        self._optimized_scheduler_timestep_keys: dict[int, tuple[Any, ...]] = {}
        self._released_network_for_fp8 = False
        self._optimized_network_cache_templates: tuple[CosmosDiTNetworkCache, ...] = ()
        self._optimized_call: CUDAGraphWrapper | None = (
            CUDAGraphWrapper(
                self._predict_flow_ext_impl,
                warmup_iters=config.cuda_graph_warmup_iters,
            )
            if config.use_cuda_graph
            else None
        )

    @property
    def network(self) -> torch.nn.Module:
        return self.transformer.network

    @network.setter
    def network(self, value: torch.nn.Module) -> None:
        cast(Any, self.transformer).network = value

    def _maybe_inject_image(
        self,
        latent: Tensor,
        cache: CosmosTransformerCache,
    ) -> Tensor:
        return self.transformer._maybe_inject_image(latent, cache)

    def _select_mask(self, cache: CosmosTransformerCache) -> Tensor:
        return self.transformer._select_mask(cache)

    def _cuda_device_index(
        self, device: torch.device | int | None
    ) -> tuple[int | None, str]:
        if not torch.cuda.is_available():
            return None, "CUDA is unavailable"
        if isinstance(device, int):
            return device, ""
        if device is None:
            try:
                return int(torch.cuda.current_device()), ""
            except Exception as e:
                return None, f"torch.cuda.current_device() failed: {e}"
        device = torch.device(device)
        if device.type != "cuda":
            return None, f"device {device} is not a CUDA device"
        if device.index is None:
            try:
                return int(torch.cuda.current_device()), ""
            except Exception as e:
                return None, f"torch.cuda.current_device() failed: {e}"
        return int(device.index), ""

    def _sage3_status(
        self, device: torch.device | int | None = None
    ) -> tuple[bool, str]:
        if sys.platform == "win32":
            return False, "Sage3 is not supported on Windows"
        built_fn = getattr(self._native_extension, "sage3_is_built", None)
        supported_fn = getattr(
            self._native_extension, "sage3_is_runtime_supported", None
        )
        if built_fn is None or supported_fn is None:
            return False, "native_extension does not expose Sage3 availability probes"
        try:
            if not bool(built_fn()):
                return False, "native_extension was built with Sage3 stubs"
        except Exception as e:
            return False, f"native_extension.sage3_is_built() failed: {e}"
        device_index, reason = self._cuda_device_index(device)
        if device_index is None:
            return False, reason
        try:
            if not bool(supported_fn(device_index)):
                return False, f"CUDA device {device_index} is not enabled for Sage3"
        except Exception as e:
            return False, f"native_extension.sage3_is_runtime_supported() failed: {e}"
        return True, ""

    def _resolve_attention_backend(
        self,
        requested: str,
        *,
        device: torch.device | int | None = None,
    ) -> str:
        requested = (requested or "auto").strip().lower()
        if requested not in {"auto", "cudnn", "sage3", "sage3_fp8"}:
            raise ValueError(
                "--native-attention-backend must be 'auto', 'cudnn', "
                f"'sage3', or 'sage3_fp8' (got {requested!r})"
            )

        if requested == "auto":
            return "cudnn"

        sage3_available, sage3_reason = self._sage3_status(device)

        if requested == "sage3_fp8" and not self._uses_fp8_dit:
            raise ValueError(
                f"--native-attention-backend={requested} requires "
                "the 'fp8_kvcache_cudnn' optimized native DiT backend."
            )
        if requested == "sage3" and self._uses_fp8_dit:
            raise ValueError(
                "--native-attention-backend=sage3 is not supported with "
                "the fp8 DiT backend; use 'sage3_fp8' or 'auto' for the "
                "default cuDNN path instead."
            )
        if requested in {"sage3", "sage3_fp8"} and not sage3_available:
            raise ValueError(
                f"--native-attention-backend={requested} requested Sage3, "
                f"but {sage3_reason}. Use --native-attention-backend=cudnn "
                "for the portable cuDNN attention path."
            )
        return requested

    def _resolve_runtime_attention_backend(self, device: torch.device | int) -> None:
        device_key = torch.device(device)
        if self._attention_backend_device == device_key:
            return
        self._attention_backend = self._resolve_attention_backend(
            self._requested_attention_backend,
            device=device_key,
        )
        self._attention_backend_device = device_key

    def after_initialize_autoregressive_cache(
        self,
        cache: CosmosTransformerCache,
    ) -> None:
        reset_template_cursor = getattr(
            self.network, "reset_cache_template_cursor", None
        )
        if callable(reset_template_cursor):
            reset_template_cursor()
        self._capture_network_cache_templates(cache)
        self._bf16_runtime = None
        self._fp8_runtime = None
        self._bf16_runtime_device = None
        self._fp8_runtime_device = None
        self._attention_backend_device = None
        self._optimized_invariant_cache.clear()
        self._optimized_rope_cache.clear()
        self._optimized_rope_freqs_cache.clear()
        self._optimized_hdmap_cache.clear()
        self._optimized_empty_hdmap_cache.clear()
        self._optimized_kv_tensor_lists.clear()
        self._optimized_runtime_config_id = None
        self._optimized_last_ar_idx = None
        self._optimized_scheduler_call_idx = 0
        self._optimized_scheduler_timestep_keys.clear()
        if self._optimized_call is not None:
            self._optimized_call.reset()

    def _capture_network_cache_templates(self, cache: CosmosTransformerCache) -> None:
        templates = [_clone_network_cache(cache.network_cache)]
        if cache.network_cache_uncond is not None:
            templates.append(_clone_network_cache(cache.network_cache_uncond))
        self._optimized_network_cache_templates = tuple(templates)
        set_cache_templates = getattr(self.network, "set_cache_templates", None)
        if callable(set_cache_templates):
            set_cache_templates(self._optimized_network_cache_templates)

    def _ensure_weights_snapshot(self) -> dict[str, Tensor]:
        """Snapshot ``self.network.state_dict()`` once. Idempotent."""
        if self._optimized_weights is not None:
            return self._optimized_weights
        if self._uses_fp8_dit:
            raise RuntimeError(
                "FP8 weights not injected. Run the offline exporter and pass "
                "fp8_prepared_path through build_fp8_dit()."
            )
        # Non-FP8 (bf16 native) path retained as-is.
        state_dict = self.network.state_dict()
        self._snapshot_cross_cache_weights(state_dict)
        self._optimized_weights = prepare_cosmos_streaming_weights(state_dict)
        return self._optimized_weights

    def _snapshot_cross_cache_weights(self, state_dict: Mapping[str, Tensor]) -> None:
        if self._cross_cache_weights is not None:
            return
        keys: list[str] = []
        if getattr(self.network.config, "use_crossattn_projection", False):
            keys.extend(["crossattn_proj.0.weight", "crossattn_proj.0.bias"])
        for block_idx in range(int(self.config.network.num_blocks)):
            prefix = f"blocks.{block_idx}.cross_attn."
            keys.extend(
                [
                    prefix + "k_proj.weight",
                    prefix + "v_proj.weight",
                    prefix + "k_norm.weight",
                ]
            )
        self._cross_cache_weights = {
            key: state_dict[key].detach().contiguous() for key in keys
        }

    def _drop_redundant_bf16_prepared_weights(self) -> None:
        """Remove BF16 prepared aliases superseded by FP8 prepared weights."""
        if self._optimized_weights is None:
            return
        for key in list(self._optimized_weights):
            if key.endswith(".weight_prepared"):
                self._optimized_weights.pop(key, None)

    def _release_network_after_fp8_snapshot(self) -> None:
        """Free the upstream BF16 DiT once FP8 weights/caches are ready.

        The inherited network is needed to load the checkpoint, patchify cache
        inputs, and initialize the FlashDreams KV caches. After the first FP8
        predict_flow call, native_extension consumes only `_optimized_weights` plus the
        cache tensors, so keeping the full PyTorch DiT resident just duplicates
        several GiB of parameters and graph-wrapper state.
        """
        if not self._uses_fp8_dit or self._released_network_for_fp8:
            return
        shape_ops = _CosmosNetworkShapeOps(
            self.network.config,
            device=next(self.network.parameters()).device,
            dtype=self.config.dtype,
            cache_templates=self._optimized_network_cache_templates,
            cross_cache_weights=self._cross_cache_weights,
        )
        # After release, optimized native's own _optimized_call owns CUDA graph capture.
        # The parent graph wrappers point at the freed PyTorch network and
        # must stay disabled when a reset initializes a fresh cache.
        self.transformer._use_cuda_graph = False
        cast(Any, self.transformer)._network_call = None
        cast(Any, self.transformer)._network_call_uncond = None
        self.network = shape_ops
        self._released_network_for_fp8 = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_fp8_runtime(
        self,
        *,
        k_cross: list[Tensor],
        v_cross: list[Tensor],
        k_self: list[Tensor],
        v_self: list[Tensor],
        tokens: int,
        cache: CosmosTransformerCache,
    ) -> dict[str, Any]:
        if not self._uses_fp8_dit:
            return {}
        device = k_self[0].device
        self._resolve_runtime_attention_backend(device)
        if self._fp8_runtime is not None and self._fp8_runtime_device == device:
            return self._fp8_runtime
        self._fp8_runtime = None
        self._fp8_runtime_device = None
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("torch.float8_e4m3fn is required for fp8_kvcache_cudnn")

        heads = self.config.network.num_heads
        model_channels = self.config.network.model_channels
        ff = int(4 * model_channels)
        lora_dim = self.config.network.adaln_lora_dim
        max_attn_tokens = max(
            tokens,
            *(int(t.size(1)) for t in k_cross),
            *(int(t.size(1)) for t in k_self),
        )
        use_sage3_fp8_attention = self._attention_backend == "sage3_fp8"
        workspace = _make_cosmos_streaming_workspace(
            batch=int(k_self[0].size(0)),
            tokens=tokens,
            max_attn_tokens=max_attn_tokens,
            num_blocks=self.config.network.num_blocks,
            model_channels=model_channels,
            heads=heads,
            ff=ff,
            lora_dim=lora_dim,
            device=device,
            dtype=torch.bfloat16,
            use_sage3_fp8_attention=use_sage3_fp8_attention,
        )

        cfg: dict[str, Any] = {
            "cosmos_linear_backend": "fp8",
            "cosmos_attention_backend": (
                self._attention_backend
                if use_sage3_fp8_attention
                else "fp8_cudnn"
            ),
            "cosmos_kv_cache_backend": "fp8",
            "cosmos_quantized_prepared": True,
            "cosmos_quantized_prepared_strict": True,
            "cosmos_workspace": workspace,
            "cosmos_attn_tc_scale_is_ones": True,
        }

        if use_sage3_fp8_attention:
            sage3_cross = [
                self._native_extension.sage3_quantize_cross_kv_bf16(k, v)
                for k, v in zip(k_cross, v_cross, strict=True)
            ]
            k_cross_sage3_fp4 = [tensors[0] for tensors in sage3_cross]
            v_cross_sage3_fp4 = [tensors[1] for tensors in sage3_cross]
            k_cross_sage3_sf = [tensors[2] for tensors in sage3_cross]
            v_cross_sage3_sf = [tensors[3] for tensors in sage3_cross]
            k_cross_fp8: list[Tensor] = []
            v_cross_fp8: list[Tensor] = []
        else:
            k_cross_fp8 = [
                t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                for t in k_cross
            ]
            v_cross_fp8 = [
                t.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
                for t in v_cross
            ]
            k_cross_sage3_fp4 = []
            v_cross_sage3_fp4 = []
            k_cross_sage3_sf = []
            v_cross_sage3_sf = []

        k_self_fp8 = [torch.zeros_like(t, dtype=torch.uint8) for t in k_self]
        v_self_fp8 = [torch.zeros_like(t, dtype=torch.uint8) for t in v_self]
        cfg.update(
            {
                "k_cross_fp8_caches": k_cross_fp8,
                "v_cross_fp8_caches": v_cross_fp8,
                "k_self_fp8_caches": k_self_fp8,
                "v_self_fp8_caches": v_self_fp8,
                "_last_rolled": {},
            }
        )
        if use_sage3_fp8_attention:
            cfg.update(
                {
                    "k_cross_sage3_fp4_caches": k_cross_sage3_fp4,
                    "v_cross_sage3_fp4_caches": v_cross_sage3_fp4,
                    "k_cross_sage3_sf_caches": k_cross_sage3_sf,
                    "v_cross_sage3_sf_caches": v_cross_sage3_sf,
                }
            )
        fp8_sdpa_layout = (
            os.environ.get("OMNIDREAMS_DIT_FP8_SDPA_LAYOUT", "").strip().lower()
        )
        if use_sage3_fp8_attention and fp8_sdpa_layout == "bhmd":
            _LOGGER.warning(
                "OMNIDREAMS_DIT_FP8_SDPA_LAYOUT=bhmd is ignored for the "
                "SageAttention-3 path with FP8 DiT state; sage3_fp8 uses "
                "Sage3 FP4 cross-attention KV cache layout."
            )
        if not use_sage3_fp8_attention and fp8_sdpa_layout == "bhmd":
            cfg.update(
                {
                    "k_cross_fp8_bhmd_caches": [
                        t.view(torch.float8_e4m3fn)
                        .permute(0, 2, 1, 3)
                        .contiguous()
                        .view(torch.uint8)
                        for t in cfg["k_cross_fp8_caches"]
                    ],
                    "v_cross_fp8_bhmd_caches": [
                        t.view(torch.float8_e4m3fn)
                        .permute(0, 2, 1, 3)
                        .contiguous()
                        .view(torch.uint8)
                        for t in cfg["v_cross_fp8_caches"]
                    ],
                    "k_self_fp8_bhmd_caches": [
                        torch.zeros(
                            (t.size(0), t.size(2), t.size(1), t.size(3)),
                            device=t.device,
                            dtype=torch.uint8,
                        )
                        for t in k_self
                    ],
                    "v_self_fp8_bhmd_caches": [
                        torch.zeros(
                            (t.size(0), t.size(2), t.size(1), t.size(3)),
                            device=t.device,
                            dtype=torch.uint8,
                        )
                        for t in v_self
                    ],
                }
            )
        self._fp8_runtime = cfg
        self._fp8_runtime_device = device
        self._release_network_after_fp8_snapshot()
        return cfg

    def _ensure_bf16_runtime(
        self,
        *,
        k_cross: list[Tensor],
        k_self: list[Tensor],
        tokens: int,
    ) -> dict[str, Any]:
        if self._uses_fp8_dit:
            return {}
        device = k_self[0].device
        self._resolve_runtime_attention_backend(device)
        if self._bf16_runtime is not None and self._bf16_runtime_device == device:
            return self._bf16_runtime
        self._bf16_runtime = None
        self._bf16_runtime_device = None

        heads = self.config.network.num_heads
        model_channels = self.config.network.model_channels
        ff = int(4 * model_channels)
        lora_dim = self.config.network.adaln_lora_dim
        max_attn_tokens = max(
            tokens,
            *(int(t.size(1)) for t in k_cross),
            *(int(t.size(1)) for t in k_self),
        )
        workspace = _make_cosmos_streaming_workspace(
            batch=int(k_self[0].size(0)),
            tokens=tokens,
            max_attn_tokens=max_attn_tokens,
            num_blocks=self.config.network.num_blocks,
            model_channels=model_channels,
            heads=heads,
            ff=ff,
            lora_dim=lora_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        self._bf16_runtime = {
            "cosmos_workspace": workspace,
            "cosmos_attn_tc_scale_is_ones": True,
        }
        if self._attention_backend == "sage3":
            self._bf16_runtime["cosmos_attention_backend"] = self._attention_backend
        self._bf16_runtime_device = device
        return self._bf16_runtime

    def _roll_fp8_self_caches_if_needed(
        self,
        runtime: dict[str, Any],
        cache: CosmosTransformerCache,
    ) -> None:
        if not runtime or "k_self_fp8_caches" not in runtime:
            return
        last_rolled: dict[int, int] = runtime["_last_rolled"]
        block_caches = cache.network_cache.block_caches
        for block_idx, block in enumerate(block_caches):
            self_attn = block.self_attn
            curr = self_attn._curr_chunk_idx
            if curr is None or curr == self_attn._prev_chunk_idx:
                continue
            if last_rolled.get(block_idx) == curr:
                continue
            if not self_attn.is_steady_state():
                continue
            _roll_fp8_cache_left_like_block_cache(
                runtime["k_self_fp8_caches"][block_idx], self_attn
            )
            _roll_fp8_cache_left_like_block_cache(
                runtime["v_self_fp8_caches"][block_idx], self_attn
            )
            if "k_self_fp8_bhmd_caches" in runtime:
                _roll_fp8_cache_left_like_block_cache(
                    runtime["k_self_fp8_bhmd_caches"][block_idx], self_attn, seq_dim=2
                )
                _roll_fp8_cache_left_like_block_cache(
                    runtime["v_self_fp8_bhmd_caches"][block_idx], self_attn, seq_dim=2
                )
            last_rolled[block_idx] = curr

    @staticmethod
    def _timesteps_key(timesteps: Tensor) -> tuple[Any, ...]:
        values = timesteps.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        return (
            str(timesteps.device),
            timesteps.dtype,
            tuple(float(v) for v in values.tolist()),
        )

    def _next_timestep_cache_key(
        self,
        *,
        ar_idx: int,
        timesteps: Tensor,
    ) -> tuple[Any, ...]:
        if self._optimized_last_ar_idx != int(ar_idx):
            self._optimized_last_ar_idx = int(ar_idx)
            self._optimized_scheduler_call_idx = 0

        call_idx = self._optimized_scheduler_call_idx
        self._optimized_scheduler_call_idx += 1

        timestep_key = self._optimized_scheduler_timestep_keys.get(call_idx)
        if timestep_key is None:
            timestep_key = self._timesteps_key(timesteps)
            self._optimized_scheduler_timestep_keys[call_idx] = timestep_key
        return ("scheduler_call", call_idx, timestep_key)

    def _ensure_invariant_tensors(
        self,
        *,
        ar_idx: int,
        timesteps: Tensor,
    ) -> _CosmosInvariantTensors | None:
        if not self._supports_block_mod_cache:
            return None

        weights = self._ensure_weights_snapshot()
        net_cfg = self.config.network
        dtype = self.config.dtype
        key = (
            id(weights),
            net_cfg.num_blocks,
            net_cfg.model_channels,
            net_cfg.adaln_lora_dim,
            dtype,
            self._next_timestep_cache_key(ar_idx=ar_idx, timesteps=timesteps),
        )
        cached = self._optimized_invariant_cache.get(key)
        if cached is not None:
            return cached

        t_emb, t_emb_silu, adaln_lora = _make_cosmos_timestep_cache(
            timesteps,
            weights,
            model_channels=net_cfg.model_channels,
            timestep_scale=float(net_cfg.timestep_scale),
            dtype=dtype,
        )
        final_shift, final_scale = _make_cosmos_final_mod_cache(
            t_emb,
            adaln_lora,
            weights,
            model_channels=net_cfg.model_channels,
            dtype=dtype,
        )
        block_mods_sa, block_mods_ca, block_mods_mlp = _make_cosmos_block_mod_cache(
            t_emb_silu,
            adaln_lora,
            weights,
            num_blocks=net_cfg.num_blocks,
            model_channels=net_cfg.model_channels,
            dtype=dtype,
        )
        cached = _CosmosInvariantTensors(
            t_emb=t_emb,
            t_emb_silu=t_emb_silu,
            adaln_lora=adaln_lora,
            final_shift=final_shift,
            final_scale=final_scale,
            block_mods_sa=block_mods_sa,
            block_mods_ca=block_mods_ca,
            block_mods_mlp=block_mods_mlp,
        )
        self._optimized_invariant_cache[key] = cached
        return cached

    def _ensure_rope_tensors(
        self,
        *,
        ar_idx: int,
        rope_freqs: Tensor,
    ) -> tuple[Tensor, Tensor] | tuple[None, None]:
        if not self._supports_block_mod_cache:
            return None, None

        key = (
            int(ar_idx),
            tuple(int(v) for v in rope_freqs.shape),
            str(rope_freqs.device),
            self.config.dtype,
        )
        cached = self._optimized_rope_cache.get(key)
        if cached is None:
            cached = _make_cosmos_rope_cache(
                rope_freqs,
                dtype=self.config.dtype,
            )
            self._optimized_rope_cache[key] = cached
        return cached

    def _ensure_rope_freqs(
        self,
        *,
        ar_idx: int,
        cache: CosmosTransformerCache,
    ) -> Tensor:
        del ar_idx
        assert cache.rope_freqs is not None, (
            "Cache.start(autoregressive_index) must populate rope_freqs before "
            "predict_flow."
        )
        return cache.rope_freqs

    def _ensure_hdmap_tensor(
        self,
        *,
        ar_idx: int,
        input_for_ext: Tensor | None,
    ) -> Tensor | None:
        if (
            not self._supports_hdmap_cache
            or input_for_ext is None
            or input_for_ext.numel() == 0
        ):
            return None

        weights = self._ensure_weights_snapshot()
        net_cfg = self.config.network
        key = (
            id(weights),
            int(ar_idx),
            tuple(int(v) for v in input_for_ext.shape),
            str(input_for_ext.device),
            self.config.dtype,
        )
        cached = self._optimized_hdmap_cache.get(key)
        if cached is None:
            cached = _make_cosmos_hdmap_cache(
                input_for_ext,
                weights,
                model_channels=net_cfg.model_channels,
                dtype=self.config.dtype,
            )
            self._optimized_hdmap_cache[key] = cached
        return cached

    def _empty_hdmap_tensor(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        key = (str(device), dtype)
        cached = self._optimized_empty_hdmap_cache.get(key)
        if cached is None:
            cached = torch.empty((0,), device=device, dtype=dtype)
            self._optimized_empty_hdmap_cache[key] = cached
        return cached

    def _ensure_kv_tensor_lists(
        self,
        *,
        cache: CosmosTransformerCache,
    ) -> _CosmosCacheTensorLists:
        net_cache = cache.network_cache
        key = id(net_cache)
        cached = self._optimized_kv_tensor_lists.get(key)
        if cached is None:
            block_caches = net_cache.block_caches
            cached = _CosmosCacheTensorLists(
                k_self=[block.self_attn._k for block in block_caches],
                v_self=[block.self_attn._v for block in block_caches],
                k_cross=[block.cross_attn._k for block in block_caches],
                v_cross=[block.cross_attn._v for block in block_caches],
            )
            self._optimized_kv_tensor_lists[key] = cached
        return cached

    def _apply_runtime_config(self, runtime_cfg: dict[str, Any]) -> None:
        if not runtime_cfg:
            return
        cfg_id = id(runtime_cfg)
        if self._optimized_runtime_config_id == cfg_id:
            return
        self._optimized_streaming_config.update(runtime_cfg)
        self._optimized_runtime_config_id = cfg_id

    def _clear_transient_ar_caches(self) -> None:
        """Drop per-AR-step tensors that are only useful within one step."""
        self._optimized_rope_cache.clear()
        self._optimized_rope_freqs_cache.clear()
        self._optimized_hdmap_cache.clear()

    def after_finalize_kv_cache(self) -> None:
        self._clear_transient_ar_caches()

    def _select_optimized_call(self, cache: CosmosTransformerCache) -> Any:
        if self._optimized_call is None:
            return self._predict_flow_ext_impl
        return (
            self._optimized_call.drain
            if cache.autoregressive_index < self.transformer._cuda_graph_capture_ar_idx
            else self._optimized_call
        )

    def _predict_flow_ext_impl(
        self,
        noisy_for_ext: Tensor,
        mask_for_ext: Tensor,
        input_for_ext: Tensor | None,
        hdmap_embed: Tensor | None,
        timestep_b: Tensor,
        rope_freqs: Tensor,
        t_emb: Tensor | None,
        t_emb_silu: Tensor | None,
        adaln_lora: Tensor | None,
        final_shift: Tensor | None,
        final_scale: Tensor | None,
        rope_cos: Tensor | None,
        rope_sin: Tensor | None,
        block_mods_sa: Tensor | None,
        block_mods_ca: Tensor | None,
        block_mods_mlp: Tensor | None,
        k_cross: list[Tensor],
        v_cross: list[Tensor],
        k_self: list[Tensor],
        v_self: list[Tensor],
        write_start: int,
    ) -> Tensor:
        config: dict[str, Any] = dict(self._optimized_streaming_config)
        if hdmap_embed is not None:
            config["cosmos_hdmap_embed"] = hdmap_embed
        if t_emb is not None:
            assert t_emb_silu is not None
            assert adaln_lora is not None
            assert final_shift is not None
            assert final_scale is not None
            assert rope_cos is not None
            assert rope_sin is not None
            assert block_mods_sa is not None
            assert block_mods_ca is not None
            assert block_mods_mlp is not None
            config.update(
                {
                    "cosmos_t_emb": t_emb,
                    "cosmos_t_emb_silu": t_emb_silu,
                    "cosmos_adaln_lora": adaln_lora,
                    "cosmos_final_shift": final_shift,
                    "cosmos_final_scale": final_scale,
                    "cosmos_rope_cos": rope_cos,
                    "cosmos_rope_sin": rope_sin,
                    "cosmos_block_mods_sa": block_mods_sa,
                    "cosmos_block_mods_ca": block_mods_ca,
                    "cosmos_block_mods_mlp": block_mods_mlp,
                }
            )
        return self._native_extension.optimized_dit_forward(
            x_new=noisy_for_ext,
            condition_mask_patched=mask_for_ext,
            hdmap_patched=input_for_ext,
            timesteps=timestep_b,
            rope_emb=rope_freqs,
            k_cross_caches=k_cross,
            v_cross_caches=v_cross,
            k_self_caches=k_self,
            v_self_caches=v_self,
            self_attn_write_start=int(write_start),
            weights=self._ensure_weights_snapshot(),
            config=config,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        """One scheduler step. Identical to the parent except the inner
        network forward runs through ``native_extension.optimized_dit_forward``.
        """
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0, (
            "CosmosTransformerCache.start(autoregressive_index) must be called "
            "before predict_flow (DiffusionModel.generate handles this)."
        )
        return_dtype = noisy_latent.dtype

        rope_freqs = self._ensure_rope_freqs(ar_idx=ar_idx, cache=cache)

        # AR step 0: inject the encoded first-frame latent into the
        # noisy input (parent's helper).
        noisy_latent = self._maybe_inject_image(noisy_latent, cache)
        condition_video_input_mask = self._select_mask(cache)

        self._ensure_weights_snapshot()

        # ---- C++/CUDA forward through native_extension.optimized_dit_forward ----
        # The C++ entry point doesn't drive cache.before_update /
        # after_update; we wrap it so the BlockKVCache state machine
        # evolves the same way the upstream eager path does.
        # The scheduler hands us a 0-D scalar timestep; native_extension's
        # bridge wants ``[B]`` (the upstream PyTorch network is fine
        # with the scalar because its first op is a broadcast multiply).
        # Repeat to ``[B]`` to match the bridge's contract.
        batch_size = noisy_latent.shape[0]
        timestep_b = timestep.reshape(1).expand(batch_size).contiguous()
        invariant_tensors = self._ensure_invariant_tensors(
            ar_idx=ar_idx,
            timesteps=timestep_b,
        )
        rope_cos, rope_sin = self._ensure_rope_tensors(
            ar_idx=ar_idx,
            rope_freqs=rope_freqs,
        )

        # NOTE: upstream's ``CosmosTransformer.predict_flow`` does NOT
        # drive ``network_cache.before_update`` / ``after_update`` -- the
        # cache state machine is advanced once per AR step at the
        # boundary by ``CosmosTransformerCache.start`` / ``.finalize``,
        # which ``DiffusionModel.generate`` invokes around the scheduler
        # loop. ``predict_flow`` is called MULTIPLE times per AR step
        # (one per scheduler timestep) and the inner network just
        # overwrites the rightmost slot each call (see the
        # ``filling-same-chunk`` and ``steady-state`` branches in
        # ``compute_self_attn_write_start``). An older revision of this
        # shim wrapped the ``optimized_dit_forward`` call in
        # ``before_update`` / ``after_update`` and tripped
        # ``BlockKVCache.before_update``'s "Must call after_update()
        # before before_update()" assert on the second scheduler
        # timestep of every AR step. We just consume the post-
        # ``before_update`` state directly here -- the C++ launcher
        # writes K/V in place at ``write_start``, and the boundary
        # ``finalize`` advances the state machine once.
        net_cache = cache.network_cache
        write_start = compute_self_attn_write_start(net_cache.block_caches[0].self_attn)
        kv_lists = self._ensure_kv_tensor_lists(cache=cache)

        # Upstream single-view CosmosTransformer flattens THW to [B, V, L, D].
        # native_extension still consumes the unflattened patch layout [B, V, T, HW, D].
        return_flattened = noisy_latent.ndim == 4
        if return_flattened:
            B, V, L, D = noisy_latent.shape
            cfg = self.config
            transformer = self.transformer
            assert (
                transformer._output_height is not None
                and transformer._output_width is not None
            ), "optimized DiT requires an initialized rollout spatial shape"
            T = cfg.len_t // cfg.network.patch_temporal
            HW = (
                transformer._output_height
                // cfg.network.patch_spatial
                * transformer._output_width
                // cfg.network.patch_spatial
            )
            assert L == T * HW, (
                f"Flattened latent length {L} does not match T*HW={T * HW}"
            )
            noisy_for_ext = noisy_latent.reshape(B, V, T, HW, D)
            mask_for_ext = condition_video_input_mask.reshape(
                B, V, T, HW, condition_video_input_mask.shape[-1]
            )
            input_for_ext = (
                None if input is None else input.reshape(B, V, T, HW, input.shape[-1])
            )
        else:
            noisy_for_ext = noisy_latent
            mask_for_ext = condition_video_input_mask
            input_for_ext = input

        ext_dtype = self.config.dtype
        noisy_for_ext = noisy_for_ext.to(dtype=ext_dtype)
        mask_for_ext = mask_for_ext.to(dtype=ext_dtype)
        if input_for_ext is not None:
            input_for_ext = input_for_ext.to(dtype=ext_dtype)
        hdmap_embed = self._ensure_hdmap_tensor(
            ar_idx=ar_idx,
            input_for_ext=input_for_ext,
        )
        hdmap_for_ext = (
            input_for_ext
            if hdmap_embed is None
            else self._empty_hdmap_tensor(
                device=noisy_for_ext.device,
                dtype=ext_dtype,
            )
        )

        runtime_cfg = self._ensure_fp8_runtime(
            k_cross=kv_lists.k_cross,
            v_cross=kv_lists.v_cross,
            k_self=kv_lists.k_self,
            v_self=kv_lists.v_self,
            tokens=int(noisy_for_ext.shape[2] * noisy_for_ext.shape[3]),
            cache=cache,
        )
        if not runtime_cfg:
            runtime_cfg = self._ensure_bf16_runtime(
                k_cross=kv_lists.k_cross,
                k_self=kv_lists.k_self,
                tokens=int(noisy_for_ext.shape[2] * noisy_for_ext.shape[3]),
            )
        if self._uses_fp8_dit:
            self._roll_fp8_self_caches_if_needed(runtime_cfg, cache)
        self._apply_runtime_config(runtime_cfg)

        predicted_flow = self._select_optimized_call(cache)(
            noisy_for_ext,
            mask_for_ext,
            hdmap_for_ext,
            hdmap_embed,
            timestep_b,
            rope_freqs,
            None if invariant_tensors is None else invariant_tensors.t_emb,
            None if invariant_tensors is None else invariant_tensors.t_emb_silu,
            None if invariant_tensors is None else invariant_tensors.adaln_lora,
            None if invariant_tensors is None else invariant_tensors.final_shift,
            None if invariant_tensors is None else invariant_tensors.final_scale,
            rope_cos,
            rope_sin,
            None if invariant_tensors is None else invariant_tensors.block_mods_sa,
            None if invariant_tensors is None else invariant_tensors.block_mods_ca,
            None if invariant_tensors is None else invariant_tensors.block_mods_mlp,
            kv_lists.k_cross,
            kv_lists.v_cross,
            kv_lists.k_self,
            kv_lists.v_self,
            int(write_start),
        )
        # ----------------------------------------------------------------------

        if return_flattened:
            predicted_flow = predicted_flow.reshape(
                noisy_latent.shape[0],
                noisy_latent.shape[1],
                noisy_latent.shape[2],
                predicted_flow.shape[-1],
            )
        if predicted_flow.dtype != return_dtype:
            predicted_flow = predicted_flow.to(dtype=return_dtype)
        return predicted_flow


__all__ = [
    "OptimizedDiTExecutor",
    "compute_self_attn_write_start",
    "prepare_cosmos_streaming_weights",
]
