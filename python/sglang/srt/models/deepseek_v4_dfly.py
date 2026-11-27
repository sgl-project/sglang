"""DeepSeek-V4.1-Flash DFly2 block-parallel draft model (AngelSpec).

Serves ``AngelSlim/DS-V4.1-Flash-DFly2-B6`` and compatible ``DFlyDeepseekV41DraftModel``
checkpoints on SGLang without allocating any draft KV cache:
  - Each draft layer reuses its paired target layer's sliding-window KV cache
    (128 tokens) and the target's shared ratio-1 global KV cache (top-512
    positions captured from the paired target layer's sparse indexer at the last
    accepted token).
  - Target's post-norm final hidden state is fused into the anchor slot via
    ``MTPFusion``.
  - The 3-layer mHC (``hc_mult=4``) MoE backbone runs over the block in parallel
    with ``DeepseekV2MoE`` (MXFP4 routed experts + BF16 shared expert).
  - Autoregressive intra-block ``HiddenStatesCorrection`` refines slots
    ``1..block_size-1`` via SwiGLU + ``DSV41AttentiveCorrection`` (context read
    on the source target layer + gated causal prefix attention over slots
    ``0..k``).
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
from typing import Callable, Iterable, List, Optional, Tuple

import msgspec
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.utils import get_num_fused_shared_experts
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils import add_prefix
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models import deepseek_v2
from sglang.srt.models.deepseek_v4 import (
    DEEPSEEK_V4_STACKED_PARAMS_MAPPING,
    _apply_wo_a_bf16_matmul,
)
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.dfly_utils import (
    DFlyRequestState,
    build_dfly_extra_topk,
    build_dfly_swa_indices,
)
from sglang.srt.utils import is_sail

logger = logging.getLogger(__name__)

_is_sail = is_sail()
_PAD_NUM_HEADS = 64
_DEFAULT_MAX_BLOCK_SLOTS = 16


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class DFlyRopeConfig(msgspec.Struct, frozen=True, kw_only=True):
    theta: float = 160000.0
    original_seq_len: int = 65536
    factor: float = 16.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0


class DFlyAttentiveCorrectionConfig(msgspec.Struct, frozen=True, kw_only=True):
    source_layer: int = -1
    block_keys: bool = True
    block_mode: str = "gated"
    block_kv_in: str = "hidden_embed"
    max_block_slots: int = _DEFAULT_MAX_BLOCK_SLOTS


class DFlyDeepseekV41Config(msgspec.Struct, frozen=True, kw_only=True):
    num_layers: int
    block_size: int
    causal_block: bool
    mask_token_id: int
    target_kv_layers: Tuple[int, ...]
    target_index_layers: Tuple[int, ...]
    target_index_topk: int
    sliding_window: int
    rope: DFlyRopeConfig
    mtp_fusion: Optional[str]
    hidden_correction_intermediate_size: Optional[int]
    attentive_correction: Optional[DFlyAttentiveCorrectionConfig]
    n_routed_experts: int
    num_experts_per_tok: int


def _probe_checkpoint_num_experts(checkpoint_path: Optional[str]) -> Optional[int]:
    """Read ``model.layers.0.ffn.gate.weight`` row count from safetensors header
    when ``dfly_config`` omits ``n_routed_experts`` (as in
    ``AngelSlim/DS-V4.1-Flash-DFly2-B6``, whose ``text_config`` mirrors the
    target's 384 experts while the draft checkpoint stores 128)."""
    if not checkpoint_path:
        return None
    from sglang.srt.utils import find_local_repo_dir

    local_dir = (
        checkpoint_path
        if os.path.isdir(checkpoint_path)
        else find_local_repo_dir(checkpoint_path)
    )
    if not local_dir or not os.path.isdir(local_dir):
        return None

    gate_key = "model.layers.0.ffn.gate.weight"
    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    shard_file: Optional[str] = None
    if os.path.isfile(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            shard_file = weight_map.get(gate_key)
        except Exception as exc:
            logger.debug("Failed reading DFly index %s: %s", index_path, exc)
    else:
        single = os.path.join(local_dir, "model.safetensors")
        if os.path.isfile(single):
            shard_file = "model.safetensors"

    if not shard_file:
        return None
    shard_path = os.path.join(local_dir, shard_file)
    if not os.path.isfile(shard_path):
        return None
    try:
        from safetensors import safe_open

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if gate_key in f.keys():
                return int(f.get_slice(gate_key).get_shape()[0])
    except Exception as exc:
        logger.debug("Failed probing DFly gate shape from %s: %s", shard_path, exc)
    return None


def parse_dsv41_dfly_config(
    hf_config,
    *,
    block_size_override: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
) -> DFlyDeepseekV41Config:
    raw = getattr(hf_config, "dfly_config", None)
    if not isinstance(raw, dict):
        raise ValueError(
            "DFlyDeepseekV41DraftModel requires a 'dfly_config' dict in config.json."
        )

    kv_layers = tuple(int(x) for x in (raw.get("target_kv_layers") or ()))
    if not kv_layers:
        raise ValueError("dfly_config.target_kv_layers must be a non-empty list.")
    num_layers = len(kv_layers)

    raw_index = raw.get("target_index_layers")
    if raw_index is None:
        raw_index = raw.get("target_index_layer")
    if isinstance(raw_index, int):
        index_layers = (int(raw_index),) * num_layers
    elif raw_index is not None:
        index_layers = tuple(int(x) for x in raw_index)
    else:
        index_layers = kv_layers
    if len(index_layers) != num_layers:
        raise ValueError(
            f"dfly_config.target_index_layers length ({len(index_layers)}) must "
            f"match target_kv_layers length ({num_layers})."
        )

    block_size = int(
        block_size_override
        if block_size_override is not None
        else raw.get("block_size", 6)
    )
    if not 2 <= block_size <= 17:
        raise ValueError(f"DFly requires block_size in [2, 17], got {block_size}.")

    mask_token_id = int(raw["mask_token_id"])
    vocab_size = int(getattr(hf_config, "vocab_size", 0))
    if vocab_size > 0 and not (0 <= mask_token_id < vocab_size):
        raise ValueError(
            f"dfly_config.mask_token_id={mask_token_id} outside vocab [0, {vocab_size})."
        )

    target_index_topk = int(
        raw.get("index_topk")
        or raw.get("target_index_topk")
        or getattr(hf_config, "index_topk", 512)
    )
    sliding_window = int(
        raw.get("context_local_window")
        or getattr(hf_config, "sliding_window", None)
        or getattr(hf_config, "window_size", 128)
    )
    causal_block = bool(raw.get("causal_block", False))

    rope_raw = raw.get("rope") or {}
    rope_scaling = getattr(hf_config, "rope_scaling", None) or {}
    rope = DFlyRopeConfig(
        theta=float(
            rope_raw.get("theta", getattr(hf_config, "compress_rope_theta", 160000.0))
        ),
        original_seq_len=int(
            rope_raw.get(
                "original_seq_len",
                rope_scaling.get("original_max_position_embeddings", 65536),
            )
        ),
        factor=float(rope_raw.get("factor", rope_scaling.get("factor", 16.0))),
        beta_fast=float(rope_raw.get("beta_fast", rope_scaling.get("beta_fast", 32.0))),
        beta_slow=float(rope_raw.get("beta_slow", rope_scaling.get("beta_slow", 1.0))),
    )

    mtp_fusion = raw.get("mtp_fusion", "slot0")
    if mtp_fusion is not None and mtp_fusion not in ("slot0", "slot1", "broadcast"):
        raise ValueError(f"Unsupported dfly_config.mtp_fusion={mtp_fusion!r}.")

    hc_raw = raw.get("hidden_correction")
    if hc_raw is None:
        hc_inter = None
    elif isinstance(hc_raw, dict):
        hc_inter = int(
            hc_raw.get("intermediate_size") or getattr(hf_config, "hidden_size")
        )
    else:
        hc_inter = int(getattr(hf_config, "hidden_size"))

    ac_raw = raw.get("attentive_correction")
    ac_cfg: Optional[DFlyAttentiveCorrectionConfig] = None
    if isinstance(ac_raw, dict):
        ac_cfg = DFlyAttentiveCorrectionConfig(
            source_layer=int(ac_raw.get("source_layer", -1)) % num_layers,
            block_keys=bool(ac_raw.get("block_keys", True)),
            block_mode=str(ac_raw.get("block_mode", "gated")),
            block_kv_in=str(ac_raw.get("block_kv_in", "hidden_embed")),
            max_block_slots=int(
                ac_raw.get("max_block_slots", _DEFAULT_MAX_BLOCK_SLOTS)
            ),
        )
        if ac_cfg.block_keys and ac_cfg.block_mode == "gated":
            if block_size > ac_cfg.max_block_slots:
                raise ValueError(
                    f"DFly block_size={block_size} exceeds attentive_correction "
                    f"max_block_slots={ac_cfg.max_block_slots}."
                )

    # Resolve draft MoE expert count & active experts per token.
    target_n_routed = int(getattr(hf_config, "n_routed_experts", 128))
    target_top_k = int(getattr(hf_config, "num_experts_per_tok", 6))
    n_routed_raw = raw.get("n_routed_experts", raw.get("num_experts"))
    if n_routed_raw is not None:
        n_routed_experts = int(n_routed_raw)
    else:
        probed = _probe_checkpoint_num_experts(
            checkpoint_path or getattr(hf_config, "_name_or_path", None)
        )
        n_routed_experts = probed if probed is not None else target_n_routed

    top_k_raw = raw.get("num_experts_per_tok")
    if top_k_raw is not None:
        num_experts_per_tok = int(top_k_raw)
    elif (
        n_routed_experts < target_n_routed
        and n_routed_experts == 128
        and target_top_k == 6
    ):
        # AngelSlim/DS-V4.1-Flash-DFly2-B6 inherits text_config (384 experts,
        # top-6) verbatim from the target while its draft MoE layers have 128
        # routed experts trained with top-3 routing (AngelSpec dfly2_dsv41_flash).
        num_experts_per_tok = 3
        logger.info(
            "DFlyDeepseekV41DraftModel: detected %d routed experts (target %d); "
            "using num_experts_per_tok=3 (override via dfly_config.num_experts_per_tok).",
            n_routed_experts,
            target_n_routed,
        )
    else:
        num_experts_per_tok = target_top_k

    return DFlyDeepseekV41Config(
        num_layers=num_layers,
        block_size=block_size,
        causal_block=causal_block,
        mask_token_id=mask_token_id,
        target_kv_layers=kv_layers,
        target_index_layers=index_layers,
        target_index_topk=target_index_topk,
        sliding_window=sliding_window,
        rope=rope,
        mtp_fusion=mtp_fusion,
        hidden_correction_intermediate_size=hc_inter,
        attentive_correction=ac_cfg,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
    )


# ---------------------------------------------------------------------------
# Math primitives (bit-exact with AngelSpec dsv41_mla / dsv41_mhc)
# ---------------------------------------------------------------------------


class DFlyDSV41RMSNorm(nn.Module):
    """FP32-accumulated RMSNorm matching ``DSV41RMSNorm`` (supports ``eps=1e-20``)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * h).to(x.dtype)


def compute_dsv41_yarn_inv_freq(
    rope_dim: int,
    rope: DFlyRopeConfig,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    pos_freqs = rope.theta ** (
        torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim
    )
    extrapolation = 1.0 / pos_freqs
    if rope.original_seq_len <= 0:
        return extrapolation
    interpolation = 1.0 / (rope.factor * pos_freqs)

    def _pair_turning(rotations: float) -> float:
        return (
            rope_dim
            * math.log(rope.original_seq_len / (rotations * 2 * math.pi))
            / (2 * math.log(rope.theta))
        )

    low = max(math.floor(_pair_turning(rope.beta_fast)), 0)
    high = min(math.ceil(_pair_turning(rope.beta_slow)), rope_dim - 1)
    if low == high:
        high += 0.001
    ramp = (
        (torch.arange(rope_dim // 2, device=device, dtype=torch.float32) - low)
        / (high - low)
    ).clamp(0.0, 1.0)
    keep = 1.0 - ramp
    return interpolation * (1.0 - keep) + extrapolation * keep


def compute_dsv41_rope_cos_sin(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    angles = positions.float().unsqueeze(-1) * inv_freq
    return angles.cos(), angles.sin()


def apply_dsv41_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing ``2 * cos.shape[-1]`` channels as adjacent ``(real, imag)`` pairs."""
    rope_dim = cos.shape[-1] * 2
    tail = x[..., -rope_dim:].float().unflatten(-1, (-1, 2))
    real, imag = tail[..., 0], tail[..., 1]
    if inverse:
        sin = -sin
    rotated = torch.stack(
        [real * cos - imag * sin, real * sin + imag * cos], dim=-1
    ).flatten(-2)
    return torch.cat([x[..., :-rope_dim], rotated.to(x.dtype)], dim=-1)


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int,
    iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult]
    )
    comb = (mixes[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :]).unflatten(
        -1, (hc_mult, hc_mult)
    )
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _hc_mixes(
    h: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    iters: int,
    eps: float,
    norm_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = h.flatten(-2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, fn.float()) * rsqrt
    return _hc_split_sinkhorn(
        mixes, scale.float(), base.float(), h.shape[-2], iters, eps
    )


def _hc_pre(h: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...k,...kd->...d", pre_mix.float(), h.float()).to(h.dtype)


def _hc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    mixed = torch.einsum("...jk,...jd->...kd", comb, residual.float())
    return (post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed).to(x.dtype)


def _load_tp_row_slice(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    """Slice ``loaded_weight`` along dim 0 for the current TP rank."""
    parallel = get_parallel()
    tp_size = parallel.tp_size
    tp_rank = parallel.tp_rank
    if tp_size == 1:
        param.data.copy_(loaded_weight)
        return
    shard_size = param.shape[0]
    start = tp_rank * shard_size
    param.data.copy_(loaded_weight.narrow(0, start, shard_size))


# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------


class DFlyDSV41MTPFusion(nn.Module):
    """Fuses anchor embedding with the target's post-norm final hidden state."""

    def __init__(self, hidden_size: int, mode: str, rms_norm_eps: float) -> None:
        super().__init__()
        self.mode = mode
        self.gate_slot = 1 if mode == "slot1" else 0
        self.enorm = DFlyDSV41RMSNorm(hidden_size, eps=rms_norm_eps)
        self.hnorm = DFlyDSV41RMSNorm(hidden_size, eps=rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            2 * hidden_size,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix="mtp_fusion.eh_proj",
        )
        self.gate = nn.Parameter(torch.zeros(hidden_size))
        self.mask_gate = (
            nn.Parameter(torch.zeros(hidden_size)) if mode == "broadcast" else None
        )

    def forward(
        self, embeds: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        """``embeds``: ``[bs, width, D]``, ``target_hidden``: ``[bs, D]``."""
        anchors = embeds[:, 0]
        fused_in = torch.cat(
            [self.enorm(anchors), self.hnorm(target_hidden.to(anchors.dtype))], dim=-1
        )
        fused, _ = self.eh_proj(fused_in)
        out = embeds.clone()
        if self.mask_gate is not None:
            out[:, 0] = out[:, 0] + self.gate.to(fused.dtype) * fused
            out[:, 1:] = out[:, 1:] + self.mask_gate.to(fused.dtype) * fused.unsqueeze(
                1
            )
        else:
            out[:, self.gate_slot] = (
                out[:, self.gate_slot] + self.gate.to(fused.dtype) * fused
            )
        return out


class DFlyDSV41Attention(nn.Module):
    """DeepSeek-V4.1 latent MQA draft attention reusing target SWA + C1 caches."""

    def __init__(
        self,
        config: DeepSeekV4Config,
        dfly_config: DFlyDeepseekV41Config,
        layer_idx: int,
        target_kv_layer: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        self.tp_size = parallel.tp_size
        self.tp_rank = parallel.tp_rank

        self.hidden_size = int(config.hidden_size)
        self.n_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.rope_dim = int(config.qk_rope_head_dim)
        self.q_lora_rank = int(config.q_lora_rank)
        self.o_lora_rank = int(config.o_lora_rank)
        self.n_groups = int(config.o_groups)
        self.eps = float(config.rms_norm_eps)
        self.scaling = self.head_dim**-0.5
        self.layer_idx = layer_idx
        self.target_kv_layer = target_kv_layer

        if self.n_groups % self.tp_size != 0 or self.n_heads % self.tp_size != 0:
            raise ValueError(
                f"DFlyDSV41Attention requires tp_size ({self.tp_size}) to divide "
                f"o_groups ({self.n_groups}) and num_attention_heads ({self.n_heads})."
            )
        self.n_local_heads = self.n_heads // self.tp_size
        self.n_local_groups = self.n_groups // self.tp_size
        self.heads_per_group = self.n_heads // self.n_groups

        self.wq_a = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wq_a", prefix),
        )
        self.q_norm = DFlyDSV41RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wq_b", prefix),
        )
        self.wkv = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv", prefix),
        )
        self.kv_norm = DFlyDSV41RMSNorm(self.head_dim, eps=self.eps)

        self.wo_a = nn.Linear(
            self.heads_per_group * self.head_dim,
            self.n_local_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_a.weight.weight_loader = _load_tp_row_slice

        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=None,
            reduce_results=True,
            prefix=add_prefix("wo_b", prefix),
        )
        self.attn_sink = nn.Parameter(
            torch.zeros(self.n_heads, dtype=torch.float32), requires_grad=False
        )

        # RadixAttention bound to ``target_kv_layer`` so DeepSeekV4AttnBackend
        # reads that target layer's SWA pool and its ratio-1 global KV source.
        self.attn = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=target_kv_layer,
            quant_config=None,
            prefix=add_prefix("attn", prefix),
        )

        self.rope_cfg = dfly_config.rope
        self.register_buffer(
            "inv_freq",
            compute_dsv41_yarn_inv_freq(self.rope_dim, dfly_config.rope),
            persistent=False,
        )
        self._freqs_cis: Optional[torch.Tensor] = None

    def _local_attn_sink(self) -> torch.Tensor:
        if _is_sail:
            return self.attn_sink
        start = self.tp_rank * self.n_local_heads
        local = self.attn_sink[start : start + self.n_local_heads]
        if self.n_local_heads < _PAD_NUM_HEADS:
            return F.pad(local, (0, _PAD_NUM_HEADS - self.n_local_heads), value=0.0)
        return local

    def _ensure_freqs_cis(
        self, positions: torch.Tensor, max_pos_hint: int = 0
    ) -> torch.Tensor:
        need = max(int(max_pos_hint), 1024)
        if (
            self._freqs_cis is None
            or self._freqs_cis.device != positions.device
            or self._freqs_cis.shape[0] <= need
        ):
            inv_freq = self.inv_freq.to(positions.device)
            t = torch.arange(need + 1, device=positions.device, dtype=torch.float32)
            angles = torch.outer(t, inv_freq)
            self._freqs_cis = torch.polar(torch.ones_like(angles), angles)
        return self._freqs_cis

    def _project_q(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q_lora, _ = self.wq_a(x)
        q_lora = self.q_norm(q_lora)
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        return apply_dsv41_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))

    def _project_output(
        self,
        o: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        is_target_verify: bool = True,
    ) -> torch.Tensor:
        o = apply_dsv41_rope(o, cos.unsqueeze(1), sin.unsqueeze(1), inverse=True)
        o = o.view(
            o.shape[0],
            self.n_local_groups,
            self.heads_per_group * self.head_dim,
        )
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if o.is_cuda:
            o = _apply_wo_a_bf16_matmul(
                o,
                wo_a,
                is_decode=False,
                is_target_verify=is_target_verify,
                fast_path=True,
            )
        else:
            o = torch.einsum("ngd,grd->ngr", o.float(), wo_a.float()).to(o.dtype)
        value = o.reshape(o.shape[0], self.n_local_groups * self.o_lora_rank)
        out, _ = self.wo_b(value)
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        swa_page_indices: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        extra_indices: torch.Tensor,
        extra_topk_lengths: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        attn_backend = get_attn_backend()
        pool = get_token_to_kv_pool()

        kv_raw, _ = self.wkv(hidden_states)
        kv_latent = apply_dsv41_rope(self.kv_norm(kv_raw), cos, sin)

        # Write draft block KV into target_kv_layer's temporary verify SWA slots.
        # Under RequestWindow this writes into workspace without calling commit();
        # under paged SWA this writes into swa_out_cache_loc which target verify
        # overwrites when target layer target_kv_layer runs later in the pass.
        swa_loc = attn_backend.get_swa_out_cache_loc(forward_batch)
        pool.set_swa_key_buffer_radix_fused(
            layer_id=self.target_kv_layer,
            swa_loc=swa_loc[: kv_latent.shape[0]],
            cache_k=kv_latent,
        )

        q = self._project_q(hidden_states, cos, sin)
        if not _is_sail and self.n_local_heads < _PAD_NUM_HEADS:
            q_in = F.pad(q, (0, 0, 0, _PAD_NUM_HEADS - self.n_local_heads))
        else:
            q_in = q

        o = attn_backend.forward_dfly_attention(
            q=q_in,
            layer=self.attn,
            forward_batch=forward_batch,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            extra_indices=extra_indices,
            extra_topk_lengths=extra_topk_lengths,
            attn_sink=self._local_attn_sink(),
        )
        if o.shape[1] != self.n_local_heads:
            o = o[:, : self.n_local_heads, :]
        return self._project_output(o, cos, sin, is_target_verify=True)

    def forward_dense_reference(
        self,
        hidden_states: torch.Tensor,
        context_latent: torch.Tensor,
        sparse_indices: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch reference matching ``AngelSpec``'s ``DSV41DraftAttention``
        for unit testing (``hidden_states``: ``[B, S, D]``, ``context_latent``:
        ``[B, C, 512]``, ``sparse_indices``: ``[B, S, K]`` int32 into
        ``[context | draft]``)."""
        bs, slots, _ = hidden_states.shape
        flat_x = hidden_states.reshape(bs * slots, self.hidden_size)
        flat_cos = cos.reshape(bs * slots, -1)
        flat_sin = sin.reshape(bs * slots, -1)

        q = self._project_q(flat_x, flat_cos, flat_sin).view(
            bs, slots, self.n_local_heads, self.head_dim
        )
        kv_raw, _ = self.wkv(flat_x)
        draft_latent = apply_dsv41_rope(self.kv_norm(kv_raw), flat_cos, flat_sin).view(
            bs, slots, self.head_dim
        )
        latent = torch.cat([context_latent.to(draft_latent.dtype), draft_latent], dim=1)

        k_len = latent.shape[1]
        valid = (sparse_indices >= 0) & (sparse_indices < k_len)
        safe_idx = sparse_indices.clamp(0, k_len - 1).long()
        # Gather keys per query: [B, S, K, D]
        gathered_k = torch.gather(
            latent[:, None, :, :].expand(bs, slots, k_len, self.head_dim),
            2,
            safe_idx.unsqueeze(-1).expand(
                bs, slots, sparse_indices.shape[-1], self.head_dim
            ),
        ).float()
        logits = torch.einsum("bshd,bskd->bshk", q.float(), gathered_k) * self.scaling
        logits = logits.masked_fill(~valid.unsqueeze(2), float("-inf"))
        start = self.tp_rank * self.n_local_heads
        local_sink = self.attn_sink[start : start + self.n_local_heads].float()
        total = torch.logaddexp(
            torch.logsumexp(logits, dim=-1, keepdim=True),
            local_sink.view(1, 1, -1, 1),
        )
        probs = torch.where(
            valid.unsqueeze(2), torch.exp(logits - total), torch.zeros_like(logits)
        )
        out = torch.einsum("bshk,bskd->bshd", probs, gathered_k).to(q.dtype)
        out = self._project_output(
            out.reshape(bs * slots, self.n_local_heads, self.head_dim),
            flat_cos,
            flat_sin,
            is_target_verify=False,
        )
        return out.view(bs, slots, self.hidden_size)


def _make_dfly_moe_config(
    config: DeepSeekV4Config, dfly_config: DFlyDeepseekV41Config
) -> DeepSeekV4Config:
    moe_cfg = copy.copy(config)
    moe_cfg.n_routed_experts = dfly_config.n_routed_experts
    moe_cfg.num_experts_per_tok = dfly_config.num_experts_per_tok
    moe_cfg.vision_n_layers = 0
    return moe_cfg


def _make_dfly_moe_quant_config(
    quant_config: Optional[QuantizationConfig],
    *,
    num_layers: int,
    prefix: str,
) -> Optional[QuantizationConfig]:
    """Keep MXFP4 quantization on routed experts while leaving BF16
    ``shared_experts`` unquantized (``AngelSlim/DS-V4.1-Flash-DFly2-B6``'s
    ``ignored_layers`` still references target layer IDs 40..42)."""
    if quant_config is None:
        return None
    cloned = copy.copy(quant_config)
    ignored = list(getattr(cloned, "ignored_layers", None) or ())
    for i in range(num_layers):
        layer_prefix = add_prefix(f"layers.{i}.mlp.shared_experts", prefix)
        for proj in ("gate_up_proj", "gate_proj", "up_proj", "down_proj"):
            full = f"{layer_prefix}.{proj}"
            if full not in ignored:
                ignored.append(full)
    cloned.ignored_layers = ignored
    return cloned


class DFlyDSV41Layer(nn.Module):
    """Single DFly2 decoder layer: mHC (hc_mult=4) + DSV41Attention + DeepseekV2MoE."""

    def __init__(
        self,
        config: DeepSeekV4Config,
        dfly_config: DFlyDeepseekV41Config,
        layer_idx: int,
        target_kv_layer: int,
        moe_quant_config: Optional[QuantizationConfig],
        prefix: str = "",
        *,
        mlp_factory: Optional[Callable[[], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.target_kv_layer = target_kv_layer
        self.hc_mult = int(getattr(config, "hc_mult", 4))
        self.hc_sinkhorn_iters = int(getattr(config, "hc_sinkhorn_iters", 20))
        self.hc_eps = float(getattr(config, "hc_eps", 1e-6))
        self.rms_norm_eps = float(config.rms_norm_eps)

        self.attn_norm = DFlyDSV41RMSNorm(config.hidden_size, eps=self.rms_norm_eps)
        self.self_attn = DFlyDSV41Attention(
            config=config,
            dfly_config=dfly_config,
            layer_idx=layer_idx,
            target_kv_layer=target_kv_layer,
            prefix=add_prefix("self_attn", prefix),
        )
        self.ffn_norm = DFlyDSV41RMSNorm(config.hidden_size, eps=self.rms_norm_eps)
        if mlp_factory is not None:
            self.mlp = mlp_factory()
            self._mlp_takes_forward_batch = False
        else:
            self.mlp = deepseek_v2.DeepseekV2MoE(
                config=_make_dfly_moe_config(config, dfly_config),
                quant_config=moe_quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=layer_idx,
                is_nextn=True,
                is_deepseek_v4=True,
                vl_correction_bias=False,
            )
            self._mlp_takes_forward_batch = True

        if self.hc_mult > 1:
            mix = (2 + self.hc_mult) * self.hc_mult
            width = self.hc_mult * int(config.hidden_size)
            self.hc_attn_fn = nn.Parameter(
                torch.zeros(mix, width, dtype=torch.float32), requires_grad=False
            )
            self.hc_attn_base = nn.Parameter(
                torch.zeros(mix, dtype=torch.float32), requires_grad=False
            )
            self.hc_attn_scale = nn.Parameter(
                torch.zeros(3, dtype=torch.float32), requires_grad=False
            )
            self.hc_ffn_fn = nn.Parameter(
                torch.zeros(mix, width, dtype=torch.float32), requires_grad=False
            )
            self.hc_ffn_base = nn.Parameter(
                torch.zeros(mix, dtype=torch.float32), requires_grad=False
            )
            self.hc_ffn_scale = nn.Parameter(
                torch.zeros(3, dtype=torch.float32), requires_grad=False
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        collapsed: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        swa_page_indices: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        extra_indices: torch.Tensor,
        extra_topk_lengths: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hc_mult > 1:
            attn_pre, attn_post, attn_comb = _hc_mixes(
                hidden,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                iters=self.hc_sinkhorn_iters,
                eps=self.hc_eps,
                norm_eps=self.rms_norm_eps,
            )
            attn_out = self.self_attn(
                positions=positions,
                hidden_states=self.attn_norm(collapsed),
                forward_batch=forward_batch,
                swa_page_indices=swa_page_indices,
                swa_topk_lengths=swa_topk_lengths,
                extra_indices=extra_indices,
                extra_topk_lengths=extra_topk_lengths,
                cos=cos,
                sin=sin,
            )
            hidden = _hc_post(attn_out, hidden, attn_post, attn_comb)
            ffn_pre, ffn_post, ffn_comb = _hc_mixes(
                hidden,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                iters=self.hc_sinkhorn_iters,
                eps=self.hc_eps,
                norm_eps=self.rms_norm_eps,
            )
            ffn_in = self.ffn_norm(_hc_pre(hidden, attn_pre))
            ffn_out = (
                self.mlp(ffn_in, forward_batch)
                if self._mlp_takes_forward_batch
                else self.mlp(ffn_in)
            )
            hidden = _hc_post(ffn_out, hidden, ffn_post, ffn_comb)
            return hidden, _hc_pre(hidden, ffn_pre)

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=self.attn_norm(hidden),
            forward_batch=forward_batch,
            swa_page_indices=swa_page_indices,
            swa_topk_lengths=swa_topk_lengths,
            extra_indices=extra_indices,
            extra_topk_lengths=extra_topk_lengths,
            cos=cos,
            sin=sin,
        )
        hidden = hidden + attn_out
        ffn_in = self.ffn_norm(hidden)
        ffn_out = (
            self.mlp(ffn_in, forward_batch)
            if self._mlp_takes_forward_batch
            else self.mlp(ffn_in)
        )
        hidden = hidden + ffn_out
        return hidden, hidden


class DFlyDSV41AttentiveCorrection(nn.Module):
    """Attentive intra-block correction for DeepSeek-V4.1 DFly2.

    For slot ``s`` (given ``inputs = [hidden_norm(h_s), embed_norm(e_{y_{s-1}})]``):
      1. Context attention against target layer ``source_target_layer``'s SWA
         window + C1 top-512 sparse latents (NO draft block keys).
      2. Gated causal prefix attention over slots ``0..s`` of the block, keyed
         by ``kv_in(inputs_k) -> kv_norm -> RoPE(pos_k)`` with ``block_rel_bias``
         and ``block_sink``, scaled by ``tanh(block_gate(inputs_s) + block_gate_bias[:, s])``.
      3. Inverse RoPE + grouped output ``wo_a`` -> ``wo_b``.
    """

    def __init__(
        self,
        config: DeepSeekV4Config,
        dfly_config: DFlyDeepseekV41Config,
        ac_config: DFlyAttentiveCorrectionConfig,
        source_target_layer: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        self.tp_size = parallel.tp_size
        self.tp_rank = parallel.tp_rank

        self.hidden_size = int(config.hidden_size)
        self.n_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.rope_dim = int(config.qk_rope_head_dim)
        self.q_lora_rank = int(config.q_lora_rank)
        self.o_lora_rank = int(config.o_lora_rank)
        self.n_groups = int(config.o_groups)
        self.eps = float(config.rms_norm_eps)
        self.scaling = self.head_dim**-0.5
        self.source_layer_idx = ac_config.source_layer
        self.source_target_layer = source_target_layer

        if self.n_groups % self.tp_size != 0 or self.n_heads % self.tp_size != 0:
            raise ValueError(
                f"DFlyDSV41AttentiveCorrection requires tp_size ({self.tp_size}) "
                f"to divide o_groups ({self.n_groups}) and num_attention_heads ({self.n_heads})."
            )
        self.n_local_heads = self.n_heads // self.tp_size
        self.n_local_groups = self.n_groups // self.tp_size
        self.heads_per_group = self.n_heads // self.n_groups

        self.q_in = ReplicatedLinear(
            2 * self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=None,
            prefix=add_prefix("q_in", prefix),
        )
        self.q_norm = DFlyDSV41RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wq_b", prefix),
        )
        self.attn_sink = nn.Parameter(
            torch.zeros(self.n_heads, dtype=torch.float32), requires_grad=False
        )
        self.wo_a = nn.Linear(
            self.heads_per_group * self.head_dim,
            self.n_local_groups * self.o_lora_rank,
            bias=False,
        )
        self.wo_a.weight.weight_loader = _load_tp_row_slice
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=None,
            reduce_results=True,
            prefix=add_prefix("wo_b", prefix),
        )

        self.block_keys = ac_config.block_keys
        self.block_gated = ac_config.block_keys and ac_config.block_mode == "gated"
        self.block_kv_in_embed = ac_config.block_kv_in == "hidden_embed"
        self.max_block_slots = ac_config.max_block_slots

        if self.block_keys:
            kv_in_dim = (2 if self.block_kv_in_embed else 1) * self.hidden_size
            self.kv_in = ReplicatedLinear(
                kv_in_dim,
                self.head_dim,
                bias=False,
                quant_config=None,
                prefix=add_prefix("kv_in", prefix),
            )
            self.kv_norm = DFlyDSV41RMSNorm(self.head_dim, eps=self.eps)
        if self.block_gated:
            self.block_sink = nn.Parameter(
                torch.zeros(self.n_local_heads, dtype=torch.float32),
                requires_grad=False,
            )
            self.block_sink.weight_loader = _load_tp_row_slice
            self.block_rel_bias = nn.Parameter(
                torch.zeros(
                    self.n_local_heads, self.max_block_slots, dtype=torch.float32
                ),
                requires_grad=False,
            )
            self.block_rel_bias.weight_loader = _load_tp_row_slice
            self.block_gate = ColumnParallelLinear(
                2 * self.hidden_size,
                self.n_heads,
                bias=False,
                quant_config=None,
                prefix=add_prefix("block_gate", prefix),
            )
            self.block_gate_bias = nn.Parameter(
                torch.zeros(
                    self.n_local_heads, self.max_block_slots, dtype=torch.float32
                ),
                requires_grad=False,
            )
            self.block_gate_bias.weight_loader = _load_tp_row_slice

        self.attn = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=source_target_layer,
            quant_config=None,
            prefix=add_prefix("attn", prefix),
        )
        self.register_buffer(
            "inv_freq",
            compute_dsv41_yarn_inv_freq(self.rope_dim, dfly_config.rope),
            persistent=False,
        )

    def _local_attn_sink(self) -> torch.Tensor:
        if _is_sail:
            return self.attn_sink
        start = self.tp_rank * self.n_local_heads
        local = self.attn_sink[start : start + self.n_local_heads]
        if self.n_local_heads < _PAD_NUM_HEADS:
            return F.pad(local, (0, _PAD_NUM_HEADS - self.n_local_heads), value=0.0)
        return local

    def compute_block_key(
        self,
        inputs: torch.Tensor,
        hidden_norm: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Compute one slot's block key ``[bs, head_dim]`` with RoPE applied."""
        kv_src = inputs if self.block_kv_in_embed else hidden_norm
        k_raw, _ = self.kv_in(kv_src)
        return apply_dsv41_rope(self.kv_norm(k_raw), cos, sin)

    def _attend_block_gated_step(
        self,
        q: torch.Tensor,
        block_keys_prefix: torch.Tensor,
        inputs: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Gated causal prefix read for a single slot ``step`` (reads ``0..step``).

        Args:
          q: ``[bs, n_local_heads, head_dim]``
          block_keys_prefix: ``[bs, step + 1, head_dim]``
          inputs: ``[bs, 2 * hidden_size]``
        Returns:
          ``[bs, n_local_heads, head_dim]`` fp32 weighted block read scaled by gate.
        """
        qf = q.float()
        kf = block_keys_prefix.float()
        logits = torch.einsum("bhm,bkm->bhk", qf, kf) * self.scaling
        back = torch.arange(step, -1, -1, device=q.device, dtype=torch.int64)
        logits = logits + self.block_rel_bias.float()[:, back].unsqueeze(0)
        total = torch.logaddexp(
            torch.logsumexp(logits, dim=-1, keepdim=True),
            self.block_sink.float().view(1, -1, 1),
        )
        read = torch.einsum("bhk,bkm->bhm", torch.exp(logits - total), kf)
        gate_proj, _ = self.block_gate(inputs)
        gate = torch.tanh(
            gate_proj.float() + self.block_gate_bias.float()[:, step].unsqueeze(0)
        )
        return read * gate.unsqueeze(-1)

    def forward_step(
        self,
        *,
        inputs: torch.Tensor,
        hidden_norm: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        step: int,
        block_keys_buf: Optional[torch.Tensor],
        context_read_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        has_context: torch.Tensor,
    ) -> torch.Tensor:
        """Run one autoregressive slot ``step`` of attentive correction.

        Args:
          inputs: ``[bs, 2 * hidden_size]``
          hidden_norm: ``[bs, hidden_size]``
          cos, sin: ``[bs, rope_dim // 2]``
          step: slot index ``1 .. block_size - 1``
          block_keys_buf: ``[bs, block_size, head_dim]`` (slots ``0..step-1`` filled)
          context_read_fn: ``(q_in, attn_sink) -> [bs, n_local_heads, head_dim]``
          has_context: ``[bs]`` bool indicating whether the request has any context key
        """
        q_lora, _ = self.q_in(inputs)
        q, _ = self.wq_b(self.q_norm(q_lora))
        q = q.view(-1, self.n_local_heads, self.head_dim)
        q = apply_dsv41_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))

        if not _is_sail and self.n_local_heads < _PAD_NUM_HEADS:
            q_in = F.pad(q, (0, 0, 0, _PAD_NUM_HEADS - self.n_local_heads))
        else:
            q_in = q

        ctx_read = context_read_fn(q_in, self._local_attn_sink())
        if ctx_read.shape[1] != self.n_local_heads:
            ctx_read = ctx_read[:, : self.n_local_heads, :]
        ctx_read = torch.where(
            has_context[:, None, None],
            ctx_read.float(),
            torch.zeros_like(ctx_read.float()),
        )

        if self.block_gated:
            assert block_keys_buf is not None
            block_keys_buf[:, step] = self.compute_block_key(
                inputs, hidden_norm, cos, sin
            )
            block_read = self._attend_block_gated_step(
                q, block_keys_buf[:, : step + 1], inputs, step
            )
            read = (ctx_read + block_read).to(q.dtype)
        else:
            read = ctx_read.to(q.dtype)

        read = apply_dsv41_rope(read, cos.unsqueeze(1), sin.unsqueeze(1), inverse=True)
        read = read.view(
            read.shape[0],
            self.n_local_groups,
            self.heads_per_group * self.head_dim,
        )
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if read.is_cuda:
            out = _apply_wo_a_bf16_matmul(
                read, wo_a, is_decode=True, is_target_verify=False, fast_path=True
            )
        else:
            out = torch.einsum("ngd,grd->ngr", read.float(), wo_a.float()).to(
                read.dtype
            )
        value = out.reshape(out.shape[0], self.n_local_groups * self.o_lora_rank)
        delta, _ = self.wo_b(value)
        return delta


class DFlyDSV41HiddenCorrection(nn.Module):
    """TreeFlash SwiGLU + optional ``DFlyDSV41AttentiveCorrection``."""

    def __init__(
        self,
        config: DeepSeekV4Config,
        dfly_config: DFlyDeepseekV41Config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        d = int(config.hidden_size)
        width = dfly_config.hidden_correction_intermediate_size or d
        eps = float(config.rms_norm_eps)

        self.hidden_norm = DFlyDSV41RMSNorm(d, eps=eps)
        self.embed_norm = DFlyDSV41RMSNorm(d, eps=eps)
        self.gate_up_proj = MergedColumnParallelLinear(
            2 * d,
            [width, width],
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.act = SiluAndMul()
        self.down_proj = RowParallelLinear(
            width,
            d,
            bias=False,
            quant_config=None,
            reduce_results=True,
            prefix=add_prefix("down_proj", prefix),
        )

        self.attention: Optional[DFlyDSV41AttentiveCorrection] = None
        if dfly_config.attentive_correction is not None:
            ac_cfg = dfly_config.attentive_correction
            source_target_layer = dfly_config.target_kv_layers[ac_cfg.source_layer]
            self.attention = DFlyDSV41AttentiveCorrection(
                config=config,
                dfly_config=dfly_config,
                ac_config=ac_cfg,
                source_target_layer=source_target_layer,
                prefix=add_prefix("attention", prefix),
            )

    def forward_swiglu(
        self, h_norm: torch.Tensor, e_norm: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([h_norm, e_norm], dim=-1)
        gate_up, _ = self.gate_up_proj(inputs)
        delta, _ = self.down_proj(self.act(gate_up))
        return delta, inputs


# ---------------------------------------------------------------------------
# Top-level draft model (HF architecture: DFlyDeepseekV41DraftModel)
# ---------------------------------------------------------------------------


class DFlyDeepseekV41DraftModel(nn.Module):
    """AngelSpec DFly2 drafter for DeepSeek-V4.1-Flash."""

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        *,
        draft_model_idx: Optional[int] = None,
        block_size_override: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        mlp_factory: Optional[Callable[[int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.num_fused_shared_experts = get_num_fused_shared_experts()
        if block_size_override is None:
            try:
                spec_cfg = get_spec()
            except ValueError:
                spec_cfg = None
            if (
                spec_cfg is not None
                and spec_cfg.speculative_algorithm == "DFLY"
                and spec_cfg.speculative_num_draft_tokens is not None
            ):
                block_size_override = int(spec_cfg.speculative_num_draft_tokens)
        self.dfly_config = parse_dsv41_dfly_config(
            config,
            block_size_override=block_size_override,
            checkpoint_path=checkpoint_path,
        )
        self.hidden_size = int(config.hidden_size)
        self.hc_mult = int(getattr(config, "hc_mult", 4))
        self.rope_dim = int(config.qk_rope_head_dim)

        moe_quant_config = _make_dfly_moe_quant_config(
            quant_config,
            num_layers=self.dfly_config.num_layers,
            prefix=prefix,
        )

        self.layers = nn.ModuleList(
            [
                DFlyDSV41Layer(
                    config=config,
                    dfly_config=self.dfly_config,
                    layer_idx=i,
                    target_kv_layer=target_kv_layer,
                    moe_quant_config=moe_quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                    mlp_factory=(
                        (lambda idx=i: mlp_factory(idx))
                        if mlp_factory is not None
                        else None
                    ),
                )
                for i, target_kv_layer in enumerate(self.dfly_config.target_kv_layers)
            ]
        )
        self.final_norm = DFlyDSV41RMSNorm(
            self.hidden_size, eps=float(config.rms_norm_eps)
        )
        self.mtp_fusion: Optional[DFlyDSV41MTPFusion] = (
            DFlyDSV41MTPFusion(
                hidden_size=self.hidden_size,
                mode=self.dfly_config.mtp_fusion,
                rms_norm_eps=float(config.rms_norm_eps),
            )
            if self.dfly_config.mtp_fusion is not None
            else None
        )
        self.hidden_correction: Optional[DFlyDSV41HiddenCorrection] = (
            DFlyDSV41HiddenCorrection(
                config=config,
                dfly_config=self.dfly_config,
                prefix=add_prefix("hidden_correction", prefix),
            )
            if self.dfly_config.hidden_correction_intermediate_size is not None
            else None
        )

        self.register_buffer(
            "inv_freq",
            compute_dsv41_yarn_inv_freq(self.rope_dim, self.dfly_config.rope),
            persistent=False,
        )

        # Target-shared modules (never registered as child submodules so
        # state_dict / weight loading never double-counts them).
        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None

        # Runtime state buffers bound by DFlyWorkerV2.alloc_memory_pool().
        self.state: Optional[DFlyRequestState] = None
        self.extra_rows: Optional[torch.Tensor] = None
        self.extra_lengths: Optional[torch.Tensor] = None
        self.candidates_out: Optional[torch.Tensor] = None

    def bind_target(self, embed_tokens: nn.Module, lm_head: nn.Module) -> None:
        self.__dict__["embed_tokens"] = embed_tokens
        self.__dict__["lm_head"] = lm_head

    def bind_buffers(
        self,
        *,
        state: DFlyRequestState,
        extra_rows: torch.Tensor,
        extra_lengths: torch.Tensor,
        candidates_out: torch.Tensor,
    ) -> None:
        self.state = state
        self.extra_rows = extra_rows
        self.extra_lengths = extra_lengths
        self.candidates_out = candidates_out

    def draft_into_verify(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Run DFly2 drafting in-place at the start of target ``TARGET_VERIFY``."""
        num_rows = input_ids.shape[0]
        width = self.dfly_config.block_size
        bs = num_rows // width
        req_pool_indices = forward_batch.req_pool_indices[:bs]

        attn_backend = get_attn_backend()
        pool = get_token_to_kv_pool()
        core_meta = attn_backend.forward_metadata.core_attn_metadata
        request_window = pool.request_window

        # 1. Build per-layer top-k global C1 indices from captured target picks.
        extra_rows, extra_lengths, ctx_extra_rows, ctx_extra_counts = (
            build_dfly_extra_topk(
                state_topk=self.state.topk,
                req_pool_indices=req_pool_indices,
                block_size=width,
                out_rows=self.extra_rows,
                out_lengths=self.extra_lengths,
            )
        )

        # 2. Build SWA indices for bidirectional draft block + 128-token context
        #    (and context-only SWA indices for attentive correction).
        swa_out_cache_loc = (
            None
            if request_window is not None
            else attn_backend.get_swa_out_cache_loc(forward_batch)
        )
        block_swa_indices, block_swa_lengths, ctx_swa_indices, ctx_swa_counts = (
            build_dfly_swa_indices(
                req_pool_indices=req_pool_indices,
                prefix_lens=forward_batch.seq_lens,
                block_size=width,
                swa_window=self.dfly_config.sliding_window,
                causal_block=self.dfly_config.causal_block,
                req_to_token=attn_backend.req_to_token,
                full_to_swa_mapping=pool.full_to_swa_index_mapping,
                swa_out_cache_loc=swa_out_cache_loc,
                request_window_layout=(
                    core_meta.request_window_layout
                    if request_window is not None
                    else None
                ),
                swa_valid_from=getattr(pool, "swa_valid_from", None),
            )
        )

        block_ids = input_ids.view(bs, width)
        try:
            drafts = self.propose(
                block_ids=block_ids,
                positions=positions[:num_rows],
                target_hidden=self.state.hidden[req_pool_indices],
                forward_batch=forward_batch,
                block_swa_indices=block_swa_indices,
                block_swa_lengths=block_swa_lengths,
                extra_rows=extra_rows,
                extra_lengths=extra_lengths,
                ctx_swa_indices=ctx_swa_indices,
                ctx_swa_counts=ctx_swa_counts,
                ctx_extra_rows=ctx_extra_rows,
                ctx_extra_counts=ctx_extra_counts,
            )
        finally:
            # Invalidate RequestWindow's workspace prepared tag so target layer 0
            # refreshes cleanly before writing its own verify SWA tokens.
            if request_window is not None:
                request_window.prepared = None

        block_ids[:, 1:] = drafts
        self.candidates_out[:num_rows].copy_(input_ids)

    @torch.no_grad()
    def propose(
        self,
        *,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        target_hidden: torch.Tensor,
        forward_batch: ForwardBatch,
        block_swa_indices: torch.Tensor,
        block_swa_lengths: torch.Tensor,
        extra_rows: torch.Tensor,
        extra_lengths: torch.Tensor,
        ctx_swa_indices: torch.Tensor,
        ctx_swa_counts: torch.Tensor,
        ctx_extra_rows: torch.Tensor,
        ctx_extra_counts: torch.Tensor,
    ) -> torch.Tensor:
        bs, width = block_ids.shape
        num_rows = bs * width
        embeds = self.embed_tokens(block_ids.reshape(-1)).view(bs, width, -1)
        if self.mtp_fusion is not None:
            embeds = self.mtp_fusion(embeds, target_hidden)

        inv_freq = self.inv_freq.to(positions.device)
        cos_flat, sin_flat = compute_dsv41_rope_cos_sin(positions, inv_freq)

        collapsed = embeds.reshape(num_rows, self.hidden_size)
        hidden = (
            collapsed.unsqueeze(1).repeat(1, self.hc_mult, 1)
            if self.hc_mult > 1
            else collapsed
        )
        for i, layer in enumerate(self.layers):
            hidden, collapsed = layer(
                positions=positions,
                hidden=hidden,
                collapsed=collapsed,
                forward_batch=forward_batch,
                swa_page_indices=block_swa_indices,
                swa_topk_lengths=block_swa_lengths,
                extra_indices=extra_rows[i],
                extra_topk_lengths=extra_lengths[i],
                cos=cos_flat,
                sin=sin_flat,
            )

        draft_hidden = self.final_norm(collapsed).view(bs, width, self.hidden_size)
        if self.hidden_correction is None:
            return self._greedy(draft_hidden[:, 1:].reshape(-1, self.hidden_size)).view(
                bs, width - 1
            )

        return self._run_hidden_correction(
            draft_hidden=draft_hidden,
            anchor_tokens=block_ids[:, 0],
            cos=cos_flat.view(bs, width, -1),
            sin=sin_flat.view(bs, width, -1),
            forward_batch=forward_batch,
            ctx_swa_indices=ctx_swa_indices,
            ctx_swa_counts=ctx_swa_counts,
            ctx_extra_rows=ctx_extra_rows,
            ctx_extra_counts=ctx_extra_counts,
        )

    def _run_hidden_correction(
        self,
        *,
        draft_hidden: torch.Tensor,
        anchor_tokens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        forward_batch: ForwardBatch,
        ctx_swa_indices: torch.Tensor,
        ctx_swa_counts: torch.Tensor,
        ctx_extra_rows: torch.Tensor,
        ctx_extra_counts: torch.Tensor,
    ) -> torch.Tensor:
        bs, width, _ = draft_hidden.shape
        hc = self.hidden_correction
        assert hc is not None
        ac = hc.attention

        h_norm_all = hc.hidden_norm(draft_hidden)

        block_keys_buf: Optional[torch.Tensor] = None
        context_read_fn = None
        has_context = None
        if ac is not None:
            src_idx = ac.source_layer_idx
            src_extra_rows = ctx_extra_rows[src_idx]
            src_extra_counts = ctx_extra_counts[src_idx]
            src_swa_lengths = ctx_swa_counts.clamp_min(1)
            src_extra_lengths = src_extra_counts.clamp_min(1)
            has_context = (ctx_swa_counts + src_extra_counts) > 0
            attn_backend = get_attn_backend()

            def _ctx_read(q_in: torch.Tensor, attn_sink: torch.Tensor) -> torch.Tensor:
                return attn_backend.forward_dfly_attention(
                    q=q_in,
                    layer=ac.attn,
                    forward_batch=forward_batch,
                    swa_page_indices=ctx_swa_indices,
                    swa_topk_lengths=src_swa_lengths,
                    extra_indices=src_extra_rows,
                    extra_topk_lengths=src_extra_lengths,
                    attn_sink=attn_sink,
                )

            context_read_fn = _ctx_read
            if ac.block_keys:
                block_keys_buf = draft_hidden.new_empty((bs, width, ac.head_dim))
                # Slot 0's prev_token is anchor_tokens (matches AngelSpec dflash.py:2528).
                e0_norm = hc.embed_norm(self.embed_tokens(anchor_tokens))
                inputs_0 = torch.cat([h_norm_all[:, 0], e0_norm], dim=-1)
                block_keys_buf[:, 0] = ac.compute_block_key(
                    inputs_0, h_norm_all[:, 0], cos[:, 0], sin[:, 0]
                )

        previous = anchor_tokens
        drafts: List[torch.Tensor] = []
        for step in range(1, width):
            h_norm = h_norm_all[:, step]
            e_norm = hc.embed_norm(self.embed_tokens(previous))
            delta, inputs = hc.forward_swiglu(h_norm, e_norm)
            if ac is not None:
                assert context_read_fn is not None and has_context is not None
                attn_delta = ac.forward_step(
                    inputs=inputs,
                    hidden_norm=h_norm,
                    cos=cos[:, step],
                    sin=sin[:, step],
                    step=step,
                    block_keys_buf=block_keys_buf,
                    context_read_fn=context_read_fn,
                    has_context=has_context,
                )
                delta = delta + attn_delta.to(delta.dtype)
            corrected = draft_hidden[:, step] + delta.to(draft_hidden.dtype)
            previous = self._greedy(corrected)
            drafts.append(previous)

        return torch.stack(drafts, dim=1)

    def _greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        head = self.lm_head
        if head is None:
            raise ValueError(
                "DFlyDeepseekV41DraftModel requires bind_target() before drafting."
            )
        shard = getattr(head, "shard_indices", None)
        if shard is None:
            logits = F.linear(hidden.float(), head.weight.float())
            return logits.argmax(dim=-1)
        weight = head.weight[: shard.num_org_elements]
        logits = F.linear(hidden, weight).float()
        local_max, local_arg = logits.max(dim=-1)
        local_ids = local_arg + shard.org_vocab_start_index
        tp_group = get_parallel().tp_group
        if tp_group.world_size == 1:
            return local_ids
        rows = hidden.shape[0]
        maxes = local_max.new_empty((tp_group.world_size * rows,))
        ids = local_ids.new_empty((tp_group.world_size * rows,))
        tp_group.all_gather_into_tensor(maxes, local_max.contiguous())
        tp_group.all_gather_into_tensor(ids, local_ids.contiguous())
        best = maxes.view(-1, rows).argmax(dim=0)
        return ids.view(-1, rows).gather(0, best[None]).squeeze(0)

    # -----------------------------------------------------------------------
    # Weight loading (AngelSlim/DS-V4.1-Flash-DFly2-B6 & compatible layouts)
    # -----------------------------------------------------------------------

    def _remap_weight_name(self, name: str) -> Optional[str]:
        if name.startswith("model."):
            name = name[len("model.") :]
        if name.startswith(("embed.", "embed_tokens.", "head.", "lm_head.")):
            return None
        if name.startswith("confidence_head.") or "rotary_emb.inv_freq" in name:
            return None

        if name.startswith("layers."):
            name = name.replace(".attn.", ".self_attn.", 1)
            name = name.replace(".ffn.", ".mlp.", 1)
            name = name.replace(".w1.", ".gate_proj.")
            name = name.replace(".w2.", ".down_proj.")
            name = name.replace(".w3.", ".up_proj.")
            name = name.replace(".gate.tid2eid", ".topk.tid2eid")
            if name.endswith(".gate.bias_vl"):
                return None
            name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
            if name.endswith(".scale"):
                name = name[: -len(".scale")] + ".weight_scale_inv"
                if _is_sail:
                    if (
                        self.quant_config is not None
                        and self.quant_config.get_name() == "fp8"
                    ):
                        if getattr(
                            self.quant_config, "weight_block_size", None
                        ) is None or (
                            getattr(self.quant_config, "is_fp4_experts", False)
                            and ".mlp.experts." in name
                        ):
                            name = name.replace(".weight_scale_inv", ".weight_scale")
                    else:
                        name = name.replace(".weight_scale_inv", ".weight_scale")
            return name

        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        n_routed_experts = self.dfly_config.n_routed_experts

        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=(n_routed_experts + self.num_fused_shared_experts),
        )

        for orig_name, loaded_weight in weights:
            mapped = self._remap_weight_name(orig_name)
            if mapped is None:
                continue

            if self.num_fused_shared_experts > 0 and ".mlp.shared_experts." in mapped:
                mapped = mapped.replace(
                    ".mlp.shared_experts.",
                    f".mlp.experts.{n_routed_experts}.",
                )

            # Stacked gate_proj / up_proj (shared_experts and hidden_correction).
            for param_name, weight_name, shard_id in DEEPSEEK_V4_STACKED_PARAMS_MAPPING:
                if weight_name not in mapped:
                    continue
                candidate = mapped.replace(weight_name, param_name)
                if candidate not in params_dict:
                    continue
                param = params_dict[candidate]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(candidate)
                break
            else:
                # Routed MoE experts.
                for (
                    param_name,
                    weight_name,
                    expert_id,
                    shard_id,
                ) in expert_params_mapping:
                    if weight_name not in mapped:
                        continue
                    candidate = mapped.replace(weight_name, param_name)
                    if candidate not in params_dict:
                        continue
                    param = params_dict[candidate]
                    param.weight_loader(
                        param,
                        loaded_weight,
                        candidate,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    loaded_params.add(candidate)
                    break
                else:
                    if mapped not in params_dict:
                        logger.warning(
                            "DFlyDeepseekV41DraftModel: skipping unexpected weight %r -> %r",
                            orig_name,
                            mapped,
                        )
                        continue
                    param = params_dict[mapped]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(mapped)

        missing = {
            name
            for name in params_dict
            if not name.endswith(
                (
                    ".k_scale",
                    ".v_scale",
                    ".q_scale",
                    ".w13_weight_bias",
                    ".w2_weight_bias",
                )
            )
            and "blockscale_swizzled" not in name
        } - loaded_params
        if missing:
            raise ValueError(
                f"DFlyDeepseekV41DraftModel: missing weights in checkpoint: {sorted(missing)}"
            )


EntryClass = [DFlyDeepseekV41DraftModel]
