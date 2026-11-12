# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""K31HT_DRAFT_CTXKV_FUSE: the MLA drafter's context-KV append in one GEMM and one epilogue kernel.

Served path (default, knob off): for each of the draft's MLA layers, DFlashMLAAttention.write_context_kv projects
the SAME context rows (DFlashDraftModel.prepare_context_hidden_for_kv returns them unchanged) with the layer's
kv_a rows of the fused a_proj, applies kv_a_layernorm to the 512-wide latent and the yarn rope to the 64-wide tail,
casts both to fp8 and scatters the first commit_lens[b] rows of each request (MLATokenToKVPool
.set_mla_kv_buffer_prefix_valid -> _write_mla_kv_buffer -> _scatter_mla_rows): ~11 launches per layer.

Knob on: one F.linear on a contiguous copy of the layers' kv_a rows (WCAT [layers x 576, hidden], built once at
init), then one Triton kernel, grid (rows, layers): FP32 RMSNorm with the layer's weight -> BF16 -> fp8; rope
from the module's FP32 cos_sin_cache (interleaved or neox, as the module) -> BF16 -> fp8; the prefix-valid mask,
the reserved slot 0 skip and the DCP owner rule of set_mla_kv_buffer(_dcp_sharded)_triton; stores into each
layer's pool buffer through a pointer table. Stream-ordered, no PDL, no cross-CTA or cross-rank sync.

Numerics: not bitwise to the served path (RMSNorm reduction order, GEMM tactic at N = layers x 576) - the
FP64-no-worse class. Triton's fp32 -> e4m3 cast saturates at 448
where torch's .to(float8_e4m3fn) yields NaN; normed values sit far below 448.

Engages only when every condition in k31ht_ctxkv_block_reason holds; otherwise the served path runs. WCAT is a
copy: a later in-place draft weight reload would leave it stale (not supported with the knob on).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_KNOB = "K31HT_DRAFT_CTXKV_FUSE"


def _parse_knob(name: str) -> bool:
    v = os.environ.get(name, "")
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    raise ValueError(f"{name} must be unset, '', '0' or '1', got {v!r}")


def k31ht_draft_ctxkv_fuse_requested() -> bool:
    return _parse_knob(_KNOB)


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@triton.jit
def _k31ht_ctxkv_kernel(
    Y,
    y_stride,
    NW,
    EPS,
    POS,
    CS,
    cs_stride,
    LOC2D,
    loc_stride,
    CL,
    n_per,
    PTRS,
    row_stride,
    NOPE_D: tl.constexpr,
    HALF: tl.constexpr,
    IS_NEOX: tl.constexpr,
    SKIP: tl.constexpr,
    DCPW: tl.constexpr,
    DCPR: tl.constexpr,
):
    m = tl.program_id(0)
    l = tl.program_id(1)
    b = m // n_per
    j = m % n_per
    loc = tl.load(LOC2D + b * loc_stride + j).to(tl.int64)
    cl = tl.load(CL + b)
    loc = tl.where(j < cl, loc, 0)
    valid = (loc != SKIP) & (loc % DCPW == DCPR)
    row = tl.where(valid, loc, 0) // DCPW
    dst = tl.load(PTRS + l).to(tl.pointer_type(tl.uint8)) + row * row_stride
    src = Y + m.to(tl.int64) * y_stride + l * (NOPE_D + 2 * HALF)
    offs = tl.arange(0, NOPE_D)
    x = tl.load(src + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / NOPE_D
    r = 1.0 / tl.sqrt(var + tl.load(EPS + l))
    w = tl.load(NW + l * NOPE_D + offs).to(tl.float32)
    yn = (x * r * w).to(tl.bfloat16).to(tl.float32).to(tl.float8e4nv)
    tl.store(dst + offs, yn.to(tl.uint8, bitcast=True), mask=valid & (offs < NOPE_D))
    i = tl.arange(0, HALF)
    pos = tl.load(POS + m).to(tl.int64)
    c = tl.load(CS + pos * cs_stride + i)
    s = tl.load(CS + pos * cs_stride + HALF + i)
    if IS_NEOX:
        ie = i
        io = i + HALF
    else:
        ie = 2 * i
        io = 2 * i + 1
    xe = tl.load(src + NOPE_D + ie).to(tl.float32)
    xo = tl.load(src + NOPE_D + io).to(tl.float32)
    oe = (xe * c - xo * s).to(tl.bfloat16).to(tl.float32).to(tl.float8e4nv)
    oo = (xo * c + xe * s).to(tl.bfloat16).to(tl.float32).to(tl.float8e4nv)
    tl.store(dst + NOPE_D + ie, oe.to(tl.uint8, bitcast=True), mask=valid & (i < HALF))
    tl.store(dst + NOPE_D + io, oo.to(tl.uint8, bitcast=True), mask=valid & (i < HALF))


def _served_dcp_owner(pool) -> Tuple[int, int]:
    """(world, rank) of the owner rule the served _scatter_mla_rows applies for this pool."""
    if pool.write_loc_is_dcp_resolved or pool.dcp_replicated:
        return 1, 0
    p = get_parallel()
    return int(p.attn_dcp_size), int(p.attn_dcp_rank)


def k31ht_ctxkv_block_reason(draft_model, pool, *, is_cuda: bool) -> Optional[str]:
    """None when the fused append reproduces the served per-layer path for this draft and pool, else the reason."""
    if not is_cuda:
        return "not CUDA"
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding.rope_variant import DeepseekScalingRotaryEmbedding
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
    from sglang.srt.models.dflash import DFlashDraftModel, DFlashMLAAttention

    if not isinstance(draft_model, DFlashDraftModel):
        return f"draft model {type(draft_model).__name__} is not a DFlashDraftModel"
    if type(draft_model).prepare_context_hidden_for_kv is not DFlashDraftModel.prepare_context_hidden_for_kv:
        return "prepare_context_hidden_for_kv is overridden (layers may not share the context rows)"
    layers = list(getattr(draft_model, "layers", []))
    if not layers:
        return "no draft layers"
    rot0 = None
    hidden = None
    for i, layer in enumerate(layers):
        a = layer.self_attn
        if not isinstance(a, DFlashMLAAttention):
            return f"layer {i} is not a DFlashMLAAttention"
        if a.q_lora_rank is None:
            return f"layer {i} has no fused a_proj (q_lora_rank None)"
        kvl, rd = int(a.kv_lora_rank), int(a.qk_rope_head_dim)
        if not (_is_pow2(kvl) and rd % 2 == 0 and _is_pow2(rd // 2)):
            return f"layer {i} kv_lora_rank {kvl} / rope dim {rd} not supported"
        if (kvl, rd) != (int(layers[0].self_attn.kv_lora_rank), int(layers[0].self_attn.qk_rope_head_dim)):
            return "layers differ in kv_lora_rank / rope dim"
        wt = a.fused_qkv_a_proj_with_mqa.weight
        if wt.dtype != torch.bfloat16 or wt.dim() != 2 or int(wt.shape[0]) != int(a.q_lora_rank) + kvl + rd:
            return f"layer {i} fused a_proj weight {tuple(wt.shape)} {wt.dtype} not the BF16 [q_lora + kv_lora + rope, hidden] form"
        if hidden is None:
            hidden = int(wt.shape[1])
        elif int(wt.shape[1]) != hidden:
            return "layers differ in hidden size"
        n = a.kv_a_layernorm
        if type(n) is not RMSNorm:
            return f"layer {i} kv_a_layernorm is {type(n).__name__}, not RMSNorm"
        if (
            not n.has_weight
            or n.cast_x_before_out_mul
            or n.override_orig_dtype is not None
            or getattr(n, "variance_size_override", None) is not None
            or tuple(n.weight.shape) != (kvl,)
            or n.weight.dtype != torch.bfloat16
        ):
            return f"layer {i} kv_a_layernorm options differ from the plain BF16-weight RMSNorm"
        rot = a.rotary_emb
        if type(rot) is not DeepseekScalingRotaryEmbedding:
            return f"layer {i} rotary_emb is {type(rot).__name__}, not DeepseekScalingRotaryEmbedding"
        cs = rot.cos_sin_cache
        if int(rot.rotary_dim) != rd or int(rot.head_size) != rd or cs.dtype != torch.float32 or cs.dim() != 2 or int(cs.shape[1]) != rd:
            return f"layer {i} rotary_emb shape / dtype not the FP32 [positions, {rd}] cache form"
        if rot0 is None:
            rot0 = rot
        elif rot is not rot0 and (
            bool(rot.is_neox_style) != bool(rot0.is_neox_style)
            or rot.cos_sin_cache.shape != rot0.cos_sin_cache.shape
            or not torch.equal(rot.cos_sin_cache, rot0.cos_sin_cache)
        ):
            return f"layer {i} rotary_emb differs from layer 0's"
    if pool is None or not isinstance(pool, MLATokenToKVPool):
        return f"draft pool {type(pool).__name__} is not an MLATokenToKVPool"
    for meth in ("set_mla_kv_buffer_prefix_valid", "set_mla_kv_buffer", "_write_mla_kv_buffer", "_scatter_mla_rows"):
        if getattr(type(pool), meth, None) is not getattr(MLATokenToKVPool, meth):
            return f"draft pool {type(pool).__name__} overrides {meth}"
    if pool.dtype != torch.float8_e4m3fn:
        return f"draft pool dtype {pool.dtype} is not float8_e4m3fn"
    if getattr(pool, "dsa_kv_cache_store_fp8", False):
        return "draft pool uses the DSA fp8 store"
    if getattr(pool, "fp8_cast_on_write", False):
        return "draft pool casts on write (fp8_cast_on_write)"
    kvw = int(layers[0].self_attn.kv_lora_rank) + int(layers[0].self_attn.qk_rope_head_dim)
    if int(pool.kv_cache_dim) != kvw:
        return f"draft pool kv_cache_dim {pool.kv_cache_dim} != {kvw}"
    rows = None
    for i, layer in enumerate(layers):
        idx = int(layer.self_attn.attn_mqa.layer_id) - int(pool.start_layer)
        if not 0 <= idx < len(pool.kv_buffer):
            return f"layer {i} has no buffer in the draft pool (layer_id {layer.self_attn.attn_mqa.layer_id})"
        buf = pool.kv_buffer[idx]
        if buf.element_size() != 1 or buf.dim() != 3 or int(buf.shape[1]) != 1 or int(buf.shape[2]) != kvw:
            return f"layer {i} pool buffer {tuple(buf.shape)} / element size {buf.element_size()} not the [rows, 1, {kvw}] byte form"
        if buf.stride(2) != 1 or buf.stride(0) != kvw:
            return f"layer {i} pool buffer is not row-contiguous"
        if rows is None:
            rows = int(buf.shape[0])
        elif int(buf.shape[0]) != rows:
            return "draft pool layer buffers differ in rows"
    return None


class K31HTCtxKvFuse:
    """Built once per worker (after the draft weights and the draft pool exist); write() replaces the per-layer loop."""

    def __init__(self, draft_model, pool):
        layers = list(draft_model.layers)
        a0 = layers[0].self_attn
        self.num_layers = len(layers)
        self.nope = int(a0.kv_lora_rank)
        self.rope = int(a0.qk_rope_head_dim)
        q = int(a0.q_lora_rank)
        with torch.no_grad():
            self.wcat = torch.cat([l.self_attn.fused_qkv_a_proj_with_mqa.weight[q:] for l in layers], 0).contiguous()
            self.nw = torch.stack([l.self_attn.kv_a_layernorm.weight for l in layers], 0).contiguous()
            self.eps = torch.tensor(
                [float(l.self_attn.kv_a_layernorm.variance_epsilon) for l in layers], dtype=torch.float32, device=self.wcat.device
            )
            self.ptrs = torch.tensor(
                [pool.kv_buffer[int(l.self_attn.attn_mqa.layer_id) - int(pool.start_layer)].data_ptr() for l in layers],
                dtype=torch.uint64,
                device=self.wcat.device,
            )
        self.row_stride = int(pool.kv_cache_dim)
        self.rot = a0.rotary_emb
        self.is_neox = bool(self.rot.is_neox_style)
        self.pool = pool

    @classmethod
    def build(cls, draft_model, pool, *, is_cuda: bool):
        reason = k31ht_ctxkv_block_reason(draft_model, pool, is_cuda=is_cuda)
        if reason is not None:
            return None, reason
        return cls(draft_model, pool), None

    def write(self, ctx_hidden: torch.Tensor, positions: torch.Tensor, cache_loc_2d: torch.Tensor, commit_lens: torch.Tensor) -> bool:
        """Write every layer's context KV for the prefix-valid rows; False (nothing written) when this call's inputs
        are outside the fused form, so the caller runs the served loop."""
        m = int(ctx_hidden.shape[0])
        if (
            ctx_hidden.dtype != self.wcat.dtype
            or ctx_hidden.dim() != 2
            or int(ctx_hidden.shape[1]) != int(self.wcat.shape[1])
            or cache_loc_2d.dim() != 2
            or int(cache_loc_2d.numel()) != m
            or positions.numel() != m
            or commit_lens.dtype != torch.int32
            or int(commit_lens.shape[0]) != int(cache_loc_2d.shape[0])
        ):
            return False
        if m == 0:
            return True
        loc2d = cache_loc_2d if cache_loc_2d.stride(1) == 1 else cache_loc_2d.contiguous()
        pos = positions if positions.is_contiguous() else positions.contiguous()
        cs = self.rot.cos_sin_cache
        world, rank = _served_dcp_owner(self.pool)
        y = F.linear(ctx_hidden, self.wcat)
        _k31ht_ctxkv_kernel[(m, self.num_layers)](
            y, y.stride(0), self.nw, self.eps, pos, cs, cs.stride(0), loc2d, loc2d.stride(0), commit_lens,
            int(cache_loc_2d.shape[1]), self.ptrs, self.row_stride,
            NOPE_D=self.nope, HALF=self.rope // 2, IS_NEOX=self.is_neox, SKIP=0, DCPW=world, DCPR=rank, num_warps=4,
        )  # fmt: skip
        return True
