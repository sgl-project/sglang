# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 hybrid attention inside a MiniMax-H3 DiT block: the per-head softmax
gate, the Video Delta linear branch and its output projection, the Ulysses
exchange that shards both branches by head, and the request-static attention
metadata. ``MiniMaxH3Attention`` owns one instance as ``hybrid`` and hands it
the raw q/k/v; nothing else in the base H3 model changes."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Mapping

import torch
from torch import nn

from sglang.kernels.ops.diffusion import (
    fused_qknorm_rope_out_of_place,
    usp_merge_heads,
)
from sglang.multimodal_gen.configs.models.dits.minimax_h3 import MiniMaxH3DiTArchConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_attn_h3 import (
    HybridWindowAttentionH3MetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.linear import RowParallelLinear
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.minimax_h3_vdn import (
    MiniMaxH3VDNLinearBranch,
    VDNSoftmaxGate,
    vdn_h3_layout_from_packed,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Attention

logger = logging.getLogger(__name__)
_FP32_DTYPE = torch.float32
_BF16_DTYPE = torch.bfloat16


class MiniMaxH3VDNHybridAttention(nn.Module):
    """out = to_out(gate_sm * window_softmax(q, k, v)) + to_out_linear(branch(q, k, v))."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        local_heads: int,
    ) -> None:
        super().__init__()
        hybrid = arch.hybrid_attention
        self.softmax_gate: VDNSoftmaxGate | None = None
        if hybrid.enable_softmax_gate:
            self.softmax_gate = VDNSoftmaxGate(
                arch.hidden_size,
                arch.num_attention_heads,
                prefix=f"{prefix}.softmax_gate",
            )
        self.linear_attention = MiniMaxH3VDNLinearBranch(
            arch, hybrid, local_heads=local_heads, prefix=f"{prefix}.linear_attention"
        )
        self.to_out_linear = RowParallelLinear(
            arch.num_attention_heads * arch.attention_head_dim,
            arch.hidden_size,
            bias=False,
            input_is_parallel=True,
            params_dtype=_BF16_DTYPE,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out_linear",
        )

    @classmethod
    def build(
        cls,
        arch: MiniMaxH3DiTArchConfig,
        quant_config: QuantizationConfig | None,
        *,
        prefix: str,
        local_heads: int,
    ) -> MiniMaxH3VDNHybridAttention | None:
        """None for the dense model and for the token refiner (VDN converts
        the transformer blocks only)."""
        if arch.hybrid_attention is None or not prefix.startswith("blocks."):
            return None
        return cls(
            arch, quant_config, prefix=f"{prefix}.hybrid", local_heads=local_heads
        )

    def forward(
        self,
        attention: MiniMaxH3Attention,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...] | None,
        max_seqlen: int,
        ulysses_active: bool,
        ring_active: bool,
    ) -> torch.Tensor:
        """VDN-H3: out = to_out(gate_sm * window_softmax) + to_out_linear(linear).

        Everything computed from x here (gates, beta) is row-local, so it runs
        on this rank's row shard before the core exchanges rows for heads.
        """
        if ring_active:
            raise NotImplementedError(
                "VDN-H3 hybrid attention does not support ring parallelism"
            )
        total = x.shape[0]
        softmax_gate = self.softmax_gate(x) if self.softmax_gate is not None else None
        beta = self.linear_attention.beta(x)
        gate_hidden, _ = self.linear_attention.output_gate.down(x)
        attention_core = (
            _hybrid_attention_core_bcg
            if attention.bcg_breakpoint
            else _minimax_h3_hybrid_attention_core_impl
        )
        softmax_out, linear_out = attention_core(
            attention,
            x,
            q,
            k,
            v,
            softmax_gate,
            beta,
            gate_hidden,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            ulysses_active=ulysses_active,
        )
        out, _ = attention.out_proj(softmax_out.reshape(total, -1))
        if linear_out is not None:
            far, _ = self.to_out_linear(linear_out)
            if far.shape[0] == total:
                out.add_(far)
            else:
                # single-rank path: the readout covers the video rows only
                meta = get_forward_context().attn_metadata
                layout = meta.layout
                out[layout.video_start : layout.video_end].add_(far)
        return out


def _vdn_frame_partial_sums(
    x: torch.Tensor,
    *,
    row_start: int,
    video_start: int,
    video_end: int,
    num_frames: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    """fp32 [F, hidden] per-frame sums of this rank's video rows: complete
    frames reduce as one [frames, tokens, hidden] sum, the two partial frames
    at the shard edges as two row sums. Deterministic, bf16 rows read once."""
    hidden = x.shape[-1]
    sums = torch.zeros(num_frames, hidden, dtype=_FP32_DTYPE, device=x.device)
    lo = max(row_start, video_start)
    hi = min(row_start + x.shape[0], video_end)
    if lo >= hi:
        return sums
    rows = x[lo - row_start : hi - row_start]
    first_frame, offset = divmod(lo - video_start, tokens_per_frame)
    lead = (tokens_per_frame - offset) % tokens_per_frame
    lead = min(lead, rows.shape[0])
    if lead:
        sums[first_frame] += rows[:lead].sum(0, dtype=_FP32_DTYPE)
        first_frame += 1
    full = (rows.shape[0] - lead) // tokens_per_frame
    if full:
        sums[first_frame : first_frame + full] = (
            rows[lead : lead + full * tokens_per_frame]
            .view(full, tokens_per_frame, hidden)
            .sum(1, dtype=_FP32_DTYPE)
        )
    tail = lead + full * tokens_per_frame
    if tail < rows.shape[0]:
        sums[first_frame + full] += rows[tail:].sum(0, dtype=_FP32_DTYPE)
    return sums


def _vdn_a2a_rows_to_heads(
    field: torch.Tensor, *, ulysses_ws: int, role: str, process_group
) -> tuple[torch.distributed.Work, torch.Tensor]:
    """Async all-to-all of one q/k/v field [L, H, d] (row shard, every head)
    -> receive buffer readable as a contiguous [S, H / ws, d] of this rank's
    heads (row shards are contiguous and rank-ordered, so no relayout on
    receipt; each field lands contiguous, which the window K/V gathers and the
    branch's conv read directly)."""
    from sglang.multimodal_gen.runtime.layers.usp import _a2a_staging_buffer

    rows, total_heads, head_dim = field.shape
    local_heads = total_heads // ulysses_ws
    send = _a2a_staging_buffer(
        role + "_send",
        (ulysses_ws, rows, local_heads, head_dim),
        field.dtype,
        field.device,
    )
    send.copy_(field.view(rows, ulysses_ws, local_heads, head_dim).permute(1, 0, 2, 3))
    recv = _a2a_staging_buffer(
        role + "_recv",
        (ulysses_ws * rows, local_heads, head_dim),
        field.dtype,
        field.device,
    )
    work = torch.distributed.all_to_all_single(
        recv, send, group=process_group, async_op=True
    )
    return work, recv


def _vdn_a2a_heads_to_rows(
    out: torch.Tensor, *, ulysses_ws: int, role: str, process_group
) -> tuple[torch.distributed.Work, torch.Tensor]:
    """Async inverse: [S, H / ws, d] head-shard output -> receive buffer
    [ws, L, H / ws, d] (source rank major); :func:`_vdn_merge_heads` turns it
    into the row shard after the wait."""
    from sglang.multimodal_gen.runtime.layers.usp import _a2a_staging_buffer

    seq_len, local_heads, head_dim = out.shape
    rows = seq_len // ulysses_ws
    recv = _a2a_staging_buffer(
        role + "_recv", (ulysses_ws, rows, local_heads, head_dim), out.dtype, out.device
    )
    work = torch.distributed.all_to_all_single(
        recv, out.contiguous(), group=process_group, async_op=True
    )
    return work, recv


def _vdn_merge_heads(recv: torch.Tensor) -> torch.Tensor:
    """[ws, L, h, d] (source rank major) -> [L, ws * h, d]: rank-major heads
    are the global head order."""
    ulysses_ws, rows, local_heads, head_dim = recv.shape
    merged = usp_merge_heads(recv.view(ulysses_ws, rows, 1, local_heads, head_dim))
    return merged.reshape(rows, ulysses_ws * local_heads, head_dim)


def _vdn_window_softmax(
    attention: MiniMaxH3Attention,
    meta,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_gate: torch.Tensor | None,
    rope_cache: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
) -> torch.Tensor:
    """Gated window softmax on out-of-place QK-normed + RoPE'd copies of the
    raw q/k [S, h, d] (the branch keeps reading the raw tensors)."""
    cos_sin_cache, positions = rope_cache
    if attention._use_fused_qknorm_rope and not torch.compiler.is_compiling():
        q_sm = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        k_sm = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        fused_qknorm_rope_out_of_place(
            q,
            k,
            q_sm,
            k_sm,
            attention.q_norm.weight,
            attention.k_norm.weight,
            cos_sin_cache,
            positions,
            is_neox=True,
            eps=attention.q_norm.eps,
            head_dim=attention.head_dim,
            rope_dim=cos_sin_cache.shape[-1],
            round_norm_before_rope=True,
        )
    else:
        from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
            _apply_qk_norm,
            _apply_rope_qk,
        )

        q_sm, k_sm = _apply_qk_norm(
            q.clone(), k.clone(), attention.q_norm, attention.k_norm, attention.head_dim
        )
        q_sm, k_sm = _apply_rope_qk(q_sm, k_sm, cos_sin_cache, positions)
    return attention._attention_impl.forward_varlen(
        q_sm,
        k_sm,
        v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        cu_seqlens_host=cu_seqlens_host,
        attn_metadata=meta,
        softmax_gate=softmax_gate,
    )


def _vdn_linear_readout(
    attention: MiniMaxH3Attention,
    meta,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    beta: torch.Tensor,
    linear_gate: torch.Tensor,
    frame_mean: torch.Tensor,
    head_range: slice | None,
) -> torch.Tensor:
    """Linear branch on the raw q/k/v [S, h, d] for the ``head_range`` heads:
    video rows in, [V, h * d] out (text rows seed the state)."""
    layout = meta.layout
    video = slice(layout.video_start, layout.video_end)
    text = slice(0, layout.text_len)
    return attention.hybrid.linear_attention(
        q_raw=q[video],
        k_raw=k[video],
        v_raw=v[video],
        beta=beta[video],
        gate=linear_gate[video],
        frame_mean=frame_mean,
        layout=layout,
        text_k_raw=k[text],
        text_v_raw=v[text],
        text_beta=beta[text],
        heads=head_range,
    )


def _minimax_h3_hybrid_attention_core_impl(
    attention: MiniMaxH3Attention,
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_gate: torch.Tensor | None,
    beta: torch.Tensor,
    gate_hidden: torch.Tensor,
    *,
    rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...] | None,
    max_seqlen: int,
    ulysses_active: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """VDN-H3 hybrid attention core: window softmax + linear branch.

    Takes the RAW (pre-QK-norm, pre-RoPE) q/k/v because the linear branch
    consumes them NoPE; QK-norm + RoPE run out of place on the softmax
    branch's copies. ``gate_hidden`` is the 128-wide output_gate.down hidden
    of the rows; the gate's ``up`` runs here (under Ulysses on the head shard
    after an all-gather, so the H x d gate never crosses the fabric). Under
    Ulysses q, k, v and the per-head scalars travel as four async all-to-alls
    that land contiguous on the head shard, norm + RoPE use the request's
    full-sequence RoPE cache, and the frame mean is one all-reduce of partial
    sums. The branch is per-head independent given (beta, gate, alpha), so head
    sharding is exact. Returns (gated softmax output [T_local, H, d], linear
    readout [T_local, H * d] or None when the window covers the whole clip).
    """
    from sglang.multimodal_gen.runtime.layers.attention.backends.hybrid_window_attn_h3 import (
        HybridWindowAttentionH3Metadata,
    )

    meta = get_forward_context().attn_metadata
    if not isinstance(meta, HybridWindowAttentionH3Metadata):
        raise RuntimeError(
            "VDN-H3 hybrid attention needs HybridWindowAttentionH3Metadata in the "
            "forward context; the MiniMax-H3 denoising stage installs it per request "
            f"(got {type(meta).__name__})."
        )
    layout = meta.layout
    softmax = functools.partial(
        _vdn_window_softmax,
        attention,
        meta,
        cu_seqlens=cu_seqlens,
        cu_seqlens_host=cu_seqlens_host,
        max_seqlen=max_seqlen,
    )
    if not ulysses_active:
        if rope_cache is None:
            raise RuntimeError("VDN-H3 hybrid attention requires the RoPE cache")
        softmax_out = softmax(q, k, v, softmax_gate=softmax_gate, rope_cache=rope_cache)
        if meta.full_cover:
            return softmax_out, None
        frame_mean = (
            x[layout.video_start : layout.video_end]
            .view(layout.num_frames, layout.tokens_per_frame, x.shape[-1])
            .mean(dim=1, dtype=_FP32_DTYPE)
        )
        readout = _vdn_linear_readout(
            attention,
            meta,
            q,
            k,
            v,
            beta=beta,
            linear_gate=attention.hybrid.linear_attention.output_gate.up_gate(
                gate_hidden
            ),
            frame_mean=frame_mean,
            head_range=None,
        )
        return softmax_out, readout

    return _vdn_ulysses_hybrid_core(
        attention,
        meta,
        x,
        q,
        k,
        v,
        softmax_gate=softmax_gate,
        beta=beta,
        gate_hidden=gate_hidden,
        softmax=softmax,
    )


def _vdn_ulysses_hybrid_core(
    attention: MiniMaxH3Attention,
    meta,
    x: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_gate: torch.Tensor | None,
    beta: torch.Tensor,
    gate_hidden: torch.Tensor,
    softmax,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Ulysses path: four async all-to-alls (q, k, v, per-head scalars) land
    contiguous on the head shard; the gate hidden is all-gathered; the frame
    mean is one all-reduce; both outputs return through the inverse."""
    from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_group

    layout = meta.layout
    if meta.rope_cache_full is None:
        raise RuntimeError("VDN-H3 under Ulysses needs the full-sequence RoPE cache")
    sp_group = get_sp_group()
    process_group = sp_group.ulysses_group
    ulysses_ws, ulysses_rank = get_ulysses_ctx()
    local_rows, head_dim = x.shape[0], q.shape[2]
    local_heads = q.shape[1] // ulysses_ws
    seq_len = local_rows * ulysses_ws

    # the q/k/v exchange is in flight while the frame sums, the small per-head
    # fields and the gate hidden go out
    inflight = [
        _vdn_a2a_rows_to_heads(
            field,
            ulysses_ws=ulysses_ws,
            role=f"vdn_{name}",
            process_group=process_group,
        )
        for name, field in (("q", q), ("k", k), ("v", v))
    ]
    frame_sums = _vdn_frame_partial_sums(
        x,
        row_start=ulysses_rank * local_rows,
        video_start=layout.video_start,
        video_end=layout.video_end,
        num_frames=layout.num_frames,
        tokens_per_frame=layout.tokens_per_frame,
    )
    frame_work = torch.distributed.all_reduce(
        frame_sums, group=sp_group.device_group, async_op=True
    )
    # the per-head scalars (beta, softmax gate) ride one more async field
    scalars = [beta] if softmax_gate is None else [beta, softmax_gate]
    inflight.append(
        _vdn_a2a_rows_to_heads(
            torch.stack(scalars, dim=-1),
            ulysses_ws=ulysses_ws,
            role="vdn_scalars",
            process_group=process_group,
        )
    )
    head_range = slice(ulysses_rank * local_heads, (ulysses_rank + 1) * local_heads)
    linear_gate = attention.hybrid.linear_attention.output_gate.up_gate(
        sp_group.all_gather(gate_hidden.contiguous(), dim=0), heads=head_range
    )

    for work, _ in inflight:
        work.wait()
    q, k, v, scalars = (recv for _, recv in inflight)
    beta = scalars[..., 0]
    if softmax_gate is not None:
        softmax_gate = scalars[..., 1]
    softmax_out = softmax(
        q, k, v, softmax_gate=softmax_gate, rope_cache=meta.rope_cache_full
    )
    linear_out = None
    if not meta.full_cover:
        frame_work.wait()
        readout = _vdn_linear_readout(
            attention,
            meta,
            q,
            k,
            v,
            beta=beta,
            linear_gate=linear_gate,
            frame_mean=frame_sums / layout.tokens_per_frame,
            head_range=head_range,
        )
        # rows go back to their owners: pad the non-video rows with zeros
        linear_out = q.new_zeros(seq_len, local_heads, head_dim)
        linear_out[layout.video_start : layout.video_end] = readout.view(
            -1, local_heads, head_dim
        )
        del readout
    else:
        frame_work.wait()
    return _vdn_return_to_rows(
        softmax_out, linear_out, ulysses_ws=ulysses_ws, process_group=process_group
    )


def _vdn_return_to_rows(
    softmax_out: torch.Tensor,
    linear_out: torch.Tensor | None,
    *,
    ulysses_ws: int,
    process_group,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Both branch outputs [S, h, d] back to their row owners with the return
    trips in flight together -> ([L, H, d], [L, H * d] or None)."""
    outs = [out for out in (softmax_out, linear_out) if out is not None]
    returning = [
        _vdn_a2a_heads_to_rows(
            out, ulysses_ws=ulysses_ws, role=f"vdn_out{i}", process_group=process_group
        )
        for i, out in enumerate(outs)
    ]
    merged = []
    for work, recv in returning:
        work.wait()
        merged.append(_vdn_merge_heads(recv))
    linear_rows = (
        merged[1].reshape(merged[1].shape[0], -1) if linear_out is not None else None
    )
    return merged[0], linear_rows


_hybrid_attention_core_bcg = eager_on_graph(True)(
    _minimax_h3_hybrid_attention_core_impl
)


def prepare_hybrid_attention_metadata(
    *,
    model,
    packed: Mapping[str, torch.Tensor],
    latent_shape: tuple[int, int, int],
    condition_rows: bool,
    server_args,
    device: torch.device,
) -> Callable[[int], Any] | None:
    """Request-static VDN-H3 hybrid attention metadata (window plan, packed
    layout, full-sequence RoPE cache under Ulysses), or None off that path.

    Unlike VSA-H3's per-step top-k, the chunk window and anchors depend only
    on the packed layout, so one metadata object serves every step.
    """
    model._resolve_attention_backend_once()
    if (
        model._resolved_attention_backend
        is not AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3
    ):
        return None
    hybrid = model.arch.hybrid_attention
    if hybrid is None:
        raise ValueError(
            "--attention-backend hybrid_window_attn_h3 needs a VDN-H3 checkpoint "
            "(transformer/config.json with hybrid_attention); this checkpoint has "
            "no linear branch. Use --attention-backend fa for MiniMax-H3."
        )
    if condition_rows:
        raise NotImplementedError(
            "VDN-H3 is trained for the t2va packed layout; condition rows "
            "(fl2va keyframes, ref2va references) are not supported."
        )

    from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import (
        _rope_cos_sin_cache,
    )

    config = server_args.attention_backend_config or {}
    max_gather_rows = int(config.get("vdn_max_gather_rows", 200_000))
    latent_t, latent_h, latent_w = latent_shape
    layout = vdn_h3_layout_from_packed(
        packed, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w
    )
    rope_cache_full = None
    ulysses_ws, _ = get_ulysses_ctx()
    ring_ws, _ = get_ring_ctx()
    if ring_ws > 1:
        raise ValueError("VDN-H3 does not support ring parallelism")
    if ulysses_ws > 1:
        # QK-norm + RoPE run after the Ulysses all-to-all on the head shard
        # over the full sequence, so every rank needs the whole cache.
        with torch.inference_mode():
            img_position_ids = (
                packed["img_position_ids"][None].to(torch.float32).to(device)
            )
            rope_freqs = model.rope(img_position_ids)
            rope_cache_full = (
                _rope_cos_sin_cache(rope_freqs, dtype=torch.bfloat16),
                torch.arange(layout.seq_len, device=device, dtype=torch.long),
            )
    metadata = HybridWindowAttentionH3MetadataBuilder().build(
        layout=layout,
        hybrid=hybrid,
        device=device,
        rope_cache_full=rope_cache_full,
        max_gather_rows=max_gather_rows,
    )
    logger.info(
        "VDN-H3 hybrid attention: frames=%d tokens/frame=%d text=%d "
        "used=%d/%d chunk=%d radius=%d anchors=%s full_cover=%s",
        layout.num_frames,
        layout.tokens_per_frame,
        layout.text_len,
        layout.used,
        layout.seq_len,
        hybrid.chunk,
        hybrid.radius,
        hybrid.anchor_frames,
        metadata.full_cover,
    )

    def build(step_index: int):
        return metadata

    return build


__all__ = ["MiniMaxH3VDNHybridAttention", "prepare_hybrid_attention_metadata"]
