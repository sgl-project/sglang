import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch_npu
from sgl_kernel_npu.norm.fused_split_qk_norm import fused_split_qk_norm

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.attention.dsa.dsa_cp import (
    dsa_cp_redistribute_heads,
    dsa_cp_restore_tokens,
    dsa_cp_slice,
    get_dsa_cp_plan,
)
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import ScatterMode, get_attn_tp_context
from sglang.srt.layers.dcp import (
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla,
    dcp_a2a_lse_reduce,
)
from sglang.srt.layers.dcp.layout import (
    dcp_extend_gather_buffer,
    plan_dcp_extend_gather,
)
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_dcp_mla_decode_phase,
    is_mla_dcp_lse_base_on_e,
)
from sglang.srt.runtime_context import get_disagg, get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator

logger = logging.getLogger(__name__)

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
_is_npu_arch35 = is_npu_arch35()
_debug_dcp_extend_memory = envs.SGLANG_DEBUG_NPU_DCP_EXTEND_MEMORY.get()


# region MHA
def forward_mha_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    layer_scatter_modes,
):
    if m.q_lora_rank is not None:
        q, latent_cache = (
            get_attn_tp_context()
            .fetch_qkv_latent()
            .split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                dim=-1,
            )
        )

        # DSA Indexer: cache quantized keys, auto-skip topk for sequences <= dsa_index_topk

        if m.use_dsa:
            q_lora = m.q_a_layernorm(q)
            q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            _ = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
                return_indices=False,
            )

        else:
            q = m.q_a_layernorm(q)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)

    else:
        q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]

    _, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)
    kv_a, _ = latent_cache.split([m.kv_lora_rank, m.qk_rope_head_dim], dim=-1)
    latent_cache = latent_cache.unsqueeze(1)

    if m.use_deepseek_yarn_rope:
        B, S = q.shape[0], 1
        cos, sin = m.rotary_emb.get_cos_sin_cache(
            positions, hidden_states.dtype, offsets=None
        )
        q_pe = torch_npu.npu_interleave_rope(
            q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
            cos,
            sin,
        )
        q_pe = q_pe.reshape(B, -1, m.qk_rope_head_dim)

        ckv_cache, k_rope_cache = get_token_to_kv_pool().get_kv_buffer(m.layer_id)
        _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache.view(-1, 1, 1, m.kv_lora_rank + m.qk_rope_head_dim),  # bnsd
            m.kv_a_layernorm.weight,
            cos,
            sin,
            forward_batch.out_cache_loc.to(torch.int64),
            k_rope_cache,
            ckv_cache,
            k_rope_scale=None,
            c_kv_scale=None,
            k_rope_offset=None,
            c_kv_offset=None,
            epsilon=m.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ" if is_fia_nz() else "PA_BNSD",
            is_output_kv=True,
        )  # adapter NZ

        k_pe = k_pe.reshape(B, -1, m.qk_rope_head_dim)
    else:
        kv_a = m.kv_a_layernorm(kv_a)
        k_pe = latent_cache[:, :, m.kv_lora_rank :]
        if m.rotary_emb is not None:
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)
        # this is for model kimi-vl-a3B-instruct
        get_token_to_kv_pool().set_kv_buffer(
            m, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
        )

    q[..., m.qk_nope_head_dim :] = q_pe

    kv = m.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, m.num_local_heads, m.qk_nope_head_dim + m.v_head_dim)
    k_nope = kv[..., : m.qk_nope_head_dim]
    v = kv[..., m.qk_nope_head_dim :]

    k = m._concat_and_cast_mha_k(k_nope, k_pe, forward_batch)
    return q, k, v, forward_batch


def forward_mha_core_npu(
    m: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: "ForwardBatch",
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_output = m.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
    attn_output = attn_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    if gate is not None:
        attn_output = m._apply_gated(attn_output, gate)
    output, _ = m.o_proj(attn_output)
    return output


# endregion


# region MLA
def forward_mla_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    layer_scatter_modes,
):
    if is_mla_preprocess_enabled():
        if not hasattr(m, "mla_preprocess"):
            m.mla_preprocess = NPUFusedMLAPreprocess(
                m.fused_qkv_a_proj_with_mqa,
                m.q_a_layernorm,
                m.kv_a_layernorm,
                m.q_b_proj,
                m.w_kc,
                m.rotary_emb,
                m.layer_id,
                m.num_local_heads,
                m.qk_nope_head_dim,
                m.qk_rope_head_dim,
                m.quant_config,
            )
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        topk_indices = None
    else:
        q_lora = None
        if m.q_lora_rank is not None:
            qkv_latent = get_attn_tp_context().fetch_qkv_latent()
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                q, latent_cache = qkv_latent.split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                    dim=-1,
                )
                k_nope = latent_cache[..., : m.kv_lora_rank]

                q = m.q_a_layernorm(q)
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)

                k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
                k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)
            else:
                if (
                    qkv_latent.shape[0] < 65536
                    and not dsa_use_prefill_cp(forward_batch)
                    and not getattr(m, "_disable_npu_fused_split_qk_norm", False)
                ):
                    q, k_nope, k_pe = fused_split_qk_norm(
                        qkv_latent,
                        m.q_a_layernorm,
                        m.kv_a_layernorm,
                        m.q_lora_rank,
                        m.kv_lora_rank,
                        m.qk_rope_head_dim,
                        eps=m.q_a_layernorm.variance_epsilon,
                    )
                else:
                    # The fused split+RMSNorm kernel is not numerically equivalent
                    # on Ascend. Keep the unfused path for models that opt out.
                    q, latent_cache = qkv_latent.split(
                        [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim],
                        dim=-1,
                    )
                    k_nope = latent_cache[..., : m.kv_lora_rank]

                    q = m.q_a_layernorm(q)

                    k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
                    k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

            # q_lora needed by indexer
            if m.use_dsa:
                q_lora = q

            q = m.q_b_proj(q)[0].view(-1, m.num_local_heads, m.qk_head_dim)
        else:
            q = m.q_proj(hidden_states)[0].view(-1, m.num_local_heads, m.qk_head_dim)
            latent_cache = m.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : m.kv_lora_rank]
            k_nope = m.kv_a_layernorm(k_nope).unsqueeze(1)
            k_pe = latent_cache[..., m.kv_lora_rank :].unsqueeze(1)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if m.rotary_emb is not None:
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        topk_indices = None
        if q_lora is not None:
            topk_indices = m.indexer(
                x=hidden_states,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                layer_id=m.layer_id,
            )

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
    )


def forward_mla_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
    topk_indices: torch.Tensor,
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_output = m.attn_mqa(
        q_nope_out,
        k_nope,
        k_nope,
        forward_batch,
        q_rope=q_pe,
        k_rope=k_pe,
        **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
    )

    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_output = attn_output.contiguous()
    if (
        attn_output.shape[0] >= 65536
        or attn_output.shape[-1] * attn_output.shape[-2] >= 65536
        or m.w_vc.shape[-1] >= 65536
    ):
        # npu_transpose_batchmatmul does not support dimensions >= 65536.
        attn_bmm_output = torch.empty(
            (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)
    else:
        # Use the numerically validated torch_npu implementation when supported.
        attn_bmm_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            m.w_vc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    if gate is not None:
        attn_bmm_output = m._apply_gated(attn_bmm_output, gate)
    output, _ = m.o_proj(attn_bmm_output)

    return output


# endregion


# region DSA
def _apply_interleaved_rope_with_half_output(rotary_emb, positions, q_pe, k_pe):
    """Apply RoPE to interleaved Q/K and return half-layout outputs."""
    rotary_emb.get_cos_sin_with_position(positions)
    cos = rotary_emb.position_cos.to(device=q_pe.device, dtype=q_pe.dtype).view(
        -1, 1, 1, q_pe.shape[-1]
    )
    sin = rotary_emb.position_sin.to(device=q_pe.device, dtype=q_pe.dtype).view(
        -1, 1, 1, q_pe.shape[-1]
    )
    q_pe = torch_npu.npu_interleave_rope(q_pe.unsqueeze(2), cos, sin).squeeze(2)
    k_pe = torch_npu.npu_interleave_rope(k_pe.unsqueeze(2), cos, sin).squeeze(2)
    return q_pe, k_pe


def forward_dsa_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    layer_scatter_modes,
    prev_topk_indices: torch.Tensor = None,
):
    dynamic_scale = None
    # Resolve DSA-CP once per forward here, because this half of the pair
    # receives layer_scatter_modes and the gate needs it; the core reads the
    # cached result back. index_topk comes from m.indexer, which exists only on
    # the layers that compute a top-k; None keeps the operator's causal crop.
    get_dsa_cp_plan(
        forward_batch,
        layer_scatter_modes,
        m.indexer.index_topk if m.indexer is not None else None,
    )
    mla_preprocess_used = (
        is_mla_preprocess_enabled()
        and not forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
    )
    if mla_preprocess_used:
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            q_lora,
            forward_batch,
            zero_allocator,
            positions,
            dynamic_scale,
        ) = npu_mla_preprocess(
            m,
            hidden_states,
            positions,
            forward_batch,
            zero_allocator,
        )
    else:
        fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
        if m.rotary_emb.is_neox_style:
            q, latent_cache = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            # overlap qk norm
            q = m.q_a_layernorm(q)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q_lora = q.clone()  # required for topk_indices

            q_event = None
            if m.alt_stream is not None:
                m.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(m.alt_stream):
                    q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)
                    # record q to ensure memory space will not be released
                    q.record_stream(m.alt_stream)
                    q_event = m.alt_stream.record_event()
            else:
                q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)

            k_nope, k_pe = latent_cache.unsqueeze(1).split(
                [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
            )
            k_nope = m.kv_a_layernorm(k_nope)
            # main stream waits for the completion of the event on the alt stream to ensure data dependency is complete
            if q_event is not None:
                torch.npu.current_stream().wait_event(q_event)
        else:
            if (
                fused_qkv_a_proj_out.shape[0] < 65535
                and not dsa_use_prefill_cp(forward_batch)
                and not getattr(m, "_disable_npu_fused_split_qk_norm", False)
            ):
                q_lora, k_nope, k_pe = fused_split_qk_norm(
                    fused_qkv_a_proj_out,
                    m.q_a_layernorm,
                    m.kv_a_layernorm,
                    m.q_lora_rank,
                    m.kv_lora_rank,
                    m.qk_rope_head_dim,
                    eps=m.q_a_layernorm.variance_epsilon,
                )
            else:
                # Keep the numerically validated unfused path for models that
                # explicitly opt out of the fused split and RMSNorm kernel.
                q, latent_cache = fused_qkv_a_proj_out.split(
                    [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
                )
                # overlap qk norm
                q = m.q_a_layernorm(q)

                q_lora = q.clone()  # required for topk_indices
                k_nope, k_pe = latent_cache.unsqueeze(1).split(
                    [m.kv_lora_rank, m.qk_rope_head_dim], dim=-1
                )
                k_nope = m.kv_a_layernorm(k_nope)
            q = m.q_b_proj(q_lora)[0].view(-1, m.num_local_heads, m.qk_head_dim)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        q_nope_out = torch_npu.npu_transpose_batchmatmul(
            q_nope,
            m.w_kc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )

        if is_mla_preprocess_enabled() and not m.rotary_emb.is_neox_style:
            # Match the half-layout RoPE outputs used by MLA preprocessing.
            q_pe, k_pe = _apply_interleaved_rope_with_half_output(
                m.rotary_emb, positions, q_pe, k_pe
            )
        else:
            if m.layer_id == get_token_to_kv_pool().start_layer:
                m.rotary_emb.sin_cos_cache = m.rotary_emb.cos_sin_cache.index_select(
                    0, positions
                )
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

    if not m.skip_topk or (m.is_nextn and prev_topk_indices is None):
        topk_indices = m.indexer(
            hidden_states,
            q_lora,
            positions,
            forward_batch,
            m.layer_id,
            layer_scatter_modes,
            dynamic_scale,
        )
    else:
        topk_indices = prev_topk_indices

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        topk_indices,
        forward_batch,
        zero_allocator,
        positions,
        mla_preprocess_used,
    )


# Gathered rows per extend-gather collective. A bf16 latent row is 1024 bytes,
# so the default caps a piece's scratch at 256 MiB of latent KV. The bytes
# moved are the same at any piece size; the trade is launch overhead against
# the scratch held beside the output.
_dcp_extend_gather_piece_rows = envs.SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS.get()
if _dcp_extend_gather_piece_rows <= 0:
    _dcp_extend_gather_piece_rows = 1 << 62

_last_dcp_extend_rows: Optional[Tuple[int, int]] = None


def _log_dcp_extend_memory(prefix_rows: int, extend_rows: int) -> None:
    """Log the previous DCP extend's peak device memory on this rank, then reset.

    Called on the first layer of each DCP extend forward, so the peak spans one
    whole forward -- plus whatever ran between the two extends, such as decode
    steps. It is the number a chunk size and ``--mem-fraction-static`` have to
    fit under; npu-smi shows only what the allocator has reserved, which at a
    stall is the whole die.
    """
    global _last_dcp_extend_rows
    if _last_dcp_extend_rows is not None:
        gib = 1 << 30
        logger.info(
            "DCP extend memory: prefix=%d extend=%d peak_allocated=%.2f GiB "
            "allocated=%.2f GiB reserved=%.2f GiB",
            *_last_dcp_extend_rows,
            torch.npu.max_memory_allocated() / gib,
            torch.npu.memory_allocated() / gib,
            torch.npu.memory_reserved() / gib,
        )
    torch.npu.reset_peak_memory_stats()
    _last_dcp_extend_rows = (prefix_rows, extend_rows)


def _pad_dcp_extend_send(shards: torch.Tensor, plan) -> torch.Tensor:
    """Lay this rank's per-request shards out at their padded send offsets."""
    send = shards.new_empty((plan.send_rows, *shards.shape[1:]))
    src = dst = 0
    for local_len, padded_len in zip(plan.local_lens, plan.padded_lens):
        send[dst : dst + local_len] = shards[src : src + local_len]
        src += local_len
        dst += padded_len
    return send


def _dcp_gather_extend_kv_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise each request's FULL prefix+extend KV, for one layer.

    A rank holds 1/c of the context, so at extend it gathers the rest rather
    than sharding the attention and merging: prefill has many query tokens and
    few heads, which makes moving the KV the cheap direction. Decode is the
    mirror image and gathers the query instead.

    The result is one contiguous run per request, which is what
    ``npu_sparse_flash_attention`` needs under a non-paged layout -- sparse
    indices relative to each request's KV start, cumulative KV lengths. The
    shared ``all_gather_kv_cache_for_mla_extend`` cannot be used: it groups all
    prefixes then all extends, so a request's KV is two disjoint runs.

    The prefix is gathered in pieces of at most
    ``SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS`` rows; each piece is written into
    its place in the output by one ``index_select`` before the next is gathered.
    Output and scratch come from ``dcp_extend_gather_buffer`` and are reused by
    every layer and forward, so nothing context-sized is allocated after the
    first extend -- a moving context-sized allocation makes ranks free and
    refill out of step and stalls the collectives for minutes.

    The two keys are gathered separately because the operator takes them as
    separate tensors and slices of a wide buffer are not contiguous. Only the
    prefix is gathered; this chunk's own KV arrives as ``k_nope``/``k_pe``.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    plan = getattr(forward_batch, "npu_dcp_extend_gather", None)
    if plan is None:
        # Sized by the shared planner for the whole context, for CUDA's
        # kernels. Nothing on this path reads it, and held it is a
        # context-sized tensor alive through every layer.
        md.dcp_kv_buffer = None
        plan = plan_dcp_extend_gather(
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            parallel.dcp_size,
            parallel.dcp_rank,
            _dcp_extend_gather_piece_rows,
        )
        plan = plan._replace(
            pieces=[
                piece._replace(index=piece.index.to(k_nope.device))
                for piece in plan.pieces
            ]
        )
        forward_batch.npu_dcp_extend_gather = plan
        # Tell the pool it may drop the rows this rank does not own from this
        # forward's KV write. getattr because a wrapper pool (SWA, hybrid) may
        # not know about this; declining leaves the pre-existing behaviour.
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(
                sum(forward_batch.extend_prefix_lens_cpu),
                sum(forward_batch.extend_seq_lens_cpu),
            )

    total_rows = plan.pieces[-1].out_end if plan.pieces else 0
    out_nope = dcp_extend_gather_buffer("latent", k_nope, total_rows)
    out_rope = dcp_extend_gather_buffer("rope", k_pe, total_rows)

    # One scratch per key, sized for the widest piece and sliced per piece. The
    # widest is not always the first: the last piece also carries this chunk's
    # own KV.
    scratch_rows = max(
        (
            (piece.send_end - piece.send_start) * parallel.dcp_size
            + piece.extend_end
            - piece.extend_start
            for piece in plan.pieces
        ),
        default=0,
    )
    scratch_nope = dcp_extend_gather_buffer("latent_scratch", k_nope, scratch_rows)
    scratch_rope = dcp_extend_gather_buffer("rope_scratch", k_pe, scratch_rows)

    send_nope = send_rope = None
    if plan.send_rows:
        send_nope, send_rope = get_token_to_kv_pool().get_mla_kv_buffer(
            m.attn_mqa,
            md.dcp_local_prefix_kv_indices,
        )
        if plan.local_lens != plan.padded_lens:
            # Served prefixes are dcp_size-aligned (the widened allocator page)
            # and need no padding; this is the general case.
            send_nope = _pad_dcp_extend_send(send_nope, plan)
            send_rope = _pad_dcp_extend_send(send_rope, plan)

    # Every rank plans the same pieces, so every rank runs -- or, for a piece
    # with no prefix rows, skips -- the same collectives in the same order.
    for piece in plan.pieces:
        gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
        rows = gathered + piece.extend_end - piece.extend_start
        for out, buf, send, own in (
            (out_nope, scratch_nope, send_nope, k_nope),
            (out_rope, scratch_rope, send_rope, k_pe),
        ):
            scratch = buf[:rows]
            if gathered:
                parallel.dcp_group.all_gather_into_tensor(
                    scratch[:gathered], send[piece.send_start : piece.send_end]
                )
            scratch[gathered:] = own[piece.extend_start : piece.extend_end]
            torch.index_select(
                scratch, 0, piece.index, out=out[piece.out_start : piece.out_end]
            )
    return out_nope, out_rope


def forward_dsa_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    topk_indices: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
    mla_preprocess_used: bool,
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # GLM-5.2 dispatches AttnForwardMethod.DSA_NPU here, not to
    # forward_absorb_core, so forward_mla.py's DCP block never runs for it and
    # DCP is composed here too. The two prepare/core pairs have different
    # shapes, so this mirrors forward_mla.py rather than sharing it.
    dcp_extend = (
        get_parallel().dcp_enabled
        and forward_batch.forward_mode.is_extend()
        and not is_dcp_mla_decode_phase(forward_batch)
        and forward_batch.attn_dcp_metadata is not None
    )
    if dcp_extend:
        # Extend under DCP: gather the context so this rank can see all of it,
        # and hand the result to the backend on the batch. Without it the
        # backend attends over its own shard with a full-span page table --
        # in bounds and wrong.
        forward_batch.npu_dcp_extend_kv = _dcp_gather_extend_kv_npu(
            m, forward_batch, k_nope, k_pe
        )

    if is_dcp_mla_decode_phase(forward_batch):
        # Every rank attends with the full head set against its own KV shard
        # and keeps its own share after the merge, so the query is gathered
        # across the group and attention runs on attn_mqa_for_dcp_decode, built
        # at num_local_heads * dcp_size.
        q_nope_out, q_pe = all_gather_q_for_mla_decode(q_nope_out=q_nope_out, q_pe=q_pe)
        # save_kv_cache stays True where the non-DCP branch takes
        # `not mla_preprocess_used`: under DCP the write goes through
        # _resolve_dcp_write, and a duplicate write is idempotent.
        attn_output, lse = m.attn_mqa_for_dcp_decode(
            q_nope_out.contiguous(),
            k_nope.contiguous(),
            k_nope.contiguous(),
            forward_batch,
            save_kv_cache=True,
            q_rope=q_pe.contiguous(),
            k_rope=k_pe.contiguous(),
            topk_indices=topk_indices,
        )
        # The partials are per-head over this rank's tokens; the merge reduces
        # the head axis back to num_local_heads, which is why the shared view
        # below is correct for both branches.
        attn_output = attn_output.view(
            -1, m.num_local_heads * get_parallel().attn_dcp_size, m.kv_lora_rank
        )
        comm_backend = get_parallel().dcp_comm_backend
        # Not cosmetic: feeding a base-e LSE to the base-2 combine is a monotone
        # reweighting, so it stays finite and plausible and only acceptance
        # degrades. "ascend" is a natural-log backend.
        base_on_e = is_mla_dcp_lse_base_on_e(m.current_attention_backend)
        if comm_backend in ("a2a", "fi_a2a"):
            attn_output = dcp_a2a_lse_reduce(
                attn_output.contiguous(),
                lse.contiguous(),
                get_parallel().dcp_group,
                is_lse_base_on_e=base_on_e,
                comm_backend=comm_backend,
            )
        else:
            attn_output = cp_lse_ag_out_rs_mla(
                attn_output,
                lse,
                get_parallel().dcp_group,
                is_lse_base_on_e=base_on_e,
            )
            attn_output = attn_output.transpose(0, 1)
    else:
        attn_mqa = m.attn_mqa
        dsa_cp_plan = get_dsa_cp_plan(forward_batch)
        if dsa_cp_plan is not None and (
            topk_indices is None or m.attn_mqa_for_dsa_cp is None
        ):
            # Both are built from the same condition as the plan, so a mismatch
            # is a wiring bug rather than a configuration. Say which, instead of
            # failing later on a shape.
            raise RuntimeError(
                "DSA-CP planned this forward but the layer is not set up for "
                f"it: attn_mqa_for_dsa_cp={m.attn_mqa_for_dsa_cp is not None}, "
                f"topk_indices={topk_indices is not None}"
            )
        if dsa_cp_plan is not None:
            # DSA-CP. Swap "this rank's heads for every token" for "every head
            # for this rank's tokens". The group holds the same (token, head)
            # pairs either way, each computed once, so the attention is
            # unchanged; what falls is the per-query top-k KV read, which is
            # what the operator is bound by.
            #
            # k_nope and k_pe stay full width: the KV write and the context
            # gather below address every token. Attention must also return the
            # padded width the batch carries, not plan.num_tokens -- SGLang
            # pads tokens to a multiple of attn_tp_size, and out_cache_loc is
            # sized by that.
            dsa_cp_rows = q_nope_out.shape[0]
            q_nope_out = dsa_cp_redistribute_heads(q_nope_out, dsa_cp_plan)
            q_pe = dsa_cp_redistribute_heads(q_pe, dsa_cp_plan)
            # The indexer ran at full width, so take this rank's rows of its
            # top-k. Padded rows get index 0, whose output is discarded.
            #
            # A separate name is load-bearing: this function returns
            # topk_indices for the next layer to reuse (only 21 of 78 layers
            # run the indexer), so rebinding it here would hand that layer a
            # slice, and the layer after would slice the slice.
            attn_topk_indices = dsa_cp_slice(topk_indices, dsa_cp_plan)
            attn_mqa = m.attn_mqa_for_dsa_cp
        else:
            attn_topk_indices = topk_indices
        attn_output = attn_mqa(
            q_nope_out.contiguous(),
            k_nope.contiguous(),
            k_nope.contiguous(),
            forward_batch,
            save_kv_cache=not mla_preprocess_used,
            q_rope=q_pe.contiguous(),
            k_rope=k_pe.contiguous(),
            topk_indices=attn_topk_indices,
        )
        if dsa_cp_plan is not None:
            # Undo the swap before anything else sees it: w_vc, o_proj and the
            # layer communicator all expect this rank's own heads for the whole
            # batch.
            attn_output = dsa_cp_restore_tokens(
                attn_output.reshape(dsa_cp_plan.rows, -1, m.kv_lora_rank),
                dsa_cp_plan,
                dsa_cp_rows,
            )
    if dcp_extend:
        # Drop the batch's reference before the MoE so a later forward on a
        # different path cannot read a stale gather. The buffers themselves are
        # reserved and survive.
        forward_batch.npu_dcp_extend_kv = None
    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    if _is_npu_arch35 or (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_draft_extend_v2()
        and not forward_batch.forward_mode.is_target_verify()
    ):
        attn_bmm_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            m.w_vc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
    else:
        attn_bmm_output = torch.empty(
            (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        attn_output = attn_output.contiguous()
        torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)

    if gate is not None:
        attn_bmm_output = m._apply_gated(attn_bmm_output, gate)
    output, _ = m.o_proj(attn_bmm_output)
    if not m.next_skip_topk:
        return output, None
    else:
        return output, topk_indices


def npu_mla_preprocess(
    m: "DeepseekV2AttentionMLA",
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
):
    dynamic_scale = None
    if not hasattr(m, "mla_preprocess"):
        m.mla_preprocess = NPUFusedMLAPreprocess(
            m.fused_qkv_a_proj_with_mqa,
            m.q_a_layernorm,
            m.kv_a_layernorm,
            m.q_b_proj,
            m.w_kc,
            m.rotary_emb,
            m.layer_id,
            m.num_local_heads,
            m.qk_nope_head_dim,
            m.qk_rope_head_dim,
            m.v_head_dim,
            m.quant_config,
        )
        if (
            get_disagg().disaggregation_mode == "decode"
            and m.mla_preprocess.uses_mlaprolog()
            and m.w_kc is not None
        ):
            m.w_kc.untyped_storage().resize_(0)
    # mlaprolog does not require additional calculation of q_lora
    if m.mla_preprocess.uses_mlaprolog():
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            q_lora,
            forward_batch,
            positions,
            dynamic_scale,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
    else:
        if m.alt_stream is not None:
            mla_event = torch.npu.Event()
            mla_event.record()
            with torch.npu.stream(m.alt_stream):
                # alt stream waits for the completion of the event on the main stream to ensure data dependency is complete
                torch.npu.current_stream().wait_event(mla_event)
                (
                    q_pe,
                    k_pe,
                    q_nope_out,
                    k_nope,
                    forward_batch,
                    zero_allocator,
                    positions,
                ) = m.mla_preprocess.forward(
                    positions, hidden_states, forward_batch, zero_allocator
                )

            fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, _ = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            q_lora = m.q_a_layernorm(q)
            torch.npu.current_stream().wait_event(m.alt_stream)
        else:
            (
                q_pe,
                k_pe,
                q_nope_out,
                k_nope,
                forward_batch,
                zero_allocator,
                positions,
            ) = m.mla_preprocess.forward(
                positions, hidden_states, forward_batch, zero_allocator
            )
            fused_qkv_a_proj_out = m.fused_qkv_a_proj_with_mqa(hidden_states)[0]
            q, _ = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            q_lora = m.q_a_layernorm(q)

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        q_lora,
        forward_batch,
        zero_allocator,
        positions,
        dynamic_scale,
    )


# endregion
