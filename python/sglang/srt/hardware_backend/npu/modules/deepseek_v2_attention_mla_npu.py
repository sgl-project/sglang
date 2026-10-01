import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

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
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
from sglang.srt.layers.attention.dsa.dsa_token_shard import (
    dsa_token_shard_narrow_a2a_enabled,
    dsa_token_shard_redistribute_heads,
    dsa_token_shard_restore_tokens,
    dsa_token_shard_slice,
    get_dsa_token_shard_plan,
)
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.dcp import (
    all_gather_q_for_mla_decode,
    cp_lse_ag_out_rs_mla_npu,
)
from sglang.srt.layers.dcp.layout import (
    dcp_extend_gather_buffer,
    dcp_interleave_size,
    plan_dcp_extend_gather,
    remap_dcp_sparse_indices,
)
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_dcp_mla_decode_phase,
)
from sglang.srt.runtime_context import get_disagg, get_parallel
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator

logger = logging.getLogger(__name__)

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
_is_npu_arch35 = is_npu_arch35()
_debug_dcp_extend_memory = envs.SGLANG_DEBUG_NPU_DCP_EXTEND_MEMORY.get()
_prefetch_dcp_extend_gather = envs.SGLANG_NPU_ENABLE_DCP_EXTEND_GATHER_PREFETCH.get()
_dcp_gather_prefetch_stream = None


def _get_dcp_gather_prefetch_stream():
    """One side stream per process, built lazily: the module is imported
    before the device is selected, and a stream made on the wrong device
    would run the gathers where no rank is looking."""
    global _dcp_gather_prefetch_stream
    if _dcp_gather_prefetch_stream is None:
        _dcp_gather_prefetch_stream = torch.npu.Stream()
    return _dcp_gather_prefetch_stream


# region MHA
def forward_mha_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
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
            if _use_ag_after_qlora and input_on_attn_tp_slices:
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
    input_on_attn_tp_slices: bool,
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
            if _use_ag_after_qlora and input_on_attn_tp_slices:
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


def _dsa_token_shard_narrow_plan(m: "DeepseekV2AttentionMLA", forward_batch):
    """The token-shard plan when the narrow exchange applies, else None. Absent
    full weights (the loader declined) put the layer back on the wide path."""
    if not dsa_token_shard_narrow_a2a_enabled():
        return None
    if getattr(m, "w_kc_full", None) is None or getattr(m, "w_vc_full", None) is None:
        return None
    return get_dsa_token_shard_plan(forward_batch)


def forward_dsa_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
    prev_topk_indices: torch.Tensor = None,
):
    dynamic_scale = None
    # Resolved here so the core can read the cached plan back.
    get_dsa_token_shard_plan(
        forward_batch,
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
            if _use_ag_after_qlora and input_on_attn_tp_slices:
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

        # Narrow exchange: the absorb is DEFERRED to after the all-to-all below,
        # so q_nope crosses the wire at 192 wide rather than 512.
        narrow_plan = _dsa_token_shard_narrow_plan(m, forward_batch)
        if narrow_plan is None:
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

        if narrow_plan is not None:
            # Record the width to restore to BEFORE the swap: the original row
            # count, possibly padded past num_tokens, is not recoverable after.
            forward_batch.npu_dsa_token_shard_input_rows = q_nope.shape[0]
            q_swapped = dsa_token_shard_redistribute_heads(
                torch.cat([q_nope, q_pe], dim=-1), narrow_plan
            )
            q_nope, q_pe = q_swapped.split(
                [m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1
            )
            q_nope_out = torch_npu.npu_transpose_batchmatmul(
                q_nope.contiguous(),
                m.w_kc_full,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
            q_pe = q_pe.contiguous()

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
            input_on_attn_tp_slices,
            dynamic_scale,
        )
        # Remap only a fresh top-k; layers that skip the indexer reuse the
        # remapped one. Decode only: extend reads the GATHERED context, whose
        # coordinates are global.
        if is_dcp_mla_decode_phase(forward_batch):
            parallel = get_parallel()
            topk_indices = remap_dcp_sparse_indices(
                topk_indices,
                parallel.attn_dcp_size,
                parallel.attn_dcp_rank,
                interleave_size=dcp_interleave_size(),
            )
    else:
        topk_indices = prev_topk_indices

    topk_indices = maybe_capture_indexer_topk(m.layer_id, topk_indices)

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


# Rows per extend-gather collective; caps the scratch, not the bytes moved.
_dcp_extend_gather_piece_rows = envs.SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS.get()
if _dcp_extend_gather_piece_rows <= 0:
    _dcp_extend_gather_piece_rows = 1 << 62

_last_dcp_extend_rows: Optional[Tuple[int, int]] = None


def _log_dcp_extend_memory(prefix_rows: int, extend_rows: int) -> None:
    """Log the previous DCP extend's peak memory, then reset. The peak is what a
    chunk size has to fit under; npu-smi shows only the reservation."""
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


def _log_dcp_shared_prefix(forward_batch, row_bytes: int) -> None:
    """Report how much of this gather is one prefix fetched once per request.

    ``walked`` coming from the radix match must equal the scheduler's own
    prefix lengths; that equality is the whole question, because a deduplicated
    gather would read sharing off the nodes rather than compare indices.
    """
    shared = getattr(forward_batch, "npu_dcp_shared_prefix", None)
    if shared is None:
        return
    prefix_lens = list(forward_batch.extend_prefix_lens_cpu)
    sent = sum(prefix_lens)
    gib = 1 << 30
    logger.info(
        "DCP shared prefix: bs=%d prefix=%s walked=%s walked_ok=%s "
        "sent_rows=%d union_rows=%d saved=%.1f%% (%.2f GiB/layer)",
        len(prefix_lens),
        prefix_lens,
        shared.walked,
        shared.walked == prefix_lens,
        sent,
        shared.union_rows,
        100.0 * (sent - shared.union_rows) / sent if sent else 0.0,
        (sent - shared.union_rows) * row_bytes / gib,
    )


def _pad_dcp_extend_send(shards: torch.Tensor, plan) -> torch.Tensor:
    """Lay this rank's per-request shards out at their padded send offsets."""
    send = shards.new_empty((plan.send_rows, *shards.shape[1:]))
    src = dst = 0
    for local_len, padded_len in zip(plan.local_lens, plan.padded_lens):
        send[dst : dst + local_len] = shards[src : src + local_len]
        src += local_len
        dst += padded_len
    return send


class _DcpGatherPrefetch:
    """Per-forward state for running each layer's prefix gather a layer ahead.

    Two scratch slots by layer parity. Both waits are unconditional, because
    getting either wrong corrupts silently: the side stream waits the main
    stream's ``release`` before refilling a slot (the index_select two layers
    back must be done), and the main stream waits the side stream's ``ready``
    before reading one.

    The __init__ syncs are once per forward, never per layer: ``wait_stream``
    depends on everything already queued, which at layer L includes layer L's
    compute and would serialise exactly what this overlaps.
    """

    __slots__ = ("stream", "pending", "release")

    def __init__(self, stream: "torch.npu.Stream"):
        self.stream = stream
        # layer_id -> (slot, ready_event, sends). ``sends`` is held only so the
        # caching allocator cannot reissue those buffers mid-collective.
        self.pending: Dict[int, tuple] = {}
        self.release: Dict[int, torch.npu.Event] = {}
        self.stream.wait_stream(torch.npu.current_stream())
        torch.npu.current_stream().wait_stream(self.stream)

    def slot_of(self, layer_id: int, start_layer: int) -> int:
        return (layer_id - start_layer) & 1


_logged_dcp_gather_prefetch = False


def _log_dcp_gather_prefetch_once(prefetching: bool, plan) -> None:
    """Say once whether the prefetch engaged; the fallback is otherwise silent."""
    global _logged_dcp_gather_prefetch
    if _logged_dcp_gather_prefetch:
        return
    _logged_dcp_gather_prefetch = True
    if prefetching:
        logger.info(
            "DCP extend gather prefetch is ACTIVE: %d scratch rows per slot, "
            "2 keys, 2 slots",
            plan.scratch_rows,
        )
    elif _prefetch_dcp_extend_gather:
        logger.info(
            "DCP extend gather prefetch was REQUESTED but is OFF: the plan has "
            "%d pieces and it needs exactly 1. Set "
            "SGLANG_NPU_DCP_EXTEND_GATHER_PIECE_ROWS=0.",
            len(plan.pieces),
        )
    else:
        logger.info("DCP extend gather prefetch is off (inline gathers)")


def _dcp_extend_gather_scratches(slot: int, k_nope, k_pe, rows: int):
    """``[(scratch, own), ...]`` for both keys. Slot 1 exists only under the
    prefetch, so slot 0 keeps the historical buffer names."""
    suffix = "" if slot == 0 else "_b"
    return [
        (dcp_extend_gather_buffer("latent_scratch" + suffix, k_nope, rows), k_nope),
        (dcp_extend_gather_buffer("rope_scratch" + suffix, k_pe, rows), k_pe),
    ]


def _dcp_extend_send_rows(pool, layer, md, plan, layer_id: int):
    """This rank's padded prefix shard for ``layer_id`` -- the gather's input.

    Prefix rows only, so it reads nothing the current forward wrote. That is
    what lets the prefetch call it a layer early.
    """
    if not plan.send_rows:
        return [None, None]
    send_nope, send_rope = pool.get_mla_kv_buffer(
        layer, md.dcp_local_prefix_kv_indices, layer_id=layer_id
    )
    if plan.local_lens != plan.padded_lens:
        # Served prefixes are cycle-aligned and need no padding; this is the
        # general case.
        send_nope = _pad_dcp_extend_send(send_nope, plan)
        send_rope = _pad_dcp_extend_send(send_rope, plan)
    return [send_nope, send_rope]


def _dcp_extend_all_gather(parallel, piece, pairs) -> None:
    """The collectives for one piece; ``pairs`` is [(scratch, send), ...]."""
    gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
    if not gathered:
        return
    for scratch, send in pairs:
        parallel.dcp_group.all_gather_into_tensor(
            scratch[:gathered], send[piece.send_start : piece.send_end]
        )


def _dcp_gather_extend_kv_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise each request's FULL prefix+extend KV, for one layer.

    Extend gathers the context rather than the query: above a tail of
    prefix/178 it moves fewer bytes (p20_dcp_extend_cost_model.py). The result
    is one contiguous run per request, which the operator needs under the
    non-paged layout -- so not ``all_gather_kv_cache_for_mla_extend``, which
    groups all prefixes then all extends.

    Output and scratch are reused buffers: a moving context-sized allocation
    makes ranks free and refill out of step and stalls the collectives for
    minutes. The keys travel separately because the operator takes them apart.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    plan = getattr(forward_batch, "npu_dcp_extend_gather", None)
    if plan is None:
        plan = plan_dcp_extend_gather(
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            parallel.dcp_size,
            parallel.dcp_rank,
            _dcp_extend_gather_piece_rows,
            dcp_interleave_size(),
        )
        plan = plan._replace(
            pieces=[
                piece._replace(index=piece.index.to(k_nope.device))
                for piece in plan.pieces
            ]
        )
        forward_batch.npu_dcp_extend_gather = plan
        # Wrapper pools (SWA, hybrid) may not offer this.
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(
                sum(forward_batch.extend_prefix_lens_cpu),
                sum(forward_batch.extend_seq_lens_cpu),
            )
        _log_dcp_shared_prefix(
            forward_batch, (k_nope.shape[-1] + k_pe.shape[-1]) * k_nope.element_size()
        )

    pool = get_token_to_kv_pool()
    total_rows = plan.pieces[-1].out_end if plan.pieces else 0
    out_nope = dcp_extend_gather_buffer("latent", k_nope, total_rows)
    out_rope = dcp_extend_gather_buffer("rope", k_pe, total_rows)

    # Prefetching needs every piece resident at once: one piece only.
    layer_id = m.attn_mqa.layer_id
    prefetching = _prefetch_dcp_extend_gather and len(plan.pieces) == 1
    _log_dcp_gather_prefetch_once(prefetching, plan)
    slot = 0
    state = None
    if prefetching:
        state = getattr(forward_batch, "npu_dcp_gather_prefetch", None)
        if state is None:
            state = _DcpGatherPrefetch(_get_dcp_gather_prefetch_stream())
            forward_batch.npu_dcp_gather_prefetch = state
        slot = state.slot_of(layer_id, pool.start_layer)

    scratches = _dcp_extend_gather_scratches(slot, k_nope, k_pe, plan.scratch_rows)

    # Every rank runs (or skips) the same collectives in the same order. Gather
    # and index_select stay INTERLEAVED per piece: pieces share one scratch, so
    # hoisting the gathers would let piece n+1 overwrite piece n unread.
    done = state.pending.pop(layer_id, None) if prefetching else None
    sends = None
    if done is not None:
        torch.npu.current_stream().wait_event(done[1])
    else:
        sends = _dcp_extend_send_rows(pool, m.attn_mqa, md, plan, layer_id)

    for piece in plan.pieces:
        gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
        rows = gathered + piece.extend_end - piece.extend_start
        if sends is not None:
            _dcp_extend_all_gather(
                parallel, piece, [(buf, s) for (buf, _), s in zip(scratches, sends)]
            )
        for out, (buf, own) in zip((out_nope, out_rope), scratches):
            scratch = buf[:rows]
            scratch[gathered:] = own[piece.extend_start : piece.extend_end]
            torch.index_select(
                scratch, 0, piece.index, out=out[piece.out_start : piece.out_end]
            )

    if prefetching:
        # The slot is free only after the index_select above.
        release = torch.npu.Event()
        release.record()
        state.release[slot] = release
        _issue_dcp_gather_prefetch(
            state, pool, m, md, plan, parallel, layer_id, k_nope, k_pe
        )
    return out_nope, out_rope


def _issue_dcp_gather_prefetch(
    state: "_DcpGatherPrefetch",
    pool,
    m: "DeepseekV2AttentionMLA",
    md,
    plan,
    parallel,
    layer_id: int,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
) -> None:
    """Start the next layer's prefix all-gathers on the side stream."""
    nxt = layer_id + 1
    if nxt > pool.start_layer + pool.layer_num - 1 or nxt in state.pending:
        return
    slot = state.slot_of(nxt, pool.start_layer)
    scratches = _dcp_extend_gather_scratches(slot, k_nope, k_pe, plan.scratch_rows)
    release = state.release.get(slot)
    with torch.npu.stream(state.stream):
        if release is not None:
            state.stream.wait_event(release)
        sends = _dcp_extend_send_rows(pool, m.attn_mqa, md, plan, nxt)
        pairs = [(scratch, s) for (scratch, _), s in zip(scratches, sends)]
        for piece in plan.pieces:
            _dcp_extend_all_gather(parallel, piece, pairs)
        for send in sends:
            if send is not None:
                send.record_stream(state.stream)
        ready = state.stream.record_event()
    state.pending[nxt] = (slot, ready, sends)


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
    # Only the narrow token-shard path sets this; DCP never does.
    w_vc_applied = False
    # GLM-5.2 dispatches DSA_NPU here, not to forward_absorb_core, so
    # forward_mla.py's DCP block never runs for it.
    dcp_extend = (
        get_parallel().dcp_enabled
        and forward_batch.forward_mode.is_extend()
        and not is_dcp_mla_decode_phase(forward_batch)
        and forward_batch.attn_dcp_metadata is not None
    )
    if (
        not dcp_extend
        and get_parallel().dcp_enabled
        and forward_batch.forward_mode.is_extend()
        and not is_dcp_mla_decode_phase(forward_batch)
        and not get_attn_backend().is_draft_worker
    ):
        # Without the metadata the backend reads this rank's shard through a
        # full-span page table and returns plausible output. The merge of
        # #37787 did exactly that, silently, by returning None for all NPU.
        raise RuntimeError(
            "DCP extend on the NPU DSA path has no attn_dcp_metadata; "
            "prepare_context_parallel_metadata_for_dcp must build it for DSA "
            "extend on NPU"
        )
    if dcp_extend:
        forward_batch.npu_dcp_extend_kv = _dcp_gather_extend_kv_npu(
            m, forward_batch, k_nope, k_pe
        )

    if is_dcp_mla_decode_phase(forward_batch):
        q_nope_out, q_pe = all_gather_q_for_mla_decode(q_nope_out=q_nope_out, q_pe=q_pe)
        # save_kv_cache stays True here: the DCP write is idempotent.
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
        attn_output = attn_output.view(
            -1, m.num_local_heads * get_parallel().attn_dcp_size, m.kv_lora_rank
        )
        # Returns the local head slice already, in natural log on both sides.
        attn_output = cp_lse_ag_out_rs_mla_npu(
            attn_output, lse, get_parallel().dcp_group
        )
    else:
        attn_mqa = m.attn_mqa
        dsa_token_shard_plan = get_dsa_token_shard_plan(forward_batch)
        if dsa_token_shard_plan is not None and (
            topk_indices is None or m.attn_mqa_for_dsa_token_shard is None
        ):
            raise RuntimeError(
                "DSA token-shard planned this forward but the layer is not set up for "
                f"it: attn_mqa_for_dsa_token_shard={m.attn_mqa_for_dsa_token_shard is not None}, "
                f"topk_indices={topk_indices is not None}"
            )
        narrow_a2a = (
            dsa_token_shard_plan is not None
            and _dsa_token_shard_narrow_plan(m, forward_batch) is not None
        )
        if dsa_token_shard_plan is not None:
            if narrow_a2a:
                # prepare already swapped, before the absorb.
                dsa_token_shard_rows = forward_batch.npu_dsa_token_shard_input_rows
            else:
                dsa_token_shard_rows = q_nope_out.shape[0]
                q_nope_out = dsa_token_shard_redistribute_heads(
                    q_nope_out, dsa_token_shard_plan
                )
                q_pe = dsa_token_shard_redistribute_heads(q_pe, dsa_token_shard_plan)
            # A separate name is load-bearing: topk_indices is returned for the
            # next layer, and rebinding it would hand that a slice of a slice.
            if getattr(forward_batch, "npu_indexer_topk_is_local", False):
                # Checked, not assumed: a full-width tensor here gives every rank
                # but 0 the wrong rows with no shape error.
                if topk_indices.shape[0] != dsa_token_shard_plan.rows:
                    raise RuntimeError(
                        "top-k is marked local to this rank but carries "
                        f"{topk_indices.shape[0]} rows, not the plan's "
                        f"{dsa_token_shard_plan.rows}"
                    )
                attn_topk_indices = topk_indices
            else:
                attn_topk_indices = dsa_token_shard_slice(
                    topk_indices, dsa_token_shard_plan
                )
            attn_mqa = m.attn_mqa_for_dsa_token_shard
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
        if dsa_token_shard_plan is not None:
            attn_output = attn_output.reshape(
                dsa_token_shard_plan.rows, -1, m.kv_lora_rank
            )
            if narrow_a2a:
                # w_vc_full: this rank holds every head right now.
                attn_output = torch_npu.npu_transpose_batchmatmul(
                    attn_output.contiguous(),
                    m.w_vc_full,
                    perm_x1=(1, 0, 2),
                    perm_x2=(0, 1, 2),
                    perm_y=(1, 0, 2),
                )
                w_vc_applied = True
            attn_output = dsa_token_shard_restore_tokens(
                attn_output,
                dsa_token_shard_plan,
                dsa_token_shard_rows,
            )
    if dcp_extend:
        # So a later forward cannot read a stale gather.
        forward_batch.npu_dcp_extend_kv = None
    if w_vc_applied:
        attn_bmm_output = attn_output
    else:
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
