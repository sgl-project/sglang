import logging
from dataclasses import fields
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch_npu
from sgl_kernel_npu.operators.matmul import batch_matmul_transpose
from sgl_kernel_npu.operators.norm import fused_split_qk_norm

import sglang.srt.layers.dcp.comm as dcp_comm
import sglang.srt.layers.dcp.layout as dcp_layout
import sglang.srt.model_executor.forward_context as forward_context
import sglang.srt.runtime_context as runtime_context
from sglang.srt.configs.model_config import get_dsa_index_topk
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.dcp import (
    forward_mla_attention_npu,
    get_mla_dcp_backend,
)
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.hardware_backend.npu.dcp_extend_gather import (
    NPUDecodeContextParallelMetadata,
    dcp_extend_crop_free,
    dcp_extend_gather_buffer,
    dcp_extend_gathers_kv,
)
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
from sglang.srt.layers.attention.dsa.dsa_token_shard import (
    dsa_token_shard_redistribute_heads,
    dsa_token_shard_restore_tokens,
    dsa_token_shard_slice,
    get_dsa_token_shard_plan,
)
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.dcp.layout import (
    DcpPackedReadPlan,
    dcp_packed_read_plan,
    remap_topk_to_packed,
)
from sglang.srt.layers.dcp.planner import prepare_decode_context_parallel_metadata
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_disagg, get_parallel
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk
from sglang.srt.utils import print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator

logger = logging.getLogger(__name__)

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
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


def _dsa_dcp_extend_gathers_kv(forward_batch: "ForwardBatch") -> bool:
    """This forward gathers the prefix KV rather than the query.

    ``build_dcp_extend_gather_plan`` decides once per forward; this only reads
    its answer.
    """
    dcp_metadata = forward_batch.attn_dcp_metadata
    return (
        isinstance(dcp_metadata, NPUDecodeContextParallelMetadata)
        and dcp_metadata.extend_gather_plan is not None
        # Per model, not per batch: never the draft model's layers.
        and not forward_context.get_attn_backend().is_draft_worker
    )


def init_modelslim_mla_kv(m: "DeepseekV2AttentionMLA", prefix: str):
    quant_config = m.quant_config
    if quant_config is None or quant_config.get_name() != "modelslim" or m.use_dsa:
        return
    scheme = quant_config.init_kv_cache(
        m.attn_mqa,
        prefix,
        kv_lora_rank=m.kv_lora_rank,
        qk_rope_head_dim=m.qk_rope_head_dim,
    )
    if scheme is not None:
        m.attn_mha.scheme = scheme


# region MHA
def forward_mha_prepare_npu(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    input_on_attn_tp_slices: bool,
):
    kv_quant_scheme = getattr(m.attn_mqa, "scheme", None)
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
            c_kv_scale=(
                kv_quant_scheme.k_inv_scale.view(1)
                if kv_quant_scheme is not None
                else None
            ),
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
            m.attn_mqa if kv_quant_scheme is not None else m,
            forward_batch.out_cache_loc,
            kv_a.unsqueeze(1),
            k_pe,
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
    kv_quant_scheme = getattr(m.attn_mqa, "scheme", None)
    q_descale = None
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
                m.v_head_dim,
                kv_quant_scheme=kv_quant_scheme,
            )
        outputs, q_descale = m.mla_preprocess.forward(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            return_q_descale=True,
        )
        q_pe, k_pe, q_nope_out, k_nope = outputs[:4]
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

        if kv_quant_scheme is not None:
            q_nope_out, q_descale = kv_quant_scheme.quantize_query(q_nope_out)
            q_pe = (q_pe.float() / (q_descale * kv_quant_scheme.k_scale)).to(
                torch.bfloat16
            )

    backend = get_mla_dcp_backend(forward_context.get_attn_backend())
    if backend.uses_dcp_partial_attention(forward_batch):
        q_nope_out, q_pe = dcp_comm.all_gather_q_for_mla_decode(q_nope_out, q_pe)

    return (
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        q_descale,
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
    q_descale: Optional[torch.Tensor],
    # Gated attention (Ling-V3 / BailingMoeV3): the subclass appends its gate
    # to inner_state, so every *_core dispatched from forward_core takes it as
    # a trailing arg. None everywhere else.
    gate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_kwargs = dict(topk_indices=topk_indices) if topk_indices is not None else {}
    if getattr(m.attn_mqa, "scheme", None) is not None:
        attn_kwargs["q_descale"] = q_descale
    backend = get_mla_dcp_backend(forward_context.get_attn_backend())
    attn_output = forward_mla_attention_npu(
        m,
        backend,
        forward_batch,
        q_nope_out,
        k_nope,
        q_pe,
        k_pe,
        **attn_kwargs,
    )

    attn_bmm_output = batch_matmul_transpose(attn_output.contiguous(), m.w_vc)

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
    input_on_attn_tp_slices: bool,
    prev_topk_indices: torch.Tensor = None,
):
    dynamic_scale = None
    # Resolved here so the core can read the cached plan back.
    get_dsa_token_shard_plan(forward_batch)
    # D2p: prime the first DSA layer's prefix gather on the side stream, before
    # the QKV, so it runs under the QKV and the indexer instead of starting with
    # them. prime() is a no-op past the forward's first DSA layer, so this runs
    # on every prepare and only the first layer actually gathers. It needs the
    # packed-read plan, so it is a no-op (the permuting path takes over) for
    # multi-request / sub-index_topk extends.
    if _enable_dcp_gather_prefetch and _dcp_extend_active(m, forward_batch):
        packed_plan = dcp_packed_read_plan(
            forward_batch, _dcp_sfa_gather_topk_width(m, prev_topk_indices)
        )
        if packed_plan is not None:
            _dcp_gather_prefetcher_get().prime(
                m,
                forward_batch.attn_dcp_metadata,
                packed_plan,
                hidden_states,
                m.layer_id,
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
        # NoPE layers (skip_rope=True, rotary_emb=None; Kimi K2/K3, GLM-5.3
        # Flash) take the unfused split path, like the MLA prepare does.
        if m.rotary_emb is None or m.rotary_emb.is_neox_style:
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

        q_nope_out = batch_matmul_transpose(q_nope, m.w_kc, impl="torch_npu")

        # NoPE layers (skip_rope=True, rotary_emb=None) skip rope entirely,
        # mirroring the None guard in forward_mla_prepare_npu.
        if (
            is_mla_preprocess_enabled()
            and m.rotary_emb is not None
            and not m.rotary_emb.is_neox_style
        ):
            # Match the half-layout RoPE outputs used by MLA preprocessing.
            q_pe, k_pe = _apply_interleaved_rope_with_half_output(
                m.rotary_emb, positions, q_pe, k_pe
            )
        else:
            if m.rotary_emb is not None:
                if m.layer_id == get_token_to_kv_pool().start_layer:
                    m.rotary_emb.sin_cos_cache = (
                        m.rotary_emb.cos_sin_cache.index_select(0, positions)
                    )
                q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if dsa_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

    backend = get_mla_dcp_backend(forward_context.get_attn_backend())
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
        # DSA layers that skip the indexer reuse ``prev_topk_indices``. Remap
        # only when a fresh global top-k is produced so shared-index layers do
        # not repeat the same DCP partitioning work.
        if backend.uses_dcp_partial_attention(forward_batch):
            parallel = runtime_context.get_parallel()
            topk_indices = dcp_layout.remap_dcp_sparse_indices(
                topk_indices,
                parallel.attn_dcp_size,
                parallel.attn_dcp_rank,
                interleave_size=forward_context.get_attn_backend().page_size,
            )
        else:
            # C1: extend packed read -- remap the fresh top-k to the rank-major
            # packed rows. Guard on DCP-active (the same predicate the prime
            # site and the gather use): without DCP (dcp_size==1) the plan would
            # still build, log "packed read ON", and identity-remap -- harmless
            # but misleading, and a wasted plan-build on every non-decode
            # forward. dcp_packed_read_plan is None for the refusal cases
            # (multi-request, no top-k, causal crop needed, FP8 KV), so this is
            # a no-op except a one-request crop-free DCP extend with the flag on.
            # Remapped HERE (where the indexer produces it) so it runs once per
            # indexer-layer, not at the operator on all 78 layers.
            if _dcp_extend_active(m, forward_batch):
                packed_plan = dcp_packed_read_plan(
                    forward_batch, topk_indices.shape[-1]
                )
                if packed_plan is not None:
                    topk_indices = remap_topk_to_packed(topk_indices, packed_plan)
    else:
        topk_indices = prev_topk_indices

    if backend.uses_dcp_partial_attention(forward_batch):
        q_nope_out, q_pe = dcp_comm.all_gather_q_for_mla_decode(q_nope_out, q_pe)

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


def prepare_dcp_extend_metadata_npu(
    m, kv_cache_dtype, seq_lens_sum, extend_prefix_lens_cpu, **planner_kwargs
) -> Optional[NPUDecodeContextParallelMetadata]:
    """DCP metadata for a DSA extend that gathers the prefix KV, else None.

    NPU decode and short-tail extends gather the query and read rank-local
    metadata off the attention backend instead.
    """
    # The gathered-KV operator path is bf16/fp16 only; any other KV dtype keeps
    # the query gather rather than raising there.
    if not m.use_dsa or kv_cache_dtype not in (torch.bfloat16, torch.float16):
        return None
    parallel = runtime_context.get_parallel()
    prefix_sum = sum(int(p) for p in extend_prefix_lens_cpu)
    if not dcp_extend_gathers_kv(
        [prefix_sum],
        [seq_lens_sum - prefix_sum],
        m.config.num_attention_heads * parallel.attn_dcp_size // parallel.attn_tp_size,
        m.config.kv_lora_rank,
        m.config.qk_rope_head_dim,
        kv_cache_dtype.itemsize,
    ):
        return None
    metadata = prepare_decode_context_parallel_metadata(
        kv_cache_dtype=kv_cache_dtype,
        seq_lens_sum=seq_lens_sum,
        extend_prefix_lens_cpu=extend_prefix_lens_cpu,
        **planner_kwargs,
        # NPU pools shard in page_size runs, and the gather reuses its own
        # buffers rather than a context-sized dcp_kv_buffer per forward.
        interleave_size=forward_context.get_attn_backend().page_size,
        allocate_kv_buffer=False,
    )
    if metadata is None:
        return None
    # The gather plan needs extend_seq_lens_cpu, so the backend fills it in.
    return NPUDecodeContextParallelMetadata(
        **{f.name: getattr(metadata, f.name) for f in fields(metadata)},
        extend_crop_free=dcp_extend_crop_free(
            extend_prefix_lens_cpu,
            get_dsa_index_topk(m.config),
        ),
    )


def _pad_dcp_extend_send(shards: torch.Tensor, plan) -> torch.Tensor:
    """Lay this rank's per-request shards out at their padded send offsets."""
    # Zeroed: no rank reads the pad rows, but they still cross the collective.
    send = shards.new_zeros((plan.send_rows, *shards.shape[1:]))
    src = dst = 0
    for local_len, padded_len in zip(plan.local_lens, plan.padded_lens):
        send[dst : dst + local_len] = shards[src : src + local_len]
        src += local_len
        dst += padded_len
    return send


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


def _log_dcp_gather_prefetch_once(prefetching: bool, plan, packed_kv: bool) -> None:
    """Say once whether the prefetch engaged; the fallback is otherwise silent."""
    global _logged_dcp_gather_prefetch
    if _logged_dcp_gather_prefetch:
        return
    _logged_dcp_gather_prefetch = True
    if prefetching:
        logger.info(
            "DCP extend gather prefetch is ACTIVE: %d scratch rows per slot, "
            "%d keys, 2 slots",
            plan.scratch_rows,
            1 if packed_kv else 2,
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
    """``[(scratch, own), ...]`` for the latent and rope keys. Slot 1 exists
    only under the prefetch, so slot 0 keeps the historical names."""
    suffix = "" if slot == 0 else "_b"
    return [
        (
            dcp_extend_gather_buffer("latent_scratch" + suffix, k_nope, rows),
            k_nope,
        ),
        (
            dcp_extend_gather_buffer("rope_scratch" + suffix, k_pe, rows),
            k_pe,
        ),
    ]


def _dcp_extend_send_rows(pool, layer, md, plan, layer_id: int):
    """This rank's padded prefix shard for ``layer_id`` -- the gather's input.

    Prefix rows only, so it reads nothing the current forward wrote. That is
    what lets the prefetch call it a layer early. One send per gathered key,
    in ``_dcp_extend_gather_scratches`` order.
    """
    if not plan.send_rows:
        return [None, None]
    send_nope, send_rope = pool.get_mla_kv_buffer(
        layer, md.dcp_local_prefix_kv_indices, layer_id_override=layer_id
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


_enable_dcp_gather_prefetch = envs.SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH.get()


if _enable_dcp_gather_prefetch and not envs.SGLANG_NPU_ENABLE_DCP_PACKED_READ.get():
    # The prefetch fills the buffer the operator reads directly. On the
    # permuting path that buffer is produced by an index_select from a scratch,
    # which cannot be prefetched a layer ahead without also double-buffering the
    # scratch and splitting the plan's index. Refuse rather than be silently
    # inert -- the whole point of the flag is that it does something.
    raise ValueError(
        "SGLANG_NPU_ENABLE_DCP_PACKED_GATHER_PREFETCH requires "
        "SGLANG_NPU_ENABLE_DCP_PACKED_READ: there is nothing to prefetch into "
        "until the operator reads the gathered buffer directly."
    )


class _DcpGatherPrefetcher:
    """C3 (packed path): gather layer l+1's prefix on a side stream while layer l computes.

    The all-gather is ~0.52 s of a chunk at a ~990k prefix. It depends on
    nothing the current layer computes: the prefix KV was written by earlier
    forwards and sits in the pool, so it can run underneath the layer before it.

    Two slots, indexed ``layer_id % 2``. Every dcp-group gather goes on the side
    stream (including the first) so one HCCL communicator stays bound to one
    stream for the life of the process; the compute stream only waits on events.
    Only this chunk's own KV is written from the compute stream, because its
    k_nope/k_pe are computed during this layer.
    """

    def __init__(self) -> None:
        # D2v2: created with an explicit priority, the vLLM-CUDA "KV I/O yields
        # to compute" pattern (vLLM makes its offload transfer streams the
        # LOWEST-priority streams it can). The lowest torch_npu exposes is 0 --
        # positive values clamp to the same pool, -1 is the HIGH pool, and a
        # stream that must yield never takes it.
        self.stream = torch.npu.Stream(
            priority=envs.SGLANG_NPU_DCP_GATHER_STREAM_PRIORITY.get()
        )
        self.ready: list = [None, None]
        self.filled: list = [None, None]
        self.prev_layer: Optional[int] = None
        self._max_layer: Optional[int] = None
        # The layer whose prefix prime() issued in prepare, before any take().
        # take() must not let _note_layer reset that slot: the prime ran before
        # the first take, so the structural "layer id stopped advancing" test
        # cannot tell the primed first layer apart from a fresh forward.
        self.primed: Optional[int] = None

    def _note_layer(self, layer_id: int) -> None:
        """Drop both slots when a new forward starts.

        Detected structurally -- layers run 0, 1, 2, ... and then the next
        forward starts over, so a layer id that does not advance means a new
        batch -- rather than by the identity of the ForwardBatch.

        The one exception is a primed forward: prime() runs before the first
        take(), and fills the first layer's slot. The structural test cannot
        see that the first take() belongs to the primed forward (the layer id
        has not advanced past the prime's), so the marker is consumed here and
        the primed slot is kept.
        """
        if self.primed is not None:
            if layer_id == self.primed:
                # The take of the very layer prime() filled: keep its slot.
                self.primed = None
                self.prev_layer = layer_id
                return
            # A take that is not the primed layer: the prime is stale (its
            # forward never consumed it) or this is a newer forward. Drop the
            # marker and fall through to the normal reset below.
            self.primed = None
        if self.prev_layer is None or layer_id <= self.prev_layer:
            self.filled = [None, None]
        self.prev_layer = layer_id

    def slots(self, plan, k_nope, k_pe, slot: int):
        return (
            dcp_extend_gather_buffer(f"packed_latent_{slot}", k_nope, plan.rows),
            dcp_extend_gather_buffer(f"packed_rope_{slot}", k_pe, plan.rows),
        )

    def _gather_prefix(self, m, md, plan, ref_nope, ref_rope, layer_id: int) -> None:
        """Issue one layer's prefix all-gather on the side stream.

        ``ref_nope``/``ref_rope`` only type the slot buffers (dtype, device, row
        shape); they are the chunk's own KV on a take() and a (1, 1, dim) slice
        of hidden_states on a prime(), before the QKV has produced them.
        """
        slot = layer_id % 2
        pool = get_token_to_kv_pool()
        out_nope, out_rope = self.slots(plan, ref_nope, ref_rope, slot)
        # record_stream stops the allocator handing the compute-allocated slot
        # to something else while the side stream is still writing it.
        out_nope.record_stream(self.stream)
        out_rope.record_stream(self.stream)
        # Wait for everything already issued on compute, so the slot's previous
        # reader -- layer_id - 2's attention -- is done before it is overwritten.
        self.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.stream):
            if plan.gathered_rows:
                parallel = get_parallel()
                send_nope, send_rope = pool.get_mla_kv_buffer(
                    m.attn_mqa,
                    md.dcp_local_prefix_kv_indices,
                    layer_id_override=layer_id,
                )
                send_nope = _pad_dcp_packed_send(send_nope, plan, parallel.dcp_rank)
                send_rope = _pad_dcp_packed_send(send_rope, plan, parallel.dcp_rank)
                # The helper waits for backend work before this stream's
                # ready event can release the compute-stream consumer.
                _dcp_issue_gather_chunks(parallel.dcp_group, out_nope, send_nope, plan)
                _dcp_issue_gather_chunks(parallel.dcp_group, out_rope, send_rope, plan)
            self.ready[slot] = self.stream.record_event()
        self.filled[slot] = layer_id

    def prime(self, m, md, plan, hidden_states, layer_id: int) -> None:
        """Issue the first layer's prefix gather at the start of the forward.

        take() issues a layer's gather from the core, AFTER that layer's indexer
        has run; with no previous layer to prefetch from, layer 0's gather then
        starts in the indexer's window, idle. prime() runs before the QKV, so
        the first gather moves bytes under the QKV and the indexer. Only the
        first DSA layer of a forward primes; the rest no-op here and are
        prefetched by the previous layer's take() as usual. The slot it fills is
        the same one take() would fill, typed from hidden_states (the QKV output
        does not exist yet) rather than from the chunk's own KV, so the two share
        one grow-only buffer.
        """
        if self.prev_layer is not None and layer_id > self.prev_layer:
            # A layer earlier in THIS forward already ran, so this is not the
            # first layer: nothing to prime, take() will handle it.
            return
        # First DSA layer of a (possibly new) forward: drop the previous
        # forward's slots, then fill this layer's.
        self.filled = [None, None]
        self.prev_layer = layer_id
        ref_nope = hidden_states.new_empty((1, 1, m.kv_lora_rank))
        ref_rope = hidden_states.new_empty((1, 1, m.qk_rope_head_dim))
        self._gather_prefix(m, md, plan, ref_nope, ref_rope, layer_id)
        self.primed = layer_id
        print_info_once(
            "DCP gather first-layer prime is ON: layer "
            f"{layer_id}'s prefix all-gather is issued before the QKV, so it "
            "runs under the QKV and the indexer instead of starting with them"
        )

    def _max_servable_layer(self) -> int:
        """Highest layer id the KV pool can serve, from the list it indexes.

        NOT from ``pool.end_layer`` -- that field is inconsistent (the pool's
        constructor defaults it inclusive while model_runner passes it
        exclusive). ``get_key_buffer`` reads ``k_buffer[layer_id - start_layer]``
        (after start_layer is applied), so that buffer's length is the only
        statement of the bound that cannot disagree with the read it guards.
        ``-1`` disables prefetching entirely (every layer gathers inline).
        """
        if self._max_layer is None:
            pool = get_token_to_kv_pool()
            k = getattr(pool, "k_buffer", None)
            start = getattr(pool, "start_layer", 0)
            self._max_layer = -1 if k is None else start + len(k) - 1
        return self._max_layer

    def take(self, m, md, plan, k_nope, k_pe, layer_id: int):
        """This layer's KV buffer, with next layer's gather already in flight."""
        self._note_layer(layer_id)
        slot = layer_id % 2
        if self.filled[slot] != layer_id:
            # Layer 0 of a forward, or a layer whose prefetch was skipped. Issue
            # it now and pay for it, exactly as the unprefetched path would.
            self._gather_prefix(m, md, plan, k_nope, k_pe, layer_id)
        else:
            # Logged on a HIT, not on construction: a prefetcher built and then
            # missing on every layer does no overlapping at all.
            print_info_once(
                "DCP gather prefetch is ON: layer l+1's prefix all-gather runs "
                "on a side stream underneath layer l's compute"
            )
        out_nope, out_rope = self.slots(plan, k_nope, k_pe, slot)
        torch.npu.current_stream().wait_event(self.ready[slot])
        # The one part that could not be prefetched: this chunk's own KV.
        _write_packed_extend_tail(out_nope, out_rope, k_nope, k_pe, plan)
        if layer_id < self._max_servable_layer():
            self._gather_prefix(m, md, plan, k_nope, k_pe, layer_id + 1)
        return out_nope, out_rope


_dcp_gather_prefetcher: Optional["_DcpGatherPrefetcher"] = None


def _dcp_gather_prefetcher_get() -> "_DcpGatherPrefetcher":
    """The process-wide prefetcher, created on first use.

    Two call sites need it -- the core (to prime the first layer's gather and
    to take a layer's buffer) -- and either can run first, so the creation
    lives here rather than in one of them.
    """
    global _dcp_gather_prefetcher
    if _dcp_gather_prefetcher is None:
        _dcp_gather_prefetcher = _DcpGatherPrefetcher()
    return _dcp_gather_prefetcher


def _dcp_extend_active(
    m: "DeepseekV2AttentionMLA", forward_batch: "ForwardBatch"
) -> bool:
    """Whether this forward's DCP extend gather runs, per layer.

    prepare and core must agree on exactly this, so the expression lives in one
    place: prepare decides whether to issue the async gather, core whether to
    wait for it.
    """
    return (
        get_parallel().dcp_enabled
        and forward_batch.forward_mode.is_extend()
        and not m.is_dcp_mla_decode_phase(forward_batch)
        and forward_batch.attn_dcp_metadata is not None
    )


def _dcp_sfa_gather_topk_width(
    m: "DeepseekV2AttentionMLA", prev_topk_indices: Optional[torch.Tensor]
) -> Optional[int]:
    """This layer's eventual top-k width, known before the indexer runs.

    The gather planner takes the width the sparse operator will be handed
    (``topk_indices.shape[-1]`` in the core): the indexer's ``index_topk`` on
    the layers that run it, and the previous layer's on the 57 that reuse it.
    The path decision (permuting vs packed) has to be made in prepare, before
    the indexer has produced the tensor, so derive the same width from the same
    condition the indexer call is gated on.
    """
    if not m.skip_topk or (m.is_nextn and prev_topk_indices is None):
        return m.indexer.index_topk if m.indexer is not None else None
    return prev_topk_indices.shape[-1] if prev_topk_indices is not None else None


def _dcp_gather_extend_kv_packed_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    plan: "DcpPackedReadPlan",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """C1: leave the gathered prefix in the order the collective produced it.

    The permuting path writes the all-gather into a scratch and then
    ``index_select``s it into a position-ordered output. This does neither: the
    collective writes straight into the buffer the operator will read, and this
    chunk's own KV is appended after it. What pays for that is
    ``remap_topk_to_packed``, arithmetic on the ``[tokens, k]`` index tensor.

    **No pieces here, deliberately.** Pieces cap the scratch held beside the
    output, and there is no scratch now -- the buffer IS the output. Piecing
    would also break the remap: one collective, one layout, one formula.
    """
    parallel = get_parallel()
    md = forward_batch.attn_dcp_metadata
    if not md.packed_write_planned:
        # Once per forward, on the first layer: carry the same once-per-forward
        # write-plan duty the permuting path does in its plan-building branch.
        md.packed_write_planned = True
        md.dcp_kv_buffer = None
        plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
        if plan_write is not None:
            plan_write(forward_batch.out_cache_loc)
        if _debug_dcp_extend_memory:
            _log_dcp_extend_memory(plan.prefix_len, plan.extend_len)

    if _enable_dcp_gather_prefetch:
        # The prefetcher is shared with the prepare-side prime, so it is created
        # (on first use) in one place: _dcp_gather_prefetcher_get().
        return _dcp_gather_prefetcher_get().take(m, md, plan, k_nope, k_pe, m.layer_id)

    pool = get_token_to_kv_pool()
    out_nope = dcp_extend_gather_buffer("packed_latent", k_nope, plan.rows)
    out_rope = dcp_extend_gather_buffer("packed_rope", k_pe, plan.rows)

    if plan.gathered_rows:
        send_nope, send_rope = pool.get_mla_kv_buffer(
            m.attn_mqa, md.dcp_local_prefix_kv_indices
        )
        send_nope = _pad_dcp_packed_send(send_nope, plan, parallel.dcp_rank)
        send_rope = _pad_dcp_packed_send(send_rope, plan, parallel.dcp_rank)
        # Sync issue, on the compute stream: stream ordering alone makes the
        # tail write and the return below see the filled prefix, no handles.
        _dcp_issue_gather_chunks(parallel.dcp_group, out_nope, send_nope, plan)
        _dcp_issue_gather_chunks(parallel.dcp_group, out_rope, send_rope, plan)
    _write_packed_extend_tail(out_nope, out_rope, k_nope, k_pe, plan)
    return out_nope, out_rope


def _write_packed_extend_tail(
    out_nope: torch.Tensor,
    out_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    plan: "DcpPackedReadPlan",
) -> None:
    """This chunk's own KV, after the gathered prefix. Not gathered: identical
    on every rank.

    ``k_nope`` can be WIDER than the request's token count (the batch is padded
    to a multiple of attn_tp_size), so take only the first plan.extend_len rows.
    The operator is given true query lengths, so it never reads a padded query
    row, and a real query's top-k only names positions below prefix+extend.
    """
    n = plan.extend_len
    assert k_nope.shape[0] >= n, (
        f"DCP packed read: this chunk's KV has {k_nope.shape[0]} rows but the "
        f"plan expects at least {n}. The plan comes from extend_seq_lens_cpu; "
        "if the batch is now NARROWER than that, the two no longer describe "
        "the same forward."
    )
    out_nope[plan.gathered_rows :] = k_nope[:n]
    out_rope[plan.gathered_rows :] = k_pe[:n]


def _dcp_issue_gather_chunks(
    group, out: torch.Tensor, send: torch.Tensor, plan: "DcpPackedReadPlan"
):
    """Fill ``out``'s prefix region with the all-gather of ``send``.

    D2v2: ``plan.chunks > 1`` issues one sub-all-gather per chunk instead of
    one monolithic collective. Chunk k moves local rows [st, st + sz) of every
    rank -- the first chunks-1 hold q = send_rows // chunks rows, the last
    q + send_rows % chunks -- and each lands in the CONTIGUOUS slice
    ``out[C*st : C*(st + sz)]`` of the very buffer the packed read expects
    (chunk-major, see DcpPackedReadPlan), so no extra buffer and no copy. With
    ``plan.chunks == 1`` this is exactly the old single collective.

    The point is not the bytes (they are the same) but the AIV episode: the
    monolithic gather was one 9-16 ms AIV kernel (93% MTE2) that starved the
    compute stream's own AIV work for its whole flight; N sub-collectives of
    roughly 1-2 ms leave the inter-collective windows in which that work runs.
    See SGLANG_NPU_DCP_GATHER_CHUNKS for the measurement.

    Order every chunk's completion before the caller consumes the prefix or
    records its ready event; communication may use a separate backend stream.
    """
    c = plan.dcp_size
    if plan.chunks <= 1:
        group.all_gather_into_tensor(out[: plan.gathered_rows], send)
        return ()
    q, rem = divmod(plan.send_rows, plan.chunks)
    n = plan.chunks
    works = []
    st = 0
    for k in range(n):
        sz = q + (rem if k == n - 1 else 0)
        works.append(
            group.all_gather_into_tensor_async(
                out[c * st : c * (st + sz)], send[st : st + sz]
            )
        )
        st += sz
    for work in works:
        if work is not None:
            work.wait()
    return tuple(works)


def _pad_dcp_packed_send(
    shards: torch.Tensor, plan: "DcpPackedReadPlan", dcp_rank: int
) -> torch.Tensor:
    """This rank's send, padded to ``send_rows`` so every rank sends alike.

    A served prefix is a whole number of ``page_size * dcp_size`` allocator
    pages, so it divides by ``dcp_size`` and no padding is needed at all -- the
    shards go to the collective untouched. Only ranks below ``prefix_len %
    dcp_size`` hold the extra row when a prefix is ragged.
    """
    local_len = plan.prefix_len // plan.dcp_size + int(
        dcp_rank < plan.prefix_len % plan.dcp_size
    )
    assert shards.shape[0] == local_len, (
        f"DCP packed read: rank {dcp_rank} holds {shards.shape[0]} prefix rows "
        f"but the plan computes {local_len} from prefix_len={plan.prefix_len} "
        f"over {plan.dcp_size} ranks"
    )
    if local_len == plan.send_rows:
        return shards.contiguous()
    send = shards.new_zeros((plan.send_rows, *shards.shape[1:]))
    send[:local_len] = shards[:local_len]
    return send


def _dcp_gather_extend_kv_npu(
    m: "DeepseekV2AttentionMLA",
    forward_batch: "ForwardBatch",
    metadata: NPUDecodeContextParallelMetadata,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    index_topk: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialise each request's FULL prefix+extend KV, for one layer.

    The result is one contiguous run per request, which the operator needs under
    the non-paged layout -- so not ``all_gather_kv_cache_for_mla_extend``, which
    groups all prefixes then all extends. Only the prefix crosses the wire: this
    chunk's own KV is already on every rank.

    Output and scratch are reused buffers: a moving context-sized allocation
    makes ranks free and refill out of step and stalls the collectives for
    minutes. The keys travel separately because the operator takes them apart.
    """
    parallel = runtime_context.get_parallel()
    packed_plan = dcp_packed_read_plan(forward_batch, index_topk)
    if packed_plan is not None:
        return _dcp_gather_extend_kv_packed_npu(
            m, forward_batch, k_nope, k_pe, packed_plan
        )
    # The byte rule decided once per forward in build_dcp_extend_gather_plan.
    plan = metadata.extend_gather_plan
    # Wrapper pools (SWA, hybrid) may not offer this.
    plan_write = getattr(get_token_to_kv_pool(), "plan_dcp_extend_write", None)
    if plan_write is not None:
        plan_write(forward_batch.out_cache_loc)
    if _debug_dcp_extend_memory:
        _log_dcp_extend_memory(
            sum(forward_batch.extend_prefix_lens_cpu),
            sum(forward_batch.extend_seq_lens_cpu),
        )

    pool = get_token_to_kv_pool()
    total_rows = plan.pieces[-1].out_end if plan.pieces else 0
    out_nope = dcp_extend_gather_buffer("latent", k_nope, total_rows)
    out_rope = dcp_extend_gather_buffer("rope", k_pe, total_rows)

    # Prefetching needs every piece resident at once: one piece only.
    layer_id = m.attn_mqa.layer_id
    prefetching = _prefetch_dcp_extend_gather and len(plan.pieces) == 1
    _log_dcp_gather_prefetch_once(prefetching, plan, False)
    slot = 0
    state = None
    if prefetching:
        state = metadata.gather_prefetch_state
        if state is None:
            state = _DcpGatherPrefetch(_get_dcp_gather_prefetch_stream())
            metadata.gather_prefetch_state = state
        slot = state.slot_of(layer_id, pool.start_layer)

    # The prefetch must carry every key this cache uses, or a dropped one is
    # read as whatever the previous layer left in its scratch.
    scratches = _dcp_extend_gather_scratches(slot, k_nope, k_pe, plan.scratch_rows)

    # Every rank runs (or skips) the same collectives in the same order. Gather
    # and index_select stay INTERLEAVED per piece: pieces share one scratch, so
    # hoisting the gathers would let piece n+1 overwrite piece n unread.
    done = state.pending.pop(layer_id, None) if prefetching else None
    sends = None
    if done is not None:
        torch.npu.current_stream().wait_event(done[1])
    else:
        sends = _dcp_extend_send_rows(pool, m.attn_mqa, metadata, plan, layer_id)

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
            state, pool, m, metadata, plan, parallel, layer_id, k_nope, k_pe
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
    """Start the next layer's prefix all-gather(s) on the side stream."""
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
    # The backend's sparse operator reads this in place of the paged pool.
    dcp_gathers_kv = _dsa_dcp_extend_gathers_kv(forward_batch)
    dcp_metadata = forward_batch.attn_dcp_metadata
    if dcp_gathers_kv:
        dcp_metadata.extend_kv = _dcp_gather_extend_kv_npu(
            m,
            forward_batch,
            dcp_metadata,
            k_nope,
            k_pe,
            topk_indices.shape[-1] if topk_indices is not None else None,
        )
    backend = get_mla_dcp_backend(forward_context.get_attn_backend())
    # Under DCP, set only for an extend that gathers the KV: the plan refuses
    # the query gather, so a forward never takes both paths.
    dsa_token_shard_plan = get_dsa_token_shard_plan(forward_batch)
    if dsa_token_shard_plan is None:
        attn_output = forward_mla_attention_npu(
            m,
            backend,
            forward_batch,
            q_nope_out.contiguous(),
            k_nope.contiguous(),
            q_pe.contiguous(),
            k_pe.contiguous(),
            save_kv_cache=not mla_preprocess_used,
            topk_indices=topk_indices,
        )
    else:
        if topk_indices is None or m.attn_mqa_for_dsa_token_shard is None:
            raise RuntimeError(
                "DSA token-shard planned this forward but the layer is not set up for "
                f"it: attn_mqa_for_dsa_token_shard={m.attn_mqa_for_dsa_token_shard is not None}, "
                f"topk_indices={topk_indices is not None}"
            )
        dsa_token_shard_rows = q_nope_out.shape[0]
        q_nope_out = dsa_token_shard_redistribute_heads(
            q_nope_out, dsa_token_shard_plan
        )
        q_pe = dsa_token_shard_redistribute_heads(q_pe, dsa_token_shard_plan)
        # A separate name is load-bearing: topk_indices is returned for the
        # next layer, and rebinding it would hand that a slice of a slice.
        topk_is_local = getattr(forward_batch, "npu_indexer_topk_is_local", None)
        if topk_is_local is None:
            # No indexer ran on this stage: a PP stage starting on a skip-top-k
            # layer reuses the previous stage's top-k on a fresh batch, and the
            # flag was set on that batch, not this one. Classify by width --
            # this rank's rows or the full batch -- because slicing a local
            # tensor again gives every rank the wrong rows with no shape error.
            if topk_indices.shape[0] == dsa_token_shard_plan.rows:
                topk_is_local = True
            elif topk_indices.shape[0] == dsa_token_shard_rows:
                topk_is_local = False
            else:
                raise RuntimeError(
                    f"DSA top-k carries {topk_indices.shape[0]} rows, neither "
                    f"this rank's {dsa_token_shard_plan.rows} nor the batch's "
                    f"{dsa_token_shard_rows}; refusing to guess its layout"
                )
            forward_batch.npu_indexer_topk_is_local = topk_is_local
        if topk_is_local:
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
        attn_output = m.attn_mqa_for_dsa_token_shard(
            q_nope_out.contiguous(),
            k_nope.contiguous(),
            k_nope.contiguous(),
            forward_batch,
            save_kv_cache=not mla_preprocess_used,
            q_rope=q_pe.contiguous(),
            k_rope=k_pe.contiguous(),
            topk_indices=attn_topk_indices,
        )
        attn_output = dsa_token_shard_restore_tokens(
            attn_output.reshape(dsa_token_shard_plan.rows, -1, m.kv_lora_rank),
            dsa_token_shard_plan,
            dsa_token_shard_rows,
        )
        attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    if dcp_gathers_kv:
        # So no later layer or forward reads a stale gather.
        dcp_metadata.extend_kv = None

    attn_bmm_output = batch_matmul_transpose(attn_output, m.w_vc, impl="torch_npu")

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
            and m.mla_preprocess.weight_quant_mode != 2
            and m.w_kc is not None
        ):
            m.w_kc.untyped_storage().resize_(0)
    # MLAProlog returns q_lora; BF16 indexers need it recomputed below.
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
        if m.mla_preprocess.weight_quant_mode == 2 and m.indexer is not None:
            from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
                NPUW8A8Int8DynamicLinearMethod,
            )

            if not isinstance(
                getattr(m.indexer.wq_b.scheme, "kernel", None),
                NPUW8A8Int8DynamicLinearMethod,
            ):
                # BF16 indexers need q_lora before INT8 quantization.
                q = m.fused_qkv_a_proj_with_mqa(hidden_states)[0][..., : m.q_lora_rank]
                q_lora = m.q_a_layernorm(q)
                dynamic_scale = None
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
            torch.npu.current_stream().wait_stream(m.alt_stream)
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
