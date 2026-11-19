from dataclasses import fields
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch_npu
from sgl_kernel_npu.norm.fused_split_qk_norm import fused_split_qk_norm

import sglang.srt.layers.dcp.comm as dcp_comm
import sglang.srt.layers.dcp.layout as dcp_layout
import sglang.srt.model_executor.forward_context as forward_context
import sglang.srt.runtime_context as runtime_context
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
    dcp_extend_gather_buffer,
    dcp_extend_gathers_kv,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
from sglang.srt.layers.attention.dsa.dsa_token_shard import (
    dsa_cp_stream_o_proj,
    dsa_token_shard_redistribute_heads,
    dsa_token_shard_restore_tokens,
    dsa_token_shard_slice,
    get_dsa_token_shard_plan,
)
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.dcp.metadata import dcp_token_sharded
from sglang.srt.layers.dcp.planner import prepare_decode_context_parallel_metadata
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.runtime_context import get_disagg
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator
_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()
_is_npu_arch35 = is_npu_arch35()


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
    input_on_attn_tp_slices: bool,
    prev_topk_indices: torch.Tensor = None,
):
    dynamic_scale = None
    # Resolved here so the core can read the cached plan back.
    dsa_cp_plan = get_dsa_token_shard_plan(forward_batch)
    token_sharded = dcp_token_sharded(forward_batch)
    if token_sharded:
        # Later layers already receive local rows; positions remain full width.
        if hidden_states.shape[0] != dsa_cp_plan.rows:
            hidden_states = dsa_token_shard_slice(hidden_states, dsa_cp_plan)
        if positions.shape[0] != dsa_cp_plan.rows:
            positions = dsa_token_shard_slice(positions, dsa_cp_plan)

    def _run_q_b(q_in: torch.Tensor) -> torch.Tensor:
        if token_sharded:
            return torch.nn.functional.linear(q_in, m.q_b_proj_qrep_weight).view(
                -1, m.num_heads, m.qk_head_dim
            )
        return m.q_b_proj(q_in)[0].view(-1, m.num_local_heads, m.qk_head_dim)

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
        if m.rotary_emb is None or m.rotary_emb.is_neox_style:
            q, latent_cache = fused_qkv_a_proj_out.split(
                [m.q_lora_rank, m.kv_lora_rank + m.qk_rope_head_dim], dim=-1
            )
            # overlap qk norm
            q = m.q_a_layernorm(q)
            if _use_ag_after_qlora and input_on_attn_tp_slices and not token_sharded:
                q = scattered_to_tp_attn_full(q, forward_batch)
                latent_cache = scattered_to_tp_attn_full(latent_cache, forward_batch)
            q_lora = q.clone()  # required for topk_indices

            q_event = None
            if m.alt_stream is not None:
                m.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(m.alt_stream):
                    q = _run_q_b(q_lora)
                    # record q to ensure memory space will not be released
                    q.record_stream(m.alt_stream)
                    q_event = m.alt_stream.record_event()
            else:
                q = _run_q_b(q_lora)

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
            q = _run_q_b(q_lora)

        q_nope, q_pe = q.split([m.qk_nope_head_dim, m.qk_rope_head_dim], dim=-1)

        w_kc = m.w_kc_qrep if token_sharded else m.w_kc
        q_nope_out = torch_npu.npu_transpose_batchmatmul(
            q_nope,
            w_kc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )

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

        if token_sharded:
            # Pool writes and the DCP gather consume full-width chunk KV.
            group = runtime_context.get_parallel().attn_tp_group
            k_nope_send = k_nope.contiguous()
            k_pe_send = k_pe.contiguous()
            k_nope = k_nope_send.new_empty(
                (dsa_cp_plan.num_tokens_pad, *k_nope_send.shape[1:])
            )
            k_pe = k_pe_send.new_empty(
                (dsa_cp_plan.num_tokens_pad, *k_pe_send.shape[1:])
            )
            forward_batch.attn_dcp_metadata.chunk_kv_works = (
                group.all_gather_into_tensor_async(k_nope, k_nope_send),
                group.all_gather_into_tensor_async(k_pe, k_pe_send),
                k_nope_send,
                k_pe_send,
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
        **{f.name: getattr(metadata, f.name) for f in fields(metadata)}
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


def _dcp_gather_extend_kv_npu(
    m: "DeepseekV2AttentionMLA",
    metadata: NPUDecodeContextParallelMetadata,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
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
    plan = metadata.extend_gather_plan
    total_rows = plan.pieces[-1].out_end if plan.pieces else 0
    out_nope = dcp_extend_gather_buffer("latent", k_nope, total_rows)
    out_rope = dcp_extend_gather_buffer("rope", k_pe, total_rows)
    scratches = [
        (
            dcp_extend_gather_buffer("latent_scratch", k_nope, plan.scratch_rows),
            k_nope,
        ),
        (
            dcp_extend_gather_buffer("rope_scratch", k_pe, plan.scratch_rows),
            k_pe,
        ),
    ]

    # Prefix rows only, so it reads nothing this forward wrote.
    sends = [None, None]
    if plan.send_rows:
        sends = list(
            get_token_to_kv_pool().get_mla_kv_buffer(
                m.attn_mqa, metadata.dcp_local_prefix_kv_indices
            )
        )
        if plan.local_lens != plan.padded_lens:
            # Served prefixes are cycle-aligned and need no padding; this is
            # the general case.
            sends = [_pad_dcp_extend_send(send, plan) for send in sends]

    # Every rank runs (or skips) the same collectives in the same order. Gather
    # and index_select stay INTERLEAVED per piece: pieces share one scratch, so
    # hoisting the gathers would let piece n+1 overwrite piece n unread.
    for piece in plan.pieces:
        gathered = (piece.send_end - piece.send_start) * parallel.dcp_size
        rows = gathered + piece.extend_end - piece.extend_start
        for out, (buf, own), send in zip((out_nope, out_rope), scratches, sends):
            if gathered:
                parallel.dcp_group.all_gather_into_tensor(
                    buf[:gathered], send[piece.send_start : piece.send_end]
                )
            scratch = buf[:rows]
            scratch[gathered:] = own[piece.extend_start : piece.extend_end]
            torch.index_select(
                scratch, 0, piece.index, out=out[piece.out_start : piece.out_end]
            )
    return out_nope, out_rope


# Two stream-ordered slots avoid a standing full O-proj replica per layer.
_o_proj_slots = [None, None]
_o_proj_gather_slots = [None, None]


def _dcp_o_proj_weight(m) -> torch.Tensor:
    shard = m.o_proj_stream_shard
    if shard is None:
        raise RuntimeError(f"layer {m.layer_id} has no streamed O-proj shard")
    parallel = runtime_context.get_parallel()
    slot = m.layer_id % 2
    full_shape = (shard.shape[0], shard.shape[1] * parallel.attn_tp_size)
    gather_shape = (parallel.attn_tp_size, *shard.shape)
    full = _o_proj_slots[slot]
    if (
        full is None
        or full.shape != full_shape
        or full.dtype != shard.dtype
        or full.device != shard.device
    ):
        full = shard.new_empty(full_shape)
        _o_proj_slots[slot] = full
    gathered = _o_proj_gather_slots[slot]
    if (
        gathered is None
        or gathered.shape != gather_shape
        or gathered.dtype != shard.dtype
        or gathered.device != shard.device
    ):
        gathered = shard.new_empty(gather_shape)
        _o_proj_gather_slots[slot] = gathered
    parallel.attn_tp_group.all_gather_into_tensor(gathered, shard)
    # Flat rank-major output becomes [out, rank, local_in], then [out, full_in].
    full.view(full_shape[0], parallel.attn_tp_size, shard.shape[1]).copy_(
        gathered.movedim(0, 1)
    )
    return full


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
    token_sharded = dcp_token_sharded(forward_batch)
    dsa_token_shard_plan = get_dsa_token_shard_plan(forward_batch)
    if token_sharded:
        works = dcp_metadata.chunk_kv_works
        if works is not None:
            for work in works[:2]:
                if work is not None:
                    work.wait()
            dcp_metadata.chunk_kv_works = None
        # Communication pads to the TP grid; cache locations contain real rows.
        k_nope = k_nope[: dsa_token_shard_plan.num_tokens]
        k_pe = k_pe[: dsa_token_shard_plan.num_tokens]
    if dcp_gathers_kv:
        dcp_metadata.extend_kv = _dcp_gather_extend_kv_npu(
            m, dcp_metadata, k_nope, k_pe
        )
    backend = get_mla_dcp_backend(forward_context.get_attn_backend())
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
        # Not forward_mla_attention_npu: no DCP partials to merge, and that
        # helper's closing view assumes per-rank heads, not the full-head
        # token slice this attention returns.
        if topk_indices is None or m.attn_mqa_for_dsa_token_shard is None:
            raise RuntimeError(
                "DSA token-shard planned this forward but the layer is not set up for "
                f"it: attn_mqa_for_dsa_token_shard={m.attn_mqa_for_dsa_token_shard is not None}, "
                f"topk_indices={topk_indices is not None}"
            )
        # The rebuild already has every head on local rows; ordinary query
        # sharding still needs the head redistribution.
        dsa_token_shard_rows = q_nope_out.shape[0]
        if not token_sharded:
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
        if token_sharded or topk_is_local:
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
        if not token_sharded:
            attn_output = dsa_token_shard_restore_tokens(
                attn_output.reshape(dsa_token_shard_plan.rows, -1, m.kv_lora_rank),
                dsa_token_shard_plan,
                dsa_token_shard_rows,
            )
            attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    if dcp_gathers_kv:
        # So no later layer or forward reads a stale gather.
        dcp_metadata.extend_kv = None

    num_heads = m.num_heads if token_sharded else m.num_local_heads
    attn_output = attn_output.view(-1, num_heads, m.kv_lora_rank)
    w_vc = m.w_vc_qrep if token_sharded else m.w_vc
    perm_x1 = (1, 0, 2)
    if not _is_npu_arch35 and num_heads * m.kv_lora_rank >= 65536:
        # A2/A3's fused input transpose requires heads * kv_lora_rank < 65536.
        attn_output = attn_output.transpose(0, 1).contiguous()
        perm_x1 = (0, 1, 2)
    attn_bmm_output = torch_npu.npu_transpose_batchmatmul(
        attn_output,
        w_vc,
        perm_x1=perm_x1,
        perm_x2=(0, 1, 2),
        perm_y=(1, 0, 2),
    )

    attn_bmm_output = attn_bmm_output.reshape(-1, num_heads * m.v_head_dim)

    if gate is not None:
        attn_bmm_output = m._apply_gated(attn_bmm_output, gate)
    if token_sharded:
        weight = (
            _dcp_o_proj_weight(m) if dsa_cp_stream_o_proj() else m.o_proj_qrep_weight
        )
        output = torch.nn.functional.linear(attn_bmm_output, weight)
    else:
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
