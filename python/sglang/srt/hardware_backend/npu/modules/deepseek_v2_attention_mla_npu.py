from typing import TYPE_CHECKING

import torch
import torch_npu
from sgl_kernel_npu.norm.fused_split_qk_norm import fused_split_qk_norm

from sglang.kernels.ops.gemm.batch_matmul_transpose_npu import (
    batch_matmul_transpose_npu,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.fp8_contracts import (
    normalize_required_fp8_scale,
)
from sglang.srt.hardware_backend.npu.attention.mla_preprocess import (
    NPUFusedMLAPreprocess,
    is_fia_nz,
    is_mla_preprocess_enabled,
)
from sglang.srt.layers.attention.dsa.dsa_npu_indexer import scattered_to_tp_attn_full
from sglang.srt.layers.attention.dsa.utils import (
    dsa_use_prefill_cp,
)
from sglang.srt.layers.communicator import ScatterMode, get_attn_tp_context
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from sglang.srt.utils import is_npu_before_atlas_a5

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
    from sglang.srt.utils import BumpAllocator
_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


def _use_explicit_npu_interleaved_rope(m: "DeepseekV2AttentionMLA") -> bool:
    return m.rotary_emb is not None and vars(m).get(
        "use_explicit_npu_interleaved_rope", False
    )


def _apply_dsa_interleave_half_rope(
    m: "DeepseekV2AttentionMLA",
    positions: torch.Tensor,
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    forward_batch: "ForwardBatch",
) -> tuple[torch.Tensor, torch.Tensor]:
    if m.qk_rope_head_dim != 64:
        raise RuntimeError(
            "NPU interleave-half RoPE for DSA requires qk_rope_head_dim=64, "
            f"got {m.qk_rope_head_dim}."
        )

    rope_cache = forward_batch.npu_dsa_interleave_half_rope_cache
    if rope_cache is None:
        m.rotary_emb.get_cos_sin_with_position(positions)
        rope_cache = (
            m.rotary_emb.position_cos.to(device=q_pe.device, dtype=q_pe.dtype),
            m.rotary_emb.position_sin.to(device=q_pe.device, dtype=q_pe.dtype),
        )
        forward_batch.npu_dsa_interleave_half_rope_cache = rope_cache
    cos, sin = rope_cache

    q_shape = q_pe.shape
    k_shape = k_pe.shape
    q_pe = torch_npu.npu_interleave_rope(
        q_pe.reshape(q_shape[0], q_shape[1], 1, q_shape[2]),
        cos,
        sin,
    ).reshape(q_shape)
    k_pe = torch_npu.npu_interleave_rope(
        k_pe.reshape(k_shape[0], k_shape[1], 1, k_shape[2]),
        cos,
        sin,
    ).reshape(k_shape)
    return q_pe, k_pe


def _get_fp8_kv_runtime_scale(
    m: "DeepseekV2AttentionMLA",
    attr_name: str,
    device: torch.device,
) -> torch.Tensor | None:
    if m.kv_cache_dtype != "fp8_e4m3":
        return None

    requires_modelslim_scale = vars(m).get("_requires_modelslim_fp8_kv_scale", False)
    if requires_modelslim_scale and not vars(m).get(
        "_modelslim_fp8_kv_scale_ready", False
    ):
        raise RuntimeError(
            "ModelSlim FP8 MLA KV scales were not validated after checkpoint "
            "loading; refusing to use uninitialized runtime descales."
        )

    scale = m._buffers.get(attr_name)
    if scale is None:
        scale = m._parameters.get(attr_name)
    if scale is None:
        if requires_modelslim_scale:
            raise RuntimeError(
                f"{attr_name} is required by the ModelSlim FP8 KV scheme, but "
                "the checkpoint-derived runtime scale is missing."
            )

        # Generic ``--kv-cache-dtype fp8_e4m3`` stores a direct FP8 cast and
        # therefore has an explicit unit-scale contract.  Keep one persistent
        # per-layer tensor for graph capture instead of allocating per forward.
        fallback_attr = f"_{attr_name}_direct_fp8_fallback"
        scale = m._buffers.get(fallback_attr)
        if scale is None:
            scale = torch.ones(
                (1, vars(m).get("num_local_kv_heads", 1)),
                dtype=torch.float32,
                device=device,
            )
            m.register_buffer(fallback_attr, scale, persistent=False)
    return normalize_required_fp8_scale(
        scale,
        name=attr_name,
        device=device,
    )


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

        if m.use_dsa and m.indexer is not None:
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

    if m.use_deepseek_yarn_rope or _use_explicit_npu_interleaved_rope(m):
        B, S = q.shape[0], 1
        if m.use_deepseek_yarn_rope:
            cos, sin = m.rotary_emb.get_cos_sin_cache(
                positions,
                hidden_states.dtype,
                offsets=None,
            )
        else:
            cos, sin = m.rotary_emb.get_cos_sin_cache(
                positions,
                hidden_states.dtype,
                offsets=None,
            )
        q_pe = torch_npu.npu_interleave_rope(
            q_pe.reshape(B, -1, S, m.qk_rope_head_dim),
            cos,
            sin,
        )
        q_pe = q_pe.reshape(B, -1, m.qk_rope_head_dim)

        ckv_cache, k_rope_cache = get_token_to_kv_pool().get_kv_buffer(m.layer_id)
        c_kv_scale = _get_fp8_kv_runtime_scale(
            m,
            "fak_descale_reciprocal",
            q.device,
        )
        _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache.view(-1, 1, 1, m.kv_lora_rank + m.qk_rope_head_dim),  # bnsd
            m.kv_a_layernorm.weight,
            cos,
            sin,
            forward_batch.out_cache_loc.to(torch.int64),
            k_rope_cache,
            ckv_cache,
            k_rope_scale=None,
            c_kv_scale=c_kv_scale,
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
    return (
        q,
        k,
        v,
        forward_batch,
        _get_fp8_kv_runtime_scale(m, "fak_descale_float", q.device),
    )


def forward_mha_core_npu(
    m: "DeepseekV2AttentionMLA",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: "ForwardBatch",
    fp8_kv_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    attn_output = m.attn_mha(
        q,
        k,
        v,
        forward_batch,
        save_kv_cache=False,
        fp8_kv_scale=fp8_kv_scale,
    )
    attn_output = attn_output.reshape(-1, m.num_local_heads * m.v_head_dim)
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
        if m._modules.get("mla_preprocess") is None:
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
                _get_fp8_kv_runtime_scale(
                    m,
                    "fak_descale_reciprocal",
                    hidden_states.device,
                ),
            )
        else:
            m.mla_preprocess.runtime_refs["fak_descale_reciprocal"] = (
                _get_fp8_kv_runtime_scale(
                    m,
                    "fak_descale_reciprocal",
                    hidden_states.device,
                )
            )
        (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            _,
            forward_batch,
            zero_allocator,
            positions,
            _,
            dequant_scale_q_nope,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        topk_indices = None
    else:
        q_lora = None
        dequant_scale_q_nope = None
        if m.q_lora_rank is not None:
            qkv_latent = get_attn_tp_context().fetch_qkv_latent()
            latent_cache = qkv_latent[..., m.q_lora_rank :]
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

        explicit_rope_cos_sin = None
        if _use_explicit_npu_interleaved_rope(m):
            explicit_rope_cos_sin = m.rotary_emb.get_cos_sin_cache(
                positions,
                hidden_states.dtype,
                offsets=None,
            )

        if m.rotary_emb is not None:
            q_pe, k_pe = m.rotary_emb(positions, q_pe, k_pe)

        if m.kv_cache_dtype == "fp8_e4m3":
            if explicit_rope_cos_sin is not None:
                cos, sin = explicit_rope_cos_sin
                cos = cos.to(q_nope_out.device)
                sin = sin.to(q_nope_out.device)
            else:
                cos = m.rotary_emb.cos_cached.to(q_nope_out.device)
                sin = m.rotary_emb.sin_cached.to(q_nope_out.device)

            q_nope_shape = q_nope_out.shape
            q_nope_out, dequant_scale_q_nope = torch_npu.npu_dynamic_quant(
                q_nope_out.reshape(-1, q_nope_shape[-1]),
                dst_type=torch.float8_e4m3fn,
            )
            q_nope_out = q_nope_out.view(q_nope_shape)
            dequant_scale_q_nope = dequant_scale_q_nope.view(q_nope_shape[:-1]).to(
                torch.float32
            )

            fp8_kv_scale = _get_fp8_kv_runtime_scale(
                m,
                "fak_descale_float",
                q_pe.device,
            )
            q_pe = (q_pe / dequant_scale_q_nope.unsqueeze(-1) / fp8_kv_scale).to(
                torch.bfloat16
            )

            ckv_cache, k_rope_cache = get_token_to_kv_pool().get_kv_buffer(m.layer_id)
            c_kv_scale = _get_fp8_kv_runtime_scale(
                m,
                "fak_descale_reciprocal",
                q_nope_out.device,
            )
            _, _, k_pe, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(
                    -1,
                    1,
                    1,
                    m.kv_lora_rank + m.qk_rope_head_dim,
                ),
                m.kv_a_layernorm.weight,
                cos,
                sin,
                forward_batch.out_cache_loc.to(torch.int64),
                k_rope_cache,
                ckv_cache,
                k_rope_scale=None,
                c_kv_scale=c_kv_scale,
                k_rope_offset=None,
                c_kv_offset=None,
                epsilon=m.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ" if is_fia_nz() else "PA_BNSD",
                is_output_kv=True,
            )
            k_pe = k_pe.reshape(-1, 1, m.qk_rope_head_dim)
            k_nope = k_nope.reshape(-1, 1, m.kv_lora_rank)
            dequant_scale_q_nope = dequant_scale_q_nope.unsqueeze(-1)

        if dsa_use_prefill_cp(forward_batch):
            if m.kv_cache_dtype == "fp8_e4m3":
                raise NotImplementedError(
                    "Ascend FP8 MLA/DSA KV cache does not support prefill CP."
                )
            # support allgather+rerrange
            k_nope, k_pe = m.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )
        topk_indices = None
        if q_lora is not None and m.indexer is not None:
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
        dequant_scale_q_nope,
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
    dequant_scale_q_nope: torch.Tensor | None = None,
) -> torch.Tensor:
    attention_kwargs = {}
    if topk_indices is not None:
        attention_kwargs["topk_indices"] = topk_indices
    if dequant_scale_q_nope is not None:
        attention_kwargs["dequant_scale_q_nope"] = dequant_scale_q_nope
        attention_kwargs["fp8_kv_scale"] = _get_fp8_kv_runtime_scale(
            m,
            "fak_descale_float",
            q_pe.device,
        )
        if m.kv_cache_dtype == "fp8_e4m3" and (
            forward_batch.forward_mode.is_decode_or_idle()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            attention_kwargs["save_kv_cache"] = False

    attn_output = m.attn_mqa(
        q_nope_out,
        k_nope,
        k_nope,
        forward_batch,
        q_rope=q_pe,
        k_rope=k_pe,
        **attention_kwargs,
    )

    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_output = attn_output.contiguous()
    if not is_npu_before_atlas_a5() and (
        m.use_dsa or _use_explicit_npu_interleaved_rope(m)
    ):
        attn_bmm_output = torch.empty(
            (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
            dtype=attn_output.dtype,
            device=attn_output.device,
        )
        batch_matmul_transpose_npu(
            tensor_a=attn_output,
            tensor_b=m.w_vc,
            tensor_c=attn_bmm_output,
        )
    else:
        # torch.ops.npu.batch_matmul_transpose is not numerically equivalent for
        # Kimi-K3, so retain its numerically validated torch_npu implementation.
        attn_bmm_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            m.w_vc,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)
    output, _ = m.o_proj(attn_bmm_output)

    return output


# endregion


# region DSA
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

        q_nope_out = torch.bmm(q_nope.transpose(0, 1), m.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        if is_mla_preprocess_enabled() and not m.rotary_emb.is_neox_style:
            q_pe, k_pe = _apply_dsa_interleave_half_rope(
                m,
                positions,
                q_pe,
                k_pe,
                forward_batch,
            )
        else:
            if m.layer_id == get_token_to_kv_pool().start_layer:
                m.rotary_emb.sin_cos_cache = m.rotary_emb.cos_sin_cache.index_select(
                    0,
                    positions,
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
        mla_preprocess_used,
        forward_batch,
        zero_allocator,
        positions,
    )


def forward_dsa_core_npu(
    m: "DeepseekV2AttentionMLA",
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    q_nope_out: torch.Tensor,
    k_nope: torch.Tensor,
    topk_indices: torch.Tensor,
    mla_preprocess_used: bool,
    forward_batch: "ForwardBatch",
    zero_allocator: "BumpAllocator",
    positions: torch.Tensor,
) -> torch.Tensor:
    attn_output = m.attn_mqa(
        q_nope_out.contiguous(),
        k_nope.contiguous(),
        k_nope.contiguous(),
        forward_batch,
        save_kv_cache=not mla_preprocess_used,
        q_rope=q_pe.contiguous(),
        k_rope=k_pe.contiguous(),
        topk_indices=topk_indices,
    )
    attn_output = attn_output.view(-1, m.num_local_heads, m.kv_lora_rank)

    attn_bmm_output = torch.empty(
        (attn_output.shape[0], m.num_local_heads, m.v_head_dim),
        dtype=attn_output.dtype,
        device=attn_output.device,
    )

    if (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_draft_extend_v2()
        and not forward_batch.forward_mode.is_target_verify()
    ):
        attn_output = attn_output.transpose(0, 1)
        torch.bmm(
            attn_output,
            m.w_vc,
            out=attn_bmm_output.view(-1, m.num_local_heads, m.v_head_dim).transpose(
                0, 1
            ),
        )
    else:
        attn_output = attn_output.contiguous()
        if is_npu_before_atlas_a5():
            torch.ops.npu.batch_matmul_transpose(attn_output, m.w_vc, attn_bmm_output)
        else:
            batch_matmul_transpose_npu(
                tensor_a=attn_output,
                tensor_b=m.w_vc,
                tensor_c=attn_bmm_output,
            )

    attn_bmm_output = attn_bmm_output.reshape(-1, m.num_local_heads * m.v_head_dim)

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
    if m._modules.get("mla_preprocess") is None:
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
            _get_fp8_kv_runtime_scale(
                m,
                "fak_descale_reciprocal",
                hidden_states.device,
            ),
        )
    else:
        m.mla_preprocess.runtime_refs["fak_descale_reciprocal"] = (
            _get_fp8_kv_runtime_scale(
                m,
                "fak_descale_reciprocal",
                hidden_states.device,
            )
        )

    # MLAProlog returns query_norm directly, so it does not require another
    # QKV-A projection and RMSNorm to produce q_lora.
    if m.mla_preprocess.uses_mlaprolog():
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
            _,
        ) = m.mla_preprocess.forward(
            positions, hidden_states, forward_batch, zero_allocator
        )
        if q_lora is None or q_lora.dim() == 0 or q_lora.shape[-1] != m.q_lora_rank:
            raise RuntimeError("MLAProlog returned an invalid query_norm.")
        if q_lora.dtype == torch.float8_e4m3fn and q_lora.numel() > 0:
            if dynamic_scale is None or dynamic_scale.numel() == 0:
                raise RuntimeError(
                    "MLAProlog returned MXFP8 query_norm without dequant scale."
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
                    _,
                    forward_batch,
                    zero_allocator,
                    positions,
                    _,
                    _,
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
                _,
                forward_batch,
                zero_allocator,
                positions,
                _,
                _,
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
