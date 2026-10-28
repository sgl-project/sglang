# SPDX-License-Identifier: Apache-2.0
"""ROCm glue of minimax_m3: the branches the model takes under _is_hip, bound there as _hip.
On gfx950 the norms quantize for the GEMMs reading them, the SwiGLU writes its down_proj's
fp8, and the sparse kernels hand the o_proj its fp8 input; the gfx950 MoE layer lives in
minimax_m3_hip_moe."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.kernels.ops.attention.minimax_sparse.common.nvfp4 import Nvfp4Cache
from sglang.kernels.ops.quantization.aiter_fp8_quant_triton import swiglu_oai_fp8
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary.residual.add_norm import (
    NormReadout,
    _is_gfx95_supported,
    _use_aiter,
)
from sglang.srt.layers.layer_boundary.residual.add_norm_hip import gemma_norm_fp8
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.layers.quantization.fp8_utils import fp8_input_quant_format
from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
from sglang.srt.model_executor.runner_utils.prefill_graph import (
    is_in_prefill_graph_scope,
)
from sglang.srt.runtime_context import get_memory, get_parallel
from sglang.srt.utils import is_gfx95_supported

_FP8_KV_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
)


def init_attention(attn) -> None:
    """The ROCm state of MiniMaxM3Attention.__init__."""
    # HiSparse stores through a slot translation the fused cache store skips
    attn._enable_hisparse = get_memory().enable_hisparse
    # gfx950's sparse kernels can hand the o_proj its input already in fp8
    attn._o_proj_quant_format = (
        o_proj_input_quant_format if is_gfx95_supported() else None
    )


def sparse_qk_index_norm_rope_cache(
    attn, positions, q, k, v, idx_q, idx_k, idx_v, forward_batch
):
    """MiniMaxM3Attention._sparse_qk_index_norm_rope_cache on ROCm, where the fused kernel
    also stores K/V into an fp8 main cache or the packed NVFP4 pool."""
    kv_pool = attn._get_sparse_kv_pool()
    nvfp4_pool = isinstance(kv_pool, MiniMaxM3Nvfp4KVPool)
    # The kernel casts K/V to an fp8 main cache or quantizes them to the packed
    # NVFP4 pool itself; other quantized caches keep the separate store.
    main_kv_dtype_ok = kv_pool is not None and (
        nvfp4_pool or kv_pool.dtype == q.dtype or kv_pool.dtype in _FP8_KV_DTYPES
    )
    # the kernel stores index K unscaled; the pool's own store divides by a non-unit
    # idx_k scale when it casts
    idx_k_scale_ok = attn.attn.idx_k_scale_float in (None, 1.0) or (
        kv_pool is not None and not nvfp4_pool and kv_pool.index_k_pool.dtype == q.dtype
    )
    if not (
        main_kv_dtype_ok
        and idx_k_scale_ok
        and not attn._enable_hisparse
        and idx_v is None
        and attn._can_use_rocm_sparse_qk_index_norm_rope(positions, q, k, idx_q, idx_k)
        and forward_batch.out_cache_loc is not None
        and v.dim() == 2
        and v.dtype == q.dtype
        and v.shape == k.shape
    ):
        return attn._sparse_qk_index_norm_rope(positions, q, k, idx_q, idx_k)
    from sglang.kernels.ops.attention.minimax_m3_qk_norm_rope import (
        sparse_qk_index_gemma_rmsnorm_rope_cache,
    )

    layer_id = attn.attn.layer_id
    if nvfp4_pool:
        k_cache, v_cache, idx_k_cache = nvfp4_layer_caches(kv_pool, layer_id)
    else:
        k_cache, v_cache = kv_pool.get_kv_buffer(layer_id)
        idx_k_cache = kv_pool.get_index_k_buffer(layer_id)
    q, k, idx_q, idx_k = sparse_qk_index_gemma_rmsnorm_rope_cache(
        q,
        k,
        v,
        idx_q,
        idx_k,
        k_cache,
        v_cache,
        idx_k_cache,
        forward_batch.out_cache_loc,
        attn.q_norm.weight.data,
        attn.k_norm.weight.data,
        attn.index_q_norm.weight.data,
        attn.index_k_norm.weight.data,
        positions,
        attn.rotary_emb.cos_sin_cache,
        attn.q_norm.variance_epsilon,
        attn.head_dim,
        attn.rotary_dim,
        attn.rotary_emb.is_neox_style,
        k_scale=attn.attn.k_scale_float,
        v_scale=attn.attn.v_scale_float,
    )
    attn._mark_sparse_kv_cached_by_fusion(forward_batch, layer_id)
    return q, k, idx_q, idx_k


def sparse_attn_kwargs(attn, forward_batch, q) -> dict:
    """The sparse attention call's ROCm keywords."""
    kwargs = {}
    # a breakable graph replays the attention with this capture-time flag on a
    # batch that does not know which layers the qk-norm/RoPE kernel stored
    precached = forward_batch.minimax_m3_precached_sparse_layers
    if precached is not None and attn.attn.layer_id in precached:
        kwargs["save_kv_cache"] = False
    if attn._o_proj_quant_format is not None:
        out_quant = attn._o_proj_quant_format(attn.o_proj, forward_batch, q.shape[0])
        # only passed when set: other backends take no out_quant
        if out_quant:
            kwargs["out_quant"] = out_quant
    return kwargs


def moe_cls():
    """The gfx950 MoE layer; it subclasses minimax_m3's MiniMaxM3MoE, hence the late import."""
    from sglang.srt.models.minimax_common.amd.minimax_m3_hip_moe import (
        MiniMaxM3MoEGfx950,
    )

    return MiniMaxM3MoEGfx950


def norm_quant(layer) -> Optional["Gfx950NormQuant"]:
    # the DP and scattered attention-TP gathers after a norm take a tensor, not its
    # (fp8, scale) tuple
    if (
        not envs.SGLANG_USE_AITER.get()
        or is_dp_attention_enabled()
        or get_parallel().enable_attn_tp_input_scattered
    ):
        return None
    return Gfx950NormQuant(layer)


def bind_fp8_norms(model) -> None:
    """Bind every decoder layer's norms to the GEMMs reading their fused fp8 output, once the
    fused qkv + index projections exist."""
    from sglang.srt.models.minimax_m3 import MiniMaxM3DecoderLayer

    for module in model.modules():
        if isinstance(module, MiniMaxM3DecoderLayer) and module._norm_quant is not None:
            module._norm_quant.bind()


@dataclass(frozen=True)
class NormFp8Readout(NormReadout):
    """The residual's norm, as ``(fp8, scale, bf16 norm)`` when the consumer's GEMM
    takes the fp8 in ``quant_format_for(rows)`` ("fp8_per_token" or "fp8") and its
    other users read the norm: one gfx95 kernel adds, applies a Gemma norm and
    quantizes as the separate aiter quant kernel does."""

    quant_format_for: Callable[[int], str] = lambda num_tokens: ""

    def update_and_read(
        self,
        update,
        hidden_states,
        residual,
        norm,
        quant_format="",
        post_residual_addition=None,
    ):
        if (
            _use_aiter
            and _is_gfx95_supported
            and residual is not None
            and update.is_plain_add
            and not quant_format
            and post_residual_addition is None
            and isinstance(norm, GemmaRMSNorm)
            and hidden_states.shape[0] > 0
        ):
            consumer_format = self.quant_format_for(hidden_states.shape[0])
            if consumer_format:
                return gemma_norm_fp8(
                    norm, hidden_states, residual, None, consumer_format, keep_bf16=True
                )
        return super().update_and_read(
            update, hidden_states, residual, norm, quant_format, post_residual_addition
        )


class Gfx950NormQuant:
    """A decoder layer's norms quantized to the fp8 the next GEMM reads: the input norm
    for the fused qkv + index projection, the FFN norm for the shared expert's gate_up
    (the dense gate_up in dense layers)."""

    def __init__(self, layer):
        self.layer = layer

    def attn_quant_format(self, num_tokens: int) -> str:
        proj = self.layer.self_attn._fused_qkv_index
        return "" if proj is None else fp8_input_quant_format(proj.weight, num_tokens)

    def _ffn_gate_up(self):
        layer = self.layer
        if not layer.is_layer_sparse:
            return layer.mlp.gate_up_proj
        if (
            layer.mlp.shared_experts is None
            or get_moe_a2a_backend().is_megamoe()
            or get_moe_a2a_backend().is_deepep()
        ):
            # only forward_normal splits the input for the shared expert
            return None
        return layer.mlp.shared_experts.gate_up_proj

    def ffn_quant_format(self, num_tokens: int) -> str:
        gate_up = self._ffn_gate_up()
        if gate_up is None:
            return ""
        return fp8_input_quant_format(gate_up.weight, num_tokens)

    def ffn_read(self) -> NormFp8Readout:
        return NormFp8Readout(quant_format_for=self.ffn_quant_format)

    def bind(self) -> None:
        """Bind each norm to the GEMM that reads its fused fp8 output."""
        layer = self.layer
        for norm, proj in (
            (layer.input_layernorm, layer.self_attn._fused_qkv_index),
            (layer.post_attention_layernorm, self._ffn_gate_up()),
        ):
            if isinstance(norm, GemmaRMSNorm) and proj is not None:
                norm.fp8_out_gemm_shape = tuple(proj.weight.shape)


class Gfx950SwigluFp8:
    """An MLP's SwiGLU-OAI writing the fp8 its down_proj's rowwise or block GEMM reads
    (fp8_input_quant_format), else the bf16 activation of ``act_fn``."""

    def __init__(self, mlp, act_fn):
        self.mlp = mlp
        self.act_fn = act_fn

    def args(self, num_tokens: int):
        """swiglu_oai_fp8's (alpha, limit, group_size, transpose_scale) at num_tokens
        rows, None where the fp8 SwiGLU does not apply."""
        if num_tokens == 0:
            return None
        quant_format = fp8_input_quant_format(self.mlp.down_proj.weight, num_tokens)
        if not quant_format:
            return None
        # "fp8" is the bpreshuffle block GEMM, which reads its scale column-major
        block = quant_format == "fp8"
        return (
            self.mlp.swiglu_alpha,
            self.mlp.swiglu_limit,
            128 if block else None,
            block,
        )

    def __call__(self, gate_up: torch.Tensor):
        args = self.args(gate_up.shape[0])
        if args is None:
            return self.act_fn(gate_up)
        alpha, limit, group_size, transpose_scale = args
        return swiglu_oai_fp8(
            gate_up,
            alpha,
            limit,
            group_size=group_size,
            transpose_scale=transpose_scale,
        )


def nvfp4_layer_caches(kv_pool, layer_id: int):
    """Main K/V and index K of one layer of the packed NVFP4 pool, write side."""
    k_payload, v_payload, k_scales, v_scales = kv_pool.main_pool.get_raw_kv_buffers(
        layer_id
    )
    idx_payload, idx_scales = kv_pool.index_k_pool.get_raw_k_buffers(
        kv_pool.index_k_layer_id_mapping[layer_id]
    )
    return (
        Nvfp4Cache(payload=k_payload, scales=k_scales),
        Nvfp4Cache(payload=v_payload, scales=v_scales),
        Nvfp4Cache(payload=idx_payload, scales=idx_scales),
    )


def o_proj_input_quant_format(o_proj, forward_batch, num_tokens: int) -> str:
    """The fp8 the sparse decode and the Gluon prefill pair kernel hand the o_proj in
    (fp8_input_quant_format), or "" where the attention output stays bf16."""
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_extend()):
        return ""
    if mode.is_extend() and is_in_prefill_graph_scope():
        # RadixAttention's prefill-graph wrapper copies the output into its own bf16 buffer
        return ""
    real_num_tokens = forward_batch.global_num_token_non_padded_cpu
    if (
        mode.is_extend()
        and not mode.is_target_verify()
        and real_num_tokens is not None
        and 0 < real_num_tokens < num_tokens
    ):
        # attention re-pads only a bf16 output to the MLP-sync padded rows
        return ""
    return fp8_input_quant_format(o_proj.weight, num_tokens)
