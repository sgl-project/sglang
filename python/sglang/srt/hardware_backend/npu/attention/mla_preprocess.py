import re
from functools import lru_cache
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.attention.fp8_contracts import (
    DSA_KV_QUANT_TILE_SIZE,
    get_dsa_fp8_packed_cache_dim,
    normalize_required_fp8_scale,
)
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.utils import get_bool_env_var, is_npu_before_atlas_a5

if TYPE_CHECKING:
    from sglang.srt.layers.quantization.base_config import QuantizationConfig


@lru_cache(maxsize=1)
def is_mla_preprocess_enabled() -> bool:
    return get_bool_env_var("SGLANG_NPU_USE_MLAPO")


@lru_cache(maxsize=1)
def is_fia_nz() -> bool:
    is_fia_nz_ = get_bool_env_var("SGLANG_USE_FIA_NZ")
    if is_fia_nz_:
        assert (
            is_mla_preprocess_enabled()
        ), "SGLANG_USE_FIA_NZ must be enable with SGLANG_NPU_USE_MLAPO"
    return is_fia_nz_


def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def transdata(nd_mat, block_size: tuple = (16, 16)):
    r = round_up(nd_mat.shape[0], block_size[0])
    c = round_up(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = F.pad(nd_mat, ((0, r_pad, 0, c_pad)))
    nz_mat = torch.permute(
        torch.reshape(
            nd_mat,
            (r // block_size[0], block_size[0], c // block_size[1], block_size[1]),
        ),
        [2, 0, 1, 3],
    )
    nz_mat = torch.reshape(
        nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3])
    )
    return nz_mat


def trans_rope_weight(weight, rope_dim):
    weight_1 = weight[..., -rope_dim::2, :].contiguous()
    weight_2 = weight[..., -rope_dim + 1 :: 2, :].contiguous()
    weight[..., -rope_dim:, :] = torch.cat([weight_1, weight_2], dim=-2)

    return weight.contiguous()


def _quantize_mla_query_for_mxfp8(
    query: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize MLAProlog input with the MXFP8 block-scale ABI."""

    query_shape = query.shape
    query_2d = query.reshape(-1, query_shape[-1]).contiguous()
    query_mxfp8, query_scale = torch.ops.npu.npu_dynamic_mx_quant(
        query_2d,
        axis=1,
        dst_type=torch.float8_e4m3fn,
        block_size=32,
        scale_alg=None,
    )

    if query_scale.dim() == 3 and query_scale.shape[-1] == 2:
        query_scale = query_scale.contiguous().reshape(query_scale.shape[0], -1)
    elif query_scale.dim() != 2:
        raise RuntimeError(
            f"Unexpected MLAProlog query scale shape: {query_scale.shape}."
        )

    if query_scale.dtype == torch.uint8:
        query_scale = query_scale.view(torch.float8_e8m0fnu)
    return query_mxfp8, query_scale


class NPUFusedMLAPreprocess(torch.nn.Module):
    def __init__(
        self,
        fused_qkv_a_proj_with_mqa,
        q_a_layernorm,
        kv_a_layernorm,
        q_b_proj,
        w_kc,
        rotary_emb,
        layer_id,
        num_local_heads,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        quant_config: Optional["QuantizationConfig"] = None,
        fak_descale_reciprocal: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.qkv_a_proj = fused_qkv_a_proj_with_mqa
        self.q_a_layernorm = q_a_layernorm
        self.kv_a_layernorm = kv_a_layernorm
        self.q_b_proj = q_b_proj
        self.w_kc = w_kc.contiguous()
        self.rotary_emb = rotary_emb
        self.layer_id = layer_id
        self.quant_config = quant_config
        self.has_preprocess_weights = False
        self.dtype = None
        # Resolve the hardware generation after model construction, rather than
        # at module import where torch-npu may not have registered the device.
        self.is_pre_a5 = is_npu_before_atlas_a5()
        self.runtime_refs = {"fak_descale_reciprocal": fak_descale_reciprocal}

        self.q_lora_rank = self.q_b_proj.input_size  # 1536
        self.kv_lora_rank = self.kv_a_layernorm.hidden_size  # 512
        self.num_local_heads = num_local_heads  # tp
        self.qk_nope_head_dim = qk_nope_head_dim  # 128
        self.qk_rope_head_dim = qk_rope_head_dim  # 64
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        q_b_proj_weight_scale = self.q_b_proj._parameters.get("weight_scale")
        self.q_b_proj_weight_scale = (
            q_b_proj_weight_scale.view(1, -1).to(torch.float)
            if q_b_proj_weight_scale is not None
            else None
        )

        quant_method = self.qkv_a_proj.quant_method
        quantization_config = (
            vars(quant_method).get("quantization_config")
            if quant_method is not None
            else None
        )
        self.use_legacy_modelslim_mlapo = (
            self.is_pre_a5
            and quantization_config is not None
            and quantization_config.get_name() == "modelslim"
        )

        ignore_patterns = (
            vars(self.quant_config).get("ignore", ())
            if self.quant_config is not None
            else ()
        )
        self.configured_for_mlaprolog = any(
            re.fullmatch(r".*kv_b_proj", pattern) for pattern in ignore_patterns
        )

    def _uses_legacy_mlapo(self) -> bool:
        return self.use_legacy_modelslim_mlapo

    def uses_mlaprolog(self) -> bool:
        """Return the single dispatch decision used by all NPU MLA callers."""

        if self._uses_legacy_mlapo():
            return False
        return self.configured_for_mlaprolog or not self.is_pre_a5

    def _get_quant_scale_ckv(self, device: torch.device) -> torch.Tensor:
        return normalize_required_fp8_scale(
            self.runtime_refs["fak_descale_reciprocal"],
            name="fak_descale_reciprocal",
            device=device,
        )

    def preprocess_weights(self, hidden_states):
        self.dummy = torch.zeros(
            (hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.qkv_a_proj_input_offset = self.qkv_a_proj.input_offset.to(dtype=torch.int8)
        self.q_b_proj_input_offset = self.q_b_proj.input_offset.to(dtype=torch.int8)

        # matmul_0 weight [7168, 2112]
        fused_qkv_a_proj_with_mqa_weight_q = self.qkv_a_proj.weight.data[
            :, : self.q_lora_rank
        ].clone()  # [7168, 1536]
        fused_qkv_a_proj_with_mqa_weight_kv = self.qkv_a_proj.weight.data[
            :, self.q_lora_rank :
        ].clone()  # [7168, 576]
        # rope fit
        fused_qkv_a_proj_with_mqa_weight_kv_t = (
            fused_qkv_a_proj_with_mqa_weight_kv.t().contiguous()
        )
        fused_qkv_a_proj_with_mqa_weight_kv_t = trans_rope_weight(
            fused_qkv_a_proj_with_mqa_weight_kv_t, self.qk_rope_head_dim
        )
        fused_qkv_a_proj_with_mqa_weight_kv = (
            fused_qkv_a_proj_with_mqa_weight_kv_t.t().contiguous()
        )
        # cat nz
        fused_qkv_a_proj_with_mqa_weight_new = torch.cat(
            (fused_qkv_a_proj_with_mqa_weight_kv, fused_qkv_a_proj_with_mqa_weight_q),
            dim=-1,
        )
        fused_qkv_a_proj_with_mqa_weight = (
            fused_qkv_a_proj_with_mqa_weight_new.t().contiguous()
        )
        fused_qkv_a_proj_with_mqa_weight_nz = (
            transdata(fused_qkv_a_proj_with_mqa_weight, block_size=(16, 32))
            .unsqueeze(0)
            .contiguous()
        )
        self.qkv_a_proj_weight_nz = npu_format_cast(fused_qkv_a_proj_with_mqa_weight_nz)

        # matmul_0 deq_scale [2112]
        fused_qkv_a_proj_with_mqa_deq_scale_q = self.qkv_a_proj.deq_scale.data[
            : self.q_lora_rank
        ].clone()  # [7168, 1536]
        fused_qkv_a_proj_with_mqa_deq_scale_kv = self.qkv_a_proj.deq_scale.data[
            self.q_lora_rank :
        ].clone()  # [7168, 576]
        # rope fit
        fused_qkv_a_proj_with_mqa_deq_scale_kv = (
            fused_qkv_a_proj_with_mqa_deq_scale_kv.reshape(
                self.kv_lora_rank + self.qk_rope_head_dim, -1
            ).contiguous()
        )
        fused_qkv_a_proj_with_mqa_deq_scale_kv = trans_rope_weight(
            fused_qkv_a_proj_with_mqa_deq_scale_kv, self.qk_rope_head_dim
        )
        fused_qkv_a_proj_with_mqa_deq_scale_kv = (
            fused_qkv_a_proj_with_mqa_deq_scale_kv.view(
                self.kv_lora_rank + self.qk_rope_head_dim
            ).contiguous()
        )
        self.qkv_a_proj_deq_scale_kvq = torch.cat(
            (
                fused_qkv_a_proj_with_mqa_deq_scale_kv,
                fused_qkv_a_proj_with_mqa_deq_scale_q,
            ),
            dim=-1,
        )

        # matmul_0 quant_bias [2112]
        fused_qkv_a_proj_with_mqa_quant_bias_q = self.qkv_a_proj.quant_bias.data[
            : self.q_lora_rank
        ].clone()  # [7168, 1536]
        fused_qkv_a_proj_with_mqa_quant_bias_kv = self.qkv_a_proj.quant_bias.data[
            self.q_lora_rank :
        ].clone()  # [7168, 576]
        # rope fit
        fused_qkv_a_proj_with_mqa_quant_bias_kv = (
            fused_qkv_a_proj_with_mqa_quant_bias_kv.reshape(
                self.kv_lora_rank + self.qk_rope_head_dim, -1
            ).contiguous()
        )
        fused_qkv_a_proj_with_mqa_quant_bias_kv = trans_rope_weight(
            fused_qkv_a_proj_with_mqa_quant_bias_kv, self.qk_rope_head_dim
        )
        fused_qkv_a_proj_with_mqa_quant_bias_kv = (
            fused_qkv_a_proj_with_mqa_quant_bias_kv.view(
                self.kv_lora_rank + self.qk_rope_head_dim
            ).contiguous()
        )
        self.qkv_a_proj_quant_bias_kvq = torch.cat(
            (
                fused_qkv_a_proj_with_mqa_quant_bias_kv,
                fused_qkv_a_proj_with_mqa_quant_bias_q,
            ),
            dim=-1,
        )

        # matmul_1 weight [1536, num_head * 192]
        q_b_proj_weight = self.q_b_proj.weight.data.clone()
        q_b_proj_weight = q_b_proj_weight.t().reshape(
            self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1
        )
        q_b_proj_weight = trans_rope_weight(q_b_proj_weight, self.qk_rope_head_dim)
        q_b_proj_weight = q_b_proj_weight.reshape(
            self.num_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), -1
        )
        q_b_proj_weight_nz = (
            transdata(q_b_proj_weight, block_size=(16, 32)).unsqueeze(0).contiguous()
        )
        self.q_b_proj_weight_nz = npu_format_cast(q_b_proj_weight_nz)

        # matmul_1 deq_scale [num_head * 192]
        q_b_proj_deq_scale = self.q_b_proj.deq_scale.data.clone()
        q_b_proj_deq_scale = q_b_proj_deq_scale.reshape(
            self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1
        )
        q_b_proj_deq_scale = trans_rope_weight(
            q_b_proj_deq_scale, self.qk_rope_head_dim
        )
        self.q_b_proj_deq_scale = q_b_proj_deq_scale.reshape(
            self.num_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )

        # matmul_1 quant_bias [num_head * 192]
        q_b_proj_quant_bias = self.q_b_proj.quant_bias.data.clone()
        q_b_proj_quant_bias = q_b_proj_quant_bias.reshape(
            self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim, -1
        )
        q_b_proj_quant_bias = trans_rope_weight(
            q_b_proj_quant_bias, self.qk_rope_head_dim
        )
        self.q_b_proj_quant_bias = q_b_proj_quant_bias.reshape(
            self.num_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )

    def mlaprolog_preprocess_weight(self):
        q_b_proj_weight_source = vars(self.q_b_proj).get("mlaprolog_weight_source")
        q_b_proj_scale_source = vars(self.q_b_proj).get("mlaprolog_weight_scale_source")
        if (q_b_proj_weight_source is None) != (q_b_proj_scale_source is None):
            raise RuntimeError(
                "MLAProlog requires q_b_proj source weight and scale to be "
                "preserved together."
            )
        if q_b_proj_weight_source is not None:
            self.q_b_proj_weight = npu_format_cast(
                q_b_proj_weight_source.transpose(0, 1).contiguous()
            )
            self.q_b_proj_scale = q_b_proj_scale_source.contiguous().view(
                torch.float8_e8m0fnu
            )
        else:
            self.q_b_proj_weight = npu_format_cast(
                self.q_b_proj.weight.data.transpose(0, 1).contiguous()
            )

        qkv_weight_source = vars(self.qkv_a_proj).get("mlaprolog_weight_source")
        qkv_scale_source = vars(self.qkv_a_proj).get("mlaprolog_weight_scale_source")
        if (qkv_weight_source is None) != (qkv_scale_source is None):
            raise RuntimeError(
                "MLAProlog requires fused QKV-A source weight and scale to be "
                "preserved together."
            )

        qkv_weight = (
            qkv_weight_source
            if qkv_weight_source is not None
            else self.qkv_a_proj.weight
        )
        qkv_weight_t = qkv_weight.data.transpose(0, 1).contiguous()
        qkv_a_proj_weight_q = qkv_weight_t[:, : self.q_lora_rank].clone()
        qkv_a_proj_weight_kv = qkv_weight_t[:, self.q_lora_rank :].clone()
        self.q_a_proj_weight = npu_format_cast(qkv_a_proj_weight_q)
        self.kv_a_proj_weight = npu_format_cast(qkv_a_proj_weight_kv)

        if qkv_scale_source is not None:
            self.qkv_a_proj_scale_q = (
                qkv_scale_source[: self.q_lora_rank, :]
                .contiguous()
                .view(torch.float8_e8m0fnu)
            )
            self.qkv_a_proj_scale_kv = (
                qkv_scale_source[self.q_lora_rank :, :]
                .contiguous()
                .view(torch.float8_e8m0fnu)
            )

    def get_sin_cos(self, positions):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
        return cos, sin

    def get_sin_cos_from_master_cache(self, positions, dtype):
        """Read full-dimension LongCat RoPE caches when they are available."""

        pos = positions.flatten()
        cos_cached_total = self.rotary_emb._buffers.get("cos_cached_total")
        sin_cached_total = self.rotary_emb._buffers.get("sin_cached_total")
        if cos_cached_total is None:
            cos_cached_total = vars(self.rotary_emb).get("cos_cached_total")
        if sin_cached_total is None:
            sin_cached_total = vars(self.rotary_emb).get("sin_cached_total")
        if cos_cached_total is None or sin_cached_total is None:
            cos_sin = self.rotary_emb.cos_sin_cache.index_select(0, pos)
            cos, sin = cos_sin.chunk(2, dim=-1)
            cos = cos.repeat(1, 2)
            sin = sin.repeat(1, 2)
        else:
            cos = cos_cached_total.index_select(0, pos)
            sin = sin_cached_total.index_select(0, pos)

        return (
            cos.to(device=positions.device, dtype=dtype),
            sin.to(device=positions.device, dtype=dtype),
        )

    def refresh_mlaprolog_runtime_cache(self, positions, forward_batch, dtype):
        cos, sin = self.get_sin_cos_from_master_cache(positions, dtype)
        cache_index_i64 = forward_batch.out_cache_loc.to(dtype=torch.int64)
        forward_batch.npu_mlaprolog_runtime_cache = (
            cos,
            sin,
            cache_index_i64,
        )
        return forward_batch.npu_mlaprolog_runtime_cache

    def get_mlaprolog_runtime_cache(self, positions, forward_batch, dtype):
        if self.layer_id == 0 or forward_batch.npu_mlaprolog_runtime_cache is None:
            return self.refresh_mlaprolog_runtime_cache(
                positions,
                forward_batch,
                dtype,
            )
        return forward_batch.npu_mlaprolog_runtime_cache

    def get_kv_cache_and_cache_idx(self, forward_batch):
        k_cache, v_cache = get_token_to_kv_pool().get_kv_buffer(self.layer_id)
        slot_mapping = forward_batch.out_cache_loc.to(dtype=torch.int32)
        return k_cache, v_cache, slot_mapping

    def forward_absorb_prepare_npu_rms_norm_cache(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch,
        zero_allocator,
    ):
        bsz, _ = hidden_states.view(-1, hidden_states.shape[-1]).shape
        self.dtype = hidden_states.dtype
        if self.layer_id == 0:
            self.cos, self.sin = self.get_sin_cos(positions)
            self.rotary_emb.cos_cached, self.rotary_emb.sin_cache = self.cos, self.sin
        else:
            self.cos, self.sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cache

        self.kvCache, self.kvCacheRope, self.slotmapping = (
            self.get_kv_cache_and_cache_idx(forward_batch)
        )

        if not self.has_preprocess_weights:
            self.has_preprocess_weights = True

        cos, sin = self.cos, self.sin

        if self.q_lora_rank is not None:
            fused_qkv_a_proj_out = self.qkv_a_proj(hidden_states)[0]
            q_lowrank, latent_cache = fused_qkv_a_proj_out.split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q_lowrank)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )  # b*s,n,d

        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim)
        q_nope = torch.matmul(q_nope.transpose(0, 1), self.w_kc).transpose(0, 1)

        q_pe = q_pe.view(-1, self.num_local_heads, 1, self.qk_rope_head_dim)
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        q_pe = torch.ops.npu.npu_interleave_rope(q_pe, cos, sin)  # (B,N,S,D)
        q_pe = q_pe.view(cos.shape[0], self.num_local_heads, self.qk_rope_head_dim)

        latent_cache = latent_cache.view(
            -1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim
        )  # (B*S,N,1,D)

        cache_mode = "PA_NZ" if is_fia_nz() else "PA_BNSD"
        self.kvCache = self.kvCache.view(
            -1,
            get_attn_backend().page_size,
            1,
            get_attn_backend().kv_lora_rank,
        )
        self.kvCacheRope = self.kvCacheRope.view(
            -1,
            get_attn_backend().page_size,
            1,
            get_attn_backend().qk_rope_head_dim,
        )
        k_rope, k_nope, _, _ = torch.ops.npu.npu_kv_rmsnorm_rope_cache(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            self.slotmapping.to(torch.int64),
            self.kvCacheRope,
            self.kvCache,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode=cache_mode,
        )

        return (
            q_pe,
            k_rope,
            q_nope,
            k_nope,
            None,
            forward_batch,
            zero_allocator,
            positions,
            None,
            None,
        )

    def forward_mlapo(self, positions, hidden_states, forward_batch, zero_allocator):
        input_dtype = hidden_states.dtype
        if not self.has_preprocess_weights:
            self.preprocess_weights(hidden_states)
            self.has_preprocess_weights = True
            self.dtype = hidden_states.dtype

        if self.layer_id == 0:
            cos, sin = self.get_sin_cos(positions)
            self.rotary_emb.cos_cached, self.rotary_emb.sin_cache = cos, sin
        else:
            cos, sin = self.rotary_emb.cos_cached, self.rotary_emb.sin_cache

        k_cache, v_cache, slot_mapping = self.get_kv_cache_and_cache_idx(forward_batch)

        q_nope_out = torch.empty(
            (hidden_states.shape[0], self.w_kc.shape[0], k_cache.shape[-1]),
            dtype=input_dtype,
            device=hidden_states.device,
        )
        q_rope_out = torch.empty(
            (hidden_states.shape[0], self.w_kc.shape[0], v_cache.shape[-1]),
            dtype=input_dtype,
            device=hidden_states.device,
        )
        if is_fia_nz():
            kv_shape, kv_rope_shape = k_cache.shape, v_cache.shape
            num_blocks, block_size, num_heads, _ = kv_shape
            k_cache = k_cache.view(
                num_blocks, num_heads * self.kv_lora_rank // 16, block_size, 16
            )
            v_cache = v_cache.view(
                num_blocks, num_heads * self.qk_rope_head_dim // 16, block_size, 16
            )
        # TODO: dummy inputs to be removed
        # https://github.com/sgl-project/sgl-kernel-npu/issues/78
        if hasattr(self.q_a_layernorm, "bias"):
            q_a_layernorm_bias = self.q_a_layernorm.bias
        else:
            q_a_layernorm_bias = self.dummy

        torch.ops.npu.mla_preprocess(
            hidden_states,
            self.dummy,
            self.dummy,
            self.qkv_a_proj_weight_nz,
            self.qkv_a_proj_deq_scale_kvq,
            self.q_a_layernorm.weight,
            q_a_layernorm_bias,
            self.q_b_proj_weight_nz,
            self.q_b_proj_deq_scale,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            self.w_kc,
            k_cache,
            v_cache,
            slot_mapping,
            quant_scale0=self.qkv_a_proj.input_scale,
            quant_offset0=self.qkv_a_proj_input_offset,
            bias0=self.qkv_a_proj_quant_bias_kvq,
            quant_scale1=self.q_b_proj.input_scale,
            quant_offset1=self.q_b_proj_input_offset,
            bias1=self.q_b_proj_quant_bias,
            cache_mode="nzcache" if is_fia_nz() else "krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            q_out0=q_nope_out,
            kv_cache_out0=k_cache,
            q_out1=q_rope_out,
            kv_cache_out1=v_cache,
        )

        if is_fia_nz():
            k_cache = k_cache.view(kv_shape)
            v_cache = v_cache.view(kv_rope_shape)

        return (
            q_rope_out,
            v_cache,
            q_nope_out,
            k_cache,
            None,
            forward_batch,
            zero_allocator,
            positions,
            None,
            None,
        )

    def forward_mlaprolog(self, positions, hidden_states, forward_batch):
        import torch_npu

        if not self.has_preprocess_weights:
            self.mlaprolog_preprocess_weight()
            self.has_preprocess_weights = True

        self.cos, self.sin, cache_index_i64 = self.get_mlaprolog_runtime_cache(
            positions,
            forward_batch,
            hidden_states.dtype,
        )
        k_cache, v_cache, _ = self.get_kv_cache_and_cache_idx(forward_batch)

        qkv_weight_source = vars(self.qkv_a_proj).get("mlaprolog_weight_source")
        q_b_weight_source = vars(self.q_b_proj).get("mlaprolog_weight_source")
        is_mxfp8 = qkv_weight_source is not None and q_b_weight_source is not None

        token_to_kv_pool = get_token_to_kv_pool()
        dsa_fp8_packed_cache = token_to_kv_pool.dsa_kv_cache_store_fp8
        active_kv_cache_dtype = token_to_kv_pool.dtype
        kr_cache = v_cache
        packed_cache_args = {}

        if dsa_fp8_packed_cache:
            if not is_mxfp8:
                raise RuntimeError(
                    "DSA FP8 per-tile MLAProlog cache requires MXFP8 QKV-A "
                    "and Q-B weights."
                )
            if is_fia_nz():
                raise RuntimeError(
                    "DSA FP8 per-tile MLAProlog cache does not support PA_NZ."
                )

            packed_cache_dim = get_dsa_fp8_packed_cache_dim(
                kv_lora_rank=self.kv_lora_rank,
                qk_rope_head_dim=self.qk_rope_head_dim,
            )
            if k_cache.shape[-1] != packed_cache_dim:
                raise RuntimeError(
                    f"Unexpected DSA FP8 cache dim {k_cache.shape[-1]}, "
                    f"expected {packed_cache_dim}."
                )
            if k_cache.dtype == torch.uint8:
                k_cache = k_cache.view(torch.float8_e4m3fn)
            if k_cache.dtype != torch.float8_e4m3fn:
                raise RuntimeError(
                    f"Unexpected DSA FP8 cache dtype {k_cache.dtype}, expected "
                    f"{torch.float8_e4m3fn}."
                )

            cache_mode = "PA_BSND"
            k_cache = k_cache.view(
                -1,
                get_attn_backend().page_size,
                1,
                packed_cache_dim,
            )
            kr_cache = torch.empty(
                (1, 1, 1, 0),
                dtype=torch.bfloat16,
                device=k_cache.device,
            )
            kv_cache_quant_mode = 3
            query_quant_mode = 0
            quant_scale_ckv = None
            packed_cache_args = {
                "ckvkr_repo_mode": 1,
                "quant_scale_repo_mode": 1,
                "tile_size": DSA_KV_QUANT_TILE_SIZE,
            }
        elif active_kv_cache_dtype == torch.float8_e4m3fn:
            cache_mode = "PA_NZ" if is_fia_nz() else "PA_BSND"
            kv_cache_quant_mode = 1
            query_quant_mode = 1
            quant_scale_ckv = self._get_quant_scale_ckv(hidden_states.device)
        else:
            cache_mode = "PA_NZ" if is_fia_nz() else "PA_BSND"
            kv_cache_quant_mode = 0
            query_quant_mode = 0
            quant_scale_ckv = None

        if is_mxfp8:
            token_x, token_x_scale = _quantize_mla_query_for_mxfp8(hidden_states)
            mla_prolog_input_args = {
                "token_x": token_x,
                "weight_dq": self.q_a_proj_weight,
                "weight_uq_qr": self.q_b_proj_weight,
                "weight_uk": self.w_kc,
                "weight_dkv_kr": self.kv_a_proj_weight,
                "rmsnorm_gamma_cq": self.q_a_layernorm.weight,
                "rmsnorm_gamma_ckv": self.kv_a_layernorm.weight,
                "rope_sin": self.sin,
                "rope_cos": self.cos,
                "kv_cache": k_cache,
                "kr_cache": kr_cache,
                "cache_index": cache_index_i64,
                "rmsnorm_epsilon_cq": self.q_a_layernorm.variance_epsilon,
                "rmsnorm_epsilon_ckv": self.kv_a_layernorm.variance_epsilon,
                "cache_mode": cache_mode,
                "query_norm_flag": True,
                "dequant_scale_w_dq": self.qkv_a_proj_scale_q,
                "dequant_scale_w_uq_qr": self.q_b_proj_scale,
                "dequant_scale_w_dkv_kr": self.qkv_a_proj_scale_kv,
                "weight_quant_mode": 3,
                "dequant_scale_x": token_x_scale.view(torch.float8_e8m0fnu),
                "kv_cache_quant_mode": kv_cache_quant_mode,
                "query_quant_mode": query_quant_mode,
                "kc_scale": 1.0,
                "qc_qr_scale": 1.0,
                "quant_scale_ckv": quant_scale_ckv,
            }
            mla_prolog_input_args.update(packed_cache_args)
            q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm = (
                torch_npu.npu_mla_prolog_v3(**mla_prolog_input_args)
            )
            return (
                q_pe,
                kr_cache,
                q_nope,
                k_cache,
                qr,
                forward_batch,
                None,
                positions,
                dequant_q_norm,
                dequant_scale_q_nope,
            )

        if active_kv_cache_dtype == torch.float8_e4m3fn:
            raise RuntimeError(
                "FP8 MLA KV cache requires the MXFP8 MLAProlog weight path; "
                "use BF16 draft KV cache for an unquantized NextN layer."
            )

        is_q_b_proj_quantized = self.q_b_proj_weight_scale is not None
        mla_prolog_input_args = {
            "token_x": hidden_states,
            "weight_dq": self.q_a_proj_weight,
            "weight_uq_qr": (
                self.q_b_proj.weight if is_q_b_proj_quantized else self.q_b_proj_weight
            ),
            "weight_uk": self.w_kc,
            "weight_dkv_kr": self.kv_a_proj_weight,
            "rmsnorm_gamma_cq": self.q_a_layernorm.weight,
            "rmsnorm_gamma_ckv": self.kv_a_layernorm.weight,
            "rope_sin": self.sin,
            "rope_cos": self.cos,
            "kv_cache": k_cache,
            "kr_cache": v_cache,
            "cache_index": cache_index_i64,
            "rmsnorm_epsilon_cq": self.q_a_layernorm.variance_epsilon,
            "rmsnorm_epsilon_ckv": self.kv_a_layernorm.variance_epsilon,
            "cache_mode": cache_mode,
            "query_norm_flag": True,
            "weight_quant_mode": 1 if is_q_b_proj_quantized else 0,
            # Non-MX target and BF16 draft layers keep the documented
            # non-quantized cache/query MLAProlog contract.
            "kv_cache_quant_mode": 0,
            "query_quant_mode": 0,
        }
        if is_q_b_proj_quantized:
            mla_prolog_input_args["dequant_scale_w_uq_qr"] = self.q_b_proj_weight_scale
        q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm = (
            torch_npu.npu_mla_prolog_v3(**mla_prolog_input_args)
        )
        if (
            is_q_b_proj_quantized
            and dequant_q_norm is not None
            and dequant_q_norm.numel() > 0
        ):
            dequant_q_norm = dequant_q_norm.view(hidden_states.shape[0])
        else:
            dequant_q_norm = None
        return (
            q_pe,
            v_cache,
            q_nope,
            k_cache,
            qr,
            forward_batch,
            None,
            positions,
            dequant_q_norm,
            dequant_scale_q_nope,
        )

    def forward(self, positions, hidden_states, forward_batch, zero_allocator):
        if self._uses_legacy_mlapo():
            return self.forward_mlapo(
                positions, hidden_states, forward_batch, zero_allocator
            )
        if self.uses_mlaprolog():
            return self.forward_mlaprolog(positions, hidden_states, forward_batch)
        return self.forward_absorb_prepare_npu_rms_norm_cache(
            positions, hidden_states, forward_batch, zero_allocator
        )
