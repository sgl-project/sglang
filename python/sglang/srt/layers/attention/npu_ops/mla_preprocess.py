import torch
import torch.nn.functional as F

from sglang.srt.utils import get_bool_env_var, is_npu

_is_npu = is_npu()
_ENABLE_MLA_PREPROCESS_FLAG = get_bool_env_var("SGLANG_NPU_USE_MLAPO")
_NPU_FORMAT_NZ = 29


def is_mla_preprocess_enabled() -> bool:
    return _is_npu and _ENABLE_MLA_PREPROCESS_FLAG


if is_mla_preprocess_enabled():
    import sgl_kernel_npu
    import torch_npu

    torch.npu.config.allow_internal_format = True
    torch.npu.set_compile_mode(jit_compile=False)


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
    ):
        super().__init__()
        self.qkv_a_proj = fused_qkv_a_proj_with_mqa
        self.q_a_layernorm = q_a_layernorm
        self.kv_a_layernorm = kv_a_layernorm
        self.q_b_proj = q_b_proj
        self.w_kc = w_kc.contiguous()
        self.rotary_emb = rotary_emb
        self.layer_id = layer_id
        self.has_preprocess_weights = False

        self.q_lora_rank = self.q_b_proj.input_size  # 1536
        self.kv_lora_rank = self.kv_a_layernorm.hidden_size  # 512
        self.num_local_heads = num_local_heads  # tp
        self.qk_nope_head_dim = qk_nope_head_dim  # 128
        self.qk_rope_head_dim = qk_rope_head_dim  # 64

    def preprocess_weights(self, hidden_states):
        self.dummy = torch.empty(
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
        self.qkv_a_proj_weight_nz = torch_npu.npu_format_cast(
            fused_qkv_a_proj_with_mqa_weight_nz, _NPU_FORMAT_NZ
        )

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
        self.q_b_proj_weight_nz = torch_npu.npu_format_cast(
            q_b_proj_weight_nz, _NPU_FORMAT_NZ
        )

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

    def get_sin_cos(self, positions):
        cos_sin = self.rotary_emb.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
        return cos, sin

    def get_kv_cache_and_cache_idx(self, forward_batch):
        k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(self.layer_id)
        slot_mapping = forward_batch.out_cache_loc.to(dtype=torch.int32)
        return k_cache, v_cache, slot_mapping

    def forward(self, positions, hidden_states, forward_batch, zero_allocator):
        input_dtype = hidden_states.dtype
        if not self.has_preprocess_weights:
            self.preprocess_weights(hidden_states)
            self.has_preprocess_weights = True
            self.dtype = hidden_states.dtype

        cos, sin = self.get_sin_cos(positions)
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

        # TODO: dummy inputs to be removed
        # https://github.com/sgl-project/sgl-kernel-npu/issues/78
        torch.ops.npu.mla_preprocess(
            hidden_states,
            self.dummy,
            self.dummy,
            self.qkv_a_proj_weight_nz,
            self.qkv_a_proj_deq_scale_kvq,
            self.q_a_layernorm.weight,
            self.q_a_layernorm.bias,
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
            cache_mode="krope_ctkv",
            quant_mode="per_tensor_quant_asymm",
            q_out0=q_nope_out,
            kv_cache_out0=k_cache,
            q_out1=q_rope_out,
            kv_cache_out1=v_cache,
        )
        return (
            q_rope_out,
            v_cache,
            q_nope_out,
            k_cache,
            forward_batch,
            zero_allocator,
            positions,
        )
