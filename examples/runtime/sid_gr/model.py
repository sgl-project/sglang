"""Qwen3 reference arithmetic with SGLang projections, weights and execution."""

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layer_boundary import append_stages
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.models.qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3Model,
)
from sglang.srt.runtime_context import get_parallel


class SIDFullReductionLinear(ColumnParallelLinear):
    """Output-sharded GEMM with the full SID dot product on each rank.

    Native row TP rounds partial dot products before reducing them. Instead,
    gather input shards and compute disjoint output rows. Zero-fill the other
    rows so the existing stage-boundary all-reduce assembles complete outputs
    without adding nonzero partials. Real weights and outputs stay sharded;
    input all-gathers and padded GEMMs are reference-path overhead.
    """

    sid_output_partition = True
    reduce_results = False

    def __init__(self, native):
        if native.bias is not None or native.reduce_results:
            raise ValueError(
                "SID full-reduction projection requires bias-free deferred reduction"
            )
        super().__init__(
            native.input_size,
            native.output_size,
            bias=False,
            gather_output=False,
            params_dtype=native.params_dtype,
            parallel_group=native.parallel_group,
        )

    def forward(self, input_, forward_batch=None):
        full_input = self.tp_group.all_gather(input_.contiguous(), dim=-1)
        local_rows = self.weight.shape[0]
        # The pinned L40S/cuBLAS BF16 decode GEMM changes accumulation when
        # N shrinks to 256 at TP=4. Zero rows retain the verified >=512-row
        # dispatch; discard them before assembling the disjoint real outputs.
        # This is an explicit reference-path compute cost, not a TP speedup.
        gemm_rows = max(local_rows, min(512, self.output_size))
        weight = self.weight
        if gemm_rows > local_rows:
            weight = F.pad(weight, (0, 0, 0, gemm_rows - local_rows))
        local_output = F.linear(full_input, weight)[..., :local_rows]
        output = local_output.new_zeros((*local_output.shape[:-1], self.output_size))
        start = self.tp_group.rank_in_group * local_output.shape[-1]
        output[..., start : start + local_output.shape[-1]] = local_output
        return output, None


class SIDLogitsProcessor(LogitsProcessor):
    def _compute_lm_head(
        self, hidden_states, lm_head, embedding_bias=None, *, out=None
    ):
        if get_parallel().tp_size == 1:
            return super()._compute_lm_head(
                hidden_states, lm_head, embedding_bias, out=out
            )
        if embedding_bias is not None:
            raise ValueError("SID reference requires a bias-free LM head")
        # Keep the full vocabulary GEMM geometry on every rank. A 128-row
        # minimum was enough for W16/128/256, but W32 selects different BF16
        # accumulation at N128 versus N256. Discard padded output rows before
        # native vocabulary gathering, normalization and beam selection.
        weight = lm_head.weight
        local_rows = weight.shape[0]
        gemm_rows = max(local_rows, self.vocab_size)
        if gemm_rows > local_rows:
            weight = F.pad(weight, (0, 0, 0, gemm_rows - local_rows))
        logits = F.linear(hidden_states.to(weight.dtype), weight)
        logits = logits[..., :local_rows].contiguous()
        # Preserve BF16 rounding even when upstream offers an FP32 buffer.
        if out is not None and out.shape == logits.shape and out.dtype == logits.dtype:
            out.copy_(logits)
            return out
        return logits


class SIDRMSNorm(RMSNorm):
    def forward(self, x, residual=None, post_residual_addition=None):
        if post_residual_addition is not None:
            raise ValueError("SID reference does not support extra residual additions")
        if residual is None:
            return F.rms_norm(x, (x.shape[-1],), self.weight, self.variance_epsilon)
        summed = x + residual  # Materialize BF16 rounding before normalization.
        return (
            F.rms_norm(summed, (summed.shape[-1],), self.weight, self.variance_epsilon),
            summed,
        )


class SIDRotaryEmbedding(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.head_dim = native.head_size
        self.register_buffer("cos_sin_cache", native.cos_sin_cache, persistent=False)

    def forward(self, positions, q, k, offsets=None):
        if offsets is not None:
            positions = positions + offsets
        cos, sin = self.cos_sin_cache[positions].chunk(2, -1)
        half = self.head_dim // 2

        def rotate(x):
            shaped = x.float().reshape(len(positions), -1, self.head_dim)
            left, right = shaped[..., :half], shaped[..., half:]
            # Separate operations match the SID Torch preset's FP32 rounding.
            result = torch.cat(
                (
                    left * cos[:, None] - right * sin[:, None],
                    left * sin[:, None] + right * cos[:, None],
                ),
                dim=-1,
            )
            return result.to(x.dtype).reshape_as(x)

        return rotate(q), rotate(k)


class SIDSiLUAndMul(nn.Module):
    def forward(self, x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up  # BF16 SiLU result before multiplication.


class SIDQwen3Attention(Qwen3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_norm = SIDRMSNorm(self.head_dim, eps=kwargs["rms_norm_eps"])
        self.k_norm = SIDRMSNorm(self.head_dim, eps=kwargs["rms_norm_eps"])
        self.rotary_emb = SIDRotaryEmbedding(self.rotary_emb)
        if get_parallel().tp_size > 1:
            self.o_proj = SIDFullReductionLinear(self.o_proj)

    def forward_prepare_native(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape_as(k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v


class SIDQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        layer_id=0,
        start_layer=0,
        quant_config=None,
        prefix="",
        alt_stream=None,
    ):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self._decode_quant_formats = None
        rope = getattr(config, "rope_parameters", None)
        theta = (rope or {}).get("rope_theta", getattr(config, "rope_theta", 1000000))
        scaling = rope or getattr(config, "rope_scaling", None)
        if scaling and scaling.get("rope_type", "default") != "default":
            raise ValueError("SID reference requires ordinary Qwen3 RoPE")
        self.self_attn = SIDQwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            start_layer=start_layer,
            rope_theta=theta,
            rope_scaling=scaling,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            prefix=add_prefix("mlp", prefix),
            reduce_results=False,
        )
        self.mlp.act_fn = SIDSiLUAndMul()
        if get_parallel().tp_size > 1:
            self.mlp.down_proj = SIDFullReductionLinear(self.mlp.down_proj)
        # Preserve the explicit Torch SiLU rounding on every supported GPU.
        self.mlp.dual_gemm.mode = None
        self.mlp.dual_gemm.down_proj = self.mlp.down_proj
        self.input_layernorm = SIDRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SIDRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        attn, ffn = self.stage_facts(config, layer_id)
        self.attn_boundary, self.ffn_boundary = append_stages(
            (attn, self.input_layernorm), (ffn, self.post_attention_layernorm)
        )


class SIDQwen3Model(Qwen3Model):
    def __init__(self, config, prefix=""):
        Qwen2Model.__init__(
            self, config, prefix=prefix, decoder_layer_type=SIDQwen3DecoderLayer
        )
        self.norm = SIDRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class SIDQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config, quant_config=None, prefix=""):
        nn.Module.__init__(self)
        parallel = get_parallel()
        if (
            quant_config is not None
            or parallel.tp_size not in (1, 2, 4)
            or parallel.pp_size != 1
            or parallel.attn_tp_size != parallel.tp_size
        ):
            raise ValueError(
                "SID reference requires unquantized TP=1/2/4, PP=1 and attention TP=TP"
            )
        if (
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        ) != (16, 4, 64):
            raise ValueError("SID reference is qualified for Q=16, KV=4, head_dim=64")
        if parallel.tp_size > 1 and (
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size,
        ) != (1024, 1024, 256):
            raise ValueError(
                "SID TP reference GEMM geometry is qualified for "
                "hidden_size=1024, intermediate_size=1024, vocab_size=256"
            )
        if config.hidden_act != "silu" or getattr(config, "sliding_window", None):
            raise ValueError("SID reference requires SiLU and full attention")
        if getattr(config, "model_type", None) != "qwen3":
            raise ValueError("SID reference currently supports dense Qwen3 only")
        self.pp_group = parallel.pp_group
        self.config, self.quant_config = config, quant_config
        self.model = SIDQwen3Model(config, prefix=add_prefix("model", prefix))
        self.lm_head = (
            self.model.embed_tokens
            if config.tie_word_embeddings
            else ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("lm_head", prefix),
            )
        )
        self.logits_processor = SIDLogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.capture_aux_hidden_states = False


EntryClass = SIDQwen3ForCausalLM
