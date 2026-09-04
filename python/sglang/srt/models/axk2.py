"""Native SGLang execution path for SKT A.X-K2.

AXK2 is a DeepSeek-v3.2 MLA/DSA MoE model, but its released checkpoints add
two non-optional operations: low-rank gated RMSNorm and a fused query/output
gate.  This module reuses SGLang's optimized MLA/DSA and MoE path while
inserting those checkpoint-compatible operations.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV32ForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix


class _RecordingRMSNorm(RMSNorm):
    """RMSNorm retaining the pre-normalized q-LoRA latent for AXK2's gate."""

    def forward(self, x, *args, **kwargs):
        self.axk2_pre_norm = x
        return super().forward(x, *args, **kwargs)


class AXK2GatedRMSNorm(torch.nn.Module):
    """RMSNorm followed by ``sigmoid(W_up(silu(W_down(y))))``."""

    def __init__(self, hidden_size, eps, rank, prefix):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=eps)
        # Rank-16 gates remain BF16 in the released checkpoint.
        self.W_down = ReplicatedLinear(
            hidden_size, rank, bias=False, quant_config=None, prefix=f"{prefix}.W_down"
        )
        self.W_up = ReplicatedLinear(
            rank, hidden_size, bias=False, quant_config=None, prefix=f"{prefix}.W_up"
        )

    def _gate(self, y):
        down, _ = self.W_down(y)
        up, _ = self.W_up(F.silu(down.float()).to(y.dtype))
        return (y.float() * torch.sigmoid(up.float())).to(y.dtype)

    def forward(self, x, residual=None, *args, **kwargs):
        result = self.norm(x, residual, *args, **kwargs)
        if residual is None:
            return self._gate(result)
        y, new_residual = result
        return self._gate(y), new_residual


class _GatedOProj(RowParallelLinear):
    """Applies AXK2's per-head gate immediately before ``o_proj``."""

    axk2_gate = None

    def forward(self, x):
        gate = self.axk2_gate
        if gate is not None:
            x = (x.float() * torch.sigmoid(gate.float())).to(x.dtype)
        return super().forward(x)


class AXK2Attention(DeepseekV2AttentionMLA):
    """DeepSeek MLA with the fused AXK2 ``q_b_proj`` query/output gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = kwargs["config"]
        prefix = kwargs["prefix"]
        quant_config = kwargs.get("quant_config")
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.q_a_layernorm = _RecordingRMSNorm(
            self.q_lora_rank, eps=config.rms_norm_eps
        )
        # Checkpoint layout: [q_lora post-norm | q_lora pre-norm] ->
        # per-head [query | output-gate]. It must remain fused: its FP8 block
        # scales cross the q/gate boundary.
        self.q_b_proj = ColumnParallelLinear(
            2 * self.q_lora_rank,
            self.num_heads * (self.qk_head_dim + self.v_head_dim),
            bias=False,
            quant_config=self._get_q_b_proj_quant_config(quant_config),
            prefix=add_prefix("q_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.o_proj = _GatedOProj(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

    def q_b_proj_forward(self, q_lora):
        q_pre_norm = self.q_a_layernorm.axk2_pre_norm
        q_and_gate = self.q_b_proj(torch.cat((q_lora, q_pre_norm), dim=-1))[0]
        q_and_gate = q_and_gate.view(
            -1, self.num_local_heads, self.qk_head_dim + self.v_head_dim
        )
        query, gate = torch.split(
            q_and_gate, (self.qk_head_dim, self.v_head_dim), dim=-1
        )
        self.o_proj.axk2_gate = gate.reshape(gate.shape[0], -1)
        return query


class AXK2ForCausalLM(DeepseekV32ForCausalLM):
    """A.X-K2 inference model using SGLang's MLA/DSA and MoE kernels."""

    fused_shared_experts_architecture = "AXK2ForCausalLM"

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__(config, quant_config, prefix)
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]
            old_attn = layer.self_attn
            attn_prefix = add_prefix("self_attn", f"model.layers.{layer_id}")
            layer.self_attn = AXK2Attention(
                config=config,
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=config.q_lora_rank,
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=old_attn.rope_theta,
                rope_scaling=config.rope_parameters,
                max_position_embeddings=config.max_position_embeddings,
                quant_config=quant_config,
                layer_id=layer_id,
                reduce_results=False,
                prefix=attn_prefix,
                alt_stream=self.model.alt_stream,
                dsa_enable_prefill_cp=self.model.dsa_enable_prefill_cp,
                mla_enable_prefill_cp=self.model.mla_enable_prefill_cp,
            )
            layer.input_layernorm = AXK2GatedRMSNorm(
                config.hidden_size,
                config.rms_norm_eps,
                config.gated_norm_rank,
                add_prefix("input_layernorm", f"model.layers.{layer_id}"),
            )
            if layer.is_layer_sparse:
                layer.post_attention_layernorm = AXK2GatedRMSNorm(
                    config.hidden_size,
                    config.rms_norm_eps,
                    config.gated_norm_rank,
                    add_prefix("post_attention_layernorm", f"model.layers.{layer_id}"),
                )
            # The communicator caches these module references during layer construction.
            layer.layer_communicator.input_layernorm = layer.input_layernorm
            layer.layer_communicator.post_attention_layernorm = (
                layer.post_attention_layernorm
            )
            layer.layer_communicator.qkv_latent_func = (
                layer.self_attn.prepare_qkv_latent
            )


EntryClass = AXK2ForCausalLM
