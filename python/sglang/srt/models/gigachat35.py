"""GigaChat 3.5 model.

GigaChat 3.5 is a DeepSeek-V3-style model (MLA attention + DeepSeek MoE) with a
*hybrid* attention stack: most layers use a Qwen3-Next Gated-Delta-Net (GDN)
linear-attention block, while a periodic subset use full MLA attention. On top of
the DeepSeek base it adds:

  * gated RMSNorm (optionally zero-centered) with a low-rank gating bottleneck,
    used as a 4-norm sandwich (pre/post attention, pre/post MLP);
  * gated attention (a sigmoid gate applied to the attention output before
    ``o_proj``);
  * an MLA query/key scaling factor (``alpha_q`` / ``alpha_kv``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.srt.configs.gigachat35 import GigaChat35Config
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layer_boundary.output import OutputTransform
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.qwen3_next import Qwen3GatedDeltaNet
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import BumpAllocator, add_prefix, make_pp_layers

_GATED_NORM_LOW_RANK = 16


class GigaChat35GatedRMSNorm(nn.Module):
    """RMSNorm (optionally zero-centered) followed by a low-rank sigmoid gate.

    ``scale`` folds the optional MLA ``alpha`` factor into the q/kv a-norms.
    Supports the fused residual-add convention used by sglang decoder layers:
    ``forward(x, residual)`` returns ``(normed, residual + x)``.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        layernorm_gating_weight: float = 2.0,
        zero_centered: bool = True,
        scale: float = 1.0,
        low_rank: int = _GATED_NORM_LOW_RANK,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.zero_centered = zero_centered
        self.layernorm_gating_weight = layernorm_gating_weight
        self.scale = scale
        self.r = low_rank
        self.weight = nn.Parameter(
            torch.zeros(hidden_size) if zero_centered else torch.ones(hidden_size)
        )
        self.gate_up_lowrank = nn.Linear(hidden_size, self.r, bias=False)
        self.gate_down_lowrank = nn.Linear(self.r, hidden_size, bias=False)

    def _norm_gate(self, x: torch.Tensor) -> torch.Tensor:
        out = x.float()
        out = out * torch.rsqrt(
            out.pow(2).mean(-1, keepdim=True) + self.variance_epsilon
        )
        if self.zero_centered:
            out = (out * (1.0 + self.weight.float())).to(x.dtype)
        else:
            out = (out * self.weight.float()).to(x.dtype)
        gate = F.linear(out, self.gate_up_lowrank.weight.to(out.dtype))
        gate = F.silu(gate.float()).to(out.dtype)
        gate = F.linear(gate, self.gate_down_lowrank.weight.to(out.dtype))
        gate = torch.sigmoid(gate.float()).to(out.dtype)
        out = (out * self.layernorm_gating_weight * gate).to(x.dtype)
        if self.scale != 1.0:
            out = (out.float() * self.scale).to(x.dtype)
        return out

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        if post_residual_addition is not None:
            x = x + post_residual_addition
        if residual is not None:
            x = x + residual
            residual = x
        out = self._norm_gate(x)
        return out if residual is None else (out, residual)


def build_norm(
    config: GigaChat35Config, hidden_size: int, scale: float = 1.0
) -> nn.Module:
    """Construct the configured norm for a GigaChat 3.5 module."""
    norm_type = getattr(config, "norm_type", "LlamaRMSNorm")
    eps = config.rms_norm_eps
    if norm_type == "LlamaRMSNorm":
        return RMSNorm(hidden_size, eps=eps)
    if norm_type == "ZeroCenteredRMSNorm":
        return GemmaRMSNorm(hidden_size, eps=eps)
    if norm_type in ("ZeroCenteredGatedNorm", "GatedNorm"):
        return GigaChat35GatedRMSNorm(
            hidden_size,
            eps=eps,
            layernorm_gating_weight=getattr(config, "layernorm_gating_weight", 2.0),
            zero_centered=(norm_type == "ZeroCenteredGatedNorm"),
            scale=scale,
        )
    raise ValueError(f"Unsupported norm_type for GigaChat 3.5: {norm_type!r}")


class GigaChat35PassthroughNorm(nn.Module):
    """Identity norm with the norm calling convention: ``forward(x)`` returns
    ``x``, ``forward(x, residual)`` returns ``(residual + x, residual + x)``.

    Used as the ``input_layernorm`` slot when a layer has no pre-norm
    (``layernorm_type="post"``); it only folds the residual so the
    attention boundary receives the folded residual.
    """

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        if post_residual_addition is not None:
            x = x + post_residual_addition
        if residual is None:
            return x
        merged = x + residual
        return merged, merged


class GigaChat35MlpPrepNorm(nn.Module):
    """Pre-MLP norm wrapper: adds the residual, then runs the optional pre-MLP
    ``post_attention_layernorm`` on the sum, unfused. The attention's own
    ``post_self_attn_layernorm`` is its declared output transform, which the
    FFN input runs before this.
    """

    def __init__(self, pre_layernorm: Optional[nn.Module]) -> None:
        super().__init__()
        self.pre_layernorm = pre_layernorm

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        post_residual_addition: Optional[torch.Tensor] = None,
    ):
        if post_residual_addition is not None:
            x = x + post_residual_addition
        if residual is None:
            return x if self.pre_layernorm is None else self.pre_layernorm(x)
        merged = x + residual
        if self.pre_layernorm is None:
            return merged, merged
        return self.pre_layernorm(merged), merged


_GIGACHAT_WEIGHT_NAME_REMAP = (
    (".gate_up_projection.", ".gate_up_lowrank."),
    (".gate_down_projection.", ".gate_down_lowrank."),
    (".self_attn.gate_proj.", ".self_attn.attn_gate."),
)


def _remap_gigachat_weight_names(weights):
    for name, loaded_weight in weights:
        for src, dst in _GIGACHAT_WEIGHT_NAME_REMAP:
            if src in name:
                name = name.replace(src, dst)
        yield name, loaded_weight


class GigaChat35GatedDeltaNet(Qwen3GatedDeltaNet):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            alt_stream=alt_stream,
            prefix=prefix,
        )
        scale = float(getattr(config, "linear_sigmoid_gate_scale", 1.0))
        zero_centered = (
            "zero_centered" in str(getattr(config, "linear_gating_type", "")).lower()
        )
        if (scale != 1.0 or zero_centered) and getattr(
            self.norm, "weight", None
        ) is not None:

            def _fold_loader(param, loaded_weight, _scale=scale, _zc=zero_centered):
                w = loaded_weight.to(param.dtype)
                if _zc:
                    w = 1.0 + w
                if _scale != 1.0:
                    w = _scale * w
                param.data.copy_(w)

            self.norm.weight.weight_loader = _fold_loader


class GigaChat35AttentionMLA(deepseek_v2.DeepseekV2AttentionMLA):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: Optional[int] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__(
            config=config,
            hidden_size=hidden_size,
            num_heads=num_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            reduce_results=reduce_results,
            layer_id=layer_id,
            prefix=prefix,
            alt_stream=alt_stream,
        )

        self._norm_type = getattr(config, "norm_type", "LlamaRMSNorm")
        use_scaling = bool(getattr(config, "use_mla_scaling_factor", False))
        q_hidden = hidden_size if q_lora_rank is None else q_lora_rank
        kv_hidden = kv_lora_rank
        alpha_q = (hidden_size / q_hidden) ** 0.5 if use_scaling else 1.0
        alpha_kv = (hidden_size / kv_hidden) ** 0.5 if use_scaling else 1.0

        if self._norm_type != "LlamaRMSNorm":
            if q_lora_rank is not None and hasattr(self, "q_a_layernorm"):
                self.q_a_layernorm = build_norm(config, q_lora_rank, scale=alpha_q)
            self.kv_a_layernorm = build_norm(config, kv_lora_rank, scale=alpha_kv)
        else:
            if use_scaling:
                if q_lora_rank is not None and hasattr(self, "q_a_layernorm"):
                    with torch.no_grad():
                        self.q_a_layernorm.weight.mul_(alpha_q)
                with torch.no_grad():
                    self.kv_a_layernorm.weight.mul_(alpha_kv)

        self.gated_attention = bool(getattr(config, "gated_attention", False))
        self._gate_input: Optional[torch.Tensor] = None
        if self.gated_attention:
            self.attn_gate = ColumnParallelLinear(
                hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("attn_gate", prefix),
                tp_rank=get_parallel().attn_tp_rank,
                tp_size=get_parallel().attn_tp_size,
            )
            self.o_proj.register_forward_pre_hook(self._o_proj_gate_hook)

    def _o_proj_gate_hook(self, module, args):
        if not self.gated_attention or self._gate_input is None:
            return None
        attn_output = args[0]
        if not isinstance(attn_output, torch.Tensor) or not isinstance(
            self._gate_input, torch.Tensor
        ):
            return None
        gate, _ = self.attn_gate(self._gate_input)
        attn_output = attn_output * torch.sigmoid(gate)
        return (attn_output, *args[1:])

    def dispatch_attn_forward_method(self, forward_batch: ForwardBatch):
        method = super().dispatch_attn_forward_method(forward_batch)
        AF = deepseek_v2.AttnForwardMethod
        fused = [
            getattr(AF, name, None)
            for name in ("MLA_FUSED_ROPE", "MLA_FUSED_ROPE_ROCM", "MLA_FUSED_ROPE_CPU")
        ]
        if method in [m for m in fused if m is not None]:
            return AF.MLA
        return method

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        **kwargs,
    ):
        if self.gated_attention:
            self._gate_input = (
                hidden_states if isinstance(hidden_states, torch.Tensor) else None
            )
        try:
            return super().forward(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                **kwargs,
            )
        finally:
            self._gate_input = None


class GigaChat35DecoderLayer(deepseek_v2.DeepseekV2DecoderLayer):
    def __init__(
        self,
        config: GigaChat35Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            is_nextn=is_nextn,
            prefix=prefix,
            alt_stream=alt_stream,
            build_stages=False,
        )

        attn_layer_id = config.num_hidden_layers if is_nextn else layer_id
        self.use_linear_attn = config.is_linear_attention_layer(attn_layer_id)

        if self.use_linear_attn:
            self.self_attn = GigaChat35GatedDeltaNet(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("self_attn", prefix),
            )
            if hasattr(self.self_attn, "out_proj"):
                self.self_attn.out_proj.reduce_results = False
            qkv_latent_func = None
        else:
            self.self_attn = GigaChat35AttentionMLA(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                v_head_dim=config.v_head_dim,
                q_lora_rank=getattr(config, "q_lora_rank", None),
                kv_lora_rank=config.kv_lora_rank,
                rope_theta=config.rope_theta,
                rope_scaling=config.rope_scaling,
                max_position_embeddings=config.max_position_embeddings,
                quant_config=quant_config,
                reduce_results=False,
                layer_id=layer_id,
                prefix=add_prefix("self_attn", prefix),
                alt_stream=alt_stream,
            )
            qkv_latent_func = self.self_attn.prepare_qkv_latent

        self.is_sparse = self.is_layer_sparse

        swiglu_limit = float(getattr(config, "swiglu_limit", None) or 0.0)
        experts = getattr(self.mlp, "experts", None)
        runner_config = getattr(experts, "moe_runner_config", None)
        if swiglu_limit > 0 and runner_config is not None:
            runner_config.gemm1_clamp_limit = swiglu_limit

        layernorm_type = getattr(config, "layernorm_type", "pre")
        self._use_pre = layernorm_type in ("pre", "pre_post")
        self._use_post = layernorm_type in ("post", "pre_post")

        self.input_layernorm = build_norm(config, config.hidden_size)
        self.post_attention_layernorm = build_norm(config, config.hidden_size)
        self.post_self_attn_layernorm = (
            build_norm(config, config.hidden_size) if self._use_post else None
        )
        self.post_feedforward_layernorm = (
            build_norm(config, config.hidden_size) if self._use_post else None
        )

        attn_prepare_layernorm = (
            self.input_layernorm if self._use_pre else GigaChat35PassthroughNorm()
        )
        mlp_prepare_layernorm = GigaChat35MlpPrepNorm(
            pre_layernorm=self.post_attention_layernorm if self._use_pre else None,
        )
        # The stage boundaries choose their fused steps from their norms when built.
        self.attn_boundary, self.ffn_boundary = self._build_stages(
            input_layernorm=attn_prepare_layernorm,
            post_attention_layernorm=mlp_prepare_layernorm,
            qkv_latent_func=qkv_latent_func,
            attn_output=(
                OutputTransform(self.post_self_attn_layernorm)
                if self.post_self_attn_layernorm is not None
                else None
            ),
            output=(
                OutputTransform(self.post_feedforward_layernorm)
                if self.post_feedforward_layernorm is not None
                else None
            ),
        )

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        if is_nextn:
            return bool(getattr(self.config, "nextn_is_sparse", False))
        return (
            self.config.n_routed_experts is not None
            and layer_id >= self.config.first_k_dense_replace
            and layer_id % self.config.moe_layer_freq == 0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)

        if self.use_linear_attn:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                input_on_attn_tp_slices=(self.attn_boundary.input_on_attn_tp_slices),
            )
        get_attn_tp_context().clear_attn_inputs()

        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)

        # Unlike deepseek_v2, no moe_output_buffer_ctx here: non-inplace MoE
        # runners then allocate their output per forward instead of recycling
        # the layer-input buffer. The default (inplace) runners are unaffected.
        with self.ffn_boundary.exit(forward_batch) as ffn_exit:
            hidden_states = self.mlp(hidden_states, forward_batch)

        hidden_states = ffn_exit.finish(hidden_states)
        return hidden_states


class GigaChat35Model(nn.Module):
    def __init__(
        self,
        config: GigaChat35Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        self.pp_group = get_parallel().pp_group

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = None

        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

        self.layers, self.start_layer, self.end_layer = make_pp_layers(
            config.num_hidden_layers,
            lambda idx, prefix: GigaChat35DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = build_norm(config, config.hidden_size)
        else:
            self.norm = None

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        total_num_layers = self.end_layer - self.start_layer
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[i](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
            )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )
        return hidden_states


class GigaChat35ForCausalLM(DeepseekV2WeightLoaderMixin, nn.Module):
    def __init__(
        self,
        config: GigaChat35Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        self.num_fused_shared_experts = 0

        self.model = GigaChat35Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_parallel().enable_dp_lm_head,
            )
        else:
            self.lm_head = None
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, inputs_embeds, pp_proxy_tensors
        )
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_nextn: bool = False
    ):
        self.do_load_weights(_remap_gigachat_weight_names(weights), is_nextn=is_nextn)

    def post_load_weights(
        self,
        is_nextn: bool = False,
        weight_names: Optional[Iterable[str]] = None,
    ) -> None:
        if not is_nextn and weight_names is None:
            full_ids = set(self.config.full_attention_layer_ids)
            weight_names = [
                f"model.layers.{lid}.self_attn.kv_b_proj.weight"
                for lid in full_ids
                if self.model.start_layer <= lid < self.model.end_layer
            ]
        return super().post_load_weights(is_nextn=is_nextn, weight_names=weight_names)


EntryClass = [GigaChat35ForCausalLM]
