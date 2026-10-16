# Modified for SGLang; see this directory's README.md for upstream source.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .configuration_neo_chat import NEOMoELLMConfig
from .modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3PreTrainedModel,
)


class Qwen3MoeMLP(nn.Module):
    """Single expert FFN. Same structure as :class:`Qwen3MLP` but the
    intermediate size is parameterised so it can be ``moe_intermediate_size``
    (per-expert) for experts and ``intermediate_size`` for any dense fallback.
    """

    def __init__(self, config, intermediate_size: Optional[int] = None):
        super().__init__()
        from transformers.activations import ACT2FN

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else config.intermediate_size
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Top-k softmax-routed MoE block matching HuggingFace's Qwen3-MoE layout.

    Parameter names (``gate.weight``, ``experts.{i}.gate_proj/up_proj/down_proj``)
    are kept identical so converted A3B checkpoints load directly via the
    ``mlp.*`` / ``mlp_mot_gen.*`` keys. The block is parameterised explicitly
    so the same class can serve both the understanding branch (``num_experts``
    experts, top-k = ``num_experts_per_tok``, width ``moe_intermediate_size``)
    and the image-generation branch (``gen_num_experts`` etc.).
    """

    def __init__(
        self,
        config: NEOMoELLMConfig,
        num_experts: Optional[int] = None,
        num_experts_per_tok: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = (
            int(num_experts) if num_experts is not None else int(config.num_experts)
        )
        self.top_k = int(
            num_experts_per_tok
            if num_experts_per_tok is not None
            else config.num_experts_per_tok
        )
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
        self.hidden_size = config.hidden_size

        expert_intermediate_size = int(
            moe_intermediate_size
            if moe_intermediate_size is not None
            else config.moe_intermediate_size
        )

        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(config, intermediate_size=expert_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_dim = orig_shape[-1]
        flat = hidden_states.view(-1, hidden_dim)
        n_tokens = flat.shape[0]

        router_logits = self.gate(flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(flat.dtype)

        output = torch.zeros(
            (n_tokens, hidden_dim), dtype=flat.dtype, device=flat.device
        )
        # (num_experts, top_k, num_tokens)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(
            2, 1, 0
        )

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            expert_layer = self.experts[expert_idx]
            current_state = flat.index_select(0, top_x)
            current_out = (
                expert_layer(current_state) * routing_weights[top_x, idx, None]
            )
            output.index_add_(0, top_x, current_out.to(flat.dtype))

        return output.view(*orig_shape)


class Qwen3MoeDecoderLayer(Qwen3DecoderLayer):
    """A Qwen3-MoE decoder block with the NEO-Unify two-branch structure.

    Mirrors ``Qwen3DecoderLayer`` from :mod:`modeling_qwen3` but uses sparse
    MoE blocks on *both* branches:

      * ``self.mlp``         - understanding-path MoE
                               (``num_experts`` / ``num_experts_per_tok`` /
                               ``moe_intermediate_size``)
      * ``self.mlp_mot_gen`` - image-generation-path MoE
                               (``gen_num_experts`` / ``gen_num_experts_per_tok`` /
                               ``gen_moe_intermediate_size``)

    Layers listed in ``mlp_only_layers`` or those not aligned with
    ``decoder_sparse_step`` fall back to a dense :class:`Qwen3MoeMLP` on the
    understanding branch (matching upstream Qwen3-MoE), while the
    generation branch still uses a sparse MoE.
    """

    def _init_mlps(self, config: NEOMoELLMConfig, layer_idx: int):
        mlp_only_layers = list(config.mlp_only_layers or [])
        decoder_sparse_step = int(config.decoder_sparse_step or 1)
        is_sparse = (
            int(config.num_experts) > 0
            and layer_idx not in mlp_only_layers
            and (layer_idx + 1) % decoder_sparse_step == 0
        )

        if is_sparse:
            self.mlp = Qwen3MoeSparseMoeBlock(
                config,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                moe_intermediate_size=config.moe_intermediate_size,
            )
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        # Image-generation branch: in the A3B checkpoint this is *also* a sparse
        # MoE block (``gen_num_experts`` experts, typically smaller than the und
        # branch's ``num_experts``). ``NEOMoELLMConfig`` defaults the gen-path
        # knobs to their und-path counterparts so legacy single-pool configs
        # keep working.
        self.mlp_mot_gen = Qwen3MoeSparseMoeBlock(
            config,
            num_experts=config.gen_num_experts,
            num_experts_per_tok=config.gen_num_experts_per_tok,
            moe_intermediate_size=config.gen_moe_intermediate_size,
        )


class Qwen3MoePreTrainedModel(Qwen3PreTrainedModel):
    config: NEOMoELLMConfig
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    _can_compile_fullgraph = False  # MoE routing has data-dependent control flow.
    _can_record_outputs = {
        "hidden_states": Qwen3MoeDecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3MoeModel(Qwen3MoePreTrainedModel, Qwen3Model):
    _decoder_layer_cls = Qwen3MoeDecoderLayer


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel, Qwen3ForCausalLM):
    _model_cls = Qwen3MoeModel


__all__ = [
    "Qwen3MoeForCausalLM",
    "Qwen3MoeModel",
    "Qwen3MoePreTrainedModel",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeSparseMoeBlock",
    "Qwen3MoeMLP",
]
