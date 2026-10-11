import copy
from typing import Iterable, Tuple

import torch
from torch import nn

from sglang.srt.layers.attention.index_topk_share import IndexTopKShareState
from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    get_attn_tp_context,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    get_embedding_tp_kwargs,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
from sglang.srt.models.hunyuan_v4 import (
    HYV4Attention,
    HYV4ForCausalLM,
    permute_hyv4_indexer_weight,
)
from sglang.srt.runtime_context import get_parallel, get_stream
from sglang.srt.utils import BumpAllocator, is_cuda


def _mtp_quant_config(quant_config):
    if quant_config is None:
        return None
    quant_config = copy.deepcopy(quant_config)
    ignored_layers = getattr(quant_config, "ignored_layers", None)
    if ignored_layers is not None:
        quant_config.ignored_layers = [
            name.replace("model.mtp_layers.0", "model.decoder").replace(
                "mtp_layers.0", "model.decoder"
            )
            for name in ignored_layers
        ]
    return quant_config


class HYV4MTPDecoderLayer(nn.Module):
    def __init__(self, config, quant_config=None, prefix="", alt_stream=None):
        super().__init__()
        self.self_attn = HYV4Attention(
            config,
            0,
            quant_config,
            f"{prefix}.self_attn",
            alt_stream,
            is_nextn=True,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DeepseekV2MoE(
            config,
            0,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
            alt_stream=alt_stream,
            is_nextn=True,
            reduce_results=False,
        )
        if hasattr(self.mlp, "shared_experts"):
            self.mlp.shared_experts.swiglu_limit = None
        self.attn_boundary, self.ffn_boundary = append_stages(
            (
                declare_attn(),
                self.input_layernorm,
                {"qkv_latent_func": self.self_attn.prepare_qkv_latent},
            ),
            (declare_ffn(sparse=True), self.post_attention_layernorm),
        )

    def forward(
        self,
        positions,
        hidden_states,
        forward_batch,
        zero_allocator,
        prev_topk_indices=None,
    ):
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        try:
            hidden_states = self.self_attn(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=prev_topk_indices,
            )
        finally:
            get_attn_tp_context().clear_attn_inputs()
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)
        return hidden_states, topk_indices


class HYV4ModelNextN(nn.Module):
    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
            **get_embedding_tp_kwargs(),
        )
        self.enorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.alt_stream = get_stream("alt") if is_cuda() else None
        with layer_stack():
            self.decoder = HYV4MTPDecoderLayer(
                config,
                quant_config,
                f"{prefix}.decoder",
                self.alt_stream,
            )
        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        residual_batch.start(forward_batch)
        topk_share = IndexTopKShareState.from_mtp_carry(forward_batch)
        hidden_states, topk_indices = self.decoder(
            positions,
            hidden_states,
            forward_batch,
            zero_allocator,
            topk_share.topk_indices,
        )
        topk_share.update(topk_indices)
        topk_share.publish()
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = residual_batch.final_norm(
                hidden_states, forward_batch, self.shared_head.norm
            )
        return hidden_states


class HYV4ForCausalLMNextN(nn.Module, DeepseekV2WeightLoaderMixin):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    @staticmethod
    def shared_experts_fusion_disable_reason(hf_config, quant_config):
        return HYV4ForCausalLM.shared_experts_fusion_disable_reason(
            hf_config, quant_config
        )

    def __init__(self, config, quant_config=None, prefix=""):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.pp_group = get_parallel().pp_group
        nextn_quant_config = _mtp_quant_config(quant_config)
        self.model = HYV4ModelNextN(
            config, nextn_quant_config, prefix=f"{prefix}.model"
        )
        self.num_fused_shared_experts = self.model.decoder.mlp.num_fused_shared_experts
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head",
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch):
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        layer_prefix = f"model.layers.{self.config.num_hidden_layers}"

        def mapped_weights():
            for name, loaded_weight in weights:
                if not name.startswith("model.mtp_layers.0."):
                    continue
                name = name.replace("model.mtp_layers.0", layer_prefix)
                if name.endswith(".final_layernorm.weight"):
                    name = name.replace(
                        ".final_layernorm.weight", ".shared_head.norm.weight"
                    )
                loaded_weight = permute_hyv4_indexer_weight(
                    name, loaded_weight, self.config
                )
                if name.endswith(".weight_scale"):
                    name += "_inv"
                yield name, loaded_weight

        self.do_load_weights(mapped_weights(), is_nextn=True)

    def post_load_weights(self, is_nextn=True, weight_names=None):
        super().post_load_weights(is_nextn=True, weight_names=weight_names)


EntryClass = [HYV4ForCausalLMNextN]
