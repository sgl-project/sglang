"""EAGLE3 draft model with MLA attention for Kimi-K2.5.

The ``kimi-k2.5-eagle3-mla`` checkpoint pairs an EAGLE3 layout
(concatenated [embed_norm, hidden_norm] pre-attention input, fc projection
over the concatenated multi-layer aux hidden states, single decoder layer,
dense MLP) with DeepSeek-V2 multi-latent attention. Sharing the MLA layout
with the Kimi-K2.5 target keeps the draft KV cache small.
"""

import copy
import logging
import re
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.communicator import AttentionInputs, get_attn_tp_context
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA, DeepseekV2MLP
from sglang.srt.utils import BumpAllocator, add_prefix

logger = logging.getLogger(__name__)


def _get_eagle_aux_layer_count(config: PretrainedConfig) -> int:
    """Number of target layers whose hidden states get concatenated into fc."""
    eagle_config = getattr(config, "eagle_config", None)
    if isinstance(eagle_config, dict):
        layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    else:
        layer_ids = getattr(eagle_config, "eagle_aux_hidden_state_layer_ids", None)
    if layer_ids is None:
        layer_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
    if layer_ids is None:
        return 3
    return len(layer_ids)


class Eagle3MLADecoderLayer(nn.Module):
    """One EAGLE3 draft layer that uses DeepSeek-V2 multi-latent attention.

    Pre-attention concatenates the input embedding and the target hidden
    state along the channel dim, doubling the input width to MLA's fused
    QKV-down projection.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
            rope_params = config.rope_parameters
            rope_theta = rope_params.get("rope_theta", 10000)
            rope_scaling = (
                rope_params if rope_params.get("rope_type") != "default" else None
            )
        else:
            rope_theta = config.rope_theta
            rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings

        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=True,
            prefix=add_prefix("self_attn", prefix),
        )

        # EAGLE3 doubles MLA's QKV-down input by concatenating
        # input_layernorm(embed) and hidden_norm(target_hidden) along the
        # feature dim. Replace the projection that DeepseekV2AttentionMLA
        # built for a single-hidden input.
        attn = self.self_attn
        if attn.q_lora_rank is None:
            raise ValueError(
                "Eagle3 MLA layer requires q_lora_rank in the draft config"
            )
        attn.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
            2 * config.hidden_size,
            attn.q_lora_rank + attn.kv_lora_rank + attn.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("self_attn.fused_qkv_a_proj_with_mqa", prefix),
        )
        # Recompute fused-proj-dependent flags so they reflect the new input dim.
        attn.has_fused_proj = True
        attn.use_min_latency_fused_a_gemm = False
        quant_method = getattr(attn.fused_qkv_a_proj_with_mqa, "quant_method", None)
        attn.is_packed_weight = (
            quant_method is not None
            and hasattr(quant_method, "quant_config")
            and quant_method.quant_config is not None
            and quant_method.quant_config.get_name()
            in {"awq", "awq_marlin", "moe_wna16"}
        )

        self.mlp = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        attn_input = torch.cat([embeds, hidden_states], dim=-1)

        # MLA's forward_absorb_prepare reads the qkv-down projection result
        # from attn_tp_context. We bypass LayerCommunicator here (the eagle3
        # draft layer is one isolated layer with custom pre-attention norms),
        # so publish the attention input ourselves.
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(
                attn_input,
                forward_batch,
                self.self_attn.prepare_qkv_latent,
            )
        )

        attn_out = self.self_attn(
            positions=positions,
            hidden_states=attn_input,
            forward_batch=forward_batch,
            zero_allocator=zero_allocator,
        )
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]

        hidden_states, residual = self.post_attention_layernorm(attn_out, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Eagle3MLAModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        target_hidden_size = (
            getattr(config, "target_hidden_size", None) or config.hidden_size
        )
        num_fc_input = _get_eagle_aux_layer_count(config)
        self.fc = nn.Linear(
            target_hidden_size * num_fc_input,
            config.hidden_size,
            bias=getattr(config, "bias", False),
        )

        if config.num_hidden_layers != 1:
            raise ValueError("EAGLE3 currently only supports 1 layer")
        self.midlayer = Eagle3MLADecoderLayer(
            config,
            layer_id=0,
            quant_config=quant_config,
            prefix=prefix,
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if input_embeds is None:
            embeds = self.embed_tokens(input_ids)
        else:
            embeds = input_embeds

        hidden_states = forward_batch.spec_info.hidden_states
        if hidden_states.shape[-1] != embeds.shape[-1]:
            hidden_states = self.fc(hidden_states)

        if hidden_states.shape[0] == 0:
            return hidden_states, [hidden_states]

        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=embeds.device,
        )
        hidden_states, residual = self.midlayer(
            positions, embeds, hidden_states, forward_batch, zero_allocator
        )

        hidden_states_to_logits, hidden_states_to_aux = self.norm(
            hidden_states, residual
        )
        return hidden_states_to_logits, [hidden_states_to_aux]


class Eagle3DeepseekV2ForCausalLM(nn.Module):
    """EAGLE3 draft model architecture with DeepSeek-V2 MLA attention.

    Used by checkpoints like ``kimi-k2.5-eagle3-mla`` that pair
    an EAGLE3 layout with multi-latent attention so the draft KV cache shape
    matches the target's MLA cache.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        # Match deepseek_nextn behavior: modelopt_fp4 is target-only and the
        # bf16 draft must not inherit the FP4 quant method.
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            logger.warning(
                "Overriding Eagle3DeepseekV2ForCausalLM quant config for "
                "modelopt_fp4 target; draft weights are bf16."
            )
            quant_config = None
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.model = Eagle3MLAModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # llama_eagle3 sets a load-from-target flag when draft_vocab_size is
        # missing. This checkpoint declares its own draft head, so keep ours.
        self.load_lm_head_from_target = False
        draft_vocab_size = getattr(config, "draft_vocab_size", None)
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            if draft_vocab_size is None:
                self.load_lm_head_from_target = True
                draft_vocab_size = config.vocab_size
                config.draft_vocab_size = draft_vocab_size
            self.lm_head = ParallelLMHead(
                draft_vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        # Logits processor sees the draft vocab.
        config_for_logits = copy.deepcopy(config)
        config_for_logits.vocab_size = draft_vocab_size or config.vocab_size
        self.logits_processor = LogitsProcessor(config_for_logits)

        self.capture_aux_hidden_states = True
        self.hot_token_id = None

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        aux_hidden_states = None
        if isinstance(hidden_states, tuple):
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed(self, embed: torch.Tensor) -> None:
        # If draft hidden size != target hidden size, embeddings can't be shared.
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size is not None
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_hot_token_id(self):
        return self.hot_token_id

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        stacked_params_mapping = [
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        cached_a_proj: dict[str, torch.Tensor] = {}

        for name, loaded_weight in weights:
            if name == "d2t" or name.endswith(".d2t"):
                # d2t stores diffs between draft id and target id; absent in
                # checkpoints whose draft_vocab_size equals vocab_size.
                self.hot_token_id = loaded_weight + torch.arange(loaded_weight.shape[0])
                continue
            if name == "t2d" or name.endswith(".t2d"):
                continue

            # Map checkpoint layout (layers.0.X, embed_tokens.X, fc, norm) to the
            # internal layout (model.midlayer.X, model.embed_tokens.X, model.fc,
            # model.norm). lm_head stays at the top level.
            mapped_name = re.sub(r"^layers\.0\.", "midlayer.", name)
            if not mapped_name.startswith("lm_head."):
                mapped_name = f"model.{mapped_name}"

            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in mapped_name:
                    continue
                target_name = mapped_name.replace(weight_name, param_name)
                if target_name not in params_dict:
                    continue
                param = params_dict[target_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                handled = True
                break
            if handled:
                continue

            if "q_a_proj" in mapped_name or "kv_a_proj_with_mqa" in mapped_name:
                cached_a_proj[mapped_name] = loaded_weight
                q_name = (
                    mapped_name
                    if "q_a_proj" in mapped_name
                    else mapped_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                )
                kv_name = (
                    mapped_name
                    if "kv_a_proj_with_mqa" in mapped_name
                    else mapped_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                )
                if q_name in cached_a_proj and kv_name in cached_a_proj:
                    fused_weight = torch.cat(
                        [cached_a_proj[q_name], cached_a_proj[kv_name]], dim=0
                    )
                    fused_name = q_name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                    if fused_name in params_dict:
                        param = params_dict[fused_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, fused_weight)
                    cached_a_proj.pop(q_name)
                    cached_a_proj.pop(kv_name)
                continue

            if mapped_name not in params_dict:
                logger.warning("Eagle3 MLA: skipping unexpected weight %s", name)
                continue
            param = params_dict[mapped_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)

        self.post_load_weights()

    def post_load_weights(self) -> None:
        """Split kv_b_proj into w_kc / w_vc tensors used by MLA absorb_core.

        DeepseekV2 normally does this in DeepseekV2WeightLoaderMixin.post_load_weights;
        we re-implement the bf16 fast-path directly here to keep the eagle3 draft
        path independent of the full DeepseekV2 weight loader.
        """
        self_attn = self.model.midlayer.self_attn
        w = self_attn.kv_b_proj.weight
        if w.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise NotImplementedError(
                f"Eagle3 MLA draft post_load_weights only supports float dtypes, got {w.dtype}"
            )
        w_kc, w_vc = w.unflatten(
            0, (-1, self_attn.qk_nope_head_dim + self_attn.v_head_dim)
        ).split([self_attn.qk_nope_head_dim, self_attn.v_head_dim], dim=1)
        self_attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
        self_attn.w_vc = w_vc.contiguous().transpose(1, 2)


EntryClass = [Eagle3DeepseekV2ForCausalLM]
