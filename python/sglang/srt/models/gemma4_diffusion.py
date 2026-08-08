# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DiffusionGemma serving model."""

import logging
import re
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import GeluAndMul
from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3_causal import Gemma3MLP, Gemma3TextScaledWordEmbedding
from sglang.srt.models.gemma4_causal import Gemma4MoE, Gemma4Router
from sglang.srt.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
)
from sglang.srt.models.gemma4_vision import Gemma4VisionEncoder
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class DiffusionGemmaAttention(nn.Module):
    """Phase-aware Gemma4 attention with DiffusionGemma K/V sharing."""

    def __init__(
        self,
        layer_id: int,
        config,  # text_config
        head_dim: int,
        num_kv_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        tp_size = get_tensor_model_parallel_world_size()

        layer_type = config.layer_types[layer_id]
        # Inclusive (HF) -> exclusive (sglang), matching gemma4_causal.
        self.sliding_window = (
            config.sliding_window - 1 if layer_type == "sliding_attention" else None
        )

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # Gemma4 q/k/v RMSNorm: *w for q/k (scale_shift=0), no-scale for v.
        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, scale_shift=0.0, with_scale=False
        )

        rope_parameters = dict(config.rope_parameters[layer_type])
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_parameters.get("rope_theta", 10000.0),
            rope_scaling={"rope_type": rope_parameters.get("rope_type", "default")},
            partial_rotary_factor=rope_parameters.get("partial_rotary_factor", 1.0),
            is_neox_style=True,
        )

        attn_common = dict(
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        # scaling = 1.0 (Gemma4 folds scale into q_norm). Two attn instances share
        # the same q/k/v/o projections, differing only in the attention mask.
        self.attn = RadixAttention(self.num_heads, self.head_dim, 1.0, **attn_common)
        self.attn_bidir = RadixAttention(
            self.num_heads,
            self.head_dim,
            1.0,
            attn_type=AttentionType.DECODER_BIDIRECTIONAL,
            **attn_common,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim))).flatten(
            -2, -1
        )
        k = self.k_norm(k.unflatten(-1, (self.num_kv_heads, self.head_dim))).flatten(
            -2, -1
        )
        v = self.v_norm(v.unflatten(-1, (self.num_kv_heads, self.head_dim))).flatten(
            -2, -1
        )

        q, k = self.rotary_emb(positions, q, k)

        attn = self.attn if forward_batch.dllm_is_encoder else self.attn_bidir
        attn_output = attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class DiffusionGemmaSelfConditioning(nn.Module):
    """Map the previous canvas distribution into decoder embeddings."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.pre_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_norm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            scale_shift=0.0,
            with_scale=False,
        )
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = GeluAndMul()

    def forward(self, inputs_embeds, signal):
        gate_up, _ = self.gate_up_proj(self.pre_norm(signal))
        h, _ = self.down_proj(self.act_fn(gate_up))
        return self.post_norm(inputs_embeds + h)


class DiffusionGemmaDecoderLayer(nn.Module):
    """Shared encoder/decoder transformer layer."""

    def __init__(
        self,
        layer_id: int,
        config,  # text_config
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        layer_type = config.layer_types[layer_id]
        is_full = layer_type == "full_attention"
        # SGLang naming: base head_dim/num_key_value_heads are the full-attention
        # layers, swa_* are the sliding-window layers (set by the gemma4 config hook).
        head_dim = (
            config.head_dim
            if is_full
            else getattr(config, "swa_head_dim", config.head_dim)
        )
        num_kv_heads = (
            config.num_key_value_heads
            if is_full
            else getattr(config, "swa_num_key_value_heads", config.num_key_value_heads)
        )

        self.self_attn = DiffusionGemmaAttention(
            layer_id=layer_id,
            config=config,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.router = Gemma4Router(
            config, quant_config=quant_config, prefix=add_prefix("router", prefix)
        )
        self.moe = Gemma4MoE(
            hidden_size=config.hidden_size,
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("moe", prefix),
        )

        eps = config.rms_norm_eps
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm_1 = RMSNorm(config.hidden_size, eps=eps)
        self.post_feedforward_layernorm_2 = RMSNorm(config.hidden_size, eps=eps)

        # Encoder and decoder use the same weights but distinct per-layer scalars.
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
        self.register_buffer("encoder_layer_scalar", torch.ones(1), persistent=True)

    def forward(self, positions, hidden_states, forward_batch):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        h1 = self.post_feedforward_layernorm_1(
            self.mlp(self.pre_feedforward_layernorm(residual))
        )
        # MoE branch: the router gates on the RAW residual (the router's input
        # norm is with_scale=False, so ln2's learned per-channel weights would
        # change the routed direction). Experts still get the ln2-normed input.
        moe_in = self.pre_feedforward_layernorm_2(residual)
        router_logits = self.router(residual)
        h2 = self.post_feedforward_layernorm_2(self.moe(moe_in, router_logits))

        hidden_states = self.post_feedforward_layernorm(h1 + h2)
        hidden_states = residual + hidden_states

        scalar = (
            self.encoder_layer_scalar
            if forward_batch.dllm_is_encoder
            else self.layer_scalar
        )
        return hidden_states * scalar


class DiffusionGemmaModel(nn.Module):
    """Shared transformer stack (stored under `model.decoder.*`) + the decoder's
    self-conditioning block."""

    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        text_config = config.text_config
        self.config = text_config

        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            text_config.pad_token_id,
            embed_scale=text_config.hidden_size**0.5,
        )
        self.self_conditioning = DiffusionGemmaSelfConditioning(
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("self_conditioning", prefix),
        )
        self.pp_group = get_pp_group()
        self.layers, self.start_layer, self.end_layer = make_layers(
            text_config.num_hidden_layers,
            lambda idx, prefix: DiffusionGemmaDecoderLayer(
                layer_id=idx,
                config=text_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        # embed_tokens applies the sqrt(hidden) scale internally.
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )

        if not forward_batch.dllm_is_encoder:
            signal = forward_batch.dllm_self_conditioning_embeds
            if signal is None:
                signal = torch.zeros_like(hidden_states)
            hidden_states = self.self_conditioning(hidden_states, signal)

        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[i](positions, hidden_states, forward_batch)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class DiffusionGemmaForBlockDiffusion(PreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.pp_group = get_pp_group()
        if self.pp_group.world_size != 1:
            raise NotImplementedError(
                "DiffusionGemma does not support pipeline parallelism"
            )
        if not (
            getattr(config, "tie_word_embeddings", False)
            and getattr(config.text_config, "tie_word_embeddings", False)
        ):
            raise NotImplementedError("DiffusionGemma requires tied word embeddings")
        if quant_config is not None:
            raise NotImplementedError(
                "DiffusionGemma does not support quantized loading"
            )

        self.config = config
        self.text_config = config.text_config
        self.quant_config = quant_config

        # Shared stack lives under `model` here, but the checkpoint stores it
        # under `model.decoder.*` (remapped in load_weights).
        self.model = DiffusionGemmaModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )
        # Vision tower and projection run only on the first PP rank (encoder input
        # embedding stage).
        self.vision_tower = None
        self.embed_vision = None
        if self.pp_group.is_first_rank:
            self.vision_tower = Gemma4VisionEncoder(
                config.vision_config,
                quant_config=quant_config,
                prefix=add_prefix("vision_tower", prefix),
                mask_pad_before_pool=True,
            )
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                self.text_config,
                quant_config=quant_config,
                prefix=add_prefix("embed_vision", prefix),
            )
        self.lm_head = self.model.embed_tokens
        # Softcapping is handled by LogitsProcessor. skip_all_gather: the tied
        # lm_head is replicated, every TP rank already has full-vocab logits.
        self.logits_processor = LogitsProcessor(
            self.text_config, skip_all_gather=True, return_full_logits=True
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_attention_sliding_window_size(self):
        # Inclusive (HF) -> exclusive (sglang).
        return self.text_config.sliding_window - 1

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        if forward_batch.dllm_is_encoder and forward_batch.contains_image_inputs():
            Gemma4ForConditionalGeneration.prepare_attn_masks(
                self, forward_batch, input_ids, torch.bool
            )

        # Image soft tokens are injected only on the encoder (causal-context) pass.
        if (
            input_embeds is None
            and self.vision_tower is not None
            and forward_batch.dllm_is_encoder
            and forward_batch.contains_image_inputs()
        ):
            hidden_states = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.model,
                data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
                positions=positions,
            )
        else:
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds
            )
        if forward_batch.dllm_is_encoder:
            # Encoder passes only write the context KV and their logits are unused.
            return None
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        vt = self.vision_tower
        embed = self.model.embed_tokens.weight
        all_embeds = []
        for item in items:
            all_pixel_values = flatten_nested_list([item.feature])
            all_position_ids = flatten_nested_list(
                [getattr(item, "image_position_ids", None)]
            )
            for pv_idx, pv in enumerate(all_pixel_values):
                if pv.dim() in (2, 3) and pv.shape[-1] == self.text_config.hidden_size:
                    all_embeds.append(pv.to(embed.device))
                    continue
                if pv_idx >= len(all_position_ids) or all_position_ids[pv_idx] is None:
                    raise ValueError(
                        f"pixel_values[{pv_idx}] has no matching image_position_ids."
                    )
                pp = all_position_ids[pv_idx]
                if pv.dim() == 2:
                    pv = pv.unsqueeze(0)
                if pp.dim() == 2:
                    pp = pp.unsqueeze(0)
                pv = pv.to(device=vt.device, dtype=embed.dtype)
                pp = pp.to(device=vt.device)
                pooled, pooler_mask = vt(pv, pp)
                for hs, mask in zip(pooled, pooler_mask):
                    real_tokens = hs[mask]
                    all_embeds.append(
                        self.embed_vision(
                            inputs_embeds=real_tokens.unsqueeze(0)
                        ).squeeze(0)
                    )
        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        return torch.empty(
            0, self.text_config.hidden_size, device=embed.device, dtype=embed.dtype
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        fused_expert_params_mapping = [
            # (param_name, ckpt_weight_name, shard_ids). gate_up_proj is [E, 2*I, H]
            ("moe.experts.w13_weight", "moe.experts.gate_up_proj", ("w1", "w3")),
            ("moe.experts.w2_weight", "moe.experts.down_proj", ("w2",)),
        ]
        num_experts = self.text_config.num_experts
        # Full-attention layers have no v_proj in the checkpoint (value == key).
        k_eq_v_layers = {
            i
            for i, lt in enumerate(self.text_config.layer_types)
            if lt == "full_attention"
        }

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            # Vision weights (model.encoder.vision_tower.* / .embed_vision.*) load
            # into the reused first-rank Gemma4 vision modules.
            if name.startswith("model.encoder.vision_tower.") or name.startswith(
                "model.encoder.embed_vision."
            ):
                if self.vision_tower is None:
                    continue
                name = Gemma4ForConditionalGeneration._remap_tower_name(
                    name[len("model.encoder.") :], params_dict
                )
                for (
                    param_name,
                    weight_name,
                    shard_id,
                ) in Gemma4ForConditionalGeneration.stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    fused = name.replace(weight_name, param_name)
                    if fused not in params_dict:
                        continue
                    param = params_dict[fused]
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(fused)
                    break
                else:
                    if name in params_dict:
                        param = params_dict[name]
                        wl = getattr(param, "weight_loader", default_weight_loader)
                        wl(param, loaded_weight)
                        loaded_params.add(name)
                continue

            # Encoder text weights are tied to the decoder, so keep only the encoder's
            # per-layer scalar, skip everything else under model.encoder.*.
            if name.startswith("model.encoder."):
                m = re.search(r"layers\.(\d+)\.layer_scalar", name)
                if m is None:
                    continue
                tgt = f"model.layers.{m.group(1)}.encoder_layer_scalar"
                if tgt in params_dict:
                    default_weight_loader(params_dict[tgt], loaded_weight)
                    loaded_params.add(tgt)
                continue

            # The real stack is stored under model.decoder.* -> our model.*
            name = name.replace("model.decoder.", "model.")
            # Route router/expert weights into the Gemma4MoE subtree.
            name = name.replace(".router.per_expert_scale", ".moe.per_expert_scale")
            if ".experts." in name and ".moe.experts." not in name:
                name = name.replace(".experts.", ".moe.experts.")

            if self.text_config.tie_word_embeddings and "lm_head" in name:
                continue

            should_dup_k_to_v = (
                ".k_proj." in name
                and k_eq_v_layers
                and (m := re.search(r"layers\.(\d+)\.", name)) is not None
                and int(m.group(1)) in k_eq_v_layers
            )

            orig_name = name
            # 1) Fused MoE experts (checked first: gate_up_proj contains "up_proj").
            for param_name, weight_name, shard_ids in fused_expert_params_mapping:
                name = orig_name
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                for i in range(num_experts):
                    chunks = loaded_weight[i].chunk(len(shard_ids), dim=0)
                    for chunk, sid in zip(chunks, shard_ids):
                        weight_loader(param, chunk, name, sid, i)
                loaded_params.add(name)
                break
            else:
                # 2) Fused QKV / gate_up (+ k->v dup for full-attention layers).
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    name = orig_name
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    if should_dup_k_to_v and weight_name == "k_proj":
                        weight_loader(param, loaded_weight, "v")
                    loaded_params.add(name)
                    break
                else:
                    # 3) Everything else (norms, scalars, router.proj/scale, embed).
                    name = orig_name
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None or name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        unloaded = dict(self.named_parameters()).keys() - loaded_params
        if unloaded:
            logger.warning(
                "Some weights were not initialized from checkpoint: %s",
                sorted(unloaded),
            )
        return loaded_params


EntryClass = DiffusionGemmaForBlockDiffusion
