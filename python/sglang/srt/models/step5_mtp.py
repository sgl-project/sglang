# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 MTP model and loading; reuses target decoder components."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from sglang.srt.configs.step5 import (
    Step5Config,
    resolve_step5_layer_selection,
)
from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.step5 import (
    Step5DecoderLayer,
    Step5ForCausalLM,
)
from sglang.srt.runtime_context import get_model, get_parallel, get_spec
from sglang.srt.utils import add_prefix


class Step5DraftModel(nn.Module):
    """Consume and return the public FP32 hidden-history ABI.

    The source narrows BOTH embedding and previous residual before their input
    norms. Concatenation and the replicated, unquantized eh projection use BF16;
    the decoder carries a BF16 residual stream. Final norm retains FP32 weight;
    the materialized BF16 sum is exported as FP32 for the existing history ABI.
    """

    def __init__(
        self,
        *,
        config: Step5Config,
        draft_model_idx: int | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        if (
            parallel.tp_size,
            parallel.attn_dp_size,
            parallel.attn_tp_size,
        ) not in ((4, 1, 4), (8, 1, 8), (8, 2, 4)) or parallel.pp_size != 1:
            raise ValueError("Step5 draft requires TP4, TP8, or DP2/TP4 with PP1")
        use_dp_attention_reduce = parallel.attn_dp_size > 1
        if (
            config.num_hidden_layers,
            config.num_nextn_predict_layers,
            config.hidden_size,
            config.vocab_size,
        ) != (92, 3, 4096, 128896) or config.tie_word_embeddings:
            raise ValueError("Step5 draft requires the frozen 92+3 checkpoint")
        self.step5_layer_selection = resolve_step5_layer_selection(
            config=config, draft_model_idx=draft_model_idx
        )
        if self.step5_layer_selection.draft_model_idx is None:
            raise ValueError("Step5 draft requires the draft architecture and depth")
        layer_id = self.step5_layer_selection.start_layer
        if layer_id in config.moe_layer_list:
            raise ValueError("Step5 draft requires the checkpoint's dense SWA layers")
        self.hidden_size = config.hidden_size
        self.mtp_image_token_id = config.mtp_image_token_id
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            use_attn_tp_group=use_dp_attention_reduce,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.enorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.hnorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.eh_proj = nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False, dtype=torch.bfloat16
        )
        layer_prefix = add_prefix(f"layers.{layer_id}", prefix)
        with layer_stack():
            self.mtp_block = Step5DecoderLayer(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                tp_rank=parallel.attn_tp_rank,
                tp_size=parallel.attn_tp_size,
                use_dp_attention_reduce=use_dp_attention_reduce,
                prefix=layer_prefix,
            )
        self.norm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mtp_image_token_id is not None:
            input_ids = torch.where(
                (input_ids >= 0) & (input_ids < self.embed_tokens.num_embeddings),
                input_ids,
                self.mtp_image_token_id,
            )
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if forward_batch.spec_info is None:
            raise ValueError("Step5 draft requires previous-depth hidden states")
        previous = forward_batch.spec_info.hidden_states
        if (
            previous is None
            or previous.dtype != torch.float32
            or previous.shape
            != (
                input_ids.numel(),
                self.hidden_size,
            )
        ):
            raise ValueError("Step5 draft requires FP32 [tokens, hidden_size] history")
        embedded = (
            self.embed_input_ids(input_ids)
            if self.mtp_image_token_id is not None or input_embeds is None
            else input_embeds
        )
        embedded = self.enorm(embedded.bfloat16())
        normalized_previous = self.hnorm(previous.bfloat16())
        hidden_states = self.eh_proj(torch.cat((embedded, normalized_previous), dim=-1))
        residual_batch.start(forward_batch)
        hidden_states = self.mtp_block(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states, residual = residual_batch.stream_of(forward_batch).export(
            hidden_states
        )
        forward_batch.residual_stream = None
        hidden_states = hidden_states + residual
        normalized = self.norm(hidden_states)
        return normalized, hidden_states.float()


class Step5MTP(nn.Module):
    kv_cache_dtype_full_attention_only = True

    def __init__(
        self,
        config: Step5Config,
        quant_config: QuantizationConfig | None = None,
        draft_model_idx: int | None = None,
        prefix: str = "",
        *,
        valid_vocab_size: int,
    ) -> None:
        super().__init__()
        if (
            type(valid_vocab_size) is not int
            or not 0 < valid_vocab_size <= config.vocab_size
        ):
            raise ValueError("Step5 draft requires a resolved effective vocabulary")
        self.config = config
        self.checkpoint_path = (
            get_spec().speculative_draft_model_path or get_model().model_path
        )
        self.quant_config = quant_config
        self.draft_model_idx = draft_model_idx
        self.model = Step5DraftModel(
            config=config,
            draft_model_idx=draft_model_idx,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.layer_selection = self.model.step5_layer_selection
        self.tie_word_embeddings = False
        self.num_fused_shared_experts = 0
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(
            config, valid_vocab_size=valid_vocab_size
        )

    @property
    def start_layer(self):
        return self.layer_selection.start_layer

    @property
    def end_layer(self):
        return self.layer_selection.end_layer

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def set_embed_and_head(self, embed: nn.Parameter, head: nn.Parameter) -> None:
        """Native sharing callback: borrow the target embedding, retain draft head."""
        current = self.model.embed_tokens.weight
        if (
            embed.shape != current.shape
            or embed.dtype != current.dtype
            or embed.device != current.device
        ):
            raise ValueError(
                "Step5 shared embedding requires the same TP shard and dtype"
            )
        self.model.embed_tokens.weight = embed

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = (
            ("self_attn.qkv_proj.weight", "self_attn.q_proj.weight", "q"),
            ("self_attn.qkv_proj.weight", "self_attn.k_proj.weight", "k"),
            ("self_attn.qkv_proj.weight", "self_attn.v_proj.weight", "v"),
            ("mlp.gate_up_proj.weight", "mlp.gate_proj.weight", 0),
            ("mlp.gate_up_proj.weight", "mlp.up_proj.weight", 1),
        )
        params_dict = dict(self.named_parameters())
        config = self.config
        layer_id = self.layer_selection.start_layer
        prefix = f"model.layers.{layer_id}."
        top_level = {
            "model.embed_tokens.weight": "model.embed_tokens.weight",
            prefix + "enorm.weight": "model.enorm.weight",
            prefix + "hnorm.weight": "model.hnorm.weight",
            prefix + "eh_proj.weight": "model.eh_proj.weight",
            prefix + "transformer.shared_head.norm.weight": "model.norm.weight",
            prefix + "transformer.shared_head.output.weight": "lm_head.weight",
        }
        expected = set(top_level)
        for name in params_dict:
            if not name.startswith("model.mtp_block."):
                continue
            local = name.removeprefix("model.mtp_block.")
            shards = [
                source
                for target, source, _ in stacked_params_mapping
                if target == local
            ]
            expected.update(prefix + source for source in (shards or [local]))
        checkpoint_depth = config.num_hidden_layers + config.num_nextn_predict_layers
        seen = set()
        weights = Step5ForCausalLM._indexed_duplicate_weights(
            weights,
            expected,
            getattr(self, "checkpoint_path", None),
        )
        for name, loaded_weight in weights:
            if name.startswith("vision_model.") or name in (
                "vit_large_projector.weight",
                "model.norm.weight",
                "lm_head.weight",
            ):
                continue
            if name.startswith("model.layers."):
                _, _, index, _ = name.split(".", 3)
                if (
                    not index.isascii()
                    or not index.isdecimal()
                    or str(int(index)) != index
                    or not 0 <= int(index) < checkpoint_depth
                ):
                    raise ValueError(f"Invalid Step5 checkpoint layer: {name}")
                if int(index) != layer_id:
                    continue
            if name not in expected:
                raise ValueError(f"Unknown Step5 draft weight: {name}")
            if name in seen:
                raise ValueError(f"Duplicate Step5 draft weight: {name}")
            seen.add(name)
            name = top_level.get(name, name.replace(prefix, "model.mtp_block.", 1))
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if not name.endswith("." + weight_name):
                    continue
                name = name[: -len(weight_name)] + param_name
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        missing = expected - seen
        if missing:
            raise ValueError(
                f"Step5 draft is missing {len(missing)} weights: {sorted(missing)}"
            )
        return seen

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ):
        hidden_states, hidden_states_before_norm = self.model(
            input_ids, positions, forward_batch, input_embeds
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=hidden_states_before_norm,
        )


EntryClass = Step5MTP
