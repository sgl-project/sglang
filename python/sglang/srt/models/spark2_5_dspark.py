from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, layer_window_locs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.models.spark2_5 import Spark2_5Attention, Spark2_5DecoderLayer


class Spark2_5DSparkDecoderLayer(Spark2_5DecoderLayer):
    attention_cls = Spark2_5Attention

    def __init__(
        self,
        config,
        layer_id: int,
        attention_cls,
        attention_conv,
        mlp_conv,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        if (
            attention_cls is not Spark2_5Attention
            or attention_conv is not None
            or mlp_conv is not None
        ):
            raise ValueError(
                "Spark2.5 DSpark requires Spark attention without convolutions."
            )
        super().__init__(
            config=config, layer_id=layer_id, quant_config=quant_config, prefix=prefix
        )
        self.attention_conv = self.mlp_conv = None
        # Draft queries attend bidirectionally within the speculative block.
        self.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY


class DSparkSpark25DraftModel(DSparkDraftModel):
    decoder_layer_cls = Spark2_5DSparkDecoderLayer
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        with layer_stack():
            super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self._target_hidden_width = self.num_context_features * config.hidden_size
        self.fc_norm = nn.ModuleList(
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for _ in range(self.num_context_features)
        )

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.out_proj = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        hidden_states = (
            self.forward_embed(input_ids) if input_embeds is None else input_embeds
        )
        residual_batch.start(forward_batch)
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = self._target_hidden_width
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(
                f"Spark2.5 DSpark requires target hidden shape [N, {expected}], "
                f"got {tuple(target_hidden.shape)}."
            )
        chunks = target_hidden.split(self.config.hidden_size, dim=-1)
        target_hidden = torch.cat(
            [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)], dim=-1
        )
        return self.hidden_norm(F.linear(target_hidden, self.fc.weight))

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        super().load_weights(
            (name.removeprefix("draft_model."), weight)
            for name, weight in weights
            if name.startswith("draft_model.")
        )

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
        swa_loc: Optional[torch.Tensor] = None,
    ) -> None:
        ctx_hidden = (
            target_hidden
            if target_hidden_is_projected
            else self.project_target_hidden(target_hidden)
        )
        prefix_valid = cache_loc_2d is not None and commit_lens is not None
        loc = cache_loc_2d if prefix_valid else cache_loc
        if prefix_valid and swa_loc is not None:
            swa_loc = swa_loc.view_as(cache_loc_2d)
        for layer in self.layers:
            attn = layer.self_attn
            qkv, _ = attn.q_k_v_proj(ctx_hidden)
            _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            # Spark attention applies RoPE without the Q/K norms of Qwen drafts.
            _, k = attn.rotary_emb(positions, torch.empty_like(k), k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            route = pool.layer_router.layer(attn.attn.layer_id)
            write_loc = KVWriteLoc(
                loc,
                window_locs=layer_window_locs(pool, attn.attn.layer_id, swa_loc),
                physical=True,
            )
            if prefix_valid:
                route.store_prefix(
                    attn.attn,
                    write_loc,
                    commit_lens,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            else:
                route.store(
                    attn.attn,
                    write_loc,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )


EntryClass = [DSparkSpark25DraftModel]
