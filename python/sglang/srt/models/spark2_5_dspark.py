from functools import partial
from typing import Callable, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
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
    context_kv_output_projection = "out_proj"

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        with layer_stack():
            super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self._target_hidden_width = self.num_context_features * config.hidden_size
        self.fc_norm = nn.ModuleList(
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for _ in range(self.num_context_features)
        )

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

    def bind_context_kv_writer(
        self, *, write_loc_resolver: Callable[[torch.Tensor], KVWriteLoc]
    ) -> Callable[..., None]:
        return partial(
            self.write_target_hidden_kv, write_loc_resolver=write_loc_resolver
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
        write_loc_resolver: Optional[Callable[[torch.Tensor], KVWriteLoc]] = None,
    ) -> None:
        ctx_hidden = (
            target_hidden
            if target_hidden_is_projected
            else self.project_target_hidden(target_hidden)
        )
        prefix_valid = cache_loc_2d is not None and commit_lens is not None
        if prefix_valid:
            columns = torch.arange(cache_loc_2d.shape[1], device=cache_loc_2d.device)
            committed = columns[None, :] < commit_lens[:, None]
            # Rejected rows may hold invalid ids; mask them before any table gather.
            full_loc = torch.where(committed, cache_loc_2d, 0)
        else:
            full_loc = cache_loc
        if swa_loc is not None:
            write_loc = KVWriteLoc(full_loc, swa_loc=swa_loc, physical=True)
        else:
            write_loc = (
                KVWriteLoc(full_loc, physical=True)
                if write_loc_resolver is None
                else write_loc_resolver(full_loc)
            )
        for layer in self.layers:
            attn = layer.self_attn
            qkv, _ = attn.q_k_v_proj(ctx_hidden)
            _, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
            # Spark attention applies RoPE without the Q/K norms of Qwen drafts.
            _, k = attn.rotary_emb(positions, torch.empty_like(k), k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            route = pool.layer_router.layer(attn.attn.layer_id)
            if prefix_valid:
                route.store_prefix_valid(
                    layer=attn.attn,
                    loc_info=write_loc,
                    commit_lens=commit_lens,
                    cache_k=k,
                    cache_v=v,
                    k_scale=attn.attn.k_scale,
                    v_scale=attn.attn.v_scale,
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
