"""Spark2.5 backbone adapter for SGLang's DSpark runtime."""

from __future__ import annotations

from copy import copy
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.kv.memory.views import MHAKVView
from sglang.srt.mem_cache.memory_pool import KVCache, KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dspark import DSparkDraftModel, VanillaMarkov
from sglang.srt.models.spark2_5 import (
    Spark2_5Attention,
    Spark2_5DecoderLayer,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)


class Spark2_5DSparkAttention(Spark2_5Attention):
    """Spark attention with the KV-only operations required by DSpark."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn.attn_type = AttentionType.ENCODER_ONLY

    def kv_proj_only(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv, _ = self.q_k_v_proj(hidden_states)
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return k, v

    def apply_k_rope(self, positions: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        dummy_q = torch.empty_like(k)
        _, k = self.rotary_emb(positions, dummy_q, k)
        return k


class Spark2_5DSparkDecoderLayer(Spark2_5DecoderLayer):
    attention_cls = Spark2_5DSparkAttention
    attention_conv = mlp_conv = None

    def __init__(
        self,
        config,
        layer_id: int,
        attention_cls=None,
        attention_conv=None,
        mlp_conv=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        if attention_conv is not None or mlp_conv is not None:
            raise ValueError("Spark2.5 DSpark does not support draft convolutions")
        if attention_cls is not None:
            self.attention_cls = attention_cls
        super().__init__(
            config=config, layer_id=layer_id, quant_config=quant_config, prefix=prefix
        )


class DSparkSpark25DraftModel(DSparkDraftModel):
    """DSpark draft model whose transformer layers follow Spark2.5."""

    decoder_layer_cls = Spark2_5DSparkDecoderLayer
    supports_fused_context_kv = False
    attention_output_projection = "out_proj"

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        config = copy(config)
        config.markov_rank = dspark_config.markov_rank
        if dspark_config.target_layer_ids is not None:
            # The parent builds fc from the DFlash config, while target capture
            # honors DSpark's prefixed IDs. Give both the same feature count.
            raw_dflash_config = getattr(config, "dflash_config", None)
            if raw_dflash_config is None:
                dflash_config = {}
            elif isinstance(raw_dflash_config, dict):
                dflash_config = dict(raw_dflash_config)
            else:
                try:
                    dflash_config = dict(raw_dflash_config)
                except (TypeError, ValueError):
                    dflash_config = dict(vars(raw_dflash_config))
            config.dflash_config = dflash_config
            config.dflash_config["target_layer_ids"] = list(
                dspark_config.target_layer_ids
            )
        with layer_stack():
            super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        target_hidden_size = int(
            getattr(config, "target_hidden_size", config.hidden_size)
        )
        if target_hidden_size != int(config.hidden_size):
            raise ValueError(
                "Spark2.5 DSpark currently requires target_hidden_size == "
                f"hidden_size, got {target_hidden_size} and "
                f"{config.hidden_size}."
            )
        # Replace the generic head with Spark's checkpoint-compatible vanilla
        # Markov head (markov_w1 embedding + markov_w2 projection).
        self.markov_head = VanillaMarkov(
            vocab_size=int(getattr(config, "draft_vocab_size", config.vocab_size)),
            markov_rank=dspark_config.markov_rank,
        )
        self.fc_norm = nn.ModuleList(
            [
                RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
                for _ in range(self.num_context_features)
            ]
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
        del get_embedding, pp_proxy_tensors
        hidden_states = (
            self.forward_embed(input_ids) if input_embeds is None else input_embeds
        )

        residual_batch.start(forward_batch)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)

        hidden_states = residual_batch.final_norm(
            hidden_states,
            forward_batch,
            self.norm,
            skip_empty=True,
        )
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        target_hidden_size = int(
            getattr(self.config, "target_hidden_size", self.config.hidden_size)
        )
        expected = self.num_context_features * target_hidden_size
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(
                "Spark2.5 DSpark target hidden shape mismatch: expected "
                f"[N, {expected}], got {tuple(target_hidden.shape)}."
            )
        chunks = target_hidden.split(target_hidden_size, dim=-1)
        normalized = torch.cat(
            [norm(chunk) for norm, chunk in zip(self.fc_norm, chunks)], dim=-1
        )
        return self.hidden_norm(F.linear(normalized, self.fc.weight))

    def load_weights(self, weights):
        """Load only the ``draft_model.*`` partition of a bundled checkpoint."""

        def draft_weights():
            for name, loaded_weight in weights:
                if name.startswith("draft_model."):
                    yield name.removeprefix("draft_model."), loaded_weight

        super().load_weights(draft_weights())

    @staticmethod
    def _set_swa_kv_buffer_prefix_valid(
        *,
        pool: KVCache,
        attn,
        full_loc_2d: torch.Tensor,
        swa_loc_2d: Optional[torch.Tensor],
        commit_lens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        """Write committed prefixes through the layer's routed MHA view."""
        kv_layer = pool.layer_router.layer(attn.attn.layer_id)
        view = kv_layer.view
        if not isinstance(view, MHAKVView):
            return False
        if not kv_layer.owner and swa_loc_2d is None:
            return False
        loc_2d = full_loc_2d if kv_layer.owner else swa_loc_2d
        # The tiled writer assumes dense token rows and ordinary/FP8 KV.
        # Other layouts and quantization formats keep their native writer.
        if (
            view.use_hnd
            or view.is_quantized_kv_cache
            or view.kv_cache_layout == "vectorized_5d"
            or (view.requires_physical_write_loc and not k.is_cuda)
        ):
            return False
        try:
            view.set_kv_buffer_prefix_valid(
                attn.attn,
                loc_2d,
                commit_lens,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
                layer_id_override=kv_layer.view_layer_id,
            )
        except NotImplementedError:
            return False
        return True

    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        swa_write_ids: Optional[
            Callable[[torch.Tensor], Optional[torch.Tensor]]
        ] = None,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
    ) -> None:
        """Write context KV at the draft runner's routed physical locations."""
        ctx_hidden = (
            target_hidden
            if target_hidden_is_projected
            else self.project_target_hidden(target_hidden)
        )
        prefix_valid = cache_loc_2d is not None and commit_lens is not None
        row_mask = None
        valid_indices = None
        if prefix_valid:
            if cache_loc_2d.ndim != 2 or (
                commit_lens.ndim != 1
                or commit_lens.shape[0] != cache_loc_2d.shape[0]
                or ctx_hidden.shape[0] != cache_loc_2d.numel()
            ):
                raise ValueError(
                    "Spark2.5 DSpark prefix locations and KV rows must match"
                )
            row_mask = torch.arange(
                cache_loc_2d.shape[1], device=cache_loc_2d.device
            ).view(1, -1) < commit_lens.to(device=cache_loc_2d.device).view(-1, 1)
            # Uncommitted entries may contain padding or stale locations. Map
            # them to the reserved slot before resolving sub-pool locations.
            full_loc = torch.where(row_mask, cache_loc_2d, 0).reshape(-1)
        else:
            full_loc = cache_loc
        swa_loc = None if swa_write_ids is None else swa_write_ids(full_loc)
        write_loc = KVWriteLoc(full_loc, swa_loc=swa_loc, physical=True)
        fallback_write_loc = None
        for layer in self.layers:
            attn = layer.self_attn
            k, v = attn.kv_proj_only(ctx_hidden)
            k = attn.apply_k_rope(positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)

            if prefix_valid:
                if self._set_swa_kv_buffer_prefix_valid(
                    pool=pool,
                    attn=attn,
                    full_loc_2d=full_loc.view_as(cache_loc_2d),
                    swa_loc_2d=(
                        None if swa_loc is None else swa_loc.view_as(cache_loc_2d)
                    ),
                    commit_lens=commit_lens,
                    k=k,
                    v=v,
                ):
                    continue
                if valid_indices is None:
                    valid_indices = row_mask.reshape(-1).nonzero().flatten()
                    fallback_write_loc = KVWriteLoc(
                        full_loc.index_select(0, valid_indices),
                        swa_loc=(
                            None
                            if swa_loc is None
                            else swa_loc.index_select(0, valid_indices)
                        ),
                        physical=True,
                    )
                if valid_indices.numel() == 0:
                    continue
                k = k.index_select(0, valid_indices)
                v = v.index_select(0, valid_indices)

            pool.set_kv_buffer(
                attn.attn,
                fallback_write_loc if prefix_valid else write_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )


EntryClass = [DSparkSpark25DraftModel]
