"""Spark2.5 backbone adapter for SGLang's DSpark runtime."""

from __future__ import annotations

from copy import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.dspark import DSparkDraftModel, VanillaMarkov
from sglang.srt.models.spark2_5 import (
    Spark2_5Attention,
    Spark2_5MLP,
    _get_attention_sliding_window_size,
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


class Spark2_5DSparkDecoderLayer(nn.Module):
    """Spark2.5 decoder layer with the constructor expected by DSpark."""

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
        super().__init__()
        if attention_conv is not None or mlp_conv is not None:
            raise ValueError("Spark2.5 DSpark does not support draft convolutions")

        layer_type = config.layer_types[layer_id]
        rope_params = config.rope_parameters[layer_type]
        attention_prefix = f"{prefix}.self_attn" if prefix else "self_attn"
        attention_cls = attention_cls or self.attention_cls
        self.self_attn = attention_cls(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", None),
            layer_id=layer_id,
            rope_theta=rope_params["rope_theta"],
            partial_rotary_factor=rope_params.get("partial_rotary_factor", 1.0),
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            quant_config=quant_config,
            sliding_window=_get_attention_sliding_window_size(config),
            layer_type=layer_type,
            headwise_attn_output_gate=getattr(
                config, "headwise_attn_output_gate", True
            ),
            prefix=attention_prefix,
        )
        mlp_prefix = f"{prefix}.mlp" if prefix else "mlp"
        self.mlp = Spark2_5MLP(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=mlp_prefix,
            reduce_results=False,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.attn_boundary, self.ffn_boundary = append_stages(
            (declare_attn(), self.input_layernorm),
            (
                declare_ffn(sparse=False, next_layer_sparse=False),
                self.post_attention_layernorm,
            ),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)

        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
        hidden_states = self.mlp(hidden_states)
        return self.ffn_boundary.finish(hidden_states, forward_batch)


class SparkX25DsparkDraftModel(DSparkDraftModel):
    """DSpark draft model whose transformer layers follow Spark2.5."""

    decoder_layer_cls = Spark2_5DSparkDecoderLayer
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        dspark_config = parse_dspark_draft_config(draft_hf_config=config)
        if dspark_config.target_layer_ids is not None:
            # The parent builds fc from the DFlash config, while target capture
            # honors DSpark's prefixed IDs. Give both the same feature count.
            config = copy(config)
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
            markov_rank=int(config.markov_rank),
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
    def _swa_write_loc(pool, full_loc: torch.Tensor) -> Optional[torch.Tensor]:
        """Resolve the SWA physical ids paired with a full physical write.

        Private SWA pools receive virtual ids and translate them directly. A
        fused draft pool receives full physical ids from ``KVLocPlan``; recover
        the virtual page before asking its shared allocator for the SWA page.
        """
        if getattr(pool, "requires_physical_write_loc", False):
            host_allocator = getattr(pool, "host_allocator", None)
            full_allocator = getattr(host_allocator, "full_attn_allocator", None)
            if full_allocator is not None:
                page_size = int(full_allocator.pool_page_size)
                full_page = full_loc // page_size
                page_offset = full_loc.remainder(page_size)
                virtual_page = full_allocator.physical_to_virtual[full_page]
                virtual_loc = virtual_page * page_size + page_offset
                return host_allocator.translate_loc_from_full_to_swa(virtual_loc)
        if hasattr(pool, "translate_loc_from_full_to_swa"):
            return pool.translate_loc_from_full_to_swa(full_loc)
        return None

    @staticmethod
    def _set_swa_kv_buffer_prefix_valid(
        *,
        pool,
        attn,
        full_loc_2d: torch.Tensor,
        swa_loc_2d: Optional[torch.Tensor],
        commit_lens: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> bool:
        """Use the underlying MHA pool's GPU prefix-valid writer for SWA."""
        layer_id = attn.attn.layer_id
        loc_2d = full_loc_2d
        if isinstance(pool, SWAKVPool):
            layer_id, is_swa_layer = pool.layers_mapping[layer_id]
            if is_swa_layer:
                if swa_loc_2d is None:
                    return False
                loc_2d = swa_loc_2d
                pool = pool.swa_kv_pool
            else:
                pool = pool.full_kv_pool
        # The tiled writer assumes dense token rows and ordinary/FP8 KV.
        # Other layouts and quantization formats keep their native writer.
        if (
            getattr(pool, "use_hnd", False)
            or getattr(pool, "is_quantized_kv_cache", False)
            or getattr(pool, "kv_cache_layout", "nhd") == "vectorized_5d"
            or (getattr(pool, "requires_physical_write_loc", False) and not k.is_cuda)
        ):
            return False
        writer = getattr(pool, "set_kv_buffer_prefix_valid", None)
        if writer is None:
            return False
        try:
            writer(
                attn.attn,
                loc_2d,
                commit_lens,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
                layer_id_override=layer_id,
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
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
        target_hidden_is_projected: bool = False,
    ) -> None:
        """Materialize Spark2.5 draft KV with hybrid-SWA locations.

        Spark2.5 draft layers are sliding-attention layers. Keep this
        location handling local to the Spark adapter so generic DSpark models
        retain their existing single-pool write path.
        """
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
            # them to the reserved slot before indexing the SWA page tables.
            full_loc = torch.where(row_mask, cache_loc_2d, 0).reshape(-1)
        else:
            full_loc = cache_loc
        swa_loc = self._swa_write_loc(pool, full_loc)
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
                        swa_loc.view_as(cache_loc_2d) if swa_loc is not None else None
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
                            swa_loc.index_select(0, valid_indices)
                            if swa_loc is not None
                            else None
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

    def prune_to_ctx_kv_injection(self) -> None:
        self.markov_head = None
        self.confidence_head = None
        for layer in self.layers:
            layer.mlp = None
            layer.self_attn.out_proj = None
        torch.cuda.empty_cache()


EntryClass = [SparkX25DsparkDraftModel]
