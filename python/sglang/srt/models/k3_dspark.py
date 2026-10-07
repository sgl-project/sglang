# SPDX-License-Identifier: Apache-2.0
"""Inferact Kimi-K3 DSpark draft (``K3DSparkModel``).

Five absorbed-MLA layers plus the dense DSpark Markov and confidence heads.
The checkpoint is not a GQA ``DSparkDraftModel``: attention weights are
``q_a_proj`` / ``kv_a_proj_with_mqa`` / ``kv_b_proj``, and target features
enter through ``context_proj`` + ``context_norm``.

The draft KV pool is replicated under DCP (same as the GQA DSpark draft).
This module writes the latent directly into that pool. The worker keeps the
draft attention off the target's DCP-reduce kernel so the launch stays the
non-DCP MLA persistent kernel.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.utils import BumpAllocator, add_prefix


class K3DSparkBackbone(nn.Module):
    supports_fused_context_kv = False

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        if getattr(config, "hidden_act", None) in (None, ""):
            config.hidden_act = "silu"
        if not hasattr(config, "n_routed_experts"):
            config.n_routed_experts = None
        if getattr(config, "mla_use_output_gate", False):
            raise ValueError(
                "K3DSparkModel with mla_use_output_gate is not implemented."
            )
        if getattr(config, "mla_use_nope", False):
            raise ValueError(
                "K3DSparkModel with mla_use_nope drops the RoPE tail; "
                "this loader expects kv_lora_rank + qk_rope_head_dim."
            )

        self.is_nemotron_35_draft = False
        draft_config = parse_dflash_draft_config(draft_hf_config=config)
        # The Inferact config has no block_size. The worker replaces gamma
        # from --speculative-num-draft-tokens; this only satisfies the mixin.
        self.block_size = int(
            draft_config.block_size or getattr(config, "block_size", None) or 16
        )
        hidden_size = int(config.hidden_size)
        num_layers = int(config.num_hidden_layers)
        rms_norm_eps = float(config.rms_norm_eps)

        if draft_config.target_layer_ids is not None:
            num_context_features = len(draft_config.target_layer_ids)
        elif draft_config.num_target_layers is not None:
            num_context_features = int(draft_config.num_target_layers)
        else:
            num_context_features = num_layers
        self.num_context_features = int(num_context_features)

        self.layers = nn.ModuleList(
            [
                DeepseekV2DecoderLayer(
                    config=config,
                    layer_id=i,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(num_layers)
            ]
        )
        self.context_proj = nn.Linear(
            self.num_context_features * hidden_size,
            hidden_size,
            bias=False,
        )
        self.context_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.final_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.embed_tokens = None

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = int(self.context_proj.in_features)
        if target_hidden.ndim != 2 or int(target_hidden.shape[-1]) != expected:
            raise ValueError(
                "K3DSpark target_hidden feature dim mismatch. "
                f"Expected [N, {expected}] "
                f"(num_context_features={self.num_context_features}), "
                f"got {tuple(target_hidden.shape)}."
            )
        return self.context_norm(self.context_proj(target_hidden))

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
        if input_embeds is None:
            if hasattr(self, "forward_embed"):
                input_embeds = self.forward_embed(input_ids)
            else:
                raise ValueError(
                    "K3DSparkModel requires the shared target embedding "
                    "(attach_shared_modules)."
                )
        hidden_states = input_embeds
        residual_batch.start(forward_batch)
        zero_allocator = BumpAllocator(
            buffer_size=len(self.layers)
            * 2
            * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for layer in self.layers:
            hidden_states, _ = layer(
                positions, hidden_states, forward_batch, zero_allocator
            )
        hidden_states = residual_batch.final_norm(
            hidden_states,
            forward_batch,
            self.final_norm,
            skip_empty=True,
        )
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        stacked = (
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        )
        pending_a: dict[str, torch.Tensor] = {}
        for name, loaded_weight in weights:
            name = name.removeprefix("model.")
            if name.startswith("embed_tokens."):
                continue
            mapped = False
            for param_name, weight_name, shard_id in stacked:
                needle = f".{weight_name}."
                if needle not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                param = params_dict.get(mapped_name)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                mapped = True
                break
            if mapped:
                continue
            if "q_a_proj" in name or "kv_a_proj_with_mqa" in name:
                pending_a[name] = loaded_weight
                q_name = (
                    name
                    if "q_a_proj" in name
                    else name.replace("kv_a_proj_with_mqa", "q_a_proj")
                )
                kv_name = (
                    name
                    if "kv_a_proj_with_mqa" in name
                    else name.replace("q_a_proj", "kv_a_proj_with_mqa")
                )
                if q_name in pending_a and kv_name in pending_a:
                    fused = torch.cat(
                        [pending_a.pop(q_name), pending_a.pop(kv_name)], dim=0
                    )
                    param_name = q_name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                    param = params_dict[param_name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, fused)
                continue
            param = params_dict.get(name)
            if param is None:
                continue
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)
        if pending_a:
            raise ValueError(
                "K3DSpark checkpoint has an unpaired q_a_proj / "
                f"kv_a_proj_with_mqa weight: {sorted(pending_a)}."
            )
        self.post_load_weights()

    def post_load_weights(self) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            weight = attn.kv_b_proj.weight
            if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise NotImplementedError(
                    "K3DSparkModel kv_b absorb only supports float weights, "
                    f"got {weight.dtype}."
                )
            w_kc, w_vc = weight.unflatten(
                0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)
            ).split([attn.qk_nope_head_dim, attn.v_head_dim], dim=1)
            attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            attn.w_vc = w_vc.contiguous().transpose(1, 2)

    def _mla_latent(self, attn, hidden: torch.Tensor, positions: torch.Tensor):
        fused = attn.fused_qkv_a_proj_with_mqa(hidden)[0]
        kv = fused[..., attn.q_lora_rank :]
        k_nope = attn.kv_a_layernorm(kv[..., : attn.kv_lora_rank]).unsqueeze(1)
        k_pe = kv[..., attn.kv_lora_rank :].unsqueeze(1)
        if attn.rotary_emb is not None:
            _, k_pe = attn.rotary_emb(positions, k_pe.new_empty(k_pe.shape), k_pe)
        return torch.cat((k_nope, k_pe), dim=-1)

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
        ctx_hidden = (
            target_hidden
            if target_hidden_is_projected
            else self.project_target_hidden(target_hidden)
        )
        if cache_loc_2d is not None and commit_lens is not None:
            width = cache_loc_2d.shape[1]
            col = torch.arange(width, device=ctx_hidden.device)
            mask = (col.unsqueeze(0) < commit_lens.to(torch.long).view(-1, 1)).reshape(
                -1
            )
            ctx_hidden = ctx_hidden[mask]
            positions = positions[mask]
            loc = cache_loc_2d.reshape(-1)[mask]
        else:
            loc = cache_loc
        if loc.numel() == 0:
            return
        # The draft pool is replicated. Its locs are the widened DCP ids, and
        # set_kv_buffer refuses that space while DCP is on. Drop DCP only for
        # the store so the row is written whole, not owner-sharded.
        for layer in self.layers:
            attn = layer.self_attn
            cache_k = self._mla_latent(attn, ctx_hidden, positions)
            with get_parallel().override(dcp_enabled=False, attn_dcp_size=1):
                pool.set_kv_buffer(
                    attn.attn_mqa,
                    KVWriteLoc(loc, physical=True),
                    cache_k,
                    cache_k,
                )


class K3DSparkModel(DSparkDraftMixin, K3DSparkBackbone):
    def write_target_hidden_kv(self, **kwargs) -> None:
        # Mixin defines the GQA writer and sits ahead of the backbone in the
        # MRO. Call the MLA writer explicitly.
        K3DSparkBackbone.write_target_hidden_kv(self, **kwargs)


EntryClass = [K3DSparkModel]
