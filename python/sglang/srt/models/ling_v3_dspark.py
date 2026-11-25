"""Native Ling MLA/MoE DSpark drafts with shared target vocabulary modules."""

from __future__ import annotations

import copy
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.draft_parallel import DraftParallelSpec, DraftTPExecution
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    should_apply_lm_head_quant_method,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.layers.utils.layer_list import make_layers
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe_v3 import BailingMoELinearDecoderLayer
from sglang.srt.models.dspark import (
    build_confidence_head,
    build_markov_head,
    gather_and_crop_vocab,
    project_through_lm_head,
)
from sglang.srt.models.utils import bind_or_assign
from sglang.srt.speculative.dspark_components.dspark_config import (
    get_dspark_sample_from_anchor,
    is_namespaced_dspark_bundle,
)
from sglang.srt.utils import BumpAllocator


class LingV3ForCausalLMDSparkMLA(nn.Module):
    draft_parallel_spec = DraftParallelSpec(
        tp_execution=DraftTPExecution.TARGET_LAYOUT, dp_deepep=True
    )
    supports_pre_gather_target_hidden_projection = False
    _derived_weight_cache_error = (
        "Online weight updates are not supported for native Ling DSpark: "
        "shared target modules and absorbed MLA weights require a server restart."
    )

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if quant_config is not None or getattr(config, "quantization_config", None):
            raise ValueError(
                "Native Ling DSpark currently requires an unquantized draft."
            )
        if (
            config.attention_type != 1
            or getattr(config, "full_attention_type", "mla") != "mla"
        ):
            raise ValueError("Native Ling DSpark requires MLA on every draft layer.")
        self.config = config
        self.quant_config = None
        self.is_bundled = is_namespaced_dspark_bundle(config)
        self.is_mrope_enabled = config.rope_parameters.get("mrope_section") is not None
        self.num_context_features = len(config.target_layer_ids)
        self.block_size = self.gamma = int(config.block_size)
        self.sample_from_anchor = get_dspark_sample_from_anchor(config)
        self.num_stages = int(config.num_hidden_layers)
        hidden_size = int(config.hidden_size)
        eps = float(config.rms_norm_eps)
        self.target_proj = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.target_norm = RMSNorm(hidden_size, eps=eps)

        def layer_fn(idx, prefix):
            # DsV3MLA normalizes RoPE fields while constructing its rotary layer.
            return BailingMoELinearDecoderLayer(
                copy.deepcopy(config), layer_id=idx, prefix=prefix
            )

        self.layers = make_layers(
            self.num_stages,
            layer_fn,
            prefix=add_prefix("layers", prefix),
            stage_facts=lambda idx: BailingMoELinearDecoderLayer.stage_facts(
                config, idx
            ),
        )
        for layer in self.layers:
            layer.attention.attn_mqa.attn_type = AttentionType.ENCODER_ONLY
            layer.attention.attn_mha.attn_type = AttentionType.ENCODER_ONLY
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.markov_head = build_markov_head(config)
        self.confidence_head = build_confidence_head(config)
        self.embed_tokens: Optional[nn.Module] = None
        self.lm_head: Optional[nn.Module] = None
        self._use_fp32_lm_head = envs.SGLANG_DSPARK_FP32_LM_HEAD.get()

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        # These target-owned modules also participate in the draft's lifecycle.
        for module in (embed_tokens, lm_head):
            module._derived_weight_cache_error = self._derived_weight_cache_error

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise ValueError("Ling DSpark requires attached target embeddings.")
        return self.embed_tokens(input_ids)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = self.num_context_features * int(self.config.hidden_size)
        if target_hidden.shape[-1] != expected:
            raise ValueError(
                f"Ling DSpark expected {expected} target features, got {target_hidden.shape[-1]}."
            )
        return self.target_norm(self.target_proj(target_hidden))

    @torch.no_grad()
    def write_target_hidden_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        pool,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d=None,
        commit_lens=None,
        target_hidden_is_projected: bool = False,
        swa_loc=None,
    ) -> None:
        if target_hidden_is_projected:
            raise ValueError(
                "Ling DSpark does not support pre-projected target hidden states."
            )
        if (cache_loc_2d is None) != (commit_lens is None):
            raise ValueError(
                "Ling DSpark requires cache_loc_2d and commit_lens together."
            )
        if (
            target_hidden.shape[0] != positions.shape[-1]
            or cache_loc.numel() != positions.shape[-1]
        ):
            raise ValueError(
                "Ling DSpark KV token rows, positions and cache locations must match."
            )
        context = self.project_target_hidden(target_hidden)
        for layer in self.layers:
            attn = layer.attention
            if hasattr(attn, "fused_qkv_a_proj_with_mqa"):
                fused, _ = attn.fused_qkv_a_proj_with_mqa(context)
                latent = fused[..., attn.q_lora_rank :]
            else:
                latent, _ = attn.kv_a_proj_with_mqa(context)
            k_nope = attn.kv_a_layernorm(latent[..., : attn.kv_lora_rank]).unsqueeze(1)
            k_rope = latent[..., attn.kv_lora_rank :].unsqueeze(1)
            dummy_q = k_rope.new_empty(
                k_rope.shape[0], attn.num_local_heads, attn.qk_rope_head_dim
            )
            _, k_rope = attn.rotary_emb(positions, dummy_q, k_rope)
            if cache_loc_2d is None:
                pool.set_mla_kv_buffer(
                    attn.attn_mqa, KVWriteLoc(cache_loc, physical=True), k_nope, k_rope
                )
            else:
                # The pool's fixed-row kernel gates rejected rows without a
                # device-to-host sync or a dynamic gather under graph capture.
                pool.set_mla_kv_buffer_prefix_valid(
                    attn.attn_mqa, cache_loc_2d, commit_lens, k_nope, k_rope
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
        if self.is_mrope_enabled and not forward_batch.forward_mode.is_idle():
            if forward_batch.mrope_positions is None:
                raise ValueError(
                    "VL Ling DSpark requires mrope_positions for draft attention."
                )
            positions = forward_batch.mrope_positions
        hidden = self.forward_embed(input_ids) if input_embeds is None else input_embeds
        residual_batch.start(forward_batch)
        allocator = BumpAllocator(
            buffer_size=max(1, len(self.layers) * 2),
            dtype=torch.float32,
            device=hidden.device,
        )
        for layer in self.layers:
            hidden = layer(
                hidden_states=hidden,
                positions=positions,
                forward_batch=forward_batch,
                zero_allocator=allocator,
            )
        hidden = residual_batch.complete_output(hidden, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden = residual_batch.final_norm(hidden, forward_batch, self.norm)
        return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden)

    def compute_base_logits(self, hidden: torch.Tensor):
        if self.lm_head is None:
            raise ValueError("Ling DSpark requires the target lm_head to be attached.")
        quantized_head = should_apply_lm_head_quant_method(
            self.lm_head, getattr(self.lm_head, "quant_method", None)
        )
        if (
            self._use_fp32_lm_head
            and self.lm_head.weight.is_floating_point()
            and not quantized_head
        ):
            local_logits = F.linear(hidden.float(), self.lm_head.weight.float())
        else:
            local_logits = project_through_lm_head(hidden, self.lm_head)
        return gather_and_crop_vocab(local_logits, self.lm_head), hidden

    def compute_confidence(self, *, anchor_tokens, draft_tokens, confidence_features):
        return self.confidence_head.predict(
            confidence_features,
            anchor_tokens=anchor_tokens,
            draft_tokens=draft_tokens,
            markov_head=self.markov_head,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        if (
            getattr(self, "embed_tokens", None) is not None
            or getattr(self, "lm_head", None) is not None
        ):
            raise ValueError(self._derived_weight_cache_error)
        params = dict(self.named_parameters())
        loaded, unexpected = set(), []
        required_projections, seen_projections = set(), set()
        # One runtime parameter contains multiple logical projections and,
        # under EP, only this rank's experts. The checkpoint iterator is global:
        # require every original projection before accepting the fused alias.
        for name in params:
            if name.endswith((".experts.w13_weight", ".experts.w2_weight")):
                prefix = name.rsplit(".experts.", 1)[0]
                projections = (
                    ("gate_proj", "up_proj")
                    if name.endswith("w13_weight")
                    else ("down_proj",)
                )
                required_projections.update(
                    f"{prefix}.experts.{expert_id}.{projection}.weight"
                    for expert_id in range(int(self.config.num_experts))
                    for projection in projections
                )
            elif name.endswith(".gate_up_proj.weight"):
                prefix = name.removesuffix(".gate_up_proj.weight")
                required_projections.update(
                    f"{prefix}.{projection}.weight"
                    for projection in ("gate_proj", "up_proj")
                )
        pending_a = {}
        expert_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=int(self.config.num_experts),
        )
        for checkpoint_name, value in weights:
            if self.is_bundled:
                if not checkpoint_name.startswith("dspark."):
                    continue
                checkpoint_name = checkpoint_name.removeprefix("dspark.")
            name = (
                "norm.weight"
                if checkpoint_name == "final_layernorm.weight"
                else checkpoint_name
            )
            if name.startswith("confidence_head.") and self.confidence_head is None:
                continue
            if value.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(
                    f"Unquantized Ling DSpark requires BF16/FP32 weights, got {value.dtype} for {checkpoint_name}."
                )
            name = name.replace(".attention.dense.", ".attention.o_proj.")
            if getattr(self.config, "q_lora_rank", None) is not None and (
                ".q_a_proj." in name or ".kv_a_proj_with_mqa." in name
            ):
                pending_a[name] = value
                q_name = name.replace("kv_a_proj_with_mqa", "q_a_proj")
                kv_name = name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if q_name not in pending_a or kv_name not in pending_a:
                    continue
                name = q_name.replace("q_a_proj", "fused_qkv_a_proj_with_mqa")
                value = torch.cat(
                    [pending_a.pop(q_name), pending_a.pop(kv_name)], dim=0
                )
            for suffix, checkpoint_suffix, shard in (
                (".gate_up_proj", ".gate_proj", 0),
                (".gate_up_proj", ".up_proj", 1),
            ):
                if checkpoint_suffix in name and ".experts." not in name:
                    candidate = name.replace(checkpoint_suffix, suffix)
                    if candidate not in params:
                        unexpected.append(checkpoint_name)
                    else:
                        params[candidate].weight_loader(params[candidate], value, shard)
                        loaded.add(candidate)
                        seen_projections.add(name)
                    break
            else:
                for suffix, checkpoint_suffix, expert_id, shard_id in expert_mapping:
                    if checkpoint_suffix not in name:
                        continue
                    candidate = name.replace(checkpoint_suffix, suffix)
                    if candidate not in params:
                        unexpected.append(checkpoint_name)
                    else:
                        params[candidate].weight_loader(
                            params[candidate],
                            value,
                            candidate,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                        loaded.add(candidate)
                        seen_projections.add(name)
                    break
                else:
                    if name not in params:
                        unexpected.append(checkpoint_name)
                    else:
                        param = params[name]
                        getattr(param, "weight_loader", default_weight_loader)(
                            param, value
                        )
                        loaded.add(name)
        missing = sorted(
            name
            for name in set(params) - loaded
            if not name.endswith(
                (
                    ".attn_mha.k_scale",
                    ".attn_mha.v_scale",
                    ".attn_mqa.k_scale",
                    ".attn_mqa.v_scale",
                )
            )
        )
        missing_projections = sorted(required_projections - seen_projections)
        if unexpected or missing or missing_projections or pending_a:
            raise ValueError(
                f"Ling DSpark checkpoint mismatch: unexpected={unexpected[:20]}, missing={missing[:20]}, missing_projections={missing_projections[:20]}, unmatched_qkv={list(pending_a)[:20]}."
            )
        self._post_load_weights()
        return loaded

    def _post_load_weights(self) -> None:
        self._post_load_mla_weights()

    def _post_load_mla_weights(self) -> None:
        for layer in self.layers:
            attn = layer.attention
            weight = attn.kv_b_proj.weight
            if weight.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(
                    "Native Ling DSpark MLA absorption requires BF16/FP32 weights."
                )
            w_kc, w_vc = weight.unflatten(
                0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)
            ).split([attn.qk_nope_head_dim, attn.v_head_dim], dim=1)
            attn.w_kc = bind_or_assign(
                attn.w_kc, w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            )
            attn.w_vc = bind_or_assign(attn.w_vc, w_vc.contiguous().transpose(1, 2))


EntryClass = [LingV3ForCausalLMDSparkMLA]
