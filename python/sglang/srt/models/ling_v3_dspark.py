"""Ling v3 MLA/MoE draft for DSpark speculative decoding."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.factories import layer_stack
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils.common import add_prefix
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.bailing_moe_v3 import (
    BailingMoELinearDecoderLayer,
)
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
from torch import nn


def _prepare_mla_weights(layers):
    """Build MLA views for BF16/FP16 attention; INT4 is routed-expert-only."""
    for layer in layers:
        attn = layer.attention
        weight = attn.kv_b_proj.weight
        if weight.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Ling DSpark requires unquantized MLA attention weights")
        key, value = weight.unflatten(
            0, (-1, attn.qk_nope_head_dim + attn.v_head_dim)
        ).split([attn.qk_nope_head_dim, attn.v_head_dim], dim=1)
        attn.w_kc = bind_or_assign(
            attn.w_kc, key.transpose(1, 2).contiguous().transpose(1, 2)
        )
        attn.w_vc = bind_or_assign(attn.w_vc, value.contiguous().transpose(1, 2))


class LingV3ForCausalLMDSparkMLA(nn.Module):
    """Ling DSpark draft using the native MLA parameterization."""

    supports_pre_gather_target_hidden_projection = False

    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.is_bundled = is_namespaced_dspark_bundle(config)
        self.num_context_features = len(config.target_layer_ids)
        self.block_size = int(config.block_size)
        self.gamma = self.block_size
        self.sample_from_anchor = get_dspark_sample_from_anchor(config)
        self.num_stages = int(config.num_hidden_layers)
        hidden_size = int(config.hidden_size)
        eps = float(getattr(config, "rms_norm_eps", 1e-6))

        self.target_proj = nn.Linear(
            self.num_context_features * hidden_size, hidden_size, bias=False
        )
        self.target_norm = RMSNorm(hidden_size, eps=eps)
        with layer_stack():
            self.layers = nn.ModuleList(
                [
                    BailingMoELinearDecoderLayer(
                        config=config,
                        quant_config=quant_config,
                        layer_id=layer_id,
                        prefix=add_prefix(f"layers.{layer_id}", prefix),
                    )
                    for layer_id in range(int(config.num_hidden_layers))
                ]
            )
        for layer in self.layers:
            layer.attention.attn_mqa.attn_type = AttentionType.ENCODER_ONLY
            layer.attention.attn_mha.attn_type = AttentionType.ENCODER_ONLY
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.markov_head = build_markov_head(config)
        self.confidence_head = build_confidence_head(config)
        self.embed_tokens: nn.Module | None = None
        self.lm_head: nn.Module | None = None
        self._use_fp32_lm_head = envs.SGLANG_DSPARK_FP32_LM_HEAD.get()

    def attach_shared_modules(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module
    ) -> None:
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head

    def forward_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.embed_tokens is None:
            raise ValueError(
                "Ling DSpark draft requires target embeddings to be attached."
            )
        return self.embed_tokens(input_ids)

    def project_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        expected = self.num_context_features * int(self.config.hidden_size)
        if target_hidden.ndim != 2 or target_hidden.shape[-1] != expected:
            raise ValueError(
                "Ling DSpark target hidden shape mismatch: expected "
                f"[N, {expected}], got {tuple(target_hidden.shape)}."
            )
        return self.target_norm(self.target_proj(target_hidden))

    @torch.inference_mode()
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
        if target_hidden_is_projected or swa_loc is not None:
            raise ValueError(
                "Ling MLA draft expects unprojected hidden states and full KV locations."
            )
        if (cache_loc_2d is None) != (commit_lens is None):
            raise ValueError(
                "Ling DSpark MLA requires cache_loc_2d and commit_lens together."
            )
        if cache_loc_2d is not None:
            verify_len = cache_loc_2d.shape[1]
            col = torch.arange(verify_len, device=cache_loc_2d.device).view(1, -1)
            committed = (col < commit_lens.to(torch.long).view(-1, 1)).reshape(-1)
            target_hidden = target_hidden[committed]
            positions = positions[committed]
            cache_loc = cache_loc_2d.reshape(-1)[committed]
            if target_hidden.numel() == 0:
                return
        context = self.project_target_hidden(target_hidden)
        for layer in self.layers:
            attn = layer.attention
            if hasattr(attn, "fused_qkv_a_proj_with_mqa"):
                # q_lora_rank fuses the query and KV down-projections into one
                # GEMM; the KV latent sits behind the leading q_lora_rank slice.
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
            pool.set_mla_kv_buffer(attn.attn_mqa, cache_loc, k_nope, k_rope)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del get_embedding, pp_proxy_tensors
        hidden_states = (
            self.forward_embed(input_ids) if input_embeds is None else input_embeds
        )
        residual_batch.start(forward_batch)
        allocator = BumpAllocator(
            buffer_size=max(1, len(self.layers) * 2),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                zero_allocator=allocator,
            )
        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = residual_batch.final_norm(
                hidden_states, forward_batch, self.norm
            )
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def compute_base_logits(self, hidden: torch.Tensor):
        if self.lm_head is None:
            raise ValueError(
                "Ling DSpark draft requires the target lm_head to be attached."
            )
        weight = self.lm_head.weight
        if self._use_fp32_lm_head and weight.is_floating_point():
            local_logits = F.linear(hidden.float(), weight.float())
        else:
            local_logits = project_through_lm_head(hidden, self.lm_head)
        return gather_and_crop_vocab(local_logits, self.lm_head), None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        loaded, unexpected, a_projections = set(), [], {}

        def load(name, value, *args, **kwargs):
            param = params.get(name)
            if param is None:
                unexpected.append(name)
                return
            getattr(param, "weight_loader", default_weight_loader)(
                param, value, *args, **kwargs
            )
            loaded.add(name)

        stacked = ((".gate_up_proj", ".gate_proj", 0), (".gate_up_proj", ".up_proj", 1))
        expert_projections = {
            "gate_proj": ("w13", "w1"),
            "up_proj": ("w13", "w3"),
            "down_proj": ("w2", "w2"),
        }
        num_experts = int(self.config.num_experts)
        for name, value in weights:
            if self.is_bundled:
                if not name.startswith("dspark."):
                    continue
                name = name.removeprefix("dspark.")
            if name.startswith("confidence_head.") and self.confidence_head is None:
                continue
            if name == "final_layernorm.weight":
                name = "norm.weight"
            name = name.replace(".attention.dense.", ".attention.o_proj.")
            if getattr(self.config, "q_lora_rank", None) is not None and name.endswith(
                (".q_a_proj.weight", ".kv_a_proj_with_mqa.weight")
            ):
                prefix, projection, _ = name.rsplit(".", 2)
                pair = a_projections.setdefault(prefix, {})
                pair[projection] = value
                if len(pair) == 2:
                    load(
                        prefix + ".fused_qkv_a_proj_with_mqa.weight",
                        torch.cat(
                            [pair["q_a_proj"], pair["kv_a_proj_with_mqa"]], dim=0
                        ),
                    )
                    del a_projections[prefix]
                continue
            prefix, separator, tail = name.partition(".mlp.experts.")
            if separator:
                expert, projection, suffix = tail.split(".", 2)
                if (
                    not expert.isdecimal()
                    or int(expert) >= num_experts
                    or projection not in expert_projections
                ):
                    unexpected.append(name)
                    continue
                stem, shard = expert_projections[projection]
                mapped = f"{prefix}.mlp.experts.{stem}_{suffix}"
                load(mapped, value, mapped, shard_id=shard, expert_id=int(expert))
                continue
            for target, source, shard in stacked:
                if source in name:
                    load(name.replace(source, target), value, shard)
                    break
            else:
                load(name, value)

        optional_scales = (
            ".attn_mha.k_scale",
            ".attn_mha.v_scale",
            ".attn_mqa.k_scale",
            ".attn_mqa.v_scale",
        )
        missing = sorted(
            name
            for name in params.keys() - loaded
            if not name.endswith(optional_scales)
        )
        if unexpected or missing or a_projections:
            raise ValueError(
                "Ling DSpark checkpoint does not match the model: "
                f"unexpected={unexpected[:20]}, missing={missing[:20]}, "
                f"incomplete_mla={list(a_projections)[:20]}."
            )
        _prepare_mla_weights(self.layers)
        return loaded


EntryClass = [LingV3ForCausalLMDSparkMLA]
