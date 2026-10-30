"""DSpark draft with MLA attention, cached as per-head K/V (V zero-padded to
the K head dim) because the draft pool indexed by target ids is an MHA pool."""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.model_config import compute_mla_mscale_scaling
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dflash import (
    DFlashDecoderLayer,
    DFlashDraftModel,
    _get_dflash_layer_attention_params,
)
from sglang.srt.models.dspark import DSparkDraftMixin, DSparkDraftModel
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.hf_transformers_utils import get_rope_config


class DFlashMLAAttention(nn.Module):
    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        # Heads shard by attention TP, as the draft KV pool does; attention CP
        # makes it narrower than TP.
        attn_tp_size = int(get_parallel().attn_tp_size)
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % attn_tp_size:
            raise ValueError(
                f"MLA DSpark draft needs num_attention_heads={self.total_num_heads} "
                f"divisible by the attention TP size {attn_tp_size}."
            )
        self.num_heads = self.total_num_heads // attn_tp_size
        self.num_kv_heads = self.num_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if self.v_head_dim > self.head_dim:
            raise ValueError(
                f"MLA DSpark draft caches V in the K head dim {self.head_dim}; "
                f"v_head_dim={self.v_head_dim} does not fit."
            )
        rms_norm_eps = float(getattr(config, "rms_norm_eps", 1e-6))

        def sub(name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        q_width = self.total_num_heads * self.head_dim
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                hidden_size,
                int(self.q_lora_rank),
                bias=False,
                quant_config=quant_config,
                prefix=sub("q_a_proj"),
            )
            self.q_a_layernorm = RMSNorm(int(self.q_lora_rank), eps=rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                int(self.q_lora_rank),
                q_width,
                bias=False,
                quant_config=quant_config,
                prefix=sub("q_b_proj"),
                parallel_group="attn_tp",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                hidden_size,
                q_width,
                bias=False,
                quant_config=quant_config,
                prefix=sub("q_proj"),
                parallel_group="attn_tp",
            )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=sub("kv_a_proj_with_mqa"),
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=sub("kv_b_proj"),
            parallel_group="attn_tp",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=sub("o_proj"),
            parallel_group="attn_tp",
            use_dp_attention_reduce=True,
        )

        rope_theta, rope_scaling = get_rope_config(config)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=int(getattr(config, "max_position_embeddings", 32768)),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=not getattr(config, "rope_interleave", True),
        )
        self.scaling = self.head_dim**-0.5
        if rope_scaling and rope_scaling.get("apply_yarn_scaling", True):
            self.scaling = compute_mla_mscale_scaling(rope_scaling, self.scaling)

        self.sliding_window_size, self.attn_type = _get_dflash_layer_attention_params(
            config, layer_id
        )
        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            attn_type=self.attn_type,
            quant_config=quant_config,
        )

    def project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q_latent, _ = self.q_a_proj(hidden_states)
            q, _ = self.q_b_proj(self.q_a_layernorm(q_latent))
        else:
            q, _ = self.q_proj(hidden_states)
        return q.view(-1, self.num_heads, self.head_dim)

    def project_kv(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-head K without RoPE, the shared un-rotated k_pe, and padded V."""
        latent_pe, _ = self.kv_a_proj_with_mqa(hidden_states)
        latent, k_pe = latent_pe.split([self.kv_lora_rank, self.qk_rope_head_dim], -1)
        kv, _ = self.kv_b_proj(self.kv_a_layernorm(latent.contiguous()))
        k_nope, v = kv.view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        ).split([self.qk_nope_head_dim, self.v_head_dim], -1)
        v = F.pad(v, (0, self.head_dim - self.v_head_dim))
        return k_nope, k_pe.unsqueeze(1).contiguous(), v

    def assemble_k(self, k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
        return torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1)], dim=-1)

    def context_kv(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k_nope, k_pe, v = self.project_kv(hidden_states)
        dummy_q = k_pe.new_empty(k_pe.shape)
        _, k_pe = self.rotary_emb(positions, dummy_q, k_pe)
        return self.assemble_k(k_nope, k_pe), v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        q = self.project_q(hidden_states)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], -1)
        k_nope, k_pe, v = self.project_kv(hidden_states)
        q_pe, k_pe = self.rotary_emb(positions, q_pe.contiguous(), k_pe)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = self.assemble_k(k_nope, k_pe)
        attn_output = self.attn(q, k, v, forward_batch)
        attn_output = attn_output.view(-1, self.num_heads, self.head_dim)[
            ..., : self.v_head_dim
        ]
        output, _ = self.o_proj(
            attn_output.reshape(-1, self.num_heads * self.v_head_dim)
        )
        return output


class DFlashMLADecoderLayer(DFlashDecoderLayer):
    attention_cls = DFlashMLAAttention


_STACKED_MLP_WEIGHTS = (
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)
_HEAD_PREFIXES = ("markov_head.", "confidence_head.")


class DFlashMLADraftModel(DFlashDraftModel):
    decoder_layer_cls = DFlashMLADecoderLayer
    supports_fused_context_kv = False

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Strict: a silently random draft parameter only shows up as low acceptance.
        params_dict = dict(self.named_parameters())
        modules = dict(self.named_modules())
        loaded_shards = {}
        for name, loaded_weight in weights:
            name = name.removeprefix("model.")
            for param_name, weight_name, shard_id in _STACKED_MLP_WEIGHTS:
                if f".{weight_name}." in name:
                    target, shard = name.replace(weight_name, param_name), shard_id
                    break
            else:
                target, shard = name, None
            param = params_dict.get(target)
            if param is None:
                raise ValueError(
                    f"MLA DSpark draft checkpoint weight {name!r} has no matching "
                    "model parameter."
                )
            module_name, _, attr = target.rpartition(".")
            expected = _checkpoint_shape(modules.get(module_name), attr, shard, param)
            if tuple(loaded_weight.shape) != expected:
                raise ValueError(
                    f"MLA DSpark draft weight {name!r} has shape "
                    f"{tuple(loaded_weight.shape)}, expected {expected}."
                )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard)
            loaded_shards.setdefault(target, set()).add(shard)

        missing = []
        for name in params_dict:
            if name.startswith(_HEAD_PREFIXES):
                continue
            shards = loaded_shards.get(name, set())
            if None in shards:
                continue
            module = modules.get(name.rpartition(".")[0])
            if isinstance(module, MergedColumnParallelLinear):
                missing += [
                    f"{name}[shard {i}]"
                    for i in range(len(module.output_sizes))
                    if i not in shards
                ]
            elif not shards:
                missing.append(name)
        if missing:
            raise ValueError(
                f"MLA DSpark draft checkpoint is missing weights: {missing}."
            )


def _checkpoint_shape(
    module: Optional[nn.Module], attr: str, shard: Optional[int], param: nn.Parameter
) -> Tuple[int, ...]:
    """The unsharded shape a checkpoint stores; TP loaders slice it per rank."""
    if not isinstance(module, LinearBase) or attr not in ("weight", "bias"):
        return tuple(param.shape)
    output_size = module.output_size if shard is None else module.output_sizes[shard]
    return (output_size, module.input_size) if attr == "weight" else (output_size,)


class MLADSparkModel(DSparkDraftMixin, DFlashMLADraftModel):
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        weights = list(weights)
        super().load_weights(weights)
        names = {name for name, _ in weights}
        missing = sorted(
            name
            for name, _ in self.markov_head.named_parameters(prefix="markov_head")
            if name not in names
        )
        if missing:
            raise ValueError(
                f"MLA DSpark draft checkpoint is missing Markov head weights: {missing}."
            )

    prune_to_ctx_kv_injection = DSparkDraftModel.prune_to_ctx_kv_injection

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
        for layer in self.layers:
            attn = layer.self_attn
            k, v = attn.context_kv(
                positions, self.prepare_context_hidden_for_kv(layer, ctx_hidden)
            )
            if cache_loc_2d is not None and commit_lens is not None:
                pool.set_kv_buffer_prefix_valid(
                    attn.attn,
                    cache_loc_2d,
                    commit_lens,
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )
            else:
                pool.set_kv_buffer(
                    attn.attn,
                    KVWriteLoc(cache_loc, physical=True),
                    k,
                    v,
                    attn.attn.k_scale,
                    attn.attn.v_scale,
                )


# fmt: off
_MTP_DSPARK_NAMES = (
    (r"backbone\.layers\.(\d+)\.attention\.wq\.(\w+)", r"layers.\1.self_attn.q_proj.\2"),
    (r"backbone\.layers\.(\d+)\.attention\.wkv_a_with_mqa\.(\w+)", r"layers.\1.self_attn.kv_a_proj_with_mqa.\2"),
    (r"backbone\.layers\.(\d+)\.attention\.kv_a_norm\.(\w+)", r"layers.\1.self_attn.kv_a_layernorm.\2"),
    (r"backbone\.layers\.(\d+)\.attention\.wkv_b\.(\w+)", r"layers.\1.self_attn.kv_b_proj.\2"),
    (r"backbone\.layers\.(\d+)\.attention\.wo\.(\w+)", r"layers.\1.self_attn.o_proj.\2"),
    (r"backbone\.layers\.(\d+)\.attention_norm\.(\w+)", r"layers.\1.input_layernorm.\2"),
    (r"backbone\.layers\.(\d+)\.ffn_norm\.(\w+)", r"layers.\1.post_attention_layernorm.\2"),
    (r"backbone\.layers\.(\d+)\.feed_forward\.w1\.(\w+)", r"layers.\1.mlp.gate_proj.\2"),
    (r"backbone\.layers\.(\d+)\.feed_forward\.w3\.(\w+)", r"layers.\1.mlp.up_proj.\2"),
    (r"backbone\.layers\.(\d+)\.feed_forward\.w2\.(\w+)", r"layers.\1.mlp.down_proj.\2"),
    (r"backbone\.target_projection\.(\w+)", r"fc.\1"),
    (r"backbone\.target_norm\.(\w+)", r"hidden_norm.\1"),
    (r"backbone\.final_norm\.(\w+)", r"norm.\1"),
    (r"backbone\.mask_embedding", r"mask_embedding"),
    (r"autoregressive_head\.embedding\.(\w+)", r"markov_head.markov_w1.\1"),
    (r"autoregressive_head\.output\.(\w+)", r"markov_head.markov_w2.\1"),
    (r"autoregressive_head\.joint_proj\.(\w+)", r"markov_head.joint_proj.\1"),
)
# fmt: on
_MTP_DSPARK_PREFIX = "mtp_dspark."


def mistral_large_4_dspark_weight_name(name: str) -> Optional[str]:
    """Map a consolidated checkpoint tensor to its draft parameter, or None
    for the target's tensors."""
    if not name.startswith(_MTP_DSPARK_PREFIX):
        return None
    name = name.removeprefix(_MTP_DSPARK_PREFIX)
    for pattern, target in _MTP_DSPARK_NAMES:
        if re.fullmatch(pattern, name):
            return re.sub(pattern, target, name)
    raise ValueError(f"Unrecognized Mistral Large 4 DSpark weight: mtp_dspark.{name}")


def _plain_yarn_rope(config) -> DeepseekScalingRotaryEmbedding:
    # get_rope would return the target's cached module, whose cos/sin cache
    # the target rewrites in place with LeRoPE; the draft is trained without it.
    rope_theta, rope_scaling = get_rope_config(config)
    return DeepseekScalingRotaryEmbedding(
        config.qk_rope_head_dim,
        config.qk_rope_head_dim,
        rope_scaling["original_max_position_embeddings"],
        rope_theta,
        not getattr(config, "rope_interleave", True),
        rope_scaling["factor"],
        torch.get_default_dtype(),
        device=torch.get_default_device(),
        **{
            key: value
            for key, value in rope_scaling.items()
            if key
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
        },
    )


class MistralLarge4DSparkModel(MLADSparkModel):
    """The DSpark draft bundled in a Mistral Large 4 checkpoint (``mtp_dspark.*``).

    Each block is ``[anchor, mask_1, ..., mask_{D-1}]``; a ``position_specific``
    checkpoint stores one mask row per slot. Base logits take the target's
    soft cap before the Markov bias is added.
    """

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        hidden_size = int(config.hidden_size)
        mask_shape = (hidden_size,)
        if config.dspark_draft_input == "position_specific":
            mask_shape = (int(config.dspark_block_size) - 1, hidden_size)
        self.mask_embedding = nn.Parameter(torch.empty(mask_shape), requires_grad=False)
        self.final_logit_softcapping = config.softmax_tanh_gating
        rotary_emb = _plain_yarn_rope(config)
        for layer in self.layers:
            layer.self_attn.rotary_emb = rotary_emb

    def embed_blocks(self, input_ids: torch.Tensor, block_size: int) -> torch.Tensor:
        if self.mask_embedding.ndim == 2 and block_size != len(self.mask_embedding) + 1:
            raise ValueError(
                "The DSpark checkpoint has position-specific masks for "
                f"{len(self.mask_embedding) + 1}-token blocks, got {block_size}; "
                "do not override --speculative-num-draft-tokens."
            )
        blocks = input_ids.view(-1, block_size)
        anchors = self.embed_tokens(blocks[:, 0].contiguous())
        masks = self.mask_embedding.to(anchors.dtype).expand(
            anchors.shape[0], block_size - 1, -1
        )
        return torch.cat((anchors.unsqueeze(1), masks), dim=1).flatten(0, 1)

    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        # A caller-embedded block looks the mask slots up in the target vocab.
        if input_ids.numel():
            input_embeds = self.embed_blocks(
                input_ids, input_ids.numel() // forward_batch.batch_size
            )
        return super().forward(
            input_ids, positions, forward_batch, input_embeds=input_embeds, **kwargs
        )

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        base_logits, confidence_features = super().compute_base_logits(hidden)
        cap = self.final_logit_softcapping
        if cap:
            base_logits = torch.tanh(base_logits / cap) * cap
        return base_logits, confidence_features

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(
            (target, weight)
            for name, weight in weights
            if (target := mistral_large_4_dspark_weight_name(name)) is not None
        )


EntryClass = [MLADSparkModel, MistralLarge4DSparkModel]
