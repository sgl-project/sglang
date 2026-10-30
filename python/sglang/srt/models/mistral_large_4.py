import math
from collections.abc import Iterable, Iterator
from typing import Optional

import regex as re
import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.linear import ColumnParallelLinear, ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import (
    _BLOCK_FP8_ALGOS,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import sharded_weight_loader
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2DecoderLayer,
    DeepseekV2Model,
    DeepseekV3ForCausalLM,
)
from sglang.srt.models.mistral_large_3 import MistralLarge3ForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk
from sglang.srt.utils import add_prefix, set_weight_attrs

_LEROPE_ALPHA = "model.lerope_alpha"


def kv_source_layer_ids(config: PretrainedConfig) -> list[int]:
    kv_sharing_config = getattr(config, "kv_sharing_config", None)
    if kv_sharing_config is None:
        return list(range(config.num_hidden_layers))
    return [int(source) for source in kv_sharing_config["source_layers_str"].split(",")]


def layer_flag(config: PretrainedConfig, name: str, layer_id: int) -> bool:
    flags = getattr(config, name, None)
    return flags is not None and bool(flags[layer_id])


class IdentityRotaryEmbedding(nn.Module):
    """A cos=1, sin=0 table: the fused indexer kernels then leave the rope slice as is."""

    def __init__(self, like: DeepseekScalingRotaryEmbedding):
        super().__init__()
        rows, width = like.cos_sin_cache.shape
        cache = torch.cat(
            [
                torch.ones(rows, width // 2, dtype=torch.float32),
                torch.zeros(rows, width // 2, dtype=torch.float32),
            ],
            dim=-1,
        ).to(like.cos_sin_cache.device)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        self.head_size = like.head_size
        self.rotary_dim = like.rotary_dim
        self.is_neox_style = like.is_neox_style

    def forward(self, positions, query, key, *args, **kwargs):
        return query, key


class MistralLarge4AttentionMLA(DeepseekV2AttentionMLA):
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        layer_id: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        skip_rope: bool = False,
        **kwargs,
    ) -> None:
        has_rope = not layer_flag(config, "nope_layer_flags", layer_id)
        # NoPE layers fold the rope width into the content head and keep a
        # zero rope slice so the latent cache layout stays uniform.
        super().__init__(
            config=config,
            layer_id=layer_id,
            qk_nope_head_dim=(
                qk_nope_head_dim if has_rope else qk_nope_head_dim + qk_rope_head_dim
            ),
            qk_rope_head_dim=qk_rope_head_dim,
            quant_config=quant_config,
            prefix=prefix,
            skip_rope=skip_rope or not has_rope,
            **kwargs,
        )
        self.has_rope = has_rope
        self.kv_source_layer_id = kv_source_layer_ids(config)[layer_id]
        if not has_rope:
            self._build_nope_projections(quant_config=quant_config, prefix=prefix)
        if self.kv_source_layer_id != layer_id:
            self.kv_a_proj_with_mqa = None
            self.kv_a_layernorm = None
            self.save_kv_cache = False

        self.ssmax_scaling_s: Optional[nn.Parameter] = None
        self.ssmax_scaling_b: Optional[nn.Parameter] = None
        if config.ssmax_scaling:
            self.ssmax_scaling_s = self._sharded_head_param()
            self.ssmax_scaling_b = self._sharded_head_param()
        if self.indexer is not None:
            self.indexer.q_input_norm = LayerNorm(
                self.hidden_size, eps=config.indexer_norm_eps
            )

    def _build_nope_projections(
        self, *, quant_config: Optional[QuantizationConfig], prefix: str
    ) -> None:
        self.qk_head_dim = self.qk_nope_head_dim
        self.scaling = self.qk_head_dim**-0.5
        self.attn_mqa.scaling = self.scaling
        self.attn_mha.scaling = self.scaling
        if hasattr(self, "attn_mqa_for_dcp_decode"):
            self.attn_mqa_for_dcp_decode.scaling = self.scaling
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.qk_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
            parallel_group="attn_tp",
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_a_proj_with_mqa", prefix),
        )

    def is_dcp_q_replicate_phase(self, forward_batch: ForwardBatch) -> bool:
        # prepare_q_latent_without_q_lora projects (and SSMax-scales) local
        # heads only, so DCP must all-gather Q rather than replicate it.
        return False

    def _sharded_head_param(self) -> nn.Parameter:
        param = nn.Parameter(
            torch.empty(self.num_local_heads, dtype=torch.float32),
            requires_grad=False,
        )
        set_weight_attrs(param, {"weight_loader": sharded_weight_loader(0)})
        return param

    def _apply_ssmax(self, q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.ssmax_scaling_s is None:
            return q
        factor = (
            torch.log1p(positions.to(torch.float32)).unsqueeze(-1)
            * self.ssmax_scaling_s
            + self.ssmax_scaling_b
        )
        return q * factor.unsqueeze(-1).to(q.dtype)

    def prepare_q_latent_without_q_lora(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        q_replicate_active: bool,
        prev_topk_indices: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        q = self.q_proj(hidden_states)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )
        # SSMax is a per-token, per-head scalar, so it commutes with RoPE.
        q = self._apply_ssmax(q, positions)
        if self.save_kv_cache:
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = self.kv_a_layernorm(latent_cache[..., : self.kv_lora_rank])
        else:
            # Attention reads the provider's cache; this placeholder is never stored.
            latent_dim = self.kv_lora_rank + (
                self.qk_rope_head_dim if self.has_rope else 0
            )
            latent_cache = hidden_states.new_zeros(hidden_states.shape[0], latent_dim)
            k_nope = latent_cache[..., : self.kv_lora_rank]
        topk_indices = self._select_topk(
            positions, hidden_states, forward_batch, prev_topk_indices
        )
        return q, latent_cache, k_nope.unsqueeze(1), topk_indices

    def _select_topk(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        prev_topk_indices: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_dsa:
            return None
        if not self.should_run_indexer(prev_topk_indices):
            return maybe_capture_indexer_topk(self.layer_id, prev_topk_indices)
        return self.indexer(
            x=hidden_states,
            q_lora=self.indexer.q_input_norm(hidden_states),
            positions=positions,
            forward_batch=forward_batch,
            layer_id=self.layer_id,
        )

    def _split_q_nope_pe(
        self, q: torch.Tensor, latent_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.has_rope:
            return super()._split_q_nope_pe(q, latent_cache)
        q_pe = q.new_zeros(*q.shape[:-1], self.qk_rope_head_dim)
        k_pe = latent_cache.new_zeros(latent_cache.shape[0], 1, self.qk_rope_head_dim)
        return q, q_pe, k_pe


class MistralLarge4DecoderLayer(DeepseekV2DecoderLayer):
    attention_cls = MistralLarge4AttentionMLA


class MistralLarge4Model(DeepseekV2Model):
    decoder_layer_cls = MistralLarge4DecoderLayer

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix=prefix)
        self.lerope_alpha: Optional[nn.Parameter] = None
        if config.lerope:
            self.lerope_alpha = nn.Parameter(
                torch.empty(config.qk_rope_head_dim // 2, dtype=torch.float32),
                requires_grad=False,
            )
        self._share_provider_attention()
        self._skip_rope_on_nope_indexers()

    def local_attention_layers(self) -> list[MistralLarge4AttentionMLA]:
        return [
            self.layers[layer_id].self_attn
            for layer_id in range(self.start_layer, self.end_layer)
        ]

    def _share_provider_attention(self) -> None:
        for attn in self.local_attention_layers():
            source = attn.kv_source_layer_id
            if source == attn.layer_id:
                continue
            if source < self.start_layer:
                raise NotImplementedError(
                    f"Layer {attn.layer_id} reads the KV cache of layer {source}, "
                    "which belongs to another pipeline stage."
                )
            provider = self.layers[source].self_attn
            # Graph capture indexes attention modules by layer_id, so a consumer
            # must hold the provider's modules rather than equal copies.
            attn.attn_mqa = provider.attn_mqa
            attn.attn_mha = provider.attn_mha
            # Under DCP every DSA attention call goes through this module.
            if hasattr(provider, "attn_mqa_for_dcp_decode"):
                attn.attn_mqa_for_dcp_decode = provider.attn_mqa_for_dcp_decode

    def _skip_rope_on_nope_indexers(self) -> None:
        identity = None
        for attn in self.local_attention_layers():
            if attn.indexer is None or not layer_flag(
                self.config, "indexer_nope_layer_flags", attn.layer_id
            ):
                continue
            if identity is None:
                identity = IdentityRotaryEmbedding(attn.indexer.rotary_emb)
            attn.indexer.rotary_emb = identity


_ML3_REMAPPING = {
    pattern: target
    for pattern, target in MistralLarge3ForCausalLM.remapping.items()
    if "fake_quantizer" not in pattern
}

# fmt: off
MISTRAL_LARGE_4_REMAPPING = {
    **_ML3_REMAPPING,
    r"layers\.(\d+)\.attention\.ssmax_scaling\.([sb])": r"model.layers.\1.self_attn.ssmax_scaling_\2",  # noqa: E501
    r"layers\.(\d+)\.attention\.indexer\.(wq_b|wk|weights_proj|k_norm|q_input_norm)\.(\w+)": r"model.layers.\1.self_attn.indexer.\2.\3",  # noqa: E501
    r"layers\.(\d+)\.attention\.indexer\.ssmax_scaling\.([sb])": r"model.layers.\1.self_attn.indexer.ssmax_scaling_\2",  # noqa: E501
    r"lerope\.alpha": _LEROPE_ALPHA,
}
# fmt: on

_ROUTED_EXPERTS = re.compile(r"layers\.(\d+)\.experts")
_PACKED_SHARDS = {".gate_proj": ".gate_up_proj", ".up_proj": ".gate_up_proj"}

_KV_PROJECTION = re.compile(r"layers\.(\d+)\.attention\.(?:wkv_a_with_mqa|kv_a_norm)\.")
# The Pixtral wrapper loads these itself; a text-only config leaves them unused.
_VISION_PREFIXES = (
    "vision_encoder.",
    "patch_merger.",
    "pre_mm_projector_norm.",
    "vision_language_adapter.",
)


def mistral_large_4_weight_name(
    name: str, *, kv_sources: list[int], block_quant: bool
) -> Optional[str]:
    """Map a consolidated checkpoint tensor to its parameter, or None to skip it."""
    if name.startswith(("mtp_", *_VISION_PREFIXES)):
        return None
    if block_quant and name.endswith(".qscale_act"):
        # Block FP8 quantizes activations dynamically.
        return None
    projection = _KV_PROJECTION.match(name)
    if projection is not None:
        layer_id = int(projection.group(1))
        if kv_sources[layer_id] != layer_id:
            return None

    for pattern, target in MISTRAL_LARGE_4_REMAPPING.items():
        if re.fullmatch(pattern, name):
            name = re.sub(pattern, target, name)
            break
    else:
        raise ValueError(f"Unrecognized Mistral Large 4 weight: {name}")

    if name.endswith(".qscale_weight"):
        suffix = "weight_scale_inv" if block_quant else "weight_scale"
        return name.removesuffix(".qscale_weight") + f".{suffix}"
    if name.endswith(".qscale_act"):
        return name.removesuffix(".qscale_act") + ".input_scale"
    return name


def mistral_large_4_modelopt_layers(quantized_layers: dict) -> dict:
    """Rename a ModelOpt ``quantized_layers`` map from checkpoint modules
    (``layers.1.attention.wq``, ``layers.1.experts``) to this model's modules."""
    mapped: dict = {}
    for name, info in quantized_layers.items():
        experts = _ROUTED_EXPERTS.fullmatch(name)
        if experts is not None:
            modules = [f"model.layers.{experts.group(1)}.mlp.experts"]
        else:
            weight = f"{name}.weight"
            for pattern, target in MISTRAL_LARGE_4_REMAPPING.items():
                if re.fullmatch(pattern, weight):
                    break
            else:
                raise ValueError(f"Unrecognized Mistral Large 4 ModelOpt layer: {name}")
            module = re.sub(pattern, target, weight).removesuffix(".weight")
            modules = [module]
            for shard, fused in _PACKED_SHARDS.items():
                if module.endswith(shard):
                    modules.append(module.removesuffix(shard) + fused)
        for module in modules:
            if mapped.setdefault(module, info) != info:
                raise ValueError(
                    f"ModelOpt layers fused into {module} use different "
                    f"quantization: {mapped[module]} and {info}"
                )
    return mapped


def _check_context_parallel_layout() -> None:
    parallel = get_parallel()
    if parallel.dcp_enabled and parallel.pp_size > 1:
        raise NotImplementedError(
            "Mistral Large 4 does not support decode context parallelism "
            "with pipeline parallelism."
        )
    if parallel.attn_cp_size <= 1:
        return
    # Zigzag writes the gathered KV inside the strategy and returns nothing,
    # so the DSA MLA forward has no KV to unpack.
    if parallel.cp_strategy != "interleave":
        raise NotImplementedError(
            "Mistral Large 4 prefill context parallelism requires "
            f"--cp-strategy interleave, got {parallel.cp_strategy!r}."
        )
    if parallel.enable_dsa_cache_layer_split:
        raise NotImplementedError(
            "Mistral Large 4 does not support --enable-dsa-cache-layer-split: "
            "its KV-sharing layers read another layer's cache."
        )
    if parallel.pp_size > 1:
        raise NotImplementedError(
            "Mistral Large 4 does not support prefill context parallelism "
            "with pipeline parallelism."
        )


class MistralLarge4ForCausalLM(DeepseekV3ForCausalLM):
    model_cls = MistralLarge4Model

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        if config.index_kv_dtype != "fp8":
            raise NotImplementedError(
                f"index_kv_dtype={config.index_kv_dtype!r} is not supported; "
                "the DSA indexer cache is FP8."
            )
        _check_context_parallel_layout()
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            quant_config.quantized_layers = mistral_large_4_modelopt_layers(
                quant_config.quantized_layers
            )
        super().__init__(config, quant_config, prefix=prefix)
        self.logits_processor.final_logit_softcapping = config.softmax_tanh_gating

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        loaded_names: set[str] = set()
        super().load_weights(self._remap_weights(weights, loaded_names))
        if self.model.lerope_alpha is not None:
            if _LEROPE_ALPHA not in loaded_names:
                raise ValueError(
                    "The config enables LeRoPE but the checkpoint has no lerope.alpha."
                )
            self._refresh_lerope()
        self._check_indexer_ssmax_positive()

    def _remap_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], loaded_names: set[str]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        kv_sources = kv_source_layer_ids(self.config)
        # Online requantization configs (e.g. nvfp4_online) still load the dense
        # layers of this block-FP8 checkpoint through the block-FP8 linear method.
        quant_config = self.quant_config
        fp8_checkpoint = isinstance(quant_config, Fp8Config) or getattr(
            quant_config, "is_checkpoint_fp8_serialized", False
        )
        block_quant = (
            fp8_checkpoint
            and getattr(quant_config, "weight_block_size", None) is not None
        )
        mixed_precision = isinstance(quant_config, ModelOptMixedPrecisionConfig)
        for name, tensor in weights:
            target = mistral_large_4_weight_name(
                name, kv_sources=kv_sources, block_quant=block_quant
            )
            if (
                mixed_precision
                and target is not None
                and target.endswith(".weight_scale")
                and quant_config.resolve_quant_algo(
                    target.removesuffix(".weight_scale")
                )
                in _BLOCK_FP8_ALGOS
            ):
                # ModelOpt stores block-FP8 scales as [out_blocks, 1, in_blocks, 1].
                target = target.removesuffix(".weight_scale") + ".weight_scale_inv"
                tensor = tensor.squeeze(-1).squeeze(1)
            if target is not None:
                loaded_names.add(target)
                yield target, tensor

    def _refresh_lerope(self) -> None:
        rotary_embs = {}
        for attn in self.model.local_attention_layers():
            for rotary_emb in (
                attn.rotary_emb,
                attn.indexer.rotary_emb if attn.indexer is not None else None,
            ):
                if rotary_emb is None or isinstance(
                    rotary_emb, IdentityRotaryEmbedding
                ):
                    continue
                if not isinstance(rotary_emb, DeepseekScalingRotaryEmbedding):
                    raise NotImplementedError(
                        f"LeRoPE does not support {type(rotary_emb).__name__}."
                    )
                rotary_embs[id(rotary_emb)] = rotary_emb
        for rotary_emb in rotary_embs.values():
            rotary_emb.set_lerope_alpha(self.model.lerope_alpha)

    def _check_indexer_ssmax_positive(self) -> None:
        log_max_position = math.log(self.config.max_position_embeddings)
        for attn in self.model.local_attention_layers():
            if attn.indexer is None or attn.indexer.ssmax_scaling_s is None:
                continue
            s = attn.indexer.ssmax_scaling_s
            b = attn.indexer.ssmax_scaling_b
            # The factor is affine in log(pos + 1), so its minimum is at an endpoint.
            min_factor = torch.minimum(b, b + log_max_position * s).min().item()
            if min_factor <= 0:
                raise ValueError(
                    f"Layer {attn.layer_id} indexer SSMax factor reaches {min_factor}; "
                    "folding it into the head gates requires a positive factor."
                )


EntryClass = MistralLarge4ForCausalLM
