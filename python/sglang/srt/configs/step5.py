# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 checkpoint configuration, vocabulary bounds and model profiles."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Union

import msgspec
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.step3p7 import Step3p7VisionEncoderConfig

# Adapted from the reviewed vLLM Step5Config at releases/v0.27.0 plus the
# Step5 architecture values are validated by public test fixtures.


def get_step5_text_config(config: PretrainedConfig) -> "Step5Config | None":
    """Resolve the Step5 backbone in a text checkpoint or a Step3.7 VLM."""
    text_config = getattr(config, "text_config", config)
    return text_config if isinstance(text_config, Step5Config) else None


def _consume_alias(*, kwargs: dict[str, Any], name: str, expected: Any) -> None:
    value = kwargs.pop(name, None)
    if value is not None and value != expected:
        raise ValueError(f"Step5 {name}={value!r} conflicts with {expected!r}.")


def _check_layer_array(*, name: str, values: list | None, total_layers: int) -> None:
    if values and len(values) < total_layers:
        raise ValueError(
            f"Step5 {name} must cover the main and MTP layers: expected at least "
            f"{total_layers} entries, got {len(values)}."
        )


def _resolve_head_dim(*, hidden_size: int, num_heads: int, head_dim: int | None) -> int:
    if num_heads <= 0:
        raise ValueError("Step5 attention head counts must be positive.")
    if head_dim is None:
        if hidden_size % num_heads:
            raise ValueError(
                "Step5 hidden_size must be divisible by num_heads when head_dim "
                f"is omitted, got hidden_size={hidden_size}, num_heads={num_heads}."
            )
        return hidden_size // num_heads
    return head_dim


def _parse_moe_layers(*, value: str | list[int] | None, name: str) -> set[int] | None:
    if value is None:
        return None
    raw = value.split(",") if isinstance(value, str) else value
    try:
        indices = [int(item) for item in raw if str(item).strip()]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Step5 {name} must contain integer layer indices.") from exc
    if len(indices) != len(set(indices)):
        raise ValueError(f"Step5 {name} contains duplicate layer indices.")
    return set(indices)


class Step5Config(PretrainedConfig):
    model_type = "step5"
    architectures = ["Step5ForCausalLM"]

    @property
    def index_head_dim(self) -> int:
        """Internal name for the checkpoint sparse_config.proxy_dim field.

        The external config keeps "proxy_dim"; runtime code uses index_head_dim
        for the indexer dimension, distinct from the main attention head_dim.
        This property adds no serialized key and leaves checkpoint JSON unchanged.
        """
        return self.sparse_config["proxy_dim"]

    def __init__(
        self,
        hidden_size: int = 5120,
        intermediate_size: int = 13312,
        num_attention_heads: int = 40,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 48,
        max_seq_len: int = 4096,
        vocab_size: int = 65536,
        valid_vocab_size: int | None = None,
        tie_word_embeddings: bool = False,
        rms_norm_eps: float = 1e-5,
        use_moe: bool = False,
        moe_intermediate_size: int = 10240,
        moe_num_experts: int = 16,
        moe_top_k: int = 4,
        rope_theta: float | list[float] | None = 500000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        share_expert_dim: int | None = None,
        norm_dtype: str = "bf16",
        norm_expert_weight: bool = True,
        bos_token_id: list[int] | int | None = None,
        eos_token_id: list[int] | int | None = None,
        pad_token_id: int | None = None,
        moe_router_activation: str = "softmax",
        moe_router_scaling_factor: float = 1.0,
        att_impl_type: str = "GQA",
        use_head_wise_attn_gate: bool = False,
        use_moe_router_bias: bool = False,
        need_fp32_gate: bool = False,
        layer_types: list[str] | None = None,
        use_rope_layers: list[bool] | None = None,
        partial_rotary_factors: list[float] | None = None,
        yarn_only_types: list[str] | None = None,
        attention_other_setting: dict[str, Any] | None = None,
        num_nextn_predict_layers: int = 0,
        swa_num_attention_heads: int | None = None,
        swiglu_limits: list[float] | None = None,
        swiglu_limits_shared: list[float] | None = None,
        zero_centered: bool = True,
        fp32_residual_connection: bool = False,
        max_position_embeddings: int | None = None,
        sliding_window: int | None = None,
        mtp_image_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._validate_sizes(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_attention_groups=num_attention_groups,
            num_hidden_layers=num_hidden_layers,
            num_nextn_predict_layers=num_nextn_predict_layers,
            max_seq_len=max_seq_len,
            max_position_embeddings=max_position_embeddings,
            kwargs=kwargs,
        )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = (
            max_seq_len if max_position_embeddings is None else max_position_embeddings
        )
        self.vocab_size = vocab_size
        self.mtp_image_token_id = mtp_image_token_id
        # Optional serving override. ModelConfig resolves tokenizer metadata;
        # vocab_size remains the physical embedding/head checkpoint width.
        self.valid_vocab_size = valid_vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.use_moe = use_moe
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        _consume_alias(kwargs=kwargs, name="num_experts_per_tok", expected=moe_top_k)
        self.num_experts_per_tok = moe_top_k
        self.share_expert_dim = (
            moe_intermediate_size * moe_top_k
            if share_expert_dim is None
            else share_expert_dim
        )
        self.norm_dtype = norm_dtype
        self.norm_expert_weight = norm_expert_weight
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.use_moe_router_bias = use_moe_router_bias
        self.need_fp32_gate = need_fp32_gate
        self.att_impl_type = att_impl_type
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.zero_centered = zero_centered
        self.fp32_residual_connection = fp32_residual_connection
        self.sliding_window = sliding_window
        self.attention_other_setting = deepcopy(attention_other_setting)
        self.swa_num_attention_heads = swa_num_attention_heads
        self._init_layer_types(layer_types=layer_types, kwargs=kwargs)
        self._init_attention_aliases(head_dim=head_dim, kwargs=kwargs)
        self._init_layer_arrays(
            rope_theta=rope_theta,
            use_rope_layers=use_rope_layers,
            partial_rotary_factors=partial_rotary_factors,
            swiglu_limits_shared=swiglu_limits_shared,
        )
        self.rope_scaling = deepcopy(rope_scaling)
        self.yarn_only_types = deepcopy(yarn_only_types)
        # Routed MoE permits missing trailing limits in the source; its
        # default is no clamp. Shared/dense MLP uses strict per-layer indexing.
        self.swiglu_limits = deepcopy(swiglu_limits)
        self._init_moe_layers(kwargs=kwargs)
        if kwargs.get("architectures") is None:
            kwargs["architectures"] = list(type(self).architectures)
        super().__init__(
            bos_token_id=1 if bos_token_id is None else bos_token_id,
            eos_token_id=[2, 3] if eos_token_id is None else eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # transformers 5.x PretrainedConfig.__init__ does not retain private
        # attributes assigned during __init__; use a public name.
        self.head_dim_was_omitted = head_dim is None

    def to_dict(self) -> dict[str, Any]:
        values = super().to_dict()
        values.pop("_head_dim_was_omitted", None)
        values.pop("head_dim_was_omitted", None)
        # Keep the source's omitted dimension across HF serialization: an
        # inferred full-attention width is not an explicit width for SWA.
        if (
            getattr(self, "head_dim_was_omitted", False)
            and self.head_dim == self.hidden_size // self.num_attention_heads
        ):
            values["head_dim"] = None
        return values

    @staticmethod
    def _validate_sizes(
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_attention_groups: int,
        num_hidden_layers: int,
        num_nextn_predict_layers: int,
        max_seq_len: int,
        max_position_embeddings: int | None,
        kwargs: dict[str, Any],
    ) -> None:
        if "max_position_embedding" in kwargs:
            raise ValueError("Use max_position_embeddings, not max_position_embedding.")
        if min(hidden_size, num_attention_heads, num_attention_groups) <= 0:
            raise ValueError(
                "Step5 hidden size and attention head counts must be positive."
            )
        if num_hidden_layers <= 0 or num_nextn_predict_layers < 0:
            raise ValueError(
                "Step5 requires positive main depth and non-negative MTP depth."
            )
        context_length = (
            max_seq_len if max_position_embeddings is None else max_position_embeddings
        )
        if context_length <= 0:
            raise ValueError("Step5 max_position_embeddings must be positive.")

    def _init_attention_aliases(
        self, *, head_dim: int | None, kwargs: dict[str, Any]
    ) -> None:
        other = self.attention_other_setting
        if other and other["attention_type"] != "sliding_attention":
            raise ValueError(
                "Step5 attention_other_setting must describe sliding_attention."
            )
        # DecoderLayer only applies this override to an explicitly listed type;
        # Attention's alternating default alone does not enable the override.
        if not self.layer_types_with_mtp or (
            "sliding_attention" not in self.layer_types_with_mtp
        ):
            other = None
        swa_heads = (
            other["num_attention_heads"] or self.num_attention_heads
            if other
            else self.num_attention_heads
        )
        swa_kv_heads = (
            other["num_attention_groups"] or self.num_attention_groups
            if other
            else self.num_attention_groups
        )
        swa_head_dim = (other["head_dim"] or head_dim) if other else head_dim
        if self.swa_num_attention_heads is not None and self.sliding_window is not None:
            swa_heads = self.swa_num_attention_heads
        self.head_dim = _resolve_head_dim(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            head_dim=head_dim,
        )
        swa_head_dim = _resolve_head_dim(
            hidden_size=self.hidden_size, num_heads=swa_heads, head_dim=swa_head_dim
        )
        for name in ("n_head_kv", "num_kv_heads"):
            _consume_alias(kwargs=kwargs, name=name, expected=self.num_attention_groups)
        if kwargs.get("multi_query") and self.num_attention_groups != 1:
            raise ValueError("Step5 multi_query conflicts with num_attention_groups.")
        aliases = {
            "num_key_value_heads": self.num_attention_groups,
            "num_attention_groups_full_attention": self.num_attention_groups,
            "head_dim_full_attention": self.head_dim,
            "v_head_dim": self.head_dim,
            "swa_num_key_value_heads": swa_kv_heads,
            "num_attention_groups_sliding_attention": swa_kv_heads,
            "num_sliding_attention_heads": swa_heads,
            "head_dim_sliding_attention": swa_head_dim,
            "swa_head_dim": swa_head_dim,
            "swa_v_head_dim": swa_head_dim,
        }
        if min(self.head_dim, swa_heads, swa_kv_heads, swa_head_dim) <= 0:
            raise ValueError("Step5 attention dimensions must be positive.")
        for name, expected in aliases.items():
            _consume_alias(kwargs=kwargs, name=name, expected=expected)
            setattr(self, name, expected)

    def _init_layer_types(self, *, layer_types: list[str] | None, kwargs: dict) -> None:
        total_layers = self.num_hidden_layers + self.num_nextn_predict_layers
        serialized = kwargs.pop("layer_types_with_mtp", None)
        if serialized is not None:
            if layer_types and list(layer_types) != list(
                serialized[: self.num_hidden_layers]
            ):
                raise ValueError(
                    "layer_types must match the dense prefix of layer_types_with_mtp."
                )
            layer_types = serialized
        _check_layer_array(
            name="layer_types", values=layer_types, total_layers=total_layers
        )
        if layer_types and any(
            value not in ("sliding_attention", "full_attention")
            for value in layer_types
        ):
            raise ValueError("Step5 layer_types contains an unknown attention type.")
        self.layer_types_with_mtp = (
            list(layer_types[:total_layers]) if layer_types else None
        )
        self.layer_types = (
            self.layer_types_with_mtp[: self.num_hidden_layers]
            if self.layer_types_with_mtp
            else None
        )
        # The source attention defaults to alternating SWA/full when absent.
        main_types = self.layer_types or [
            "sliding_attention" if i % 2 == 0 else "full_attention"
            for i in range(self.num_hidden_layers)
        ]
        pattern = [int(value == "sliding_attention") for value in main_types]
        hybrid = self.sliding_window is not None and any(pattern)
        _consume_alias(kwargs=kwargs, name="is_hybrid_swa", expected=hybrid)
        _consume_alias(kwargs=kwargs, name="hybrid_layer_pattern", expected=pattern)
        self.is_hybrid_swa = hybrid
        self.hybrid_layer_pattern = pattern

    def _init_layer_arrays(
        self,
        *,
        rope_theta: float | list[float] | None,
        use_rope_layers: list[bool] | None,
        partial_rotary_factors: list[float] | None,
        swiglu_limits_shared: list[float] | None,
    ) -> None:
        arrays = {
            "use_rope_layers": use_rope_layers,
            "partial_rotary_factors": partial_rotary_factors,
            "swiglu_limits_shared": swiglu_limits_shared,
        }
        if isinstance(rope_theta, list):
            if not rope_theta:
                raise ValueError("Step5 rope_theta cannot be an empty list.")
            arrays["rope_theta"] = rope_theta
        for name, values in arrays.items():
            _check_layer_array(
                name=name,
                values=values,
                total_layers=self.num_hidden_layers + self.num_nextn_predict_layers,
            )
            setattr(self, name, deepcopy(values))
        self.rope_theta = deepcopy(rope_theta)

    def _init_moe_layers(self, *, kwargs: dict[str, Any]) -> None:
        enum = _parse_moe_layers(
            value=kwargs.pop("moe_layers_enum", None), name="moe_layers_enum"
        )
        listed = _parse_moe_layers(
            value=kwargs.pop("moe_layer_list", None), name="moe_layer_list"
        )
        if enum is not None and listed is not None and enum != listed:
            raise ValueError(
                "Step5 moe_layers_enum and moe_layer_list describe different layers."
            )
        indices = enum if enum is not None else listed
        if indices is None:
            indices = set(range(1, self.num_hidden_layers))
        total_layers = self.num_hidden_layers + self.num_nextn_predict_layers
        if any(not 0 <= index < total_layers for index in indices):
            raise ValueError(f"Step5 MoE layer indices must be in [0, {total_layers}).")
        self.moe_layer_list = sorted(indices)
        self.moe_layers_enum = ",".join(str(index) for index in self.moe_layer_list)


class Step5MTPConfig(Step5Config):
    """Independent registration for standalone Step5 MTP checkpoint configs."""

    model_type = "step5_mtp"
    architectures = ["Step5MTP"]

    def __init__(self, num_nextn_predict_layers: int = 1, **kwargs: Any) -> None:
        super().__init__(num_nextn_predict_layers=num_nextn_predict_layers, **kwargs)


if TYPE_CHECKING:
    pass


class Step5LayerSelection(msgspec.Struct, frozen=True, kw_only=True):
    """Absolute attention IDs owned by one target or one MTP runner.

    ``draft_model_idx`` remains a relative checkpoint depth. Only this resolver
    translates it; model and pool construction must consume the resulting IDs.
    The model owns this value; its runner reuses it after load. The shared
    ModelConfig keeps the complete partition for its role.
    """

    draft_model_idx: int | None
    start_layer: int
    end_layer: int
    full_attention_layer_ids: tuple[int, ...]
    swa_attention_layer_ids: tuple[int, ...]


def get_step5_mtp_layer_ids(
    *, config: Step5Config
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the entire checkpoint's absolute (SWA, FULL) draft partition."""
    start = config.num_hidden_layers
    depth = config.num_nextn_predict_layers
    if type(start) is not int or start <= 0 or type(depth) is not int or depth <= 0:
        raise ValueError("Step5 MTP requires positive main and draft depths.")
    end = start + depth
    types = config.layer_types_with_mtp
    if types is None or len(types) < end:
        raise ValueError("Step5 MTP requires explicit attention types for every layer.")
    swa = tuple(i for i in range(start, end) if types[i] == "sliding_attention")
    full = tuple(i for i in range(start, end) if types[i] == "full_attention")
    if len(swa) + len(full) != depth:
        raise ValueError("Step5 MTP contains an unknown attention type.")
    return swa, full


def resolve_step5_layer_selection(
    *,
    config: Step5Config,
    draft_model_idx: int | None,
) -> Step5LayerSelection:
    """Resolve from the exact config and relative-index kwargs of the loader.

    This metadata boundary supports a declared SWA-only draft partition. It does
    not enable a speculative algorithm or qualify any draft execution depth.
    """
    pass

    if not isinstance(config, Step5Config):
        raise TypeError("Step5 layer selection requires the registered Step5 config.")
    architecture = config.architectures[0]
    if architecture not in ("Step5ForCausalLM", "Step5MTP"):
        raise ValueError("Step5 layer selection requires a Step5 architecture.")
    draft = architecture == "Step5MTP"
    if not config.is_hybrid_swa:
        raise ValueError("Step5 layer selection requires native hybrid SWA memory.")
    if draft:
        swa, full = get_step5_mtp_layer_ids(config=config)
        if full:
            raise ValueError("Step5 draft FULL/DSA attention is not supported.")
        if type(draft_model_idx) is not int or not (
            0 <= draft_model_idx < config.num_nextn_predict_layers
        ):
            raise ValueError("Step5 draft_model_idx must be a valid relative depth.")
        layer_id = swa[draft_model_idx]
        return Step5LayerSelection(
            draft_model_idx=draft_model_idx,
            start_layer=layer_id,
            end_layer=layer_id + 1,
            full_attention_layer_ids=(),
            swa_attention_layer_ids=(layer_id,),
        )
    if draft_model_idx is not None:
        raise ValueError("Step5 target must not have a draft_model_idx.")
    # This is the same checkpoint-owned pattern consumed by the native hybrid
    # config adapter; it includes only main layers, even for a bundled MTP file.
    pattern = config.hybrid_layer_pattern
    full = tuple(i for i, kind in enumerate(pattern) if kind == 0)
    swa = tuple(i for i, kind in enumerate(pattern) if kind == 1)
    if len(pattern) != config.num_hidden_layers or len(full) + len(swa) != len(pattern):
        raise ValueError("Step5 target pattern must cover every main layer once.")
    return Step5LayerSelection(
        draft_model_idx=None,
        start_layer=0,
        end_layer=config.num_hidden_layers,
        full_attention_layer_ids=full,
        swa_attention_layer_ids=swa,
    )


class Step5IndexGeometry(msgspec.Struct, frozen=True, kw_only=True):
    index_head_dim: int
    index_k_heads: int
    region_size: int


def parse_step5_index_geometry(config) -> Step5IndexGeometry:
    """Shared model/pool interpretation of the compressed-key storage geometry."""
    sparse = config.sparse_config
    geometry = Step5IndexGeometry(
        index_head_dim=config.index_head_dim,
        index_k_heads=sparse["sparse_indexer_num_k_heads"],
        region_size=sparse["region_block_size"],
    )
    values = (geometry.index_head_dim, geometry.index_k_heads, geometry.region_size)
    if any(type(value) is not int for value in values) or values != (256, 1, 8):
        raise ValueError("Step5 index kernels require D256/H1/R8 geometry")
    return geometry


class Step5AttentionProfile(msgspec.Struct, frozen=True, kw_only=True):
    hidden_size: int
    head_dim: int
    num_heads: int
    num_kv_heads: int
    index_head_dim: int
    index_num_heads: int
    index_num_kv_heads: int
    provider_groups: int
    region_size: int
    index_rope_dim: int
    sliding_window: int


def parse_step5_attention_profile(config) -> Step5AttentionProfile:
    sparse = config.sparse_config
    index = parse_step5_index_geometry(config)
    profile = Step5AttentionProfile(
        hidden_size=config.hidden_size,
        head_dim=config.head_dim,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        # External config spelling is handled by Step5Config.index_head_dim.
        index_head_dim=index.index_head_dim,
        index_num_heads=sparse["sparse_indexer_num_heads"],
        index_num_kv_heads=index.index_k_heads,
        provider_groups=sparse["num_provider_groups"],
        region_size=index.region_size,
        index_rope_dim=sparse["sparse_indexer_rope_dim"],
        sliding_window=config.sliding_window,
    )
    # Current QK-norm, scoring and R8 compressor kernel capabilities.
    if (
        profile.hidden_size,
        profile.head_dim,
        profile.num_heads,
        profile.num_kv_heads,
        profile.index_head_dim,
        profile.index_num_heads,
        profile.index_num_kv_heads,
        profile.provider_groups,
        profile.region_size,
        profile.index_rope_dim,
        profile.sliding_window,
    ) != (4096, 192, 64, 4, 256, 16, 1, 4, 8, 32, 512):
        raise ValueError("Step5 kernels require the D192/R8/index-D256 profile")
    if (
        config.num_sliding_attention_heads,
        config.swa_num_key_value_heads,
        config.swa_head_dim,
    ) != (profile.num_heads, profile.num_kv_heads, profile.head_dim):
        raise ValueError("Step5 FULL/SWA attention geometry must agree")
    if (
        config.zero_centered is not True
        or config.norm_dtype.lower() not in ("fp32", "float32", "float")
        or config.use_head_wise_attn_gate is not True
    ):
        raise ValueError("Step5 requires FP32 Gemma norms and per-head gates")
    required = {
        "enabled": True,
        "sparse_indexer_use_rope": True,
        "sparse_indexer_q_norm_type": "rmsnorm",
        "sparse_indexer_k_norm_type": "layernorm",
        "sparse_indexer_csa_z_norm_type": "none",
    }
    if any(sparse[key] != value for key, value in required.items()):
        raise ValueError("Step5 indexer differs from the supported norm/RoPE profile")
    for layer_id, kind in enumerate(config.layer_types_with_mtp):
        if kind not in ("full_attention", "sliding_attention"):
            raise ValueError("Step5 attention has an unknown layer type")
        full = kind == "full_attention"
        width = 64 if full else profile.head_dim
        if (
            config.rope_theta[layer_id] != (10_000_000 if full else 10_000)
            or config.partial_rotary_factors[layer_id] != width / profile.head_dim
            or (
                config.use_rope_layers is not None
                and not config.use_rope_layers[layer_id]
            )
        ):
            raise ValueError(
                "Step5 attention RoPE differs from the supported table profile"
            )
    return profile


def resolve_step5_vocab_size(
    *, config: Step5Config, tokenizer_vocab_size: int | None
) -> int:
    """Resolve once before model/serving construction; never resize HF weights."""
    explicit = config.valid_vocab_size
    if explicit is not None and (
        type(explicit) is not int or not 0 < explicit <= config.vocab_size
    ):
        raise ValueError("Step5 valid_vocab_size must be in (0, physical vocab_size]")
    if tokenizer_vocab_size is None:
        if explicit is None:
            raise ValueError(
                "Step5 skip_tokenizer_init requires valid_vocab_size in "
                "--json-model-override-args"
            )
        return explicit
    bound = tokenizer_vocab_size
    if bound > config.vocab_size:
        raise ValueError("Step5 tokenizer token-ID bound exceeds physical vocab_size")
    if explicit is not None and explicit > bound:
        raise ValueError("Step5 valid_vocab_size exceeds tokenizer token-ID bound")
    return bound if explicit is None else explicit


def get_step5_position_padding() -> int:
    """Execution headroom, not an extension of the advertised model context.

    The public speculative worker computes a complete window before enforcing
    each request's output limit. Keep RoPE, indexer and MTP input validation
    within the same native request-row allocation. R8 storage needs full groups.
    """
    from sglang.srt.mem_cache.allocation_sizing import (
        get_req_to_token_extra_context_len,
    )
    from sglang.srt.runtime_context import get_spec

    if get_spec().speculative_algorithm is None:
        return 0
    return get_req_to_token_extra_context_len() // 8 * 8


class Step5VLConfig(PretrainedConfig):
    """Open-source Step5 VL checkpoint (model_type "step5_vl").

    Wraps the Step5 text backbone (``text_config``) and a perception-encoder
    vision tower. Text-only serving unwraps ``text_config``; multimodal
    serving goes through the Step3.7-compatible wrapper/processor.
    """

    model_type = "step5_vl"
    architectures = ["Step5ForConditionalGeneration"]

    def __init__(
        self,
        vision_config: Optional[Union[dict, PretrainedConfig]] = None,
        text_config: Optional[Union[dict, PretrainedConfig]] = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 128001,
        image_token_len: int = 169,
        patch_token_len: Optional[int] = None,
        im_start_token: str = "<im_start>",
        im_end_token: str = "<im_end>",
        im_patch_token: str = "<im_patch>",
        use_im_start_end: bool = True,
        vision_select_layer: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(vision_config, dict):
            vision_config = Step3p7VisionEncoderConfig(**vision_config)
        self.vision_config = vision_config
        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = Step5Config(**text_config)
        self.text_config = text_config
        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.image_token_id = image_token_id
        self.image_token_len = image_token_len
        self.patch_token_len = patch_token_len
        self.im_start_token = im_start_token
        self.im_end_token = im_end_token
        self.im_patch_token = im_patch_token
        self.use_im_start_end = use_im_start_end
        self.vision_select_layer = vision_select_layer
        super().__init__(**kwargs)
