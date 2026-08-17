from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
from transformers import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig

from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.runtime_context import get_exec


class InklingModelConfig(PretrainedConfig):
    model_type = "inkling_model"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        vocab_size: int = 201024,
        hidden_size: int = 1536,
        intermediate_size: int = 768,
        dense_intermediate_size: Optional[int] = None,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        d_rel: int = 16,
        rel_extent: int = 1024,
        local_layer_ids: Optional[list[int]] = None,
        sliding_window_size: int = 512,
        swa_num_attention_heads: Optional[int] = None,
        swa_num_key_value_heads: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        mtp_local_layer_ids: Optional[list[int]] = None,
        mtp_local_extent: Optional[int] = None,
        mtp_swa_num_attention_heads: Optional[int] = None,
        mtp_swa_num_key_value_heads: Optional[int] = None,
        mtp_swa_head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
        q_bias: bool = False,
        o_bias: bool = False,
        use_embed_norm: bool = False,
        use_sconv: bool = False,
        sconv_kernel_size: int = 4,
        chain_hidden_post_norm: bool = False,
        dense_mlp_idx: int = 0,
        n_routed_experts: int = 0,
        n_shared_experts: int = 0,
        num_experts_per_tok: int = 1,
        route_scale: float = 1.0,
        use_gate_bias: bool = False,
        use_global_scale: bool = False,
        norm_after_topk: bool = True,
        gate_activation: Literal["sigmoid", "softmax"] = "sigmoid",
        shared_expert_sink: bool = False,
        shared_experts_size: int = 1,
        inference_moe_w13_interleaved: bool = True,
        log_scaling_n_floor: int | None = None,
        log_scaling_alpha: float = 0.1,
        unpadded_vocab_size: Optional[int] = None,
        padded_vocab_size: Optional[int] = None,
        logits_mup_width_multiplier: Optional[float] = None,
        final_logit_softcapping: Optional[float] = None,
        num_nextn_predict_layers: int = 8,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        if v_head_dim is None:
            v_head_dim = head_dim
        if swa_num_attention_heads is None:
            swa_num_attention_heads = num_attention_heads
        if swa_num_key_value_heads is None:
            swa_num_key_value_heads = num_key_value_heads
        if swa_head_dim is None:
            swa_head_dim = head_dim
        if swa_v_head_dim is None:
            swa_v_head_dim = swa_head_dim
        if dense_intermediate_size is None:
            dense_intermediate_size = intermediate_size
        if local_layer_ids is None:
            local_layer_ids = []
        # Per-depth banded MTP attention: a depth listed in mtp_local_layer_ids
        # is a sliding-window block with its own window (mtp_local_extent);
        # other depths stay full-attention.
        if mtp_local_layer_ids is None:
            mtp_local_layer_ids = []
        if mtp_local_extent is None:
            mtp_local_extent = sliding_window_size
        if mtp_swa_num_attention_heads is None:
            mtp_swa_num_attention_heads = swa_num_attention_heads
        if mtp_swa_num_key_value_heads is None:
            mtp_swa_num_key_value_heads = swa_num_key_value_heads
        if mtp_swa_head_dim is None:
            mtp_swa_head_dim = swa_head_dim
        if mtp_local_layer_ids:
            local_id_set = set(mtp_local_layer_ids)
            assert len(local_id_set) == len(
                mtp_local_layer_ids
            ), f"mtp_local_layer_ids must be unique: {mtp_local_layer_ids}"
            assert all(0 <= i < num_nextn_predict_layers for i in local_id_set), (
                f"mtp_local_layer_ids must be in [0, {num_nextn_predict_layers}): "
                f"{mtp_local_layer_ids}"
            )
            # The draft KV pool and the sconv conv-state cache are still sized
            # from the trunk's swa geometry; a head geometry that differs is not
            # wired through yet.
            assert (
                mtp_swa_num_key_value_heads == swa_num_key_value_heads
                and mtp_swa_head_dim == swa_head_dim
            ), (
                "banded MTP head geometry must match the trunk swa geometry: "
                f"kv_heads {mtp_swa_num_key_value_heads} vs "
                f"{swa_num_key_value_heads}, head_dim {mtp_swa_head_dim} vs "
                f"{swa_head_dim}"
            )

        if padded_vocab_size is None:
            padded_vocab_size = vocab_size
            vocab_size = (
                unpadded_vocab_size
                if (
                    unpadded_vocab_size is not None
                    and unpadded_vocab_size < padded_vocab_size
                )
                else vocab_size
            )

        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dense_intermediate_size = dense_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.local_layer_ids = local_layer_ids
        self.sliding_window_size = sliding_window_size
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.swa_v_head_dim = swa_v_head_dim
        self.mtp_local_layer_ids = mtp_local_layer_ids
        self.mtp_local_extent = mtp_local_extent
        self.mtp_swa_num_attention_heads = mtp_swa_num_attention_heads
        self.mtp_swa_num_key_value_heads = mtp_swa_num_key_value_heads
        self.mtp_swa_head_dim = mtp_swa_head_dim
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.q_bias = q_bias
        self.o_bias = o_bias
        self.use_embed_norm = use_embed_norm
        self.use_sconv = use_sconv
        self.sconv_kernel_size = sconv_kernel_size
        self.chain_hidden_post_norm = chain_hidden_post_norm
        self.dense_mlp_idx = dense_mlp_idx
        self.n_routed_experts = n_routed_experts
        self.num_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.route_scale = route_scale
        self.use_gate_bias = use_gate_bias
        self.use_global_scale = use_global_scale
        self.norm_after_topk = norm_after_topk
        self.gate_activation = gate_activation
        self.shared_expert_sink = shared_expert_sink
        self.shared_experts_size = shared_experts_size
        self.inference_moe_w13_interleaved = inference_moe_w13_interleaved
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.unpadded_vocab_size = self.vocab_size
        self.logits_mup_width_multiplier = logits_mup_width_multiplier
        self.final_logit_softcapping = final_logit_softcapping
        self.num_nextn_predict_layers = num_nextn_predict_layers

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def conv_layer_ids(self) -> list[int]:
        return list(range(self.num_hidden_layers))

    @property
    def linear_layer_ids(self) -> list[int]:
        return self.conv_layer_ids

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return list(range(self.num_hidden_layers))

    @property
    def mamba_chunk_size(self) -> int:
        # Floor at 64: mamba_cache_chunk_size = max(mamba_chunk_size, page_size),
        # and a floor of 1 lets the radix tree adopt another request's KV at
        # tiny shared prefixes, whose different kernel-rounding perturbs decode logits.
        return 64

    @property
    def mamba2_cache_params(self) -> Optional[InklingConvCacheParams]:
        from sglang.srt.runtime_context import get_parallel

        try:
            tp_size = get_parallel().attn_tp_size
        except (AssertionError, RuntimeError):
            tp_size = 1

        def tp_local_kv_conv_dim(num_kv_heads: int, head_dim: int) -> int:
            return max(1, num_kv_heads // tp_size) * head_dim

        full_kv_conv_dim = tp_local_kv_conv_dim(self.num_key_value_heads, self.head_dim)
        local_kv_conv_dim = tp_local_kv_conv_dim(
            self.swa_num_key_value_heads, self.swa_head_dim
        )
        stream_dim = self.hidden_size

        if get_exec().comm.enable_scattered_sconv:
            # Scattered sconv: the attn/mlp output sconvs run on the [T, H/P]
            # hidden shard, so their conv-state caches shard with them.
            assert (
                self.hidden_size % tp_size == 0
            ), f"hidden_size {self.hidden_size} not divisible by attn tp {tp_size}"
            stream_dim = self.hidden_size // tp_size
        conv_len = self.sconv_kernel_size - 1
        shape = InklingConvStateShape(
            conv=[
                (conv_len, full_kv_conv_dim),
                (conv_len, full_kv_conv_dim),
                (conv_len, local_kv_conv_dim),
                (conv_len, local_kv_conv_dim),
                (conv_len, stream_dim),
                (conv_len, stream_dim),
            ],
            temporal=(0, 0, 0),
        )
        dtype = InklingStateDType(conv=torch.bfloat16, temporal=torch.bfloat16)
        return InklingConvCacheParams(
            shape=shape, layers=self.conv_layer_ids, dtype=dtype
        )


class InklingAudioConfig(PretrainedConfig):
    model_type = "inkling_audio_model"

    def __init__(
        self,
        *,
        decoder_dmodel: Optional[int] = None,
        n_mel_bins: int = 80,
        mel_vocab_size: int = 16,
        dmel_min_value: float = -1.5,
        dmel_max_value: float = 2.0,
        use_audio_norm: bool = False,
        audio_mode: Literal["dmel", "flow"] = "dmel",
        **kwargs: Any,
    ) -> None:
        self.decoder_dmodel = decoder_dmodel
        self.n_mel_bins = n_mel_bins
        self.mel_vocab_size = mel_vocab_size
        self.dmel_min_value = dmel_min_value
        self.dmel_max_value = dmel_max_value
        self.use_audio_norm = use_audio_norm
        self.audio_mode = audio_mode
        super().__init__(**kwargs)


class InklingVisionConfig(PretrainedConfig):
    model_type = "inkling_vision_model"

    def __init__(
        self,
        *,
        vision_encoder_type: Literal["linear", "hmlp"] = "hmlp",
        decoder_dmodel: Optional[int] = None,
        patch_size: int = 16,
        temporal_patch_size: int = 1,
        n_channels: int = 3,
        n_layers: int = 1,
        use_vision_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        self.vision_encoder_type = vision_encoder_type
        self.decoder_dmodel = decoder_dmodel
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.use_vision_norm = use_vision_norm
        super().__init__(**kwargs)


class InklingMMConfig(PretrainedConfig):
    model_type = "inkling_mm_model"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {
        "text_config": InklingModelConfig,
        "audio_config": InklingAudioConfig,
        "vision_config": InklingVisionConfig,
    }

    def __init__(
        self,
        *,
        text_config: Optional[dict[str, Any] | InklingModelConfig] = None,
        audio_config: Optional[dict[str, Any] | InklingAudioConfig] = None,
        vision_config: Optional[dict[str, Any] | InklingVisionConfig] = None,
        mtp_config: Optional[dict[str, Any]] = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.mtp_config = mtp_config
        self.text_config = (
            text_config
            if isinstance(text_config, InklingModelConfig)
            else InklingModelConfig(**(text_config or {}))
        )
        if isinstance(mtp_config, dict) and mtp_config.get("local_layer_ids"):
            # Banded MTP head: the checkpoint declares its sliding-window draft
            # depths on mtp_config. Canonicalize onto text_config so every
            # consumer (hybrid layer-id split, draft pool routing, the MTP block
            # construction) reads one source of truth.
            self.text_config.mtp_local_layer_ids = list(mtp_config["local_layer_ids"])
            if mtp_config.get("local_extent") is not None:
                self.text_config.mtp_local_extent = mtp_config["local_extent"]
        self.audio_config = (
            audio_config
            if isinstance(audio_config, InklingAudioConfig)
            else InklingAudioConfig(**(audio_config or {}))
        )
        self.vision_config = (
            vision_config
            if isinstance(vision_config, InklingVisionConfig)
            else InklingVisionConfig(**(vision_config or {}))
        )
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_text_config(self, *args: Any, **kwargs: Any) -> InklingModelConfig:
        return self.text_config

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers

    @property
    def num_attention_heads(self) -> int:
        return self.text_config.num_attention_heads

    @property
    def num_key_value_heads(self) -> int:
        return self.text_config.num_key_value_heads

    @property
    def head_dim(self) -> int:
        return self.text_config.head_dim

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return self.text_config.full_attention_layer_ids

    @property
    def linear_layer_ids(self) -> list[int]:
        return self.text_config.linear_layer_ids

    @property
    def conv_layer_ids(self) -> list[int]:
        return self.text_config.conv_layer_ids

    @property
    def mamba_chunk_size(self) -> int:
        return self.text_config.mamba_chunk_size

    @property
    def mamba2_cache_params(self) -> Optional[InklingConvCacheParams]:
        return self.text_config.mamba2_cache_params


@dataclass(kw_only=True, frozen=True)
class InklingConvStateShape:
    conv: list[tuple[int, int]]
    temporal: tuple[int, int, int]

    # Conv tuples read (K-1, dim) — the overlapping dedup view would alias
    # along the dim axis, so the dedup conv-intermediate layout must stay off.
    disable_conv_window_dedup: bool = True


@dataclass(kw_only=True, frozen=True)
class InklingStateDType:
    conv: torch.dtype = torch.bfloat16
    temporal: torch.dtype = torch.bfloat16


@dataclass(kw_only=True, frozen=True)
class InklingConvCacheParams(BaseLinearStateParams):
    dtype: InklingStateDType = field(default_factory=InklingStateDType)
    shape: InklingConvStateShape


for _model_type, _config_cls in {
    "inkling_model": InklingModelConfig,
    "inkling_audio_model": InklingAudioConfig,
    "inkling_vision_model": InklingVisionConfig,
    "inkling_mm_model": InklingMMConfig,
}.items():
    try:
        CONFIG_MAPPING.register(_model_type, _config_cls)
    except Exception:
        CONFIG_MAPPING._extra_content[_model_type] = _config_cls
