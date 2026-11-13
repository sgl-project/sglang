# coding=utf-8
# Copyright 2026 Liquid AI and the SGLang team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""LFM3 (Liquid Foundation Model 3) configuration."""

from transformers import CONFIG_MAPPING, PretrainedConfig

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateShape,
    mamba2_state_dtype,
)


# TODO: Replace this wrapper with native config inheritance once LFM3 support
# lands in upstream Transformers.
class Lfm3ServingConfig(PretrainedConfig):
    """Carry checkpoint-resolved values without duplicating its config schema."""

    has_no_defaults_at_init = True

    def __post_init__(self, **kwargs):
        # Transformers otherwise discards use_cache as a generation parameter.
        if "use_cache" in kwargs:
            self.use_cache = kwargs.pop("use_cache")
        super().__post_init__(**kwargs)
        for name, value in vars(self).items():
            if isinstance(value, dict) and "model_type" in value:
                model_type = value["model_type"]
                config_class = (
                    CONFIG_MAPPING[model_type]
                    if model_type in CONFIG_MAPPING
                    else Lfm3ServingConfig
                )
                setattr(self, name, config_class(**value))

    def to_dict(self):
        # Generic nested configs retain their checkpoint model type.
        return {**super().to_dict(), "model_type": self.model_type}

    @classmethod
    def from_hf_config(cls, config: PretrainedConfig):
        values = config.to_dict()
        # Preserve resolved nested config classes from checkpoint code.
        for name in config.sub_configs:
            values[name] = getattr(config, name)
        return cls(**values)


class Lfm3Config(Lfm3ServingConfig):
    """LFM3 configuration with SGLang hybrid-cache metadata."""

    model_type = "lfm3"
    _attention_layer_types = ("full_attention", "sliding_attention")

    @property
    def full_attention_layer_ids(self) -> list[int]:
        """Indices of the attention layers (both full and sliding-window).

        These are the only layers that hold KV-cache slots; conv layers use the
        conv-state cache instead. Sliding-window behaviour is applied inside
        RadixAttention via `sliding_window_size`, not by partitioning this list.
        """
        return [
            i
            for i, lt in enumerate(self.layer_types)
            if lt in self._attention_layer_types
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        """Short-conv layer indices (use the MambaPool conv-state cache)."""
        return [i for i, lt in enumerate(self.layer_types) if lt == "conv"]

    @property
    def mamba_chunk_size(self) -> int:
        # ShortConv has no chunked SSM scan, so this only feeds the mamba
        # cache bookkeeping. 256 (the NemotronH/FLA convention) instead of 1:
        # the extra_buffer path asserts the conv window fits inside a chunk
        # (conv_L_cache - 1 < chunk) and that mamba_track_interval (default
        # 256) >= chunk, so chunk=1 made --mamba-scheduler-strategy
        # extra_buffer fail at init and chunk must not exceed 256. It also
        # sets the prefill caching granularity, max(chunk, page_size), and
        # bounds Mamba2Metadata.prepare_mixed's chunk bookkeeping
        # (chunk=1 built one metadata entry per token).
        return 256

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams | None:
        """Conv-state cache params for HybridReqToTokenPool.

        Like LFM2's ShortConv, this keeps only a (kernel_size - 1)-token conv cache and
        no SSM temporal state, so it reuses Mamba2StateShape with state_size=0 and
        a synthetic head layout to satisfy that type's divisibility checks.
        """
        from sglang.srt.runtime_context import get_parallel

        conv_layer_ids = self.linear_layer_ids
        if not conv_layer_ids:
            return None

        # tp from the attention group when distributed, else 1 for a standalone
        # AutoConfig build (same pattern as Lfm2Config). The conv-state shape
        # below is divisibility-safe at every tp; the attention KV-head count is
        # the real TP upper bound (so the 1.x checkpoints, with 4 KV heads,
        # support tp in {1, 2, 4}).
        try:
            tp_size = get_parallel().attn_tp_size
        except (AssertionError, RuntimeError):
            tp_size = 1

        shape = Mamba2StateShape.create(
            tp_world_size=tp_size,
            intermediate_size=self.hidden_size,
            n_groups=1,
            num_heads=tp_size,
            head_dim=self.hidden_size,
            state_size=0,
            conv_kernel=int(self.conv_L_cache),
        )

        return Mamba2CacheParams(
            shape=shape,
            layers=conv_layer_ids,
            dtype=mamba2_state_dtype(self),
        )


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=Lfm3Config,
        arch_names=["Lfm3ForCausalLM"],
        backend_class_name=(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend.Mamba2AttnBackend"
        ),
        support_mamba_cache_extra_buffer=True,
    )
)
