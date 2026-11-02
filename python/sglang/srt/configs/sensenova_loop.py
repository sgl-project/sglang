# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from sglang.srt.configs.linear_attn_model_registry import (
    LinearAttnModelSpec,
    register_linear_attn_model,
)
from sglang.srt.configs.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig


class SenseNovaLoopTextConfig(Qwen3_5TextConfig):
    """Serving slots follow execution order; checkpoint layers remain physical."""

    def __init__(self, execution_plan=(), mhc=None, layer_types=None, **kwargs):
        super().__init__(layer_types=layer_types, **kwargs)
        self.execution_plan = tuple(execution_plan)
        self.mhc = mhc
        self.layer_types = layer_types

    @property
    def layers_block_type(self):
        return [
            "attention" if kind == "full_attention" else "linear_attention"
            for kind in self.layer_types
        ]


class SenseNovaLoopConfig(Qwen3_5Config):
    model_type = "sensenova_loop"
    has_no_defaults_at_init = True
    sub_configs = {
        **Qwen3_5Config.sub_configs,
        "text_config": SenseNovaLoopTextConfig,
    }

    def __init__(self, loop_settings=None, mhc=None, **kwargs):
        super().__init__(**kwargs)
        self.loop_settings = loop_settings
        self.mhc = mhc
        text = self.text_config.to_dict()
        depth = text.get("physical_num_hidden_layers", text["num_hidden_layers"])
        kinds = text.get("physical_layer_types", text["layer_types"])
        start, end, count = (
            loop_settings[key] for key in ("start", "end", "eval_count")
        )
        self.execution_plan = (
            tuple(range(start))
            + tuple(range(start, end)) * count
            + tuple(range(end, depth))
        )
        text.update(
            physical_num_hidden_layers=depth,
            physical_layer_types=kinds,
            num_hidden_layers=len(self.execution_plan),
            layer_types=[kinds[i] for i in self.execution_plan],
            execution_plan=self.execution_plan,
            mhc=mhc,
        )
        self.text_config = SenseNovaLoopTextConfig(**text)
        self.architectures = ["SenseNovaLoopForConditionalGeneration"]


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=SenseNovaLoopTextConfig,
        backend_class_name="sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
        arch_names=["SenseNovaLoopForConditionalGeneration"],
        unwrap_text_config=True,
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=True,
    )
)
