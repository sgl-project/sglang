# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math

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

    def __init__(self, loop_settings=None, mhc=None, loop_expanded=False, **kwargs):
        if loop_expanded:
            raise ValueError(
                "SenseNova Loop requires a canonical, unexpanded checkpoint"
            )
        super().__init__(**kwargs)
        self.loop_settings = loop_settings
        self.mhc = mhc
        text = self.text_config.to_dict()
        depth = text.get("physical_num_hidden_layers", text["num_hidden_layers"])
        kinds = text.get("physical_layer_types", text["layer_types"])
        self.execution_plan = self._execution_plan(depth=depth, layer_types=kinds)
        text.update(
            physical_num_hidden_layers=depth,
            physical_layer_types=kinds,
            num_hidden_layers=len(self.execution_plan),
            layer_types=[kinds[i] for i in self.execution_plan],
            execution_plan=self.execution_plan,
            mhc=mhc,
        )
        self.text_config = SenseNovaLoopTextConfig(**text)
        # The canonical HF checkpoint also contains a vision tower. This native
        # entry point implements its text backbone and retains the original files.
        self.architectures = ["SenseNovaLoopForCausalLM"]
        self.language_model_only = True
        # Text uses the same position on all three mRoPE axes, so ordinary
        # 1-D RoPE is equivalent and does not require a multimodal processor.
        for rope in (self.text_config.rope_parameters, self.text_config.rope_scaling):
            rope.pop("mrope_section", None)
            rope.pop("mrope_interleaved", None)

    def _execution_plan(self, *, depth, layer_types):
        loop = self.loop_settings
        if not isinstance(loop, dict) or loop.get("use_gate", True):
            raise ValueError("SenseNova Loop requires explicit, ungated loop_settings")
        start, end, count = (loop.get(key) for key in ("start", "end", "eval_count"))
        if any(type(value) is not int for value in (depth, start, end, count)) or not (
            0 <= start < end <= depth and count >= 1
        ):
            raise ValueError("Invalid Loop interval or eval_count")
        if loop.get("counts") != [count] or loop.get("probabilities") != [1.0]:
            raise ValueError("Only fixed-count Loop checkpoints are supported")
        if len(layer_types) != depth or set(layer_types) - {
            "full_attention",
            "linear_attention",
        }:
            raise ValueError("Invalid physical layer_types")
        if "full_attention" not in layer_types:
            raise ValueError(
                "SenseNova Loop requires at least one full-attention layer"
            )
        if self.mhc is not None:
            mhc_start, mhc_end, streams, iterations = (
                self.mhc.get(key) for key in ("start", "end", "streams", "iterations")
            )
            if any(
                type(value) is not int
                for value in (mhc_start, mhc_end, streams, iterations)
            ) or not (
                0 <= mhc_start < mhc_end <= depth
                and streams in (2, 4)
                and iterations >= 1
            ):
                raise ValueError("Invalid mHC interval, stream count, or iterations")
            if type(self.mhc.get("fixed_identity")) is not bool:
                raise ValueError("mHC fixed_identity must be a boolean")
            scale = self.mhc.get("pre_scale", 1.0)
            if (
                not isinstance(scale, (int, float))
                or not math.isfinite(scale)
                or scale <= 0
            ):
                raise ValueError("Invalid mHC pre_scale")
            if set(self.mhc) - {
                "start",
                "end",
                "streams",
                "iterations",
                "fixed_identity",
                "pre_scale",
            }:
                raise ValueError("Unknown mHC configuration fields")
            boundaries = (0, start, end, depth)
            if not any(
                left <= mhc_start < mhc_end <= right
                for left, right in zip(boundaries, boundaries[1:])
            ):
                raise ValueError("mHC cannot cross a Loop phase boundary")
        return (
            tuple(range(start))
            + tuple(range(start, end)) * count
            + tuple(range(end, depth))
        )


register_linear_attn_model(
    LinearAttnModelSpec(
        config_class=SenseNovaLoopTextConfig,
        backend_class_name="sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend",
        arch_names=["SenseNovaLoopForCausalLM"],
        unwrap_text_config=True,
        uses_mamba_radix_cache=True,
        support_mamba_cache=True,
        support_mamba_cache_extra_buffer=True,
    )
)
