# SPDX-License-Identifier: Apache-2.0
"""LingBot-World-specific pipeline stages."""


def __getattr__(name: str):
    if name == "LingBotWorldCausalDMDDenoisingStage":
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world.lingbot_world_causal_denoising import (
            LingBotWorldCausalDMDDenoisingStage,
        )

        return LingBotWorldCausalDMDDenoisingStage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LingBotWorldCausalDMDDenoisingStage",
]
