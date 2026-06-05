# SPDX-License-Identifier: Apache-2.0
"""SANA-WM pipeline stages (package).

The realtime serving framework drives ``SanaWMRealtimeStage`` over the
``/v1/realtime_video`` WebSocket. Base stages + helpers are re-exported here
for back-compat.
"""

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.base import (
    SanaWMBeforeDenoisingStage,
    SanaWMDecodingStage,
    SanaWMDenoisingStage,
    SanaWMTextEncodingStage,
    _align_sana_wm_cfg_text_conditions,
    configure_sana_wm_ltx2_vae_for_long_video,
    parse_sana_wm_action_string,
    sana_wm_action_to_camera_to_world,
)

__all__ = [
    "SanaWMBeforeDenoisingStage",
    "SanaWMDecodingStage",
    "SanaWMDenoisingStage",
    "SanaWMTextEncodingStage",
    "_align_sana_wm_cfg_text_conditions",
    "configure_sana_wm_ltx2_vae_for_long_video",
    "parse_sana_wm_action_string",
    "sana_wm_action_to_camera_to_world",
    "SanaWMRealtimeStage",
]


def __getattr__(name):
    # Lazy: ``realtime_stage`` imports from this package, so an eager import
    # here would hit a partially-initialized package. Defer to stay cycle-free.
    if name == "SanaWMRealtimeStage":
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.realtime_stage import (
            SanaWMRealtimeStage,
        )

        globals()["SanaWMRealtimeStage"] = SanaWMRealtimeStage
        return SanaWMRealtimeStage
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
