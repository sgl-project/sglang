# SPDX-License-Identifier: Apache-2.0
"""SANA-WM pipeline stages (package).

Split out of the former flat ``sana_wm.py`` module into a package to mirror
mickqian's layout. The base stages live in ``sana_wm_base``; streaming, refiner,
streaming_refiner, realtime and realtime_engine are sibling submodules imported
by their explicit path.

The base stages + a few helpers are re-exported here so that existing callers
of ``...model_specific_stages.sana_wm import X`` (the pipeline and tests) keep
working unchanged. Pure relocation — no behavior change.
"""

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.sana_wm_base import (
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
]
