# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams-Transfer pipeline configuration.

Same distilled Cosmos3 Omni checkpoint layout as Cosmos-Dreams, with a
``control_video`` conditioning contract instead of an action contract.
"""

import os
from dataclasses import dataclass

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    CONTROL_VIDEO_CONDITIONING_MODE,
    HISTORY_MODE_AUTO,
    HISTORY_MODE_FULL,
    HISTORY_MODE_SLIDING,
    CosmosDreamsManifest,
    CosmosDreamsTransferHistoryProfile,
    resolve_transfer_history_profile,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
)


@dataclass
class CosmosDreamsTransferConfig(CosmosDreamsConfig):
    conditioning_mode: str = CONTROL_VIDEO_CONDITIONING_MODE
    # The control clip arrives as control_path, never as an image; TI2V would
    # make the server warmup attach a synthetic image the request validator rejects.
    task_type: ModelTaskType = ModelTaskType.T2V

    def _validate_history_settings(self, manifest: CosmosDreamsManifest) -> None:
        self.transfer_history_profile(manifest)

    # "auto" follows the export's training config: the reference slides only when
    # kv_cache_inference_size was set (Sim-Depth: 51 frames, 1 sink) and keeps the
    # whole clip otherwise. The artifact's no_eviction flag is an exporter constant.
    history_mode: str = HISTORY_MODE_AUTO

    def transfer_history_profile(
        self,
        manifest: CosmosDreamsManifest,
        *,
        trained_window: tuple[int | None, int] | None = None,
    ) -> CosmosDreamsTransferHistoryProfile:
        """K/V retention for the control/RGB rollout.

        ``trained_window`` is ``(kv_cache_inference_size, attention_sink_size)`` of
        the training config; ``None`` reads it from the export's root ``config.json``.
        """
        mode = self.history_mode
        if mode == HISTORY_MODE_AUTO:
            if trained_window is None:
                trained_window = training_kv_window(self.model_path)
            window, sink = trained_window
            if window is None:
                mode = HISTORY_MODE_FULL
            else:
                if (window, sink) != (manifest.window_frames, manifest.sink_frames):
                    raise ValueError(
                        "Cosmos-Dreams control_video export disagrees with its training "
                        f"config: artifact window_frames={manifest.window_frames}, "
                        f"sink_frames={manifest.sink_frames}; training "
                        f"kv_cache_inference_size={window}, attention_sink_size={sink}."
                    )
                mode = HISTORY_MODE_SLIDING
        return resolve_transfer_history_profile(manifest, history_mode=mode)

    def adjust_num_frames(self, num_frames: int, *, log_adjustment: bool = True) -> int:
        # The rollout consumes whole latent chunks after frame 0 and the control
        # stage trims clips to that partition; rounding here keeps warmup probes
        # on counts the pipeline accepts (17, 33, 49, ... pixel frames at 4x4).
        del log_adjustment
        chunk, factor = self.chunk_size, self.temporal_compression_factor
        latent = max(1 + chunk, (max(int(num_frames), 1) - 1) // factor + 1)
        aligned_latent = 1 + ((latent - 1) // chunk) * chunk
        return 1 + (aligned_latent - 1) * factor

    # The transfer stages tokenize with the Transfer system prompt themselves;
    # recorded here so generic prompt plumbing reports the truth.
    use_system_prompt: bool = True


def training_kv_window(model_path: str | None) -> tuple[int | None, int]:
    """``(kv_cache_inference_size, attention_sink_size)`` from ``model.config`` of the
    export's root ``config.json`` (the imaginaire4 training config the exporter dumps);
    ``(None, 0)`` when the export has no such record."""
    if not model_path:
        return None, 0
    from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
        load_dict,
        prepare_diffusers_component_path_for_loading,
    )

    root = prepare_diffusers_component_path_for_loading(model_path)
    config_path = os.path.join(root, "config.json")
    if not os.path.isfile(config_path):
        return None, 0
    model_config = load_dict(config_path).get("model", {}).get("config", {})
    window = model_config.get("kv_cache_inference_size")
    sink = model_config.get("attention_sink_size", 0) or 0
    if window is not None and (isinstance(window, bool) or not isinstance(window, int)):
        raise ValueError(
            "Cosmos-Dreams training config kv_cache_inference_size must be an int, "
            f"got {window!r}."
        )
    return window, int(sink)
