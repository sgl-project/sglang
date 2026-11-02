# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams-Transfer pipeline configuration.

Same distilled Cosmos3 Omni checkpoint layout as Cosmos-Dreams, with a
``control_video`` conditioning contract instead of an action contract.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    CONTROL_VIDEO_CONDITIONING_MODE,
    TRANSFER_RESIZE_DIRECT,
    TRANSFER_RESIZE_MODES,
    CosmosDreamsManifest,
    CosmosDreamsTransferHistoryProfile,
    resolve_transfer_history_profile,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.cosmos_dreams import (
    CosmosDreamsConfig,
    resolve_history_mode,
)


@dataclass
class CosmosDreamsTransferConfig(CosmosDreamsConfig):
    conditioning_mode: str = CONTROL_VIDEO_CONDITIONING_MODE
    # The control clip arrives as control_path, never as an image; TI2V would
    # make the server warmup attach a synthetic image the request validator rejects.
    task_type: ModelTaskType = ModelTaskType.T2V

    def _validate_history_settings(self, manifest: CosmosDreamsManifest) -> None:
        if (
            self.kv_cache_inference_size is not None
            or self.attention_sink_size is not None
        ):
            raise ValueError(
                "Cosmos-Dreams-Transfer reads its K/V window from the export's training "
                "config; kv_cache_inference_size/attention_sink_size overrides apply to "
                "the action pipelines only."
            )
        self.transfer_history_profile(manifest)

    # How the control clip reaches the trained canvas. The current Hub export
    # (``*_direct_resize_caption_mix*``) trained without reflection padding: full
    # frame, per-modality interpolation, at most 3% aspect distortion, and its
    # reference preset leaves the caption without the control-adherence sentence.
    # ``center_crop`` reproduces the Sep/Oct-1 exports (aspect-preserving resize
    # plus center crop, control sentence per the artifact).
    transfer_resize_mode: str = TRANSFER_RESIZE_DIRECT

    def update_config_from_dict(self, args, prefix: str = "") -> None:
        super().update_config_from_dict(args, prefix)
        if self.transfer_resize_mode not in TRANSFER_RESIZE_MODES:
            raise ValueError(
                "Cosmos-Dreams-Transfer transfer_resize_mode must be one of "
                f"{TRANSFER_RESIZE_MODES}, got {self.transfer_resize_mode!r}."
            )

    @property
    def emphasize_control_default(self) -> bool | None:
        """Request default for the control-adherence sentence: the direct-resize
        reference preset keeps captions as supplied; legacy exports follow the artifact."""
        return False if self.transfer_resize_mode == TRANSFER_RESIZE_DIRECT else None

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
        mode = resolve_history_mode(
            manifest,
            history_mode=self.history_mode,
            model_path=self.model_path,
            trained_window=trained_window,
        )
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


def register():
    from sglang.multimodal_gen.configs.sample.cosmos_dreams_transfer import (
        CosmosDreamsTransferSamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    # Cosmos-Dreams-Transfer: the same causal recipe conditioned on a control
    # video (edge, blur, depth, or seg) instead of actions.
    register_configs(
        sampling_param_cls=CosmosDreamsTransferSamplingParams,
        pipeline_config_cls=CosmosDreamsTransferConfig,
        hf_model_paths=[
            "nvidia/Cosmos3-Nano-Sim-Transfer",
            "nvidia/Cosmos3-Nano-Sim-Depth",
        ],
        # Matches the release ``Cosmos3NanoSimTransferPipeline`` ``_class_name``.
        model_detectors=[
            lambda hf_id: "cosmos3nanosimtransferpipeline" in hf_id.lower()
        ],
    )
