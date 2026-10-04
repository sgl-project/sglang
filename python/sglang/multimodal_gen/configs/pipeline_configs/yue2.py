# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)


@dataclass
class Yue2PipelineConfig(PipelineConfig):
    """YuE2 AR-NAR music-generation pipeline configuration."""

    task_type: ModelTaskType = ModelTaskType.T2I  # closest existing task type
    vae_precision: str = "fp32"
    vae_decode_precision: str = "fp32"
    vae_tiling: bool = True
    dit_precision: str = "bf16"
    output_audio_sample_rate: int = 48000
    output_audio_channels: int = 2

    def accepts_audio_input(self) -> bool:
        return False

    @property
    def requires_audio_output(self) -> bool:
        return True

    def supports_disaggregation(self) -> bool:
        return False

    def supports_native_grouped_requests(self) -> bool:
        """YuE2 batches the semantic AR phase via ``run_grouped_requests``.

        The scheduler's dynamic-batch signature still gates which requests may
        share a group; the AR stage additionally requires CoT/full, no CFG, and
        no external ABC, and falls back to per-request decoding otherwise.
        """
        return True

    # NOTE (yiakwy) : as 2-stage streaming method, we evalate cost to merge requests
    def estimate_request_cost(self, batch) -> float:
        """Relative admission cost = estimated AR token volume of one request.

        The batched resource is the AR decode KV: it scales with the prompt
        prefix (prefill + KV base), the ABC planning budget, and the semantic
        codec budget. Prefix length is estimated from the text because the
        tokenised prefix is only built later (in ``Yue2PrepareRequestStage``).
        Falls back to the base estimator (image latent volume) and finally
        ``1.0`` for requests without the YuE2 sampling fields.
        """
        params = getattr(batch, "sampling_params", None)
        semantic = float(getattr(params, "semantic_max_tokens", 0) or 0) if params else 0.0
        abc = float(getattr(params, "abc_max_tokens", 0) or 0) if params else 0.0
        if semantic <= 0 and abc <= 0:
            # Not a YuE2 request (fields absent): defer to the base estimator.
            try:
                return super().estimate_request_cost(batch)
            except Exception:
                return 1.0

        text = f"{getattr(params, 'style', '') or ''}\n{getattr(params, 'lyrics', '') or ''}"
        prefix = len(text) / 3.0 + 32.0
        return prefix + abc + semantic

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            dit_layerwise_offload_modes=("auto", "memory"),
            keep_resident_min_available_gb=24,
            keep_resident_components=("dit", "vae"),
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=False,
            speed_mode_enable_torch_compile_by_default=False,
        )

    def validate_server_args(self, server_args) -> None:
        if getattr(server_args, "disagg_role", None) not in (None, "monolithic"):
            raise ValueError("YuE2 supports only monolithic deployment")


def register():
    from sglang.multimodal_gen.registry import register_configs

    from sglang.multimodal_gen.configs.sample.yue2 import Yue2SamplingParams

    register_configs(
        sampling_param_cls=Yue2SamplingParams,
        pipeline_config_cls=Yue2PipelineConfig,
        hf_model_paths=["YuE2-3B", "m-a-p/yue2-3b"],
        model_detectors=[lambda value: "yue2-3b" in value.lower()],
    )
