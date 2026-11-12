# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)


@dataclass
class PhysisH3PipelineConfig(MiniMaxH3PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2V
    native_only_components: tuple[str, ...] = ("text_encoder", "transformer")
    output_audio_sample_rate: int | None = None
    output_audio_channels: int | None = None
    output_av_drift_tolerance_s: float | None = None
    physis_history_chunks: int = 18
    physis_bootstrap_chunks: int = 5

    def supports_causal_pipeline(self) -> bool:
        return True

    def causal_pipeline_num_steps(self) -> int:
        return 4

    def validate_server_args(self, server_args) -> None:
        if not current_platform.is_cuda():
            raise ValueError("Physis H3 requires NVIDIA CUDA")
        backend = self.resolve_transformer_attention_backend(server_args)
        if backend not in (None, AttentionBackendEnum.TORCH_SDPA):
            raise ValueError("Physis H3 requires torch_sdpa attention")
        if backend is None:
            server_args.attention_backend = "torch_sdpa"
        if (
            server_args.kv_cache_quant_config.enabled
            or server_args.cache_dit_config is not None
            or envs.SGLANG_CACHE_DIT_ENABLED
        ):
            raise ValueError("Physis H3 does not support KV quantization or cache-dit")
        if server_args.encoder_parallel not in ("auto", "replicate"):
            raise ValueError(
                "Physis H3 encodes text on the decoder rank; use replicated encoders"
            )
        server_args.encoder_parallel = "replicate"
        if self.physis_history_chunks < 0:
            raise ValueError("physis_history_chunks must be non-negative")
        if not 1 <= self.physis_bootstrap_chunks <= self.physis_history_chunks + 1:
            raise ValueError(
                "physis_bootstrap_chunks must be between 1 and history_chunks + 1"
            )
        if (
            server_args.tp_size != 1
            or server_args.ring_degree != 1
            or server_args.dp_size != 1
            or server_args.enable_cfg_parallel
        ):
            raise ValueError(
                "Physis H3 supports Ulysses or causal pipeline parallelism"
            )
        if (
            server_args.enable_torch_compile
            or server_args.enable_breakable_cuda_graph
            or server_args.use_fsdp_inference
            or server_args.has_layerwise_offload_components()
            or server_args.lora_path
            or server_args.comfyui_mode
            or server_args.quantization
            or server_args.component_quantizations
            or server_args.minimax_h3_adaln_online
            or server_args.minimax_h3_adaln_cache_path
        ):
            raise ValueError("Physis H3 requires eager, unquantized native DiT")
        if server_args.batching_max_size != 1:
            raise ValueError("Physis H3 requires one request at a time")
        if server_args.warmup_mode != "off":
            raise ValueError(
                "Physis H3 requires --warmup-mode off; warm with a real request"
            )


def register():
    from sglang.multimodal_gen.configs.sample.physis_h3 import PhysisH3SamplingParams
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=PhysisH3SamplingParams,
        pipeline_config_cls=PhysisH3PipelineConfig,
        hf_model_paths=["Physis-AI/physisserve-minimaxh3_33b-same_step-ls-dev"],
        model_detectors=[lambda model_id: "physisserve-minimaxh3" in model_id.lower()],
    )
