# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 pipeline config: the MiniMax-H3 deployment envelope for the hybrid
attention checkpoint."""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)


@dataclass
class VDNH3PipelineConfig(MiniMaxH3PipelineConfig):
    """VDN-H3 (hybrid window-softmax + Video Delta linear attention, 8-NFE DMD2
    distill, t2va only): the deployment envelope; the arch config comes from
    the materialized ``transformer/config.json``."""

    def validate_quality_deployment(self, server_args) -> None:
        raise ValueError(
            'quality="high" is audited only for the base MiniMax-H3 50-step '
            "4xH200 deployment; the VDN-H3 8-step hybrid checkpoint has no "
            'audited high-quality deployment. Use quality="lossless".'
        )

    def validate_server_args(self, server_args) -> None:
        if server_args.model_variant is not None:
            raise ValueError(
                "VDN-H3 ships one t2va weight partition; --model-variant does "
                "not apply. FL2VA and Ref2VA tasks were not trained; use "
                "MiniMaxAI/MiniMax-H3 for those."
            )
        quantization = (server_args.quantization or "").lower()
        if quantization in ("none", "bf16"):
            server_args.quantization = None
        elif (
            current_platform.is_blackwell() or current_platform.is_sm120()
        ) and quantization in ("", "fp8"):
            # the default: online MXFP8, 0.93 s/NFE on 8x B200 against 1.05
            # per-channel fp8. The block-scaled GEMM needs SM100+; on SM120
            # (RTX PRO 6000) it is a cutlass sm120 block-scaled kernel that beats
            # the per-channel path's generic GEMM by 37% (1 GPU: 20.3 vs 22.8
            # s/NFE on the paper workload).
            server_args.quantization = "mxfp8"
        # an unset backend would resolve to the platform default (dense FA)
        if server_args.attention_backend is None and not (
            server_args.component_attention_backends or {}
        ).get("transformer"):
            server_args.attention_backend = "hybrid_window_attn_h3"
        selected_backend = self.resolve_transformer_attention_backend(server_args)
        if selected_backend is not AttentionBackendEnum.HYBRID_WINDOW_ATTN_H3:
            # the base-H3+LoRA equivalence smoke runs plain attention on purpose
            config = server_args.attention_backend_config or {}
            if not bool(config.get("vdn_h3_dense_smoke", False)):
                raise ValueError(
                    "VDN-H3 requires --attention-backend hybrid_window_attn_h3 for "
                    f"the transformer (got {selected_backend}); a dense backend "
                    "would skip the linear branch and the softmax gates and "
                    "produce the wrong model, not a slower one. Pass "
                    "--attention-backend-config '{\"vdn_h3_dense_smoke\": true}' "
                    "only for the base-H3+LoRA equivalence smoke."
                )
        if int(server_args.ring_degree or 1) > 1:
            raise ValueError(
                "VDN-H3 does not support --ring-degree > 1; use Ulysses sequence "
                "parallelism."
            )
        if server_args.enable_torch_compile or server_args.enable_breakable_cuda_graph:
            # BCG keeps one pool per captured segment and exhausts 183 GB at 104k rows
            raise ValueError(
                "VDN-H3 hybrid attention is not validated under torch.compile or "
                "the breakable CUDA graph yet; disable them."
            )
        super().validate_server_args(server_args)


__all__ = ["VDNH3PipelineConfig"]
