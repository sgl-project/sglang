# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from typing import Any

import torch

from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsFusedMoEMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Nvfp4MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


class MiniMaxM3Nvfp4Config(Fp8Config):
    """MXFP8 projections and NVFP4 routed experts in static M3.1 checkpoints."""

    def __init__(
        self,
        *,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        packed_modules_mapping: dict[str, list[str]] | None = None,
        kv_cache_quant_algo: str = "FP8",
        index_k_quant_type: str = "nvfp4",
        attention_quant_algo: str = "FP8",
        group_size: int = 16,
    ) -> None:
        if activation_scheme != "dynamic":
            raise ValueError("M3.1 NVFP4 requires dynamic activation quantization")
        if group_size != 16:
            raise ValueError("M3.1 NVFP4 requires group_size=16")
        if (
            not isinstance(kv_cache_quant_algo, str)
            or kv_cache_quant_algo.upper() != "FP8"
        ):
            raise ValueError("M3.1 NVFP4 requires kv_cache_quant_algo='FP8'")
        if (
            not isinstance(attention_quant_algo, str)
            or attention_quant_algo.upper() != "FP8"
        ):
            raise ValueError("M3.1 NVFP4 requires attention_quant_algo='FP8'")
        if (
            not isinstance(index_k_quant_type, str)
            or index_k_quant_type.lower() != "nvfp4"
        ):
            raise ValueError("M3.1 NVFP4 requires index_k_quant_type='nvfp4'")
        super().__init__(
            is_checkpoint_fp8_serialized=True,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=[1, 32],
            packed_modules_mapping=packed_modules_mapping,
            use_mxfp8=True,
            kv_cache_quant_algo="FP8",
        )
        self.index_k_quant_type = "nvfp4"
        self.attention_quant_algo = "FP8"
        self.group_size = group_size
        self.is_minimax_m3_nvfp4 = True
        # MX-scaled matmul is Blackwell-only; elsewhere the projections run as
        # block-FP8 [128,128], converted at load.
        self.convert_mxfp8_to_block_fp8 = not get_platform().is_blackwell

    def get_min_capability(self) -> int:
        # Overrides the inherited MXFP8 gate of 100; the block-FP8 projections
        # and W4A16 experts need FP8 tensor cores. Validated on SM90 and SM100.
        return 90

    @classmethod
    def get_name(cls) -> str:
        return "minimax_m3_nvfp4"

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> MiniMaxM3Nvfp4Config:
        if config.get("weight_block_size", [1, 32]) != [1, 32]:
            raise ValueError("M3.1 MXFP8 projections require weight_block_size=[1, 32]")
        ignored_layers = config.get(
            "ignored_layers", config.get("modules_to_not_convert")
        )
        if ignored_layers:
            normalized = []
            for name in ignored_layers:
                name = name.removeprefix("model.")
                normalized.extend((name, f"model.{name}"))
            ignored_layers = list(dict.fromkeys(normalized))
        return cls(
            activation_scheme=config.get("activation_scheme", "dynamic"),
            ignored_layers=ignored_layers,
            packed_modules_mapping=config.get("packed_modules_mapping", {}),
            kv_cache_quant_algo=config.get("kv_cache_quant_algo", "FP8"),
            index_k_quant_type=config.get("index_k_quant_type", "nvfp4"),
            attention_quant_algo=config.get("attention_quant_algo", "FP8"),
            group_size=config.get("group_size", 16),
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if not isinstance(layer, FusedMoE) or is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        ):
            return super().get_quant_method(layer, prefix)
        layer.scheme = MiniMaxM3Nvfp4MoE()
        return CompressedTensorsFusedMoEMethod(self)


class MiniMaxM3Nvfp4MoE(CompressedTensorsW4A4Nvfp4MoE):
    """MegaMoE and TRT-LLM experts on Blackwell, Marlin W4A16 elsewhere."""

    # Cleared because __init__ below supplies a non-Blackwell expert path.
    requires_blackwell = False

    def __init__(self):
        super().__init__()
        # Marlin unpacks the packed NVFP4 weights in-register and multiplies on
        # bf16 tensor cores, keeping the 4-bit footprint without FP4 hardware.
        self.use_marlin = not get_platform().is_blackwell
        self.use_flashinfer_trtllm = (
            not self.use_marlin
            and get_moe_runner_backend().is_flashinfer_trtllm_routed()
        )
        self.use_mega_moe_prefill = (
            self.use_flashinfer_trtllm and get_moe_a2a_backend().is_megamoe()
        )
        if (
            not self.use_marlin
            and not self.use_flashinfer_trtllm
            and not get_moe_a2a_backend().is_megamoe()
        ):
            raise ValueError(
                "M3.1 NVFP4 experts require --moe-a2a-backend megamoe or "
                "--moe-runner-backend flashinfer_trtllm_routed"
            )

    @classmethod
    def get_min_capability(cls) -> int:
        # Marlin W4A16 covers SM80 and above.
        return 80

    @property
    def load_up_proj_weight_first(self) -> bool:
        # Compressed-tensors loaders do not perform the ModelOpt W31 remap.
        return False

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        if self.use_marlin:
            from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend

            self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)
        elif self.use_flashinfer_trtllm:
            import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm  # noqa: F401
            from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend

            if not self.use_mega_moe_prefill:
                self.runner = MoeRunner(
                    MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, moe_runner_config
                )

    def _process_marlin_weights(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            prepare_moe_nvfp4_layer_for_marlin,
        )

        # The Marlin helper reads the ModelOpt parameter names.
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed

        # Checkpoints store encode scales; Marlin dequant wants the decode
        # multiplier. W4A16 leaves the FC2 input in bf16, so its scale is unused.
        w13_global_scale = layer.w13_weight_global_scale.data.float()
        if w13_global_scale.dim() > 1:
            # Marlin carries one global scale per expert, so gate and up must
            # agree; a checkpoint where they differ would be silently rescaled.
            if w13_global_scale.shape[1] >= 2 and not torch.allclose(
                w13_global_scale[:, 0], w13_global_scale[:, 1]
            ):
                logger.warning_once(
                    "Gate and up global scales differ; Marlin keeps only the "
                    "gate scale and accuracy may be affected."
                )
            w13_global_scale = w13_global_scale[:, 0]
        layer.w13_weight_scale_2 = torch.nn.Parameter(
            w13_global_scale.reciprocal().contiguous(), requires_grad=False
        )
        layer.w2_weight_scale_2 = torch.nn.Parameter(
            layer.w2_weight_global_scale.data.float().reciprocal().contiguous(),
            requires_grad=False,
        )
        prepare_moe_nvfp4_layer_for_marlin(layer)

        # The helper allocates repacked weights and rebinds w13_weight/w2_weight
        # to them; the packed parameters still own the original storage, which
        # would otherwise stay resident for every layer.
        layer.register_parameter("w13_weight_packed", None)
        layer.register_parameter("w2_weight_packed", None)

        layer._mega_moe_nvfp4 = False
        layer._mega_moe_weights_built = False

    def get_marlin_quant_info(self, layer: torch.nn.Module):
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardDispatcher,
        )

        # Under expert parallelism the dispatcher marks non-local experts with
        # -1; without the map the kernel indexes weights with it. Only the
        # standard dispatcher carries the mapping.
        expert_map = None
        global_num_experts = -1
        if isinstance(layer.dispatcher, StandardDispatcher):
            expert_map = layer.dispatcher.local_expert_mapping
            if expert_map is not None:
                global_num_experts = self.moe_runner_config.num_experts

        return MarlinMoeQuantInfo(
            w13_qweight=layer.w13_weight,
            w2_qweight=layer.w2_weight,
            w13_scales=layer.w13_weight_scale,
            w2_scales=layer.w2_weight_scale,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            w13_global_scale=layer.w13_weight_scale_2,
            w2_global_scale=layer.w2_weight_scale_2,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
        )

    def _process_trtllm_weights(self, layer: torch.nn.Module) -> None:
        if self.use_mega_moe_prefill:
            _set_mega_moe_alphas(layer)
        # Activations are dynamic: the kernel quantizes both GEMM inputs per
        # token, so the static input scales must be neutral.
        layer.w13_input_global_scale.data.fill_(1.0)
        layer.w2_input_global_scale.data.fill_(1.0)
        super().process_weights_after_loading(layer)

        # TRT-LLM applies the SwiGLU clamp and beta before the g1 dequant scale.
        cfg = layer.moe_runner_config
        g1 = layer.g1_alphas.data
        clamp = cfg.gemm1_clamp_limit or cfg.swiglu_limit
        layer.gemm1_clamp_limit = (
            None if clamp is None else (clamp / g1).to(torch.float32)
        )
        layer.gemm1_alpha = layer.gemm1_beta = None
        if layer._minimax_activation == "swigluoai":
            layer.gemm1_alpha = torch.full_like(g1, cfg.gemm1_alpha)
            layer.gemm1_beta = (cfg.gemm1_beta / g1).to(torch.float32)
        if self.use_mega_moe_prefill:
            _alias_mega_moe_to_trtllm_weights(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.use_marlin:
            self._process_marlin_weights(layer)
            return
        if self.use_flashinfer_trtllm:
            self._process_trtllm_weights(layer)
            return
        import deep_gemm
        from deep_gemm.utils.math import transform_ue4m3_sf_into_required_layout

        from sglang.srt.layers.moe.mega_moe import check_mega_moe_shapes

        w13 = layer.w13_weight_packed.data
        w2 = layer.w2_weight_packed.data
        check_mega_moe_shapes(w13.shape[2] * 2, w13.shape[1] // 2, "nvfp4xnvfp4")
        _set_mega_moe_alphas(layer)
        w13_sf = transform_ue4m3_sf_into_required_layout(
            layer.w13_weight_scale.data, w13.shape[1]
        )
        w2_sf = transform_ue4m3_sf_into_required_layout(
            layer.w2_weight_scale.data, w2.shape[1]
        )
        l1, l2 = deep_gemm.transform_weights_for_mega_moe(
            (w13.view(torch.int8), w13_sf),
            (w2.view(torch.int8), w2_sf),
            mma_type="nvfp4xnvfp4",
            activation=layer._minimax_activation,
        )
        layer.mega_l1_weights = l1
        layer.mega_l2_weights = l2
        layer.w13_weight_packed.data = l1[0]
        layer.w13_weight_scale.data = l1[1]
        layer.w2_weight_packed.data = l2[0]
        layer.w2_weight_scale.data = l2[1]
        layer._mega_moe_nvfp4 = True
        layer._mega_moe_weights_built = True

    def apply_weights(self, layer, dispatch_output):
        if self.use_marlin:
            return self.runner.run(dispatch_output, self.get_marlin_quant_info(layer))
        if self.use_flashinfer_trtllm:
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                FlashInferTrtllmFp4MoeQuantInfo,
            )

            quant_info = FlashInferTrtllmFp4MoeQuantInfo(
                w13_weight=layer.w13_weight,
                w2_weight=layer.w2_weight,
                w13_weight_scale=layer.w13_weight_scale,
                w2_weight_scale=layer.w2_weight_scale,
                g1_scale_c=layer.g1_scale_c,
                g1_alphas=layer.g1_alphas,
                g2_alphas=layer.g2_alphas,
                w13_input_scale_quant=layer.w13_input_scale_quant[:1],
                global_num_experts=layer.num_experts,
                local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
                local_num_experts=layer.num_local_experts,
                intermediate_size_per_partition=layer.intermediate_size_per_partition,
                routing_method_type=layer.routing_method_type,
                use_per_token_activation=True,
                gemm1_alpha=layer.gemm1_alpha,
                gemm1_beta=layer.gemm1_beta,
                gemm1_clamp_limit=layer.gemm1_clamp_limit,
            )
            if self.use_mega_moe_prefill:
                from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                    fused_experts_none_to_flashinfer_trtllm_fp4,
                )

                return fused_experts_none_to_flashinfer_trtllm_fp4(
                    dispatch_output,
                    quant_info,
                    layer.moe_runner_config,
                    use_routed_topk=True,
                )
            return self.runner.run(dispatch_output, quant_info)
        raise RuntimeError("M3.1 NVFP4 experts must execute through the MegaMoE path")


def _set_mega_moe_alphas(layer: torch.nn.Module) -> None:
    scales = (
        layer.w13_weight_global_scale.data,
        layer.w2_weight_global_scale.data,
        layer.w2_input_global_scale.data,
    )
    if any(not torch.all(torch.isfinite(s) & (s > 0)) for s in scales):
        raise ValueError(
            "M3.1 NVFP4 weight and FC2 input scales must be finite and positive"
        )
    # Checkpoints store encode scales. Apply each inverse before the
    # nonlinear FC1 activation, and the FC2 input inverse at its epilogue.
    layer.mega_l1_alphas = scales[0].float().reciprocal().contiguous()
    # TRT-LLM preparation overwrites the source scale in place.
    layer.mega_l2_act_scales = scales[2].float().clone()
    layer.mega_l2_alphas = (scales[1].float().reciprocal() / scales[2]).contiguous()


def _alias_mega_moe_to_trtllm_weights(layer: torch.nn.Module) -> None:
    from sglang.srt.layers.moe.mega_moe import check_mega_moe_shapes

    num_experts, l1_rows, l1_k_bytes = layer.w13_weight.shape
    _, l2_rows, l2_k_bytes = layer.w2_weight.shape
    check_mega_moe_shapes(l1_k_bytes * 2, l1_rows // 2, "nvfp4xnvfp4")
    layer.mega_l1_weights = (
        layer.w13_weight.data.view(torch.int8),
        layer.w13_weight_scale.data.view(torch.int32).view(
            num_experts, l1_rows, l1_k_bytes // 32
        ),
    )
    layer.mega_l2_weights = (
        layer.w2_weight.data.view(torch.int8),
        layer.w2_weight_scale.data.view(torch.int32).view(
            num_experts, l2_rows, l2_k_bytes // 32
        ),
    )
    layer._mega_moe_nvfp4 = True
    layer._mega_moe_trtllm_layout = True
    layer._mega_moe_weights_built = True
