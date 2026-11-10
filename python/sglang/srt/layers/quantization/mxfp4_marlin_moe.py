"""MXFP4 MoE → Marlin W4A16.

Hopper (SM90/SM120) and SM70 (V100 ``marlin_v100``) share this method.
Do not invent a third format and do not feed this path into
``sm70_nvfp4_moe_decode`` (that kernel is NVFP4: E4M3 g16 + FP32 global).
Decode M<=4 on SM70 uses ``sm70_dsv41_mxfp4_moe_decode`` (same packed
MXFP4 layout); kill-switch ``SGLANG_DSV41_MOE_GEMV=0``. Spilled decode
hits use D4-H host GEMV (``SGLANG_DSV41_HOST_GEMV``) instead of a second
landing-pool GEMV.

Exact SM70 pack (DeepSeek-V4.1-Flash experts):

* Weights: MXFP4 e2m1, two nibbles per uint8 (low=even-K, high=odd-K),
  LUT ``{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}``. Checkpoint ``[E, N, K/2]``.
  ``view(int32).transpose(1, 2)`` → GPTQ ``[E, K/8, N]``, then
  ``gptq_marlin_moe_repack`` from marlin_v100 (not the SM70 JIT stub).
  Packed ``[E, K/16, N*(num_bits/2)]`` int32.
* Scales: UE8M0 g32, checkpoint ``[E, N, K/32]`` (byte 127 = 1.0).
  SM70 consumes raw E8M0 ``[E, K/32, N]``. No Hopper ``marlin_permute_scales``,
  no 16-bit pair-swap, no NVFP4 S0E5M3, no global_scale.
* Activations: FP16. Hopper upcasts FP16+E8M0 to BF16 because sgl-kernel
  never instantiates that combo; marlin_v100 does, so SM70 must not upcast.
* FP16 E8M0 dequant is exact iff scale bytes ∈ {0} ∪ [113, 142]. Otherwise
  fail loud (Hopper BF16 would keep the exponent).
* 384 routed experts / top-6 / hidden=5120 / intermediate=2304 fit one
  ``moe_wna16_marlin_gemm``; there is no expert-count cap. Do not drop
  experts to match the Qwen NVFP4 decode kernel.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils import log_info_on_rank0, round_up

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import RoutingMethodType

logger = logging.getLogger(__name__)


def build_marlin_moe_quant_info(layer: Module) -> MarlinMoeQuantInfo:
    """Build the Marlin quant_info for an MXFP4 MoE layer.

    Single source for the runner inputs shared by the marlin path of
    ``Mxfp4MoEMethod.apply`` and :class:`Mxfp4MarlinMoEMethod`, including
    the dispatcher's EP mapping (global -> local expert ids) when EP is on.
    """
    expert_map = getattr(layer.dispatcher, "local_expert_mapping", None)
    global_num_experts = layer.dispatcher.num_experts if expert_map is not None else -1
    return MarlinMoeQuantInfo(
        w13_qweight=layer.w13_weight,
        w2_qweight=layer.w2_weight,
        w13_scales=layer.w13_weight_scale,
        w2_scales=layer.w2_weight_scale,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        is_k_full=True,
        w13_bias=getattr(layer, "w13_weight_bias", None),
        w2_bias=getattr(layer, "w2_weight_bias", None),
        expert_map=expert_map,
        global_num_experts=global_num_experts,
    )


class Mxfp4MarlinMoEMethod:
    """MXFP4 (E8M0 scales) MoE quantization method using the Marlin backend."""

    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_moe_runner(self, layer, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.runner = MoeRunner(MoeRunnerBackend.MARLIN, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return self.runner.runner_backend

    def get_moe_routing_method_type(
        self, layer: Module, requested_method: RoutingMethodType | None
    ) -> RoutingMethodType | None:
        return requested_method

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        layer._dsv4_mxfp4_backend = None  # set in process_weights_after_loading
        fp4_block_k = 32
        # Hopper repacking pads the loaded gate/up halves and down columns to
        # the required tile sizes. Keep their logical widths until then: early
        # padding to 128 turns a TP8 width of 288 into 384 instead of 320.
        if not get_platform().is_sm90:
            intermediate_size_per_partition = round_up(
                intermediate_size_per_partition, 128
            )
        hidden_size = round_up(hidden_size, 256)
        self.hidden_pad = hidden_size - layer.hidden_size

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # Store loader scales in E8M0; uint8 127 encodes 1.0.
        def _e8m0_ones(*shape: int) -> torch.Tensor:
            return torch.full(shape, 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        w13_weight_scale = torch.nn.Parameter(
            _e8m0_ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            _e8m0_ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.marlin_utils import (
            check_moe_marlin_supports_layer,
        )
        from sglang.srt.layers.quantization.marlin_utils_fp4 import (
            deinterleave_moe_mxfp4_w13_for_marlin,
            prepare_moe_mxfp4_layer_for_marlin,
        )

        # Let the FP8 base method handle ROCm normalization, etc.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        platform = get_platform()
        if platform.is_sm70:
            from sglang.srt.layers.quantization.marlin_utils import (
                _sm70_marlin_v100_available,
            )

            if not _sm70_marlin_v100_available():
                raise RuntimeError(
                    "MXFP4 Marlin on SM70 requires marlin_v100 "
                    "(scripts/setup_v100_marlin.sh). Refusing the SM70 JIT "
                    "repack stub, which would silently zero expert weights."
                )
        elif not platform.is_sm90 and not platform.is_sm120:
            raise RuntimeError(
                "MXFP4 Marlin requires SM90, SM120, or SM70+marlin_v100."
            )

        if not check_moe_marlin_supports_layer(layer, 32, allow_tile_padding=True):
            raise RuntimeError(
                "Current MXFP4 MoE layer does not satisfy Marlin constraints."
            )

        # NOTE: the Marlin MoE runner consumes w13 in the checkpoint's
        # native ``[w1; w3]`` order -- see ``silu_and_mul`` in
        # fused_marlin_moe.py which expects ``gate = intermediate[:, :N]``
        # (first half) and ``up = intermediate[:, N:]`` (second half).
        # Unlike the flashinfer trtllm_fp4 kernel (which wants [w3, w1]),
        # we must *not* call ``reorder_w1w3_to_w3w1`` here.

        log_info_on_rank0(
            logger,
            f"Preparing MXFP4 experts for Marlin backend (layer: {self.prefix})...",
        )
        if self.runner.config.gemm1_alpha is not None:
            deinterleave_moe_mxfp4_w13_for_marlin(layer)
        prepare_moe_mxfp4_layer_for_marlin(layer)
        layer._dsv4_mxfp4_backend = "marlin"

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")
        hidden_states = dispatch_output.hidden_states
        target_hidden_size = layer.w13_weight.shape[1] * 16
        if hidden_states.shape[-1] == target_hidden_size:
            hidden_states_padded = hidden_states
        else:
            hidden_states_padded = torch.nn.functional.pad(
                hidden_states,
                (0, target_hidden_size - hidden_states.shape[-1]),
                mode="constant",
                value=0.0,
            )

        quant_info = build_marlin_moe_quant_info(layer)
        padded_dispatch = dispatch_output._replace(hidden_states=hidden_states_padded)
        runner_output = self.runner.run(
            padded_dispatch,
            quant_info=quant_info,
        )

        hs = runner_output.hidden_states
        # D4-H: CPU workers overlap this GPU GEMV; join adds host y.
        # Always join when the request ran so CUDA graph capture cannot skip
        # it via a host-side occupancy check. D4-G landing is the fallback.
        if getattr(layer, "_dsv41_host_gemv_pending", False):
            from sglang.srt.layers.moe.dsv41_expert_spill import spill_join_host_gemv

            spill_join_host_gemv(layer, hs)
        else:
            land_ids = getattr(layer, "_dsv41_land_ids", None)
            pool = getattr(layer, "_dsv41_landing_pool", None)
            if land_ids is not None and pool is not None and pool.quant_info is not None:
                land_topk = topk_output._replace(topk_ids=land_ids)
                land_out = self.runner.run(
                    padded_dispatch._replace(topk_output=land_topk),
                    quant_info=pool.quant_info,
                )
                hs = hs + land_out.hidden_states
        return StandardCombineInput(hidden_states=hs)
