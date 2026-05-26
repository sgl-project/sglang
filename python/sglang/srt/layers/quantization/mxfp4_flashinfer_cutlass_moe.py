"""DeepSeek-V4 MXFP4 expert backend backed by FlashInfer's SM90 cutlass
mixed-input MoE GEMM (FlashInfer PR #3084).

Sibling of :class:`Mxfp4MarlinMoEMethod` and :class:`Mxfp4FlashinferTrtllmMoEMethod`.
Wired into :func:`Fp8MoEConfig.get_quant_method` when
``is_fp4_experts=True`` and ``--moe-runner-backend flashinfer_mxfp4`` is
selected on a Hopper (SM90) device. SM100 still routes to
:class:`Mxfp4FlashinferTrtllmMoEMethod` (trtllm-gen).

Performance trade-off vs Marlin (kernel-level on H100, GPT-OSS-like body):
  - decode  (M <=   64) :  Marlin     +12-15 %
  - tie     (M ~=  256)
  - prefill (M >= 1024) :  FlashInfer +24-36 %

PD-disaggregated prefill workers are the natural fit; decode workers should
keep the Marlin default.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.utils import is_flashinfer_available, log_info_on_rank0
from sglang.srt.utils.common import next_power_of_2

# Silence the TRT-LLM cutlass autotune trace embedded inside FlashInfer's
# cutlass_fused_moe. Its C++ logger reads TLLM_LOG_LEVEL on first kernel launch;
# setdefault preserves any explicit user override.
os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

if is_flashinfer_available():
    from flashinfer.fused_moe import cutlass_fused_moe as flashinfer_cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    try:
        from flashinfer.fused_moe import (
            interleave_moe_scales_for_sm90_mixed_gemm,
            interleave_moe_weights_for_sm90_mixed_gemm,
        )

        _FI_HAS_SM90_CUTLASS_MXFP4 = True
    except ImportError:
        interleave_moe_scales_for_sm90_mixed_gemm = None
        interleave_moe_weights_for_sm90_mixed_gemm = None
        _FI_HAS_SM90_CUTLASS_MXFP4 = False
else:
    _FI_HAS_SM90_CUTLASS_MXFP4 = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

# MXFP4 group/block size (E8M0 scale per 32 fp4 weights).
_GROUP_SIZE = 32


class Mxfp4FlashinferCutlassMoEMethod:
    """DeepSeek-V4 W4A16 MXFP4 MoE via FlashInfer's SM90 mixed-input cutlass
    grouped GEMM. The fused kernel does GEMM1 + clamped SwiGLU + GEMM2 in one
    call after a one-shot weight/scale interleave at load time."""

    def __init__(self, fp8_method, prefix: str):
        if not _FI_HAS_SM90_CUTLASS_MXFP4:
            raise RuntimeError(
                "Mxfp4FlashinferCutlassMoEMethod requires FlashInfer >= 0.6.11 "
                "(PR #3084 SM90 mixed-input helpers). Older builds lack "
                "interleave_moe_{weights,scales}_for_sm90_mixed_gemm; "
                "either upgrade flashinfer-python or fall back to "
                "--moe-runner-backend marlin."
            )
        self._fp8 = fp8_method
        self.prefix = prefix
        self._swiglu_alpha_tensor: torch.Tensor | None = None
        self._swiglu_beta_tensor: torch.Tensor | None = None
        self._swiglu_limit_tensor: torch.Tensor | None = None

    # --- Lifecycle ---------------------------------------------------------

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        # SM90 mixed-input GEMM: contraction dim K must be a multiple of 128
        # (interleave factor = 128 / group_size = 4). For DSv4 (hidden=7168,
        # inter=2048) both are already multiples of 128; we assert rather than
        # silently pad here, since padding the FP8-base buffers in-place would
        # require deeper changes.
        if hidden_size % 128 != 0 or intermediate_size_per_partition % 128 != 0:
            raise ValueError(
                "Mxfp4FlashinferCutlassMoEMethod requires hidden_size and "
                "intermediate_size_per_partition to be multiples of 128 "
                f"(got hidden={hidden_size}, "
                f"intermediate={intermediate_size_per_partition})."
            )
        # Raw weight shapes match what the fp8 base method allocates for fp4
        # experts (uint8 4-bit packed weights, fp32 E8M0 scales). Delegate.
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        self.moe_runner_config = moe_runner_config

        # DSv4 uses standard SwiGLU plus a config-driven activation clamp.
        # We pass all three (alpha, beta, limit) as explicit per-expert tensors
        # rather than mixing tensors with None: the cutlass SwiGLU kernel
        # branches on whether each is None, and partial-None inputs land in
        # less-tested code paths. ``alpha=1.0``, ``beta=0.0`` reproduce plain
        # ``silu(gate) * up``; ``limit`` enforces the activation clamp the
        # checkpoint was trained with.
        swiglu_limit = getattr(moe_runner_config, "swiglu_limit", None)
        if swiglu_limit is not None:
            E = layer.num_local_experts
            device = layer.w13_weight.device
            self._swiglu_alpha_tensor = torch.ones(
                E, dtype=torch.float32, device=device
            )
            self._swiglu_beta_tensor = torch.zeros(
                E, dtype=torch.float32, device=device
            )
            self._swiglu_limit_tensor = torch.full(
                (E,), float(swiglu_limit), dtype=torch.float32, device=device
            )
        else:
            self._swiglu_alpha_tensor = None
            self._swiglu_beta_tensor = None
            self._swiglu_limit_tensor = None

    def process_weights_after_loading(self, layer: Module) -> None:
        from sglang.srt.layers.quantization.utils import reorder_w1w3_to_w3w1

        # Run the fp8 base hook first (ROCm normalization, mxfp8 requant, ...).
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        # cutlass_fused_moe expects fc1 in [w3; w1] = [up; gate] order, just
        # like the trtllm-gen path. The HF / FP8 loader emits [w1; w3].
        w13, w13_s = reorder_w1w3_to_w3w1(
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        layer.w13_weight = Parameter(w13, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_s, requires_grad=False)

        log_info_on_rank0(
            logger,
            f"Preparing DSv4 MXFP4 experts for FlashInfer SM90 cutlass "
            f"(layer: {self.prefix})...",
        )

        # FP8 base stores scales as fp32 numerical values (= 2**e). The
        # FlashInfer SM90 helper reads raw E8M0 bytes (uint8 with the
        # exponent + 127 bias). Cast through float8_e8m0fnu to extract the
        # raw byte without losing the exponent.
        w13_scale_u8 = (
            layer.w13_weight_scale_inv.data.to(torch.float8_e8m0fnu)
            .view(torch.uint8)
            .contiguous()
        )
        w2_scale_u8 = (
            layer.w2_weight_scale_inv.data.to(torch.float8_e8m0fnu)
            .view(torch.uint8)
            .contiguous()
        )

        # C++ byte interleave on packed 4-bit weights.
        w13_il = interleave_moe_weights_for_sm90_mixed_gemm(
            layer.w13_weight.data.view(torch.uint8).contiguous(), "fp4"
        )
        w2_il = interleave_moe_weights_for_sm90_mixed_gemm(
            layer.w2_weight.data.view(torch.uint8).contiguous(), "fp4"
        )
        # Pure-PyTorch reshape+permute on E8M0 block scales.
        w13_s_il = interleave_moe_scales_for_sm90_mixed_gemm(
            w13_scale_u8, group_size=_GROUP_SIZE
        )
        w2_s_il = interleave_moe_scales_for_sm90_mixed_gemm(
            w2_scale_u8, group_size=_GROUP_SIZE
        )

        layer.w13_weight = Parameter(w13_il, requires_grad=False)
        layer.w2_weight = Parameter(w2_il, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_s_il, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_s_il, requires_grad=False)

        layer._dsv4_mxfp4_backend = "flashinfer_cutlass_sm90"
        torch.cuda.empty_cache()

    # --- Forward -----------------------------------------------------------

    def apply(
        self,
        layer: Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        topk_output = dispatch_output.topk_output
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        x = dispatch_output.hidden_states
        topk_weights = topk_output.topk_weights
        topk_ids = topk_output.topk_ids

        output_dtype = torch.bfloat16
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            out = torch.empty(
                x.shape[0], x.shape[-1], dtype=output_dtype, device=x.device
            )

        flashinfer_cutlass_fused_moe(
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=layer.w13_weight,
            fc2_expert_weights=layer.w2_weight,
            output_dtype=output_dtype,
            quant_scales=[
                layer.w13_weight_scale_inv.view(torch.int32),
                layer.w2_weight_scale_inv.view(torch.int32),
            ],
            fc1_expert_biases=None,  # DSv4 has no MoE expert bias.
            fc2_expert_biases=None,
            swiglu_alpha=self._swiglu_alpha_tensor,  # ones: standard SiLU gate
            swiglu_beta=self._swiglu_beta_tensor,  # zeros: standard up
            swiglu_limit=self._swiglu_limit_tensor,
            tp_size=layer.moe_tp_size,
            tp_rank=layer.moe_tp_rank,
            ep_size=layer.moe_ep_size,
            ep_rank=layer.moe_ep_rank,
            use_w4_group_scaling=True,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            output=out,
        )

        return StandardCombineInput(hidden_states=out)
