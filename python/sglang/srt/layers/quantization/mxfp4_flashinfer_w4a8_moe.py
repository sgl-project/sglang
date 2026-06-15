"""DeepSeek-V4 W4A8 MXFP4 expert backend backed by FlashInfer's SM90 CuTe-DSL
W4A8 grouped-GEMM MoE (``flashinfer.fused_moe.w4a8_mxfp4_moe``).

This is the **W4A8** sibling of :class:`Mxfp4FlashinferCutlassMoEMethod` (which is
**W4A16**: FP4 weight x BF16 activation). Same MXFP4 weights / E8M0 per-32 block scales /
clamped SwiGLU; the only difference is the activation is cast to FP8 e4m3 so the expert
GEMMs run on Hopper's FP8 tensor cores (2x the BF16 throughput). Selected by
``--moe-runner-backend flashinfer_mxfp4 --flashinfer-mxfp4-moe-activation fp8`` on a Hopper
(SM90) device with ``is_fp4_experts=True``; ``--flashinfer-mxfp4-moe-activation bf16``
(default) keeps the W4A16 cutlass path.

Performance (kernel-level on H20-3e, DSv4 shape E=256/top_k=6/H=4096/I=2048, vs the W4A16
cutlass path):
  - tie     (M ~= 48, T ~= 2048)
  - prefill (M >= 96, T >= 4096)           : W4A8 1.6-1.9x -- WINS.
So this is a **prefill-worker** kernel (the same PD-disaggregated split the W4A16 cutlass
path already documents); decode workers should not select it.

Caveats / follow-ups:
  - **EP not yet supported** (the W4A8 kernel has no ep_rank masking; it expects topk_ids
    to index the *local* experts). ``moe_ep_size == 1`` (single GPU / pure TP) only for now.
  - The W4A8 kernel casts the activation straight to FP8 (no per-token-group input scale)
    and requantizes the SwiGLU intermediate per-token; DSv4's reference FP8 path uses
    per-token-group-128. Validate accuracy on a real checkpoint; a SmoothQuant / per-token-
    group input scale is a follow-up.
  - The packed-FP4 nibble convention is assumed to match the W4A8 kernel's (low nibble =
    even K). If a real DSv4 MXFP4 checkpoint differs, repack in
    ``process_weights_after_loading`` (numerically obvious in the first eval run).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.utils import is_flashinfer_available, log_info_on_rank0

if is_flashinfer_available():
    try:
        from flashinfer.fused_moe import (
            interleave_w4a8_fc1_gate_up,
            w4a8_mxfp4_moe,
        )

        _FI_HAS_W4A8_MXFP4 = True
    except ImportError:
        interleave_w4a8_fc1_gate_up = None
        w4a8_mxfp4_moe = None
        _FI_HAS_W4A8_MXFP4 = False
else:
    _FI_HAS_W4A8_MXFP4 = False

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

# MXFP4 group/block size (E8M0 scale per 32 fp4 weights).
_GROUP_SIZE = 32


class Mxfp4FlashinferW4A8MoEMethod:
    # Model-wide dequant exponent bias, chosen by the FIRST MoE layer's scale range and
    # shared by all layers (one value => one compiled kernel config for the whole
    # model). Why it exists: the W4A8 kernel dequantizes FP4*2^(scale-127) into FP8
    # e4m3 via exponent addition, which mis-encodes/underflows once products drop below
    # e4m3's normal floor (2^-6) -- and real DSv4 scales (~2^-8..2^-4) push ~25% of
    # weights there. The fix lifts the scales by +bias at load time and the kernel
    # multiplies the FP32 accumulator by 2^-bias in the epilogue (mathematically exact).
    _global_dequant_bias: Optional[int] = None

    """DeepSeek-V4 W4A8 MXFP4 MoE via FlashInfer's SM90 CuTe-DSL grouped GEMM.

    Mirrors :class:`Mxfp4FlashinferCutlassMoEMethod` but: (a) the fc1 weight/scale are
    laid out gate/up *interleaved* (row 2j = gate_j, 2j+1 = up_j) -- what the W4A8 fused
    SwiGLU epilogue needs -- rather than the cutlass mixed-gemm byte interleave; (b)
    ``apply`` calls the stateless ``w4a8_mxfp4_moe`` directly (serving-safe: the grouped
    GEMM compiles once per config and takes the routing-dependent sizes at runtime) and
    returns a ``StandardCombineInput`` without going through ``MoeRunner`` (whose
    fused-func key is owned by the W4A16 cutlass path)."""

    def __init__(self, fp8_method, prefix: str):
        if not _FI_HAS_W4A8_MXFP4:
            raise RuntimeError(
                "Mxfp4FlashinferW4A8MoEMethod requires a flashinfer build exporting "
                "flashinfer.fused_moe.w4a8_mxfp4_moe / interleave_w4a8_fc1_gate_up "
                "(the SM90 W4A8 MXFP4 MoE). Upgrade flashinfer-python, or drop "
                "--flashinfer-mxfp4-moe-activation fp8 to use the W4A16 cutlass path."
            )
        self._fp8 = fp8_method
        self.prefix = prefix
        self._swiglu_alpha: Optional[float] = None
        self._swiglu_beta: Optional[float] = None
        self._swiglu_limit: Optional[float] = None
        self._dequant_exp_bias: int = 0  # set in process_weights_after_loading

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
        # The W4A8 grouped GEMM needs the FP4 contraction dim K a multiple of 32 (E8M0
        # block) with the packed/contiguous dim 16-byte aligned (=> K % 32 == 0). DSv4
        # (hidden=4096, inter=2048) satisfies this for any sane TP; assert rather than pad.
        if hidden_size % 32 != 0 or intermediate_size_per_partition % 32 != 0:
            raise ValueError(
                "Mxfp4FlashinferW4A8MoEMethod requires hidden_size and "
                "intermediate_size_per_partition to be multiples of 32 "
                f"(got hidden={hidden_size}, "
                f"intermediate={intermediate_size_per_partition})."
            )
        # Same raw fp4 buffers the fp8 base allocates (uint8 4-bit packed weights, fp32
        # E8M0 scales) as the W4A16 path -- only the load-time repack differs.
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            **extra_weight_attrs,
        )

    def create_moe_runner(self, layer: Module, moe_runner_config) -> None:
        # EP is not yet supported (the W4A8 kernel has no ep_rank expert masking; it
        # treats topk_ids as local-expert indices). Single GPU / pure TP only for now.
        if getattr(layer, "moe_ep_size", 1) > 1:
            raise NotImplementedError(
                "Mxfp4FlashinferW4A8MoEMethod does not support expert parallelism "
                f"(moe_ep_size={layer.moe_ep_size}) yet; use ep_size=1 (pure TP) or "
                "select --flashinfer-mxfp4-moe-activation bf16 (W4A16)."
            )
        self.moe_runner_config = moe_runner_config

        # DSv4 uses standard SwiGLU plus a config-driven activation clamp. Pass all
        # three scalars (alpha=1, beta=0, limit) as plain Python floats: the W4A8
        # fused SwiGLU epilogue bakes each uniform value as a compile-time constant
        # anyway, and the float form lets w4a8_mxfp4_moe skip its per-call
        # uniformity check + .item() on tensors -- two device->host syncs per
        # parameter per MoE call, the dominant host stall of its otherwise
        # sync-free forward. None limit => plain silu(gate)*up.
        swiglu_limit = getattr(moe_runner_config, "swiglu_limit", None)
        if swiglu_limit is not None:
            self._swiglu_alpha = 1.0
            self._swiglu_beta = 0.0
            self._swiglu_limit = float(swiglu_limit)
        else:
            self._swiglu_alpha = None
            self._swiglu_beta = None
            self._swiglu_limit = None

    def process_weights_after_loading(self, layer: Module) -> None:
        # Run the fp8 base hook first (ROCm normalization, mxfp8 requant, ...). After it,
        # w13_weight is packed FP4 [E, 2I, H/2] in [w1; w3] = [gate; up] STACKED order
        # (the HF/FP8 loader emits [w1; w3]) and the scales are fp32 (= 2**e).
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_w4a8_mxfp4_weights_built", False):
            return

        log_info_on_rank0(
            logger,
            f"Preparing DSv4 MXFP4 experts for FlashInfer SM90 W4A8 "
            f"(layer: {self.prefix})...",
        )

        # Scales: fp32 (= 2**e) -> raw UE8M0 byte (exponent + 127 bias), which the W4A8
        # kernel reads directly.
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
        w13_u8 = layer.w13_weight.data.view(torch.uint8).contiguous()  # [E, 2I, H/2]
        w2_u8 = layer.w2_weight.data.view(torch.uint8).contiguous()  # [E, H, I/2]

        # Dequant exponent re-centering (see _global_dequant_bias): lift the UE8M0
        # scales so every FP4*scale product encodes in FP8 e4m3's NORMAL range. The
        # feasible bias window for exactness is [122 - s_min, 133 - s_max] (lower bound:
        # smallest nonzero product 0.5*2^(s_min+B-127) >= 2^-6; upper: largest product
        # 6*2^(s_max+B-127) <= 448). One model-wide value (first layer's midpoint, env
        # SGLANG_W4A8_DEQUANT_EXP_BIAS overrides) keeps a single compiled kernel config.
        s_min = min(int(w13_scale_u8.min()), int(w2_scale_u8.min()))
        s_max = max(int(w13_scale_u8.max()), int(w2_scale_u8.max()))
        b_low, b_high = max(0, 122 - s_min), 133 - s_max
        if Mxfp4FlashinferW4A8MoEMethod._global_dequant_bias is None:
            env_bias = os.environ.get("SGLANG_W4A8_DEQUANT_EXP_BIAS")
            if env_bias is not None:
                bias = int(env_bias)
            elif b_low <= b_high:
                bias = (b_low + b_high) // 2
            else:
                bias = max(0, b_high)
            Mxfp4FlashinferW4A8MoEMethod._global_dequant_bias = bias
            log_info_on_rank0(
                logger,
                f"W4A8 dequant exponent bias = {bias} (layer scale range "
                f"[{s_min}, {s_max}], feasible [{b_low}, {b_high}])",
            )
        bias = Mxfp4FlashinferW4A8MoEMethod._global_dequant_bias
        if not (b_low <= bias <= b_high):
            logger.warning(
                "W4A8 dequant bias %d outside layer %s feasible range [%d, %d] "
                "(scales [%d, %d]); dequant may underflow/saturate on this layer.",
                bias,
                self.prefix,
                b_low,
                b_high,
                s_min,
                s_max,
            )
        self._dequant_exp_bias = bias
        w13_scale_u8 = w13_scale_u8 + bias
        w2_scale_u8 = w2_scale_u8 + bias

        # fc1: the W4A8 fused SwiGLU needs gate/up *interleaved* (row 2j = gate_j,
        # 2j+1 = up_j), built from the stacked [gate; up] = [0:I]; [I:2I]. The raw
        # [w1; w3] is already [gate; up] (no [w3; w1] reorder, unlike the cutlass path).
        # interleave_w4a8_fc1_gate_up applies the same row permutation to weight + scale.
        w13_il, w13_s_il = interleave_w4a8_fc1_gate_up(w13_u8, w13_scale_u8)
        # fc2 (down) is used as-is: packed MXFP4 [E, H, I/2] + raw E8M0 [E, H, I/32].

        layer.w13_weight = Parameter(w13_il, requires_grad=False)
        layer.w2_weight = Parameter(w2_u8, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_s_il, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_scale_u8, requires_grad=False)

        layer._w4a8_mxfp4_weights_built = True
        layer._dsv4_mxfp4_backend = "flashinfer_w4a8_sm90"
        torch.cuda.empty_cache()

    # --- Forward -----------------------------------------------------------

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

        topk_output = dispatch_output.topk_output
        # The W4A8 kernel needs explicit topk_ids / topk_weights (it does no internal
        # routing). Materialize if a bypassed-format topk slipped through.
        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = topk_output.to_standard()
        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        x = dispatch_output.hidden_states
        output_dtype = torch.bfloat16
        # Match the W4A16 FlashInfer MXFP4 path and keep this output allocation
        # consistent with the symmetric allocator policy used around MoE
        # dispatch/combine and CP reduce-scatter paths.
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            out = torch.empty(
                x.shape[0], x.shape[-1], dtype=output_dtype, device=x.device
            )

        # The stateless flashinfer entry point is serving-safe: the grouped GEMM keeps
        # num_groups static (all experts are always passed, empty ones at M=0) and all
        # routing-dependent metadata -- per-group sizes/pointers and the persistent
        # scheduler's cluster totals -- is written and read on DEVICE, so the compiled
        # kernel is reused across arbitrary routings (one cute.compile per config, no
        # per-step recompile), no capacity padding, and the forward performs no
        # device->host sync.
        w4a8_mxfp4_moe(
            input=x,
            token_selected_experts=topk_ids.to(torch.int),
            token_final_scales=topk_weights,
            fc1_expert_weights=layer.w13_weight,
            fc2_expert_weights=layer.w2_weight,
            output_dtype=output_dtype,
            quant_scales=[
                layer.w13_weight_scale_inv,  # UE8M0 uint8 [E, 2I, H/32]
                layer.w2_weight_scale_inv,  # UE8M0 uint8 [E, H, I/32]
            ],
            swiglu_alpha=self._swiglu_alpha,
            swiglu_beta=self._swiglu_beta,
            swiglu_limit=self._swiglu_limit,
            dequant_exp_bias=self._dequant_exp_bias,
            output=out,
        )
        return StandardCombineInput(hidden_states=out)
