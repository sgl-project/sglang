"""
DeepSeek W4A16 MoE quantization method (SM90 / H200).

Wraps Fp8MoEMethod to reuse the FP4-expert weight creation/loading, then
overrides process_weights_after_loading to pre-interleave FP4 weights and
MXFP4 block scales for the SM90 mixed-input CUTLASS kernel exposed by
flashinfer-ai/flashinfer PR #3084, and overrides apply to call
cutlass_fused_moe(..., use_w4_group_scaling=True) directly.

This is the H200 counterpart to mxfp4_deepseek.py. The two share the same
DSv4 FP4 checkpoint (SGLANG_DSV4_MODE=2604 SGLANG_DSV4_FP4_EXPERTS=1): weight
shapes and dtypes are identical; only the post-load layout and the kernel
call differ.

Usage: --moe-runner-backend flashinfer_w4a16 --moe-a2a-backend none
       with SGLANG_DSV4_MODE=2604 SGLANG_DSV4_FP4_EXPERTS=1
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    is_flashinfer_available,
    log_info_on_rank0,
    set_weight_attrs,
)
from sglang.srt.utils.common import next_power_of_2

if is_flashinfer_available():
    from flashinfer.fused_moe import (
        cutlass_fused_moe,
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput


from sglang.srt.debug_utils.sunrise_debug_utils import sunrise_moe_code_path_checker
from sglang.srt.environ import envs


def _fp32_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Convert float32 → UE8M0 (float8_e8m0fnu) and assert lossless round-trip.

    UE8M0 stores only the 8-bit biased exponent (no mantissa), so only exact
    powers of 2 round-trip bit-exactly. DSv4 MXFP4 block scales should already
    be powers of 2 per the MXFP4 spec; if this assert fires, the checkpoint
    isn't actually MXFP4-quantized and we must feed the kernel the raw E8M0
    bytes instead of round-tripping through fp32.
    """
    assert x.dtype == torch.float32, f"expected float32 input, got {x.dtype}"
    ans = x.to(torch.float8_e8m0fnu)
    rt = ans.float()
    mismatch = rt != x
    if mismatch.any():
        bad_orig = x[mismatch][:5].tolist()
        bad_rt = rt[mismatch][:5].tolist()
        raise AssertionError(
            f"fp32→UE8M0 lossy: {int(mismatch.sum())}/{x.numel()} elements "
            f"changed; min/max input = {x.min().item()}/{x.max().item()}; "
            f"first 5 (orig → round-trip): {list(zip(bad_orig, bad_rt))}"
        )
    return ans


# MXFP4 4-bit codebook and dequant helper, copied verbatim (module-level
# constant `_MXFP4_LUT` + body of `_dequant_mxfp4_on_device`) from flashinfer
# PR #3084 branch at
#   flashinfer-sunrise/tests/moe/test_trtllm_cutlass_fused_moe.py
#   (commit 77746b81, lines 2419-2452)
# so the bf16 API path sees weights bit-equivalent to what the SM90
# mixed-input kernel dequants inside itself.
_MXFP4_LUT = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _dequant_mxfp4(
    w_fp4_u8: torch.Tensor, w_scale_ue8m0_u8: torch.Tensor
) -> torch.Tensor:
    """[E, rows, K/2] uint8 FP4 packed + [E, rows, K/32] uint8 UE8M0 → [E, rows, K] bf16."""
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=w_fp4_u8.device)
    lo = w_fp4_u8 & 0x0F
    hi = (w_fp4_u8 >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(*w_fp4_u8.shape[:-1], -1)
    values = lut[nib.long()]
    scale = torch.exp2(w_scale_ue8m0_u8.to(torch.float32) - 127.0)
    scale = scale.repeat_interleave(32, dim=-1)
    return (values * scale).to(torch.bfloat16)


class DeepSeekW4A16MoEMethod:
    """W4A16 MoE method for DeepSeek-family models with FP4 expert weights on SM90.

    Wraps Fp8MoEMethod for weight creation/loading, but overrides
    post-loading processing to pre-interleave FP4 weights and MXFP4 scales
    for the flashinfer SM90 mixed-input CUTLASS kernel, and directly calls
    cutlass_fused_moe(..., use_w4_group_scaling=True) in apply().
    """

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        # Kept for parity with mxfp4_deepseek; unused by cutlass_fused_moe.
        self.flashinfer_mxfp4_moe_precision = (
            get_global_server_args().flashinfer_mxfp4_moe_precision
        )

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config

        # Sanity check: v5 (260415) ckpt's HF config has swiglu_limit=10.0;
        # v4 (260409) does not. Same check as mxfp4_deepseek.
        swiglu_limit = moe_runner_config.swiglu_limit
        is_260415 = envs.SGLANG_DSV4_2604_SUBMODE.get() == "260415"
        assert is_260415 == (swiglu_limit is not None), (
            f"swiglu_limit must be non-None iff submode=260415 "
            f"(got submode={envs.SGLANG_DSV4_2604_SUBMODE.get()!r}, "
            f"swiglu_limit={swiglu_limit!r})"
        )
        self._swiglu_limit_tensor = (
            torch.full(
                (layer.num_local_experts,),
                swiglu_limit,
                dtype=torch.float32,
                device=layer.w13_weight.device,
            )
            if swiglu_limit is not None
            else None
        )

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        """Create FP4-packed weights with TP-aware shapes.

        Shapes and dtypes are identical to mxfp4_deepseek (same checkpoint);
        the only difference is the post-load layout produced by
        process_weights_after_loading.
        """
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        fp4_block_k = 32

        # FP4 packed weights: 2 values per byte -> physical K = logical K / 2
        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = Parameter(
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

        # Block scales: one float32 scale per fp4_block_k FP4 elements along K
        w13_weight_scale = Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float32,
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
        from sglang.srt.layers.quantization.utils import reorder_w1w3_to_w3w1

        # Let Fp8MoEMethod do its processing first (FP4 view conversion, etc.).
        # When SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1 is set, that path builds the
        # mega-MoE weight tuples on the layer; we must then skip the
        # reorder/interleave below since mega wants the checkpoint's
        # [gate(w1), up(w3)] row order.
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        # Reorder w13 from checkpoint [w1(gate), w3(up)] to kernel [w3(up), w1(gate)].
        # flashinfer's SM90 W4A16 test (`test_moe_bf16_mxfp4`) computes its
        # reference as `w3, w1 = torch.chunk(w31, 2, dim=0)` — i.e. w3 (up) is
        # the first half along dim -2, w1 (gate) is the second. Same row order
        # as the B200 TRT-LLM routed kernel.
        w13_w, w13_s = reorder_w1w3_to_w3w1(
            layer.w13_weight.data, layer.w13_weight_scale_inv.data
        )
        layer.w13_weight = Parameter(w13_w, requires_grad=False)
        layer.w13_weight_scale_inv = Parameter(w13_s, requires_grad=False)

        log_info_on_rank0(
            logger,
            f"Interleaving FP4 expert weights/scales for SM90 W4A16 kernel "
            f"(layer: {self.prefix})...",
        )

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data

        # Convert float32 block scales to UE8M0 (1 byte per element) before the
        # layout interleave. The SM90 kernel reads E8M0 uint8 bytes.
        if w13_scale.dtype == torch.float32:
            w13_scale = _fp32_to_ue8m0(w13_scale)
            w2_scale = _fp32_to_ue8m0(w2_scale)

        # bf16-weight debug path: dequant FP4+UE8M0 → bf16 once. Two downstream
        # consumers share this:
        #   - BF16_API: apply() calls plain bf16 cutlass_fused_moe (skips the
        #     SM90 mixed-input kernel and the SM90 interleave).
        #   - TORCH_REF: apply() calls a pure-torch MoE forward (skips the
        #     flashinfer bf16 grouped GEMM too).
        # Both are independent numerical references for W4A16 acc drops.
        use_bf16_api = envs.SGLANG_HACK_DEBUG_W4A16_USE_BF16_API.get()
        use_torch_ref = envs.SGLANG_HACK_DEBUG_W4A16_USE_TORCH_REF.get()
        assert not (use_bf16_api and use_torch_ref), (
            "SGLANG_HACK_DEBUG_W4A16_USE_BF16_API and "
            "SGLANG_HACK_DEBUG_W4A16_USE_TORCH_REF are mutually exclusive"
        )
        if use_bf16_api or use_torch_ref:
            consumer = "bf16-API" if use_bf16_api else "torch-ref"
            log_info_on_rank0(
                logger,
                f"Dequant FP4 → bf16 for {consumer} path (layer: {self.prefix})...",
            )
            w13_bf16 = _dequant_mxfp4(
                w13.contiguous().view(torch.uint8),
                w13_scale.contiguous().view(torch.uint8),
            )
            w2_bf16 = _dequant_mxfp4(
                w2.contiguous().view(torch.uint8),
                w2_scale.contiguous().view(torch.uint8),
            )
            layer.w13_weight = Parameter(w13_bf16, requires_grad=False)
            layer.w2_weight = Parameter(w2_bf16, requires_grad=False)
            # Drop scale parameters — bf16 path does not read them. Replace
            # with zero-size placeholders to keep any attribute-existence
            # checks happy.
            layer.w13_weight_scale_inv = Parameter(
                torch.empty(0, device=w13_bf16.device), requires_grad=False
            )
            layer.w2_weight_scale_inv = Parameter(
                torch.empty(0, device=w2_bf16.device), requires_grad=False
            )
            torch.cuda.empty_cache()
            return

        # Pre-interleave MXFP4 weights and scales (runs once at load time).
        # Shapes after interleave:
        #   weights: same as input (byte-permutation only).
        #   scales:  [E, rows, K/32]  ->  [E, K/(32*4), rows*4]   uint8.
        w13_u8 = w13.contiguous().view(torch.uint8)
        w2_u8 = w2.contiguous().view(torch.uint8)
        w13_scale_u8 = w13_scale.contiguous().view(torch.uint8)
        w2_scale_u8 = w2_scale.contiguous().view(torch.uint8)

        w13_il = interleave_moe_weights_for_sm90_mixed_gemm(w13_u8, "fp4")
        w2_il = interleave_moe_weights_for_sm90_mixed_gemm(w2_u8, "fp4")
        w13_scale_il = interleave_moe_scales_for_sm90_mixed_gemm(
            w13_scale_u8, group_size=32
        )
        w2_scale_il = interleave_moe_scales_for_sm90_mixed_gemm(
            w2_scale_u8, group_size=32
        )

        layer.w13_weight = Parameter(w13_il, requires_grad=False)
        layer.w2_weight = Parameter(w2_il, requires_grad=False)
        # Keep interleaved scales as uint8 — .view(torch.int32) at apply-time.
        layer.w13_weight_scale_inv = Parameter(w13_scale_il, requires_grad=False)
        layer.w2_weight_scale_inv = Parameter(w2_scale_il, requires_grad=False)

        torch.cuda.empty_cache()

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        # --- Step A: Prepare weights and sizes ---
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        use_bf16_api = envs.SGLANG_HACK_DEBUG_W4A16_USE_BF16_API.get()
        use_torch_ref = envs.SGLANG_HACK_DEBUG_W4A16_USE_TORCH_REF.get()
        if use_bf16_api or use_torch_ref:
            # bf16 weights path: weights already dequanted to bf16 in
            # process_weights_after_loading; no scale tensors to pass.
            quant_scales_arg = None
        else:
            quant_scales_arg = [
                layer.w13_weight_scale_inv.view(torch.int32),
                layer.w2_weight_scale_inv.view(torch.int32),
            ]

        # w13/w2 are pre-interleaved uint8 (W4A16) or plain bf16 (bf16-API);
        # logical shapes come from the layer-configured sizes rather than
        # tensor dims (interleave preserves numel but the 3D view no longer
        # maps 1:1 to [E, 2*I, H/2]).
        hidden_size = layer.hidden_size

        # --- Step B: Determine routing ---
        if TopKOutputChecker.format_is_standard(topk_output):
            topk_ids = topk_output.topk_ids
            topk_weights = topk_output.topk_weights
        else:
            raise ValueError(
                f"Unsupported topk output format for W4A16 MoE: {topk_output.format}"
            )

        # Undo StandardDispatcher's global->local+sentinel mapping so the
        # flashinfer kernel (which expects global expert ids plus ep_rank/ep_size
        # for local filtering) gets what it wants. Mirror the mxfp4_deepseek
        # logic gated on SGLANG_OPT_MXFP4_SKIP_DISPATCHER_MAPPING.
        if not envs.SGLANG_OPT_MXFP4_SKIP_DISPATCHER_MAPPING.get():
            local_expert_offset = layer.moe_ep_rank * layer.num_local_experts
            topk_ids = torch.where(
                topk_ids >= 0,
                topk_ids + local_expert_offset,
                topk_ids,
            )

        # --- Step C: Activations ---
        # W4A16 path: bf16 activations, no quantization needed.
        assert hidden_states.dtype == torch.bfloat16, (
            f"W4A16 expects bf16 activations, got {hidden_states.dtype}"
        )
        x = hidden_states
        origin_dim = x.shape[-1]
        if hidden_size != origin_dim:
            x = torch.nn.functional.pad(
                x, (0, hidden_size - origin_dim), mode="constant", value=0.0
            )

        # --- Step D: Allocate output with symmetric memory for TP all-reduce ---
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            num_tokens = x.shape[0]
            symm_output = torch.empty(
                num_tokens, hidden_size, dtype=torch.bfloat16, device=x.device
            )

        # --- Step E: Call kernel ---
        # DSv4 260415 ships a per-MoE-layer sanity counter that deepseek_v4.py
        # asserts is bumped exactly once per forward (see deepseek_v4.py:2014).
        # Mirror the mxfp4_deepseek bump so the checker is satisfied.
        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "260415" and (
            self._swiglu_limit_tensor is not None
        ):
            sunrise_moe_code_path_checker.observed += 1

        swiglu_limit_arg = (
            None
            if envs.SGLANG_HACK_DEBUG_W4A16_REMOVE_SWIGLU_LIMIT.get()
            else self._swiglu_limit_tensor
        )

        _moe_fn = cutlass_fused_moe
        if use_torch_ref:
            from sglang.srt.debug_utils.w4a16_moe_ref_related import (
                torch_ref_cutlass_fused_moe as _moe_fn,
            )

        _moe_fn(
            input=x,
            token_selected_experts=topk_ids.to(torch.int32).contiguous(),
            token_final_scales=topk_weights.to(torch.float32).contiguous(),
            fc1_expert_weights=w13,
            fc2_expert_weights=w2,
            output_dtype=torch.bfloat16,
            quant_scales=quant_scales_arg,
            swiglu_limit=swiglu_limit_arg,
            ep_size=layer.moe_ep_size,
            ep_rank=layer.moe_ep_rank,
            tp_size=1,
            tp_rank=0,
            use_w4_group_scaling=not use_bf16_api,
            tune_max_num_tokens=next_power_of_2(x.shape[0]),
            output=symm_output,
        )
        output = symm_output

        # Apply routed_scaling_factor (DSv4 = 1.5). cutlass_fused_moe has no
        # routed_scaling_factor parameter, so unless we hand this to the fused
        # shared-add fast path, we multiply post-hoc. See mxfp4_deepseek for
        # the same rationale.
        if not envs.SGLANG_OPT_MXFP4_FUSE_RSF_SHARED_ADD.get():
            rsf = layer.moe_runner_config.routed_scaling_factor
            if rsf is not None and rsf != 1.0:
                output.mul_(rsf)

        return StandardCombineInput(hidden_states=output)
