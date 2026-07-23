from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.kernels.ops.model.inkling.inkling_gate_topk_renorm import (
    ensure_gate_gemv_fused_scratch,
    inkling_gate_gemv,
    inkling_gate_gemv_fused,
)
from sglang.srt.configs.inkling import InklingModelConfig
from sglang.srt.distributed import (
    get_tensor_model_parallel_group,
)
from sglang.srt.environ import GateGemvMode, envs
from sglang.srt.layers.moe import get_moe_runner_backend
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.triton_utils.gate_topk import gate_topk
from sglang.srt.layers.moe.moe_runner.triton_utils.inkling_moe import (
    FUSED_PREPROCESS_WIN_TOKENS,
    compute_grouped_gemm_metadata,
    fused_moe_preprocess,
    get_src2dst,
    grouped_gemm_triton,
    post_reorder,
    pre_reorder,
    select_grouped_gemm_block_m,
    silu_and_mul_helion,
)
from sglang.srt.layers.moe.moe_runner.triton_utils.sigmoid_gate_topk_renorm import (
    sigmoid_gate_topk_renorm,
)
from sglang.srt.layers.moe.topk import PackedTopKOutput, StandardTopKOutput
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp4_utils import get_fp4_gemm_runner_backend
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.models.inkling_common.dense_mlp import (
    InklingBatchDenseMLP,
    InklingDenseMLP,
)
from sglang.srt.models.inkling_common.kernels.comm import (
    get_ar_buffer,
    reduce_scatter_hidden,
    stash_ar_shared,
    symm_mem_all_reduce,
)
from sglang.srt.models.inkling_common.util import (
    bf16_routed_uses_stock_fused_moe,
    lora_compatible_layout_enabled,
    use_inkling_shared_fused_moe,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer
from sglang.srt.utils import add_prefix, is_cuda, is_hip

_FP32_GEMM_UPCAST = is_hip()


def _mm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _FP32_GEMM_UPCAST:
        return torch.mm(a.float(), b.float())
    return torch.mm(a, b, out_dtype=torch.float32)


def _addmm_fp32(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _FP32_GEMM_UPCAST:
        return torch.addmm(bias.float(), a.float(), b.float())
    return torch.addmm(bias, a, b, out_dtype=torch.float32)


_INKLING_FUSED_GATE_OUT_FEATURES = 258
# Pad the expert dimension for aligned cuBLAS access. The loader initializes
# the padding once, avoiding a per-forward copy.
_INKLING_FUSED_GATE_OUT_FEATURES_PADDED = (
    (_INKLING_FUSED_GATE_OUT_FEATURES + 7) // 8 * 8
)

# Use the specialized GEMV kernel only for small batches at this hidden size.
_INKLING_GATE_GEMV_HIDDEN = 6144
_GATE_GEMV_MAX_TOKENS = 4


def _load_gate_weight_padded(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Copy the checkpoint rows into the pre-padded gate weight; zero the tail.

    Reached only when shapes differ (param padded, checkpoint not); see
    InklingForConditionalGeneration._load_regular_param.
    """
    n = loaded_weight.shape[0]
    param.data[:n].copy_(loaded_weight)
    param.data[n:].zero_()


def inkling_fused_gate_linear_with_fp32_out(
    input: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    assert weight.ndim == 2, f"{weight.shape=}"
    assert weight.size(0) == _INKLING_FUSED_GATE_OUT_FEATURES_PADDED, f"{weight.shape=}"
    assert input.ndim == 2, f"{input.shape=}"
    return _mm_fp32(input, weight.T)[:, :_INKLING_FUSED_GATE_OUT_FEATURES]


def linear_with_fp32_out(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None
) -> torch.Tensor:
    leading_dims = list(input.shape[:-1])
    flat_input = input.flatten(0, -2)

    if bias is None:
        out = _mm_fp32(flat_input, weight.T)
    else:
        out = _addmm_fp32(bias, flat_input, weight.T)

    out = out.view(*leading_dims, weight.shape[0])
    return out


def linear_with_pad(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None):
    assert w.ndim == 2, f"{w.shape=}"
    w_size0 = w.size(0)
    pad_size = 8 - (w_size0 % 8) if w_size0 % 8 != 0 else 0
    if pad_size > 0:
        w = torch.cat([w, w.new_zeros((pad_size, w.size(1)))], dim=0)
    y = linear_with_fp32_out(x, w, bias)
    if pad_size > 0:
        y = y[..., :-pad_size]
    return y


def _logsigmoid_normalize(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.logsigmoid(logits)
    return torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True))


def _renorm_topk_logits(
    logits_TG: torch.Tensor,
    topk_indices_TK: torch.Tensor,
    n_shared_experts: int,
    gate_activation: str,
) -> torch.Tensor:
    routed_logits = (
        logits_TG[..., :-n_shared_experts] if n_shared_experts > 0 else logits_TG
    )
    # gate_topk now emits int32 ids (for the SRT MoE path); torch.gather requires
    # int64, so widen here on this (CPU/non-sigmoid) fallback path.
    topk_logits = routed_logits.gather(-1, topk_indices_TK.long())
    if n_shared_experts > 0:
        shared_logits = logits_TG[..., -n_shared_experts:]
        topk_logits = torch.cat([topk_logits, shared_logits], dim=-1)
    if gate_activation == "sigmoid":
        return _logsigmoid_normalize(topk_logits)
    return topk_logits.softmax(dim=-1, dtype=torch.float32)


@triton.jit
def _inkling_compute_logsigmoid_norm(logits, mask_a):
    abs_logits = tl.abs(logits)
    min_logits = tl.minimum(logits, 0.0)
    log_probs = min_logits - libdevice.log1p(libdevice.exp(-abs_logits))

    max_log_probs = tl.max(log_probs, axis=1)[:, None]
    exp_shifted = libdevice.exp(log_probs - max_log_probs)
    sum_exp = tl.sum(
        tl.where(mask_a[None, :], exp_shifted, 0.0), axis=1, keep_dims=True
    )
    logsumexp = max_log_probs + libdevice.log(sum_exp)
    return libdevice.exp(log_probs - logsumexp)


@triton.jit(do_not_specialize=["T", "route_scale"])
def _renorm_topk_logits_fwd_kernel(
    logits_ptr,
    indices_ptr,
    routed_weights_ptr,
    shared_weights_ptr,
    global_scale_ptr,
    route_scale,
    T,
    G: tl.constexpr,
    stride_logits_0,
    K: tl.constexpr,
    S: tl.constexpr,
    A_POW2: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    A: tl.constexpr = K + S
    pid = tl.program_id(0).to(tl.int64)

    offs_t = pid * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    mask_t = offs_t < T
    offs_a = tl.arange(0, A_POW2)
    mask_a = offs_a < A

    mask_k = mask_t[:, None] & (offs_a < K)[None, :]
    indices = tl.load(
        indices_ptr + offs_t[:, None] * K + offs_a[None, :], mask=mask_k, other=0
    )
    routed_logits = tl.load(
        logits_ptr + offs_t[:, None] * stride_logits_0 + indices,
        mask=mask_k,
        other=float("-inf"),
    ).to(tl.float32)

    if S > 0:
        offs_s = offs_a - K
        mask_s = mask_t[:, None] & (offs_s[None, :] >= 0) & (offs_s[None, :] < S)
        shared_logits = tl.load(
            logits_ptr + offs_t[:, None] * stride_logits_0 + (G - S) + offs_s[None, :],
            mask=mask_s,
            other=float("-inf"),
        ).to(tl.float32)
        active_logits = tl.where((offs_a < K)[None, :], routed_logits, shared_logits)
    else:
        active_logits = routed_logits

    weights = _inkling_compute_logsigmoid_norm(active_logits, mask_a)
    weights *= route_scale
    weights *= tl.load(global_scale_ptr).to(weights.dtype)

    offs_tk = offs_t[:, None] * K + offs_a[None, :]
    mask_tk = mask_t[:, None] & (offs_a < K)[None, :]
    tl.store(routed_weights_ptr + offs_tk, weights, mask=mask_tk)

    if S > 0:
        offs_s = offs_a - K
        mask_ts = mask_t[:, None] & (offs_s[None, :] >= 0) & (offs_s[None, :] < S)
        offs_ts = offs_t[:, None] * S + offs_s[None, :]
        tl.store(shared_weights_ptr + offs_ts, weights, mask=mask_ts)


def renorm_topk_logits_scaled(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    n_shared_experts: int,
    route_scale: float,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not logits.is_cuda:
        weights = _renorm_topk_logits(logits, topk_indices, n_shared_experts, "sigmoid")
        weights = weights * route_scale
        weights = weights * global_scale
        topk = topk_indices.shape[-1]
        shared = weights[..., topk:].contiguous() if n_shared_experts > 0 else None
        return weights[..., :topk].contiguous(), shared

    logits = logits.contiguous()
    topk_indices = topk_indices.contiguous()
    tokens, gate_experts = logits.shape
    topk = topk_indices.shape[-1]
    active = topk + n_shared_experts
    active_pow2 = triton.next_power_of_2(active)
    block_size_t = max(1, 1024 // active_pow2)
    routed_weights = torch.empty(
        (tokens, topk), dtype=logits.dtype, device=logits.device
    )
    shared_weights = (
        torch.empty(
            (tokens, n_shared_experts), dtype=logits.dtype, device=logits.device
        )
        if n_shared_experts > 0
        else None
    )
    _renorm_topk_logits_fwd_kernel[(triton.cdiv(tokens, block_size_t),)](
        logits,
        topk_indices,
        routed_weights,
        shared_weights,
        global_scale,
        route_scale,
        tokens,
        gate_experts,
        logits.stride(0),
        topk,
        n_shared_experts,
        active_pow2,
        block_size_t,
    )
    return routed_weights, shared_weights


class InklingGate(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_routed_experts: int,
        n_shared_experts: int,
        experts_per_token: int,
        route_scale: float,
        layer_id: int,
        prefix: str = "",
        norm_after_topk: bool = True,
        use_global_scale: bool = False,
        use_gate_bias: bool = False,
        gate_activation: Literal["sigmoid", "softmax"] = "sigmoid",
        shared_expert_sink: bool = False,
    ):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.n_total_experts = n_routed_experts + n_shared_experts
        self.topk = experts_per_token
        self.layer_id = layer_id
        self.prefix = prefix
        self.norm_after_topk = norm_after_topk
        self.gate_activation = gate_activation
        self.shared_expert_sink = shared_expert_sink
        # The fused gate emits pre-packed topk only when the routed experts consume it
        # (the SRT MoeRunner apply path). The unquantized forward_moe path needs standard
        # topk tensors, so the owning InklingMoE flips this off for unquantized experts.
        self.emit_packed_topk = True

        if use_global_scale:
            self.global_scale = nn.Parameter(
                torch.empty(1, dtype=torch.float32), requires_grad=False
            )
        else:
            self.global_scale = None
        self.route_scale = route_scale
        # Rows pre-padded to a multiple of 8 (see _INKLING_FUSED_GATE_OUT_FEATURES_PADDED);
        # the loader fills the real rows.
        padded_experts = (self.n_total_experts + 7) // 8 * 8
        self.weight = nn.Parameter(
            torch.zeros(padded_experts, d_model), requires_grad=False
        )
        self.weight.weight_loader = _load_gate_weight_padded
        if use_gate_bias:
            self.bias = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.bias = None
        # The FUSED gate mode uses persistent scratch buffers; they must exist
        # before CUDA graph capture (with --skip-server-warmup nothing runs
        # eagerly first, so allocating lazily would land in the capture pool).
        if (
            envs.SGLANG_OPT_GATE_GEMV_MODE.get() >= GateGemvMode.FUSED
            and torch.cuda.is_available()
            and torch.version.hip is None
        ):
            ensure_gate_gemv_fused_scratch(torch.device("cuda"))

    def forward_fused(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # Small decode batches use the specialized GEMV path. FUSED also runs
        # the gate epilogue in the same launch.
        # sigmoid_gate_topk_renorm sees the identical [tokens, 264]-padded slice
        # either way, so the top-k path below is oblivious to the choice.
        gemv_mode = envs.SGLANG_OPT_GATE_GEMV_MODE.get()
        if (
            gemv_mode != GateGemvMode.OFF
            and x.shape[0] <= _GATE_GEMV_MAX_TOKENS
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and x.shape[-1] == _INKLING_GATE_GEMV_HIDDEN
            and torch.version.hip is None
        ):
            if gemv_mode >= GateGemvMode.FUSED:
                # Single launch: outputs are bitwise-identical to the pair
                # (shared GEMV + epilogue code paths, asserted in tests).
                return inkling_gate_gemv_fused(
                    x,
                    self.weight,
                    self.bias,
                    self.global_scale,
                    self.route_scale,
                    return_packed=self.emit_packed_topk,
                    enable_pdl=is_arch_support_pdl(),
                )
            logits = inkling_gate_gemv(x, self.weight, enable_pdl=is_arch_support_pdl())
        else:
            logits = inkling_fused_gate_linear_with_fp32_out(x, self.weight)
        # Fused sigmoid[+bias] select-top-k + logsigmoid-renorm in one launch.
        # Pre-packed topk is consumed only by the SRT MoeRunner apply path (quantized
        # experts). Unquantized experts use forward_moe, which needs standard topk
        # tensors; packed mode returns None for routed_weights/topk_indices and would
        # crash there. InklingMoE sets emit_packed_topk from the experts' quant method.
        return_packed_topk = self.emit_packed_topk
        gate_output = sigmoid_gate_topk_renorm(
            logits,
            self.topk,
            self.n_shared_experts,
            self.route_scale,
            self.global_scale,
            self.bias,
            return_packed_topk=return_packed_topk,
        )
        routed_weights, topk_indices, shared_gammas, packed_topk_ids = gate_output
        return routed_weights, topk_indices, shared_gammas, packed_topk_ids

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        # R3 (rollout routing replay) needs the plain [T, K] topk indices captured on
        # the standard path; the fused kernel's packed output never exposes them, so
        # bypass the fused shortcut whenever an experts capturer is active.
        if (
            get_global_experts_capturer() is None
            and envs.SGLANG_OPT_USE_FUSED_GATE_TOPK.get()
            and self.n_total_experts == _INKLING_FUSED_GATE_OUT_FEATURES
            and self.gate_activation == "sigmoid"
            and self.norm_after_topk
            and self.global_scale is not None
            and self.bias is not None
        ):
            return self.forward_fused(x)

        # self.weight is pre-padded; pass the real rows so scores stay [tokens, experts].
        scores = linear_with_pad(x, self.weight[: self.n_total_experts], None)
        assert scores.ndim == 2, f"{scores.shape=} should be TE"

        logits = scores
        if self.gate_activation == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        routed_scores = (
            scores[..., : -self.n_shared_experts]
            if self.n_shared_experts > 0
            else scores
        )
        bias_for_topk = self.bias
        routed_scores_for_topk = (
            routed_scores + bias_for_topk
            if bias_for_topk is not None
            else routed_scores
        )
        _, topk_indices = gate_topk(routed_scores_for_topk, self.topk)
        # R3 (rollout routing replay): feed the routed-expert selection to sglang's global
        # experts capturer so --use-rollout-routing-replay can ship it back to the trainer,
        # which replays the exact same routing during training. Inkling's gate uses its own
        # gate_topk (not srt/layers/moe/topk.py), so the standard capture call is re-added here.
        # NOTE: this inlines cap.capture() rather than going through topk.py's
        # capture_routed_experts_if_allowed, so disable_routed_experts_capture_for_draft
        # (which only rewires TopK modules) cannot opt a InklingGate out. Moot while Inkling MTP
        # draft blocks are forced dense; give the gate an allow_capture flag if a draft
        # ever carries MoE.
        if (cap := get_global_experts_capturer()) is not None:
            cap.capture(layer_id=self.layer_id, topk_indices=topk_indices)
        if self.norm_after_topk:
            if self.gate_activation == "sigmoid" and self.global_scale is not None:
                routed_weights, shared_gammas = renorm_topk_logits_scaled(
                    logits,
                    topk_indices,
                    self.n_shared_experts,
                    self.route_scale,
                    self.global_scale,
                )
            else:
                routed_weights = _renorm_topk_logits(
                    logits, topk_indices, self.n_shared_experts, self.gate_activation
                )
                if self.global_scale is not None:
                    routed_weights = (
                        routed_weights * self.route_scale * self.global_scale
                    )
                else:
                    routed_weights = routed_weights * self.route_scale
                if self.shared_expert_sink and self.n_shared_experts > 0:
                    shared_gammas = routed_weights[
                        ..., -self.n_shared_experts :
                    ].contiguous()
                    routed_weights = routed_weights[..., : self.topk].contiguous()
                else:
                    shared_gammas = None
        else:
            # int32 topk ids (from gate_topk) -> int64 for torch.gather.
            routed_weights = routed_scores.gather(dim=-1, index=topk_indices.long())
            if self.global_scale is not None:
                routed_weights = routed_weights * self.route_scale * self.global_scale
            else:
                routed_weights = routed_weights * self.route_scale
            shared_gammas = None

        return routed_weights, topk_indices, shared_gammas, None


def make_forward_inputs_2d(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w2_weight: torch.Tensor,
):
    *outer, hidden_size = hidden_states.shape
    *_outer, top_k = topk_ids.shape
    *__outer, _top_k = topk_weights.shape
    assert _outer == __outer, f"{topk_ids.shape=} {topk_weights.shape=}"
    assert top_k == _top_k, f"{topk_ids.shape=} {topk_weights.shape=}"

    hidden_states = hidden_states.view(-1, hidden_size)
    topk_weights = topk_weights.view(-1, top_k)
    topk_ids = topk_ids.view(-1, top_k)

    assert hidden_states.is_contiguous()
    assert topk_weights.is_contiguous()
    assert topk_ids.is_contiguous()
    num_experts, _, intermediate_size = w2_weight.shape
    del outer, intermediate_size

    return hidden_states, topk_weights, topk_ids, top_k, num_experts


def run_moe_preprocess(topk_ids: torch.Tensor, num_experts: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
]:
    n = topk_ids.numel()
    # Decode/verify band: the whole preprocess (cast + stable sort + src2dst +
    # offsets + counts + block schedule, ~10 launches) collapses into one
    # single-CTA kernel; outputs are bit-identical (see test_moe_preprocess).
    if 0 < n <= FUSED_PREPROCESS_WIN_TOKENS:
        outs = fused_moe_preprocess(topk_ids.view(-1), num_experts)
        return (*outs, select_grouped_gemm_block_m(n))
    block_size_m = select_grouped_gemm_block_m(n)
    topk_ids = topk_ids.to(torch.int16)
    reorder_topk_ids, reorder_ids = torch.sort(topk_ids.view(-1), stable=True)
    src2dst = get_src2dst(reorder_ids)
    (
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
    ) = compute_grouped_gemm_metadata(
        reorder_topk_ids, num_experts, block_size_m=block_size_m
    )
    return (
        src2dst,
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
        reorder_topk_ids,
        block_size_m,
    )


def apply_grouped_bias(x: torch.Tensor, bias: torch.Tensor, reorder_ids: torch.Tensor):
    return x + bias.index_select(index=reorder_ids.int(), dim=0)


def activation(
    activation_type: str,
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    use_interleaved: bool = True,
):
    if activation_type == "silu_and_mul":
        assert (
            gateup_output.is_contiguous()
        ), f"{gateup_output.shape=} {gateup_output.stride()=}"
        assert gateup_output.ndim == 2, f"{gateup_output.shape=}"
        out_dtype = None
        if gateup_output.numel() == 0:
            return gateup_output.new_zeros(
                *gateup_output.shape[:-1], gateup_output.shape[-1] // 2, dtype=out_dtype
            )

        return silu_and_mul_helion(
            gateup_output, topk_weights, out_dtype, use_interleaved=use_interleaved
        )
    raise ValueError(f"Unsupported activation: {activation_type}")


def moe_tp_forward(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_weight_E_2f_D: torch.Tensor,
    w2_weight_EDf: torch.Tensor,
    w13_bias_E_2f: torch.Tensor | None = None,
    w2_bias_ED: torch.Tensor | None = None,
    activation_type: str = "silu_and_mul",
    use_interleaved: bool = True,
) -> torch.Tensor:
    orig_shape: torch.Size = hidden_states.shape
    hidden_states_TD, topk_weights_TK, topk_ids_TK, top_k, num_experts = (
        make_forward_inputs_2d(hidden_states, topk_weights, topk_ids, w2_weight_EDf)
    )
    del hidden_states, topk_weights, topk_ids

    (
        src2dst,
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
        reorder_topk_ids,
        block_size_m,
    ) = run_moe_preprocess(topk_ids_TK, num_experts)

    gateup_input_TK_D = pre_reorder(hidden_states_TD, src2dst, top_k)

    gateup_output_TK_2f = grouped_gemm_triton(
        gateup_input_TK_D,
        w13_weight_E_2f_D,
        num_experts,
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
        block_size_m=block_size_m,
    )

    if w13_bias_E_2f is not None:
        gateup_output_TK_2f = apply_grouped_bias(
            gateup_output_TK_2f, w13_bias_E_2f, reorder_topk_ids
        )

    down_input_TK_f = activation(
        activation_type, gateup_output_TK_2f, use_interleaved=use_interleaved
    )

    down_output_TK_D = grouped_gemm_triton(
        down_input_TK_f,
        w2_weight_EDf,
        num_experts,
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
        block_size_m=block_size_m,
    )

    if w2_bias_ED is not None:
        down_output_TK_D = apply_grouped_bias(
            down_output_TK_D, w2_bias_ED, reorder_topk_ids
        )

    return post_reorder(down_output_TK_D, src2dst, topk_weights_TK).view(orig_shape)


class InklingSharedFusedMoE(FusedMoE):
    """Sink shared experts (E = n_shared, top_k = n_shared) as a stock FusedMoE.

    Every token routes to all sink experts weighted by the gate's gammas, so
    the fused kernel computes sum_j gamma_j * expert_j(x) -- the same math as
    the bmm-based InklingBatchDenseMLP it replaces. Always shards over the full TP
    group at EP=1 (see __init__), independent of the routed experts' EP setting.
    """

    # Duck-typed marker: srt/lora keys this module's buffers under *_shared_moe so
    # a layer can carry both a routed and a sink FusedMoE without collision.
    is_shared_fused_moe = True

    def __init__(
        self,
        n_shared_experts: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        prefix: str,
        quant_config: QuantizationConfig | None,
        inference_moe_w13_interleaved: bool,
    ) -> None:
        # FusedMoE.__init__ reads get_parallel() once and caches it on self, so
        # scoping the override to just this call is sufficient for the module's lifetime.
        with get_parallel().override(
            moe_ep_size=1,
            moe_ep_rank=0,
            moe_tp_size=get_parallel().tp_size,
            moe_tp_rank=get_parallel().tp_rank,
        ):
            super().__init__(
                num_experts=n_shared_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_id=layer_id,
                top_k=n_shared_experts,
                num_fused_shared_experts=0,
                reduce_results=False,
                quant_config=quant_config,
                prefix=prefix,
                activation="silu",
                # TopK takes the gate's gamma-weighted topk verbatim, no re-routing.
                routing_method_type=RoutingMethodType.TopK,
                is_gated=True,
                use_weight_loader_fused=True,
                with_bias=False,
                # InklingMoE feeds the same hidden_states to routed experts AND sink; an
                # inplace runner (stock triton under --enable-lora bf16) corrupts the sink.
                inplace=False,
            )
        # Arms the fp4 w13 de-interleave in ModelOpt; bf16 is de-interleaved
        # separately at load time in models/inkling.py.
        self.inference_moe_w13_interleaved = inference_moe_w13_interleaved


def _build_inkling_shared_experts(
    *,
    n_shared_experts: int,
    shared_expert_sink: bool,
    shared_experts_size: int,
    inference_moe_w13_interleaved: bool,
    hidden_size: int,
    intermediate_size: int,
    layer_id: int,
    prefix: str,
    moe_tp_rank: int,
    moe_tp_size: int,
    quant_config: QuantizationConfig | None = None,
) -> nn.Module | None:
    if n_shared_experts <= 0:
        return None
    if shared_expert_sink:
        shared_prefix = add_prefix("shared_experts", prefix)
        shared_sink_serves_fp4 = InklingBatchDenseMLP._resolve_fp4_strategy(
            quant_config, shared_prefix
        ).serves_fp4
        use_fused_shared = use_inkling_shared_fused_moe(
            inference_moe_w13_interleaved=inference_moe_w13_interleaved,
            shared_sink_serves_fp4=shared_sink_serves_fp4,
        )
        if use_fused_shared:
            # moe_tp_rank/moe_tp_size are derived internally; kept as args only
            # for the legacy and non-sink InklingDenseMLP paths below.
            return InklingSharedFusedMoE(
                n_shared_experts=n_shared_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                layer_id=layer_id,
                prefix=shared_prefix,
                quant_config=quant_config,
                inference_moe_w13_interleaved=inference_moe_w13_interleaved,
            )
        dense_kwargs = dict(
            n_shared_experts=n_shared_experts,
            d_model=hidden_size,
            shared_d_mlp=intermediate_size,
            layer_id=layer_id,
            prefix=shared_prefix,
            quant_config=quant_config,
            inference_moe_w13_interleaved=inference_moe_w13_interleaved,
            tp_rank=moe_tp_rank,
            tp_size=moe_tp_size,
            tp_group=get_tensor_model_parallel_group(),
        )
        return InklingBatchDenseMLP(
            **dense_kwargs,
            linearized_bf16=(
                lora_compatible_layout_enabled()
                or envs.SGLANG_OPT_LINEARIZED_SHARED_SINK.get()
            ),
        )
    return InklingDenseMLP(
        hidden_size=hidden_size,
        intermediate_size=n_shared_experts * shared_experts_size * intermediate_size,
        use_global_scale=False,
        fused=True,
        layer_id=layer_id,
        prefix=add_prefix("shared_experts", prefix),
        quant_config=quant_config,
        tp_rank=moe_tp_rank,
        tp_size=moe_tp_size,
        tp_group=get_tensor_model_parallel_group(),
    )


class InklingMoE(nn.Module):
    def __init__(
        self,
        config: InklingModelConfig,
        layer_id: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.quant_config = quant_config
        self.layer_id = layer_id
        self.prefix = prefix
        hidden_size = config.hidden_size
        self.n_shared_experts = config.n_shared_experts
        self.shared_expert_sink = config.shared_expert_sink
        self.shared_experts_size = config.shared_experts_size
        self.intermediate_dim = config.intermediate_size
        self.n_routed_experts = config.n_routed_experts
        self.experts_per_token = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.norm_after_topk = config.norm_after_topk
        self.use_global_scale = config.use_global_scale
        self.use_gate_bias = config.use_gate_bias
        self.gate_activation = config.gate_activation
        self.inference_moe_w13_interleaved = config.inference_moe_w13_interleaved
        self.moe_ep_size = get_parallel().moe_ep_size

        self.gate = InklingGate(
            d_model=hidden_size,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts if self.shared_expert_sink else 0,
            experts_per_token=self.experts_per_token,
            route_scale=self.route_scale,
            layer_id=layer_id,
            prefix=prefix,
            norm_after_topk=self.norm_after_topk,
            use_global_scale=self.use_global_scale,
            use_gate_bias=self.use_gate_bias,
            gate_activation=self.gate_activation,
            shared_expert_sink=self.shared_expert_sink,
        )

        # Routed experts: the standard SGLang FusedMoE (no shared experts here). The Inkling
        # gate pre-routes (produces topk), so routing_method_type=TopK makes the runner
        # honor the supplied topk rather than re-route. reduce_results=False because the
        # single all_reduce in forward() covers routed + shared together.
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_dim,
            layer_id=layer_id,
            top_k=self.experts_per_token,
            num_fused_shared_experts=0,
            reduce_results=False,
            quant_config=self.quant_config,
            prefix=add_prefix("experts", prefix),
            activation="silu",
            routing_method_type=RoutingMethodType.TopK,
            is_gated=True,
            use_weight_loader_fused=True,
            with_bias=False,
            # See InklingSharedFusedMoE above: hidden_states is shared with the sink;
            # inplace runners corrupt it.
            inplace=False,
        )
        # ModelOptNvFp4FusedMoEMethod.process_weights_after_loading de-interleaves the
        # Inkling interleaved-w13 layout only when this attr is truthy; the stock FusedMoE
        # never sets it, so set it explicitly (else w13 is not de-interleaved).
        self.experts.inference_moe_w13_interleaved = (
            config.inference_moe_w13_interleaved
        )

        if (
            is_hip()
            and (
                get_moe_runner_backend().is_aiter()
                or get_moe_runner_backend().is_auto()
            )
            and isinstance(self.experts.quant_method, UnquantizedFusedMoEMethod)
            and not lora_compatible_layout_enabled()
            and not bf16_routed_uses_stock_fused_moe(self.quant_config)
        ):
            self.experts._skip_aiter_moe_shuffle = True

        # Inkling shared expert is a separate dense MLP (gammas + sink) — NOT FusedMoE's
        # num_fused_shared_experts mechanism. Owned here; it runs with reduce_scatter so
        # it does not self-all-reduce (the final all_reduce below covers routed+shared).
        self.shared_experts = _build_inkling_shared_experts(
            n_shared_experts=self.n_shared_experts,
            shared_expert_sink=self.shared_expert_sink,
            shared_experts_size=self.shared_experts_size,
            inference_moe_w13_interleaved=config.inference_moe_w13_interleaved,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_dim,
            layer_id=layer_id,
            prefix=prefix,
            # Shared expert is a replicated dense MLP: shard over the full tp group, not
            # moe_tp (the single full-tp all_reduce in forward() reconstructs it).
            moe_tp_rank=get_parallel().tp_rank,
            moe_tp_size=get_parallel().tp_size,
            quant_config=self.quant_config,
        )
        if isinstance(self.shared_experts, InklingSharedFusedMoE):
            # Static "route to every sink expert" ids; int32 for the trtllm topk
            # packer. Expanded per token in _forward_shared. Built lazily at first
            # forward rather than as a registered buffer: sglang's own
            # release/resume stashes and restores named_buffers, but RL
            # trainer-side flows that rebuild or diff engine state re-ship only
            # *parameters*, so a buffer allocated in the memory-saver pool comes
            # back as uninitialized memory there, silently corrupting every
            # shared-expert gather. A runtime allocation lives outside the pool
            # in both flows; the hot path is a plain attribute read.
            self._shared_topk_ids = None
        self.alt_stream = (
            alt_stream if is_cuda() and self.shared_experts is not None else None
        )
        # --enable-scattered-sconv: the output reduction becomes a hidden-dim
        # reduce-scatter (the consumer mlp_sconv runs on the [T, H/P] shard).
        from sglang.srt.runtime_context import get_server_args

        self.scattered_sconv = get_server_args().enable_scattered_sconv
        # Fold the shared-expert partials into the custom AR kernels (or their
        # stage-in copies) instead of a separate torch.add per MoE layer.
        self._fused_ar_shared = envs.SGLANG_OPT_USE_INKLING_FUSED_AR_SHARED.get()
        # The alt-stream fused sink races with the marlin routed GEMM on the shared
        # input under allocator churn (NaN in MTP draft extend); clone breaks the
        # shared storage. Scoped to the confirmed combo, others stay zero-copy.
        lora_enabled = lora_compatible_layout_enabled()
        self._clone_fused_sink_input = (
            self.shared_expert_sink
            and self.shared_experts is not None
            and (
                not isinstance(self.shared_experts, InklingBatchDenseMLP)
                or lora_enabled
            )
            and (
                get_fp4_gemm_runner_backend().is_marlin()
                or get_moe_runner_backend().is_marlin()
            )
        )

        # Packed topk is for the stock SRT apply path: quantized layers, or bf16 on the
        # trtllm_routed runner (unquantized ckpts only — a quantized ckpt's excluded
        # bf16 layers resolve to the triton runner, which needs standard topk).
        # moe_tp_forward and MoE-LoRA also need standard topk (LoRA packs internally).
        self.gate.emit_packed_topk = not lora_compatible_layout_enabled() and (
            not isinstance(self.experts.quant_method, UnquantizedFusedMoEMethod)
            or bf16_routed_uses_stock_fused_moe(self.quant_config)
        )

    def _forward_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        packed_topk_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            isinstance(self.experts.quant_method, UnquantizedFusedMoEMethod)
            and not lora_compatible_layout_enabled()
            and not bf16_routed_uses_stock_fused_moe(self.quant_config)
        ):
            # Preserve the Inkling triton grouped-GEMM for the unquantized (bf16) path; the
            # FusedMoE still owns the w13/w2 weights created by the unquantized method.
            # Skipped under --enable-lora (moe_tp_forward would drop the routed LoRA
            # delta) and for unquantized ckpts on trtllm_routed (its prep made w13/w2
            # 4-D block-shuffled): those run the stock forward (w13 [up||gate] at load).
            if self.moe_ep_size > 1:
                # moe_tp_forward has no local_expert_offset: remap the gate's global topk
                # ids to this rank's local experts and zero the non-local weights.
                local = self.n_routed_experts // self.moe_ep_size
                lo = get_parallel().moe_ep_rank * local
                is_local = (topk_ids >= lo) & (topk_ids < lo + local)
                topk_weights = topk_weights * is_local.to(topk_weights.dtype)
                topk_ids = torch.where(
                    is_local, topk_ids - lo, torch.zeros_like(topk_ids)
                )
            return moe_tp_forward(
                hidden_states,
                topk_weights,
                topk_ids,
                cast(torch.Tensor, self.experts.w13_weight),
                cast(torch.Tensor, self.experts.w2_weight),
                None,
                None,
                "silu_and_mul",
                use_interleaved=self.inference_moe_w13_interleaved,
            )
        # NVFP4 routed experts: feed the gate's topk straight into FusedMoE.forward. For
        # flashinfer_trtllm_routed the dispatcher is a pass-through, so a PackedTopKOutput
        # reaches the same trtllm_fp4_block_scale_routed_moe kernel as before.
        router_logits = hidden_states.new_empty((hidden_states.shape[0], 0))
        if packed_topk_ids is not None:
            topk_output = PackedTopKOutput(
                packed_topk_ids=packed_topk_ids, router_logits=router_logits
            )
        else:
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=router_logits,
            )
        return self.experts(hidden_states, topk_output)

    def _forward_shared(
        self,
        hidden_states: torch.Tensor,
        shared_gammas: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.shared_experts is None:
            return None
        if self.shared_expert_sink:
            if isinstance(self.shared_experts, InklingBatchDenseMLP):
                assert shared_gammas is not None
                return self.shared_experts(
                    hidden_states, gammas=shared_gammas, use_reduce_scatter=True
                )
            assert shared_gammas is not None
            # Every token selects all E=n_shared experts, weighted by the gate's
            # gammas; ids/weights must be contiguous int32/fp32 (trtllm packer asserts both).
            num_tokens = hidden_states.shape[0]
            if (
                self._shared_topk_ids is None
                or self._shared_topk_ids.device != hidden_states.device
            ):
                self._shared_topk_ids = torch.arange(
                    self.n_shared_experts,
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
            topk_output = StandardTopKOutput(
                topk_weights=shared_gammas.to(torch.float32),
                topk_ids=self._shared_topk_ids.unsqueeze(0)
                .expand(num_tokens, -1)
                .contiguous(),
                router_logits=hidden_states.new_empty((num_tokens, 0)),
            )
            return self.shared_experts(hidden_states, topk_output)
        return self.shared_experts(hidden_states, use_reduce_scatter=True)

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch | None = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """Return local routed and shared partial sums when ``reduce=False``.

        The caller is then responsible for tensor-parallel reduction.
        """
        del forward_batch
        topk_weights, topk_ids, shared_gammas, packed_topk_ids = self.gate(x)

        allow_lora_overlap = True
        if lora_compatible_layout_enabled():
            # ===== TO BE REFACTORED ====
            from sglang.srt.lora.trtllm_lora_temp.inkling_dense import (
                allow_inkling_moe_two_stream,
            )

            # ===== END TO BE REFACTORED ====

            allow_lora_overlap = allow_inkling_moe_two_stream(
                self.shared_experts, self.experts, x.shape[0]
            )

        use_two_stream = (
            self.alt_stream is not None
            and x.is_cuda
            and x.shape[0] > 0
            and envs.SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP.get()
            and get_is_capture_mode()
            and allow_lora_overlap
        )
        if use_two_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                sink_x = x.clone() if self._clone_fused_sink_input else x
                shared_out = self._forward_shared(sink_x, shared_gammas)
            out = self._forward_routed(x, topk_weights, topk_ids, packed_topk_ids)
            current_stream.wait_stream(self.alt_stream)
        else:
            out = self._forward_routed(x, topk_weights, topk_ids, packed_topk_ids)
            shared_out = self._forward_shared(x, shared_gammas)

        if not reduce:
            if shared_out is not None:
                if self._fused_ar_shared:
                    # Hand the shared partials to the consuming fused-AR call
                    # (register fold in the decode/verify kernels; pre-add in
                    # the scattered/extend consumers) -- deletes the separate
                    # {routed + shared} add on the fold paths.
                    stash_ar_shared(shared_out)
                    return out
                tp = get_tensor_model_parallel_group()
                buf = get_ar_buffer(tp, out.shape[0], out.shape[1], out.dtype)
                if buf is not None:
                    torch.add(out, shared_out, out=buf)
                    return buf
                return out + shared_out
            return out

        tp = get_tensor_model_parallel_group()
        if shared_out is not None:
            if self._fused_ar_shared and not self.scattered_sconv:
                # The AR dispatch folds in-kernel on the fold paths and pre-adds
                # during its stage-in otherwise -- never worse than the explicit
                # add below.
                return symm_mem_all_reduce(out, tp, shared=shared_out)
            buf = get_ar_buffer(tp, out.shape[0], out.shape[1], out.dtype)
            if buf is not None:
                torch.add(out, shared_out, out=buf)
                if self.scattered_sconv:
                    # Scattered sconv: reduce + scatter hidden -> the [T, H/P]
                    # shard mlp_sconv consumes; all-gather happens after it.
                    return reduce_scatter_hidden(buf, tp, input_is_ar_buffer=True)
                return symm_mem_all_reduce(buf, tp, input_is_ar_buffer=True)
            out = out + shared_out

        if self.scattered_sconv:
            return reduce_scatter_hidden(out, tp)
        return symm_mem_all_reduce(out, tp)
