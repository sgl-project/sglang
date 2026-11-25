"""Oracles for the MiniMax-M3.1 low-batch FFN tests: one TP rank's MoE layer in
SGLang's processed layout (random weights at checkpoint magnitudes), the
stock routed call it replaces (``moe_fused_gate`` top-k, then the TRT-LLM-gen
routed experts with ``do_finalize=False``), and an MXFP8 shared-expert
reference in torch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import torch

# Import quantization before flashinfer_trtllm; the reverse is a circular import.
# isort: off
from sglang.srt.layers.quantization import minimax_m3_nvfp4
from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm

# isort: on
from sglang.kernels.ops.moe.minimax_m3_low_batch.shapes import (
    EXPERTS,
    HIDDEN,
    SWIGLU_ALPHA,
    SWIGLU_LIMIT,
    TOP_K,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend


def _random_bytes(shape, generator, low: int, high: int, dtype=torch.uint8):
    return (
        torch.randint(low, high, shape, generator=generator, device="cuda")
        .to(torch.uint8)
        .view(dtype)
    )


def build_routed_layer(columns: int, generator: torch.Generator) -> torch.nn.Module:
    """One rank's NVFP4 experts (``columns`` intermediate columns), processed by
    ``MiniMaxM3Nvfp4MoE._process_trtllm_weights`` exactly as at load time.
    ``layer.quant_info`` is the payload the scheme hands the stock runner."""
    layer = torch.nn.Module()
    e4m3 = torch.float8_e4m3fn
    params = dict(
        w13_weight_packed=_random_bytes(
            (EXPERTS, 2 * columns, HIDDEN // 2), generator, 0, 256
        ),
        # E4M3 block scales between 0.5 and 4.
        w13_weight_scale=_random_bytes(
            (EXPERTS, 2 * columns, HIDDEN // 16), generator, 0x30, 0x48, e4m3
        ),
        w13_weight_global_scale=torch.full((EXPERTS, 2), 1.5e4, device="cuda"),
        w13_input_global_scale=torch.ones((EXPERTS, 2), device="cuda"),
        w2_weight_packed=_random_bytes(
            (EXPERTS, HIDDEN, columns // 2), generator, 0, 256
        ),
        w2_weight_scale=_random_bytes(
            (EXPERTS, HIDDEN, columns // 16), generator, 0x30, 0x48, e4m3
        ),
        w2_weight_global_scale=torch.full((EXPERTS,), 1.5e4, device="cuda"),
        w2_input_global_scale=torch.ones((EXPERTS,), device="cuda"),
    )
    for name, tensor in params.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    layer.num_experts = layer.num_local_experts = EXPERTS
    layer.hidden_size = HIDDEN
    layer.moe_ep_rank = 0
    layer.intermediate_size_per_partition = columns
    layer.routing_method_type = None
    layer._minimax_activation = "swigluoai"
    layer._mega_moe_trtllm_layout = False
    layer.moe_runner_config = MoeRunnerConfig(
        num_experts=EXPERTS,
        num_local_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size_per_partition=columns,
        top_k=TOP_K,
        gemm1_alpha=SWIGLU_ALPHA,
        gemm1_beta=1.0,
        gemm1_clamp_limit=SWIGLU_LIMIT,
        inplace=False,
    )
    with mock.patch.object(
        minimax_m3_nvfp4,
        "get_moe_runner_backend",
        return_value=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
    ):
        scheme = minimax_m3_nvfp4.MiniMaxM3Nvfp4MoE()
    assert scheme.use_flashinfer_trtllm and not scheme.use_mega_moe_prefill
    scheme._process_trtllm_weights(layer)
    captured = {}
    scheme.runner = SimpleNamespace(
        run=lambda dispatch, quant_info: captured.setdefault("q", quant_info) and None
    )
    scheme.apply_weights(layer, None)
    layer.quant_info = captured["q"]
    layer.scheme = scheme
    return layer


def build_shared(columns: int, generator: torch.Generator) -> SimpleNamespace:
    """One rank's MXFP8 shared expert: E4M3 ``gate_up`` [2I, 6144] (rows
    ``[gate; up]``) and ``down`` [6144, I], canonical E8M0 scales [N, K/32]
    and the 128x4 swizzled copies SGLang's FlashInfer MXFP8 linears keep."""
    from flashinfer import block_scale_interleave

    def linear(n: int, k: int):
        values = torch.randn((n, k), generator=generator, device="cuda") * 64
        weight = values.clamp(-448, 448).to(torch.float8_e4m3fn)
        scale = _random_bytes((n, k // 32), generator, 114, 118)
        return weight, scale, block_scale_interleave(scale).contiguous()

    gate_up, gate_up_scale, gate_up_swizzled = linear(2 * columns, HIDDEN)
    down, down_scale, down_swizzled = linear(HIDDEN, columns)
    return SimpleNamespace(
        columns=columns,
        gate_up=gate_up,
        gate_up_scale=gate_up_scale,
        gate_up_swizzled=gate_up_swizzled,
        down=down,
        down_scale=down_scale,
        down_swizzled=down_swizzled,
    )


def stock_topk(
    logits: torch.Tensor, bias: torch.Tensor, routed_scaling_factor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(weights, ids)`` as the stock sigmoid top-4 router returns them."""
    from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate

    weights, ids = moe_fused_gate(
        logits,
        bias,
        TOP_K,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=True,
    )
    return weights, ids.int()


def stock_routed_deferred(
    x: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    layer: torch.nn.Module,
):
    """The stock routed call on ``x`` stopped after GEMM2: a
    ``FlashInferTrtllmDeferredFinalizeOutput``."""
    dispatch = StandardDispatchOutput(
        hidden_states=x,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=weights, topk_ids=ids, router_logits=None
        ),
    )
    with (
        flashinfer_trtllm.flashinfer_trtllm_deferred_finalize_context(),
        mock.patch.object(flashinfer_trtllm, "is_allocation_symmetric", lambda: False),
    ):
        return flashinfer_trtllm.fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch, layer.quant_info, layer.moe_runner_config, use_routed_topk=True
        ).hidden_states


def weighted_rows(
    routed: torch.Tensor, weights: torch.Tensor, permuted: torch.Tensor, rows: int
) -> torch.Tensor:
    """FP32 ``sum_k weights[t, k] * routed[permuted[t, k]]`` for the first
    ``rows`` rows ``t`` (FlashInfer may hand back more)."""
    index = permuted.reshape(-1, TOP_K)[:rows].long()
    routed = routed.reshape(-1, routed.shape[-1]).float()
    picked = routed[index.reshape(-1)].view(*index.shape, routed.shape[-1])
    weights = weights.reshape(-1, TOP_K)[:rows].float()
    return (picked * weights.unsqueeze(-1)).sum(1)


def _mxfp8_fake_quant(values: torch.Tensor) -> torch.Tensor:
    """MXFP8 round trip of FP32 rows: per 32 values an E8M0 scale (amax / 448
    rounded up to a power of two) and saturating E4M3."""
    blocks = values.view(values.shape[0], -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.where(
        amax > 0, torch.ceil(torch.log2(amax / 448)), torch.zeros_like(amax)
    )
    scale = torch.exp2(exponent.clamp(-127, 127))
    codes = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (codes.float() * scale).view_as(values)


def _dequant_weight(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    exponent = scale.view(torch.uint8).float() - 127
    return weight.float() * torch.exp2(exponent).repeat_interleave(32, dim=1)


def shared_reference(x: torch.Tensor, shared: SimpleNamespace) -> torch.Tensor:
    """BF16 [rows, 6144]: MXFP8 x, FC1 rounded to BF16, SwiGLU-OAI (gate
    first), MXFP8 h, FC2 rounded to BF16."""
    columns = shared.columns
    gate_up = (
        _mxfp8_fake_quant(x.float())
        @ _dequant_weight(shared.gate_up, shared.gate_up_scale).t()
    )
    gate_up = gate_up.bfloat16().float()
    gate = gate_up[:, :columns].clamp(max=SWIGLU_LIMIT)
    up = gate_up[:, columns:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    h = gate * torch.sigmoid(SWIGLU_ALPHA * gate) * (up + 1)
    out = _mxfp8_fake_quant(h) @ _dequant_weight(shared.down, shared.down_scale).t()
    return out.bfloat16()


def relative_errors(got: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    """Per-row relative L2 error (rows where ``want`` is zero report the
    absolute norm of ``got``)."""
    got, want = got.float(), want.float()
    norm = want.norm(dim=-1)
    diff = (got - want).norm(dim=-1)
    return torch.where(norm > 0, diff / norm.clamp(min=1e-30), diff)


def near_tie_rows(
    logits: torch.Tensor, bias: torch.Tensor, gap: float = 1e-6
) -> torch.Tensor:
    """Rows whose 4th and 5th biased sigmoid scores are within ``gap``: the
    top-4 set may legitimately differ there between two implementations."""
    biased = torch.sigmoid(logits.float()) + bias.float()
    top = biased.topk(TOP_K + 1, dim=-1).values
    return (top[:, TOP_K - 1] - top[:, TOP_K]) < gap


def poison_allocator(megabytes: int = 512) -> None:
    """Fill freed caching-allocator memory with NaN bytes, so a buffer read
    before it is written shows up."""
    block = torch.empty(megabytes << 20, dtype=torch.uint8, device="cuda")
    block.fill_(0xFF)
    del block
