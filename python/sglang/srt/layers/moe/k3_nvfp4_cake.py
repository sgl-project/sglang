"""Opt-in TP8-local K3 NVFP4 SiTU experiment using FlashInfer's Cake backend."""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_logged_enabled = False

# Each stream owns scratch. Layer calls on that stream are ordered; CUDA graphs
# retain the same scratch addresses. Different shapes retain separate buffers.
_workspaces = {}

# Qualified on actual layer-12 TP0 weights; M112 regresses against TRTLLM.
_QUALIFIED_TOKENS = (1, 8, 16, 32, 64, 128)

# Token counts that run Cake padded up to the next qualified count. Layer-12
# TP0 shard, uniform top-16 routing, every M in 1..64 and 72..128 by 8, net of
# the radix top-k + bf16 latent cast Cake needs and the fused route+quant it
# gives up (SGLANG_K3_NVFP4_ROUTE_QUANT): M <= 4 and 33..40 favor trtllm-gen.
_PAD_RANGES = ((5, 32), (41, 128))


def padded_tokens(num_tokens: int):
    """Cake token count for ``num_tokens`` routed tokens, or None for trtllm-gen."""
    if not any(lo <= num_tokens <= hi for lo, hi in _PAD_RANGES):
        return None
    return next(t for t in _QUALIFIED_TOKENS if t >= num_tokens)


@triton.jit
def _finalize_fp32(expert, weights, mapping, out, H: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.full((B,), 0, tl.float32)
    for k in range(16):
        row = tl.load(mapping + token * 16 + k)
        weight = tl.load(weights + token * 16 + k).to(tl.float32)
        value = tl.load(expert + row * H + col, col < H, 0).to(tl.float32)
        acc = acc + value * weight
    tl.store(out + token * H + col, acc, col < H)


def _prepare_prefix(workspace, num_tokens):
    # Experimental integration with FlashInfer 0.7.2's prepared stage API.
    # Retain quant/routing/FC1/FC2; replace only the BF16-weight finalizer.
    from flashinfer.fused_moe.cake_kimi_k3_situ import _prepared
    from flashinfer.jit.cake_kimi_k3_situ import MODULES, cake_situ_stage_entries

    prepared = _prepared(workspace, num_tokens)
    assert prepared["stages"][-1] == "finalize"
    fc2_args = MODULES[prepared["modules"]["fc2"]]["arg_plan"]
    assert all(
        name != "partial_scale" for _, name in fc2_args
    ), "Q8 partials need a different finalizer"
    submit, full_plan, full_slots = prepared["submit"]
    finalizer_header = (
        len(cake_situ_stage_entries(prepared["modules"]["finalize"], prepared["arch"]))
        + 1
    )
    prefix_size = len(full_plan) - finalizer_header - len(prepared["launches"][-1][1])
    plan = list(full_plan[:prefix_size])
    slots = tuple((index, name) for index, name in full_slots if index < prefix_size)
    assert all(name not in ("weights", "out") for _, name in slots)
    views = prepared["views"]
    return submit, plan, slots, views["expert_output"], views["token_to_permuted"]


def _run_fp32_prefix(state, layer, x, ids, weights, out):
    import tvm_ffi

    submit, plan, slots, expert_output, mapping = state
    call = dict(
        x=x,
        ids=ids.view(-1),
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        sf1=layer.w13_weight_scale.view(torch.uint8).view(896 * 6, 56, 2, 256),
        sf2=layer.w2_weight_scale.view(torch.uint8).view(896 * 28, 6, 2, 256),
        qx=layer.w13_input_scale_quant,
        qa=layer.g1_scale_c,
        decode1=layer.g1_alphas,
        decode2=layer.g2_alphas,
        alpha=layer.gemm1_alpha,
        beta=layer.gemm1_beta,
    )
    for index, name in slots:
        plan[index] = call[name]
    with tvm_ffi.use_torch_stream():
        submit(*plan)
    _finalize_fp32[(out.shape[0], triton.cdiv(x.shape[1], 256))](
        expert_output,
        weights,
        mapping,
        out,
        x.shape[1],
        256,
        enable_fp_fusion=False,
    )


def try_cake_situ(layer, x, routing, out, *, max_tokens=128, cake_tokens=None):
    """Write the rank-local expert result, or return False for unsupported input.

    The model remains responsible for shared experts, all-reduce and latent
    normalization. This path intentionally does not handle deferred finalize.

    ``cake_tokens`` > ``x.shape[0]`` runs Cake at that padded count: ``x`` must
    then be the leading rows of a contiguous ``[cake_tokens, hidden]`` buffer
    whose tail rows are finite. Padded rows reuse row 0's experts, so they read
    no extra expert weights, and the finalize only writes the real rows.
    """
    global _logged_enabled

    num_tokens = x.shape[0]
    cake_tokens = cake_tokens or num_tokens
    if cake_tokens > num_tokens:
        if not (
            x.is_contiguous()
            and x.untyped_storage().nbytes()
            >= x.storage_offset() * x.element_size()
            + cake_tokens * x.shape[1] * x.element_size()
        ):
            return False
        x = torch.as_strided(x, (cake_tokens, x.shape[1]), (x.shape[1], 1))

    config = layer.moe_runner_config
    method = layer.quant_method
    if not (
        getattr(method, "use_flashinfer_trtllm_weight_layout", False)
        and not getattr(method.quant_config, "use_per_token_activation", False)
        and layer.moe_tp_size == 8
        and layer.moe_ep_size == 1
        and layer.num_local_experts == 896
        and layer.intermediate_size_per_partition == 384
        and not layer.reduce_results
        and config.activation == "situ"
        and config.routed_scaling_factor in (None, 1.0)
        and x.ndim == 2
        and 0 < x.shape[0] <= max_tokens
        and x.shape[0] in _QUALIFIED_TOKENS
        and x.shape[1] == 3584
        and x.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and out.shape == (num_tokens, x.shape[1])
        and out.device == x.device
        and out.dtype == x.dtype
        and out.is_contiguous()
        and tuple(layer.w13_weight.shape) == (896, 768, 1792)
        and tuple(layer.w2_weight.shape) == (896, 3584, 192)
    ):
        return False

    if not hasattr(layer, "_cake_situ_scale_compatible"):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm K3 Cake weights before CUDA graph capture")
        # Cake uses one FC1 dequant scale for both halves. Validate once after
        # loading, rather than silently changing unequal gate/up scales.
        layer._cake_situ_scale_compatible = torch.equal(
            layer.g1_alphas, layer.g1_alphas_up
        )
    if not layer._cake_situ_scale_compatible:
        return False

    from flashinfer.fused_moe import (
        cake_fused_moe_prepare_workspace,
        cutlass_fused_moe_workspace_size,
    )
    from flashinfer.tllm_enums import ActivationType

    from sglang.srt.layers.moe.topk import TopKOutputChecker

    if TopKOutputChecker.format_is_bypassed(routing):
        routing = routing.to_standard()
    if not TopKOutputChecker.format_is_standard(routing):
        return False
    if routing.topk_ids.shape != (num_tokens, 16):
        return False

    stream = torch.cuda.current_stream(x.device)
    key = (x.device.index, stream.cuda_stream, x.shape[0])
    entry = _workspaces.get(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the K3 Cake shape before CUDA graph capture")
        size = cutlass_fused_moe_workspace_size(
            max_num_tokens=x.shape[0],
            hidden_size=3584,
            intermediate_size=384,
            num_experts_total=896,
            top_k=16,
            x_dtype=torch.bfloat16,
            weight_dtype=torch.uint8,
            activation_type=ActivationType.Situ,
            tp_size=8,
            backend="cake",
        )
        workspace = torch.empty(size, dtype=torch.uint8, device=x.device)
        cake_fused_moe_prepare_workspace(workspace, x.shape[0])
        entry = (workspace, _prepare_prefix(workspace, x.shape[0]))
        _workspaces[key] = entry
    workspace, prefix = entry

    if not _logged_enabled:
        logger.info(
            "K3 Cake NVFP4 SiTU with FP32 finalize enabled for TP8, token counts %s",
            _QUALIFIED_TOKENS,
        )
        _logged_enabled = True

    ids = routing.topk_ids.to(torch.int32)
    if cake_tokens > num_tokens:
        ids = torch.cat([ids, ids[:1].expand(cake_tokens - num_tokens, -1)])
    _run_fp32_prefix(
        prefix,
        layer,
        x,
        ids.contiguous(),
        routing.topk_weights.contiguous(),
        out,
    )
    return True
