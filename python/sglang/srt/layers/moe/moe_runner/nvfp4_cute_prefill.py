"""Qwen3.8 NVFP4 routed experts on FlashInfer's Rubin CuTe-DSL MoE (SGLANG_QWEN_MOE_PREFILL_CUTE_RUBIN, default off).

Meant for a prefill-only worker. The worker keeps --moe-runner-backend flashinfer_trtllm and all of its glue
(routed-input pre-quant, deferred output, the fused finalize + shared-expert gate), but its routed experts are prepared
in the CuTe-DSL v2 layout in place of trtllm-gen's (one copy, the same bytes), and each routed call runs:

  routing from the served router logits (q38_prefill_routing: trtllm-gen's RenormalizeNaive top-10, bit for bit)
  -> moe_sort -> FlashInfer 0.7.2 gather GEMM1 + SwiGLU + NVFP4 quant (Rubin)
  -> GEMM2 in deterministic mode, each (token, slot) row stored to its own slot (rubin_cutedsl_finalize_fusion:
     FlashInfer 0.7.2 + #6307 + #6950, the first release with both is 0.7.3)
  -> the served finalize through an identity index (expanded rows are already in (token, slot) order).

Same weights, same routing and the same finalize arithmetic as the trtllm-gen path; only the GEMMs' accumulation order
differs. Every routed call of a converted layer must take this path: the trtllm-gen layout no longer exists.
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple, Optional

import torch

from sglang.srt.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)

ENV = "SGLANG_QWEN_MOE_PREFILL_CUTE_RUBIN"
ENABLED = get_bool_env_var(ENV, "false")

NUM_EXPERTS = 512
TOP_K = 10
HIDDEN = 2560
INTERMEDIATE = 640
_SF_VEC = 16
# Fastest of FlashInfer 0.7.2's Rubin tactics for both GEMMs at 16K and 32K rows (bench mb-h641-fc2fin-2):
# tile 128, MMA tiler (128, 256, 256), instruction (128, 256, 128), one CTA.
_TILE = 128
_MMA_TILER = (128, 256, 256)
_MMA_INST = (128, 256, 128)
_CLUSTER = (1, 1)


class CutePrefillWeights(NamedTuple):
    w13: torch.Tensor  # [E, 2I, H/2] uint8; [up, gate] halves interleaved in 64-row groups
    w13_sf: torch.Tensor  # GEMM1 E4M3 block scales, 128x4 swizzle then MMA layout
    w2: torch.Tensor  # [E, H, I/2] uint8, as loaded
    w2_sf: torch.Tensor  # GEMM2 block scales, MMA layout
    w1_alpha: torch.Tensor  # [E] fp32: gate weight_scale_2 / activation quant scale
    fc2_input_scale: torch.Tensor  # [1] fp32: GEMM1 output quant scale (1 / max down input_scale)
    w2_alpha: torch.Tensor  # [E] fp32


_logged: set = set()


def _log_once(key: str, level: int, msg: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.log(level, msg, *args)


def _ineligible_reason(layer, use_per_token_activation: bool) -> Optional[str]:
    import flashinfer
    from flashinfer.cute_dsl.utils import is_rubin_cute_dsl_available

    from sglang.srt.layers.moe.utils import (
        MoeRunnerBackend,
        RoutingMethodType,
        get_moe_a2a_backend,
    )
    from sglang.srt.runtime_context import get_lora

    cfg = layer.moe_runner_config
    scale_2 = layer.w13_weight_scale_2.data
    checks = (
        (torch.cuda.get_device_capability(layer.w13_weight.device) == (10, 7), "the device is not SM107"),
        # The routed and experimental TRT-LLM backends share this weight prep but take other runner paths.
        (layer.runner_backend.value == MoeRunnerBackend.FLASHINFER_TRTLLM.value, "the MoE backend is not flashinfer_trtllm"),
        # LoRA builds its own trtllm-gen quant info from these weights (trtllm_lora_temp/lora_layer.py).
        (not get_lora().enable_lora, "LoRA is enabled"),
        # The vendored GEMM2 and the dispatcher calls below follow FlashInfer 0.7.2's sources.
        (flashinfer.__version__ == "0.7.2", f"FlashInfer is {flashinfer.__version__}, not 0.7.2"),
        (is_rubin_cute_dsl_available(), "CuTe DSL has no Rubin helpers"),
        (get_moe_a2a_backend().is_none(), "an MoE all-to-all backend is set"),
        (
            layer.moe_tp_size == 1
            and layer.moe_ep_size == 1
            and layer.num_experts == layer.num_local_experts == NUM_EXPERTS,
            "needs TP1/EP1 with 512 local experts",
        ),
        (
            layer.hidden_size == HIDDEN
            and layer.intermediate_size_per_partition == INTERMEDIATE
            and layer.top_k == TOP_K,
            "needs hidden 2560, intermediate 640, top-10",
        ),
        (cfg.activation == "silu" and cfg.is_gated, "needs gated SiLU"),
        (
            cfg.gemm1_alpha is None and cfg.gemm1_clamp_limit is None and cfg.swiglu_limit is None,
            "a SwiGLU alpha or limit is set",
        ),
        (cfg.routed_scaling_factor in (None, 1.0), "a routed scaling factor is set"),
        (layer.routing_method_type == RoutingMethodType.RenormalizeNaive, "routing is not RenormalizeNaive"),
        (not use_per_token_activation, "per-token activation scales"),
        (layer.w13_input_scale_quant.numel() == 1, "the activation quant scale is not a scalar"),
        # GEMM1 takes one alpha per expert for both halves.
        (scale_2.dim() == 1 or torch.equal(scale_2[:, 0], scale_2[:, 1]), "gate and up weight_scale_2 differ"),
    )
    for ok, reason in checks:
        if not ok:
            return reason
    return None


def prepare_weights(layer, *, use_per_token_activation: bool) -> bool:
    """Convert the layer's routed experts to the CuTe-DSL layout in place of trtllm-gen's.

    Returns False (logged once) when the layer does not qualify; the caller then prepares trtllm-gen's layout.
    """
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
        interleave_w13_halves,
        resolve_cutedsl_standard_scales,
    )
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _compute_g1_scale_c
    from sglang.srt.layers.quantization.utils import swizzle_blockscale
    from sglang.srt.layers.utils import copy_or_rebind_param

    reason = _ineligible_reason(layer, use_per_token_activation)
    if reason is not None:
        _log_once("not-taken", logging.WARNING, "%s set but not taken: %s; trtllm-gen layout kept", ENV, reason)
        return False

    num_experts, gate_up, hidden_half = layer.w13_weight.shape
    hidden = hidden_half * 2
    intermediate = layer.w2_weight.shape[2] * 2
    # The SwiGLU GEMM1 reads [up, gate] (the trtllm-gen load order) interleaved in 64-row groups,
    # as the flashinfer_cutedsl runner prepares it.
    w13 = interleave_w13_halves(layer.w13_weight.data.view(torch.uint8), group_size=64, dim=1).contiguous()
    w13_sf = interleave_w13_halves(
        layer.w13_weight_scale.data.view(torch.uint8), group_size=64, dim=1
    ).contiguous().view(torch.float8_e4m3fn)
    w13_sf_mma = convert_sf_to_mma_layout(
        swizzle_blockscale(w13_sf).contiguous().view(torch.uint8).reshape(-1),
        m=gate_up,
        k=hidden,
        num_groups=num_experts,
        sf_vec_size=_SF_VEC,
    )
    del w13_sf
    w2_sf_mma = convert_sf_to_mma_layout(
        swizzle_blockscale(layer.w2_weight_scale.data).contiguous().view(torch.uint8).reshape(-1),
        m=hidden,
        k=intermediate,
        num_groups=num_experts,
        sf_vec_size=_SF_VEC,
    )
    w1_alpha, fc2_input_scale, w2_alpha, used_input_scale = resolve_cutedsl_standard_scales(layer)
    # The served activation quant (w13_input_scale_quant) feeds GEMM1, so its scale must be GEMM1's.
    if not torch.equal(used_input_scale.reshape(()), layer.w13_input_scale_quant.reshape(()).float()):
        raise RuntimeError(f"{ENV}: GEMM1 activation scale differs from the served routed-input quant")

    copy_or_rebind_param(layer, "w13_weight", w13)
    del w13
    copy_or_rebind_param(layer, "w13_weight_scale", w13_sf_mma)
    copy_or_rebind_param(layer, "w2_weight_scale", w2_sf_mma)
    if not layer.w2_weight.data.is_contiguous():
        copy_or_rebind_param(layer, "w2_weight", layer.w2_weight.data.contiguous())
    # Read by apply()'s quant-info constructor only; no trtllm-gen kernel runs on this layer.
    copy_or_rebind_param(
        layer,
        "g1_scale_c",
        _compute_g1_scale_c(
            layer.w2_input_scale_quant,
            layer.g1_alphas,
            layer.g1_alphas_up,
            True,
            activation=layer.moe_runner_config.activation,
        ),
    )
    layer.trtllm_padded_hidden_size = None
    layer._nvfp4_cute_prefill = CutePrefillWeights(
        w13=layer.w13_weight.data,
        w13_sf=layer.w13_weight_scale.data,
        w2=layer.w2_weight.data,
        w2_sf=layer.w2_weight_scale.data,
        w1_alpha=w1_alpha.contiguous(),
        fc2_input_scale=fc2_input_scale.reshape(1).contiguous(),
        w2_alpha=w2_alpha.contiguous(),
    )
    _log_once(
        "prepared",
        logging.INFO,
        "%s: routed experts of layer %s and later in the CuTe-DSL layout, not trtllm-gen's "
        "(w13 %s, w13_sf %s, w2 %s, w2_sf %s; one copy); tactic tile=%d mma=%s inst=%s",
        ENV,
        layer.layer_id,
        tuple(layer.w13_weight.shape),
        tuple(layer.w13_weight_scale.shape),
        tuple(layer.w2_weight.shape),
        tuple(layer.w2_weight_scale.shape),
        _TILE,
        _MMA_TILER,
        _MMA_INST,
    )
    return True


# GEMM2 kernels compiled by this process, by tactic and top-k (problem sizes are runtime arguments).
_FC2_KERNELS: dict = {}


def _fc2_unfused(
    *,
    a: torch.Tensor,
    a_scale: torch.Tensor,
    weights: CutePrefillWeights,
    tile_idx_to_expert_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    token_final_scales: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Rubin GEMM2 storing each permuted row, alpha-scaled, to its expanded (token, slot) row of `out`.

    The SM107 part of FlashInfer 0.7.2's blockscaled_contiguous_grouped_gemm_finalize_fusion dispatcher with
    #6950's use_fused_finalize=False, on the vendored kernel class.

    This launch must stay non-PDL: the finalize kernels after it read the expert weights and the identity index
    before their PDL wait, which is safe only because everything before this launch has completed.
    """
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import get_cutlass_dtype, get_max_active_clusters, make_ptr

    from sglang.kernels.ops.moe.rubin_cutedsl_finalize_fusion import (
        Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel as Fc2Kernel,
    )

    permuted_m = a.shape[0]
    num_experts, n, k_half = weights.w2.shape
    k = k_half * 2
    num_tokens, top_k = token_final_scales.shape
    fp4 = get_cutlass_dtype("float4_e2m1fn")
    sf = get_cutlass_dtype("float8_e4m3fn")
    bf16 = get_cutlass_dtype("bfloat16")
    gmem = cute.AddressSpace.gmem
    ptrs = (
        make_ptr(fp4, a.data_ptr(), gmem, assumed_align=32),
        make_ptr(fp4, weights.w2.data_ptr(), gmem, assumed_align=32),
        make_ptr(sf, a_scale.data_ptr(), gmem, assumed_align=16),
        make_ptr(sf, weights.w2_sf.data_ptr(), gmem, assumed_align=16),
        make_ptr(bf16, out.data_ptr(), gmem, assumed_align=32),
        make_ptr(cutlass.Float32, weights.w2_alpha.data_ptr(), gmem),
        make_ptr(cutlass.Int32, tile_idx_to_expert_idx.data_ptr(), gmem),
        make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), gmem),
        make_ptr(cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), gmem),
        make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), gmem),
        make_ptr(cutlass.Float32, token_final_scales.data_ptr(), gmem, assumed_align=16),
    )
    sizes = (permuted_m, n, k, num_experts, num_tokens, top_k)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    key = (_MMA_TILER, _MMA_INST, _CLUSTER, top_k)
    compiled = _FC2_KERNELS.get(key)
    if compiled is None:
        # The vendored epilogue copies whole N tiles (no column bound in 0.7.2).
        if n % _MMA_TILER[1] or not Fc2Kernel.can_implement(
            a_dtype=fp4, b_dtype=fp4, sf_dtype=sf, sf_vec_size=_SF_VEC, c_dtype=bf16,
            mma_inst_shape=_MMA_INST, mma_tiler=_MMA_TILER, cluster_shape_mn=_CLUSTER,
            m=permuted_m, n=n, k=k, l=num_experts, a_major="k", b_major="k", c_major="n",
        ):
            raise RuntimeError(f"{ENV}: the Rubin GEMM2 cannot run n={n} k={k} at tactic {_MMA_TILER}")
        kernel = Fc2Kernel(
            sf_vec_size=_SF_VEC,
            mma_inst_shape=_MMA_INST,
            mma_tiler=_MMA_TILER,
            cluster_shape_mn=_CLUSTER,
            raster_along_m=False,
            topK=top_k,
            use_fused_finalize=False,
        )
        compiled = cute.compile(
            kernel.wrapper,
            *ptrs,
            *sizes,
            tile_size=_TILE,
            scaling_vector_size=_SF_VEC,
            max_active_clusters=get_max_active_clusters(_CLUSTER[0] * _CLUSTER[1]),
            stream=stream,
        )
        _FC2_KERNELS[key] = compiled
    compiled(*ptrs, *sizes, stream=stream)


# Read-only [0, 1, ...) int32 index shared by every layer: the expanded rows are already in (token, slot) order.
_IDENTITY: dict = {}
# Default capacity in tokens of the buffers below (the 32K prefill pass); a larger eager call grows them.
_DEFAULT_ROWS = 32768
# GEMM2's expanded rows [tokens * 10, H] bf16, one buffer for every layer. The dual-stream glue record_streams the
# deferred output, so a fresh tensor per layer is not reused inside a graph capture (48 x 1.68 GB at 32K rows).
# Sharing is safe: a layer's finalize is queued on the main stream before the next layer's experts start, and the
# routed stream waits on the main stream first.
_EXPANDED: dict = {}
# Buffers a captured graph may still address; never freed.
_RETIRED: list = []


def _identity_index(n: int, device: torch.device) -> torch.Tensor:
    cached = _IDENTITY.get(device)
    if cached is not None and cached.numel() >= n:
        return cached[:n]
    if torch.cuda.is_current_stream_capturing():
        # Filled by the graph's own replay; never cached outside it.
        return torch.arange(n, dtype=torch.int32, device=device)
    cached = torch.arange(max(n, _DEFAULT_ROWS * TOP_K), dtype=torch.int32, device=device)
    _IDENTITY[device] = cached
    return cached[:n]


def _expanded_rows(n: int, device: torch.device) -> torch.Tensor:
    cached = _EXPANDED.get(device)
    if cached is not None and cached.shape[0] >= n:
        return cached[:n]
    if torch.cuda.is_current_stream_capturing():
        # Correct, but each layer then keeps its own buffer inside the capture (the r1 memory fault).
        _log_once(
            f"expanded-capture-{n}",
            logging.WARNING,
            "%s: GEMM2 expanded rows allocated per call inside a graph capture (%d rows, no shared buffer yet); "
            "graph memory grows by %.2f GB per MoE layer",
            ENV,
            n,
            n * HIDDEN * 2 / 1e9,
        )
        return torch.empty((n, HIDDEN), dtype=torch.bfloat16, device=device)
    if cached is not None:
        _RETIRED.append(cached)
    cached = torch.empty((max(n, _DEFAULT_ROWS * TOP_K), HIDDEN), dtype=torch.bfloat16, device=device)
    _EXPANDED[device] = cached
    _log_once(
        f"expanded-{cached.shape[0]}",
        logging.INFO,
        "%s: GEMM2 expanded-row buffer of %d rows (%.2f GB), shared by all layers",
        ENV,
        cached.shape[0],
        cached.numel() * cached.element_size() / 1e9,
    )
    return cached[:n]


def _routed_input(dispatch_output, input_scale_quant: torch.Tensor):
    """The routed input's NVFP4 linear-layout quant: the one the glue already made, else the runner's own call."""
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        Nvfp4RoutedInputPreQuant,
        quantize_hidden_states_fp4,
    )

    hidden_states = dispatch_output.hidden_states
    pre = dispatch_output.hidden_states_pre_quant
    if isinstance(pre, Nvfp4RoutedInputPreQuant):
        if pre.ready is not None:
            torch.cuda.current_stream().wait_event(pre.ready)
        x_fp4, x_sf = pre.x_fp4, pre.x_sf
    elif pre is not None:
        x_fp4, x_sf = pre  # the router front's Nvfp4RouterInput(packed, scales)
    else:
        x_fp4, x_sf = quantize_hidden_states_fp4(hidden_states, input_scale_quant)
    m = hidden_states.shape[0]
    return x_fp4.reshape(m, HIDDEN // 2), x_sf.view(torch.float8_e4m3fn).reshape(m, HIDDEN // _SF_VEC)


def run_routed_experts(dispatch_output, quant_info):
    """The routed MoE of one call: a FlashInferTrtllmDeferredFinalizeOutput when the caller defers the finalize,
    else the finalized [M, H] bf16 rows (shared expert not added), as the trtllm-gen call returns them."""
    from flashinfer import ActivationType
    from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
        blockscaled_contiguous_gather_grouped_gemm_act_fusion,
    )
    from flashinfer.fused_moe.cute_dsl.moe_utils import moe_sort

    from sglang.kernels.ops.moe import q38_prefill_routing
    from sglang.kernels.ops.moe.moe_finalize_fuse_shared import moe_finalize_fuse_shared
    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        FlashInferTrtllmDeferredFinalizeOutput,
        is_deferred_finalize_enabled,
        trtllm_moe_enable_pdl,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    weights: CutePrefillWeights = quant_info.cute_prefill
    topk_output = dispatch_output.topk_output
    router_logits = topk_output.router_logits if TopKOutputChecker.format_is_bypassed(topk_output) else None
    if (
        router_logits is None
        or dispatch_output.hidden_states_scale is not None
        or topk_output.topk_config.correction_bias is not None
        or not q38_prefill_routing.covered(router_logits)
    ):
        raise RuntimeError(f"{ENV}: the CuTe-DSL routed experts take bf16 activations and [M, 512] bf16 router logits")
    started = time.perf_counter() if not _FC2_KERNELS else None
    m = router_logits.shape[0]
    pdl = trtllm_moe_enable_pdl(m)

    # Before the first kernel of the call: a lazily built index or buffer must precede the non-PDL GEMM2.
    identity = _identity_index(m * TOP_K, router_logits.device)
    expanded = _expanded_rows(m * TOP_K, router_logits.device)
    x_fp4, x_sf = _routed_input(dispatch_output, quant_info.w13_input_scale_quant)
    ids, expert_weights = q38_prefill_routing.route(router_logits)
    scales_f32 = expert_weights.float()  # moe_sort and the Rubin GEMM2 take fp32 router scales (unused unfused)
    tile_to_expert, tile_to_limit, _, permuted_to_expanded, _, num_tiles = moe_sort(
        token_selected_experts=ids,
        token_final_scales=scales_f32,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=0,
        num_local_experts=NUM_EXPERTS,
        tile_tokens_dim=_TILE,
        enable_pdl=pdl,
    )
    inter, inter_sf = blockscaled_contiguous_gather_grouped_gemm_act_fusion(
        a=x_fp4,
        b=weights.w13,
        a_scale=x_sf,
        b_scale=weights.w13_sf,
        alpha=weights.w1_alpha,
        tile_idx_to_expert_idx=tile_to_expert,
        tile_idx_to_mn_limit=tile_to_limit,
        token_id_mapping=permuted_to_expanded,
        num_non_exiting_tiles=num_tiles,
        global_scale=weights.fc2_input_scale,
        topk=TOP_K,
        a_dtype="float4_e2m1fn",
        b_dtype="float4_e2m1fn",
        c_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        sf_vec_size=_SF_VEC,
        quantize_output=True,
        mma_tiler_mn=_MMA_TILER[:2],
        cluster_shape_mn=_CLUSTER,
        mma_tiler=_MMA_TILER,
        mma_inst_shape=_MMA_INST,
        enable_pdl=pdl,
        activation_type=ActivationType.Swiglu.value,
        gated=True,
    )
    _fc2_unfused(
        a=inter,
        a_scale=inter_sf,
        weights=weights,
        tile_idx_to_expert_idx=tile_to_expert,
        num_non_exiting_tiles=num_tiles,
        tile_idx_to_mn_limit=tile_to_limit,
        permuted_idx_to_expanded_idx=permuted_to_expanded,
        token_final_scales=scales_f32,
        out=expanded,
    )
    deferred = is_deferred_finalize_enabled()
    if started is not None:
        _log_once(
            "engaged",
            logging.INFO,
            "%s engaged: routed experts on the Rubin CuTe-DSL gather GEMM1 + SwiGLU and the deterministic GEMM2 "
            "(FlashInfer 0.7.2 + #6307 + #6950), served finalize through an identity index; first call rows=%d "
            "deferred=%s pdl=%s, %.1f s including JIT builds (mostly FlashInfer's moe_utils for moe_sort)",
            ENV,
            m,
            deferred,
            pdl,
            time.perf_counter() - started,
        )
    if deferred:
        return FlashInferTrtllmDeferredFinalizeOutput(
            gemm2_out=expanded,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=identity,
            top_k=TOP_K,
        )
    return moe_finalize_fuse_shared(expanded, identity, expert_weights, None, TOP_K, enable_pdl=pdl)
