"""Prepared expert-tile routing metadata for FlashInfer's native FP4 MoE body.

Activations enter the native expert path unchanged; only routing setup is
replaced. Routes are sorted by expert and each distinct expert owns one padded
CTA tile, so this covers token counts up to the tile size. Refresh metadata on
every call because expert kernels may rewrite the expanded-to-permuted map.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import get_device_sm


@triton.jit
def _max_scan(a, b):
    return tl.maximum(a, b)


@triton.jit
def prepare_expert_tile_metadata(
    IDS,
    W,
    TOTAL,
    MAP,
    PERM,
    EW,
    CTA,
    LIMIT,
    COUNT,
    TILE: tl.constexpr,
    ROUTES: tl.constexpr,
    BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    PACKED: tl.constexpr,
):
    r = tl.arange(0, BLOCK)
    valid = r < ROUTES
    raw = tl.load(IDS + r, mask=valid, other=0)
    if PACKED:
        ids = raw >> 16
        weights = (raw & 65535).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    else:
        ids = raw
        weights = tl.load(W + r, mask=valid, other=0.0)
    ids = tl.where(valid, ids, 1 << 20)
    keys = tl.sort(ids * BLOCK + r, descending=False)
    expert = keys // BLOCK
    original = keys % BLOCK
    prev = tl.gather(expert, tl.maximum(r - 1, 0), 0)
    nxt = tl.gather(expert, tl.minimum(r + 1, BLOCK - 1), 0)
    first = valid & ((r == 0) | (expert != prev))
    last = valid & ((r == ROUTES - 1) | (expert != nxt))
    rank = tl.cumsum(first.to(tl.int32), 0) - 1
    start = tl.associative_scan(tl.where(first, r, 0), 0, _max_scan)
    within = r - start
    position = rank * TILE + within
    tl.store(MAP + original, position, valid)
    tl.store(PERM + position, original // TOP_K, valid)
    tl.store(EW + r, weights, valid)
    tl.store(CTA + rank, expert, first)
    tl.store(LIMIT + rank, rank * TILE + within + 1, last)
    unique = tl.sum(first.to(tl.int32), 0)
    tl.store(TOTAL, unique * TILE)
    tl.store(COUNT, unique)


def prepared_routing_sm_supported() -> bool:
    return get_device_sm() in (100, 103, 107)


def _flashinfer_core():
    from flashinfer.fused_moe import core as c

    return c if hasattr(c, "TrtllmMoERoutingMetadataSlot") else None


def _run_prepared(
    c,
    cache_owner,
    prepared_metadata,
    kwargs,
    *,
    hidden_size,
    dtype_act,
    dtype_weights,
    routing_input_mode,
    topk_ids,
    expert_weights,
):
    x = kwargs["hidden_states"]
    num_tokens = x.shape[0]
    top_k = kwargs["top_k"]
    num_experts = kwargs["num_experts"]
    runtime = c.get_trtllm_moe_sm100_module()
    runner = runtime.MoERunner(
        runtime.moe_op,
        top_k=top_k,
        num_local_experts=num_experts,
        dtype_act=dtype_act,
        dtype_weights=dtype_weights,
        fp8_quantization_type=c.Fp8QuantizationType.NoneFp8,
        hidden_size=hidden_size,
        intermediate_size=kwargs["intermediate_size"],
        activation_type=kwargs["activation_type"],
        weight_layout=c.WeightLayout.MajorK,
        use_shuffled_weight=True,
        use_per_token_scaling=False,
        num_experts=num_experts,
        num_fused_shared_experts=0,
    )
    inputs = c.MoeRunnerInputs(
        output=kwargs["output"],
        routing_logits=None,
        topk_ids=topk_ids,
        expert_weights=expert_weights,
        hidden_states=x,
        hidden_states_scale=kwargs["hidden_states_scale"],
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    runner_kwargs = dict(kwargs)
    runner_kwargs.update(
        routing_input_mode=routing_input_mode,
        per_token_scale=None,
        num_fused_shared_experts=0,
        norm_topk_prob=True,
        routing_replay_out=None,
    )
    tuning = runner._make_tuning_config(
        inputs,
        tune_max_num_tokens=num_tokens,
        routing_input_mode=routing_input_mode,
        use_cold_l2_cache=True,
        use_cuda_graph=True,
    )
    _, tactic = c.AutoTuner.get().choose_one(
        "flashinfer::trtllm_fp4_block_scale_moe",
        [runner],
        tuning,
        inputs.to_list(),
        **runner_kwargs,
    )
    if tactic == -1 or tuple(tactic) == (-1, -1):
        tactic = (8, -1)
    tile = int(tactic[0])
    # Only the eight-token tile has been validated with deferred K3 outputs.
    if tile != 8 or num_tokens > tile:
        return None
    key = (bool(kwargs["do_finalize"]), num_tokens, tuple(tactic), x.device)
    cache = getattr(cache_owner, "_prepared_fp4_moe", None)
    if cache is None:
        cache = cache_owner._prepared_fp4_moe = {}
    if key not in cache:

        def zeros(n):
            return torch.zeros(n, device=x.device, dtype=torch.int32)

        routes = num_tokens * top_k
        metadata = [
            zeros(1),
            zeros(routes),
            zeros(routes * tile + 1),
            torch.empty_like(expert_weights),
            zeros(2 * num_experts),
            zeros(num_experts),
            zeros(routes),
            zeros(routes),
            zeros(1),
        ]
        body = runner.forward(
            inputs.to_list(),
            tactic=tactic,
            do_preparation=True,
            da_routing_metadata=metadata,
            **runner_kwargs,
        )
        cache[key] = (metadata, body)
    metadata, body = cache[key]
    if prepared_metadata is not None:
        metadata = prepared_metadata
    else:
        prepare_expert_tile_metadata[(1,)](
            topk_ids,
            expert_weights,
            *[metadata[i] for i in (0, 1, 2, 3, 6, 7, 8)],
            TILE=tile,
            ROUTES=num_tokens * top_k,
            BLOCK=triton.next_power_of_2(num_tokens * top_k),
            TOP_K=top_k,
            PACKED=routing_input_mode == c.RoutingInputMode.PackedPrecomputed,
            num_warps=4,
        )
    result = runner.forward(
        inputs.to_list(),
        tactic=tactic,
        da_routing_metadata=metadata,
        da_body_workspace=body,
        **runner_kwargs,
    )
    if kwargs["do_finalize"]:
        return kwargs["output"]
    return tuple(c._torch_view_of_ffi_tensor(tensor) for tensor in result)


def _k3_prepared_covered(x, kwargs, width):
    """K3 M<=8 MXFP4 expert shape covered by the prepared path."""
    return (
        x.is_cuda
        and x.ndim == 2
        and 1 <= x.shape[0] <= 8
        and x.shape[1] == width
        and kwargs["top_k"] == 16
        and kwargs["num_experts"] == kwargs["local_num_experts"] == 896
        and kwargs["local_expert_offset"] == 0
        and kwargs["intermediate_size"] == 384
        and (not kwargs["do_finalize"] or x.shape[0] > 1)
        and prepared_routing_sm_supported()
    )


def try_prepared_k3_mxfp4(cache_owner, prepared_metadata=None, **kwargs):
    """Return native output/deferred tensors on the covered M<=8 MXFP4 path, or None."""
    x = kwargs["hidden_states"]
    if not (
        _k3_prepared_covered(x, kwargs, 3584)
        and x.dtype == torch.float8_e4m3fn
        and kwargs["topk_ids"].shape == (x.shape[0], 16)
    ):
        return None
    c = _flashinfer_core()
    if c is None:
        return None
    return _run_prepared(
        c,
        cache_owner,
        prepared_metadata,
        kwargs,
        hidden_size=3584,
        dtype_act=c.DtypeTrtllmGen.MxE4m3,
        dtype_weights=c.DtypeTrtllmGen.MxE2m1,
        routing_input_mode=c.RoutingInputMode.PackedPrecomputed,
        topk_ids=kwargs["topk_ids"],
        expert_weights=torch.empty(
            (x.shape[0], 16), device=x.device, dtype=torch.bfloat16
        ),
    )
