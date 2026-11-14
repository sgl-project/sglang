# ruff: noqa: C408, SIM102
"""FlyDSL Muse Spark MoE: same contract as ``muse_moe_triton``.

BF16 and Quark MXFP4 share one Python entry (``w13.dtype == uint8`` selects MXFP4)
and keep Triton / checkpoint weight layouts — no aiter preshuffle. Alignment is
reused from the Triton module. Grouped GEMM and finalize (per-expert RMSNorm
then weighted sum) are FlyDSL kernels.
"""

import functools

import torch

from sglang.srt.rocm_muse_spark.muse_moe_triton import moe_align_block_size

# Pre-fusion Triton tiles. Do not copy muse_moe_triton._pick_config — decode-fusion
# returned mid/large MXFP4 to BLOCK_K=256 / BLOCK_N=64, which FlyDSL did not sweep.
_SMALL_M = dict(
    BLOCK_M=16,
    BLOCK_N=64,
    BLOCK_K=64,
    GROUP_M=1,
    num_warps=4,
    num_stages=2,
    waves_per_eu=2,
)
_MID_M = dict(BLOCK_M=16, BLOCK_N=128, BLOCK_K=64, GROUP_M=4, num_warps=4, num_stages=2)
_LARGE_M = dict(
    BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=2
)
_SMALL_M_FP4 = dict(
    BLOCK_M=16, BLOCK_N=64, BLOCK_K=128, GROUP_M=1, num_warps=4, num_stages=2
)
_MID_M_FP4 = dict(
    BLOCK_M=16, BLOCK_N=128, BLOCK_K=128, GROUP_M=4, num_warps=4, num_stages=2
)
_LARGE_M_FP4 = dict(
    BLOCK_M=64, BLOCK_N=128, BLOCK_K=128, GROUP_M=8, num_warps=8, num_stages=2
)


def _require_flydsl():
    try:
        import flydsl.expr as fx
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    except ImportError as exc:
        raise RuntimeError(
            "MUSE_MOE_BACKEND=flydsl requires FlyDSL and aiter.ops.flydsl"
        ) from exc
    return fx, _run_compiled


def _pick_flydsl_config(M: int, topk: int, E: int, mxfp4: bool) -> dict:
    rows = M * topk
    table = (
        {"small": _SMALL_M_FP4, "mid": _MID_M_FP4, "large": _LARGE_M_FP4}
        if mxfp4
        else {"small": _SMALL_M, "mid": _MID_M, "large": _LARGE_M}
    )
    if rows <= 64:
        cfg = dict(table["small"])
    elif rows <= 1024:
        cfg = dict(table["mid"])
    else:
        cfg = dict(table["large"])
    # BF16 benefits from twice as many N workgroups throughout the decode
    # range. MXFP4's vectorized K transpose prefers the wider N tile.
    if not mxfp4 and M * topk <= 1024:
        cfg["BLOCK_N"] = 64
    if M <= 4:
        # Conc=4 has eight local routes on the slowest EP rank in the common
        # case. The narrower tile keeps that imbalanced rank below Triton.
        cfg["BLOCK_N"] = 32
        if not mxfp4:
            # Four K partitions turn the latency-bound BF16 decode GEMM into
            # enough independent work to fill the device. Once K is split,
            # one-wave N16 groups outperform the wider unsplit decode tile.
            cfg["BLOCK_N"] = 16
            cfg["FLYDSL_K_UNROLL"] = 4
            cfg["FLYDSL_SPLIT_K"] = 4
    if not mxfp4 and 4 < M <= 16:
        # BF16 has no unpack work to cover HBM latency.  A K64 gate/up tile
        # reaches the streaming roof while the down projection benefits from
        # the lower loop overhead of K128.  Keep the overrides stage-local:
        # using either reduction tile for both GEMMs loses on imbalanced EP
        # ranks, which set decode latency.
        cfg["BLOCK_N"] = 64
        cfg["num_warps"] = 4
        cfg["GEMM1_FLYDSL_K_UNROLL"] = 2
        cfg["GEMM2_FLYDSL_K_UNROLL"] = 4
    if not mxfp4 and 32 < M <= 128:
        # Verify widths at EP4 (64 local experts), six routings each: 33..64 rows -9% mean / never slower than the
        # mid tile (BM32 BN64 BK64, K unroll 2 / 4); 65..128 rows -9% with the narrow BM16 BN32 BK128 tile.
        if M <= 64:
            cfg.update(
                BLOCK_M=32,
                BLOCK_N=64,
                BLOCK_K=64,
                num_warps=4,
                GEMM1_FLYDSL_K_UNROLL=2,
                GEMM2_FLYDSL_K_UNROLL=4,
            )
        else:
            cfg.update(
                BLOCK_M=16,
                BLOCK_N=32,
                BLOCK_K=128,
                num_warps=4,
                GEMM1_FLYDSL_K_UNROLL=2,
                GEMM2_FLYDSL_K_UNROLL=2,
            )
    if M * topk > 1024:
        use_wide_bf16_m = not mxfp4 and M >= 4096
        cfg["num_warps"] = 8 if use_wide_bf16_m else 4
        cfg["BLOCK_M"] = 32 if mxfp4 and M <= 1024 else (128 if use_wide_bf16_m else 64)
        cfg["FLYDSL_K_UNROLL"] = (2 if M >= 4096 else 4) if mxfp4 else 1
    if mxfp4 and M <= 1024:
        # Through M=1024 an MXFP4 launch is pure weight streaming: alignment
        # gives every local expert its own BLOCK_M tile, so no tile choice buys
        # weight reuse and the tile is instead picked to amortize everything
        # that is *not* the weight stream. Two waves per workgroup halve the
        # activation fragments that all waves in a workgroup otherwise re-load
        # identically, the wider N tile pays for those loads with more weight
        # columns, and the 512-element K tile halves both the trip count and
        # the E8M0 fetches per column. BF16 wants the opposite (four waves):
        # with no FP4 unpack to overlap it needs the waves to hide latency.
        # End-to-end cost is set by the slowest EP rank, so the N tile is chosen
        # on the worst routing rather than the average one. Up to M=16 the
        # narrower tiles stay close on one local route per token and hold up far
        # better when a rank draws four (BN64 vs BN128 at M=16: +18% vs +40%);
        # from M=32 up BN128 wins on every routing measured.
        cfg["BLOCK_M"] = 32 if M > 512 else 16
        cfg["BLOCK_N"] = 32 if M <= 4 else (64 if M <= 16 else 128)
        cfg["num_warps"] = 2
        cfg["FLYDSL_K_UNROLL"] = 16
        if M > 128:
            # Down projection re-reads the intermediate once per N tile, and
            # that traffic grows with the routed rows while the weight stream
            # stays fixed, so it is what makes the two stages diverge as M
            # climbs. Halving the re-reads is worth 5-9% from M=256 up; below
            # M=64 the narrower weight stream matters more and this loses.
            cfg["GEMM2_BLOCK_N"] = 256
    return cfg


@functools.lru_cache(maxsize=64)
def _cached_finalize_launch(
    H: int,
    topk: int,
    has_norm: bool,
    has_scale: bool,
    weight_dtype: str,
    split_k: int = 1,
):
    from aiter.ops.flydsl.muse_moe_finalize import build_finalize

    return build_finalize(H, topk, has_norm, has_scale, weight_dtype, split_k)


@functools.lru_cache(maxsize=16)
def _cached_splitk_silu_launch(I: int, split_k: int):
    from aiter.ops.flydsl.muse_moe_finalize import build_splitk_silu

    return build_splitk_silu(I, split_k)


def _launch_gemm(
    a,
    b,
    b_scale,
    c,
    sorted_ids,
    expert_ids,
    num_post_padded,
    N,
    K,
    num_valid,
    topk,
    a_row_div_topk,
    fuse_silu,
    mxfp4,
    cfg,
    split_k=1,
):
    from aiter.ops.flydsl.muse_moe_gemm import launch_muse_moe_gemm

    # Gate/up carries twice as many accumulator fragments as down projection,
    # so one tile is not generally optimal for both stages.  Preserve the
    # shared defaults while allowing the selector (and tuning tools) to
    # specialize either launch without duplicating a full config dictionary.
    stage = "GEMM1" if fuse_silu else "GEMM2"
    stage_cfg = dict(cfg)
    prefix = stage + "_"
    for key, value in cfg.items():
        if key.startswith(prefix):
            stage_cfg[key[len(prefix) :]] = value

    launch_muse_moe_gemm(
        a,
        b,
        b_scale,
        c,
        sorted_ids,
        expert_ids,
        num_post_padded,
        N,
        K,
        num_valid,
        topk,
        a_row_div_topk,
        fuse_silu,
        mxfp4,
        stage_cfg,
        split_k,
    )


def muse_moe_flydsl(
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    local_expert_start: int,
    post_norm_weight: torch.Tensor | None,
    eps: float = 1e-8,
    cfg: dict | None = None,
    post_norm_gain: float = 0.0,
    has_norm: bool = True,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Muse MoE with native-layout FlyDSL grouped GEMMs and finalize."""
    fx, _run_compiled = _require_flydsl()

    if hidden_states.ndim != 2 or topk_ids.ndim != 2 or w13.ndim != 3 or w2.ndim != 3:
        raise ValueError(
            "expected activations/routing [M,K]/[M,topk] and weights [E,N,K]"
        )
    if (
        hidden_states.dtype != torch.bfloat16
        or w13.dtype not in (torch.bfloat16, torch.uint8)
        or w2.dtype != w13.dtype
    ):
        raise ValueError(
            "Muse FlyDSL requires BF16 activations and matching BF16 or packed uint8 weights"
        )
    tensors = [topk_scores, topk_ids, w13, w2]
    tensors += [t for t in (w13_scale, w2_scale, post_norm_weight) if t is not None]
    if not hidden_states.is_cuda or any(
        t.device != hidden_states.device for t in tensors
    ):
        raise ValueError("all Muse MoE tensors must be on the same GPU")
    if not torch.cuda.get_device_properties(
        hidden_states.device
    ).gcnArchName.startswith("gfx950"):
        raise ValueError("Muse FlyDSL MFMA kernels currently require gfx950")
    M, K = hidden_states.shape
    topk = topk_ids.shape[1]
    mxfp4 = w13.dtype == torch.uint8
    E, two_i, kw = w13.shape
    I = two_i // 2
    _, H, iw = w2.shape
    if not E or not topk or not K or not H or two_i % 2 or w2.shape[0] != E:
        raise ValueError("invalid expert, top-k, or GEMM dimensions")
    if not w13.is_contiguous() or not w2.is_contiguous():
        raise ValueError(
            "weights must use contiguous native [E,N,K] / packed [E,N,K/2] layouts"
        )
    if post_norm_weight is not None and has_norm:
        if post_norm_weight.shape != (H,) or post_norm_weight.dtype not in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ):
            raise ValueError("post-norm weight must be BF16/FP16/FP32 with shape [H]")
    # Masked buffer operations use 0x7fffffff as an OOB byte offset.
    if max(M * K, M * topk * I, M * topk * H) * 2 >= 0x7FFFFFFF:
        raise ValueError(
            "Muse FlyDSL activation/scratch buffers must each be smaller than 2 GiB"
        )
    if max(two_i * kw, H * iw) * w13.element_size() >= 0x100000000:
        raise ValueError("each expert weight matrix must be smaller than 4 GiB")
    if mxfp4:
        assert w13_scale is not None and w2_scale is not None, (
            "MXFP4 weights need e8m0 scales"
        )
        assert kw * 2 == K and iw * 2 == I, (
            f"packed shape mismatch: w13 {tuple(w13.shape)} w2 {tuple(w2.shape)} K={K}"
        )
        assert w13_scale.shape == (E, two_i, K // 32) and w2_scale.shape == (
            E,
            H,
            I // 32,
        ), "scale shape mismatch"
        assert w13.stride(2) == 1 and w2.stride(2) == 1, (
            "packed fp4 must be contiguous along K"
        )
        assert hidden_states.dtype == torch.bfloat16, (
            "MXFP4 path expects bf16 activations"
        )
        if any(
            s.dtype != torch.uint8 or not s.is_contiguous()
            for s in (w13_scale, w2_scale)
        ):
            raise ValueError("MXFP4 scales must use contiguous uint8 E8M0 layout")
    else:
        assert K == kw and I == iw, "shape mismatch between w13/w2/hidden"
    if cfg is None:
        cfg = _pick_flydsl_config(M, topk, E, mxfp4)
    assert (
        K % cfg["BLOCK_K"] == 0 and I % cfg["BLOCK_K"] == 0 and cfg["BLOCK_K"] % 32 == 0
    )
    assert I % cfg["BLOCK_N"] == 0 and H % cfg["BLOCK_N"] == 0

    assert topk_ids.shape[0] == M and topk_scores.shape == topk_ids.shape, (
        f"routing/hidden shape mismatch: hidden {tuple(hidden_states.shape)} ids {tuple(topk_ids.shape)} scores {tuple(topk_scores.shape)}"
    )
    if M == 0:
        return hidden_states.new_empty((0, H))
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    ids32 = topk_ids.to(torch.int32).contiguous()
    scores32 = topk_scores.to(torch.float32).contiguous()
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        ids32, E, local_expert_start, cfg["BLOCK_M"]
    )
    num_valid = M * topk
    split_k = int(cfg.get("FLYDSL_SPLIT_K", 1))

    inter = torch.empty(
        (num_valid, I), dtype=hidden_states.dtype, device=hidden_states.device
    )
    if split_k > 1:
        part1 = torch.empty(
            (split_k, num_valid, 2 * I),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        _launch_gemm(
            hidden_states,
            w13,
            w13_scale,
            part1,
            sorted_ids,
            expert_ids,
            num_post,
            I,
            K,
            num_valid,
            topk,
            True,
            True,
            mxfp4,
            cfg,
            split_k,
        )
        stream = fx.Stream(torch.cuda.current_stream(hidden_states.device).cuda_stream)
        _run_compiled(
            _cached_splitk_silu_launch(int(I), split_k),
            part1,
            inter,
            int(num_valid),
            int(num_valid * ((I + 1023) // 1024)),
            stream,
        )
        y = torch.empty(
            (split_k, num_valid, H), dtype=torch.float32, device=hidden_states.device
        )
    else:
        _launch_gemm(
            hidden_states,
            w13,
            w13_scale,
            inter,
            sorted_ids,
            expert_ids,
            num_post,
            I,
            K,
            num_valid,
            topk,
            True,
            True,
            mxfp4,
            cfg,
        )
        y = torch.empty(
            (num_valid, H), dtype=hidden_states.dtype, device=hidden_states.device
        )
    _launch_gemm(
        inter,
        w2,
        w2_scale,
        y,
        sorted_ids,
        expert_ids,
        num_post,
        H,
        I,
        num_valid,
        topk,
        False,
        False,
        mxfp4,
        cfg,
        split_k,
    )

    out = torch.empty((M, H), dtype=hidden_states.dtype, device=hidden_states.device)
    has_scale = has_norm and post_norm_weight is not None
    w_arg = post_norm_weight.contiguous() if has_scale else out
    stream = fx.Stream(torch.cuda.current_stream(hidden_states.device).cuda_stream)
    weight_dtype = {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }[w_arg.dtype]
    launch = _cached_finalize_launch(
        int(H), int(topk), bool(has_norm), bool(has_scale), weight_dtype, split_k
    )
    _run_compiled(
        launch,
        y,
        w_arg,
        scores32,
        ids32,
        out,
        int(local_expert_start),
        int(E),
        float(eps),
        float(post_norm_gain),
        int(M),
        stream,
    )
    return out
