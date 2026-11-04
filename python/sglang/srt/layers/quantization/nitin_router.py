"""Opt-in router-only cluster split-K routes from the small-M BF16 candidate."""

from __future__ import annotations

import atexit
from collections import Counter
import hashlib
import json
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_ROUTER = (512, 2560)
ROUTES = {
    **{(m, *_ROUTER): ("cl", (64, 8, 4, 5, 128)) for m in (5, 6, 7, *range(9, 17))},
    **{(m, *_ROUTER): ("cl", (64, 16, 8, 5, 64)) for m in range(17, 33)},
}
# FlashInfer 0.7.2's dense_bf16_gemm_sm100_splitk.py: the "cl" routes call its compile path directly.
_FI_SPLITK_SHA256 = "f5edd4c11d95822c836a363dfa1dd19f7843b4e354e091c24c0acac12f52d9d2"

_routes: Optional[dict] = None
_fi = None  # (module, CUstream ctor)
_compiled: dict = {}
_logged: set = set()
_quiet = False
_observed_buckets: set = set()
_non_graph_calls = Counter()


def flush_non_graph_counts() -> None:
    """Host calls only; CUDA graph replays do not increment these counts."""
    if _non_graph_calls:
        logger.info(
            "Q38_NITIN_ROUTER_COUNT scope=non_graph_calls counts=%s",
            json.dumps(dict(sorted(_non_graph_calls.items()))),
        )


def _observe(x, weight, *, result: str, reason: str, kind: str) -> None:
    """Observe dispatch metadata without device reads, launches or synchronization."""
    if _quiet or tuple(weight.shape) != _ROUTER:
        return
    from sglang.srt.model_executor.forward_context import (
        get_forward_context,
        has_forward_context,
    )
    from sglang.srt.runtime_context import get_parallel

    if not has_forward_context():
        return
    context = get_forward_context()
    forward_mode = context.moe_forward_mode
    if forward_mode is None or context.moe_batch_size is None:
        return
    if forward_mode.is_target_verify():
        mode = "verify"
    elif forward_mode.is_draft_extend_v2():
        mode = "draft"
    elif forward_mode.is_decode():
        mode = "decode"
    elif forward_mode.is_extend():
        mode = "extend"
    else:
        return
    rank = get_parallel().tp_rank
    bs = context.moe_batch_size
    m = x.numel() // x.shape[-1]
    key = (rank, bs, m, mode, kind, result, reason)
    if torch.cuda.is_current_stream_capturing():
        if key not in _observed_buckets:
            _observed_buckets.add(key)
            logger.info(
                "Q38_NITIN_ROUTER_BUCKET rank=%s bs=%s M=%s mode=%s "
                "kind=%s result=%s reason=%s capture=1",
                *key,
            )
    else:
        _non_graph_calls[
            f"rank={rank},bs={bs},M={m},mode={mode},kind={kind},"
            f"result={result},reason={reason}"
        ] += 1


atexit.register(flush_non_graph_counts)


def _cluster_smem(mma_m: int, mma_n: int, split_k: int, stages: int, cta_k: int) -> int:
    """FlashInfer 0.7.2 dense_bf16_gemm_sm100_splitk._smem_bytes with the K tile as a parameter."""

    def align(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    cursor = align(mma_m * cta_k * 2 * stages, 1024) + mma_n * cta_k * 2 * stages
    cursor = align(align(cursor, 8) + 2 * stages * 8 + 3 * 8, 4) + 4
    if split_k == 1:
        return cursor
    return align(align(cursor, 128) + (split_k - 1) * mma_m * mma_n * 4, 8) + 8


def install(device_sm: int, splitk_enabled: bool, cutedsl_available: bool) -> bool:
    """Called by initialize_bf16_gemm_config; returns whether the routes are on."""
    global _routes, _fi
    _routes = None
    if os.environ.get("SGLANG_Q38C_NITIN_ROUTER", "0") != "1":
        return False
    if not (splitk_enabled and cutedsl_available and device_sm == 107):
        logger.warning("Nitin router: needs SM107 and the cutedsl/split-K BF16 backend; off")
        return False
    import cuda.bindings.driver as cuda_driver
    from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as fi_splitk

    with open(fi_splitk.__file__, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != _FI_SPLITK_SHA256:
        logger.warning("Nitin router: FlashInfer split-K source %s is not 0.7.2's; off", digest[:16])
        return False
    for (m, n, k), (kind, tactic) in ROUTES.items():
        if kind == "cl":
            mma_m, mma_n, split_k, stages, cta_k = tactic
            if (k % (split_k * cta_k) or not 1 <= stages <= k // (split_k * cta_k)
                    or _cluster_smem(*tactic) > fi_splitk._SMEM_CAPACITY_BYTES):
                logger.warning("Nitin router: route %s at %s is invalid; off", tactic, (m, n, k))
                return False
    _fi = (fi_splitk, cuda_driver.CUstream)
    _routes = ROUTES
    kinds = [r[0] for r in ROUTES.values()]
    logger.info(
        "Nitin router: %d SM107 keys installed (cluster split-K %d, TGV %d)",
        len(kinds), kinds.count("cl"), kinds.count("tgv") + kinds.count("tgvc"),
    )
    return True


def _cluster_out(x_2d: torch.Tensor, weight: torch.Tensor, out: torch.Tensor, tactic) -> None:
    fi_splitk, stream_ctor = _fi
    mma_m, mma_n, split_k, stages, cta_k = tactic
    tensors = fi_splitk._to_cute_swap(x_2d, weight.T, out, None)
    key = (x_2d.get_device(), x_2d.dtype, tactic, tensors[4])
    compiled = _compiled.get(key)
    if compiled is None:
        fi_tactic = fi_splitk.SplitKTactic(mma_m, mma_n, split_k, stages)
        if cta_k == fi_splitk._CTA_K:
            compiled = fi_splitk._get_compiled_splitk_kernel(
                device_index=x_2d.get_device(), dtype=x_2d.dtype, tactic=fi_tactic, use_pdl=True,
                has_bias=False, leading_dims=tensors[4],
            )
        else:
            # The kernel body reads only self.cta_k; FlashInfer's _CTA_K is host-side validation.
            kernel = fi_splitk.SplitKDenseGemmKernel(tactic=fi_tactic, use_pdl=True, has_bias=False)
            kernel.cta_k = cta_k
            with torch.cuda.device(x_2d.device):
                rep = fi_splitk._make_compile_repr_tensors(x_2d.dtype, False, *tensors[4])
                compiled = fi_splitk.cute_ext.compile(
                    fi_splitk._bmm_no_bias, kernel, *rep[:3],
                    stream_ctor(torch.cuda.current_stream().cuda_stream),
                )
        _compiled[key] = compiled
    compiled(*tensors[:3], stream_ctor(torch.cuda.current_stream(x_2d.device).cuda_stream))


def gemm(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    """One kernel for a listed (M, N, K) with the stock bits, or None (the stock route runs)."""
    k = x.shape[-1]
    m = x.numel() // k
    route = _routes.get((m, weight.shape[0], weight.shape[1]))
    if route is None or not (
        x.dtype == weight.dtype == torch.bfloat16
        and x.is_contiguous() and weight.is_contiguous()
    ):
        _observe(
            x, weight, result="NATIVE",
            reason="shape_not_listed" if route is None else "dtype_or_layout",
            kind="native" if route is None else route[0],
        )
        return None
    x_2d = x.view(-1, k)
    if not (x_2d.stride() == (k, 1) and weight.stride() == (k, 1)
            and x_2d.data_ptr() % 32 == 0 and weight.data_ptr() % 32 == 0):
        _observe(x, weight, result="NATIVE", reason="stride_or_alignment", kind=route[0])
        return None
    kind, tactic = route
    out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
    _cluster_out(x_2d, weight, out, tactic)
    _observe(x, weight, result="ACTIVE", reason="listed_route", kind=kind)
    if not _quiet and (kind, tuple(weight.shape)) not in _logged:
        _logged.add((kind, tuple(weight.shape)))
        logger.info("Nitin router: %s m=%d n=%d k=%d route=%s", kind, m, weight.shape[0], k, tactic)
    return out.view(*x.shape[:-1], weight.shape[0])


def precompile() -> None:
    """JIT-compile every key at load, without the served-path engagement lines."""
    global _quiet
    if _routes is None:
        return
    device = torch.cuda.current_device()
    weights = {}
    _quiet = True
    try:
        for (m, n, k) in sorted(_routes):
            if (n, k) not in weights:
                weights[(n, k)] = torch.zeros(n, k, dtype=torch.bfloat16, device=device)
            gemm(torch.zeros(m, k, dtype=torch.bfloat16, device=device), weights[(n, k)])
    finally:
        _quiet = False
