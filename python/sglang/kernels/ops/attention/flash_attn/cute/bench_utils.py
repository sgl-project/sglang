"""Shared benchmark utilities: attention_ref, cuDNN helpers, flops calculation."""

import math

import torch

try:
    import cudnn
except ImportError:
    cudnn = None


# ── FLOPS calculation ────────────────────────────────────────────────────────


def flops(
    batch,
    nheads,
    seqlen_q,
    seqlen_k,
    headdim,
    headdim_v,
    causal=False,
    window_size=(None, None),
    has_qv=False,
):
    if causal:
        avg_seqlen = (max(0, seqlen_k - seqlen_q) + seqlen_k) / 2
    else:
        if window_size == (None, None):
            avg_seqlen = seqlen_k
        else:
            row_idx = torch.arange(seqlen_q, device="cuda")
            col_left = (
                torch.maximum(
                    row_idx + seqlen_k - seqlen_q - window_size[0], torch.tensor(0)
                )
                if window_size[0] is not None
                else torch.zeros_like(row_idx)
            )
            col_right = (
                torch.minimum(
                    row_idx + seqlen_k - seqlen_q + window_size[1],
                    torch.tensor(seqlen_k - 1),
                )
                if window_size[1] is not None
                else torch.full_like(row_idx, seqlen_k - 1)
            )
            avg_seqlen = (col_right - col_left + 1).float().mean().item()
    eff_headdim = headdim + headdim_v if has_qv else headdim
    return batch * nheads * 2 * seqlen_q * avg_seqlen * (eff_headdim + headdim_v)


# ── Bandwidth calculation ────────────────────────────────────────────────────


def bandwidth_fwd_bytes(
    batch,
    nheads,
    nheads_kv,
    seqlen_q,
    seqlen_k,
    headdim,
    headdim_v,
    dtype_bytes=2,
    has_qv=False,
):
    """HBM traffic for one attention pass: read Q,K,V + write O."""
    q = batch * nheads * seqlen_q * headdim
    qv = batch * nheads * seqlen_q * headdim_v if has_qv else 0
    k = batch * nheads_kv * seqlen_k * headdim
    v = batch * nheads_kv * seqlen_k * headdim_v
    o = batch * nheads * seqlen_q * headdim_v
    return (q + qv + k + v + o) * dtype_bytes


def bandwidth_bwd_bytes(
    batch, nheads, nheads_kv, seqlen_q, seqlen_k, headdim, headdim_v, dtype_bytes=2
):
    """HBM traffic for one attention pass: read Q,K,V,dO + write dQ,dK,dV."""
    q = batch * nheads * seqlen_q * headdim
    k = batch * nheads_kv * seqlen_k * headdim
    v = batch * nheads_kv * seqlen_k * headdim_v
    do = batch * nheads * seqlen_q * headdim_v
    dq = q
    dk = k
    dv = v
    return (q + k + v + do + dq + dk + dv) * dtype_bytes


# ── Reference attention ─────────────────────────────────────────────────────

_attention_ref_mask_cache = {}


def attention_ref(q, k, v, causal=False):
    """Standard attention reference implementation.

    Args:
        q, k, v: (batch, seqlen, nheads, headdim) tensors.
        causal: whether to apply causal mask.
    """
    softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    if causal:
        if scores.shape[-2] not in _attention_ref_mask_cache:
            mask = torch.tril(
                torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool),
                diagonal=0,
            )
            _attention_ref_mask_cache[scores.shape[-2]] = mask
        else:
            mask = _attention_ref_mask_cache[scores.shape[-2]]
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhts,bshd->bthd", attn, v)


# ── cuDNN graph helpers ─────────────────────────────────────────────────────

_TORCH_TO_CUDNN_DTYPE = {
    torch.float16: "HALF",
    torch.bfloat16: "BFLOAT16",
    torch.float32: "FLOAT",
    torch.int32: "INT32",
    torch.int64: "INT64",
}


def _build_cudnn_graph(io_dtype, tensors, build_fn):
    """Build a cuDNN graph.  Returns (graph, variant_pack, workspace)."""
    assert cudnn is not None, "cuDNN is not available"
    cudnn_dtype = getattr(cudnn.data_type, _TORCH_TO_CUDNN_DTYPE[io_dtype])
    graph = cudnn.pygraph(
        io_data_type=cudnn_dtype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    graph_tensors = {name: graph.tensor_like(t.detach()) for name, t in tensors.items()}
    variant_pack = build_fn(graph, graph_tensors)
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )
    return graph, variant_pack, workspace


def cudnn_fwd_setup(q, k, v, causal=False, window_size_left=None):
    """Build a cuDNN forward SDPA graph.

    Args:
        q, k, v: (batch, nheads, seqlen, headdim) tensors (cuDNN layout).
        causal: whether to apply causal mask.
        window_size_left: sliding window size (None for no window).

    Returns:
        (fwd_fn, o_gpu, stats_gpu) where fwd_fn is a zero-arg callable.
    """
    b, nheads, seqlen_q, headdim = q.shape
    headdim_v = v.shape[-1]
    o_gpu = torch.empty(b, nheads, seqlen_q, headdim_v, dtype=q.dtype, device=q.device)
    stats_gpu = torch.empty(
        b, nheads, seqlen_q, 1, dtype=torch.float32, device=q.device
    )

    def build(graph, gt):
        o, stats = graph.sdpa(
            name="sdpa",
            q=gt["q"],
            k=gt["k"],
            v=gt["v"],
            is_inference=False,
            attn_scale=1.0 / math.sqrt(headdim),
            use_causal_mask=causal or window_size_left is not None,
            sliding_window_length=(
                window_size_left
                if window_size_left is not None and not causal
                else None
            ),
        )
        o.set_output(True).set_dim(o_gpu.shape).set_stride(o_gpu.stride())
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)
        return {gt["q"]: q, gt["k"]: k, gt["v"]: v, o: o_gpu, stats: stats_gpu}

    graph, variant_pack, workspace = _build_cudnn_graph(
        q.dtype, {"q": q, "k": k, "v": v}, build
    )

    def fwd_fn():
        graph.execute(variant_pack, workspace)
        return o_gpu

    return fwd_fn, o_gpu, stats_gpu


def cudnn_bwd_setup(q, k, v, o, g, lse, causal=False, window_size_left=None):
    """Build a cuDNN backward SDPA graph.

    Args:
        q, k, v, o, g, lse: (batch, nheads, seqlen, dim) tensors (cuDNN layout).
        causal: whether to apply causal mask.
        window_size_left: sliding window size (None for no window).

    Returns:
        bwd_fn: zero-arg callable that returns (dq, dk, dv).
    """
    headdim = q.shape[-1]
    dq_gpu, dk_gpu, dv_gpu = (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
    )

    def build(graph, gt):
        dq, dk, dv = graph.sdpa_backward(
            name="sdpa_backward",
            q=gt["q"],
            k=gt["k"],
            v=gt["v"],
            o=gt["o"],
            dO=gt["g"],
            stats=gt["lse"],
            attn_scale=1.0 / math.sqrt(headdim),
            use_causal_mask=causal or window_size_left is not None,
            sliding_window_length=(
                window_size_left
                if window_size_left is not None and not causal
                else None
            ),
            use_deterministic_algorithm=False,
        )
        dq.set_output(True).set_dim(dq_gpu.shape).set_stride(dq_gpu.stride())
        dk.set_output(True).set_dim(dk_gpu.shape).set_stride(dk_gpu.stride())
        dv.set_output(True).set_dim(dv_gpu.shape).set_stride(dv_gpu.stride())
        return {
            gt["q"]: q,
            gt["k"]: k,
            gt["v"]: v,
            gt["o"]: o,
            gt["g"]: g,
            gt["lse"]: lse,
            dq: dq_gpu,
            dk: dk_gpu,
            dv: dv_gpu,
        }

    graph, variant_pack, workspace = _build_cudnn_graph(
        q.dtype,
        {"q": q, "k": k, "v": v, "o": o, "g": g, "lse": lse},
        build,
    )

    def bwd_fn():
        graph.execute(variant_pack, workspace)
        return dq_gpu, dk_gpu, dv_gpu

    return bwd_fn
