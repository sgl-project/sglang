"""ERNIE FlashIndexer for the DSA indexer: gate params, prefill head selection, Triton kernel."""

from __future__ import annotations

import logging
import os
from typing import Iterable, Iterator, Optional, Tuple

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

# Checkpoint param names, relative to the indexer module.
FLASHINDEXER_GATE_PARAM_SUFFIX = "linear_gate.weight"
FLASHINDEXER_BIAS_PARAM_SUFFIX = "flashindexer_bias"

# Checkpoint subfolder holding the gate tensors, outside the main weight index.
FLASHINDEXER_WEIGHTS_SUBFOLDER = "flashindexer"

# Heads kept per token; 8 is a native DeepGEMM head count, so no padding is needed.
FLASHINDEXER_TOPK_HEADS = 8
_BLOCK_M = 32
_BLOCK_K = 64

_logged = False


def _gate_weights_error(model_path: str, offline: bool = False) -> RuntimeError:
    why = (
        "is not in the local cache, and HF_HUB_OFFLINE forbids fetching it"
        if offline
        else f"has no '{FLASHINDEXER_WEIGHTS_SUBFOLDER}/' subfolder"
    )
    return RuntimeError(
        f"--enable-flashindexer needs the FlashIndexer gate tensors, but "
        f"{model_path!r} {why}. Drop --enable-flashindexer to run all indexer heads."
    )


def _cached_snapshot_roots(
    model_path: str, revision: Optional[str], download_dir: Optional[str]
) -> Iterator[str]:
    """Snapshot dirs already on disk for this repo, the explicit cache first."""
    # local_files_only: a filesystem lookup, so a cached run never hits the Hub.
    from huggingface_hub import snapshot_download

    seen = set()
    for cache_dir in (download_dir, None):
        try:
            root = snapshot_download(
                model_path,
                revision=revision,
                cache_dir=cache_dir,
                allow_patterns=[f"{FLASHINDEXER_WEIGHTS_SUBFOLDER}/*"],
                local_files_only=True,
            )
        except Exception:
            continue
        if root not in seen:
            seen.add(root)
            yield root


def _first_gate_subfolder(roots: Iterable[str]) -> Optional[str]:
    # Return the subfolder: a local path the main index cannot filter files out of.
    for root in roots:
        candidate = os.path.join(root, FLASHINDEXER_WEIGHTS_SUBFOLDER)
        if os.path.isdir(candidate) and os.listdir(candidate):
            return candidate
    return None


def resolve_flashindexer_weights_dir() -> str:
    """Local dir with the gate tensors, since a Hub id cannot be path-joined."""
    from huggingface_hub import constants as hf_constants

    from sglang.srt.model_loader.weight_utils import get_lock
    from sglang.srt.runtime_context import get_model

    model_path = get_model().model_path
    revision = get_model().revision
    download_dir = get_model().download_dir

    if os.path.isdir(model_path):
        found = _first_gate_subfolder([model_path])
        if found is None:
            raise _gate_weights_error(model_path)
        return found

    found = _first_gate_subfolder(
        _cached_snapshot_roots(model_path, revision, download_dir)
    )
    if found is not None:
        return found
    if hf_constants.HF_HUB_OFFLINE:
        raise _gate_weights_error(model_path, offline=True)

    # Fetch just the subfolder into the main cache, under the loader's own lock.
    with get_lock(model_path, download_dir):
        found = _first_gate_subfolder(
            _cached_snapshot_roots(model_path, revision, download_dir)
        )
        if found is not None:
            return found

        from huggingface_hub import snapshot_download

        root = snapshot_download(
            model_path,
            revision=revision,
            cache_dir=download_dir,
            allow_patterns=[f"{FLASHINDEXER_WEIGHTS_SUBFOLDER}/*"],
        )

    found = _first_gate_subfolder([root])
    if found is None:
        raise _gate_weights_error(model_path)
    logger.info("FlashIndexer gate weights downloaded to %s", found)
    return found


def attach_flashindexer_gate(indexer: nn.Module) -> None:
    """Give a DSA indexer the learned FlashIndexer gate and its load-time bias."""
    n_heads, hidden_size = indexer.n_heads, indexer.hidden_size
    if not is_cuda():
        raise ValueError("--enable-flashindexer needs CUDA with the DeepGEMM indexer.")
    # The kernel spans all heads with one tl.arange and tiles hidden by _BLOCK_K.
    if (
        n_heads < FLASHINDEXER_TOPK_HEADS
        or n_heads & (n_heads - 1)
        or hidden_size % _BLOCK_K
    ):
        raise ValueError(
            f"FlashIndexer needs a power-of-two head count >= "
            f"{FLASHINDEXER_TOPK_HEADS} and hidden size divisible by {_BLOCK_K}, "
            f"got n_heads={n_heads}, hidden={hidden_size}."
        )
    indexer.linear_gate = nn.Linear(
        hidden_size, n_heads, bias=False, dtype=torch.bfloat16
    )
    indexer.flashindexer_bias = nn.Parameter(
        torch.zeros(n_heads, dtype=torch.float32), requires_grad=False
    )


def flashindexer_select_heads(
    indexer: nn.Module,
    q: torch.Tensor,
    weights: torch.Tensor,
    hidden: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep the top-FLASHINDEXER_TOPK_HEADS heads per token; a no-op without the FlashIndexer gate."""
    gate = getattr(indexer, "linear_gate", None)
    if gate is None:
        return q, weights
    assert isinstance(hidden, torch.Tensor), "FlashIndexer needs the bf16 hidden"
    global _logged
    if not _logged:
        _logged = True
        logger.info(
            "[DSA FlashIndexer] prefill head selection on: topk_heads=%d",
            FLASHINDEXER_TOPK_HEADS,
        )
    # q may be sliced to the unpadded token count; keep weights and hidden aligned.
    tokens = q.shape[0]
    return flashindexer_select_heads_fused(
        q, weights[:tokens], gate.weight, indexer.flashindexer_bias, hidden[:tokens]
    )


@triton.jit
def _flashindexer_fused_gemm_kernel(
    hidden_ptr,  # [B, HIDDEN] bf16
    weight_ptr,  # [NH, HIDDEN] bf16 (Linear weight, row = head)
    bias_ptr,  # [NH] fp32
    w_gate_ptr,  # [B, NH] fp32 or bf16 (per-row per-head weights; promoted in-kernel)
    q_i32_ptr,  # [B, NH, HDI32] int32
    q_sel_i32_ptr,  # [B, TOPK, HDI32] int32 (out)
    w_sel_ptr,  # [B, TOPK] fp32 or bf16 (out)
    B,
    HIDDEN: tl.constexpr,
    NH: tl.constexpr,
    TOPK: tl.constexpr,
    HDI32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    m_mask = offs_m < B
    offs_n = tl.arange(0, NH)  # [NH]

    # (a) Gate GEMM: gates[BM, NH] = hidden[BM, K] @ weight^T[K, NH].
    acc = tl.zeros((BLOCK_M, NH), dtype=tl.float32)
    for k0 in range(0, HIDDEN, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]
        h_tile = tl.load(
            hidden_ptr + offs_m[:, None] * HIDDEN + offs_k[None, :],
            mask=m_mask[:, None],
            other=0.0,
        )  # [BM,BK] bf16
        # weight[n,k] -> need [BK,NH]: w_tile[k,n] = weight_ptr[n*HIDDEN + k]
        w_tile = tl.load(
            weight_ptr + offs_n[None, :] * HIDDEN + offs_k[:, None]
        )  # [BK,NH] bf16
        acc = tl.dot(h_tile, w_tile, acc)  # [BM,NH]

    # (b) Raw gate logits, no activation: rank by gate + bias, reweight by the gate.
    gates = acc  # [BM,NH]
    scores = gates + tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
    # PDL: wait for compute_q after the gate GEMM, before the w_vec/q_i32 loads.
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    w_vec = tl.load(
        w_gate_ptr + offs_m[:, None] * NH + offs_n[None, :],
        mask=m_mask[:, None],
        other=0.0,
    ).to(tl.float32)  # [BM,NH]; promote in-register so caller can pass bf16

    # (c) TOPK passes of argmax over heads, batched over the BM rows.
    d = tl.arange(0, HDI32)  # [HDI32]
    NEG_INF = -float("inf")
    cur = scores
    for i in range(TOPK):
        idx = tl.argmax(cur, axis=1).to(tl.int32)  # [BM] i-th largest head
        sel_mask = offs_n[None, :] == idx[:, None]  # [BM,NH]
        cur = tl.where(sel_mask, NEG_INF, cur)
        gate_at = tl.sum(tl.where(sel_mask, gates, 0.0), axis=1)  # [BM]
        w_at = tl.sum(tl.where(sel_mask, w_vec, 0.0), axis=1)  # [BM]
        tl.store(w_sel_ptr + offs_m * TOPK + i, w_at * gate_at, mask=m_mask)
        # (d) Gather the picked head's q as one [BM, HDI32] int32 row per pass.
        q_off = offs_m[:, None] * NH * HDI32 + idx[:, None] * HDI32 + d[None, :]
        q_vec = tl.load(q_i32_ptr + q_off, mask=m_mask[:, None], other=0)  # [BM,HDI32]
        out_off = offs_m[:, None] * TOPK * HDI32 + i * HDI32 + d[None, :]
        tl.store(q_sel_i32_ptr + out_off, q_vec, mask=m_mask[:, None])

    # PDL: all stores are done, so let the downstream logits kernel launch.
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def flashindexer_select_heads_fused(
    q: torch.Tensor,
    weights: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    hidden: torch.Tensor,
    topk_heads: int = FLASHINDEXER_TOPK_HEADS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gate GEMM, top-k and q gather in one kernel; q [T, NH, D], weights [T, NH]."""
    tokens, num_heads, head_dim = q.shape
    assert weights.shape == (tokens, num_heads), tuple(weights.shape)
    assert hidden.shape[0] == tokens, tuple(hidden.shape)
    # Fp4Indexer selects on its bf16 q (quantized after), so count bytes, not elems.
    head_dim_i32 = head_dim * q.element_size() // 4
    q_sel_i32 = q.new_empty((tokens, topk_heads, head_dim_i32), dtype=torch.int32)
    # w_sel keeps the caller's weight dtype (fp32 or bf16).
    w_sel = weights.new_empty((tokens, topk_heads))

    # PDL chains compute_q -> FlashIndexer -> logits on SM90+; not under TRITON_INTERPRET.
    pdl_enabled = is_arch_support_pdl() and os.environ.get("TRITON_INTERPRET") != "1"
    pdl_kwargs = (
        {"USE_GDC": True, "launch_pdl": True} if pdl_enabled else {"USE_GDC": False}
    )
    _flashindexer_fused_gemm_kernel[(triton.cdiv(tokens, _BLOCK_M),)](
        hidden.contiguous(),
        gate_weight.contiguous(),
        bias.contiguous(),
        weights.contiguous(),
        q.contiguous().view(torch.int32),
        q_sel_i32,
        w_sel,
        tokens,
        HIDDEN=hidden.shape[-1],
        NH=num_heads,
        TOPK=topk_heads,
        HDI32=head_dim_i32,
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        **pdl_kwargs,
    )
    return q_sel_i32.view(torch.uint8).view(q.dtype), w_sel
