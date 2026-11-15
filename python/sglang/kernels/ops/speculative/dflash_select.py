"""DFlash2 draft selection after the lm_head projection (local top-K, cross-rank merge, lattice, walk).

``dflash_local_topk`` (K1) publishes one rank's sorted top-K words of each logits row into the TP group's push plane
(the producer half of a round); ``dflash_select_walk`` (K2) polls every rank's words, merges them, scores the candidate
lattice, walks it and commits the round (protocol notes in ``csrc/speculative/dflash_select_walk.cuh``). The plane is
the push plane of the TP group's shared symmetric workspace (``get_dflash_select_workspace_request``), with a multicast
address. ``dflash_select_reference`` is the torch chain the kernels replace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

    from sglang.kernels.ops.communication.all_reduce import PushPlane
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        WorkspaceRequest,
    )


def _get_local_topk_min_blocks(num_rows: int) -> int:
    """The fewest CTAs per SM that keep every row's CTA resident in one wave (a second wave costs 5-8 us after the
    GEMM on GB300, measured in the lm_head GEMM -> K1 chain), at most 2: rows past 2 x the SM count take another wave
    (not tuned). The kernel derives its warp count from the shard width and rejects unsupported widths."""
    return min(max(-(-num_rows // _get_num_sms()), 1), 2)


@cache_once
def _get_num_sms() -> int:
    return current_arch().num_sms


def _check_arch() -> None:
    if not current_arch().is_sm(10):
        raise RuntimeError("dflash_select kernels require SM100 (B200 / B300 / GB300)")


@cache_once
def _jit_local_topk_module(k: int, width: int, min_blocks: int) -> Module:
    _check_arch()
    args = make_cpp_args(k, width, min_blocks, is_arch_support_pdl())
    return load_jit(
        "dflash_local_topk",
        *args,
        cuda_files=["speculative/dflash_local_topk.cuh"],
        cuda_wrappers=[("run", f"sglang::DFlashLocalTopKKernel<{args}>::run")],
    )


@cache_once
def _jit_select_walk_module(k: int, gamma: int) -> Module:
    _check_arch()
    args = make_cpp_args(k, gamma, is_arch_support_pdl())
    return load_jit(
        "dflash_select_walk",
        *args,
        cuda_files=["speculative/dflash_select_walk.cuh"],
        cuda_wrappers=[("run", f"sglang::DFlashSelectWalkKernel<{args}>::run")],
    )


def get_dflash_select_workspace_request(max_rows: int, k: int) -> WorkspaceRequest:
    """The push slot K1 and K2 need on the TP group's symmetric workspace (``request_group_workspace``)."""
    from sglang.srt.distributed.device_communicators.symm_workspace import (
        WorkspaceRequest,
    )

    # One u32 word per candidate.
    return WorkspaceRequest(owner="dflash_select", push_slot_bytes=max_rows * k * 4)


def dflash_local_topk(
    logits: torch.Tensor,
    plane: PushPlane,
    k: int,
    *,
    _min_blocks: Optional[int] = None,
) -> None:
    """Publishes the top-k of each logits row [N, W] bf16 (this rank's shard) as int32 words (order16(value) << 16 |
    position + 1), sorted (value desc, position asc), -0.0 counted as +0.0, into slot (phase, plane.rank) of every
    rank's push workspace. Rows with more than 128 elements at or above the kernel's threshold (heavy ties) take a slow
    exact path in the same kernel. Test hook: _min_blocks overrides the dispatched CTAs per SM."""
    num_rows, width = logits.shape
    min_blocks = _min_blocks or _get_local_topk_min_blocks(num_rows)
    _jit_local_topk_module(k, width, min_blocks).run(logits, plane)


def dflash_select_walk(
    plane: PushPlane,
    k: int,
    pred_codebook: torch.Tensor,
    succ_codebook: torch.Tensor,
    hp: torch.Tensor,
    anchor_ids: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    uniforms: torch.Tensor,
    output_multiplier: float,
    out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consumes the round every rank's K1 published on ``plane`` (and commits it). Returns tokens int64 [bs, gamma],
    candidate_ids int64 [bs, gamma, K], q fp32 [bs, gamma, K]."""
    bs, gamma = uniforms.shape
    if out is None:
        dev = uniforms.device
        out = (
            torch.empty((bs, gamma), dtype=torch.int64, device=dev),
            torch.empty((bs, gamma, k), dtype=torch.int64, device=dev),
            torch.empty((bs, gamma, k), dtype=torch.float32, device=dev),
        )
    tokens, candidate_ids, q = out
    _jit_select_walk_module(k, gamma).run(
        plane,
        pred_codebook,
        succ_codebook,
        hp,
        anchor_ids,
        temperatures,
        greedy_mask,
        uniforms,
        tokens,
        candidate_ids,
        q,
        float(output_multiplier),
        pred_codebook.shape[0] // plane.world_size,
    )
    return tokens, candidate_ids, q


# ----------------------------------------------------------------------------------------------------------------
# Reference


def reference_candidates(
    shards: list[torch.Tensor], k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact global top-K of the concatenated shards, ordered (value desc, id asc): ids int64, values fp32."""
    full = torch.cat(shards, dim=-1).float()
    full = torch.where(full == 0, torch.zeros_like(full), full)  # -0.0 == +0.0
    vals, idx = torch.sort(full, dim=-1, descending=True, stable=True)
    return idx[:, :k].contiguous(), vals[:, :k].contiguous()


def reference_lattice(
    pred_codebook: torch.Tensor,
    succ_codebook: torch.Tensor,
    candidate_ids: torch.Tensor,  # [bs, gamma, K]
    unary: torch.Tensor,  # [bs, gamma, K] fp32
    hp: torch.Tensor,  # [bs, gamma, R] bf16
    anchor_ids: torch.Tensor,  # [bs]
) -> torch.Tensor:
    """score[b, e, i, c] = unary[b, e, c] + bf16(sum_r bf16(A[prev_i, r] * hp[r]) * B[c, r]), fp32 accumulation."""
    bs, gamma, k = candidate_ids.shape
    prev = torch.cat(
        [anchor_ids[:, None, None].expand(-1, 1, k), candidate_ids[:, :-1]], dim=1
    )
    pred = (pred_codebook[prev] * hp[:, :, None]).float()  # bf16 products
    succ = succ_codebook[candidate_ids].float()
    dots = torch.einsum("blpr,blcr->blpc", pred.double(), succ.double()).float()
    return unary[:, :, None] + dots.bfloat16().float()


def reference_walk(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The walk of ``selector_walk_triton`` in torch: tokens, q rows and the path indices."""
    bs, gamma, k = candidate_ids.shape
    tokens = torch.empty((bs, gamma), dtype=torch.int64, device=scores.device)
    q = torch.empty((bs, gamma, k), dtype=torch.float32, device=scores.device)
    path = torch.empty((bs, gamma), dtype=torch.int64, device=scores.device)
    prev = torch.zeros(bs, dtype=torch.int64, device=scores.device)
    ar = torch.arange(bs, device=scores.device)
    for e in range(gamma):
        row = scores[ar, e, prev]  # [bs, K]
        probs = torch.softmax(row / temperatures[:, None], dim=-1)
        sampled = (uniforms[:, e : e + 1] >= probs.cumsum(-1)).sum(-1).clamp_max(k - 1)
        greedy_idx = row.argmax(-1)
        idx = torch.where(greedy_mask, greedy_idx, sampled)
        q[:, e] = torch.where(
            greedy_mask[:, None],
            torch.nn.functional.one_hot(idx, k).float(),
            probs,
        )
        tokens[:, e] = candidate_ids[ar, e, idx]
        path[:, e] = idx
        prev = idx
    return tokens, q, path


def dflash_select_reference(
    shards: list[torch.Tensor],  # TP x [bs * gamma, W] bf16
    pred_codebook: torch.Tensor,
    succ_codebook: torch.Tensor,
    hp: torch.Tensor,  # [bs * gamma, R] bf16
    anchor_ids: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    uniforms: torch.Tensor,
    output_multiplier: float,
    k: int,
):
    bs, gamma = uniforms.shape
    ids, vals = reference_candidates(shards, k)
    ids = ids.view(bs, gamma, k)
    unary = (vals * output_multiplier).view(bs, gamma, k)
    scores = reference_lattice(
        pred_codebook, succ_codebook, ids, unary, hp.view(bs, gamma, -1), anchor_ids
    )
    tokens, q, path = reference_walk(ids, scores, uniforms, temperatures, greedy_mask)
    return tokens, ids, q, scores, path
