"""DFlash2 fused selection: the target lm_head shard and the selector's hidden projection run as one GEMM
(``sglang.kernels.ops.gemm.lmhead``), whose output feeds the local top-K and the cross-rank merge + lattice + walk
(``sglang.kernels.ops.speculative.dflash_select``). This module holds the weight layout that needs and the conditions
it runs under."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch
from torch import nn

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.gemm.lmhead import lmhead_gemm
from sglang.kernels.ops.speculative.dflash_select import (
    dflash_local_topk,
    dflash_select_walk,
    get_dflash_select_workspace_request,
)
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.distributed.device_communicators.symm_workspace import (
    request_group_workspace,
)
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.runtime_context import get_exec, get_model

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import PushPlane
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
    from sglang.srt.models.dflash import CandidateSelector
    from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


# Fixed by the select kernels (csrc/speculative/dflash_local_topk.cuh, dflash_select_walk.cuh).
_TOP_KS = (16, 32)
_SELECTOR_RANK = 256
_MAX_GAMMA = 8
_SHARD_WIDTH_ALIGN, _SHARD_WIDTH_MIN, _SHARD_WIDTH_MAX = 1024, 8192, 32768
# The lm_head GEMM's k per stage (lmhead_mma.cuh BLOCK_K; the UMMA kernel needs 64).
_HIDDEN_ALIGN = 256


def _get_fused_select_blocker(
    *,
    lm_head: ParallelLMHead,
    selector: CandidateSelector,
    gamma: int,
    final_logit_softcapping: Optional[float],
) -> Optional[str]:
    """Why the fused selection cannot run, or None when it can. Every input is the same on all TP ranks, so the
    ranks agree: the kernels synchronize them every round, and a rank on another path would hang the rest."""
    if not current_arch().is_sm(10):
        return "requires SM100"
    if lm_head.tp_size < 2:
        return "requires a vocab-sharded lm_head (TP >= 2)"
    if lm_head.use_attn_tp_group:
        return "dp attention is not supported"
    if lm_head.weight.dtype != torch.bfloat16 or should_apply_lm_head_quant_method(
        lm_head, lm_head.quant_method
    ):
        return "requires a dense bf16 lm_head"
    if lm_head.num_embeddings_padded != lm_head.org_vocab_size:
        return "requires a vocabulary without added or padding rows"
    shard_width = lm_head.org_vocab_size // lm_head.tp_size
    if not (
        shard_width % _SHARD_WIDTH_ALIGN == 0
        and _SHARD_WIDTH_MIN <= shard_width <= _SHARD_WIDTH_MAX
    ):
        return f"unsupported lm_head shard width {shard_width}"
    if lm_head.weight.shape[1] % _HIDDEN_ALIGN != 0:
        return f"unsupported hidden size {lm_head.weight.shape[1]}"
    tables = (
        selector.hidden_projection.weight,
        selector.predecessor_codebook,
        selector.successor_codebook,
    )
    if (
        any(t.dtype != torch.bfloat16 for t in tables)
        or tables[0].shape[0] != _SELECTOR_RANK
    ):
        return f"requires bf16 selector tables of rank {_SELECTOR_RANK}"
    if selector.predecessor_codebook.shape[0] != lm_head.org_vocab_size:
        return "the selector codebooks and the lm_head disagree on the vocabulary"
    if selector.top_k not in _TOP_KS:
        return f"unsupported selector top-k {selector.top_k}"
    if get_model().weight_cache_mode != "off":
        # IPC-shared weights are updated in place by their owner; a merged copy would go stale.
        return "zero-copy IPC weight cache is not supported"
    if gamma > _MAX_GAMMA:
        return f"unsupported gamma {gamma}"
    if final_logit_softcapping is not None:
        return "final_logit_softcapping is not supported"
    return None


def _merge_lmhead_projection(
    *,
    target_model: nn.Module,
    lm_head: ParallelLMHead,
    projection: nn.Linear,
    memory_saver_adapter: TorchMemorySaverAdapter,
) -> torch.Tensor:
    """Moves the lm_head shard [Vs, H] and the selector projection [R, H] into one [Vs + R, H] buffer and points both
    parameters at their rows of it, so one GEMM computes the logits and the projection. Every parameter of the target
    sharing the head's storage (a tied embedding) moves too, so the old storage is freed.

    Must run after both models load and before the KV cache is sized (the copy needs one more shard for a moment)
    and before any graph capture (graphs bake the head's address). The buffer lives in the memory saver's weights
    region, like the parameters it replaces, so releasing the weights releases it too."""
    head, proj = lm_head.weight, projection.weight
    assert (
        head.is_contiguous() and proj.is_contiguous() and head.shape[1] == proj.shape[1]
    ), (head.shape, proj.shape)
    num_head_rows = head.shape[0]
    features = get_exec().features
    with memory_saver_adapter.region(
        GPU_MEMORY_TYPE_WEIGHTS,
        # Back up if either model's weights are backed up: the buffer holds both.
        enable_cpu_backup=features.enable_weights_cpu_backup
        or features.enable_draft_weights_cpu_backup,
    ):
        merged = head.new_empty((num_head_rows + proj.shape[0], head.shape[1]))
    merged[:num_head_rows].copy_(head)
    merged[num_head_rows:].copy_(proj)
    head_ptr = head.data_ptr()
    aliases = [
        p
        for p in target_model.parameters()
        if p.data_ptr() == head_ptr and p.shape == head.shape
    ]
    for param in aliases:
        param.data = merged[:num_head_rows]
    proj.data = merged[num_head_rows:]
    return merged


class FusedSelect:
    """One worker's fused selection: lm_head + projection GEMM, local top-K (K1), merge + lattice + walk (K2).

    K1 and K2 synchronize the TP ranks through the push plane every call, so every rank must make the same calls:
    callers choose this path from inputs that are equal on all ranks (the batch size)."""

    def __init__(
        self,
        *,
        merged_weight: torch.Tensor,
        shard_width: int,
        plane: PushPlane,
        selector: CandidateSelector,
        output_multiplier: float,
        max_rows: int,
    ) -> None:
        self.merged_weight = merged_weight  # [shard_width + R, H]: the lm_head shard, then the projection
        self.shard_width = shard_width
        self.plane = plane
        self.selector = selector
        self.output_multiplier = output_multiplier
        self.max_rows = max_rows  # rows (bs x gamma) the push slot holds

    def fits(self, num_rows: int) -> bool:
        return num_rows <= self.max_rows

    def __call__(
        self,
        *,
        pred_hidden: torch.Tensor,
        anchor_ids: torch.Tensor,
        temperatures: torch.Tensor,
        greedy_mask: torch.Tensor,
        uniforms: torch.Tensor,
        out: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """pred_hidden [bs, gamma, H] -> tokens int64 [bs, gamma], candidate_ids int64 [bs, gamma, K],
        q fp32 [bs, gamma, K] (written into ``out`` when given)."""
        bs, gamma, hidden = pred_hidden.shape
        # [bs * gamma, shard_width + R]: the logits shard, then hp.
        logits_hp = lmhead_gemm(
            pred_hidden.reshape(bs * gamma, hidden), self.merged_weight
        )
        dflash_local_topk(
            logits_hp[:, : self.shard_width], self.plane, self.selector.top_k
        )
        return dflash_select_walk(
            self.plane,
            self.selector.top_k,
            self.selector.predecessor_codebook,
            self.selector.successor_codebook,
            logits_hp[:, self.shard_width :],
            anchor_ids,
            temperatures,
            greedy_mask,
            uniforms,
            self.output_multiplier,
            out=out,
        )


def build_fused_select(
    *,
    target_model: nn.Module,
    lm_head: ParallelLMHead,
    selector: CandidateSelector,
    gamma: int,
    max_bs: int,
    output_multiplier: float,
    final_logit_softcapping: Optional[float],
    memory_saver_adapter: TorchMemorySaverAdapter,
) -> Tuple[Optional[FusedSelect], Optional[str]]:
    """The fused selection, or (None, why not). Collective over the lm_head's TP group (the workspace request).

    Merges the weights (``_merge_lmhead_projection``), so it carries the same placement constraint: after both
    models load, before the KV cache is sized and before any graph capture."""
    blocker = _get_fused_select_blocker(
        lm_head=lm_head,
        selector=selector,
        gamma=gamma,
        final_logit_softcapping=final_logit_softcapping,
    )
    if blocker is not None:
        return None, blocker
    if max_bs == 0:
        return (
            None,
            "no decode CUDA graph batch sizes (the push slot is sized by the largest)",
        )
    max_rows = max_bs * gamma
    if lm_head.tp_group is None:
        return None, "No TP-group for LMHead"
    lease = request_group_workspace(
        lm_head.tp_group, get_dflash_select_workspace_request(max_rows, selector.top_k)
    )
    if lease is None or not lease.has_multicast:
        return (
            None,
            "requires a symmetric workspace with multicast on the lm_head's TP group",
        )
    shard_width = lm_head.org_vocab_size // lm_head.tp_size
    assert lm_head.shard_indices.org_vocab_start_index == lease.rank * shard_width, (
        "shard order != group rank order"
    )
    merged = _merge_lmhead_projection(
        target_model=target_model,
        lm_head=lm_head,
        projection=selector.hidden_projection,
        memory_saver_adapter=memory_saver_adapter,
    )
    push_plane = lease.get_communicator().push
    assert push_plane is not None
    return (
        FusedSelect(
            merged_weight=merged,
            shard_width=shard_width,
            plane=push_plane,
            selector=selector,
            output_multiplier=output_multiplier,
            max_rows=max_rows,
        ),
        None,
    )
