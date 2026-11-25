"""DFLASH sampled verify on each row's top-k support (SGLANG_OPT_DFLASH_SPARSE_VERIFY).

With TP > 1 and a softcapped target, verify leaves each rank holding its vocab
shard of the logits. For a sampled batch whose top-k the support reproduces,
each rank packs its shard's shortlist, the shortlists are all-gathered and
merged into the top-k of the full softcapped rows, and the chain sampler runs
on that support instead of on the all-gathered full-vocab logits.
"""

import logging
from typing import Optional, Tuple

import torch

from sglang.kernels.ops.speculative.dflash_sparse_verify import (
    merge_shard_topk,
    pack_shard_topk,
    shard_topk_width,
    sparse_chain_speculative_sampling,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo, adjusts_logits
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

# Measured at top_k 64. A larger cap widens each row's merge sort to
# tp_size * next_power_of_2(top_k + 16) keys.
_SPARSE_VERIFY_MAX_TOP_K = 64

# merge_shard_topk's (values, ids, argmax) of the full softcapped verify rows.
SparseSupport = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class DFlashSparseVerify:
    """Built once per worker by ``create``; None where the sparse path cannot run."""

    def __init__(self, *, softcap: float, vocab_size: int, tp_size: int) -> None:
        self._softcap = softcap
        self._vocab_size = vocab_size
        self._tp_size = tp_size
        # Verify steps accepted on the support; reported in /server_info.
        self.verify_ct = 0

    @classmethod
    def create(
        cls,
        *,
        logits_processor: Optional[LogitsProcessor],
        tp_size: int,
        has_selector: bool,
        block_verification: bool,
        lm_head: Optional[torch.nn.Module],
    ) -> Optional["DFlashSparseVerify"]:
        if not (
            envs.SGLANG_OPT_DFLASH_SPARSE_VERIFY.get()
            and is_cuda()
            and has_selector
            and not block_verification
            and tp_size > 1
            # Only a softcapped target defers the verify gather to a vocab shard.
            and logits_processor is not None
            and bool(logits_processor.final_logit_softcapping)
            # Verify gets this head's vocab shard (its logical width, whatever the
            # weight layout); a padded one carries columns past the vocabulary.
            and isinstance(lm_head, VocabParallelEmbedding)
            and tp_size * lm_head.num_embeddings_per_partition
            == logits_processor.vocab_size
        ):
            return None
        if get_parallel().tp_rank == 0:
            logger.info(
                "DFLASH sparse verify enabled for sampled batches with top_k <= %d.",
                _SPARSE_VERIFY_MAX_TOP_K,
            )
        return cls(
            softcap=logits_processor.final_logit_softcapping,
            vocab_size=logits_processor.vocab_size,
            tp_size=tp_size,
        )

    def eligible(
        self,
        *,
        batch: ScheduleBatch,
        sampling_info: SamplingBatchInfo,
        max_top_k: int,
        shard_width: int,
    ) -> bool:
        """Whether the top-k support reproduces this sampled batch's dense verify."""
        if batch.has_grammar or batch.return_logprob:
            return False
        # max_top_k <= 64 implies every row samples with a top-k (top_k -1 is TOP_K_ALL).
        return (
            max_top_k <= _SPARSE_VERIFY_MAX_TOP_K
            and not sampling_info.need_top_p_sampling
            and not sampling_info.need_min_p_sampling
            and shard_topk_width(max_top_k) <= shard_width
            and self._tp_size * shard_width == self._vocab_size
            and not adjusts_logits(sampling_info, has_grammar=batch.has_grammar)
            # A sampling mask is captured from the full target distribution.
            and sampling_info.sampling_mask_batch_indices is None
        )

    def support(
        self, *, tp_group, logits_shard: torch.Tensor, max_top_k: int
    ) -> SparseSupport:
        """The top-k of the full softcapped rows, from this rank's vocab shard."""
        num_rows, shard_width = logits_shard.shape
        local = pack_shard_topk(
            logits_shard,
            max_top_k=max_top_k,
            col_offset=tp_group.rank_in_group * shard_width,
        )
        gathered = torch.empty(
            (tp_group.world_size * num_rows, local.shape[1]),
            dtype=local.dtype,
            device=local.device,
        )
        tp_group.all_gather_into_tensor(gathered, local)
        return merge_shard_topk(
            gathered,
            world=tp_group.world_size,
            softcap=self._softcap,
            max_top_k=max_top_k,
        )

    def prewarm(
        self,
        *,
        device: torch.device,
        block_size: int,
        selector_top_k: int,
        tp_group,
    ) -> None:
        """Compile the kernels at the top_k cap, for one and two requests (both
        row-count specializations), without a collective."""
        shard_width = self._vocab_size // self._tp_size
        top_k = _SPARSE_VERIFY_MAX_TOP_K
        if shard_topk_width(top_k) > shard_width:
            return
        fp32 = torch.float32
        q_shape = (block_size - 1, selector_top_k)
        for bs in (1, 2):
            local = pack_shard_topk(
                torch.zeros((bs * block_size, shard_width), dtype=fp32, device=device),
                max_top_k=top_k,
                col_offset=tp_group.rank_in_group * shard_width,
            )
            sup_vals, sup_ids, _ = merge_shard_topk(
                local.repeat(self._tp_size, 1),
                world=self._tp_size,
                softcap=self._softcap,
                max_top_k=top_k,
            )
            sparse_chain_speculative_sampling(
                sup_vals=sup_vals,
                sup_ids=sup_ids,
                candidates=torch.zeros(
                    (bs, block_size), dtype=torch.int64, device=device
                ),
                q_ids=torch.zeros((bs, *q_shape), dtype=torch.int64, device=device),
                q_vals=torch.zeros((bs, *q_shape), dtype=fp32, device=device),
                temperatures=torch.ones((bs, 1), dtype=fp32, device=device),
                top_ks=torch.full((bs,), top_k, dtype=torch.int32, device=device),
                max_top_k=top_k,
                vocab_size=self._vocab_size,
                uniform_samples=torch.zeros(
                    (bs, block_size - 1), dtype=fp32, device=device
                ),
                uniform_samples_final=torch.zeros((bs,), dtype=fp32, device=device),
            )

    def accept(
        self,
        *,
        candidates: torch.Tensor,
        support: SparseSupport,
        candidate_ids: torch.Tensor,
        q_rows: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        max_top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The dense selector accept's (accept_len, bonus) on the merged support."""
        bs, block = candidates.shape
        sup_vals, sup_ids, _ = support
        # The dense path's two draws, same shapes and order, so the RNG stream
        # continues as it would there.
        uniform_samples = torch.rand(
            (bs, block - 1), dtype=torch.float32, device=candidates.device
        )
        uniform_samples_final = torch.rand(
            (bs,), dtype=torch.float32, device=candidates.device
        )
        self.verify_ct += 1
        return sparse_chain_speculative_sampling(
            sup_vals=sup_vals,
            sup_ids=sup_ids,
            candidates=candidates,
            q_ids=candidate_ids,
            q_vals=q_rows.float(),
            temperatures=sampling_info.temperatures,
            top_ks=sampling_info.top_ks,
            max_top_k=max_top_k,
            vocab_size=self._vocab_size,
            uniform_samples=uniform_samples,
            uniform_samples_final=uniform_samples_final,
        )
