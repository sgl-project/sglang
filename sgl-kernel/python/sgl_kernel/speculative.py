import logging

import torch
from sgl_kernel.utils import get_cuda_stream

logger = logging.getLogger(__name__)


def tree_speculative_sampling_target_only(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    uniform_samples: torch.Tensor,
    uniform_samples_for_final_sampling: torch.Tensor,
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    threshold_single: float = 1.0,
    threshold_acc: float = 1.0,
    deterministic: bool = True,
) -> None:
    with torch.no_grad():
        # target_probs, draft_probs: [bs, num_draft_tokens, vocab]
        diff = target_probs - draft_probs
        # 1. Per drafted token mean difference
        per_pos_mean_abs = diff.mean(dim=-1)  # [bs, num_draft_tokens]
        # 2. Batch-level summary (mean over positions)
        per_seq_mean_abs = per_pos_mean_abs.mean(dim=-1)  # [bs]
        # 3. Global stats
        global_mean = per_pos_mean_abs.mean()
        global_max = per_pos_mean_abs.max()
        logger.debug("=========== <Probs diff stats> ===========")
        logger.debug("target_probs: %s", target_probs)
        logger.debug("draft_probs: %s", draft_probs)
        logger.debug("diff: %s", diff)
        logger.debug("per_seq_mean_abs: %s", per_seq_mean_abs)
        logger.debug("global mean `q-p` per position: %s", global_mean.item())
        logger.debug("global max  `q-p` per position: %s", global_max.item())
        logger.debug("=========== </Probs diff stats> ===========")
    torch.ops.sgl_kernel.tree_speculative_sampling_target_only.default(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        uniform_samples,
        uniform_samples_for_final_sampling,
        target_probs,
        draft_probs,
        threshold_single,
        threshold_acc,
        deterministic,
        get_cuda_stream(),
    )


def verify_tree_greedy(
    predicts: torch.Tensor,  # mutable
    accept_index: torch.Tensor,  # mutable
    accept_token_num: torch.Tensor,  # mutable
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    torch.ops.sgl_kernel.verify_tree_greedy.default(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        get_cuda_stream(),
    )


def build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    torch.ops.sgl_kernel.build_tree_kernel_efficient.default(
        parent_list,
        selected_index,
        verified_seq_len,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk,
        depth,
        draft_token_num,
        tree_mask_mode,
    )


def segment_packbits(
    x: torch.Tensor,
    input_indptr: torch.Tensor,
    output_indptr: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> None:
    torch.ops.sgl_kernel.segment_packbits.default(
        x,
        input_indptr,
        output_indptr,
        y,
        batch_size,
        torch.cuda.current_stream().cuda_stream,
    )
