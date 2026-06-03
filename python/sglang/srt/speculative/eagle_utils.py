from __future__ import annotations

import logging
import math
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.speculative.triton_ops.spec_tree import (
    sgl_build_tree_kernel_efficient_triton,
    verify_tree_greedy_kernel_triton,
)
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu
from sglang.srt.utils.async_probe import maybe_detect_oob

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()
_is_xpu = is_xpu()

logger = logging.getLogger(__name__)

if _is_cuda or _is_hip or _is_musa:
    from sgl_kernel import (
        build_tree_kernel_efficient as sgl_build_tree_kernel_efficient,
    )


def per_step_draft_out_cache_loc(
    out_cache_loc: torch.Tensor,
    batch_size: int,
    topk: int,
    num_steps: int,
) -> torch.Tensor:
    """Per-step slice of the multi-step EAGLE draft out_cache_loc buffer.

    Single source of truth for the layout shared by EagleWorkerV2.draft_forward
    (per-step write target) and DeepseekV4AttnBackend (per-step compression
    write target baked into metadata).
    """
    expected = batch_size * topk * num_steps
    assert out_cache_loc.shape[0] == expected, (
        f"out_cache_loc.shape[0]={out_cache_loc.shape[0]} != "
        f"batch_size * topk * num_steps = {batch_size}*{topk}*{num_steps}={expected}"
    )
    return (
        out_cache_loc.view(batch_size, topk, num_steps)
        .permute(2, 0, 1)
        .reshape(num_steps, -1)
    )


def _eagle_prefill_tail_tokens(
    batch: ScheduleBatch, next_token_ids: torch.Tensor
) -> torch.Tensor:
    """Per-seq tail token for EAGLE prefill rotation; uses next prompt token for
    non-final chunks (chunked-prefill chain consistency, see PR #26329)."""
    tail_tokens = next_token_ids.to(batch.input_ids.dtype)
    next_prompt_token = batch.chunked_req_next_prompt_token
    if next_prompt_token is not None:
        for i, r in enumerate(batch.reqs):
            if r is batch.chunked_req:
                tail_tokens = tail_tokens.clone()
                tail_tokens[i] = next_prompt_token
                break
    return tail_tokens

def organize_draft_results(
    score_list: List[torch.Tensor],
    token_list: List[torch.Tensor],
    parents_list: List[torch.Tensor],
    num_draft_token: int,
):
    score_list = torch.cat(score_list, dim=1).flatten(1)
    ss_token_list = torch.cat(token_list, dim=1)
    top_scores = torch.topk(score_list, num_draft_token - 1, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values
    maybe_detect_oob(
        top_scores_index,
        0,
        ss_token_list.shape[1],
        "organize_draft_results: top_scores_index OOB for gather on ss_token_list",
    )
    draft_tokens = torch.gather(ss_token_list, index=top_scores_index, dim=1)

    if len(parents_list) > 1:
        parent_list = torch.cat(parents_list[:-1], dim=1)
    else:
        batch_size = parents_list[0].shape[0]
        parent_list = torch.empty(
            batch_size, 0, dtype=torch.long, device=parents_list[0].device
        )

    return parent_list, top_scores_index, draft_tokens


class TreeMaskMode(IntEnum):
    FULL_MASK = 0
    QLEN_ONLY = 1
    QLEN_ONLY_BITPACKING = 2


def sgl_build_tree_kernel_efficient_pytorch(
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
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,  # TODO: QLEN_ONLY_BITPACKING needs to be added
):
    # TODO: Add support for QLEN_ONLY_BITPACKING mode
    if tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        raise NotImplementedError(
            "QLEN_ONLY_BITPACKING is not supported in PyTorch implementation"
        )

    prefix_sums = torch.cumsum(verified_seq_len, dim=0)
    for batch_idx in range(verified_seq_len.shape[0]):
        seq_len_prefix_sum = prefix_sums[batch_idx - 1] if batch_idx > 0 else 0
        seq_tree_idx = (
            draft_token_num * draft_token_num * batch_idx
            + seq_len_prefix_sum * draft_token_num
        )

        seq_len = verified_seq_len[batch_idx]
        for draft_token_idx in range(draft_token_num):
            if tree_mask_mode == TreeMaskMode.FULL_MASK:
                token_tree_idx = (
                    seq_tree_idx
                    + (seq_len + draft_token_num) * draft_token_idx
                    + seq_len
                    + 1
                )
            else:
                token_tree_idx = (
                    draft_token_num * draft_token_num * batch_idx
                    + draft_token_num * draft_token_idx
                    + 1
                )

            tree_mask[token_tree_idx - 1] = True
            for i in range(draft_token_num - 1):
                tree_mask[token_tree_idx + i] = False

            if draft_token_idx == 0:
                positions[batch_idx * draft_token_num] = seq_len

                retrive_index_offset = batch_idx * draft_token_num
                for i in range(draft_token_num - 1, 0, -1):
                    current_token_idx = retrive_index_offset + i
                    retrive_index[batch_idx][i] = current_token_idx
                    parent_tb_idx = int(selected_index[batch_idx][i - 1]) // topk
                    parent_position = 0
                    found_parent = parent_tb_idx == 0
                    if parent_tb_idx > 0:
                        parent_token_idx = parent_list[batch_idx][parent_tb_idx]
                        while parent_position < draft_token_num - 1:
                            if (
                                selected_index[batch_idx][parent_position]
                                == parent_token_idx
                            ):
                                parent_position += 1
                                found_parent = True
                                break
                            parent_position += 1

                    if not found_parent:
                        logger.warning(
                            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
                            "Please check if the logprob has nan. The token will be ignored to keep proceeding."
                        )
                        continue

                    if retrive_next_token[batch_idx][parent_position] == -1:
                        retrive_next_token[batch_idx][parent_position] = i
                    else:
                        origin_next_token = retrive_next_token[batch_idx][
                            parent_position
                        ].item()
                        retrive_next_token[batch_idx][parent_position] = i
                        retrive_next_sibling[batch_idx][i] = origin_next_token

                retrive_index[batch_idx][0] = batch_idx * draft_token_num
            else:
                cur_position = draft_token_idx - 1
                position = 0
                for _ in range(depth):
                    position += 1
                    tree_mask[token_tree_idx + cur_position] = True
                    parent_tb_idx = int(selected_index[batch_idx][cur_position]) // topk
                    if parent_tb_idx == 0:
                        break

                    parent_token_idx = parent_list[batch_idx][parent_tb_idx]
                    found = False
                    for cp in range(draft_token_num - 1):
                        if selected_index[batch_idx][cp] == parent_token_idx:
                            cur_position = cp
                            found = True
                            break
                    if not found:
                        break

                positions[batch_idx * draft_token_num + draft_token_idx] = (
                    position + seq_len
                )


def build_tree_kernel_efficient(
    bonus_tokens: torch.Tensor,
    parent_list: List[torch.Tensor],
    top_scores_index: torch.Tensor,
    draft_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_sum: int,
    topk: int,
    spec_steps: int,
    num_verify_tokens: int,
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
    tree_mask_buf: Optional[torch.Tensor] = None,
    position_buf: Optional[torch.Tensor] = None,
):
    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()

    # seq_lens_sum == sum(seq_lens); seq_lens: sequence length without draft tokens
    bs = seq_lens.numel()
    device = seq_lens.device
    # e.g. for bs=1, tree_mask: num_draft_token, seq_lens_sum + num_draft_token (flattened)
    # where each row indicates the attending pattern of each draft token
    # if use_partial_packed_tree_mask is True, tree_mask: num_draft_token (flattened, packed)
    if tree_mask_buf is not None:
        tree_mask = tree_mask_buf
        if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
            tree_mask.fill_(True)
        elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
            tree_mask.fill_(0)
        elif tree_mask_mode == TreeMaskMode.FULL_MASK:
            tree_mask.fill_(True)
        else:
            raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY:
        tree_mask = torch.full(
            (num_verify_tokens * bs * num_verify_tokens,),
            True,
            dtype=torch.bool,
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        packed_dtypes = [torch.uint8, torch.uint16, torch.uint32]
        packed_dtype_idx = int(math.ceil(math.log2((num_verify_tokens + 7) // 8)))
        tree_mask = torch.zeros(
            (num_verify_tokens * bs,),
            dtype=packed_dtypes[packed_dtype_idx],
            device=device,
        )
    elif tree_mask_mode == TreeMaskMode.FULL_MASK:
        tree_mask = torch.full(
            (
                seq_lens_sum * num_verify_tokens
                + num_verify_tokens * num_verify_tokens * bs,
            ),
            True,
            device=device,
        )
    else:
        raise NotImplementedError(f"Invalid tree mask: {tree_mask_mode=}")

    # TODO: make them torch.empty and fuse them into `sgl_build_tree_kernel`
    retrieve_buf = torch.full(
        (3, bs, num_verify_tokens), -1, device=device, dtype=torch.long
    )
    retrieve_index, retrieve_next_token, retrieve_next_sibling = retrieve_buf
    # position: where each token belongs to
    # e.g. if depth of each draft token is [0, 1, 1, 2] and the prompt length is 7
    # then, positions = [7, 8, 8, 9]
    if position_buf is not None:
        positions = position_buf
    else:
        positions = torch.empty(
            (bs * num_verify_tokens,), device=device, dtype=torch.long
        )

    if _is_npu:
        torch.ops.npu.build_tree_kernel_efficient(
            parent_list.to(dtype=torch.int64),
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    elif _is_xpu:
        # Try Triton implementation first, fallback to PyTorch if not available
        try:
            sgl_build_tree_kernel_triton(
                parent_list,
                top_scores_index,
                seq_lens,
                tree_mask,
                positions,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                topk,
                spec_steps,
                num_verify_tokens,
                tree_mask_mode,
            )
        except (AttributeError, RuntimeError):
            # Reinitialize buffers to original state in case Triton partially corrupted them
            if tree_mask_mode == TreeMaskMode.QLEN_ONLY:
                tree_mask.fill_(True)
            elif tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
                tree_mask.fill_(0)
            elif tree_mask_mode == TreeMaskMode.FULL_MASK:
                tree_mask.fill_(True)
            retrive_index.fill_(-1)
            retrive_next_token.fill_(-1)
            retrive_next_sibling.fill_(-1)

            # Fallback to PyTorch implementation
            sgl_build_tree_kernel_efficient_pytorch(
                parent_list,
                top_scores_index,
                seq_lens,
                tree_mask,
                positions,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                topk,
                spec_steps,
                num_verify_tokens,
                tree_mask_mode,
            )
    else:
        sgl_build_tree_kernel_efficient(
            parent_list,
            top_scores_index,
            seq_lens,
            tree_mask,
            positions,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            topk,
            spec_steps,
            num_verify_tokens,
            tree_mask_mode,
        )
    return (
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_tokens,
    )


def verify_tree_greedy_pytorch(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    batch_size = candidates.shape[0]
    num_speculative_tokens = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]

    for bx in range(batch_size):
        last_accepted_retrive_idx = retrive_index[bx][0]
        accept_index[bx][0] = last_accepted_retrive_idx
        num_accepted_tokens = 0
        cur_index = 0

        for j in range(1, num_speculative_tokens):
            cur_index = retrive_next_token[bx][cur_index]
            while cur_index != -1:
                draft_index = retrive_index[bx][cur_index]
                draft_token_id = candidates[bx][cur_index]
                target_token_id = target_predict[
                    last_accepted_retrive_idx // num_draft_tokens
                ][last_accepted_retrive_idx % num_draft_tokens]

                if draft_token_id == target_token_id:
                    # accept token
                    predicts[last_accepted_retrive_idx] = target_token_id
                    num_accepted_tokens += 1
                    accept_index[bx][num_accepted_tokens] = draft_index
                    last_accepted_retrive_idx = draft_index
                    break
                else:
                    cur_index = retrive_next_sibling[bx][cur_index]

            if cur_index == -1:
                break

        accept_token_num[bx] = num_accepted_tokens
        predicts[last_accepted_retrive_idx] = target_predict[
            last_accepted_retrive_idx // num_draft_tokens
        ][last_accepted_retrive_idx % num_draft_tokens]


def sgl_build_tree_kernel_triton(
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
    tree_mask_mode: TreeMaskMode = TreeMaskMode.FULL_MASK,
):
    """Triton-based implementation."""
    # TODO: Add support for QLEN_ONLY_BITPACKING mode
    if tree_mask_mode == TreeMaskMode.QLEN_ONLY_BITPACKING:
        raise NotImplementedError(
            "QLEN_ONLY_BITPACKING is not supported in Triton implementation"
        )

    batch_size = verified_seq_len.shape[0]
    seq_len_prefix_sum = torch.cumsum(verified_seq_len, dim=0) - verified_seq_len

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    sgl_build_tree_kernel_efficient_triton[grid](
        parent_list,
        selected_index,
        verified_seq_len,
        seq_len_prefix_sum,
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        topk=topk,
        depth=depth,
        draft_token_num=draft_token_num,
        tree_mask_mode=int(tree_mask_mode),
        batch_size=batch_size,
        parent_list_stride=(
            parent_list.stride(0) if parent_list.dim() > 1 else parent_list.shape[0]
        ),
        selected_index_stride=selected_index.stride(0),
    )


def verify_tree_greedy_triton(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
):
    """Triton-based implementation."""
    batch_size = candidates.shape[0]
    num_speculative_tokens = accept_index.shape[1]
    num_draft_tokens = candidates.shape[1]

    # Launch kernel with one program per batch item
    grid = (batch_size,)

    verify_tree_greedy_kernel_triton[grid](
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        target_predict,
        batch_size=batch_size,
        num_speculative_tokens=num_speculative_tokens,
        num_draft_tokens=num_draft_tokens,
    )


def verify_tree_greedy_func(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    topk: int = -1,
):
    if _is_cuda or _is_hip or _is_musa:
        from sgl_kernel import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,  # mutable
            accept_index=accept_index,  # mutable
            accept_token_num=accept_token_num,  # mutable
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )

    elif _is_npu:
        from sgl_kernel_npu.sample.verify_tree_greedy import verify_tree_greedy

        verify_tree_greedy(
            predicts=predicts,
            accept_index=accept_index,
            accept_token_num=accept_token_num,
            candidates=candidates,
            # kwarg LHS retained as `retrive_*` to match sgl_kernel op schema.
            retrive_index=retrieve_index,
            retrive_next_token=retrieve_next_token,
            retrive_next_sibling=retrieve_next_sibling,
            target_predict=target_predict,
        )
    elif _is_xpu:
        # Try Triton implementation first, fallback to PyTorch if not available
        try:
            verify_tree_greedy_triton(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrive_index=retrieve_index,
                retrive_next_token=retrieve_next_token,
                retrive_next_sibling=retrieve_next_sibling,
                target_predict=target_predict,
            )
        except (AttributeError, RuntimeError):
            # Reinitialize buffers to original state in case Triton partially corrupted them
            accept_index.fill_(-1)
            accept_token_num.fill_(0)

            # Fallback to PyTorch implementation
            verify_tree_greedy_pytorch(
                predicts=predicts,
                accept_index=accept_index,
                accept_token_num=accept_token_num,
                candidates=candidates,
                retrive_index=retrieve_index,
                retrive_next_token=retrieve_next_token,
                retrive_next_sibling=retrieve_next_sibling,
                target_predict=target_predict,
            )
    return predicts, accept_index, accept_token_num


def get_draft_hidden_dim(model_runner: ModelRunner) -> int:
    """Derive the hidden dimension of target hidden states fed to the draft model."""
    hf_config = model_runner.model_config.hf_config
    eagle_config = getattr(hf_config, "eagle_config", {})
    use_aux = eagle_config.get("use_aux_hidden_state", False)
    spec_algorithm = model_runner.spec_algorithm

    if spec_algorithm is not None and spec_algorithm.is_eagle3() and use_aux:
        base = getattr(hf_config, "target_hidden_size", None)
        if base is None:
            base = model_runner.model_config.hidden_size
        layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids", [])
        num_aux = max(len(layer_ids), 1)
        return base * num_aux
    return model_runner.model_config.spec_hidden_size
