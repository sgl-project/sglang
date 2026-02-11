import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_npu

_is_npu = is_npu()


def parallel_decoding_update_input_ids_vectorized(
    input_ids_1d: torch.Tensor,  # [B*blk]
    full_logits_2d: torch.Tensor,  # [B*blk, V]
    mask_id: int,
    block_size: int,
    threshold: float,
    finished: Optional[
        torch.Tensor
    ] = None,  # [B] bool, True means do NOT update this row
    force_at_least_one: bool = True,
) -> None:
    """
    Vectorized update for parallel decoding:
      - reshape to [B, blk] and update each row in-place
      - replaces per-batch Python loop + .item() sync points
      - optionally gates out finished rows

    Semantics (matches your old code):
      - Only positions where input_ids == mask_id are candidates for replacement
      - Replace with argmax token if its confidence > threshold
      - If a row has any mask positions but none pass threshold, force select 1 position (max confidence)
      - Rows with no mask positions are left unchanged (equivalent to `continue`)
      - If `finished` is provided, finished rows are always left unchanged
    """
    assert (
        input_ids_1d.dim() == 1
    ), f"input_ids_1d must be 1D, got {tuple(input_ids_1d.shape)}"
    assert (
        full_logits_2d.dim() == 2
    ), f"full_logits_2d must be 2D, got {tuple(full_logits_2d.shape)}"
    assert (
        input_ids_1d.shape[0] == full_logits_2d.shape[0]
    ), f"len mismatch: {input_ids_1d.shape[0]} vs {full_logits_2d.shape[0]}"
    assert (
        input_ids_1d.shape[0] % block_size == 0
    ), f"len(input_ids_1d) must be divisible by block_size, got {input_ids_1d.shape[0]}, block_size={block_size}"
    assert 0.0 < threshold <= 1.0, f"threshold should be in (0,1], got {threshold}"

    B = input_ids_1d.shape[0] // block_size
    blk = block_size
    V = full_logits_2d.shape[1]
    dev = input_ids_1d.device

    # Views (no copy)
    input_ids = input_ids_1d.view(B, blk)  # [B, blk]
    logits = full_logits_2d.view(B, blk, V)  # [B, blk, V]

    # Candidate mask
    mask = input_ids.eq(mask_id)  # [B, blk]
    if finished is not None:
        assert finished.shape == (
            B,
        ), f"finished must be shape [B]={B}, got {tuple(finished.shape)}"
        alive = (~finished).view(B, 1)  # [B,1]
        eff_mask = mask & alive
    else:
        eff_mask = mask

    # Argmax token id per position and its log-prob.
    # logp = log softmax(logits)[argmax] = max_logit - logsumexp(logits)
    max_logit, x = logits.max(dim=-1)  # [B, blk], [B, blk]
    lse = torch.logsumexp(logits, dim=-1)  # [B, blk]
    logp = max_logit - lse  # [B, blk]

    neg_inf = torch.full_like(logp, float("-inf"))
    confidence = torch.where(eff_mask, logp, neg_inf)  # [B, blk]

    log_thr = math.log(threshold)
    transfer = confidence > log_thr  # [B, blk] bool

    # Apply threshold updates first
    new_ids = torch.where(transfer, x, input_ids)  # [B, blk]

    # Force-at-least-one (graph-friendly): unconditional sparse per-row write, gated to no-op when not needed
    if force_at_least_one:
        row_has_any = eff_mask.any(dim=1)  # [B]
        hit_any = transfer.any(dim=1)  # [B]
        need_force = row_has_any & (~hit_any)  # [B]

        # Best idx per row (even if all -inf -> 0); gated by need_force
        best_idx = confidence.argmax(dim=1)  # [B] long

        # idx2: where to write per row:
        #   - need_force: best_idx
        #   - else: 0 (safe position)
        zero_idx = torch.zeros_like(best_idx)
        idx2 = torch.where(need_force, best_idx, zero_idx)  # [B]

        rows = torch.arange(B, device=dev)  # [B]

        # val2: what to write per row:
        #   - need_force: x[row, best_idx]
        #   - else: new_ids[row, 0] (no-op)
        force_val = x[rows, best_idx]  # [B]
        noop_val = new_ids[rows, zero_idx]  # [B] == new_ids[:,0]
        val2 = torch.where(need_force, force_val, noop_val)  # [B]

        # Single-point update per row (typically much cheaper than [B,blk] one-hot scatter_)
        new_ids[rows, idx2] = val2

    # In-place write-back (preserve storage / better for graph capture)
    input_ids.copy_(new_ids)


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        # Here, the forward_batch full logits contains all the blocks
        # such as [dllm_block_size * batch_size, hidden_size]
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each block
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask_index = block_input_ids == self.mask_id
            start = self.block_size - torch.sum(block_mask_index).item()
            start_list.append(start)

        if _is_npu:
            skip_attn_backend_init = False

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            if _is_npu:
                out = model_runner.forward(
                    forward_batch, skip_attn_backend_init, pp_proxy_tensors=None
                )
                skip_attn_backend_init = True
            else:
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            assert batch_size == forward_batch.input_ids.shape[0] // self.block_size
            if _is_npu:
                parallel_decoding_update_input_ids_vectorized(
                    input_ids_1d=forward_batch.input_ids,
                    full_logits_2d=logits_output.full_logits,
                    mask_id=self.mask_id,
                    block_size=self.block_size,
                    threshold=self.threshold,
                    finished=None,
                    force_at_least_one=True,
                )
                continue

            for batch_id in range(batch_size):
                curr_block_start = batch_id * self.block_size
                curr_block_end = curr_block_start + self.block_size
                block_input_ids = forward_batch.input_ids[
                    curr_block_start:curr_block_end,
                ]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[
                    curr_block_start:curr_block_end,
                ]

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True

                block_input_ids[transfer_index] = x[transfer_index]

        if _is_npu:
            out = model_runner.forward(
                forward_batch, skip_attn_backend_init, pp_proxy_tensors=None
            )
        else:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Here next token ids is tricky to implement the dynamic lengths,
        # so we return a list of tensors
        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
