import math
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

def joint_threshold_update_step_vectorized(
    input_ids_1d: torch.Tensor,          # [B*blk]
    full_logits_2d: torch.Tensor,         # [B*blk, V]
    prompt_mask_2d: torch.Tensor,         # [B, blk]
    finished: torch.Tensor,              # [B]
    post_edit_steps: torch.Tensor,        # [B]
    mask_id: int,
    blk: int,
    threshold: float,
    edit_threshold: float,
    max_post_edit_steps: int,
    penalty_lambda: float,
):
    B = input_ids_1d.shape[0] // blk
    V = full_logits_2d.shape[1]

    input_ids = input_ids_1d.view(B, blk)
    logits = full_logits_2d.view(B, blk, V)

    active = ~finished

    # ---------- penalty ----------
    if penalty_lambda > 0:
        prev_ids = input_ids[:, :-1]
        logits[:, 1:, :].scatter_(
            dim=2,
            index=prev_ids.unsqueeze(-1),
            src=torch.full_like(prev_ids.unsqueeze(-1), -penalty_lambda, dtype=logits.dtype),
            reduce="add",
        )

    # ---------- argmax + logp ----------
    max_logit, x = logits.max(dim=-1)
    lse = torch.logsumexp(logits, dim=-1)
    logp = max_logit - lse

    mask_pos = input_ids.eq(mask_id)
    has_mask = mask_pos.any(dim=1)

    # ---------- post-edit ----------
    no_mask_active = active & (~has_mask)
    post_edit_steps.add_(no_mask_active.to(post_edit_steps.dtype))
    exceeded = post_edit_steps > max_post_edit_steps
    finished |= (no_mask_active & exceeded)

    # eligible rows (match original semantics)
    eligible = active & (~(no_mask_active & exceeded))

    # ---------- M2T ----------
    log_thr = math.log(threshold)
    neg_inf = torch.full_like(logp, float("-inf"))
    conf_m2t = torch.where(mask_pos, logp, neg_inf)

    m2t = (conf_m2t > log_thr) & (eligible & has_mask).view(B, 1)

    # force-one if needed
    hit_any = m2t.any(dim=1)
    need_force = (eligible & has_mask) & (~hit_any)

    best_idx = conf_m2t.argmax(dim=1)
    rows = torch.arange(B, device=input_ids.device)

    m2t[rows, best_idx] |= need_force

    # ---------- T2T ----------
    edit_mask = (~mask_pos) & (~prompt_mask_2d)

    if edit_threshold > 0:
        log_edit_thr = math.log(edit_threshold)
        t2t = (logp > log_edit_thr) & (input_ids != x) & edit_mask
    else:
        # p > 0 always true -> reduce to mismatch & edit_mask
        t2t = (input_ids != x) & edit_mask

    t2t = t2t & eligible.view(B, 1)

    # ---------- combine ----------
    transfer = m2t | t2t
    any_transfer_row = transfer.any(dim=1)

    finished |= (eligible & (~any_transfer_row))

    # apply update
    input_ids.copy_(torch.where(transfer, x, input_ids))

    return any_transfer_row.any()


class JointThreshold(DllmAlgorithm):

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get("max_post_edit_steps", 16)
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)

    def run(self, model_runner: ModelRunner, forward_batch: ForwardBatch):
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        mask_index = forward_batch.input_ids == self.mask_id
        if not mask_index.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # ---------- build prompt_mask_2d ----------
        input_2d = forward_batch.input_ids.view(batch_size, self.block_size)
        prompt_mask_2d = input_2d != self.mask_id
        start_list = prompt_mask_2d.sum(dim=1).tolist()

        post_edit_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        any_changed_flag = torch.tensor(False, device=device)

        if _is_npu:
            skip_attn_backend_init = False

        max_iterations = self.block_size + self.max_post_edit_steps
        for _ in range(max_iterations):
            if finished.all():
                break
            if _is_npu:
                out = model_runner.forward(
                    forward_batch, skip_attn_backend_init, pp_proxy_tensors=None
                )
                skip_attn_backend_init = True
            else:
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            if _is_npu:
                changed_any = joint_threshold_update_step_vectorized(
                    forward_batch.input_ids,
                    logits_output.full_logits,
                    prompt_mask_2d,
                    finished,
                    post_edit_steps,
                    self.mask_id,
                    self.block_size,
                    self.threshold,
                    self.edit_threshold,
                    self.max_post_edit_steps,
                    self.penalty_lambda,
                )
                any_changed_flag = changed_any   # only last step matters
                continue

            # ---------- original CPU path ----------
            any_changed_in_last_step = False
            for i in range(batch_size):
                if finished[i]:
                    continue

                block_start = i * self.block_size
                block_end = block_start + self.block_size

                curr_input_ids = forward_batch.input_ids[block_start:block_end]
                curr_logits = logits_output.full_logits[block_start:block_end]
                curr_prompt_mask = prompt_mask_2d[i]

                if self.penalty_lambda > 0:
                    prev_ids = curr_input_ids[:-1]
                    curr_logits[1:, :].scatter_(
                        1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                    )

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(F.softmax(curr_logits, dim=-1), -1, x.unsqueeze(-1)),
                    -1,
                )

                mask_index = curr_input_ids == self.mask_id
                has_mask = mask_index.any()

                mask_transfer_index = torch.zeros_like(mask_index)
                if has_mask:
                    confidence = torch.where(mask_index, p, -np.inf)
                    mask_transfer_index = confidence > self.threshold
                    if not mask_transfer_index.any():
                        _, idx = torch.topk(confidence, 1)
                        mask_transfer_index[idx] = True
                else:
                    post_edit_steps[i] += 1
                    if post_edit_steps[i] > self.max_post_edit_steps:
                        finished[i] = True
                        continue

                edit_mask = ~mask_index & ~curr_prompt_mask
                if self.edit_threshold > 0:
                    edit_transfer_index = (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                else:
                    edit_transfer_index = (curr_input_ids != x) & edit_mask

                transfer_index = mask_transfer_index | edit_transfer_index
                if not transfer_index.any():
                    finished[i] = True
                    continue

                curr_input_ids[transfer_index] = x[transfer_index]
                any_changed_in_last_step = True

        # ---------- extra forward ----------
        if _is_npu:
            if any_changed_flag.item():
                out = model_runner.forward(forward_batch, skip_attn_backend_init, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        else:
            if any_changed_in_last_step:
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids = forward_batch.input_ids.view(batch_size, -1)
        next_token_ids_list = [next_token_ids[i, start_list[i]:] for i in range(batch_size)]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = JointThreshold

