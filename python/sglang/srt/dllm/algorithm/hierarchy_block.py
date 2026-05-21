from typing import List, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class HierarchyBlock(DllmAlgorithm):
    """Fast dLLM v2 hierarchical block decoding with token inheritance."""

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.sub_block_size = config.algorithm_config.get("sub_block_size", 8)
        self.token_shift = config.algorithm_config.get("token_shift", 1)

        self.last_inherited_token = None

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        # Per-request inheritance state (last_inherited_token, positions[0]==0
        # new-request detection) assumes a single in-flight request. DLLM is
        # launched with --max-running-requests=1, so enforce that here.
        assert forward_batch.batch_size == 1, (
            "HierarchyBlock requires batch_size == 1; "
            f"got {forward_batch.batch_size}"
        )

        total_len = len(forward_batch.input_ids)
        block_mask = forward_batch.input_ids == self.mask_id
        num_masked = block_mask.sum().item()
        num_sub_blocks = total_len // self.sub_block_size
        block_start = total_len - num_masked

        # Detect new request (positions start from 0) and clear inheritance.
        is_new_request = False
        if forward_batch.positions is not None and forward_batch.positions[0] == 0:
            is_new_request = True
            self.last_inherited_token = None

        # Handle token inheritance for all-mask blocks.
        first_token_is_mask = forward_batch.input_ids[0] == self.mask_id
        if (
            first_token_is_mask
            and self.last_inherited_token is not None
            and not is_new_request
        ):
            forward_batch.input_ids[0] = self.last_inherited_token

        # Process sub-blocks.
        for sub_idx in range(num_sub_blocks):
            rel_start = sub_idx * self.sub_block_size
            rel_end = rel_start + self.sub_block_size

            while True:
                sub_mask = forward_batch.input_ids[rel_start:rel_end] == self.mask_id
                if sub_mask.sum().item() == 0:
                    break

                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
                logits_output, can_run_cuda_graph = (
                    out.logits_output,
                    out.can_run_graph,
                )
                full_logits = logits_output.full_logits
                assert full_logits is not None

                # Token shift: [L0, L0, L1, ..., LN-2]
                if self.token_shift > 0:
                    shifted_full = torch.cat([full_logits[:1], full_logits[:-1]], dim=0)
                else:
                    shifted_full = full_logits

                sub_logits = shifted_full[rel_start:rel_end, :]

                # Compute predictions and confidence.
                preds = sub_logits.argmax(dim=-1)
                probs = F.softmax(sub_logits, dim=-1)
                conf = probs.gather(dim=-1, index=preds.unsqueeze(-1)).squeeze(-1)
                conf = torch.where(sub_mask, conf, float("-inf"))

                # Confidence-based unmask: accept tokens above threshold, or
                # fall back to the single highest-confidence token.
                unmask = conf > self.threshold
                if unmask.sum().item() == 0:
                    unmask[conf.argmax()] = True
                unmask = unmask & sub_mask

                forward_batch.input_ids[rel_start:rel_end] = torch.where(
                    unmask, preds, forward_batch.input_ids[rel_start:rel_end]
                )

        # Final forward to get the inherited token for the next block.
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        full_logits = logits_output.full_logits

        if full_logits is not None:
            self.last_inherited_token = full_logits[-1].argmax().item()

        # Wrap per-request tokens in a list to match the contract expected by
        # Scheduler.process_batch_result_dllm (see low_confidence.py /
        # joint_threshold.py). DLLM currently runs with max_running_requests=1,
        # so the list always has one element.
        return (
            logits_output,
            [forward_batch.input_ids[block_start:]],
            can_run_cuda_graph,
        )


Algorithm = HierarchyBlock
