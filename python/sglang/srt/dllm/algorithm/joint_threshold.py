import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class JointThreshold(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.5)
        self.edit_threshold = config.algorithm_config.get("edit_threshold", 0)
        self.max_post_edit_steps = config.algorithm_config.get(
            "max_post_edit_steps", 16
        )
        self.penalty_lambda = config.algorithm_config.get("penalty_lambda", 0)

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> tuple[LogitsProcessorOutput | torch.Tensor, torch.Tensor | None, bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device

        mask_index = forward_batch.input_ids == self.mask_id
        if not mask_index.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        start_list = []
        prompt_masks = []
        for i in range(batch_size):
            block_start = i * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]

            prompt_mask = block_input_ids != self.mask_id
            prompt_masks.append(prompt_mask)
            start_list.append(prompt_mask.sum().item())

        post_edit_steps = torch.zeros(batch_size, dtype=torch.int32, device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Controls whether to perform an additional forward pass for KV cache persistence.
        # For certain decoding rounds where the terminal step yields no state change,
        # this can be set to False to bypass the overhead of an idle forward pass.
        any_changed_in_last_step = False

        max_iterations = self.block_size + self.max_post_edit_steps
        for _ in range(max_iterations):
            if finished.all():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            any_changed_in_last_step = False

            for i in range(batch_size):
                if finished[i]:
                    continue

                block_start = i * self.block_size
                block_end = block_start + self.block_size

                curr_input_ids = forward_batch.input_ids[block_start:block_end]
                curr_logits = logits_output.full_logits[block_start:block_end]
                curr_prompt_mask = prompt_masks[i]

                if self.penalty_lambda > 0:
                    prev_ids = curr_input_ids[:-1]
                    curr_logits[1:, :].scatter_(
                        1, prev_ids.unsqueeze(-1), -self.penalty_lambda, reduce="add"
                    )

                x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )

                mask_index = curr_input_ids == self.mask_id
                has_mask = mask_index.any()

                # Mask to token (M2T)
                mask_transfer_index = torch.zeros_like(mask_index)
                if has_mask:
                    confidence = torch.where(mask_index, p, -np.inf)
                    mask_transfer_index = confidence > self.threshold

                    if not mask_transfer_index.any():
                        _, select_index = torch.topk(confidence, k=1)
                        mask_transfer_index[select_index] = True
                else:
                    post_edit_steps[i] += 1
                    if post_edit_steps[i] > self.max_post_edit_steps:
                        finished[i] = True
                        continue

                # Token to token (T2T)
                edit_mask = ~mask_index & ~curr_prompt_mask
                edit_transfer_index = (
                    (p > self.edit_threshold) & (curr_input_ids != x) & edit_mask
                )

                transfer_index = mask_transfer_index | edit_transfer_index
                if not transfer_index.any():
                    finished[i] = True
                    continue

                curr_input_ids[transfer_index] = x[transfer_index]
                any_changed_in_last_step = True

        if any_changed_in_last_step:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        next_token_ids = torch.reshape(forward_batch.input_ids, (batch_size, -1))
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = JointThreshold
