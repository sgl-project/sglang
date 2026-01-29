from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class LowConfidenceFDFO(DllmAlgorithm):
    """Low Confidence algorithm for DLLM. Requiring first done first out mode"""

    requires_fdfo_mode: bool = True

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)

    def _pick_tokens(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor
    ) -> None:
        """PyTorch implementation of pick_tokens (fallback)."""
        batch_size = forward_batch.batch_size
        vocab_size = full_logits.shape[-1]
        full_logits = full_logits.view(batch_size, self.block_size, vocab_size)
        input_ids = forward_batch.input_ids.view(batch_size, self.block_size)
        block_mask_index = input_ids == self.mask_id
        x = torch.argmax(full_logits, dim=-1)
        probs = torch.nn.functional.softmax(full_logits, dim=-1)
        confidence = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(
            block_mask_index,
            confidence,
            torch.tensor(-float("inf"), device=confidence.device),
        )
        transfer_index = confidence > self.threshold
        has_transfer = transfer_index.sum(dim=1) > 0
        _, top1_indices = torch.topk(confidence, k=1, dim=1)
        top1_indices = top1_indices.squeeze(-1)
        batch_indices = torch.arange(batch_size, device=top1_indices.device)
        top1_mask = torch.zeros_like(transfer_index, dtype=torch.bool)
        top1_mask[batch_indices, top1_indices] = True
        transfer_index = torch.where(
            has_transfer.unsqueeze(-1), transfer_index, top1_mask
        )
        x = torch.where(block_mask_index, x, input_ids)
        input_ids = torch.where(transfer_index, x, input_ids)

        forward_batch.input_ids = input_ids.view(-1)

    def _post_forward_process(
        self, forward_batch: ForwardBatch, full_logits: torch.Tensor
    ) -> List[List[int]]:
        # Validate batch size
        batch_size = forward_batch.batch_size
        expected_batch_size = forward_batch.input_ids.shape[0] // self.block_size
        if batch_size != expected_batch_size:
            raise RuntimeError(
                f"Batch size mismatch: forward_batch.batch_size={batch_size}, "
                f"but input_ids shape {forward_batch.input_ids.shape[0]} / "
                f"block_size {self.block_size} = {expected_batch_size}"
            )

        # Compute mask counts per block BEFORE _pick_tokens modifies input_ids
        # Convert mask counts to CPU once to avoid multiple D2H transfers
        mask_counts_cpu = (
            (forward_batch.input_ids == self.mask_id)
            .view(batch_size, self.block_size)
            .sum(dim=1)
            .tolist()
        )

        # Update input_ids based on confidence
        self._pick_tokens(forward_batch, full_logits)

        # Build output token IDs list with decode markers
        # Reshape and convert to CPU list in one operation
        next_token_ids = forward_batch.input_ids.view(
            batch_size, self.block_size
        ).tolist()

        next_token_ids_list = []
        accept_length_per_req_cpu = []
        for i in range(batch_size):
            next_token_ids_list.append(next_token_ids[i])
            accept_length_per_req_cpu.append(
                self.block_size if mask_counts_cpu[i] == 0 else 0
            )

        return next_token_ids_list, accept_length_per_req_cpu

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], List[List[int]], List[int], bool
    ]:
        # Forward pass through the model (no preprocessing needed)
        # each block may contain mask or finished tokens
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        # Post-forward processing: pick tokens and build output list
        next_token_ids_list, accept_length_per_req_cpu = self._post_forward_process(
            forward_batch, logits_output.full_logits
        )

        return (
            logits_output,
            next_token_ids_list,
            accept_length_per_req_cpu,
            can_run_cuda_graph,
        )


Algorithm = LowConfidenceFDFO
