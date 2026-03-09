from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


def sample_tokens_with_entropy(logits, temperature=1.0):
    """Compute entropy and sample tokens. Copied from d3LLM."""
    original_probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(original_probs + 1e-8)
    entropy = -torch.sum(original_probs * log_probs, dim=-1)

    if temperature == 0:
        samples = torch.argmax(logits, dim=-1)
    else:
        scaled_logits = logits / temperature
        # Convert to probabilities and sample
        probs = torch.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return entropy, samples


class EntropyThreshold(DllmAlgorithm):

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
        # Per-item token counts: supports variable-length sequences (needs_full_prefill)
        item_lens = forward_batch.extend_seq_lens_cpu
        offsets = [0]
        for l in item_lens:
            offsets.append(offsets[-1] + l)
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            next_token_ids = []
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each batch item
        for i in range(batch_size):
            block_input_ids = forward_batch.input_ids[offsets[i] : offsets[i + 1]]
            block_mask_index = block_input_ids == self.mask_id
            start = item_lens[i] - torch.sum(block_mask_index).item()
            start_list.append(start)

        nfe = 0
        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            nfe += 1
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            for batch_id in range(batch_size):
                s, e = offsets[batch_id], offsets[batch_id + 1]
                block_input_ids = forward_batch.input_ids[s:e]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[s:e]

                # Entropy-based selection (matching d3LLM's entropy_threshold algorithm)
                mask_logits = curr_logits[block_mask_index]
                entropy, x0 = sample_tokens_with_entropy(mask_logits, temperature=0)

                x = block_input_ids.clone()
                full_entropy = torch.full_like(
                    block_input_ids, float("inf"), dtype=curr_logits.dtype
                )

                x[block_mask_index] = x0
                full_entropy[block_mask_index] = entropy

                num_mask = block_mask_index.sum().item()
                selected_entropy, select_index = torch.topk(
                    full_entropy, num_mask, largest=False
                )
                transfer_index = torch.zeros_like(block_input_ids, dtype=torch.bool)

                # Always accept the lowest-entropy token; accept others if entropy < threshold
                transfer_index[select_index[0]] = True
                for k in range(1, num_mask):
                    if selected_entropy[k] < self.threshold:
                        transfer_index[select_index[k]] = True
                    else:
                        transfer_index[select_index[k]] = False

                block_input_ids[transfer_index] = x[transfer_index]

        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        nfe += 1
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        # Build per-sequence next_token_ids using variable-length offsets
        next_token_ids_list = [
            forward_batch.input_ids[offsets[i] + start_list[i] : offsets[i + 1]]
            for i in range(batch_size)
        ]

        # Token per forward: attach nfe so it flows via customized_info -> meta_info
        logits_output.customized_info = {"nfe": [nfe] * batch_size}
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = EntropyThreshold
