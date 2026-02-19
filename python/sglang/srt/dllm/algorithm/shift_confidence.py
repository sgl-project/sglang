from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

try:
    import torch_npu
except ImportError:
    pass

def sample_tokens(logits, neg_entropy=False):
    probs = torch.softmax(logits, dim=-1)
    confidence, x0 = probs.max(dim=-1)

    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)

    return confidence, x0

class ShiftConfidence(DllmAlgorithm):
    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.9)
        self.num_small_blocks = config.algorithm_config.get("num_small_blocks", 4)
        self.alg = config.algorithm_config.get("alg", "confidence_threshold")
        self.eos_token_id = config.algorithm_config.get("eos_token_id")
        self.next_token_cache = None
        self.pad_len = 0
        self.pad_id = config.pad_id
        self.fia_length = 32768

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        prompt_flag = forward_batch.input_ids[0].item() != self.mask_id
        is_first_block = forward_batch.positions[0] == 1 or forward_batch.positions[0] == 0
        small_block_length = self.block_size // self.num_small_blocks

        if self.block_size % self.num_small_blocks != 0:
            raise ValueError(f"block_size ({self.block_size}) must be divisible by num_small_blocks ({self.num_small_blocks})")
        
        mask_index = forward_batch.input_ids == self.mask_id
        mask_count = torch.sum(mask_index).item()

        if forward_batch.model_specific_states is None:
            forward_batch.model_specific_states = {}

        if is_first_block:
            pad_len = torch.sum(forward_batch.input_ids == self.pad_id).item()
            if pad_len!=self.pad_len:
                self.pad_len = pad_len
        forward_batch.model_specific_states["pad_len"] = self.pad_len

        if self.pad_len:
            padding_mask = torch.ones(self.fia_length, dtype=torch.bool, device=forward_batch.input_ids.device)
            padding_mask[:self.pad_len] = False
            padding_mask = torch.logical_and(
                padding_mask.unsqueeze(-2),
                padding_mask.unsqueeze(-1),
            )
            padding_mask = padding_mask[forward_batch.seq_lens[0]-forward_batch.input_ids.shape[0]:forward_batch.seq_lens[0]]
        
        if not prompt_flag:
            if self.next_token_cache is not None:
                forward_batch.input_ids[0] = self.next_token_cache
                self.next_token_cache = None

            forward_batch.model_specific_states["attention_mask"] = padding_mask if self.pad_len>0 else None

            for small_block_idx in range(self.num_small_blocks):
                small_block_start = small_block_idx * small_block_length
                small_block_end = small_block_start + small_block_length

                while True:
                    mask_index = forward_batch.input_ids[small_block_start:small_block_end] == self.mask_id
                    mask_count = torch.sum(mask_index).item()
                    if mask_count == 0:
                        break

                    out = model_runner.forward(
                        forward_batch, pp_proxy_tensors=None
                    )
                    logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

                    original_logits = logits_output.full_logits 
                    shifted_logits = torch.cat([original_logits[:1, :], original_logits[:-1, :]], dim=0)
                    shifted_logits = shifted_logits[small_block_start:small_block_end]

                    confidence, x = sample_tokens(shifted_logits, neg_entropy=(self.alg == 'entropy'))
                    confidence = torch.where(mask_index, confidence, -np.inf)

                    transfer_index = (F.one_hot(torch.max(confidence, dim=0)[1], num_classes=small_block_length) == 1)
                    if self.alg == 'confidence_threshold':
                        transfer_index |= (confidence > self.threshold)

                    forward_batch.input_ids[small_block_start:small_block_end][transfer_index] = x[transfer_index]
                
                if self.eos_token_id and (forward_batch.input_ids[small_block_start:small_block_end] == self.eos_token_id).any():
                    break  

        causal_mask = torch.tril(torch.ones(self.fia_length, self.fia_length, device=forward_batch.input_ids.device, dtype=torch.bool))
        causal_mask = causal_mask[forward_batch.seq_lens[0]-forward_batch.input_ids.shape[0]:forward_batch.seq_lens[0]]
        forward_batch.model_specific_states["attention_mask"] = (padding_mask & causal_mask) if self.pad_len > 0 else causal_mask
        
        out = model_runner.forward(
            forward_batch, pp_proxy_tensors=None
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        
        last_logits = logits_output.full_logits[len(forward_batch.input_ids)-mask_count-1, :]
        self.next_token_cache = torch.argmax(last_logits, dim=-1)

        if prompt_flag:
            next_token_ids = []
        else :
            next_token_ids = [forward_batch.input_ids]

        return logits_output, next_token_ids, can_run_cuda_graph

Algorithm = ShiftConfidence