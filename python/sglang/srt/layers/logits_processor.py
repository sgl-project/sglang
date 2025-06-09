# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Logits processing."""

import dataclasses
import logging
from typing import List, Optional, Union

import torch
import triton
import triton.language as tl
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.srt.layers.dp_attention import (
    attn_tp_all_gather,
    dp_gather_replicate,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_size,
    get_local_attention_dp_rank,
    get_local_attention_dp_size,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import dump_to_file

logger = logging.getLogger(__name__)


from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import dump_to_file

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class LogitsProcessorOutput:
    ## Part 1: This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logits of the next tokens.       shape: [#seq, vocab_size]
    next_token_logits: torch.Tensor
    # Used by speculative decoding (EAGLE)
    # The last hidden layers
    hidden_states: Optional[torch.Tensor] = None

    ## Part 2: This part will be assigned in python/sglang/srt/layers/sampler.py::Sampler
    # The logprobs of the next tokens.                              shape: [#seq]
    next_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in output positions. shape: [#seq, k]
    next_token_top_logprobs_val: Optional[List] = None
    next_token_top_logprobs_idx: Optional[List] = None
    # The logprobs and ids of the requested token ids in output positions. shape: [#seq, n] (n is the number of requested token ids)
    next_token_token_ids_logprobs_val: Optional[List] = None
    next_token_token_ids_logprobs_idx: Optional[List] = None

    ## Part 3: Prefill-only. This part will be assigned in python/sglang/srt/layers/logits_processor.py::LogitsProcessor
    # The logprobs of input tokens.        shape: [#token]
    input_token_logprobs: Optional[torch.Tensor] = None
    # The logprobs and ids of the top-k tokens in input positions.  shape: [#seq, #token, k]
    input_top_logprobs_val: List = None
    input_top_logprobs_idx: List = None
    # The logprobs and ids of the requested token ids in input positions. shape: [#seq, n] (n is the number of requested token ids)
    input_token_ids_logprobs_val: Optional[List] = None
    input_token_ids_logprobs_idx: Optional[List] = None


@dataclasses.dataclass
class LogitsMetadata:
    forward_mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    extend_return_logprob: bool = False
    extend_return_top_logprob: bool = False
    extend_token_ids_logprob: bool = False
    extend_seq_lens: Optional[torch.Tensor] = None
    extend_seq_lens_cpu: Optional[List[int]] = None
    extend_logprob_start_lens_cpu: Optional[List[int]] = None
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None
    top_logprobs_nums: Optional[List[int]] = None
    extend_input_logprob_token_ids_gpu: Optional[torch.Tensor] = None
    token_ids_logprobs: Optional[List[List[int]]] = None

    # logits and logprobs post processing
    temp_scaled_logprobs: bool = False
    temperature: torch.Tensor = None
    top_p_normalized_logprobs: bool = False
    top_p: torch.Tensor = None

    # DP attention metadata. Not needed when DP attention is not used.
    # Number of tokens in the request.
    global_num_tokens_gpu: Optional[torch.Tensor] = None
    # The start position of local hidden states.
    dp_local_start_pos: Optional[torch.Tensor] = None
    dp_local_num_tokens: Optional[torch.Tensor] = None
    gathered_buffer: Optional[torch.Tensor] = None
    # Buffer to gather logits from all ranks.
    forward_batch_gathered_buffer: Optional[torch.Tensor] = None
    # Number of tokens to sample per DP rank
    global_num_tokens_for_logprob_cpu: Optional[torch.Tensor] = None
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor] = None

    # for padding
    padded_static_len: int = -1

    @classmethod
    def from_forward_batch(cls, forward_batch: ForwardBatch):
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.return_logprob
            and not forward_batch.forward_mode.is_target_verify()
        ):
            extend_return_top_logprob = any(
                x > 0 for x in forward_batch.top_logprobs_nums
            )
            extend_token_ids_logprob = any(
                x is not None for x in forward_batch.token_ids_logprobs
            )
            extend_return_logprob = False
            extend_logprob_pruned_lens_cpu = []
            for extend_len, start_len in zip(
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_logprob_start_lens_cpu,
            ):
                if extend_len - start_len > 0:
                    extend_return_logprob = True
                extend_logprob_pruned_lens_cpu.append(extend_len - start_len)
        else:
            extend_return_logprob = extend_return_top_logprob = (
                extend_token_ids_logprob
            ) = extend_logprob_pruned_lens_cpu = False

        return cls(
            forward_mode=forward_batch.forward_mode,
            capture_hidden_mode=forward_batch.capture_hidden_mode,
            extend_return_logprob=extend_return_logprob,
            extend_return_top_logprob=extend_return_top_logprob,
            extend_token_ids_logprob=extend_token_ids_logprob,
            extend_seq_lens=forward_batch.extend_seq_lens,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            extend_logprob_start_lens_cpu=forward_batch.extend_logprob_start_lens_cpu,
            extend_logprob_pruned_lens_cpu=extend_logprob_pruned_lens_cpu,
            top_logprobs_nums=forward_batch.top_logprobs_nums,
            token_ids_logprobs=forward_batch.token_ids_logprobs,
            extend_input_logprob_token_ids_gpu=forward_batch.extend_input_logprob_token_ids_gpu,
            padded_static_len=forward_batch.padded_static_len,
            global_num_tokens_gpu=forward_batch.global_num_tokens_gpu,
            dp_local_start_pos=forward_batch.dp_local_start_pos,
            dp_local_num_tokens=forward_batch.dp_local_num_tokens,
            gathered_buffer=forward_batch.gathered_buffer,
            forward_batch_gathered_buffer=forward_batch.gathered_buffer,
            global_num_tokens_for_logprob_cpu=forward_batch.global_num_tokens_for_logprob_cpu,
            global_num_tokens_for_logprob_gpu=forward_batch.global_num_tokens_for_logprob_gpu,
        )

    def compute_dp_attention_metadata(self, hidden_states: torch.Tensor):
        if self.global_num_tokens_for_logprob_cpu is None:
            # we are capturing cuda graph
            return

        cumtokens = torch.cumsum(self.global_num_tokens_for_logprob_gpu, dim=0)
        dp_rank = get_local_attention_dp_rank()
        if dp_rank == 0:
            dp_local_start_pos = torch.zeros_like(
                self.global_num_tokens_for_logprob_gpu[0]
            )
        else:
            dp_local_start_pos = cumtokens[dp_rank - 1]
        dp_local_num_tokens = self.global_num_tokens_for_logprob_gpu[dp_rank]
        gathered_buffer = torch.zeros(
            (
                sum(self.global_num_tokens_for_logprob_cpu),
                hidden_states.shape[1],
            ),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        self.dp_local_start_pos = dp_local_start_pos
        self.dp_local_num_tokens = dp_local_num_tokens
        self.gathered_buffer = gathered_buffer


class LogitsProcessor(nn.Module):
    def __init__(
        self, config, skip_all_gather: bool = False, logit_scale: Optional[float] = None
    ):
        super().__init__()
        self.config = config
        self.logit_scale = logit_scale
        self.use_attn_tp_group = global_server_args_dict["enable_dp_lm_head"]
        if self.use_attn_tp_group:
            self.attn_tp_size = get_attention_tp_size()
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and self.attn_tp_size > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = False
        else:
            self.do_tensor_parallel_all_gather = (
                not skip_all_gather and get_tensor_model_parallel_world_size() > 1
            )
            self.do_tensor_parallel_all_gather_dp_attn = (
                self.do_tensor_parallel_all_gather and get_attention_dp_size() != 1
            )
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )
        if (
            self.final_logit_softcapping is not None
            and self.final_logit_softcapping < 0
        ):
            self.final_logit_softcapping = None

        self.debug_tensor_dump_output_folder = global_server_args_dict.get(
            "debug_tensor_dump_output_folder", None
        )

        # 添加用于捕获隐藏状态的标志
        self.capture_hidden_states = False

    def forward(
        self,
        input_ids,
        hidden_states,
        lm_head: VocabParallelEmbedding,
        logits_metadata: Union[LogitsMetadata, ForwardBatch],
        aux_hidden_states: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        """Process hidden states to get the logits.

        Args:
            input_ids: input token ids.
            hidden_states: hidden states from transformer.
            lm_head: the LM head.
            logits_metadata: metadata for logits processing.
            aux_hidden_states: auxiliary hidden states for tasks like speculative generation.

        Returns:
            LogitsProcessorOutput: processed logits.
        """
        if isinstance(logits_metadata, ForwardBatch):
            logits_metadata = LogitsMetadata.from_forward_batch(logits_metadata)

        # Set the flag for capturing hidden states
        self.capture_hidden_states = (
            logits_metadata.capture_hidden_mode != CaptureHiddenMode.NULL
        )
        
        if logits_metadata.forward_mode.is_prefill():
            # During prefill, we need to return the logits for all output tokens
            # and optionally the logits for all input tokens.
            # If the hidden states contains all token positions, we need to
            # separate the output positions and input positions.

            # Get the logits for the output tokens (next tokens).
            sampled_logits = self._get_logits(hidden_states, lm_head, logits_metadata)
            
            # Check if we need to return additional information.
            if (
                logits_metadata.forward_mode.is_extend()
                and logits_metadata.extend_return_logprob
            ):
                if not logits_metadata.forward_mode.is_extend_only_output():
                    # Calculate input token logprobs.
                    # Hidden states extraction logic here...
                    if (
                        logits_metadata.padded_static_len > 0
                        and len(logits_metadata.extend_seq_lens_cpu) > 0
                    ):
                        # For static batching.
                        extend_start_pos = 0
                        for i, l in enumerate(logits_metadata.extend_seq_lens_cpu):
                            if i < len(logits_metadata.extend_logprob_start_lens_cpu):
                                extend_start_pos += logits_metadata.extend_logprob_start_lens_cpu[
                                    i
                                ]
                            else:
                                extend_start_pos += l
                        pruned_hidden_states = hidden_states[
                            : extend_start_pos, :
                        ].float()
                    else:
                        # For dynamic batching.
                        pruned_hidden_states = hidden_states.float()

                    if self.capture_hidden_states:
                        hidden_states_to_store = pruned_hidden_states.clone()
                    else:
                        hidden_states_to_store = None

                    # Compute logits for ALL (pruned_len) tokens.
                    all_logits = self._get_logits(
                        pruned_hidden_states, lm_head, logits_metadata
                    )
                    all_logprobs = torch.nn.functional.log_softmax(all_logits, dim=-1)
                    
                    # Get the logprob of top-k tokens
                    if logits_metadata.extend_return_top_logprob:
                        (
                            input_top_logprobs_val,
                            input_top_logprobs_idx,
                        ) = self.get_top_logprobs(all_logprobs, logits_metadata)
                    else:
                        input_top_logprobs_val = input_top_logprobs_idx = None

                    # Get the logprob of given token id
                    if logits_metadata.extend_token_ids_logprob:
                        (
                            input_token_ids_logprobs_val,
                            input_token_ids_logprobs_idx,
                        ) = self.get_token_ids_logprobs(all_logprobs, logits_metadata)
                    else:
                        input_token_ids_logprobs_val = input_token_ids_logprobs_idx = None

                    input_token_logprobs = all_logprobs[
                        torch.arange(all_logprobs.shape[0], device=all_logprobs.device),
                        logits_metadata.extend_input_logprob_token_ids_gpu,
                    ]

                    return LogitsProcessorOutput(
                        next_token_logits=sampled_logits,
                        input_token_logprobs=input_token_logprobs,
                        input_top_logprobs_val=input_top_logprobs_val,
                        input_top_logprobs_idx=input_top_logprobs_idx,
                        hidden_states=hidden_states_to_store,
                        input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                        input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    )
                else:
                    # Only return next token logits.
                    return LogitsProcessorOutput(
                        next_token_logits=sampled_logits,
                        hidden_states=hidden_states
                        if self.capture_hidden_states
                        else None,
                    )
            else:
                # Just prefill without extend options.
                return LogitsProcessorOutput(
                    next_token_logits=sampled_logits,
                    hidden_states=hidden_states if self.capture_hidden_states else None,
                )
        else:
            # During decode, we only need to return the logits for the next token.
            sampled_logits = self._get_logits(hidden_states, lm_head, logits_metadata)

            # Return aux_hidden_states for speculative decoding if needed
            if aux_hidden_states is not None:
                return LogitsProcessorOutput(
                    next_token_logits=sampled_logits,
                    hidden_states=aux_hidden_states
                    if self.capture_hidden_states
                    else None,
                )
            else:
                return LogitsProcessorOutput(
                    next_token_logits=sampled_logits,
                    hidden_states=hidden_states if self.capture_hidden_states else None,
                )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        logits_metadata: LogitsMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get logits from hidden_states."""
        if self.do_tensor_parallel_all_gather_dp_attn:
            logits_metadata.compute_dp_attention_metadata(hidden_states)
            hidden_states, local_hidden_states = (
                logits_metadata.gathered_buffer,
                hidden_states.clone(),
            )
            dp_gather_replicate(hidden_states, local_hidden_states, logits_metadata)

        # Compute logits
        if hasattr(lm_head, "weight") and lm_head.weight is not None:
            # Use weight-based computation
            logits = torch.matmul(hidden_states.to(lm_head.weight.dtype), lm_head.weight.T)
        else:
            # For GGUF or other models that use a forward pass for the head
            logits = lm_head(hidden_states, embedding_bias)

        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)

        if self.do_tensor_parallel_all_gather:
            if self.use_attn_tp_group:
                global_logits = torch.empty(
                    (self.config.vocab_size, logits.shape[0]),
                    device=logits.device,
                    dtype=logits.dtype,
                )
                global_logits = global_logits.T
                attn_tp_all_gather(
                    list(global_logits.tensor_split(self.attn_tp_size, dim=-1)), logits
                )
                logits = global_logits
            else:
                logits = tensor_model_parallel_all_gather(logits)

        if self.do_tensor_parallel_all_gather_dp_attn:
            logits, global_logits = (
                torch.empty(
                    (local_hidden_states.shape[0], logits.shape[1]),
                    device=logits.device,
                    dtype=logits.dtype,
                ),
                logits,
            )
            dp_scatter(logits, global_logits, logits_metadata)

            logits = logits[:, : self.config.vocab_size].float()

        if self.final_logit_softcapping:
            fused_softcap(logits, self.final_logit_softcapping)

        return logits

    @staticmethod
    def get_top_logprobs(all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata):
        max_k = max(logits_metadata.top_logprobs_nums)
        ret = all_logprobs.topk(max_k, dim=1)
        values = ret.values.tolist()
        indices = ret.indices.tolist()

        input_top_logprobs_val, input_top_logprobs_idx = [], []

        pt = 0
        for k, pruned_len in zip(
            logits_metadata.top_logprobs_nums,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_top_logprobs_val.append([])
                input_top_logprobs_idx.append([])
                continue

            input_top_logprobs_val.append(
                [values[pt + j][:k] for j in range(pruned_len)]
            )
            input_top_logprobs_idx.append(
                [indices[pt + j][:k] for j in range(pruned_len)]
            )
            pt += pruned_len

        return input_top_logprobs_val, input_top_logprobs_idx

    @staticmethod
    def get_token_ids_logprobs(
        all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
    ):
        input_token_ids_logprobs_val, input_token_ids_logprobs_idx = [], []
        pt = 0
        for token_ids, pruned_len in zip(
            logits_metadata.token_ids_logprobs,
            logits_metadata.extend_logprob_pruned_lens_cpu,
        ):
            if pruned_len <= 0:
                input_token_ids_logprobs_val.append([])
                input_token_ids_logprobs_idx.append([])
                continue

            input_token_ids_logprobs_val.append(
                [all_logprobs[pt + j, token_ids].tolist() for j in range(pruned_len)]
            )
            input_token_ids_logprobs_idx.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len

        return input_token_ids_logprobs_val, input_token_ids_logprobs_idx

    @staticmethod
    def compute_temp_top_p_normalized_logprobs(
        last_logits: torch.Tensor, logits_metadata: LogitsMetadata
    ) -> torch.Tensor:
        """
        compute logprobs for the output token from the given logits.

        Returns:
            torch.Tensor: logprobs from logits
        """
        # Scale logits if temperature scaling is enabled
        if logits_metadata.temp_scaled_logprobs:
            last_logits = last_logits / logits_metadata.temperature

        # Normalize logprobs if top_p normalization is enabled
        # NOTE: only normalize logprobs when top_p is set and not equal to 1.0
        if (
            logits_metadata.top_p_normalized_logprobs
            and (logits_metadata.top_p != 1.0).any()
        ):
            from sglang.srt.layers.sampler import top_p_normalize_probs_torch

            probs = torch.softmax(last_logits, dim=-1)
            del last_logits
            probs = top_p_normalize_probs_torch(probs, logits_metadata.top_p)
            return torch.log(probs)
        else:
            return torch.nn.functional.log_softmax(last_logits, dim=-1)


@triton.jit
def fused_softcap_kernel(
    full_logits_ptr,
    softcapping_value,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load values
    x = tl.load(full_logits_ptr + offsets, mask=mask)

    # Perform operations in-place
    x = x / softcapping_value

    # Manual tanh implementation using exp
    exp2x = tl.exp(2 * x)
    x = (exp2x - 1) / (exp2x + 1)

    x = x * softcapping_value

    # Store result
    tl.store(full_logits_ptr + offsets, x, mask=mask)


def fused_softcap(full_logits, final_logit_softcapping):
    n_elements = full_logits.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 1, 1)

    fused_softcap_kernel[grid](
        full_logits_ptr=full_logits,
        softcapping_value=final_logit_softcapping,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return full_logits
