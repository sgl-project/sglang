from __future__ import annotations

import dataclasses
import logging
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.utils.common import crash_on_warnings

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.speculative.eagle_info import EagleVerifyOutput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes
# ============================================================


class LogprobStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclasses.dataclass
class LogprobResult:
    """Unified result from logprob computation, shared by input and output processors."""

    token_logprobs: Optional[torch.Tensor] = None
    top_logprobs_val: Optional[List] = None
    top_logprobs_idx: Optional[List] = None
    token_ids_logprobs_val: Optional[List[Union[List[float], torch.Tensor]]] = None
    token_ids_logprobs_idx: Optional[List] = None


InputLogprobsResult = LogprobResult


# ============================================================
# Core Utilities
# ============================================================


def compute_temp_top_p_normalized_logprobs(
    last_logits: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_p: Optional[torch.Tensor] = None,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    compute logprobs for the output token from the given logits.

    Returns:
        torch.Tensor: logprobs from logits
    """
    if top_p is None:
        top_p = logits_metadata.top_p
    if temperature is None:
        temperature = logits_metadata.temperature

    # Scale logits if temperature scaling is enabled
    if logits_metadata.temp_scaled_logprobs:
        last_logits = last_logits / temperature

    # Normalize logprobs if top_p normalization is enabled
    # NOTE: only normalize logprobs when top_p is set and not equal to 1.0
    if logits_metadata.top_p_normalized_logprobs and (top_p != 1.0).any():
        from sglang.srt.layers.sampler import top_p_normalize_probs_torch

        probs = torch.softmax(last_logits, dim=-1)
        del last_logits
        probs = top_p_normalize_probs_torch(probs, top_p)
        return torch.log(probs)
    else:
        return torch.nn.functional.log_softmax(last_logits, dim=-1)


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
    values = values.tolist()
    indices = indices.tolist()

    top_logprobs_val = []
    top_logprobs_idx = []

    if stage == LogprobStage.DECODE:
        for i, k in enumerate(top_logprobs_nums):
            top_logprobs_val.append(values[i][:k])
            top_logprobs_idx.append(indices[i][:k])
    else:
        pt = 0
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0:
                top_logprobs_val.append([])
                top_logprobs_idx.append([])
                continue

            top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])
            top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])
            pt += pruned_len

    return top_logprobs_val, top_logprobs_idx


def get_top_logprobs_prefill(
    all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
):
    return get_top_logprobs_raw(
        all_logprobs,
        logits_metadata.top_logprobs_nums,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
    )


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
):
    return get_top_logprobs_raw(logprobs, top_logprobs_nums, stage=LogprobStage.DECODE)


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[Optional[List[int]]],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    delay_cpu_copy: bool = False,
):
    vals, idxs = [], []
    if stage == LogprobStage.DECODE:
        for i, token_ids in enumerate(token_ids_logprobs):
            if token_ids is None:
                vals.append([])
                idxs.append([])
            else:
                vals.append(logprobs[i, token_ids].tolist())
                idxs.append(token_ids)
    else:  # prefill
        pt = 0
        for token_ids, pruned_len in zip(
            token_ids_logprobs, extend_logprob_pruned_lens_cpu
        ):
            if pruned_len <= 0:
                vals.append([])
                idxs.append([])
                continue
            pos_logprobs = logprobs[pt : pt + pruned_len, token_ids]
            vals.append(pos_logprobs if delay_cpu_copy else pos_logprobs.tolist())
            idxs.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len
    return vals, idxs


def get_token_ids_logprobs_prefill(
    all_logprobs, logits_metadata: LogitsMetadata, delay_cpu_copy=False
):
    return get_token_ids_logprobs_raw(
        all_logprobs,
        logits_metadata.token_ids_logprobs,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
        delay_cpu_copy=delay_cpu_copy,
    )


def get_token_ids_logprobs(logprobs, token_ids_logprobs):
    return get_token_ids_logprobs_raw(
        logprobs, token_ids_logprobs, stage=LogprobStage.DECODE
    )


def get_token_ids_logprobs_batch_optimized(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[List[int]],
) -> Tuple[List, List]:
    """
    Vectorized batch processing for token ID logprobs extraction.

    Uses a single GPU kernel call for the entire batch instead of multiple
    separate calls, significantly improving performance for large batches.

    Args:
        logprobs: Log probabilities tensor [batch_size, vocab_size]
        token_ids_logprobs: List of token IDs to extract logprobs for

    Example:
        # Input: batch_size=3, vocab_size=5
        logprobs = torch.tensor([
            [-1.2, -2.1, -0.8, -3.0, -1.5],  # batch 0
            [-0.5, -1.8, -2.2, -1.1, -2.7],  # batch 1
            [-2.0, -0.9, -1.4, -2.8, -1.6],  # batch 2
        ])
        token_ids_logprobs = [[1, 3], [2], [0, 2, 4]]

        # Output:
        # values = [tensor([-2.1, -3.0]), tensor([-2.2]), tensor([-2.0, -1.4, -1.6])]
        # indices = [[1, 3], [2], [0, 2, 4]]
    """
    batch_size = len(token_ids_logprobs)
    device = logprobs.device

    # Step 1: Calculate lengths for each request, treating None as empty list
    # Example: [[1, 3], [2], [0, 2, 4]] -> token_lengths = tensor([2, 1, 3])
    token_lengths = torch.tensor(
        [len(token_ids or []) for token_ids in token_ids_logprobs], device=device
    )
    total_tokens = int(token_lengths.sum().item())  # 2 + 1 + 3 = 6

    # Handle edge case where no tokens are requested
    if total_tokens == 0:
        return [logprobs.new_empty(0) for _ in token_ids_logprobs], [
            [] for _ in token_ids_logprobs
        ]

    # Step 2: Build flattened indices using torch operations
    # Example: row_indices = [0, 0, 1, 2, 2, 2] (batch indices repeated by their lengths)
    row_indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device), token_lengths
    )
    # Example: col_indices = [1, 3, 2, 0, 2, 4] (flattened token IDs from all requests)
    col_indices = torch.tensor(
        [
            token_id
            for token_ids in token_ids_logprobs
            for token_id in (token_ids or [])
        ],
        device=device,
        dtype=torch.long,
    )

    # Step 3: Single vectorized gather operation
    # Example: logprobs[row_indices, col_indices] -> [-2.1, -3.0, -2.2, -2.0, -1.4, -1.6]
    gathered_logprobs = logprobs[row_indices, col_indices]

    # Step 4: Split results back per request using torch operations
    # Example: split tensor [6] into chunks of sizes [2, 1, 3] -> [tensor(2), tensor(1), tensor(3)]
    split_logprobs = torch.split_with_sizes(
        gathered_logprobs, token_lengths.tolist(), dim=0
    )

    # Step 5: Format output to match expected return structure
    # Example: Convert split tensors back to list format with proper empty handling
    # i=0: [1,3] -> append split_logprobs[0] and [1,3]
    # i=1: [2] -> append split_logprobs[1] and [2]
    # i=2: [0,2,4] -> append split_logprobs[2] and [0,2,4]
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []

    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None and len(token_ids) > 0:
            output_token_ids_logprobs_val.append(split_logprobs[i])
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append(logprobs.new_empty(0))
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx


def get_top_logprobs_chunk(
    logprobs: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_k_nums: List[int],
    pruned_lens: List[int],
    input_top_logprobs_val: List,
    input_top_logprobs_idx: List,
    split_pruned_len: int,
) -> int:
    """Get top-k logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing top-k and pruned length info
        top_k_nums: List of top-k numbers for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_top_logprobs_val: List to store top-k logprob values
        input_top_logprobs_idx: List to store top-k token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """
    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

    max_k = max(logits_metadata.top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    pt = 0
    next_split_pruned_len = 0
    for n, (k, pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):
        if n == 0:
            # For the first sequence, adjust the pruned length
            pruned_len -= split_pruned_len
        else:
            # After the first sequence, no split in the middle
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no top-k logprobs to process
            input_top_logprobs_val.append([])
            input_top_logprobs_idx.append([])
            continue

        # Get the top-k logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= len(values):
                next_split_pruned_len = split_pruned_len + j
                break
            # Append the top-k logprobs
            val.append(values[pt + j][:k])
            idx.append(indices[pt + j][:k])

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_top_logprobs_val[-1].extend(val)
                input_top_logprobs_idx[-1].extend(idx)
            else:
                input_top_logprobs_val.append(val)
                input_top_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def get_token_ids_logprobs_chunk(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[int],
    pruned_lens: List[int],
    input_token_ids_logprobs_val: List,
    input_token_ids_logprobs_idx: List,
    split_pruned_len: int = 0,
):
    """Get token_ids logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        token_ids_logprobs: List of token IDs for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_token_ids_logprobs_val: List to store token logprob values
        input_token_ids_logprobs_idx: List to store token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """

    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

    pt = 0
    next_split_pruned_len = 0
    for n, (token_ids, pruned_len) in enumerate(
        zip(
            token_ids_logprobs,
            pruned_lens,
        )
    ):
        # Adjust pruned length for first sequence
        if n == 0:
            pruned_len -= split_pruned_len
        else:
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no token ids logprobs to process
            input_token_ids_logprobs_val.append([])
            input_token_ids_logprobs_idx.append([])
            continue

        # Get the token ids logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= logprobs.shape[0]:
                next_split_pruned_len = split_pruned_len + j
                break
            if token_ids is not None:
                val.append(logprobs[pt + j, token_ids].tolist())
                idx.append(token_ids)

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_token_ids_logprobs_val[-1].extend(val)
                input_token_ids_logprobs_idx[-1].extend(idx)
            else:
                input_token_ids_logprobs_val.append(val)
                input_token_ids_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


# ============================================================
# Output Logprob Processor
# ============================================================


class OutputLogprobProcessor(nn.Module):
    """Processes output (decode) logprobs: logits -> log_softmax -> utilities.

    Handles the full output logprob pipeline including preprocessing,
    log_softmax computation, and extraction of top-k / token-id logprobs.

    Entry points:
      - forward:                  logits -> preprocess -> temperature scale -> log_softmax
                                  (used by Sampler before sampling, and for scoring)
      - process_output_logprobs:  attach pre-computed logprob results to LogitsProcessorOutput
                                  (used by Sampler after sampling)
    """

    def __init__(self, use_nan_detection: bool = False):
        super().__init__()
        self.use_nan_detection = use_nan_detection

    def preprocess_logits(
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> torch.Tensor:
        """Apply custom logit processors and handle NaN detection."""
        # Apply the custom logit processors if registered in the sampling info
        if sampling_info.has_custom_logit_processor:
            from sglang.srt.layers.sampler import apply_custom_logit_processor

            apply_custom_logit_processor(logits, sampling_info)

        # Detect and handle NaN values in logits
        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        return logits

    def compute_logprobs_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities from raw logits via log_softmax."""
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def compute_logprobs_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities from a probability distribution."""
        return torch.log(probs)

    def forward(
        self,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        is_all_greedy: bool,
        return_original_logprob: bool,
        rl_on_policy: bool,
        use_ascend_backend: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Preprocess logits and compute logprobs / probs for sampling.

        Single entry point that handles preprocessing, original-logprob caching,
        temperature scaling, and log_softmax / softmax computation.

        Args:
            logits: Raw logits from the model [batch, vocab]
            sampling_info: Batch sampling metadata (temperatures, etc.)
            return_logprob: Whether output logprobs are needed
            is_all_greedy: If True, skip temperature-scaled logprob computation
            return_original_logprob: If True, output logprobs use pre-temperature logits
            rl_on_policy: If True, use bfloat16 log_softmax to match the RL trainer
            use_ascend_backend: If True, return unmodified logits for ascend sampling

        Returns:
            (logits, output_logprobs, sampling_input):
            - logits: preprocessed logits (for greedy argmax or ascend sampling)
            - output_logprobs: logprobs for output reporting (None if not return_logprob)
            - sampling_input: input fed to the sampler — logprobs for RL on-policy,
              probs (in-place softmax) for standard path, None for greedy / ascend
        """
        logits = self.preprocess_logits(logits, sampling_info)

        if is_all_greedy:
            output_logprobs = (
                self.compute_logprobs_from_logits(logits) if return_logprob else None
            )
            return logits, output_logprobs, None

        # If requested, cache original logprobs before temperature scaling.
        output_logprobs = None
        if return_logprob and return_original_logprob:
            output_logprobs = self.compute_logprobs_from_logits(logits)

        # In RL on-policy mode, we use log_softmax to compute logprobs to match the trainer.
        if rl_on_policy:
            # TODO: use more inplace ops to save memory
            logits_div_temperature = (
                logits.bfloat16().div(sampling_info.temperatures).bfloat16()
            )
            sampling_input = self.compute_logprobs_from_logits(logits_div_temperature)
            del logits_div_temperature
            if return_logprob and not return_original_logprob:
                output_logprobs = sampling_input
            return logits, output_logprobs, sampling_input

        # Ascend backend: return unmodified logits for fused sampling kernels.
        if use_ascend_backend:
            if return_logprob and output_logprobs is None:
                output_logprobs = self.compute_logprobs_from_logits(
                    logits.div(sampling_info.temperatures)
                )
            return logits, output_logprobs, None

        # Standard path: in-place op to save memory
        logits.div_(sampling_info.temperatures)
        logits[:] = torch.softmax(logits, dim=-1)
        probs = logits
        if return_logprob and output_logprobs is None:
            output_logprobs = self.compute_logprobs_from_probs(probs)
        return probs, output_logprobs, probs

    def process_output_logprobs(
        self,
        logits_output: LogitsProcessorOutput,
        logprobs: torch.Tensor,
        batch_next_token_ids: torch.Tensor,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[Optional[List[int]]],
    ):
        """Compute and attach logprob results to logits_output."""
        result = self.from_logprobs(
            logprobs, batch_next_token_ids, top_logprobs_nums, token_ids_logprobs
        )
        logits_output.next_token_logprobs = result.token_logprobs
        logits_output.next_token_top_logprobs_val = result.top_logprobs_val
        logits_output.next_token_top_logprobs_idx = result.top_logprobs_idx
        logits_output.next_token_token_ids_logprobs_val = result.token_ids_logprobs_val
        logits_output.next_token_token_ids_logprobs_idx = result.token_ids_logprobs_idx

    def from_logprobs(
        self,
        logprobs: torch.Tensor,
        batch_next_token_ids: torch.Tensor,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[Optional[List[int]]],
    ) -> LogprobResult:
        """Extract logprob results from pre-computed logprobs."""

        # clamp to avoid -inf values
        logprobs.clamp_(min=torch.finfo(logprobs.dtype).min)

        next_token_top_logprobs_val = next_token_top_logprobs_idx = None
        if any(x > 0 for x in top_logprobs_nums):
            next_token_top_logprobs_val, next_token_top_logprobs_idx = get_top_logprobs(
                logprobs, top_logprobs_nums
            )

        next_token_token_ids_logprobs_val = next_token_token_ids_logprobs_idx = None
        if any(x is not None for x in token_ids_logprobs):
            next_token_token_ids_logprobs_val, next_token_token_ids_logprobs_idx = (
                get_token_ids_logprobs(logprobs, token_ids_logprobs)
            )

        token_logprobs = logprobs[
            torch.arange(len(batch_next_token_ids), device=logprobs.device),
            batch_next_token_ids,
        ]

        return LogprobResult(
            token_logprobs=token_logprobs,
            top_logprobs_val=next_token_top_logprobs_val,
            top_logprobs_idx=next_token_top_logprobs_idx,
            token_ids_logprobs_val=next_token_token_ids_logprobs_val,
            token_ids_logprobs_idx=next_token_token_ids_logprobs_idx,
        )

    def extract_logprobs_for_scoring(
        self,
        logprobs: torch.Tensor,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[Optional[List[int]]],
        needs_token_ids_logprobs: bool,
        needs_top_logprobs: bool,
    ) -> LogprobResult:
        """Extract top-k and token-id logprobs for scoring (prefill-only) requests."""
        next_token_top_logprobs_val = next_token_top_logprobs_idx = None
        next_token_token_ids_logprobs_val = next_token_token_ids_logprobs_idx = None

        # Prefill-only path (scoring): no per-token logprobs needed
        # Handle top logprobs if requested
        if needs_top_logprobs:
            (
                next_token_top_logprobs_val,
                next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums)

        # Handle token_ids logprobs if requested
        if needs_token_ids_logprobs:
            (
                next_token_token_ids_logprobs_val,
                next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_batch_optimized(logprobs, token_ids_logprobs)
        return LogprobResult(
            top_logprobs_val=next_token_top_logprobs_val,
            top_logprobs_idx=next_token_top_logprobs_idx,
            token_ids_logprobs_val=next_token_token_ids_logprobs_val,
            token_ids_logprobs_idx=next_token_token_ids_logprobs_idx,
        )


# ============================================================
# Input Logprob Processor
# ============================================================


class InputLogprobProcessor(nn.Module):
    """Processes input (prefill) logprobs:
    (hidden_states, lm_head, get_logits_fn) -> chunk -> log_softmax -> utilities.

    Single entry point: forward() handles both chunked and non-chunked paths.
    Logits are computed internally via get_logits_fn(hidden_states, lm_head, logits_metadata).
    """

    def __init__(self, chunk_size: int = 4096):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        pruned_states: torch.Tensor,
        sample_indices: Optional[torch.Tensor],
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head,
        get_logits_fn: Callable,
        logits_metadata: LogitsMetadata,
        enable_chunk: bool = False,
    ) -> Tuple[LogprobResult, torch.Tensor]:
        if enable_chunk:
            return self._forward_by_chunk(
                pruned_states,
                sample_indices,
                input_logprob_indices,
                token_to_seq_idx,
                lm_head,
                get_logits_fn,
                logits_metadata,
            )

        logits = get_logits_fn(pruned_states, lm_head, logits_metadata)
        sampled_logits = (
            logits[sample_indices] if sample_indices is not None else logits
        )
        input_logits = logits[input_logprob_indices]
        del logits

        input_logprobs = compute_temp_top_p_normalized_logprobs(
            input_logits, logits_metadata
        )

        # Get the logprob of top-k tokens
        if logits_metadata.extend_return_top_logprob:
            (
                input_top_logprobs_val,
                input_top_logprobs_idx,
            ) = get_top_logprobs_prefill(input_logprobs, logits_metadata)
        else:
            input_top_logprobs_val = input_top_logprobs_idx = None

        # Get the logprob of given token id
        if logits_metadata.extend_token_ids_logprob:
            (
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_prefill(input_logprobs, logits_metadata)
        else:
            input_token_ids_logprobs_val = input_token_ids_logprobs_idx = None

        input_token_logprobs = input_logprobs[
            torch.arange(input_logprobs.shape[0], device=input_logprobs.device),
            logits_metadata.extend_input_logprob_token_ids_gpu,
        ]

        return (
            LogprobResult(
                token_logprobs=input_token_logprobs,
                top_logprobs_val=input_top_logprobs_val,
                top_logprobs_idx=input_top_logprobs_idx,
                token_ids_logprobs_val=input_token_ids_logprobs_val,
                token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            ),
            sampled_logits,
        )

    def _forward_by_chunk(
        self,
        pruned_states: torch.Tensor,
        sample_indices: torch.Tensor,
        input_logprob_indices: torch.Tensor,
        token_to_seq_idx: list[int],
        lm_head,
        get_logits_fn: Callable,
        logits_metadata: LogitsMetadata,
    ) -> Tuple[LogprobResult, torch.Tensor]:
        """Compute input logprobs by chunking hidden states to limit peak memory."""
        # The peak memory usage is proportional to the chunk size.
        chunk_size = self.chunk_size
        total_size = pruned_states.shape[0]
        num_chunks = (total_size + chunk_size - 1) // chunk_size

        input_token_logprobs = []
        if logits_metadata.extend_return_top_logprob:
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
        else:
            input_top_logprobs_val = None
            input_top_logprobs_idx = None
        if logits_metadata.extend_token_ids_logprob:
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
        else:
            input_token_ids_logprobs_val = None
            input_token_ids_logprobs_idx = None

        # If a single sequence is split into multiple chunks, we need to keep track
        # of the pruned length of the sequences in the previous chunks.
        split_len_topk = 0
        split_len_token_ids = 0

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)

            # Get indices for this chunk
            chunk_mask = (input_logprob_indices >= start_idx) & (
                input_logprob_indices < end_idx
            )
            global_indices = input_logprob_indices[chunk_mask]
            chunk_indices = global_indices - start_idx
            # Get the positions in the original array where chunk_mask is True
            # This is needed to correctly index into extend_input_logprob_token_ids_gpu
            mask_indices = torch.nonzero(chunk_mask, as_tuple=True)[0]

            # Get the logits for this chunk
            chunk_states = pruned_states[start_idx:end_idx]
            chunk_logits = get_logits_fn(chunk_states, lm_head, logits_metadata)
            # Initialize sampled_logits on first chunk
            if i == 0:
                sampled_logits = torch.empty(
                    (sample_indices.shape[0], chunk_logits.shape[1]),
                    dtype=chunk_logits.dtype,
                    device=chunk_logits.device,
                )

            # Handle sampled logits for the chunk if needed
            # This must be done before the continue statement to ensure all sampled_logits are filled
            chunk_sample_mask = (sample_indices >= start_idx) & (
                sample_indices < end_idx
            )
            if chunk_sample_mask.any():
                chunk_sample_indices = sample_indices[chunk_sample_mask] - start_idx
                sampled_logits[chunk_sample_mask] = chunk_logits[chunk_sample_indices]

            # If there are no input logprobs in this chunk, skip the rest
            if chunk_indices.numel() == 0:
                continue

            # Compute the logprobs of the chunk
            chunk_input_logprobs = chunk_logits[chunk_indices]
            # Only index per-token arrays when the corresponding feature is active.
            # Otherwise these tensors can be per-sequence (or scalars), which can
            # cause out-of-bounds indexing on GPU.
            chunk_temperature = (
                logits_metadata.temperature[global_indices]
                if logits_metadata.temp_scaled_logprobs
                and logits_metadata.temperature is not None
                else None
            )
            chunk_top_p = (
                logits_metadata.top_p[global_indices]
                if logits_metadata.top_p_normalized_logprobs
                and logits_metadata.top_p is not None
                else None
            )
            chunk_input_logprobs = compute_temp_top_p_normalized_logprobs(
                chunk_input_logprobs,
                logits_metadata,
                chunk_top_p,
                chunk_temperature,
            )

            # For each chunk, we need to get the slice of the token_to_seq_idx
            chunk_slice = slice(
                token_to_seq_idx[start_idx], token_to_seq_idx[end_idx] + 1
            )

            # Get the logprob of top-k tokens
            if logits_metadata.extend_return_top_logprob:
                top_k_nums = logits_metadata.top_logprobs_nums[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_topk = get_top_logprobs_chunk(
                    chunk_input_logprobs,
                    logits_metadata,
                    top_k_nums,
                    pruned_lens,
                    input_top_logprobs_val,
                    input_top_logprobs_idx,
                    split_len_topk,
                )

            # Get the logprob of given token id
            if logits_metadata.extend_token_ids_logprob:
                token_ids_logprobs = logits_metadata.token_ids_logprobs[chunk_slice]
                pruned_lens = logits_metadata.extend_logprob_pruned_lens_cpu[
                    chunk_slice
                ]
                split_len_token_ids = get_token_ids_logprobs_chunk(
                    chunk_input_logprobs,
                    token_ids_logprobs,
                    pruned_lens,
                    input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx,
                    split_len_token_ids,
                )

            # Get the logprob of the requested token ids
            chunk_input_token_logprobs = chunk_input_logprobs[
                torch.arange(
                    chunk_input_logprobs.shape[0], device=chunk_input_logprobs.device
                ),
                logits_metadata.extend_input_logprob_token_ids_gpu[mask_indices],
            ]
            input_token_logprobs.append(chunk_input_token_logprobs)

        # Concatenate the results
        input_token_logprobs = torch.cat(input_token_logprobs, dim=0)

        return (
            LogprobResult(
                token_logprobs=input_token_logprobs,
                top_logprobs_val=input_top_logprobs_val,
                top_logprobs_idx=input_top_logprobs_idx,
                token_ids_logprobs_val=input_token_ids_logprobs_val,
                token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            ),
            sampled_logits,
        )


# ============================================================
# Speculative Decoding Helpers
# ============================================================


def add_output_logprobs_for_spec_v1(
    batch: ScheduleBatch,
    res: Union[EagleVerifyOutput, NgramVerifyInput],
    logits_output: Optional[LogitsProcessorOutput] = None,
):
    # Extract args
    if logits_output is None:
        logits_output = res.logits_output

    if hasattr(res, "accept_length_per_req_cpu"):
        accept_length_per_req_cpu = res.accept_length_per_req_cpu
    else:
        # FIXME: Get a NgramVerifyOutput class and use that instead of this hack.
        accept_length_per_req_cpu = res.accept_length.tolist()

    top_logprobs_nums = batch.top_logprobs_nums
    token_ids_logprobs = batch.token_ids_logprobs
    accepted_indices = res.accepted_indices
    assert len(accepted_indices) == len(logits_output.next_token_logits)

    temperatures = batch.sampling_info.temperatures
    num_draft_tokens = batch.spec_info.draft_token_num
    # acceptance indices are the indices in a "flattened" batch.
    # dividing it to num_draft_tokens will yield the actual batch index.
    temperatures = temperatures[accepted_indices // num_draft_tokens]
    if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
    else:
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits / temperatures, dim=-1
        )
    batch_next_token_ids = res.verified_id
    num_tokens_per_req = [accept + 1 for accept in accept_length_per_req_cpu]

    # We should repeat top_logprobs_nums to match num_tokens_per_req.
    top_logprobs_nums_repeat_interleaved = [
        num
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    token_ids_logprobs_repeat_interleaved = [
        token_ids
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    # Extract logprobs
    should_top_logprobs = any(x > 0 for x in top_logprobs_nums)
    should_token_ids_logprobs = any(x is not None for x in token_ids_logprobs)
    if should_top_logprobs:
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            logprobs,
            top_logprobs_nums_repeat_interleaved,
        )

    if should_token_ids_logprobs:
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            logprobs,
            token_ids_logprobs_repeat_interleaved,
        )

    logits_output.next_token_logprobs = logprobs[
        torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
        batch_next_token_ids,
    ]

    # Add output logprobs to the request
    pt = 0
    next_token_logprobs = logits_output.next_token_logprobs.tolist()
    verified_ids = batch_next_token_ids.tolist()
    token_top_logprobs_val = logits_output.next_token_top_logprobs_val
    token_top_logprobs_idx = logits_output.next_token_top_logprobs_idx
    for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
        for _ in range(num_tokens):
            # TODO: add token_ids_logprobs to each request
            if req.return_logprob:
                req.output_token_logprobs_val.append(next_token_logprobs[pt])
                req.output_token_logprobs_idx.append(verified_ids[pt])
                if req.top_logprobs_num > 0:
                    assert (
                        should_top_logprobs
                    ), "Inconsistent state: should_top_logprobs is False"
                    req.output_top_logprobs_val.append(token_top_logprobs_val[pt])
                    req.output_top_logprobs_idx.append(token_top_logprobs_idx[pt])
            pt += 1
