import logging
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = get_global_server_args().enable_nan_detection
        self.tp_sync_group = get_tp_group().device_group

        if is_dp_attention_enabled():
            self.tp_sync_group = get_attention_tp_group().device_group

    def _preprocess_logits(
        self, logits: torch.Tensor, sampling_info: SamplingBatchInfo
    ) -> torch.Tensor:
        """Apply custom logit processors and handle NaN detection."""
        # Apply the custom logit processors if registered in the sampling info
        if sampling_info.has_custom_logit_processor:
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

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
        positions: torch.Tensor,
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
            positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.
        """
        logits = logits_output.next_token_logits

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits, sampling_info)

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        else:
            can_sample_directly_from_probs = (
                not sampling_info.need_top_p_sampling
                and not sampling_info.need_top_k_sampling
                and not sampling_info.need_min_p_sampling
            )

            # If requested, cache probabilities from original logits before temperature scaling.
            if return_logprob and SGLANG_RETURN_ORIGINAL_LOGPROB:
                probs_without_temp_scaling = torch.softmax(logits, dim=-1)

            if get_global_server_args().rl_on_policy_target is not None:
                logits_div_temperature = (
                    logits.bfloat16().div(sampling_info.temperatures).bfloat16()
                )
                logprobs_via_logsoftmax_kernel = torch.log_softmax(
                    logits_div_temperature, dim=-1
                )

            # Post process logits
            logits.div_(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits
            del logits

            if can_sample_directly_from_probs:
                # when we don't need top-k, top-p, or min-p sampling, we can directly sample from the probs
                batch_next_token_ids = sampling_from_probs_torch(
                    probs,
                    sampling_seed=sampling_info.sampling_seed,
                    positions=positions,
                )
            else:
                if get_global_server_args().sampling_backend == "flashinfer":
                    if sampling_info.need_min_p_sampling:
                        probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                        probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                        batch_next_token_ids = min_p_sampling_from_probs(
                            probs, sampling_info.min_ps
                        )
                    else:
                        batch_next_token_ids = top_k_top_p_sampling_from_probs(
                            probs.contiguous(),
                            sampling_info.top_ks,
                            sampling_info.top_ps,
                            filter_apply_order="joint",
                            check_nan=self.use_nan_detection,
                        )
                elif get_global_server_args().sampling_backend == "pytorch":
                    # A slower fallback implementation with torch native operations.
                    batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                        probs,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        sampling_info.min_ps,
                        sampling_info.need_min_p_sampling,
                        sampling_info.sampling_seed,
                        positions,
                    )
                else:
                    raise ValueError(
                        f"Invalid sampling backend: {get_global_server_args().sampling_backend}"
                    )

            if return_logprob:
                if get_global_server_args().rl_on_policy_target is not None:
                    logprobs = logprobs_via_logsoftmax_kernel
                    del logprobs_via_logsoftmax_kernel
                # clamp to avoid -inf
                elif SGLANG_RETURN_ORIGINAL_LOGPROB:
                    logprobs = torch.log(probs_without_temp_scaling).clamp(
                        min=torch.finfo(probs_without_temp_scaling.dtype).min
                    )
                    del probs_without_temp_scaling
                else:
                    logprobs = torch.log(probs).clamp(min=torch.finfo(probs.dtype).min)

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                batch_next_token_ids,
            ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

        return batch_next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
    ) -> None:
        """
        Compute logprobs for requested token IDs without performing sampling.

        Optimized for prefill-only scoring requests that need token probabilities
        but don't require next token generation.
        """

        if logits_output.next_token_logits is None:
            logger.warning("No logits available for logprob computation")
            return

        # Check if any requests actually need logprobs computation
        needs_token_ids_logprobs = any(
            token_ids is not None and len(token_ids) > 0
            for token_ids in token_ids_logprobs
        )
        needs_top_logprobs = any(x > 0 for x in top_logprobs_nums)

        if not (needs_token_ids_logprobs or needs_top_logprobs):
            return

        # Preprocess logits (custom processors and NaN handling)
        logits = self._preprocess_logits(logits_output.next_token_logits, sampling_info)

        # Compute logprobs
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Handle top logprobs if requested
        if needs_top_logprobs:
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(logprobs, top_logprobs_nums)

        # Handle token_ids logprobs if requested
        if needs_token_ids_logprobs:
            (
                logits_output.next_token_token_ids_logprobs_val,
                logits_output.next_token_token_ids_logprobs_idx,
            ) = get_token_ids_logprobs_batch_optimized(logprobs, token_ids_logprobs)


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
    sampling_seed: Optional[torch.Tensor],
    positions: torch.Tensor,
):
    """
    A top-k, top-p and min-p sampling implementation with native pytorch operations.
    When sampling_seed is not None, deterministic inference will be enabled, it will sample
    with the sampling_seed of each request.
    """
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0
    if sampling_seed is not None:
        sampled_index = multinomial_with_seed(probs_sort, sampling_seed, positions)
    else:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def multinomial_with_seed(
    inputs: torch.Tensor, seed: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """
    Samples n elements from an input tensor `inputs` of shape (n, m) using
    a unique random seed for each row. This is a deterministic batched alternative to
    `torch.multinomial`.

    Args:
        inputs: A float tensor of shape (n, m) representing n categorical
                distributions with m categories each. The values are treated
                as weights and do not need to sum to 1.
        seed:   An integer tensor of shape (n,) containing the random seed
                for each corresponding row in `inputs`.
        positions: The positions of the tokens in the sequence. Used for deterministic sampling
                to get the unique seed for each position.

    Returns:
        A tensor of shape (n,) where the i-th element is an index sampled
        from the distribution in `inputs[i]` using `seed[i]`.
    """
    n, m = inputs.shape
    col_indices = torch.arange(m, device=inputs.device).unsqueeze(0)
    step_seed = (seed * 19349663) ^ (positions * 73856093)
    seed_expanded = step_seed.unsqueeze(-1)
    hashed = (seed_expanded * 8589934591) ^ (col_indices * 479001599)
    uniform_samples = (hashed % (2**24)).float() / (2**24)
    epsilon = 1e-10
    uniform_samples = uniform_samples.clamp(epsilon, 1.0 - epsilon)
    gumbel_noise = -torch.log(-torch.log(uniform_samples))
    log_probs = torch.log(inputs + epsilon)
    perturbed_log_probs = log_probs + gumbel_noise
    return torch.argmax(perturbed_log_probs, dim=1, keepdim=True)


def sampling_from_probs_torch(
    probs: torch.Tensor,
    sampling_seed: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
):
    """A sampling implementation with native pytorch operations, without
    top-k, top-p, or min-p filtering."""
    if sampling_seed is not None:
        sampled_index = multinomial_with_seed(probs, sampling_seed, positions)
    else:
        sampled_index = torch.multinomial(probs, num_samples=1)
    batch_next_token_ids = sampled_index.view(-1).to(torch.int32)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
):
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])

    return (
        output_top_logprobs_val,
        output_top_logprobs_idx,
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


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return (
        output_token_ids_logprobs_val,
        output_token_ids_logprobs_idx,
    )


def apply_custom_logit_processor(
    logits: torch.Tensor,
    sampling_batch_info: SamplingBatchInfo,
    num_tokens_in_batch: int = 1,
):
    """Apply custom logit processors to the logits.
    This function will modify the logits in-place.
    num_tokens_in_batch is needed to support spec decoding, where each batch can contain multiple
    tokens. By default, we assume each batch contains only 1 token.
    """

    assert logits.shape[0] == len(sampling_batch_info) * num_tokens_in_batch, (
        f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
        f"sampling_batch_info ({len(sampling_batch_info)}) x num_tokens_in_batch "
        f"({num_tokens_in_batch})"
    )

    for _, (
        processor,
        batch_mask,
    ) in sampling_batch_info.custom_logit_processor.items():
        # Get the batch indices that need to be processed
        batch_indices = batch_mask.nonzero(as_tuple=True)[0]

        assert batch_mask.shape[0] == len(sampling_batch_info), (
            f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )
        batch_mask = torch.repeat_interleave(batch_mask, num_tokens_in_batch)

        # Apply the processor to the logits
        logits[batch_mask] = processor(
            logits[batch_mask],
            [sampling_batch_info.custom_params[i] for i in batch_indices],
        )

        logger.debug(
            f"Custom logit processor {processor.__class__.__name__} is applied."
        )
