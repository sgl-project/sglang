import logging

import torch
from flashinfer.sampling import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
)
from vllm.model_executor.custom_op import CustomOp

# TODO: move this dict to another place
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)


class Sampler(CustomOp):
    def __init__(self):
        super().__init__()

    def forward_cuda(self, logits: torch.Tensor, sampling_info: SamplingBatchInfo):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(sampling_info.temperatures)
        if sampling_info.logit_bias is not None:
            logits.add_(sampling_info.logit_bias)

        if sampling_info.vocab_mask is not None:
            logits = logits.masked_fill(~sampling_info.vocab_mask, float("-inf"))

        logits = sampling_info.penalizer_orchestrator.apply(logits)

        probs = torch.softmax(logits, dim=-1)

        if not global_server_args_dict["disable_flashinfer_sampling"]:
            max_top_k_round, batch_size = 32, probs.shape[0]
            uniform_samples = torch.rand(
                (max_top_k_round, batch_size), device=probs.device
            )
            if sampling_info.min_ps.any():
                probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                batch_next_token_ids, success = min_p_sampling_from_probs(
                    probs, uniform_samples, sampling_info.min_ps
                )
            else:
                batch_next_token_ids, success = top_k_top_p_sampling_from_probs(
                    probs, uniform_samples, sampling_info.top_ks, sampling_info.top_ps
                )
        else:
            # Here we provide a slower fallback implementation.
            batch_next_token_ids, success = top_k_top_p_min_p_sampling_from_probs_torch(
                probs, sampling_info.top_ks, sampling_info.top_ps, sampling_info.min_ps
            )

        if not torch.all(success):
            logging.warning("Sampling failed, fallback to top_k=1 strategy")
            probs = probs.masked_fill(torch.isnan(probs), 0.0)
            argmax_ids = torch.argmax(probs, dim=-1)
            batch_next_token_ids = torch.where(
                success, batch_next_token_ids, argmax_ids
            )

        return batch_next_token_ids

    def forward_native():
        raise NotImplementedError("Native forward is not implemented yet.")


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    min_p_thresholds = probs_sort[:, 0] * min_ps
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    try:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    except RuntimeError as e:
        logger.warning(f"Sampling error: {e}")
        batch_next_token_ids = torch.zeros(
            (probs_sort.shape[0],), dtype=torch.int32, device=probs.device
        )
        success = torch.zeros(probs.shape[0], dtype=torch.bool, device=probs.device)
        return batch_next_token_ids, success

    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    success = torch.ones(probs.shape[0], dtype=torch.bool, device=probs.device)
    return batch_next_token_ids, success
