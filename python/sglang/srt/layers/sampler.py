import logging
from typing import Union

import torch
from torch import nn

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, is_flashinfer_available

if is_flashinfer_available():
    from flashinfer.sampling import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detectioin = global_server_args_dict["enable_nan_detection"]

    def forward(
        self,
        logits: Union[torch.Tensor, LogitsProcessorOutput],
        sampling_info: SamplingBatchInfo,
    ):
        if isinstance(logits, LogitsProcessorOutput):
            logits = logits.next_token_logits

        logits = logits.contiguous()

        if self.use_nan_detectioin and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
        else:
            # Post process logits
            logits.div_(sampling_info.temperatures)
            probs = torch.softmax(logits, dim=-1)
            logits = None
            del logits

            if global_server_args_dict["sampling_backend"] == "flashinfer":
                max_top_k_round, batch_size = 32, probs.shape[0]
                uniform_samples = torch.rand(
                    (max_top_k_round, batch_size), device=probs.device
                )
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids, success = min_p_sampling_from_probs(
                        probs, uniform_samples, sampling_info.min_ps
                    )
                else:
                    batch_next_token_ids, success = top_k_top_p_sampling_from_probs(
                        probs,
                        uniform_samples,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                    )

                if self.use_nan_detectioin and not torch.all(success):
                    logger.warning("Detected errors during sampling!")
                    batch_next_token_ids = torch.zeros_like(batch_next_token_ids)
            elif global_server_args_dict["sampling_backend"] == "pytorch":
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                )
            else:
                raise ValueError(
                    f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                )

        return batch_next_token_ids.to(torch.int32)


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
    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids
