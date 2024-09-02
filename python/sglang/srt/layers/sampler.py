import dataclasses
import logging
from typing import Tuple, Union

import torch
from flashinfer.sampling import (
    min_p_sampling_from_probs,
    top_k_renorm_prob,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_prob,
)
from torch.library import custom_op as torch_custom_op
from vllm.model_executor.custom_op import CustomOp

from sglang.srt.layers.logits_processor import LogitsProcessorOutput

# TODO: move this dict to another place
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SampleOutput:
    success: torch.Tensor
    probs: torch.Tensor
    batch_next_token_ids: torch.Tensor


class Sampler(CustomOp):
    def __init__(self):
        super().__init__()
        # FIXME: torch.multinomial has too many bugs
        self.forward_native = self.forward_cuda
        self.is_torch_compile = False

    def _apply_penalties(self, logits: torch.Tensor, sampling_info: SamplingBatchInfo):
        # min-token, presence, frequency
        if sampling_info.linear_penalties is not None:
            logits += sampling_info.linear_penalties

        # repetition
        if sampling_info.scaling_penalties is not None:
            logits = torch.where(
                logits > 0,
                logits / sampling_info.scaling_penalties,
                logits * sampling_info.scaling_penalties,
            )

        return logits

    def _get_probs(self, logits: torch.Tensor, sampling_info: SamplingBatchInfo):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(sampling_info.temperatures)
        if self.is_torch_compile:
            # FIXME: Temporary workaround for unknown bugs in torch.compile
            logits.add_(0)

        if sampling_info.logit_bias is not None:
            logits.add_(sampling_info.logit_bias)

        if sampling_info.vocab_mask is not None:
            logits = logits.masked_fill(sampling_info.vocab_mask, float("-inf"))

        logits = self._apply_penalties(logits, sampling_info)

        return torch.softmax(logits, dim=-1)

    def forward_cuda(
        self,
        logits: Union[torch.Tensor, LogitsProcessorOutput],
        sampling_info: SamplingBatchInfo,
    ):
        if isinstance(logits, LogitsProcessorOutput):
            logits = logits.next_token_logits

        probs = self._get_probs(logits, sampling_info)

        if not global_server_args_dict["disable_flashinfer_sampling"]:
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
                batch_next_token_ids, success = flashinfer_top_k_top_p(
                    probs, uniform_samples, sampling_info.top_ks, sampling_info.top_ps
                )
        else:
            # Here we provide a slower fallback implementation.
            batch_next_token_ids, success = top_k_top_p_min_p_sampling_from_probs_torch(
                probs, sampling_info.top_ks, sampling_info.top_ps, sampling_info.min_ps
            )

        return SampleOutput(success, probs, batch_next_token_ids)

    def forward_native(
        self,
        logits: Union[torch.Tensor, LogitsProcessorOutput],
        sampling_info: SamplingBatchInfo,
    ):
        if isinstance(logits, LogitsProcessorOutput):
            logits = logits.next_token_logits

        probs = self._get_probs(logits, sampling_info)

        batch_next_token_ids, success = top_k_top_p_min_p_sampling_from_probs_torch(
            probs, sampling_info.top_ks, sampling_info.top_ps, sampling_info.min_ps
        )

        return SampleOutput(success, probs, batch_next_token_ids)


@torch_custom_op("my_lib::flashinfer_top_k_top_p", mutates_args={})
def flashinfer_top_k_top_p(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # NOTE: we do not use min_p neither in CUDA nor in torch.compile
    return top_k_top_p_sampling_from_probs(probs, uniform_samples, top_ks, top_ps)


@flashinfer_top_k_top_p.register_fake
def _(
    probs: torch.Tensor,
    uniform_samples: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bs = probs.shape[0]
    return (
        torch.ones(bs, dtype=torch.bool, device=probs.device),
        torch.zeros(bs, dtype=torch.int32, device=probs.device),
    )


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
        # FIXME: torch.multiomial does not support num_samples = 1
        sampled_index = torch.multinomial(probs_sort, num_samples=2, replacement=True)[
            :, :1
        ]
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
