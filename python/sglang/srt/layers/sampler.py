import logging

import torch
from flashinfer.sampling import top_k_top_p_sampling_from_probs
from torch import distributed as dist
from torch import nn
from vllm.distributed import get_tensor_model_parallel_group

# TODO: move this dict to another place
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_info import SamplingInfo

logger = logging.getLogger(__name__)


class Sampler(nn.Module):
    def __init__(self):
        pass

    def sample(
        self, logits: torch.Tensor, sampling_info: SamplingInfo, is_multi_node_tp: bool
    ):
        # Post process logits
        logits = logits.contiguous()
        logits.div_(sampling_info.temperatures)
        if sampling_info.logit_bias is not None:
            logits.add_(sampling_info.logit_bias)

        logits = sampling_info.penalizer_orchestrator.apply(logits)

        probs = torch.softmax(logits, dim=-1)

        if not global_server_args_dict["disable_flashinfer_sampling"]:
            max_top_k_round, batch_size = 32, probs.shape[0]
            uniform_samples = torch.rand(
                (max_top_k_round, batch_size), device=probs.device
            )
            batch_next_token_ids, success = top_k_top_p_sampling_from_probs(
                probs,
                uniform_samples,
                sampling_info.top_ks,
                sampling_info.top_ps,
            )
        else:
            # Here we provide a slower fallback implementation.
            batch_next_token_ids, success = top_k_top_p_sampling_from_probs_torch(
                probs, sampling_info.top_ks, sampling_info.top_ps
            )

        if not torch.all(success):
            logging.warning("Sampling failed, fallback to top_k=1 strategy")
            probs = probs.masked_fill(torch.isnan(probs), 0.0)
            argmax_ids = torch.argmax(probs, dim=-1)
            batch_next_token_ids = torch.where(
                success, batch_next_token_ids, argmax_ids
            )

        sampling_info.penalizer_orchestrator.cumulate_output_tokens(
            batch_next_token_ids
        )

        if is_multi_node_tp:
            # If the tensor parallelism spans across multiple nodes, there is some indeterminism
            # that can cause the TP workers to generate different tokens, so we need to
            # sync here
            dist.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=get_tensor_model_parallel_group().device_group,
            )

        return batch_next_token_ids


def top_k_top_p_sampling_from_probs_torch(
    probs: torch.Tensor, top_ks: torch.Tensor, top_ps: torch.Tensor
):
    """A top-k and top-k sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort.div_(probs_sort.max(dim=-1, keepdim=True)[0])
    try:
        sampled_index = torch.multinomial(probs_sort, num_samples=1)
    except RuntimeError:
        batch_next_token_ids = torch.zeros(
            (probs_sort.shape[0],), dtype=torch.int32, device=probs.device
        )
        success = torch.zeros(probs.shape[0], dtype=torch.bool, device=probs.device)
        return batch_next_token_ids, success

    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    success = torch.ones(probs.shape[0], dtype=torch.bool, device=probs.device)
    return batch_next_token_ids, success
