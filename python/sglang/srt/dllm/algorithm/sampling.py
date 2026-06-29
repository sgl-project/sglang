from typing import Optional, Tuple

import torch

from sglang.srt.layers.sampler import (
    sampling_from_probs_torch,
    top_k_top_p_min_p_sampling_from_probs_torch,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL

_SAMPLING_EPS = 1e-6


def sample_block_tokens(
    logits: torch.Tensor,
    sampling_info: Optional[SamplingBatchInfo],
    batch_id: int,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one proposal token per DLLM block position.

    Returns sampled token ids and their probabilities under the temperature-scaled
    distribution before top-k/top-p/min-p filtering. When the request is greedy,
    this preserves the previous argmax behavior.
    """
    if sampling_info is None:
        return _greedy_sample(logits)

    num_positions = logits.shape[0]
    temperature = sampling_info.temperatures[batch_id].to(
        device=logits.device, dtype=logits.dtype
    )
    top_k = sampling_info.top_ks[batch_id].to(device=logits.device)
    if int(top_k.item()) <= 1 or float(temperature.item()) < _SAMPLING_EPS:
        return _greedy_sample(logits)

    probs = torch.softmax(logits / temperature, dim=-1)

    top_p = sampling_info.top_ps[batch_id].to(device=logits.device)
    min_p = sampling_info.min_ps[batch_id].to(device=logits.device)
    need_top_k_sampling = int(top_k.item()) != TOP_K_ALL
    need_top_p_sampling = float(top_p.item()) != 1.0
    need_min_p_sampling = float(min_p.item()) > 0.0

    sampling_seed = None
    if sampling_info.sampling_seed is not None:
        sampling_seed = (
            sampling_info.sampling_seed[batch_id]
            .to(device=logits.device)
            .view(1)
            .expand(num_positions)
        )
        if positions is None:
            positions = torch.arange(
                num_positions, device=logits.device, dtype=torch.int64
            )
        else:
            positions = positions.to(device=logits.device)

    if need_top_k_sampling or need_top_p_sampling or need_min_p_sampling:
        token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
            probs,
            top_k.view(1).expand(num_positions),
            top_p.view(1).expand(num_positions),
            min_p.view(1).expand(num_positions),
            need_min_p_sampling,
            sampling_seed,
            positions,
        )
    else:
        token_ids = sampling_from_probs_torch(
            probs, sampling_seed=sampling_seed, positions=positions
        )

    token_ids = token_ids.to(dtype=torch.long)
    token_probs = probs.gather(1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_ids, token_probs


def _greedy_sample(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=-1)
    token_ids = torch.argmax(probs, dim=-1)
    token_probs = probs.gather(1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_ids.to(dtype=torch.long), token_probs
