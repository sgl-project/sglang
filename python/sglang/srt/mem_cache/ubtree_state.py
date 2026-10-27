"""Shared eligibility and scratch sizing for UBTree's compact KDA state."""

from math import prod

from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.platforms import current_platform


def use_ubtree_factorized_kda(
    *,
    enabled: bool,
    cache_params: BaseLinearStateParams,
    spec_state_size: int,
    speculative_num_draft_tokens: int | None,
    speculative_eagle_topk: int | None,
) -> bool:
    return bool(
        enabled
        and current_platform.is_cuda()
        and spec_state_size == 1
        and cache_params.is_kda
        and speculative_eagle_topk is not None
        and speculative_eagle_topk > 1
        and speculative_num_draft_tokens in (32, 49)
    )


def ubtree_spec_state_bytes_per_req(
    cache_params: BaseLinearStateParams, num_draft_tokens: int
) -> int:
    """Raw v/k/g/beta plus dense conv windows, across all KDA layers.

    The caller supplies the active request count plus the pool's padding slot;
    persistent temporal/conv states are budgeted separately.
    """
    hv, v_dim, k_dim = cache_params.shape.temporal
    h_k = cache_params.shape.num_k_heads_per_tp
    activation_bytes = cache_params.dtype.conv.itemsize
    per_node = (
        (hv * v_dim + h_k * k_dim) * activation_bytes  # raw v/k
        + hv * (k_dim + 1) * 4  # fp32 g/beta
        + sum(prod(shape) for shape in cache_params.shape.conv) * activation_bytes
    )
    return per_node * num_draft_tokens * len(cache_params.layers)
