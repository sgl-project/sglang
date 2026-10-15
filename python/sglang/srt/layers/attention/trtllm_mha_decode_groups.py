"""Length-sorted request groups for TRT-LLM decode attention, built once per forward.

With ``SGLANG_TRTLLM_MHA_DECODE_SEQ_LEN_SPLITS`` above one, every decode attention call
sorts the requests by KV length and gathers the query, block table and seq lens of each
group before its FMHA launch. Only the query depends on the layer: the order and the
gathered block tables and seq lens are the same for every call of a forward.
``DecodeGroupPlans`` builds them at the first call and later calls reuse them, so a call
keeps one query gather, the FMHA launches on contiguous slices, and one output scatter.
The groups and their order are the stock ones, so the output is bitwise unchanged.
"""

from __future__ import annotations

import logging
from typing import Callable, NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)


class DecodeGroupPlan(NamedTuple):
    # [bs] request indices sorted by KV length (torch.argsort, as the stock path).
    order: torch.Tensor
    # [bs, max_pages] and [bs]: the block tables and seq lens in `order`.
    block_tables: torch.Tensor
    seq_lens: torch.Tensor


class DecodeGroupPlans:
    """Plans of the current forward, keyed by the metadata tensors a call passes in."""

    def __init__(self):
        self._plans: dict[tuple, DecodeGroupPlan] = {}
        self._logged = False

    def reset(self) -> None:
        # Call at every forward-metadata init: a plan is only valid within one forward,
        # and buffers of the next forward may reuse the same addresses.
        self._plans = {}

    def get(
        self, *, block_tables: torch.Tensor, seq_lens: torch.Tensor
    ) -> DecodeGroupPlan:
        key = (
            block_tables.data_ptr(),
            tuple(block_tables.shape),
            seq_lens.data_ptr(),
            seq_lens.shape[0],
        )
        plan = self._plans.get(key)
        if plan is None:
            if not self._logged:
                self._logged = True
                logger.info(
                    "Decode length groups built once per forward (%d requests)",
                    seq_lens.shape[0],
                )
            order = torch.argsort(seq_lens)
            plan = DecodeGroupPlan(
                order=order,
                block_tables=block_tables.index_select(0, order),
                seq_lens=seq_lens.index_select(0, order),
            )
            self._plans[key] = plan
        return plan


def group_sizes(num_requests: int, num_splits: int) -> list[int]:
    """Section sizes of torch.tensor_split(x, num_splits) for len(x) == num_requests."""
    base, extra = divmod(num_requests, num_splits)
    return [base + 1 if i < extra else base for i in range(num_splits)]


def run_planned_groups(
    *,
    query: torch.Tensor,
    plan: DecodeGroupPlan,
    num_splits: int,
    q_len_per_req: int,
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    run_group: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Run the length groups of `plan`; same groups and order as the per-call stock path.

    ``run_group(query, block_tables, seq_lens, out)`` launches one FMHA and writes into
    ``out``. Returns [num_tokens, num_heads, head_dim].
    """
    num_requests = plan.order.shape[0]
    num_heads, head_dim = query.shape[-2], query.shape[-1]
    query_by_request = query.view(num_requests, q_len_per_req, num_heads, head_dim)
    sorted_query = query_by_request.index_select(0, plan.order)
    sorted_output = torch.empty(
        query_by_request.shape, dtype=out_dtype, device=query.device
    )
    start = 0
    for size in group_sizes(num_requests, num_splits):
        end = start + size
        run_group(
            sorted_query[start:end].reshape(-1, num_heads, head_dim),
            plan.block_tables[start:end],
            plan.seq_lens[start:end],
            sorted_output[start:end].view(-1, num_heads, head_dim),
        )
        start = end
    output_by_request = (
        out.view_as(query_by_request)
        if out is not None
        else torch.empty_like(sorted_output)
    )
    output_by_request.index_copy_(0, plan.order, sorted_output)
    return output_by_request.view(-1, num_heads, head_dim)
