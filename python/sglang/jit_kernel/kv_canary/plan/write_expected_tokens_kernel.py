from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_plan_write_expected_tokens_module() -> "Module":
    return load_jit(
        "kv_canary_plan_write_expected_tokens",
        cuda_files=["kv_canary/canary_plan_write_expected_tokens.cuh"],
        cuda_wrappers=[
            (
                "plan_write_expected_tokens",
                "PlanWriteExpectedTokensKernel::run",
            ),
        ],
    )


def launch_plan_write_expected_tokens_kernel(
    *,
    req_pool_indices: torch.Tensor,
    prefix_lens: torch.Tensor,
    write_offsets: torch.Tensor,
    expected_token_pool: Optional[torch.Tensor],
    expected_token_valid_lens: Optional[torch.Tensor],
    out_expected_input_tokens: torch.Tensor,
    slot_token_offset: int,
) -> None:
    """Fill ``out_expected_input_tokens`` per write entry from
    ``expected_token_pool``. For each write entry ``i`` owned by req ``r``:

    - ``write_pos = prefix_lens[r] + (i - write_offsets[r])``
    - ``sot_pos = write_pos + slot_token_offset``
    - if ``0 <= sot_pos < expected_token_valid_lens[req_pool_indices[r]]``:
      ``out_expected_input_tokens[i] = expected_token_pool[req_pool_indices[r], sot_pos]``
    - else: ``out_expected_input_tokens[i] = -1`` (sentinel: WRITE kernel skips this entry).

    When ``expected_token_pool`` or ``expected_token_valid_lens`` is ``None``, every entry
    receives the ``-1`` sentinel.
    """
    if (expected_token_pool is None) != (expected_token_valid_lens is None):
        raise ValueError(
            "kv-canary: expected_token_pool and expected_token_valid_lens must both be "
            "provided or both be None"
        )
    module = _jit_plan_write_expected_tokens_module()
    module.plan_write_expected_tokens(
        req_pool_indices,
        prefix_lens,
        write_offsets,
        expected_token_pool,
        expected_token_valid_lens,
        out_expected_input_tokens,
        int(slot_token_offset),
    )
