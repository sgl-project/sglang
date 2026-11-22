from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_penalty_counts_module() -> Module:
    return load_jit(
        "sampling_penalty_counts",
        cuda_files=["sampling/penalty_counts.cuh"],
        cuda_wrappers=[("accumulate_penalty_counts", "accumulate_penalty_counts")],
    )


def accumulate_penalty_counts(
    counts: torch.Tensor,
    lens: torch.Tensor,
    batch_indices: torch.Tensor,
    penalty_indices: torch.Tensor,
    token_ids: torch.Tensor,
    accept_mask: Optional[torch.Tensor],
) -> None:
    """Count the tokens of penalized request ``i``, row ``batch_indices[i]`` of
    ``token_ids`` (``[bs, k]``), into row ``penalty_indices[i]`` of ``counts``
    and ``lens`` in place. A token counts when it is in the vocab and, if
    ``accept_mask`` is given, marked there.
    """
    _jit_penalty_counts_module().accumulate_penalty_counts(
        counts, lens, batch_indices, penalty_indices, token_ids, accept_mask
    )
