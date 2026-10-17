"""Output-ring encoding for pipeline-parallel speculative proposals."""

from __future__ import annotations

from typing import Any, List, Mapping, MutableMapping, Optional

import torch

from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput

_NEXT_STEPS_KEY = "spec_next_num_steps"
_NEXT_WIDTH_KEY = "spec_next_num_draft_tokens"


def read_next_config(
    tensors: Mapping[str, Any],
) -> tuple[Optional[int], Optional[int]]:
    """Return ``(steps, width)`` for the next-round proposal, if present."""
    return tensors.get(_NEXT_STEPS_KEY), tensors.get(_NEXT_WIDTH_KEY)


def write_next_config(
    tensors: MutableMapping[str, Any], *, steps: int, width: int
) -> None:
    """Publish both next-round configuration fields together."""
    tensors[_NEXT_STEPS_KEY] = steps
    tensors[_NEXT_WIDTH_KEY] = width


def encode_chain(
    chain: torch.Tensor,
    *,
    batch_size: int,
    logical_width: int,
    capacity: int,
) -> torch.Tensor:
    """Convert a logical proposal to the fixed-width PP wire shape."""
    if capacity < logical_width:
        raise ValueError(
            "PP speculative capacity is smaller than the proposal: "
            f"capacity={capacity}, logical={logical_width}"
        )
    chain = chain.to(torch.int64).reshape(batch_size, logical_width)
    if capacity > logical_width:
        chain = torch.nn.functional.pad(chain, (0, capacity - logical_width), value=0)
    return chain


def decode_relay(
    *,
    rids: List[str],
    tensors: Mapping[str, Any],
    steps: int,
    logical_width: int,
    fixed_capacity: bool,
) -> PPSpecRelayInput:
    """Decode an output-ring proposal into logical per-request state."""
    chain = tensors.get("spec_next_chain")
    if chain is None:
        return PPSpecRelayInput.degenerate(
            rids=rids,
            bonus_tokens=tensors["spec_bonus_tokens"],
            num_draft_tokens=logical_width,
            speculative_num_steps=steps,
        )

    if fixed_capacity:
        if chain.ndim != 2 or chain.shape[0] != len(rids):
            raise ValueError(
                "Invalid fixed-capacity PP speculative chain shape: "
                f"shape={tuple(chain.shape)}, batch={len(rids)}"
            )
        if chain.shape[1] < logical_width:
            raise ValueError(
                "PP speculative chain is narrower than its logical width: "
                f"wire={chain.shape[1]}, logical={logical_width}"
            )
        # Materialize the logical slice so request state does not retain the
        # padded wire storage or its wider row stride.
        chain = chain[:, :logical_width].contiguous()
    else:
        chain = chain.reshape(len(rids), logical_width)

    return PPSpecRelayInput(
        rids=rids,
        tokens=chain.to(torch.int64),
        parents=tensors.get("spec_next_parents"),
        top_scores=tensors.get("spec_next_top_scores"),
        speculative_num_steps=steps,
    )
