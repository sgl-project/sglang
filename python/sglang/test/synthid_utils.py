from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.sampling.watermarking.synthid.state import SynthIDStatePool

SYNTHID_TEST_KEYS = tuple(range(1, 31))


def synthid_state(pool: SynthIDStatePool) -> dict[str, torch.Tensor]:
    """A CPU copy of every state tensor of the pool."""
    return {
        name: value.to("cpu", copy=True)
        for name, value in vars(pool).items()
        if isinstance(value, torch.Tensor)
    }


def assert_synthid_state_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name
