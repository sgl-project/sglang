from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.sampling.watermarking.synthid.watermark import SynthIDWatermark

SYNTHID_TEST_KEYS = tuple(range(1, 31))

STATE_FIELDS = ("context", "context_history", "history_cursor", "num_calls")


def synthid_state(watermark: SynthIDWatermark) -> dict[str, torch.Tensor]:
    return {
        name: getattr(watermark.state, name).to("cpu", copy=True)
        for name in STATE_FIELDS
    }


def assert_synthid_state_equal(
    actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> None:
    for name in STATE_FIELDS:
        assert torch.equal(actual[name], expected[name]), name
