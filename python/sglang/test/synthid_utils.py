"""Shared helpers for the SynthID watermark tests and benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from sglang.srt.sampling.watermarking.synthid.watermark import SynthIDWatermark

# Public reference keys from google-deepmind/synthid-text. These are suitable
# for compatibility tests and benchmarks, not for a production watermark.
REFERENCE_SYNTHID_KEYS = (
    654,
    400,
    836,
    123,
    340,
    443,
    597,
    160,
    57,
    29,
    590,
    639,
    13,
    715,
    468,
    990,
    966,
    226,
    324,
    585,
    118,
    504,
    421,
    521,
    129,
    669,
    732,
    225,
    90,
    960,
)

STATE_FIELDS = ("context", "context_history", "history_cursor", "num_calls")


def synthid_state(watermark: SynthIDWatermark) -> dict[str, torch.Tensor]:
    """A CPU copy of every slot's state."""
    return {
        name: getattr(watermark.state, name).to("cpu", copy=True)
        for name in STATE_FIELDS
    }
