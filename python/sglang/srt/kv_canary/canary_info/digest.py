from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.ops.kv_canary.consts import sampling_digest_from_values

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_params import SamplingParams


def sampling_digest(params: SamplingParams) -> int:
    return sampling_digest_from_values(
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        min_p=params.min_p,
    )
