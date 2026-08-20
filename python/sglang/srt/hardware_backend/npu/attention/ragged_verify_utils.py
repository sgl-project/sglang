from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout


def get_npu_bucketed_ragged_verify_layout(
    *,
    spec_info,
    layout: RaggedVerifyLayout,
    padded_bs: int,
    cap: int,
) -> RaggedVerifyLayout:
    """Memoize the NPU derivative of a live compact verify layout.

    NPU attention and DSV4 compressor metadata consume the same bucketed
    geometry several times during one replay.  Keep that derivative private
    to the speculative input so the public ragged-layout and graph-runner
    interfaces remain unchanged.
    """
    padded_bs = int(padded_bs)
    cap = int(cap)
    cached = getattr(spec_info, "_npu_ragged_verify_bucket_cache", None)
    if (
        cached is not None
        and cached[0] is layout
        and cached[1] == padded_bs
        and cached[2] == cap
    ):
        return cached[3]

    padded = layout.padded_to_bucket(padded_bs=padded_bs, cap=cap)
    # Retaining layout prevents a later source from colliding through a
    # recycled Python object id. A new source or geometry replaces the cache.
    setattr(
        spec_info,
        "_npu_ragged_verify_bucket_cache",
        (layout, padded_bs, cap, padded),
    )
    return padded
