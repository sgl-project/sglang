"""Strategy dispatch for prefill context parallel CUDA graphs."""

from sglang.srt.layers.cp.base import (
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)

PREFILL_CP_BCG_SUPPORTED_COMBINATIONS: set[
    tuple[ContextParallelStrategyKind, CPAttentionBackendKind]
] = {
    (ContextParallelStrategyKind.ZIGZAG, CPAttentionBackendKind.TRTLLM_MHA),
    (ContextParallelStrategyKind.INTERLEAVE, CPAttentionBackendKind.DSA),
}


def get_cp_bcg_input_type(strategy: ContextParallelStrategyKind):
    # Keep runtime adapters lazy: config resolution also queries CP BCG support.
    if strategy == ContextParallelStrategyKind.INTERLEAVE:
        from sglang.srt.layers.cp.bcg.interleave_bcg import InterleaveCPBCGInput

        return InterleaveCPBCGInput
    if strategy == ContextParallelStrategyKind.ZIGZAG:
        from sglang.srt.layers.cp.bcg.zigzag_bcg import ZigzagCPBCGInput

        return ZigzagCPBCGInput
    raise ValueError(f"Unsupported CP BCG strategy: {strategy}")
