"""Strategy dispatch for prefill context parallel CUDA graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.layers.cp.base import (
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


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


def supports_prefill_cp_bcg(server_args: ServerArgs) -> bool:
    """Return whether the selected prefill-CP configuration supports BCG."""

    cfg = resolving_view(server_args)
    resolved = resolved_view(server_args)
    if not (
        cfg.enable_prefill_cp
        and cfg.pp_size == 1
        and resolved.attn_cp_size == cfg.tp_size
    ):
        return False
    prefill_attention_backend, _ = attention_backends_of(resolved)
    try:
        combination = (
            ContextParallelStrategyKind.from_string(cfg.cp_strategy),
            CPAttentionBackendKind.from_string(prefill_attention_backend),
        )
    except ValueError:
        # Other backends (or an unresolved backend/strategy) cannot use CP BCG.
        return False
    if combination not in PREFILL_CP_BCG_SUPPORTED_COMBINATIONS:
        return False
    return get_cp_bcg_input_type(combination[0]).supports(server_args)
