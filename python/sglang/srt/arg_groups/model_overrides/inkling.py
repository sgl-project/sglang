"""Config-time override declarations for inkling."""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    model_config_of,
    resolving_view,
)
from sglang.srt.configs.inkling2 import INKLING2_ARCHS
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def _mega_moe_fits(text_config: Any) -> bool:
    from sglang.srt.layers.moe.mega_moe import check_mega_moe_shapes

    # NVFP4 has the strictest MegaMoE alignment of the expert precisions;
    # Inkling2Experts pads the intermediate width to a multiple of 256 for it.
    try:
        check_mega_moe_shapes(
            hidden=text_config.hidden_size,
            intermediate=-(-text_config.intermediate_size // 256) * 256,
            mma_type="nvfp4xnvfp4",
        )
    except ValueError:
        return False
    return True


@_register_for(
    "InklingForConditionalGeneration",
    *INKLING2_ARCHS,
    "InklingForConditionalGenerationMTP",
)
def _inkling_overrides(server_args: Any, hf_config: Any) -> dict:
    """Inkling architecture defaults: SWA / mamba KV-pool ratios tuned for the
    hybrid-SWA layout, the extra-buffer mamba strategy, and the unified radix
    tree (which Inkling requires — models/inkling.py asserts it). The full-graph
    prefill default is set separately (inline, before cuda-graph resolution) —
    see ServerArgs.__post_init__ / _apply_inkling_prefill_cuda_graph_default. The
    server-arg defaults each yield to an explicit user value (the ratios read
    `None`, the mamba strategy its "auto" token); the prefill declaration is
    materialized
    before _parse_cuda_graph_config folds cuda_graph_backend_prefill into
    prefill.backend, and an explicit --cuda-graph-backend-prefill /
    --disable-prefill-cuda-graph still wins. The unified-radix env write follows
    the MiniMax-M3 handler precedent (env is not a resolvable server-arg)."""
    cfg = resolving_view(server_args)

    overrides: Dict[str, Any] = {}
    inkling2 = any(arch in INKLING2_ARCHS for arch in hf_config.architectures)
    if inkling2:
        if cfg.tool_call_parser is None:
            overrides["tool_call_parser"] = "inkling2"
        if cfg.reasoning_parser is None:
            overrides["reasoning_parser"] = "inkling"
        moe_ranks = cfg.tp_size // cfg.moe_dp_size
        # On SM100 / SM103 / SM107, quantized experts above the routed-kernel batch
        # run only on MegaMoE at full speed.
        if (
            cfg.moe_a2a_backend == "none"
            and moe_ranks > 1
            and get_platform().is_cuda
            and get_platform().device_sm in (100, 103, 107)
            and not cfg.encoder_only
            and not (cfg.enable_two_batch_overlap or cfg.enable_single_batch_overlap)
            and model_config_of(server_args).quantization == "inkling2"
            and _mega_moe_fits(hf_config.get_text_config())
        ):
            overrides["moe_a2a_backend"] = "megamoe"
        # Inkling2's MegaMoE path slices the replicated attention-TP tokens itself;
        # the scheduler-side attn-TP gather would only pad batches to attn_tp_size.
        megamoe = overrides.get("moe_a2a_backend", cfg.moe_a2a_backend) == "megamoe"
        if megamoe and cfg.attn_dp_size == 1:
            overrides["disable_attn_tp_gather"] = True
        # NVFP4 experts need expert parallelism. ep_size == 1 is the field
        # default; any other value is the user's and wins.
        if cfg.ep_size == 1 and moe_ranks > 1:
            overrides["ep_size"] = moe_ranks
        # SM120 hosts may lack PCIe P2P, and the default assumes it works:
        # custom all-reduce would exchange IPC handles before finding out.
        if get_platform().is_sm120:
            overrides["enable_p2p_check"] = True
            # Page size 1 left 0.5 GB after the KV pool; requests ran out of memory.
            if cfg.page_size is None:
                overrides["page_size"] = 128
            # MXFP8 verify runs only on the bias decode kernel, which gives every
            # request the same draft count, not compact ragged verify's own.
            if cfg.kv_cache_dtype == "mxfp8":
                from sglang.srt.speculative.ragged_verify import (
                    RaggedVerifyMode,
                    read_ragged_verify_mode,
                )
                from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

                algorithm = SpeculativeAlgorithm.from_string(cfg.speculative_algorithm)
                if (
                    algorithm.supports_ragged_verify()
                    and read_ragged_verify_mode() is RaggedVerifyMode.COMPACT
                ):
                    raise ValueError(
                        "Inkling2 with --kv-cache-dtype mxfp8 on SM120 verifies "
                        "every request's whole draft block; use "
                        "SGLANG_RAGGED_VERIFY_MODE=static or cap-accept."
                    )
    # NOTE: the full-graph prefill default is NOT set here. cuda-graph config is
    # resolved in __post_init__ before declarations are materialized, so a
    # cuda_graph_backend_prefill declared here lands too late (the breakable
    # default would already have been auto-disabled for this multimodal arch).
    # It is set inline before _handle_cuda_graph_config instead.
    if cfg.swa_full_tokens_ratio is None:
        overrides["swa_full_tokens_ratio"] = 0.1
    if cfg.mamba_full_memory_ratio is None:
        overrides["mamba_full_memory_ratio"] = 0.1
    # Inkling requires the extra-buffer mamba strategy (inkling.py asserts
    # enable_mamba_extra_buffer()); the generic "auto" resolution does not cover
    # Inkling, so pin it here. Yields to an explicit --mamba-radix-cache-strategy.
    #
    # Compared against the unresolved token rather than the class default: the
    # default only answers "unset" while nothing has declared the field first,
    # and `_mamba_radix_cache_resolution` would -- from the slot just above
    # `collect_model_override_declarations` -- for an architecture whose
    # linear-attention spec sets `uses_mamba_radix_cache`. Inkling has no such
    # spec today, but giving it one would silently stop this pin from firing.
    if cfg.mamba_radix_cache_strategy == "auto":
        overrides["mamba_radix_cache_strategy"] = "extra_buffer"
    # The generic resolution never sets the arch-derived leaf for Inkling, and
    # handle_mamba_radix_cache rejects extra_buffer on a model without it.
    if not cfg.disable_radix_cache:
        overrides["uses_mamba_radix_cache"] = True
    # Inkling attention runs only on the fa4 (Hopper/Blackwell) or triton backends --
    # models/inkling_common/attn.py asserts attention_backend in {fa4, triton}.
    # The generic resolver would otherwise pick trtllm_mha (SM100) / fa3
    # (Hopper), so a bare launch fails on the first attention forward. Pin a
    # supported default when the user left every attention-backend flag unset
    # (mirrors the MiniMax-M3 SM100 fa4-default above); an explicit
    # --attention-backend / --prefill/decode-attention-backend still wins.
    if is_attention_backend_not_set(cfg):
        # Inkling2 attention asserts fa4, which SM120 runs on its FA4 forward host.
        if (
            get_platform().is_sm90
            or get_platform().is_sm100
            or (inkling2 and get_platform().is_sm120)
        ):
            inkling_attn_backend = "fa4"
        else:
            inkling_attn_backend = "triton"
        overrides["attention_backend"] = inkling_attn_backend
        logger.info(
            f"Use {inkling_attn_backend} as the attention backend for Inkling "
            "(requires fa4 or triton)."
        )
    return overrides
