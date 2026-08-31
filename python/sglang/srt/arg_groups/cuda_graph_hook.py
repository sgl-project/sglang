# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the CUDA-graph capture configuration."""

from __future__ import annotations

import logging
from typing import Any

from sglang.srt.arg_groups.overrides import (
    attention_backends_of,
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.connector import ConnectorType
from sglang.srt.model_executor.cuda_graph_config import (
    ALLOWED_BACKENDS_PER_PHASE,
    Backend,
    CudaGraphConfig,
    Phase,
    default_cuda_graph_config,
    with_phase,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import (
    is_cpu,
    is_mps,
    parse_connector_type,
)
from sglang.srt.utils.hf_transformers_utils import check_gguf_file

logger = logging.getLogger(__name__)


def parse_cuda_graph_config(server_args: Any):
    """Resolve cuda_graph_config from explicit JSON, per-phase
    convenience flags, legacy global flags, and defaults.
    Precedence (highest first): explicit JSON > convenience > legacy > defaults.
    Also populates server_args._cuda_graph_config_locked — the set of
    (phase, key) tuples that came from non-default sources; the
    auto-disable cascade respects this lock (the old
    --enforce-piecewise-cuda-graph semantics generalized).
    """
    cfg = resolving_view(server_args)
    raw_input = cfg.cuda_graph_config
    if isinstance(raw_input, CudaGraphConfig):
        explicit_input = raw_input.to_dict()
    else:
        explicit_input = raw_input or {}
    config = default_cuda_graph_config()
    locked: set = set()

    def _set(phase: str, key: str, value: Any) -> None:
        setattr(getattr(config, phase), key, value)
        locked.add((phase, key))

    # ---- Legacy global flags (lowest precedence above defaults) ----
    if cfg.disable_cuda_graph:
        _set(Phase.DECODE, "backend", Backend.DISABLED)
        _set(Phase.PREFILL, "backend", Backend.DISABLED)

    # ---- Boolean per-phase off-switches ----
    # Below the explicit backend selectors so --cuda-graph-backend-*
    # wins if both are given.
    if cfg.disable_prefill_cuda_graph:
        _set(Phase.PREFILL, "backend", Backend.DISABLED)
    if cfg.disable_decode_cuda_graph:
        _set(Phase.DECODE, "backend", Backend.DISABLED)

    # ---- Per-phase convenience flags ----
    if cfg.cuda_graph_backend_decode is not None:
        _set(Phase.DECODE, "backend", cfg.cuda_graph_backend_decode)
    if cfg.cuda_graph_backend_prefill is not None:
        _set(Phase.PREFILL, "backend", cfg.cuda_graph_backend_prefill)
    if cfg.cuda_graph_max_bs_decode is not None:
        _set(Phase.DECODE, "max_bs", cfg.cuda_graph_max_bs_decode)
    if cfg.cuda_graph_max_bs_prefill is not None:
        _set(Phase.PREFILL, "max_bs", cfg.cuda_graph_max_bs_prefill)
    if cfg.cuda_graph_bs_decode is not None:
        _set(Phase.DECODE, "bs", cfg.cuda_graph_bs_decode)
    if cfg.cuda_graph_bs_prefill is not None:
        _set(Phase.PREFILL, "bs", cfg.cuda_graph_bs_prefill)
    if cfg.cuda_graph_tc_compiler is not None:
        # Written to both phases so the value is in place when TC_PIECEWISE
        # decode is implemented; today decode ignores it.
        _set(Phase.DECODE, "tc_compiler", cfg.cuda_graph_tc_compiler)
        _set(Phase.PREFILL, "tc_compiler", cfg.cuda_graph_tc_compiler)

    # ---- Explicit JSON config (highest precedence) ----
    for phase, phase_config in explicit_input.items():
        if not isinstance(phase_config, dict):
            continue
        for key, value in phase_config.items():
            _set(phase, key, value)

    declare_resolution(
        server_args,
        "_parse_cuda_graph_config",
        cuda_graph_config=config,
    )
    server_args._cuda_graph_config_locked = locked


def apply_cuda_graph_compatibility(server_args: Any):
    """Auto-disable prefill cuda graph for incompatible configs.
    Rules are split per backend — TcPiecewise and Breakable have
    different constraints. Skipped when the user explicitly set the
    prefill backend (this folds in the old
    --enforce-piecewise-cuda-graph contract).
    """

    cfg = resolving_view(server_args)
    if (Phase.PREFILL, "backend") in server_args._cuda_graph_config_locked:
        return

    # PP prefill graph replay is opt-in. It is most useful for small
    # aggregate forwards, while enabling it implicitly would also capture
    # large buckets that can be slower than eager. An explicit backend
    # selection bypasses this default policy.
    if cfg.pp_size > 1 and cfg.cuda_graph_config.prefill.backend == Backend.BREAKABLE:
        logger.info(
            "Disabling breakable prefill CUDA graph by default for pipeline "
            "parallelism. Set --cuda-graph-backend-prefill=breakable to opt in."
        )
        declare_resolution(
            server_args,
            "_apply_cuda_graph_compatibility",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )
        return

    # Breakable is the CUDA default but not multimodal-compatible;
    # piecewise-allowlisted archs run their validated decoder prefill
    # there instead. Archs also on the breakable allowlist keep it --
    # this runs first, so piecewise would otherwise silently win.
    if (
        cfg.cuda_graph_config.prefill.backend == Backend.BREAKABLE
        and model_config_of(server_args).is_multimodal_piecewise_cuda_graph_supported
        and not model_config_of(
            server_args
        ).is_multimodal_breakable_cuda_graph_supported
        # Keep trtllm_mla on the preferred breakable path, which now serves
        # MLA by falling back to the flashinfer MLA impl for extend.
        and attention_backends_of(resolved_view(server_args))[0] != "trtllm_mla"
    ):
        logger.info(
            "Using tc_piecewise CUDA graph for validated multimodal " "decoder prefill."
        )
        declare_resolution(
            server_args,
            "_apply_cuda_graph_compatibility",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.TC_PIECEWISE
            ),
        )

    if cfg.cuda_graph_config.prefill.backend == Backend.TC_PIECEWISE:
        disable_tc_piecewise_cudagraph_if_incompatible(server_args)
    elif cfg.cuda_graph_config.prefill.backend == Backend.BREAKABLE:
        disable_breakable_cudagraph_if_incompatible(server_args)
    elif cfg.cuda_graph_config.prefill.backend == Backend.FULL:
        disable_full_prefill_cudagraph_if_incompatible(server_args)


def disable_tc_piecewise_cudagraph_if_incompatible(server_args: Any):
    """TcPiecewise (torch.compile + piecewise) is incompatible with
    these configurations. Most are torch.compile / dynamo limitations.
    """

    cfg = resolving_view(server_args)

    rules = [
        (
            "model-arch blacklist",
            lambda: model_config_of(server_args).is_piecewise_cuda_graph_disabled_model,
        ),
        ("DP attention", lambda: resolved_view(server_args).enable_dp_attention),
        ("full torch.compile mode", lambda: cfg.enable_torch_compile),
        ("pipeline parallelism (pp_size > 1)", lambda: cfg.pp_size > 1),
        (
            "non-CUDA hardware (HIP/NPU/CPU/MPS/XPU)",
            lambda: get_platform().is_hip
            or get_platform().is_npu
            or is_cpu()
            or is_mps()
            or get_platform().is_xpu,
        ),
        (
            "OOT platform without piecewise support",
            lambda: current_platform.is_out_of_tree()
            and not current_platform.support_piecewise_cuda_graph(),
        ),
        (
            "MoE A2A backend",
            lambda: resolved_view(server_args).moe_a2a_backend != "none",
        ),
        # Dynamo blocks LoRA under tc_piecewise (per-batch LoRABatchInfo
        # rebinds break guards); breakable/full support LoRA.
        ("LoRA", lambda: bool(cfg.lora_paths) or cfg.enable_lora),
        (
            "multimodal model",
            lambda: model_config_of(server_args).is_multimodal
            and not model_config_of(
                server_args
            ).is_multimodal_piecewise_cuda_graph_supported,
        ),
        (
            "GGUF quantization",
            lambda: cfg.load_format == "gguf"
            or resolved_view(server_args).quantization == "gguf"
            or check_gguf_file(cfg.model_path),
        ),
        ("DLLM (diffusion LLM)", lambda: cfg.dllm_algorithm is not None),
        (
            "CPU offload / hierarchical cache",
            lambda: cfg.cpu_offload_gb > 0 or cfg.enable_hierarchical_cache,
        ),
        (
            "deterministic inference",
            lambda: cfg.enable_deterministic_inference,
        ),
        ("PD disaggregation", lambda: cfg.disaggregation_mode != "null"),
        ("symmetric memory", lambda: cfg.enable_symm_mem),
        (
            "expert distribution recorder",
            lambda: cfg.enable_eplb
            or cfg.expert_distribution_recorder_mode is not None,
        ),
        (
            "context parallel (attn_cp_size > 1)",
            lambda: resolved_view(server_args).attn_cp_size > 1,
        ),
        ("CUDA graph debug mode", lambda: cfg.debug_cuda_graph),
        (
            "DSA prefill context parallelism",
            lambda: cfg.enable_dsa_prefill_context_parallel,
        ),
        # Capture builds a dummy extend forward with attn_dcp_metadata=None.
        (
            "decode context parallel (dcp_size > 1)",
            lambda: cfg.dcp_size > 1,
        ),
    ]
    for _name, predicate in rules:
        if predicate():
            declare_resolution(
                server_args,
                "_disable_tc_piecewise_cudagraph_if_incompatible",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
                ),
            )
            # One decision, one declaration: every rule declares the same
            # value, so a later match would only append a duplicate entry.
            break


def disable_breakable_cudagraph_if_incompatible(server_args: Any):
    """Breakable (segmented capture, no torch.compile). Breakable enforces
    memory-saver rejection in its own __init__; config-time rules can be
    added here as they're discovered.
    """

    cfg = resolving_view(server_args)
    from sglang.srt.configs.model_config import (
        is_deepseek_v4,
        uses_kda_attention,
    )
    from sglang.srt.layers.cp.bcg import supports_prefill_cp_bcg

    rules = [
        (
            "KDA hybrid linear attention",
            lambda: uses_kda_attention(model_config_of(server_args).hf_config),
        ),
        # DSV4 is BCG-compatible but introduces heavy memory pressure: the
        # c4 indexer scratch is pinned in the capture pool and OOMs. Disable.
        (
            "DeepSeek-V4 (heavy capture-pool memory pressure)",
            lambda: is_deepseek_v4(model_config_of(server_args).hf_config),
        ),
        # CP all_gather replay size mismatch under BCG.
        (
            "context parallel (attn_cp_size > 1)",
            lambda: resolved_view(server_args).attn_cp_size > 1
            and not supports_prefill_cp_bcg(server_args),
        ),
        # Capture builds a dummy extend forward with attn_dcp_metadata=None.
        (
            "decode context parallel (dcp_size > 1)",
            lambda: cfg.dcp_size > 1,
        ),
        # TBO capture is unsupported.
        (
            "two-batch overlap",
            lambda: cfg.enable_two_batch_overlap,
        ),
        (
            "unvalidated a2a backend",
            lambda: resolved_view(server_args).moe_a2a_backend
            not in ("none", "deepep", "megamoe", "flashinfer"),
        ),
        # Multimodal prefill replay faults under BCG; allowlisted archs opt back in.
        (
            "multimodal model",
            lambda: model_config_of(server_args).is_multimodal
            and not model_config_of(
                server_args
            ).is_multimodal_breakable_cuda_graph_supported,
        ),
    ]
    for name, predicate in rules:
        if predicate():
            logger.warning(
                "Breakable CUDA graph is incompatible with %s; "
                "disabling prefill CUDA graph.",
                name,
            )
            declare_resolution(
                server_args,
                "_disable_breakable_cudagraph_if_incompatible",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
                ),
            )
            return


def disable_full_prefill_cudagraph_if_incompatible(server_args: Any):
    """Full prefill CG: empty rule list today; see the experimental warning."""
    cfg = resolving_view(server_args)
    rules = []
    for name, predicate in rules:
        if predicate():
            logger.warning(
                "Full prefill CUDA graph is incompatible with %s; "
                "disabling prefill CUDA graph.",
                name,
            )
            declare_resolution(
                server_args,
                "_disable_full_prefill_cudagraph_if_incompatible",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
                ),
            )
            return


def disable_prefill_cuda_graph_for_deepseek_trtllm_mla(server_args: Any):
    """Disable prefill CUDA graph for dsr1 by default when using the trtllm_mla
    attention backend. Under any captured prefill CUDA graph (tc_piecewise or
    breakable) trtllm_mla falls back to FlashAttention for prefill and regresses
    performance, so disable whichever prefill graph backend is in effect.
    """

    cfg = resolving_view(server_args)

    if (Phase.PREFILL, "backend") in server_args._cuda_graph_config_locked:
        return
    if cfg.cuda_graph_config.prefill.backend == Backend.DISABLED:
        return
    if (
        "DeepseekV3ForCausalLM"
        not in model_config_of(server_args).hf_config.architectures
    ):
        return
    prefill_attention_backend, _ = attention_backends_of(resolved_view(server_args))
    if prefill_attention_backend != "trtllm_mla":
        return
    logger.warning(
        "Disabling prefill CUDA graph (%s) by default for the DeepSeek-V3 arch on "
        "the trtllm_mla attention backend (a captured prefill graph forces a "
        "FlashAttention fallback that regresses prefill). Set the prefill cuda graph "
        "backend explicitly (e.g. --cuda-graph-backend-prefill tc_piecewise) to override.",
        cfg.cuda_graph_config.prefill.backend,
    )
    declare_resolution(
        server_args,
        "_disable_prefill_cuda_graph_for_deepseek_trtllm_mla",
        cuda_graph_config=with_phase(
            cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
        ),
    )


def apply_deepep_adjustments(server_args: Any):
    """Config adjustments required by the DeepEP a2a backend."""
    cfg = resolving_view(server_args)
    if resolved_view(server_args).moe_a2a_backend != "deepep":
        return

    # Non-multiple-of-8 prefill buckets can hang DeepEP a2a capture under
    # breakable CUDA graph
    if cfg.cuda_graph_config.prefill.backend == Backend.BREAKABLE:
        bs = cfg.cuda_graph_config.prefill.bs
        if bs is None:
            # 2048 = documented prefill default; max_bs unresolved here.
            max_bs = cfg.cuda_graph_config.prefill.max_bs or 2048
            bs = generate_prefill_cuda_graph_batch_sizes(max_bs)
        aligned = sorted({((b + 7) // 8) * 8 for b in bs})
        if aligned != sorted(bs):
            logger.info(
                "Breakable prefill CUDA graph with DeepEP requires bucket "
                "sizes divisible by 8; aligning %s -> %s.",
                sorted(bs),
                aligned,
            )
            declare_resolution(
                server_args,
                "_apply_deepep_adjustments",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config,
                    Phase.PREFILL,
                    bs=aligned,
                    max_bs=aligned[-1],
                ),
            )


def apply_inkling_prefill_cuda_graph_default(server_args: Any):
    """Inkling opts into full-graph prefill CUDA-graph capture. Must run
    before _handle_cuda_graph_config: the generic breakable default is
    auto-disabled for this multimodal arch, and declarative model overrides
    materialize too late to steer cuda-graph resolution. Honors an explicit
    --cuda-graph-backend-prefill / --disable-prefill-cuda-graph."""

    cfg = resolving_view(server_args)
    if (
        cfg.cuda_graph_backend_prefill is not None
        or cfg.disable_prefill_cuda_graph
        or parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE
    ):
        return
    arch = model_config_of(server_args).hf_config.architectures[0]
    if arch in (
        "InklingForConditionalGeneration",
        "InklingForConditionalGenerationMTP",
    ):
        declare_resolution(
            server_args,
            "_apply_inkling_prefill_cuda_graph_default",
            cuda_graph_backend_prefill=Backend.FULL,
        )


def apply_muse_glimmer_prefill_cuda_graph_max_bs_default(server_args: Any):

    cfg = resolving_view(server_args)
    if (
        cfg.cuda_graph_max_bs_prefill is not None
        or parse_connector_type(cfg.model_path) == ConnectorType.INSTANCE
    ):
        return
    arch = model_config_of(server_args).hf_config.architectures[0]
    if arch in ("MuseGlimmerForCausalLM", "MuseGlimmerForConditionalGeneration"):
        declare_resolution(
            server_args,
            "_apply_muse_glimmer_prefill_cuda_graph_max_bs_default",
            cuda_graph_max_bs_prefill=512,
        )


def handle_cuda_graph_config(server_args: Any):
    cfg = resolving_view(server_args)

    parse_cuda_graph_config(server_args)
    apply_cuda_graph_compatibility(server_args)
    apply_deepep_adjustments(server_args)
    apply_cuda_graph_disaggregation_roles(server_args)
    validate_cuda_graph_config(server_args)
    # Warn on the final resolved config (not inside the compat cascade —
    # that path is skipped when the user explicitly sets the backend,
    # which is the only way to get 'full' for prefill today).
    if cfg.cuda_graph_config.prefill.backend == Backend.FULL:
        logger.warning(
            "cuda_graph_config[prefill].backend='full' is experimental. "
            "Use breakable or tc_piecewise for production workloads."
        )


def validate_cuda_graph_config(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.cuda_graph_config is None:
        return
    for phase in Phase.ALL:
        backend = getattr(cfg.cuda_graph_config, phase).backend
        if backend not in ALLOWED_BACKENDS_PER_PHASE[phase]:
            raise ValueError(
                f"--cuda-graph-config[{phase}].backend={backend!r} not allowed; "
                f"allowed: {ALLOWED_BACKENDS_PER_PHASE[phase]}"
            )


def generate_prefill_cuda_graph_batch_sizes(max_bs: int):
    """
    Generate the list of batch sizes for prefill CUDA graph capture
    based on max_bs. For tc_piecewise prefill, bs carries the
    captured token count (one shape knob per phase).
    """
    capture_sizes = (
        list(range(4, 33, 4))
        + list(range(48, 257, 16))
        + list(range(288, 513, 32))
        + list(range(576, 1024 + 1, 64))
        + list(range(1280, 4096 + 1, 256))
        + list(range(4608, max_bs + 1, 512))
    )

    capture_sizes = [s for s in capture_sizes if s <= max_bs]

    return capture_sizes


def generate_decode_cuda_graph_batch_sizes(server_args: Any, max_bs: int):
    """
    Generate the list of batch sizes for CUDA graph capture based on max_bs.
    This integrates the logic from cuda_graph_runner.py.
    """
    cfg = resolving_view(server_args)
    # Handle disable_cuda_graph_padding as the first condition for both spec and non-spec
    if cfg.disable_cuda_graph_padding:
        capture_bs = list(range(1, max_bs + 1))
    elif cfg.speculative_algorithm is None:
        # Normal case:
        capture_bs = (
            [1, 2, 4, 8, 12]
            + list(range(16, 257, 8))
            + list(range(272, 512, 16))
            + list(range(512, max_bs + 1, 32))
        )
    else:
        # Spec decoding case: less padding for smaller batch sizes
        capture_bs = (
            list(range(1, 9, 1))
            + list(range(10, 33, 2))
            + list(range(40, 65, 4))
            + list(range(72, 257, 8))
            + list(range(272, max_bs + 1, 16))
        )

    capture_bs = [bs for bs in capture_bs if bs <= max_bs]

    if max_bs not in capture_bs:
        capture_bs.append(max_bs)

    return capture_bs


def generate_cpu_graph_batch_sizes(server_args: Any):
    """
    Generate the list of batch sizes for CPU graph capture based on torch_compile_max_bs.
    """
    cfg = resolving_view(server_args)
    if cfg.disable_cuda_graph_padding:
        capture_bs = list(range(1, cfg.torch_compile_max_bs + 1))
    else:
        capture_bs = sorted(
            set().union(
                range(1, 17),
                range(18, 31, 2),
                range(32, 81, 4),
                range(84, cfg.torch_compile_max_bs + 1, 8),
                {cfg.torch_compile_max_bs},
            )
        )
    capture_bs = [bs for bs in capture_bs if bs <= cfg.torch_compile_max_bs]

    return capture_bs


def apply_cuda_graph_disaggregation_roles(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.disaggregation_mode == "prefill":
        if (Phase.DECODE, "backend") not in server_args._cuda_graph_config_locked:
            declare_resolution(
                server_args,
                "_apply_cuda_graph_disaggregation_roles",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
                ),
            )
    elif cfg.disaggregation_mode == "decode":
        if (Phase.PREFILL, "backend") not in server_args._cuda_graph_config_locked:
            declare_resolution(
                server_args,
                "_apply_cuda_graph_disaggregation_roles",
                cuda_graph_config=with_phase(
                    cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
                ),
            )
