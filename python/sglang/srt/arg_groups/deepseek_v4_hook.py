from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.model_override_base import (
    attention_backends_of,
    context_parallel_attn_dp_size,
)
from sglang.srt.arg_groups.overrides import (
    _deepseek_v4_kv_cache_dtype,
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import attn_dp_enabled_of, get_platform
from sglang.srt.utils.common import is_gfx95_supported, is_npu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Apply DeepSeek V4 environment defaults, request limits, and validation."""
    cfg = resolving_view(server_args)

    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
    # (MI355X), which breaks the disaggregation nightly. Keep the previous
    # (dense prefill) behavior on ROCm until the sparse kernel is validated
    # there;
    if get_platform().is_hip:
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL by default on ROCm/HIP "
            f"for {model_arch}; set it explicitly to override."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)

    # The kv-cache dtype default moved to the resolution pipeline
    # (arg_groups/overrides.py: _deepseek_v4_kv_cache_dtype), invoked here at
    # its legacy slot.

    run_post_process_pass(server_args, _deepseek_v4_kv_cache_dtype)
    validate_deepseek_v41_sm80_experimental(server_args)

    if cfg.dsv4_attn_backend == "trtllm":
        from sglang.srt.utils.common import is_sm100_supported

        assert cfg.device == "cuda" and is_sm100_supported(), (
            "--dsv4-attn-backend trtllm requires an SM100/SM103 (Blackwell) GPU."
        )
        # The resolution pipeline materializes "auto" as fp8_e4m3 on CUDA.
        assert cfg.kv_cache_dtype in ("auto", "fp8_e4m3"), (
            "--dsv4-attn-backend trtllm requires kv_cache_dtype=fp8_e4m3, "
            f"got {cfg.kv_cache_dtype}."
        )
        assert not cfg.enable_hisparse, (
            "--dsv4-attn-backend trtllm does not support enable_hisparse."
        )
        assert not (
            cfg.attn_cp_size > 1 or cfg.dcp_size > 1 or cfg.enable_prefill_cp
        ), (
            "--dsv4-attn-backend trtllm does not support context parallelism "
            "(prefill CP, attention CP, or decode CP)."
        )
        # The trtllm backend stores KV in a 512-byte uniform-FP8 layout while
        # FlashMLA uses the 584-byte packed layout; the PD handshake only
        # compares kv_cache_dtype, so mismatched prefill/decode backends would
        # pass the check and transfer garbage. Reject until the handshake
        # carries a layout identifier and the path is tested (#37838).
        assert cfg.disaggregation_mode == "null", (
            "--dsv4-attn-backend trtllm does not support PD disaggregation yet "
            "(uniform-FP8 KV layout is not part of the PD handshake; see "
            "https://github.com/sgl-project/sglang/issues/37838)."
        )
        # The trtllm-gen semaphore buffer is sized from the prefill chunk
        # bound; with chunking disabled a single long request has no bound.
        assert cfg.chunked_prefill_size is not None and cfg.chunked_prefill_size > 0, (
            "--dsv4-attn-backend trtllm requires chunked prefill "
            "(--chunked-prefill-size > 0)."
        )
        logger.info(
            "DeepSeek V4 attention: trtllm backend enabled "
            "(uniform-FP8 KV pool, decode + sparse prefill)."
        )

    if cfg.max_running_requests is None:
        declare_resolution(
            server_args,
            "apply_deepseek_v4_defaults",
            max_running_requests=256,
        )
        logger.warning(
            f"Setting max_running_requests to {cfg.max_running_requests} for {model_arch}."
        )

    if cfg.speculative_algorithm is not None:
        assert cfg.speculative_algorithm in (
            "EAGLE",
            "DSPARK",
        ), (
            f"Only EAGLE and DSPARK speculative algorithms are supported for {model_arch}"
        )
        if cfg.speculative_algorithm == "EAGLE":
            assert cfg.speculative_eagle_topk == 1, (
                f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"
            )


def validate_deepseek_v41_sm80_experimental(server_args: ServerArgs) -> None:
    """Fail closed for opt-in V4.1 SM80 eager or BS=1 full-decode experiments."""
    # Do not query CUDA or change declarations for ordinary serving. In particular,
    # EnvBool accepts more spellings than this experimental opt-in permits.
    if envs.SGLANG_DSV41_SM80_EXPERIMENTAL.get() != "1":
        return
    cfg = resolving_view(server_args)
    model_config = model_config_of(server_args)
    if model_config.hf_text_config.model_type != "deepseek_v41" or cfg.device != "cuda":
        return

    import torch

    # Startup runs before worker device selection. Probe the actual allocation
    # base rather than CUDA's current/default device or a cached platform gate.
    if torch.cuda.get_device_capability(cfg.base_gpu_id) != (8, 0):
        return

    prefix = "DeepSeek-V4.1 experimental SM80"
    unsupported = (
        ("TRT-LLM attention", cfg.dsv4_attn_backend == "trtllm"),
        (
            "unified KV layout",
            envs.SGLANG_HACK_FLASHMLA_BACKEND.get() == "unified_kv_triton"
            or envs.SGLANG_DSV4_UNIFIED_KV_FP8.get(),
        ),
        ("Q8 sparse prefill", cfg.dsv4_prefill_backend == "flashmla_sparse_q8"),
        ("FP4 indexer", cfg.enable_deepseek_v4_fp4_indexer),
        (
            "context parallelism (CP/DCP)",
            cfg.enable_prefill_cp or cfg.attn_cp_size > 1 or cfg.dcp_size > 1,
        ),
        ("DP attention", attn_dp_enabled_of(cfg)),
        ("pipeline parallelism", cfg.pp_size > 1),
        ("multiple nodes", cfg.nnodes > 1),
        ("PD disaggregation", cfg.disaggregation_mode != "null"),
        (
            "speculative decoding / MTP",
            cfg.speculative_algorithm is not None
            or cfg.speculative_draft_model_path is not None
            or cfg.decoupled_spec_role != "null",
        ),
        ("mixed prefill/decode", cfg.enable_mixed_chunk),
        ("decoder SWA bounded replay", cfg.enable_decoder_swa_bounded_replay),
        ("encoder SWA bounded replay", cfg.enable_encoder_swa_bounded_replay),
        ("HiCache", cfg.enable_hierarchical_cache),
        ("HiSparse", cfg.enable_hisparse),
        ("LoRA", cfg.enable_lora),
        (
            "vision / multimodal pipeline (use --language-model-only)",
            model_config.is_multimodal and not cfg.language_model_only,
        ),
        ("two-batch overlap", cfg.enable_two_batch_overlap),
    )
    for feature, enabled in unsupported:
        if enabled:
            raise ValueError(f"{prefix} does not support {feature}; disable it.")

    # Validate the resolved config, not just legacy booleans: explicit per-phase
    # selectors and JSON override those booleans and can re-enable capture.
    if envs.SGLANG_DSV41_SM80_FULL_GRAPH.get() == "1":
        decode_graph = cfg.cuda_graph_config.decode
        if (
            decode_graph.backend != Backend.FULL
            or cfg.cuda_graph_config.prefill.backend != Backend.DISABLED
        ):
            raise ValueError(
                f"{prefix} full-graph experiment requires decode backend full "
                "and prefill backend disabled."
            )
        # Unresolved defaults must not widen the capture set after this guard.
        if (
            cfg.max_running_requests != 1
            or decode_graph.bs != [1]
            or decode_graph.max_bs != 1
        ):
            raise ValueError(
                f"{prefix} full-graph experiment requires max_running_requests=1 "
                "and resolved decode bs=[1], max_bs=1."
            )
        if envs.SGLANG_DSV41_SM80_FFN_GRAPH.get() == "1":
            raise ValueError(
                f"{prefix} full-graph experiment forbids "
                "SGLANG_DSV41_SM80_FFN_GRAPH=1 (nested capture)."
            )
    elif (
        cfg.cuda_graph_config.decode.backend != Backend.DISABLED
        or cfg.cuda_graph_config.prefill.backend != Backend.DISABLED
    ):
        raise ValueError(
            f"{prefix} requires eager decode and prefill: use "
            "--disable-decode-cuda-graph --disable-prefill-cuda-graph "
            "and remove CUDA graph backend overrides."
        )

    if cfg.dsv4_attn_backend not in ("auto", "flashmla"):
        raise ValueError(
            f"{prefix} requires --dsv4-attn-backend flashmla, "
            f"got {cfg.dsv4_attn_backend!r}."
        )
    prefill_backend, decode_backend = attention_backends_of(resolved_view(server_args))
    if (prefill_backend, decode_backend) != ("dsv4", "dsv4"):
        raise ValueError(
            f"{prefix} requires dsv4 attention for both phases, "
            f"got prefill={prefill_backend!r}, decode={decode_backend!r}."
        )
    if cfg.kv_cache_dtype != "fp8_e4m3":
        raise ValueError(
            f"{prefix} requires --kv-cache-dtype fp8_e4m3, got {cfg.kv_cache_dtype!r}."
        )
    if cfg.dsv4_attn_backend == "auto":
        # Keep the existing dsv4 selector; the device-aware backend owns fallback.
        declare_resolution(
            server_args,
            "validate_deepseek_v41_sm80_experimental",
            dsv4_attn_backend="flashmla",
        )


def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    cfg = resolving_view(server_args)
    if not cfg.enable_prefill_cp:
        return

    if cfg.cp_strategy not in ("interleave", "zigzag"):
        raise ValueError(
            f"DeepSeekV4 only supports interleave/zigzag CP strategy, got {cfg.cp_strategy}"
        )

    if cfg.cp_strategy == "zigzag" and not is_npu():
        raise ValueError(
            "DeepSeekV4 zigzag CP requires the NPU backend; CUDA/HIP backends "
            "reindex with interleave order."
        )
    if get_platform().is_hip:
        prefill_backend, decode_backend = attention_backends_of(
            resolved_view(server_args)
        )
        if (prefill_backend, decode_backend) != ("dsv4", "dsv4"):
            raise ValueError(
                "DeepSeekV4 prefill CP on HIP requires the dsv4 attention "
                f"backend for both phases, got prefill={prefill_backend!r}, "
                f"decode={decode_backend!r}."
            )
        unsupported = (
            ("multiple nodes", cfg.nnodes > 1),
            (
                "DeepSeek-V4.1",
                model_config_of(server_args).hf_config.model_type == "deepseek_v41",
            ),
            (
                "--enable-decoder-swa-bounded-replay",
                cfg.enable_decoder_swa_bounded_replay,
            ),
            ("--enable-two-batch-overlap", cfg.enable_two_batch_overlap),
        )
        for feature, enabled in unsupported:
            if enabled:
                raise ValueError(
                    f"DeepSeekV4 prefill CP on HIP does not support {feature} yet."
                )

    attn_dp_size = context_parallel_attn_dp_size(cfg, "DeepSeek-V4 context parallelism")
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        attn_dp_size=attn_dp_size,
        dp_size=1,
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        moe_dense_tp_size=1,
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        attn_cp_size=cfg.tp_size // attn_dp_size,
    )
    if not is_npu():
        assert attn_dp_size == 1, (
            "For round-robin split mode, dp attention is not supported."
        )
        assert cfg.tp_size <= 8, (
            "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
        )
    supported_a2a_backends = ("none", "deepep", "megamoe", "mori")
    if cfg.moe_a2a_backend not in supported_a2a_backends:
        raise ValueError(
            f"DeepSeekV4 CP supports moe_a2a_backend in {supported_a2a_backends}, "
            f"got {cfg.moe_a2a_backend!r}."
        )
    if model_config_of(server_args).hf_config.model_type != "deepseek_v41":
        # The CP-aware sparse prefill chunk cache is validated on V4.1 only.
        logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL because DeepSeekV4 "
            "context parallelism is enabled."
        )
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"strategy={cfg.cp_strategy}, "
        f"attn_dp_size={attn_dp_size}, moe_dense_tp_size={cfg.moe_dense_tp_size}, "
        f"attn_cp_size={cfg.attn_cp_size}, ep_size={cfg.ep_size}, tp_size={cfg.tp_size}"
    )


def validate_deepseek_v41_features(server_args: ServerArgs) -> None:
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )

    cfg = resolving_view(server_args)
    if model_config_of(server_args).hf_config.model_type != "deepseek_v41":
        if cfg.enable_encoder_swa_bounded_replay:
            raise ValueError(
                "--enable-encoder-swa-bounded-replay requires DeepSeek-V4.1"
            )
        return
    if (
        cfg.dsv4_attn_backend == "trtllm"
        and cfg.cuda_graph_config.prefill.backend
        not in (Backend.DISABLED, Backend.BREAKABLE)
    ):
        raise ValueError(
            "DeepSeek-V4.1 TRT-LLM requires --cuda-graph-backend-prefill "
            "disabled or breakable"
        )
    if cfg.enable_encoder_swa_bounded_replay:
        incompatible = (
            (
                "hardware other than CUDA or gfx950",
                not (
                    get_platform().is_cuda
                    or (get_platform().is_hip and is_gfx95_supported())
                ),
            ),
            (
                "prefill CUDA graphs",
                cfg.cuda_graph_config.prefill.backend != Backend.DISABLED,
            ),
            ("DP attention", attn_dp_enabled_of(cfg)),
            ("context parallelism", cfg.attn_cp_size > 1),
            ("external cache linker", cfg.enable_unified_cache_external_linker),
            ("unified memory", cfg.enable_unified_memory),
            ("PD disaggregation", cfg.disaggregation_mode != "null"),
            ("mixed prefill/decode", cfg.enable_mixed_chunk),
            ("LoRA", cfg.enable_lora),
            ("radix sessions", cfg.enable_session_radix_cache),
        )
        for feature, enabled in incompatible:
            if enabled:
                raise ValueError(
                    f"--enable-encoder-swa-bounded-replay does not support {feature} yet"
                )
        if (
            cfg.max_running_requests is None
            or cfg.max_running_requests <= 0
            or not cfg.chunked_prefill_size
            or cfg.chunked_prefill_size < 128
        ):
            raise ValueError(
                "encoder SWA replay requires explicit --max-running-requests and --chunked-prefill-size >= 128"
            )

    unsupported = (
        (
            "speculative decoding other than DSpark",
            cfg.speculative_algorithm is not None
            and str(cfg.speculative_algorithm).upper() != "DSPARK",
        ),
        ("HiSparse", cfg.enable_hisparse),
        ("the unified KV layout", is_unified_kv_triton()),
        (
            "TRT-LLM with encoder SWA bounded replay",
            cfg.dsv4_attn_backend == "trtllm" and cfg.enable_encoder_swa_bounded_replay,
        ),
        ("two-batch overlap", cfg.enable_two_batch_overlap),
        ("pipeline parallelism", cfg.pp_size > 1),
    )
    for feature, enabled in unsupported:
        if enabled:
            raise ValueError(
                f"DeepSeek-V4.1 does not support {feature} yet; disable it to "
                "serve this model."
            )

    if cfg.disaggregation_mode != "null" and cfg.speculative_algorithm is not None:
        from sglang.srt.speculative.ragged_verify import (
            RaggedVerifyMode,
            read_ragged_verify_mode,
        )

        # CP size may still be implicit here; validate_deepseek_v4_cp resolves it.
        supported_prefill_cp = (
            cfg.disaggregation_mode == "prefill"
            and cfg.enable_prefill_cp
            and cfg.cp_strategy == "interleave"
        )
        if (
            read_ragged_verify_mode() is not RaggedVerifyMode.STATIC
            or cfg.disaggregation_transfer_backend not in ("mooncake", "nixl")
            or (
                (cfg.enable_prefill_cp or cfg.attn_cp_size != 1)
                and not supported_prefill_cp
            )
            or cfg.dcp_size != 1
        ):
            raise ValueError(
                "DeepSeek-V4.1 DSpark PD requires static verify, Mooncake or NIXL, "
                "DCP=1, and CP=1 on decode. Prefill CP requires "
                "--enable-prefill-cp with the interleave strategy. "
                "DP attention is supported when "
                "both servers use the same block size and target/draft KV layout."
            )

    prefill_graph = cfg.cuda_graph_config.prefill
    cp_breakable_prefill = (
        cfg.enable_prefill_cp
        and cfg.cp_strategy == "interleave"
        and cfg.tp_size > 1
        and prefill_graph.backend == Backend.BREAKABLE
    )
    if (
        prefill_graph.backend != Backend.DISABLED
        and prefill_graph.max_seq_len is None
        and not cp_breakable_prefill
    ):
        declare_resolution(
            server_args,
            "validate_deepseek_v41_features",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, max_seq_len=16 * 1024
            ),
        )
        logger.warning(
            "Setting cuda_graph_config[prefill].max_seq_len to 16384 for "
            "DeepSeek-V4.1; longer contexts run eager prefill."
        )

    if cfg.enable_decoder_swa_bounded_replay:
        # Late layers see a per-request tail slice, not the captured prefill shape.
        # The breakable backend is admitted: the tail switch runs as eager breaks and
        # the tail is a fixed-capacity row block inside the captured segments.
        incompatible = (
            (
                "the prefill CUDA graph",
                cfg.cuda_graph_config.prefill.backend
                not in (Backend.DISABLED, Backend.BREAKABLE),
            ),
            # Late-layer communication uses one row count per attention-DP rank.
            # Attention TP/CP shards need a separate tail padding layout.
            (
                "DP attention other than attention TP 1 with supported MoE backends",
                attn_dp_enabled_of(cfg)
                and not (
                    cfg.attn_dp_size == cfg.tp_size
                    and cfg.attn_cp_size == 1
                    and (
                        (get_platform().is_hip and cfg.moe_a2a_backend == "none")
                        or (
                            get_platform().is_cuda
                            and cfg.moe_a2a_backend in ("none", "megamoe")
                        )
                    )
                ),
            ),
        )
        for feature, enabled in incompatible:
            if enabled:
                raise ValueError(
                    "--enable-decoder-swa-bounded-replay cannot be combined with "
                    f"{feature} yet; disable one of them."
                )
