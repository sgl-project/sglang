from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arg_groups.overrides import (
    _deepseek_v4_kv_cache_dtype,
    declare_resolution,
    model_config_of,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import is_npu

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
            "DeepSeekV4 zigzag CP requires the NPU backend; the CUDA backend "
            "reindexes with interleave order."
        )

    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        enable_dp_attention=True,
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        moe_dense_tp_size=1,
    )
    declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        attn_cp_size=cfg.tp_size // cfg.dp_size,
    )
    if not is_npu():
        assert cfg.dp_size == 1, (
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
        f"dp_size={cfg.dp_size}, moe_dense_tp_size={cfg.moe_dense_tp_size}, "
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
    if cfg.enable_encoder_swa_bounded_replay:
        from sglang.srt.model_executor.cuda_graph_config import Backend

        incompatible = (
            (
                "non-CUDA/ROCm hardware",
                not (get_platform().is_cuda or get_platform().is_hip),
            ),
            (
                "prefill CUDA graphs",
                cfg.cuda_graph_config.prefill.backend != Backend.DISABLED,
            ),
            ("DP attention", cfg.enable_dp_attention),
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
        # The trtllm-gen path has no uniform-FP8 pool for V4.1's ratio-1/2 layers.
        ("the trtllm DSv4 attention backend", cfg.dsv4_attn_backend == "trtllm"),
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

        if (
            read_ragged_verify_mode() is not RaggedVerifyMode.STATIC
            or cfg.disaggregation_transfer_backend != "mooncake"
            or cfg.dp_size != 1
            or cfg.enable_dp_attention
            or cfg.attn_cp_size != 1
            or cfg.dcp_size != 1
        ):
            raise ValueError(
                "DeepSeek-V4.1 DSpark PD requires static verify, Mooncake, "
                "DP=1 and CP=1. Both servers must enable DSpark with the same "
                "block size and TP size."
            )

    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase

    prefill_graph = cfg.cuda_graph_config.prefill
    if prefill_graph.backend != Backend.DISABLED and prefill_graph.max_seq_len is None:
        # The captured low-ratio indexer scores a static context width; 16k
        # keeps it inside the candidate window at under 1 ms per layer.
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
        from sglang.srt.model_executor.cuda_graph_config import Backend

        # Late layers see a per-request tail slice, not the captured prefill shape.
        incompatible = (
            (
                "the prefill CUDA graph",
                cfg.cuda_graph_config.prefill.backend != Backend.DISABLED,
            ),
            # input_ids_global is a DP-wide gather, so the tail slice cannot apply.
            ("DP attention", cfg.enable_dp_attention),
        )
        for feature, enabled in incompatible:
            if enabled:
                raise ValueError(
                    "--enable-decoder-swa-bounded-replay cannot be combined with "
                    f"{feature} yet; disable one of them."
                )
