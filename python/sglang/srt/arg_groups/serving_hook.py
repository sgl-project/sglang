# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for serving-surface and multimodal entry validation."""

from __future__ import annotations

import json
import logging
import os
import random
import socket
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, with_phase
from sglang.srt.runtime_context import get_platform
from sglang.srt.utils.common import (
    configure_media_url_security,
    get_device,
    is_gfx95_supported,
    is_mnnvl_fabric_device,
)
from sglang.utils import is_in_ci

logger = logging.getLogger(__name__)


def handle_ssl_validation(server_args: Any):
    """Ensure SSL arguments are consistent and referenced files exist."""
    cfg = resolving_view(server_args)
    if cfg.ssl_keyfile and not cfg.ssl_certfile:
        raise ValueError(
            "--ssl-keyfile requires --ssl-certfile to be specified as well."
        )
    if cfg.ssl_certfile and not cfg.ssl_keyfile:
        raise ValueError(
            "--ssl-certfile requires --ssl-keyfile to be specified as well."
        )
    if not cfg.ssl_certfile and not cfg.ssl_keyfile:
        if cfg.ssl_ca_certs:
            raise ValueError(
                "--ssl-ca-certs has no effect without --ssl-certfile and --ssl-keyfile."
            )
        if cfg.ssl_keyfile_password:
            raise ValueError(
                "--ssl-keyfile-password has no effect without --ssl-certfile and --ssl-keyfile."
            )
    # Validate files exist early to avoid late failures after model loading.
    if cfg.ssl_keyfile and not os.path.isfile(cfg.ssl_keyfile):
        raise ValueError(
            f"SSL key file not found: '{cfg.ssl_keyfile}'. "
            f"Please check the --ssl-keyfile path."
        )
    if cfg.ssl_certfile and not os.path.isfile(cfg.ssl_certfile):
        raise ValueError(
            f"SSL certificate file not found: '{cfg.ssl_certfile}'. "
            f"Please check the --ssl-certfile path."
        )
    if cfg.ssl_ca_certs and not os.path.isfile(cfg.ssl_ca_certs):
        raise ValueError(
            f"SSL CA certificates file not found: '{cfg.ssl_ca_certs}'. "
            f"Please check the --ssl-ca-certs path."
        )
    if cfg.enable_ssl_refresh and not (cfg.ssl_certfile and cfg.ssl_keyfile):
        raise ValueError(
            "--enable-ssl-refresh requires --ssl-certfile and --ssl-keyfile "
            "to be specified."
        )

    if cfg.enable_http2:
        if not 0 < cfg.http2_max_concurrent_streams < 2**32:
            raise ValueError(
                "--http2-max-concurrent-streams must be between 1 and 4294967295."
            )
        if not 1024 <= cfg.http2_initial_connection_window_size < 2**31:
            raise ValueError(
                "--http2-initial-connection-window-size must be between "
                "1024 and 2147483647."
            )

        try:
            import granian  # noqa: F401
        except ImportError:
            raise ValueError(
                "--enable-http2 requires the 'granian' package. "
                'Install it with: pip install "sglang[http2]"'
            )

        if cfg.enable_ssl_refresh:
            raise ValueError(
                "--enable-ssl-refresh is not supported with --enable-http2. "
                "Granian does not support SSL certificate hot-reloading. "
                "Use Uvicorn (the default) or handle certificate rotation externally."
            )


def handle_asr_validation(server_args: Any):
    """Validate transcription/ASR-specific server args."""
    cfg = resolving_view(server_args)
    if cfg.asr_max_buffer_seconds <= 0:
        raise ValueError(
            f"--asr-max-buffer-seconds must be positive "
            f"(got {cfg.asr_max_buffer_seconds})."
        )
    if cfg.asr_max_concurrent_sessions <= 0:
        raise ValueError(
            f"--asr-max-concurrent-sessions must be positive "
            f"(got {cfg.asr_max_concurrent_sessions})."
        )


def handle_multimodal(server_args: Any):
    """Validate mm_process_config structure before model loading."""
    cfg = resolving_view(server_args)
    if (
        cfg.mm_preprocess_cache_size_mb is not None
        and cfg.mm_preprocess_cache_size_mb < 0
    ):
        raise ValueError("mm_preprocess_cache_size_mb must be non-negative")
    if cfg.mm_process_config is not None:
        if not isinstance(cfg.mm_process_config, dict):
            raise TypeError(
                f"mm_process_config must be a dict, "
                f"but got {type(cfg.mm_process_config)}"
            )
        for key in ("image", "video", "audio"):
            if key in cfg.mm_process_config and not isinstance(
                cfg.mm_process_config[key], dict
            ):
                raise TypeError(
                    f"mm_process_config['{key}'] must be a dict, "
                    f"but got {type(cfg.mm_process_config[key])}"
                )


def handle_crash_dump_env(server_args: Any):
    cfg = resolving_view(server_args)
    if not cfg.crash_dump_folder:
        return
    _CUDA_COREDUMP_DEFAULTS = {
        "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "1",
        "CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1",
        "CUDA_COREDUMP_SHOW_PROGRESS": "1",
        "CUDA_COREDUMP_GENERATION_FLAGS": (
            "skip_nonrelocated_elf_images,skip_global_memory,"
            "skip_shared_memory,skip_local_memory,skip_constbank_memory"
        ),
        "CUDA_COREDUMP_FILE": f"{cfg.crash_dump_folder}/%h/core.cuda.%t.%p",
        "CUDA_COREDUMP_PIPE": "/tmp/corepipe.cuda.%h.%p",
    }
    for key, value in _CUDA_COREDUMP_DEFAULTS.items():
        if key not in os.environ:
            os.environ[key] = value
            logger.info("Auto-set %s=%s (from --crash-dump-folder)", key, value)

    coredump_dir = os.path.dirname(
        os.environ["CUDA_COREDUMP_FILE"].replace("%h", socket.gethostname())
    )
    if "%" in coredump_dir:
        logger.warning(
            "Cannot pre-create CUDA coredump directory %s: only %%h is "
            "supported in the directory part of CUDA_COREDUMP_FILE; "
            "coredumps may fail to write.",
            coredump_dir,
        )
    elif coredump_dir:
        try:
            os.makedirs(coredump_dir, exist_ok=True)
        except OSError as e:
            logger.warning(
                "Failed to create CUDA coredump directory %s: %s; "
                "coredumps may fail to write.",
                coredump_dir,
                e,
            )


def handle_media_url_security(server_args: Any):
    """Normalize and publish the media URL policy before workers start."""
    cfg = resolving_view(server_args)
    declare_resolution(
        server_args,
        "_handle_media_url_security",
        allowed_media_domains=configure_media_url_security(
            cfg.allowed_media_domains,
            cfg.media_url_max_file_size_mb,
        ),
    )


def handle_load_balance_method(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.disaggregation_mode not in ("null", "prefill", "decode"):
        raise ValueError(f"Invalid disaggregation_mode={cfg.disaggregation_mode!r}")

    if cfg.load_balance_method == "auto":
        # Default behavior:
        # - non-PD: round_robin
        # - PD prefill: follow_bootstrap_room
        # - PD decode: round_robin
        declare_resolution(
            server_args,
            "_handle_load_balance_method",
            load_balance_method=(
                "follow_bootstrap_room"
                if cfg.disaggregation_mode == "prefill"
                else "round_robin"
            ),
        )
        return


def handle_grammar_backend(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.grammar_backend is None:
        declare_resolution(
            server_args, "_handle_grammar_backend", grammar_backend="xgrammar"
        )


def handle_debug_utils(server_args: Any):
    cfg = resolving_view(server_args)
    if is_in_ci() and cfg.soft_watchdog_timeout is None:
        logger.info("Set soft_watchdog_timeout since in CI")
        declare_resolution(
            server_args, "_handle_debug_utils", soft_watchdog_timeout=300
        )


def handle_deprecated_args(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.disable_fast_image_processor:
        if cfg.image_processor_backend not in {"auto", "pil"}:
            raise ValueError(
                "--disable-fast-image-processor conflicts with "
                f"--image-processor-backend={cfg.image_processor_backend}."
            )
        logger.warning(
            "--disable-fast-image-processor is deprecated; use "
            "--image-processor-backend=pil instead."
        )
        declare_resolution(
            server_args, "_handle_deprecated_args", image_processor_backend="pil"
        )

    # Handle deprecated tool call parsers
    deprecated_tool_call_parsers = {"qwen25": "qwen", "glm45": "glm"}
    if cfg.tool_call_parser in deprecated_tool_call_parsers:
        logger.warning(
            f"The tool_call_parser '{cfg.tool_call_parser}' is deprecated. Please use '{deprecated_tool_call_parsers[cfg.tool_call_parser]}' instead."
        )
        declare_resolution(
            server_args,
            "_handle_deprecated_args",
            tool_call_parser=deprecated_tool_call_parsers[cfg.tool_call_parser],
        )

    # When user passes --enable-flashinfer-allreduce-fusion, enable with auto backend
    if (
        cfg.enable_flashinfer_allreduce_fusion
        and cfg.flashinfer_allreduce_fusion_backend is None
    ):
        logger.warning(
            "--enable-flashinfer-allreduce-fusion is deprecated. "
            "Please use --flashinfer-allreduce-fusion-backend=auto instead."
        )
        declare_resolution(
            server_args,
            "_handle_deprecated_args",
            flashinfer_allreduce_fusion_backend="auto",
        )
    declare_resolution(
        server_args,
        "_handle_deprecated_args",
        enable_flashinfer_allreduce_fusion=False,
    )
    # Deprecated attention-backend alias: "compressed" -> "dsv4".
    renamed = {}
    for attr in (
        "attention_backend",
        "decode_attention_backend",
        "prefill_attention_backend",
        "speculative_draft_attention_backend",
    ):
        if getattr(server_args, attr, None) == "compressed":
            logger.warning(
                "--%s=compressed is deprecated; use 'dsv4' instead.",
                attr.replace("_", "-"),
            )
            renamed[attr] = "dsv4"
    if renamed:
        declare_resolution(server_args, "_handle_deprecated_args", **renamed)

    # --grpc-mode is a deprecated alias for --smg-grpc-mode.
    if cfg.grpc_mode and not cfg.smg_grpc_mode:
        logger.warning(
            "--grpc-mode is deprecated and will be removed in a future "
            "version. Use --smg-grpc-mode for the legacy SMG gRPC server, "
            "or --grpc-port for the native gRPC server."
        )
        declare_resolution(
            server_args,
            "_handle_deprecated_args",
            smg_grpc_mode=True,
        )

    # Native gRPC tuning knob is env-only; --grpc-port (CLI) enables the
    # native server, falling back to SGLANG_GRPC_PORT.
    declare_resolution(
        server_args,
        "_handle_deprecated_args",
        grpc_worker_threads=envs.SGLANG_GRPC_WORKER_THREADS.get(),
    )

    grpc_port_env = envs.SGLANG_GRPC_PORT.get()
    if cfg.grpc_port is None and grpc_port_env is not None:
        declare_resolution(
            server_args,
            "_handle_deprecated_args",
            grpc_port=grpc_port_env,
        )

    # Legacy SMG defaults its port to --port + 10000. Derive/validate only
    # when gRPC is in use, so HTTP-only high ports don't fail validation.
    legacy_grpc = cfg.smg_grpc_mode or cfg.grpc_mode
    if legacy_grpc and cfg.grpc_port is None:
        declare_resolution(
            server_args,
            "_handle_deprecated_args",
            grpc_port=cfg.port + 10000,
        )

    if cfg.grpc_port is not None:
        if not (1 <= cfg.grpc_port <= 65535):
            raise ValueError(
                "--grpc-port / SGLANG_GRPC_PORT "
                f"({cfg.grpc_port}) must be between 1 and 65535"
            )
        if cfg.grpc_worker_threads is not None and cfg.grpc_worker_threads < 1:
            raise ValueError(
                f"SGLANG_GRPC_WORKER_THREADS ({cfg.grpc_worker_threads}) must be >= 1"
            )

    # Native gRPC is incompatible with launch paths it doesn't wire into.
    # Legacy takes precedence over grpc_port, keeping re-runs idempotent.
    native_grpc = cfg.grpc_port is not None and not legacy_grpc
    if cfg.sidecar_args is not None:
        if cfg.sidecar is None:
            raise ValueError("--sidecar-args requires --sidecar.")
        if not isinstance(cfg.sidecar_args, list) or not all(
            isinstance(arg, str) for arg in cfg.sidecar_args
        ):
            raise ValueError("--sidecar-args must be a JSON array of strings.")
    if cfg.sidecar is not None:
        if not cfg.sidecar.strip():
            raise ValueError("--sidecar must not be empty.")
        if legacy_grpc:
            raise ValueError(
                "--sidecar requires SGLang's native gRPC server; "
                "it cannot be combined with --smg-grpc-mode/--grpc-mode."
            )
        if cfg.grpc_port is None:
            raise ValueError("--sidecar requires --grpc-port or SGLANG_GRPC_PORT.")
    if native_grpc:
        if cfg.use_ray:
            raise ValueError(
                "--grpc-port is not supported with --use-ray: the Ray "
                "serve launch path does not start the native gRPC server."
            )
        if cfg.encoder_only:
            raise ValueError(
                "--grpc-port is not supported with --encoder-only: "
                "encoder disaggregation uses its own server."
            )
        if cfg.tokenizer_worker_num > 1:
            raise ValueError(
                "Native gRPC does not yet support --tokenizer-worker-num > 1. "
                "Unset --grpc-port or set --tokenizer-worker-num 1."
            )
        if cfg.api_key or cfg.admin_api_key:
            raise ValueError(
                "--grpc-port is incompatible with --api-key/--admin-api-key: "
                "the native gRPC listener bypasses HTTP auth middleware."
            )


def handle_environment_variables(server_args: Any):
    cfg = resolving_view(server_args)
    handle_multimodal_feature_transport(server_args)
    envs.SGLANG_ENABLE_TORCH_COMPILE.set("1" if cfg.enable_torch_compile else "0")
    if cfg.mamba_ssm_dtype is not None:
        envs.SGLANG_MAMBA_SSM_DTYPE.set(cfg.mamba_ssm_dtype)
    envs.SGLANG_DISABLE_OUTLINES_DISK_CACHE.set(
        "1" if cfg.disable_outlines_disk_cache else "0"
    )
    envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(
        "1" if cfg.enable_deterministic_inference else "0"
    )
    if cfg.enable_deterministic_inference:
        envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.set("0")
    if cfg.debug_cuda_graph:
        if not (get_platform().is_cuda or get_platform().is_hip):
            logger.warning(
                "--debug-cuda-graph is not supported on non CUDA/HIP devices. "
                "Disabling breakable CUDA graph."
            )
            declare_resolution(
                server_args, "_handle_environment_variables", debug_cuda_graph=False
            )
        else:
            envs.SGLANG_USE_BREAKABLE_CUDA_GRAPH.set("1")
            logger.warning(
                "Debug mode for CUDA graph is enabled via breakable CUDA graph. "
                "All operations will run eagerly through the graph capture/replay path."
            )
    if cfg.enable_deepseek_v4_fp4_indexer and not (
        get_platform().is_sm100 or get_platform().is_sm120 or is_gfx95_supported()
    ):
        raise ValueError(
            "--enable-deepseek-v4-fp4-indexer requires SM100, SM120, or gfx95 GPUs "
            "with FP4 indexer support."
        )
    # FP8 W_o GEMM needs DeepGEMM JIT. Enable exactly where the runtime can run
    # it, mirroring the forward scale split: the ue8m0 path
    # (DEEPGEMM_SCALE_UE8M0, true sm100, default on) or an sm90 opt-in
    # fp32-scale path (use FP4 expert ckpt). Disable in every other case.
    if get_platform().is_cuda and envs.SGLANG_OPT_FP8_WO_A_GEMM.get():
        from sglang.srt.layers import deep_gemm_wrapper

        sm = get_platform().device_sm
        explicit = envs.SGLANG_OPT_FP8_WO_A_GEMM.is_set()
        supported = deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 or (
            deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and get_platform().is_sm90
            and explicit
        )
        if not supported and explicit:
            logger.warning(
                "Disabling SGLANG_OPT_FP8_WO_A_GEMM: requires DeepGEMM JIT "
                "and sm100+ (Blackwell), or explicit opt-in on sm90; "
                "detected sm%d.",
                sm,
            )
        if not supported:
            envs.SGLANG_OPT_FP8_WO_A_GEMM.set(False)


def handle_other_validations(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.default_chat_template_kwargs is not None and not isinstance(
        cfg.default_chat_template_kwargs, dict
    ):
        raise ValueError("--default-chat-template-kwargs must decode to a JSON object")

    # Handle optimistic prefill validation
    if cfg.optimistic_prefill_attempts > 0 and cfg.disaggregation_mode == "prefill":
        if cfg.pp_size > 1:
            logger.warning("Optimistic prefill does not support pp_size > 1")
            declare_resolution(
                server_args,
                "_handle_other_validations",
                optimistic_prefill_attempts=0,
            )
        elif cfg.enable_hierarchical_cache and (
            cfg.hicache_storage_backend is not None
            or cfg.hicache_write_policy != "write_back"
        ):
            logger.warning(
                "Optimistic prefill only supports L2 hierarchical cache "
                "with write-back policy"
            )
            declare_resolution(
                server_args,
                "_handle_other_validations",
                optimistic_prefill_attempts=0,
            )
        elif resolved_view(server_args).uses_mamba_radix_cache:
            logger.warning(
                "Optimistic prefill does not support models that use mamba radix cache."
            )
            declare_resolution(
                server_args,
                "_handle_other_validations",
                optimistic_prefill_attempts=0,
            )

    # Handle model inference tensor dump.
    if cfg.debug_tensor_dump_output_folder is not None:
        logger.warning(
            "Cuda graph and server warmup are disabled because of using tensor dump mode"
        )
        declare_resolution(
            server_args,
            "_handle_other_validations",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
            ),
        )
        declare_resolution(
            server_args,
            "_handle_other_validations",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )
        declare_resolution(
            server_args, "_handle_other_validations", skip_server_warmup=True
        )

    if cfg.msprobe_dump_config is not None:
        logger.warning(
            "When msProbe is enabled, "
            "cuda graph is disabled because msProbe only supports dump in eager mode, "
            "warmup is disabled(skip_server_warmup=True) because there is no need to dump data for this stage."
        )
        declare_resolution(
            server_args,
            "_handle_other_validations",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.DECODE, backend=Backend.DISABLED
            ),
        )
        declare_resolution(
            server_args,
            "_handle_other_validations",
            cuda_graph_config=with_phase(
                cfg.cuda_graph_config, Phase.PREFILL, backend=Backend.DISABLED
            ),
        )
        declare_resolution(
            server_args, "_handle_other_validations", skip_server_warmup=True
        )

    # Validate limit_mm_per_prompt modalities
    if cfg.limit_mm_data_per_request:
        if isinstance(cfg.limit_mm_data_per_request, str):
            declare_resolution(
                server_args,
                "_handle_other_validations",
                limit_mm_data_per_request=json.loads(cfg.limit_mm_data_per_request),
            )

        if isinstance(cfg.limit_mm_data_per_request, dict):
            allowed_modalities = {"image", "video", "audio"}
            for modality in cfg.limit_mm_data_per_request.keys():
                if modality not in allowed_modalities:
                    raise ValueError(
                        f"Invalid modality '{modality}' in --limit-mm-data-per-request."
                        f"Allowed modalities are: {list(allowed_modalities)}"
                    )

    # Validate preferred_sampling_params
    if cfg.preferred_sampling_params:
        if isinstance(cfg.preferred_sampling_params, str):
            declare_resolution(
                server_args,
                "_handle_other_validations",
                preferred_sampling_params=json.loads(cfg.preferred_sampling_params),
            )

        # Validate preferred_sampling_params doesn't use tokenizer-dependent features
        if cfg.skip_tokenizer_init:
            from sglang.srt.sampling.sampling_params import SamplingParams

            test_params = SamplingParams(**cfg.preferred_sampling_params)
            # raises if tokenizer-dependent features used
            test_params.normalize(None)


def handle_missing_default_values(server_args: Any):
    from sglang.srt.arg_groups.model_path_hook import handle_modelscope_paths

    cfg = resolving_view(server_args)
    if cfg.tokenizer_path is None:
        declare_resolution(
            server_args,
            "_handle_missing_default_values",
            tokenizer_path=cfg.model_path,
        )
    if cfg.served_model_name is None:
        declare_resolution(
            server_args,
            "_handle_missing_default_values",
            served_model_name=cfg.model_path,
        )
    if cfg.device is None:
        declare_resolution(
            server_args,
            "_handle_missing_default_values",
            device=get_device(),
        )
    # strip device index from user if any (e.g. "cuda:0" -> "cuda")
    declare_resolution(
        server_args,
        "_handle_missing_default_values",
        device=cfg.device.split(":")[0],
    )
    if cfg.random_seed is None:
        declare_resolution(
            server_args,
            "_handle_missing_default_values",
            random_seed=random.randint(0, 1 << 30),
        )
    if cfg.mm_process_config is None:
        declare_resolution(
            server_args, "_handle_missing_default_values", mm_process_config={}
        )

    # Handle ModelScope model downloads
    if envs.SGLANG_USE_MODELSCOPE.get():
        handle_modelscope_paths(server_args)

    # In speculative scenario:
    # - If `speculative_draft_model_quantization` is specified, the draft model uses this quantization method.
    # - Otherwise, the draft model defaults to the same quantization as the target model.
    if cfg._speculative_draft_quantization_explicitly_set is None:
        declare_resolution(
            server_args,
            "_handle_missing_default_values",
            _speculative_draft_quantization_explicitly_set=cfg.speculative_draft_model_quantization
            is not None,
        )
    if cfg.speculative_draft_model_quantization is None:
        declare_resolution(
            server_args,
            "_handle_missing_default_values",
            speculative_draft_model_quantization=cfg.quantization,
        )

    # Resolve --quantization unquant before model config validation. Record
    # the explicit opt-out so later auto-detection does not re-enable
    # quantization.
    if cfg.quantization == "unquant":
        declare_resolution(
            server_args,
            "_handle_missing_default_values",
            quantization=None,
        )
        server_args._quantization_explicitly_unset = True
    else:
        server_args._quantization_explicitly_unset = False
    if cfg.speculative_draft_model_quantization == "unquant":
        declare_resolution(
            server_args,
            "_handle_missing_default_values",
            speculative_draft_model_quantization=None,
        )


def handle_return_hidden_states_mode(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.return_hidden_states_mode not in (None, "last", "full"):
        raise ValueError(
            "return_hidden_states_mode must be one of: None, 'last', or 'full'."
        )
    if cfg.return_hidden_states_mode is None:
        if cfg.enable_return_hidden_states:
            declare_resolution(
                server_args,
                "_handle_return_hidden_states_mode",
                return_hidden_states_mode="full",
            )
    else:
        declare_resolution(
            server_args,
            "_handle_return_hidden_states_mode",
            enable_return_hidden_states=True,
        )


def handle_prefill_delayer_env_compat(server_args: Any):
    if envs.SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE.get():
        declare_resolution(
            server_args,
            "_handle_prefill_delayer_env_compat",
            enable_prefill_delayer=True,
        )
    if x := envs.SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES.get():
        declare_resolution(
            server_args,
            "_handle_prefill_delayer_env_compat",
            prefill_delayer_max_delay_passes=x,
        )
    if x := envs.SGLANG_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK.get():
        declare_resolution(
            server_args,
            "_handle_prefill_delayer_env_compat",
            prefill_delayer_token_usage_low_watermark=x,
        )


def handle_tokenizer_batching(server_args: Any):
    cfg = resolving_view(server_args)
    if cfg.enable_tokenizer_batch_encode and cfg.enable_dynamic_batch_tokenizer:
        raise ValueError(
            "Cannot enable both --enable-tokenizer-batch-encode and --enable-dynamic-batch-tokenizer. "
            "Please choose one tokenizer batching approach."
        )

    if cfg.skip_tokenizer_init and not envs.SGLANG_RUST_SERVER.get():
        # Tokenizer workers still serve HTTP / state / output work, so
        # their fanout is preserved; detokenizer workers only decode.
        if cfg.detokenizer_worker_num != 1:
            logger.warning(
                "skip_tokenizer_init=True leaves no decode work for detokenizer workers; "
                f"forcing detokenizer_worker_num=1 (requested {cfg.detokenizer_worker_num})."
            )
            declare_resolution(
                server_args, "_handle_tokenizer_batching", detokenizer_worker_num=1
            )

        if cfg.enable_tokenizer_batch_encode:
            logger.warning(
                "skip_tokenizer_init=True ignores --enable-tokenizer-batch-encode; disabling it."
            )
            declare_resolution(
                server_args,
                "_handle_tokenizer_batching",
                enable_tokenizer_batch_encode=False,
            )

        if cfg.enable_dynamic_batch_tokenizer:
            logger.warning(
                "skip_tokenizer_init=True ignores --enable-dynamic-batch-tokenizer; disabling it."
            )
            declare_resolution(
                server_args,
                "_handle_tokenizer_batching",
                enable_dynamic_batch_tokenizer=False,
            )

        logger.info(
            "skip_tokenizer_init=True: string-based stop conditions (stop, stop_regex) "
            "and min_new_tokens are unavailable."
        )


def handle_multimodal_feature_transport(server_args: Any):
    """Resolve multimodal feature transport before tokenizer workers start.

    CUDA IPC is opt-in because its fixed pool on ``base_gpu_id`` reduces the
    memory left for model/KV-cache allocations. Multi-node MNNVL deployments
    may still auto-select CUDA VMM. The legacy CUDA IPC flag and environment
    variable remain supported so existing deployments map to this policy.
    """

    cfg = resolving_view(server_args)
    requested_transport = cfg.mm_feature_transport
    legacy_ipc_is_set = envs.SGLANG_USE_CUDA_IPC_TRANSPORT.is_set()
    legacy_ipc_enabled = envs.SGLANG_USE_CUDA_IPC_TRANSPORT.get()

    if cfg.keep_mm_feature_on_device:
        if requested_transport not in (None, "cuda_ipc"):
            raise ValueError(
                "--keep-mm-feature-on-device conflicts with "
                f"--mm-feature-transport={requested_transport}. Use only "
                "--mm-feature-transport=cuda_ipc."
            )
        requested_transport = "cuda_ipc"
        logger.warning(
            "--keep-mm-feature-on-device is deprecated; using "
            "--mm-feature-transport=cuda_ipc instead."
        )

    if requested_transport is None:
        if legacy_ipc_is_set:
            requested_transport = "cuda_ipc" if legacy_ipc_enabled else "cpu"
            logger.warning(
                "SGLANG_USE_CUDA_IPC_TRANSPORT is deprecated; use "
                "--mm-feature-transport=%s instead.",
                requested_transport,
            )
        elif cfg.encoder_only:
            requested_transport = "cpu"
            logger.info(
                "Multimodal feature transport auto-resolved to cpu for "
                "encoder-only serving; encoder outputs use "
                "--encoder-transfer-backend instead."
            )
        elif (
            model_config_of(server_args).is_multimodal
            and get_platform().is_cuda
            and cfg.disaggregation_mode == "null"
        ):
            # A full GPU pool always degrades to CPU transport per tensor.
            # Keep CUDA IPC opt-in because even an idle pool consumes HBM
            # that would otherwise back the KV cache. Multi-node
            # auto-selection is limited to GB200/GB300 systems where the
            # runtime already enables the MNNVL/IMEX communication stack.
            if cfg.nnodes == 1:
                requested_transport = "cpu"
            elif is_mnnvl_fabric_device() and os.path.exists(
                "/dev/nvidia-caps-imex-channels/channel0"
            ):
                from sglang.srt.model_loader.utils import (
                    supports_cuda_vmm_feature_transport,
                )

                if supports_cuda_vmm_feature_transport(model_config_of(server_args)):
                    requested_transport = "cuda_vmm"
                    logger.info(
                        "Multimodal feature transport auto-resolved to "
                        "cuda_vmm (multi-node GB200/GB300 MNNVL). Pass "
                        "--mm-feature-transport=cpu to opt out."
                    )
                else:
                    requested_transport = "cpu"
                    logger.info(
                        "Multimodal feature transport auto-resolved to cpu: "
                        "the model has not opted into CUDA VMM transport."
                    )
            else:
                requested_transport = "cpu"
                if is_mnnvl_fabric_device():
                    logger.info(
                        "Multimodal feature transport auto-resolved to cpu: "
                        "GB200/GB300 was detected but no IMEX channel is "
                        "mounted. Configure the MNNVL compute domain or pass "
                        "--mm-feature-transport=cuda_vmm after doing so."
                    )
        else:
            requested_transport = "cpu"
    elif legacy_ipc_is_set and legacy_ipc_enabled != (
        requested_transport == "cuda_ipc"
    ):
        logger.warning(
            "--mm-feature-transport=%s overrides the conflicting legacy "
            "SGLANG_USE_CUDA_IPC_TRANSPORT=%s setting.",
            requested_transport,
            int(legacy_ipc_enabled),
        )

    if cfg.encoder_only and requested_transport in ("cuda_ipc", "cuda_vmm"):
        logger.warning(
            "--mm-feature-transport=%s does not control encoder-only "
            "output transfer; using cpu for this inactive transport. Select "
            "--encoder-transfer-backend for encoder outputs.",
            requested_transport,
        )
        requested_transport = "cpu"

    if requested_transport == "cuda_vmm":
        if not get_platform().is_cuda:
            raise ValueError("--mm-feature-transport=cuda_vmm requires NVIDIA CUDA.")
        if cfg.pp_size != 1:
            raise ValueError(
                "--mm-feature-transport=cuda_vmm does not support pipeline parallelism."
            )
        if envs.SGLANG_RUST_SERVER.get():
            raise ValueError(
                "--mm-feature-transport=cuda_vmm is not supported with "
                "SGLANG_RUST_SERVER."
            )
        pool_budget_mb = envs.SGLANG_MM_FEATURE_CACHE_MB.get()
        handle_kind = "CUDA FABRIC" if cfg.nnodes > 1 else "POSIX FD"
        logger.info(
            "Using CUDA VMM for multimodal features with %s sharing: "
            "reserving up to %d MiB on base GPU %d across %d tokenizer "
            "worker(s). This reduces KV cache headroom; a full pool falls "
            "back to inline CPU transport.",
            handle_kind,
            pool_budget_mb,
            cfg.base_gpu_id,
            cfg.tokenizer_worker_num,
        )

    if requested_transport == "cuda_ipc":
        if not get_platform().is_cuda:
            raise ValueError("--mm-feature-transport=cuda_ipc requires NVIDIA CUDA.")
        if cfg.nnodes != 1:
            raise ValueError(
                "--mm-feature-transport=cuda_ipc only supports a single node."
            )

        pool_budget_mb = envs.SGLANG_MM_FEATURE_CACHE_MB.get()
        logger.info(
            "Using CUDA IPC for multimodal features: reserving up to %d MiB "
            "on base GPU %d across %d tokenizer worker(s). This reduces KV "
            "cache headroom; a full pool falls back to CPU transport.",
            pool_budget_mb,
            cfg.base_gpu_id,
            cfg.tokenizer_worker_num,
        )
        logger.info(
            "CUDA IPC pool-handle caching is %s. It reuses mappings to the "
            "existing bounded pool without reserving another pool; set "
            "SGLANG_USE_IPC_POOL_HANDLE_CACHE=0 to disable it.",
            ("enabled" if envs.SGLANG_USE_IPC_POOL_HANDLE_CACHE.get() else "disabled"),
        )

    declare_resolution(
        server_args,
        "_handle_multimodal_feature_transport",
        mm_feature_transport=requested_transport,
    )
    # The bounded IPC pool owns device residency. Do not retain unpooled
    # tensors after a pool miss, which would make HBM use request-dependent.
    declare_resolution(
        server_args,
        "_handle_multimodal_feature_transport",
        keep_mm_feature_on_device=False,
    )
    envs.SGLANG_USE_CUDA_IPC_TRANSPORT.set(
        "1" if requested_transport == "cuda_ipc" else "0"
    )


_ssl_verify_warned = False


def ssl_verify_of(cfg: Any):
    """What to pass as the requests library's ``verify=``.

    A CA file means validate against it. SSL configured without one means
    verification off -- self-signed certificates in development -- and that is
    worth saying out loud, once. No SSL means the system CA bundle.

    The warning is once per process: the message is about how this process was
    configured, and a second engine repeating it says nothing new.
    """
    global _ssl_verify_warned
    if cfg.ssl_ca_certs:
        return cfg.ssl_ca_certs
    if cfg.ssl_certfile:
        if not _ssl_verify_warned:
            logger.warning(
                "SSL is enabled but --ssl-ca-certs was not provided. Certificate "
                "verification is DISABLED for internal health checks. For "
                "production deployments, provide --ssl-ca-certs or use CA-signed "
                "certificates."
            )
            _ssl_verify_warned = True
        return False
    return True
