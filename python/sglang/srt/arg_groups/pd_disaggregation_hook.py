from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from sglang.srt.arg_groups.arg_utils import record_fields
from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    model_config_of,
    resolved_view,
    resolving_view,
)
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def handle_pd_disaggregation(server_args: ServerArgs) -> None:
    """Validate and normalize PD-disaggregation server args."""
    cfg = resolving_view(server_args)

    from sglang.srt.disaggregation.compression.protocol import validate_mode

    compression = validate_mode(envs.SGLANG_PD_KV_COMPRESSION.get())
    host_compression = validate_mode(envs.SGLANG_HICACHE_KV_COMPRESSION.get())
    if (
        compression != "off"
        or host_compression != "off"
        or envs.SGLANG_PD_KV_COMPRESSION_FORCE.get()
    ):
        _validate_pd_compression(server_args)
    if compression != "off":
        envs.SGLANG_DISAGG_STAGING_BUFFER.set(True)
        envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE.set(True)
        envs.SGLANG_DISAGGREGATION_QUEUE_SIZE.set(1)
        envs.SGLANG_DISAGGREGATION_THREAD_POOL_SIZE.set(1)
        if "SGLANG_DISAGG_STAGING_POOL_SIZE_MB" not in os.environ:
            envs.SGLANG_DISAGG_STAGING_POOL_SIZE_MB.set(512)

    # "mooncake_tcp" is mooncake with the TCP transport forced: set MC_FORCE_TCP
    # so mooncake installs TcpTransport instead of RDMA, rewrite the backend to
    # mooncake, and skip RDMA HCA selection. Must run before backend-name checks.
    if cfg.disaggregation_transfer_backend == "mooncake_tcp":
        os.environ.setdefault("MC_FORCE_TCP", "1")
        declare_resolution(
            server_args,
            "handle_pd_disaggregation",
            disaggregation_transfer_backend="mooncake",
        )
        declare_resolution(
            server_args,
            "handle_pd_disaggregation",
            disaggregation_ib_device=None,
        )
        logger.info(
            "disaggregation transfer backend 'mooncake_tcp' -> mooncake "
            "with MC_FORCE_TCP=1 (TCP transport, no RDMA)"
        )

    if cfg.disaggregation_mode == "prefill" and cfg.dcp_size > 1:
        logger.warning(
            "DCP on a PD prefill server is supported when prefill and decode "
            "use the same DCP layout, but it usually adds communication "
            "overhead without improving prefill performance."
        )

    if cfg.disaggregation_mode == "decode" and cfg.dcp_size > 1:
        # Fake transfer moves no KV and is only used for synthetic decode
        # benchmarks, so it does not need the DCP relayout from Mooncake/NIXL.
        if cfg.disaggregation_transfer_backend not in (
            "mooncake",
            "nixl",
            "fake",
        ):
            raise ValueError(
                "PD decode DCP requires --disaggregation-transfer-backend "
                "mooncake, nixl, or fake for synthetic benchmarking, got "
                f"{cfg.disaggregation_transfer_backend!r}."
            )
        if cfg.disaggregation_decode_enable_radix_cache:
            raise ValueError(
                "PD decode DCP currently requires chunk cache; "
                "--disaggregation-decode-enable-radix-cache is not supported."
            )
        if cfg.enable_hierarchical_cache:
            raise ValueError(
                "PD decode DCP currently requires chunk cache; "
                "--enable-hierarchical-cache is not supported."
            )

    if cfg.disaggregation_mode == "decode":
        if cfg.disaggregation_decode_enable_radix_cache:
            if cfg.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if cfg.disaggregation_transfer_backend == "fake":
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --disaggregation-transfer-backend fake"
                )
            if cfg.speculative_algorithm is not None:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with speculative decoding "
                    f"(--speculative-algorithm {cfg.speculative_algorithm})"
                )

            if resolved_view(server_args).enable_dp_attention:
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disable_radix_cache=False,
            )
            logger.warning("EXPERIMENTAL: Radix cache is enabled for decode server")
        else:
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disable_radix_cache=True,
            )
            logger.warning("KV cache is forced as chunk cache for decode server")

        # Default the number of *extra* decode req_to_token slots reserved for
        # in-transfer (being-received-from-prefill) requests, on top of the
        # max_running_requests-derived pool. Large batches get none; small
        # per-worker batches reserve 2x the batch as cheap overlap headroom.
        if cfg.disaggregation_decode_extra_slots is None:
            extra_slots = 0
            if cfg.max_running_requests is not None:
                per_worker = cfg.max_running_requests // max(1, cfg.dp_size)
                if per_worker <= 32:
                    extra_slots = per_worker * 2
            declare_resolution(
                server_args,
                "handle_pd_disaggregation",
                disaggregation_decode_extra_slots=extra_slots,
            )

    elif cfg.disaggregation_mode == "prefill":
        assert cfg.disaggregation_transfer_backend != "fake", (
            "Prefill server does not support 'fake' as the transfer backend"
        )

        if envs.SGLANG_RUST_SERVER.get():
            _alias_bootstrap_port_to_api_port(server_args)

    if cfg.disaggregation_mode in ("prefill", "decode"):
        if (
            envs.SGLANG_DISAGG_STAGING_BUFFER.get()
            and cfg.disaggregation_transfer_backend not in ("mooncake", "nixl")
        ):
            raise ValueError(
                f"SGLANG_DISAGG_STAGING_BUFFER requires "
                f"disaggregation_transfer_backend='mooncake' or 'nixl', "
                f"got '{cfg.disaggregation_transfer_backend}'."
            )


def _alias_bootstrap_port_to_api_port(server_args: ServerArgs) -> None:
    """Rust-server prefill serves the KV bootstrap registry on the api listener
    itself, so the resolved bootstrap port must BE the api port — every internal
    consumer (KVManager registration, PrefillBootstrapQueue) reads the resolved
    field and agrees automatically. Decode is untouched: there the field names
    the PREFILL side's bootstrap port and must stay as the operator set it.
    """
    cfg = resolving_view(server_args)
    default_port = next(
        f.default
        for f in record_fields(type(server_args))
        if f.name == "disaggregation_bootstrap_port"
    )
    if cfg.disaggregation_bootstrap_port not in (
        default_port,
        cfg.port,
    ):
        raise ValueError(
            "SGLANG_RUST_SERVER serves the PD KV bootstrap registry on the api "
            "port itself; --disaggregation-bootstrap-port "
            f"{cfg.disaggregation_bootstrap_port} conflicts with --port "
            f"{cfg.port}. Drop --disaggregation-bootstrap-port (decode "
            "nodes and the PD router must then target the prefill api port)."
        )
    if cfg.disaggregation_bootstrap_port != cfg.port:
        logger.info(
            "SGLANG_RUST_SERVER: KV bootstrap registry is served on the api "
            "port; disaggregation_bootstrap_port %d -> %d",
            cfg.disaggregation_bootstrap_port,
            cfg.port,
        )
        declare_resolution(
            server_args,
            "_alias_bootstrap_port_to_api_port",
            disaggregation_bootstrap_port=cfg.port,
        )


def handle_encoder_disaggregation(server_args: Any):
    from sglang.srt.arg_groups.model_hook import handle_language_model_only
    from sglang.srt.arg_groups.validation_hook import validate_ib_devices
    from sglang.srt.server_args import resolve_encoder_transfer_backend

    cfg = resolving_view(server_args)
    handle_language_model_only(server_args)
    if cfg.enable_prefix_mm_cache and not cfg.encoder_only:
        raise ValueError(
            "--enable-prefix-mm-cache requires --encoder-only to be enabled"
        )
    if cfg.encoder_only and cfg.language_only:
        raise ValueError("Cannot set --encoder-only and --language-only together")
    if cfg.encoder_only and not cfg.disaggregation_mode == "null":
        raise ValueError(
            "Cannot set --encoder-only and --disaggregation-mode prefill/decode together"
        )

    if cfg.language_only and len(cfg.encoder_urls) == 0:
        logger.info(
            "--language-only is set without --encoder-urls. Encoders are "
            "expected to register dynamically via the "
            "EncoderBootstrapServer."
        )

    # Validate IB devices when mooncake backend is used
    if (
        cfg.disaggregation_transfer_backend == "mooncake"
        and cfg.disaggregation_mode in ("prefill", "decode")
    ) or cfg.encoder_transfer_backend == "mooncake":
        declare_resolution(
            server_args,
            "_handle_encoder_disaggregation",
            disaggregation_ib_device=validate_ib_devices(cfg.disaggregation_ib_device),
        )

    # Validate model type for encoder disaggregation
    hf_config = model_config_of(server_args).hf_config
    model_arch = hf_config.architectures[0]
    if cfg.encoder_transfer_backend == "auto":
        declare_resolution(
            server_args,
            "_handle_encoder_disaggregation",
            encoder_transfer_backend=resolve_encoder_transfer_backend(
                cfg.encoder_transfer_backend, model_arch, cfg.tp_size
            ),
        )
        if cfg.encoder_only or cfg.language_only:
            logger.info(
                "Encoder transfer backend auto-resolved to %s for %s at TP%d.",
                cfg.encoder_transfer_backend,
                model_arch,
                cfg.tp_size,
            )
    if (cfg.encoder_only or cfg.language_only) and model_arch not in [
        "Qwen2VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "InternS2PreviewForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
        "Qwen2AudioForConditionalGeneration",
        "Qwen2_5OmniForConditionalGeneration",
        "Dots3NoteForCausalLM",
        "KimiVLForConditionalGeneration",
        "KimiK25ForConditionalGeneration",
        "KimiK3ForConditionalGeneration",
        "MiMoV2ForCausalLM",
        "Glm5NextForConditionalGeneration",
    ]:
        raise ValueError(
            f"Model type {model_arch} is not supported for encoder disaggregation. "
            f"Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, "
            f"Qwen2Audio, Qwen2.5Omni, Dots3-Note, Kimi, MiMoV2, GLM5Next."
        )


def _validate_pd_compression(server_args: ServerArgs) -> None:
    """Fail closed on combinations outside the prototype's validation scope."""
    cfg = resolving_view(server_args)
    errors = []
    if cfg.disaggregation_mode not in ("prefill", "decode"):
        errors.append("P/D mode is required")
    if (
        cfg.disaggregation_transfer_backend != "mooncake"
        or os.getenv("MC_FORCE_TCP") == "1"
    ):
        errors.append("Mooncake RDMA is required")
    if (
        any(
            getattr(cfg, key, 1) != 1
            for key in ("tp_size", "pp_size", "dp_size", "attn_cp_size", "dcp_size")
        )
        or cfg.enable_prefill_cp
    ):
        errors.append("TP/PP/DP/CP/DCP must all be 1")
    if cfg.page_size != 1 or cfg.attention_backend != "flashinfer":
        errors.append("--page-size 1 --attention-backend flashinfer are required")
    host_compression = envs.SGLANG_HICACHE_KV_COMPRESSION.get()
    prefill_cache = (
        cfg.disaggregation_mode == "prefill" and cfg.enable_hierarchical_cache
    )
    if prefill_cache and cfg.disable_radix_cache:
        errors.append("Prefill HiCache requires radix cache enabled")
    if (
        (not cfg.disable_radix_cache and not prefill_cache)
        or (cfg.enable_hierarchical_cache and cfg.disaggregation_mode != "prefill")
        or cfg.enable_hisparse
        or cfg.disaggregation_decode_enable_radix_cache
        or cfg.disaggregation_decode_enable_offload_kvcache
    ):
        errors.append(
            "only Prefill HiCache is supported; Decode radix/HiCache, HiSparse and incremental offload must be disabled"
        )
    if host_compression != "off":
        if (
            getattr(cfg, "radix_cache_backend", None) is not None
            or getattr(cfg, "enable_lmcache", False)
            or getattr(cfg, "enable_flexkv", False)
            or getattr(cfg, "enable_unified_cache_external_linker", False)
            or getattr(cfg, "enable_session_radix_cache", False)
            or getattr(cfg, "enable_streaming_session", False)
        ):
            errors.append(
                "compressed L2 requires the built-in cache without external linker or sessions"
            )
        if envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get():
            errors.append("compressed L2 cannot use the C++ radix tree")
        if not prefill_cache or cfg.disable_radix_cache:
            errors.append(
                "compressed L2 requires Prefill HiCache with radix cache enabled"
            )
        if cfg.hicache_size <= 0 or cfg.hicache_host_memory_mode != "cache":
            errors.append(
                "compressed L2 requires an explicit positive --hicache-size and cache mode"
            )
        if (
            cfg.hicache_write_policy != "write_through"
            or cfg.hicache_storage_backend is not None
        ):
            errors.append("compressed L2 requires write_through and no L3 backend")
        if envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.get() != "python":
            errors.append("compressed L2 requires the Python unified TreeCore")
    if (
        cfg.disaggregation_mode == "decode"
        and cfg.disaggregation_decode_retraction_backup != "cpu_tensor"
    ):
        errors.append(
            "Decode requires --disaggregation-decode-retraction-backup cpu_tensor"
        )
    if envs.SGLANG_PD_KV_COMPRESSION_FORCE.get():
        if (
            envs.SGLANG_PD_KV_COMPRESSION.get() != "lz4"
            or (
                cfg.enable_hierarchical_cache
                and (not prefill_cache or host_compression != "lz4")
            )
            or (not cfg.enable_hierarchical_cache and host_compression != "off")
        ):
            errors.append(
                "FORCE is test-only: LZ4 P/D; optional Prefill LZ4 L2; Decode L2 off"
            )
        if not envs.SGLANG_PD_KV_COMPRESSION_VERIFY.get():
            errors.append("FORCE requires VERIFY=1")
    if envs.SGLANG_KV_COMPRESSION_WORKSPACE_MB.get() <= 0:
        errors.append("compression workspace budget must be positive")
    if getattr(cfg, "enable_lora", False):
        errors.append("LoRA is outside this draft's scope")
    if not cfg.disable_overlap_schedule or not cfg.disable_cuda_graph:
        errors.append("--disable-overlap-schedule --disable-cuda-graph are required")
    if cfg.speculative_algorithm is not None:
        errors.append("speculative decoding is unsupported")
    if not cfg.chunked_prefill_size or cfg.chunked_prefill_size <= 0:
        errors.append("a positive chunked-prefill-size is required")
    if envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get():
        errors.append("custom Mooncake memory pools are outside the prototype scope")
    if envs.SGLANG_RUST_SERVER.get():
        errors.append("the Python worker/bootstrap server is required")
    if model_config_of(server_args).hf_config.architectures != ["Qwen3ForCausalLM"]:
        errors.append("only Qwen3ForCausalLM is in the prototype scope")
    if errors:
        raise ValueError("P/D compression prototype: " + "; ".join(errors))
