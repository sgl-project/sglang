# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for serving-surface and multimodal entry validation."""

from __future__ import annotations

import logging
import os
import socket
from typing import Any

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.utils.common import configure_media_url_security
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
                "--http2-max-concurrent-streams must be between 1 and " "4294967295."
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
