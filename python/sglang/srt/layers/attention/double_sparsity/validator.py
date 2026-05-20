"""Server-args validator for standalone Double Sparsity.

Mirrors the role of ``arg_groups.hisparse_hook.validate_hisparse`` but lives
inside the Double Sparsity package per the upstream-shaped path budget
(``arg_groups/`` is intentionally out of scope for this feature).

Enforces, at server startup:

* mutual-exclusion with ``--enable-hisparse`` (DEC-8),
* rejection of ``--disaggregation-mode`` (HiSparse owns the PD path),
* presence of ``--double-sparsity-config`` JSON with at least
  ``channel_mask_path``,
* JSON page-size pairing with ``--page-size``,
* backend / KV-dtype pairing: ``fp8_e4m3 ↔ flashmla_kv``,
  ``bfloat16 ↔ flashmla_sparse``,
* unsupported page-size rejection (page must be in ``{32, 64, 128}``),
* capability check (DEC-10): the running model must expose the ``nsa.Indexer``
  hook surface — proxied via :func:`is_deepseek_nsa` so GLM-5.1 falls in for
  free once it ships the same indexer interface,
* channel-mask file existence + content-hash verification (delegates to
  :func:`channel_mask.load_channel_mask`),
* DEC-2 radix-cache permission: refuse to serve with radix cache enabled
  until the M3-B page-stability fixture has been recorded as passing
  (``server_args._double_sparsity_radix_fixture_passed = True``), unless the
  developer override ``SGLANG_DS_RADIX_OVERRIDE=1`` is set.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


_SUPPORTED_PAGE_SIZES = (32, 64, 128)

_BACKEND_BY_DTYPE = {
    "bfloat16": {"flashmla_sparse"},
    "fp8_e4m3": {"flashmla_kv"},
}


def validate_double_sparsity(server_args: "ServerArgs") -> None:
    """Validate ``--enable-double-sparsity`` constraints.

    Called from ``ServerArgs.check_server_args`` after ``validate_hisparse``.
    The early-return semantics keep this safe to call unconditionally.
    """

    if not getattr(server_args, "enable_double_sparsity", False):
        return

    if getattr(server_args, "enable_hisparse", False):
        raise ValueError(
            "Double Sparsity and HiSparse are mutually exclusive; there are no plans "
            "to integrate them. Pick exactly one of --enable-double-sparsity or "
            "--enable-hisparse."
        )

    if getattr(server_args, "disaggregation_mode", None) not in (None, "null"):
        raise ValueError(
            "Standalone Double Sparsity does not support --disaggregation-mode. "
            f"Got --disaggregation-mode={server_args.disaggregation_mode!r}. "
            "Drop the disaggregation flag or use HiSparse instead "
            "(HiSparse is the PD-disaggregated sparsity path)."
        )

    payload = getattr(server_args, "double_sparsity_config", None)
    if payload is None or (isinstance(payload, str) and not payload.strip()):
        raise ValueError(
            "--enable-double-sparsity requires --double-sparsity-config to be set "
            "with at least 'channel_mask_path'. Example: --double-sparsity-config "
            '\'{"top_k": 2048, "page_size": 64, '
            '"channel_mask_path": "/path/to/channel_mask.safetensors", '
            '"device_buffer_size": 4096}\'.'
        )

    from sglang.srt.layers.attention.double_sparsity.config import (
        parse_double_sparsity_config,
    )

    config = parse_double_sparsity_config(payload)

    if not config.channel_mask_path:
        raise ValueError(
            "Double Sparsity requires 'channel_mask_path' in --double-sparsity-config."
        )

    # Page size pairing (config vs server) + supported set.
    server_page_size = getattr(server_args, "page_size", None)
    if config.page_size not in _SUPPORTED_PAGE_SIZES:
        raise ValueError(
            f"Double Sparsity page_size={config.page_size} is not supported. "
            f"Supported values: {_SUPPORTED_PAGE_SIZES}."
        )
    if server_page_size is not None and config.page_size != server_page_size:
        raise ValueError(
            f"Double Sparsity config page_size={config.page_size} does not match "
            f"--page-size={server_page_size}. The two must agree at startup."
        )

    # Backend / KV dtype pairing.
    kv_cache_dtype = getattr(server_args, "kv_cache_dtype", None)
    if kv_cache_dtype in ("auto", None):
        # The model's default dtype is resolved later; let the runtime decide.
        pass
    elif kv_cache_dtype not in _BACKEND_BY_DTYPE:
        raise ValueError(
            f"Double Sparsity requires --kv-cache-dtype in {sorted(_BACKEND_BY_DTYPE)}, "
            f"got {kv_cache_dtype!r}."
        )
    else:
        allowed = _BACKEND_BY_DTYPE[kv_cache_dtype]
        for attr, label in (
            ("nsa_prefill_backend", "prefill"),
            ("nsa_decode_backend", "decode"),
        ):
            backend = getattr(server_args, attr, None)
            if backend is not None and backend not in allowed:
                raise ValueError(
                    f"Double Sparsity with --kv-cache-dtype={kv_cache_dtype} requires "
                    f"--nsa-{label}-backend in {sorted(allowed)}, but got {backend!r}."
                )

    # Capability check (DEC-10): the running model must expose the NSA Indexer
    # hook surface that Double Sparsity replaces. is_deepseek_nsa() recognises
    # DeepSeek V3.2 today; GLM-5.1 should also pass once it ships the same
    # interface (DEC-6 deferred-but-hard requirement).
    if callable(getattr(server_args, "get_model_config", None)):
        try:
            hf_config = server_args.get_model_config().hf_config  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Double Sparsity capability check could not resolve hf_config: %s",
                exc,
            )
            hf_config = None
        if hf_config is not None:
            from sglang.srt.configs.model_config import is_deepseek_nsa

            if not is_deepseek_nsa(hf_config):
                raise ValueError(
                    "Double Sparsity currently requires a model that exposes the NSA "
                    "Indexer hook surface (e.g. DeepSeek V3.2). The capability check "
                    "via is_deepseek_nsa(hf_config) returned False for this model. "
                    "Note: this generalizes to GLM-5.1 once it ships the same "
                    "indexer interface (DEC-6 / DEC-10)."
                )

    # Channel-mask file existence + content-hash verification. The loader
    # raises DoubleSparsityChannelMaskMissing for absent files and
    # DoubleSparsityChannelMaskCorrupt for hash mismatch, schema drift,
    # dtype mismatch, NaN / +/-Inf / all-zero per-row weights. Both are
    # typed so operators get a stable error class at boot.
    from sglang.srt.layers.attention.double_sparsity.channel_mask import (
        load_channel_mask,
    )

    mask = load_channel_mask(config.channel_mask_path)

    # Cross-check mask vs runtime config (dtype / page_size / label_dim). The
    # head_dim check is best-effort: the running attention layer is the
    # authoritative head_dim, but it's not yet constructed here, so we
    # accept the mask's recorded head_dim if it is internally consistent.
    if kv_cache_dtype not in (None, "auto") and mask.dtype != kv_cache_dtype:
        raise ValueError(
            f"channel mask dtype={mask.dtype!r} does not match "
            f"--kv-cache-dtype={kv_cache_dtype!r}."
        )
    if server_page_size is not None and mask.page_size != server_page_size:
        raise ValueError(
            f"channel mask page_size={mask.page_size} does not match "
            f"--page-size={server_page_size}."
        )

    # DEC-2 radix-cache permission. Default: refuse until the M3-B page-
    # stability fixture has been recorded as passing for this configuration.
    # Developers can override with SGLANG_DS_RADIX_OVERRIDE=1.
    if not getattr(server_args, "disable_radix_cache", True):
        fixture_passed = bool(
            getattr(server_args, "_double_sparsity_radix_fixture_passed", False)
        )
        if not fixture_passed and os.environ.get(
            "SGLANG_DS_RADIX_OVERRIDE"
        ) != "1":
            raise ValueError(
                "Double Sparsity requires --disable-radix-cache until the "
                "M3-B page-stability fixture has been recorded as passing "
                "(server_args._double_sparsity_radix_fixture_passed = True). "
                "Set SGLANG_DS_RADIX_OVERRIDE=1 to override during development."
            )

    setattr(server_args, "_double_sparsity_parsed_config", config)
    setattr(server_args, "_double_sparsity_channel_mask", mask)

    # The mask passed schema + content-hash + runtime-pairing checks. Mark
    # the readiness gauge so /metrics reflects a healthy DS startup.
    # The metrics helper is best-effort (silent when prometheus_client is
    # missing); no error handling needed at the call site.
    from sglang.srt.layers.attention.double_sparsity import metrics as _ds_metrics

    _ds_metrics.mark_channel_mask_valid(True)
