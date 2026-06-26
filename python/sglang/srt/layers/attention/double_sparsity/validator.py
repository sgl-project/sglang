"""Server-args validator for standalone Double Sparsity.

Mirrors the role of ``arg_groups.hisparse_hook.validate_hisparse`` but lives
inside the Double Sparsity package per the upstream-shaped path budget
(``arg_groups/`` is intentionally out of scope for this feature).

Enforces, at server startup:

* mutual-exclusion with ``--enable-hisparse``,
* rejection of ``--disaggregation-mode`` (HiSparse owns the PD path),
* presence of ``--double-sparsity-config`` JSON with at least
  ``channel_mask_path``,
* JSON page-size pairing with ``--page-size``,
* backend / KV-dtype pairing: ``fp8_e4m3 ↔ flashmla_kv``,
  ``bfloat16 ↔ flashmla_sparse``,
* unsupported page-size rejection (page must be in ``{32, 64, 128}``),
* capability check: the running model must expose the ``nsa.Indexer``
  hook surface — proxied via :func:`is_deepseek_dsa` so GLM-5.1 falls in for
  free once it ships the same indexer interface,
* channel-mask file existence + content-hash verification (delegates to
  :func:`channel_mask.load_channel_mask`),
* radix cache is supported with Double Sparsity (no extra gate).
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


def validate_double_sparsity(server_args: ServerArgs) -> None:
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

    if getattr(server_args, "enable_hierarchical_cache", False):
        raise ValueError(
            "Standalone Double Sparsity does not support --enable-hierarchical-cache. "
            "The hierarchical cache builds a per-token DSA indexer host sidecar from the "
            "device KV pool, but Double Sparsity replaces the DSA indexer with "
            "query-signature selection and does not allocate that sidecar, so its host "
            "offload and cached-prefix label semantics are undefined. Drop "
            "--enable-hierarchical-cache (Double Sparsity targets the in-GPU radix cache)."
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

    # Production-path selector-variant safety (startup gate). This
    # ds_scorer_is_graph_safe() guard covers any non-graph-safe variant: raw-dot
    # "off", cosine (the in-kernel per-head division + resident key-norm cache, on
    # both the FP8 and BF16 KV paths), head_agg (mean), and anchor_mode
    # (recency/global/strided) are all graph-safe so it does not fire for them; it
    # remains the single startup gate so a future non-graph-safe variant can re-enable it.
    from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
        ds_scorer_is_graph_safe,
    )

    if not ds_scorer_is_graph_safe(config) and not getattr(
        server_args, "disable_cuda_graph", False
    ):
        raise ValueError(
            "Double Sparsity selector variant is not graph-safe under CUDA graph "
            "capture. Re-run with --disable-cuda-graph, or use the default selector."
        )

    # include_current_slot (the current-decode-slot force-include) is defined for
    # single-token decode only — it force-includes exactly logical seq_len-1 per
    # row. Speculative / MTP / multi-token-accept decode advances several tokens
    # per step, so the single seq_len-1 rule would force-include the wrong slot.
    # Fail closed at startup rather than silently mis-select.
    if getattr(config, "include_current_slot", False) and (
        getattr(server_args, "speculative_algorithm", None) is not None
    ):
        raise ValueError(
            "Double Sparsity 'include_current_slot' (the current-decode-slot "
            "force-include) is single-token-decode only and is not supported with "
            "speculative/MTP decoding (speculative_algorithm="
            f"{getattr(server_args, 'speculative_algorithm', None)!r}); a "
            "multi-token-accept step would force-include the wrong slot. Disable "
            "speculative decoding or set 'include_current_slot' to false."
        )

    # rope_aware_score (the long-context RoPE recovery) is validated ONLY for the
    # loop14/15 config: single-token MLA decode on CUDA, fp8 KV, CUDA graphs ON, no
    # speculative/MTP/DCP. The selection-site guard in
    # deepseek_v2._select_topk_indices is the in-loop backstop, but CPU/NPU and
    # graphs-off decode can bypass that selector entirely, so this startup gate is the
    # authoritative fail-closed: every non-validated runtime is rejected before serving
    # rather than silently scoring no-PE (or silently ignoring the flag).
    if getattr(config, "rope_aware_score", False):
        device = getattr(server_args, "device", None)
        if device not in (None, "cuda"):
            raise ValueError(
                "Double Sparsity 'rope_aware_score' is validated only on the CUDA MLA "
                f"path; got --device={device!r}. Set rope_aware_score=false."
            )
        if getattr(server_args, "disable_cuda_graph", False):
            raise ValueError(
                "Double Sparsity 'rope_aware_score' is validated only with CUDA graphs "
                "ON (the loop14/15 config); it is not supported with "
                "--disable-cuda-graph. Set rope_aware_score=false."
            )
        if getattr(server_args, "speculative_algorithm", None) is not None:
            raise ValueError(
                "Double Sparsity 'rope_aware_score' is not supported with speculative/"
                "MTP decoding (speculative_algorithm="
                f"{getattr(server_args, 'speculative_algorithm', None)!r}); the rope "
                "query is validated for single-token decode only. Set "
                "rope_aware_score=false."
            )
        if int(getattr(server_args, "dcp_size", 1) or 1) > 1:
            raise ValueError(
                "Double Sparsity 'rope_aware_score' is not supported with decode "
                f"context parallel (--dcp-size={getattr(server_args, 'dcp_size', None)!r}); "
                "not validated. Set rope_aware_score=false."
            )
        # fp8 KV requirement. '--kv-cache-dtype auto' resolves to fp8 only on SM>=10
        # (bf16 on H100/H200), so an explicit non-fp8 dtype is rejected here and the
        # resident-latent dtype is re-checked at the selection site (the auto->bf16
        # backstop in _select_topk_indices).
        rope_kv_dtype = getattr(server_args, "kv_cache_dtype", None)
        if rope_kv_dtype not in (None, "auto") and "fp8" not in str(rope_kv_dtype):
            raise ValueError(
                "Double Sparsity 'rope_aware_score' is validated only for fp8 KV "
                f"cache; got --kv-cache-dtype={rope_kv_dtype!r}. Pin "
                "--kv-cache-dtype fp8_e4m3 or set rope_aware_score=false."
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
            ("dsa_prefill_backend", "prefill"),
            ("dsa_decode_backend", "decode"),
        ):
            backend = getattr(server_args, attr, None)
            if backend is not None and backend not in allowed:
                raise ValueError(
                    f"Double Sparsity with --kv-cache-dtype={kv_cache_dtype} requires "
                    f"--dsa-{label}-backend in {sorted(allowed)}, but got {backend!r}."
                )

    # Capability check: the running model must expose the DSA Indexer
    # hook surface that Double Sparsity replaces.
    hf_config = None
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
            from sglang.srt.configs.model_config import is_deepseek_dsa

            if not is_deepseek_dsa(hf_config):
                raise ValueError(
                    "Double Sparsity currently requires a model that exposes the DSA "
                    "Indexer hook surface (e.g. DeepSeek V3.2). The capability check "
                    "via is_deepseek_dsa(hf_config) returned False for this model."
                )

    # top_k vs get_dsa_index_topk(hf_config) boot assert.
    # DS must pick the same number of tokens as the DSA lightning indexer so the
    # apples-to-apples comparison holds (top_k == index_topk). A mismatch stays an
    # explicit ablation gated by SGLANG_DS_ALLOW_TOPK_MISMATCH=1.
    if hf_config is not None:
        try:
            from sglang.srt.configs.model_config import get_dsa_index_topk

            model_topk = get_dsa_index_topk(hf_config)
            if model_topk > 0:
                if config.top_k != model_topk:
                    if os.environ.get("SGLANG_DS_ALLOW_TOPK_MISMATCH") != "1":
                        raise ValueError(
                            f"Double Sparsity top_k={config.top_k} does not match the model's "
                            f"DSA index_topk={model_topk}. The two must agree for the "
                            "apples-to-apples comparison. Either set "
                            f"top_k={model_topk} in --double-sparsity-config or set "
                            "SGLANG_DS_ALLOW_TOPK_MISMATCH=1 to override."
                        )
                    else:
                        logger.warning(
                            "Double Sparsity top_k=%d != model DSA index_topk=%d; "
                            "SGLANG_DS_ALLOW_TOPK_MISMATCH=1 override active.",
                            config.top_k,
                            model_topk,
                        )
        except ImportError:
            logger.debug("get_dsa_index_topk not available; skipping top_k assertion.")

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

    setattr(server_args, "_double_sparsity_parsed_config", config)
    setattr(server_args, "_double_sparsity_channel_mask", mask)

    # The mask passed schema + content-hash + runtime-pairing checks. Mark
    # the readiness gauge so /metrics reflects a healthy DS startup.
    # The metrics helper is best-effort (silent when prometheus_client is
    # missing); no error handling needed at the call site.
    from sglang.srt.layers.attention.double_sparsity import metrics as _ds_metrics

    _ds_metrics.mark_channel_mask_valid(True)
