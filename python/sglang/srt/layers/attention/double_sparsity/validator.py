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
  hook surface — proxied via :func:`is_deepseek_dsa` so GLM-5.1 falls in for
  free once it ships the same indexer interface,
* channel-mask file existence + content-hash verification (delegates to
  :func:`channel_mask.load_channel_mask`),
* DEC-2 radix-cache permission: refuse to serve with radix cache enabled
  until the M3-B page-stability fixture has been recorded as passing
  (``server_args._double_sparsity_radix_fixture_passed = True``), unless the
  developer override ``SGLANG_DS_RADIX_OVERRIDE=1`` is set.
"""

from __future__ import annotations

import hashlib
import json
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

    # Lifted-budget decode opt-in gate. The backend path is wired
    # (ds_lifted_budget_decode_available() == True): the selector widens to
    # lifted_budget_top_k and decode routes the selected slots through the
    # request-local compact remap -> dequantize_k_cache_paged_out ->
    # flash_mla_sparse_fwd path. As of the production-graph hardening this path is
    # CUDA-graph-safe (fixed-shape compact builder + alloc-free `out=` dequant +
    # preallocated DSGraphState scratch, proven zero-alloc under graph replay), so
    # it no longer requires --disable-cuda-graph. The only remaining gate here is
    # fail-closed if a future build ever ships the flag without a wired backend
    # (the top_k == index_topk / lifted_budget_top_k shape checks live in the
    # model-topk block below).
    from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
        ds_lifted_budget_decode_available,
    )

    if getattr(config, "enable_lifted_budget_decode", False):
        if not ds_lifted_budget_decode_available():
            raise ValueError(
                "Double Sparsity enable_lifted_budget_decode is recognized but the "
                "opt-in lifted-budget decode backend path is not implemented in this "
                "build, so the flag cannot be honored and the server fails closed "
                "rather than booting into a silent no-op or the flashmla_kv "
                "'indices.shape[-1] == dsa_index_topk' assert."
            )
        if getattr(server_args, "speculative_algorithm", None) is not None:
            # The lifted-budget graph scratch (DSGraphState.lifted_compact_kv etc.)
            # is sized by max_bs, but speculative target-verify expands the decode
            # rows to bs * num_draft_tokens, which would undersize/overflow the
            # scratch. The Loop-7 lifted op-point is non-speculative; fail closed
            # rather than risk a wrong-output / OOB scratch slice.
            raise ValueError(
                "Double Sparsity enable_lifted_budget_decode is not supported with "
                "speculative decoding (--speculative-algorithm="
                f"{server_args.speculative_algorithm!r}): the lifted-budget CUDA-graph "
                "scratch is sized by max_bs, but speculative target-verify expands the "
                "decode rows. Disable one of the two."
            )

    # Production-path selector-variant safety (future-proof guard). As of R9 ALL
    # non-learned variants — scorer_norm (cosine/hybrid) + head_agg (mean) [R6] and
    # anchor_mode (recency/global/strided) [R9] — are graph-safe, so
    # ds_scorer_is_graph_safe() is True and this guard does not fire. It remains as
    # the single startup gate so any future non-graph-safe variant can re-enable it.
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

    # Recall-oracle diagnostic safety. The oracle hook does host syncs
    # (.item()/dict build) that are illegal during CUDA-graph capture, and under
    # graph replay the Python would not re-run at all (so it would record
    # nothing). It must run the eager selector path, which requires
    # --disable-cuda-graph.
    if getattr(config, "recall_oracle", False) and not getattr(
        server_args, "disable_cuda_graph", False
    ):
        raise ValueError(
            "Double Sparsity recall_oracle diagnostic is not supported under CUDA "
            "graph capture (the oracle hook host-syncs, and under graph replay it "
            "would record nothing). Re-run with --disable-cuda-graph, or disable "
            "recall_oracle."
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
    # Option B apples-to-apples comparison holds (top_k == index_topk). Two
    # sanctioned exceptions, each with its OWN mechanism:
    #   * top_k > index_topk (a WIDER budget) is the Tier-2.A lifted-budget path:
    #     opt in via enable_lifted_budget_decode + lifted_budget_top_k. It is NOT
    #     unlocked by SGLANG_DS_ALLOW_TOPK_MISMATCH (which is an equality-mismatch
    #     ablation, not the lifted-budget ABI).
    #   * any other mismatch (e.g. top_k < index_topk) stays an explicit ablation
    #     gated by SGLANG_DS_ALLOW_TOPK_MISMATCH=1.
    if hf_config is not None:
        try:
            from sglang.srt.configs.model_config import get_dsa_index_topk

            model_topk = get_dsa_index_topk(hf_config)
            lifted = config.enable_lifted_budget_decode
            if model_topk > 0:
                if lifted and config.top_k != model_topk:
                    raise ValueError(
                        f"Double Sparsity enable_lifted_budget_decode requires the base "
                        f"top_k to equal DSA index_topk ({model_topk}); got "
                        f"top_k={config.top_k}. The base budget stays the DSA budget — "
                        "lifted_budget_top_k is the SEPARATE, wider lifted selection "
                        "width (it is what widens the decode, not top_k)."
                    )
                if lifted and config.lifted_budget_top_k <= model_topk:
                    raise ValueError(
                        f"Double Sparsity lifted_budget_top_k={config.lifted_budget_top_k} "
                        f"must be > DSA index_topk={model_topk} when "
                        "enable_lifted_budget_decode is set (the lifted budget widens "
                        "selection beyond index_topk)."
                    )
                if config.top_k > model_topk and not lifted:
                    raise ValueError(
                        f"Double Sparsity top_k={config.top_k} > DSA index_topk={model_topk}. "
                        "A wider-than-index_topk budget is the Tier-2.A lifted-budget path: "
                        "set enable_lifted_budget_decode=true + lifted_budget_top_k in "
                        "--double-sparsity-config. (Do NOT use SGLANG_DS_ALLOW_TOPK_MISMATCH "
                        "for this — that is an equality-mismatch ablation, not the "
                        "lifted-budget ABI.)"
                    )
                if config.top_k != model_topk and not (lifted and config.top_k > model_topk):
                    if os.environ.get("SGLANG_DS_ALLOW_TOPK_MISMATCH") != "1":
                        raise ValueError(
                            f"Double Sparsity top_k={config.top_k} does not match the model's "
                            f"DSA index_topk={model_topk}. The two must agree for the "
                            "Option B apples-to-apples comparison. Either set "
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
                elif lifted:
                    logger.info(
                        "Double Sparsity lifted-budget decode enabled: top_k=%d, "
                        "lifted_budget_top_k=%d, DSA index_topk=%d.",
                        config.top_k, config.lifted_budget_top_k, model_topk,
                    )
        except ImportError:
            logger.debug(
                "get_dsa_index_topk not available; skipping top_k assertion."
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
                "table-free radix fixture has been recorded as passing "
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


def record_radix_fixture_passed(
    server_args: "ServerArgs",
    *,
    artifact_path: "str | None" = None,
) -> None:
    """Record that the M3-B radix-cache stability fixture has passed
    for the current ServerArgs configuration.

    After this is called, ``validate_double_sparsity`` will accept a
    DS launch with ``--disable-radix-cache`` removed (radix cache ON),
    without requiring the developer override
    ``SGLANG_DS_RADIX_OVERRIDE=1``.

    The intended flow (table-free DS):

    1. Operator runs the radix correctness probes on hardware (value
       equivalence, not bit-identity): recall@2048 radix-on-vs-off within
       the +/-0.5pp bar overall and per length, cross-rank selection
       identity within each path, a clean eviction/partial-hit/page-
       boundary edge probe (no stale-slot reuse), no dense fallback, and
       the cold/warm selection flips characterized as value-neutral
       near-cutoff reshuffling.
    2. On all passing, the operator writes the table-free fixture state
       (``write_radix_fixture_state(...)``) and the launcher passes
       ``--double-sparsity-radix-fixture-artifact <state-file>`` so
       ``apply_radix_fixture_artifact`` records this flag at boot and the
       server log records exactly which artifact authorized the flip.
    3. ``serve_double_sparsity.sh`` removes the ``--disable-radix-cache``
       flag once a matching table-free artifact is supplied.

    ``artifact_path`` is optional but recommended: when supplied, the
    audit WARNING records the path + SHA256 of its contents so a grep
    over server logs surfaces both the flip event AND the evidence
    it claims.
    """
    setattr(server_args, "_double_sparsity_radix_fixture_passed", True)

    artifact_suffix = ""
    if artifact_path is not None:
        try:
            with open(artifact_path, "rb") as fh:
                artifact_sha = hashlib.sha256(fh.read()).hexdigest()
            artifact_suffix = (
                f" artifact={artifact_path} artifact_sha256={artifact_sha}"
            )
        except OSError as exc:
            artifact_suffix = (
                f" artifact={artifact_path} artifact_sha256=<unreadable:{exc}>"
            )

    logger.warning(
        "DS radix-cache fixture recorded as PASSED for this ServerArgs; "
        "validator will now accept --disable-radix-cache removal. "
        "Source: record_radix_fixture_passed().%s",
        artifact_suffix,
    )


# ----- No-env-override radix-cache flip via a config-bound state file --------
#
# The radix flip is authorized by a config-bound state file, NOT an env var.
# After the operator runs the radix correctness probes on hardware and they pass,
# they write a state file via `write_radix_fixture_state(...)` recording the
# value-equivalence criterion (recall@2048 radix-on-vs-off within +/-0.5pp overall
# and per length, cross-rank selection identity, a clean eviction/partial-hit/
# page-boundary edge probe, no dense fallback, and the cold/warm selection flips
# documented as value-neutral near-cutoff reshuffling) plus a fingerprint of the
# exact serving config (model / TP / page / KV dtype / channel-mask SHA / table-
# free selector mode). `serve_double_sparsity.sh` then passes
# `--double-sparsity-radix-fixture-artifact <state-file>`; `check_server_args`
# calls `apply_radix_fixture_artifact(server_args)` BEFORE `validate_double_sparsity`,
# which verifies the schema + that the state matches THIS boot's config and only
# then records the fixture-passed flag. A legacy label-capture artifact, a
# superseded bit-identity artifact, or one recorded for a different
# model/mask/config/mode does not authorize this boot (fail-closed).

RADIX_FIXTURE_STATE_SCHEMA = "ds_radix_fixture_state_v1"
# Superseded table-free schema. v1 gated cold/warm SELECTED-INDEX bit-identity,
# which is NOT the radix authorization criterion: a radix cache hit changes the
# decode query (v_h) upstream of DS selection — the cached fp8 latent bytes are
# bit-identical, so the divergence is in the model forward, shared with the
# shipped dense/DSA paths — so the cold/warm top-k reshuffles a small fraction of
# borderline near-cutoff tokens. That reshuffle is value-neutral: recall@2048 is
# unchanged. v1 artifacts are rejected; regenerate under v2.
RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA_V1 = "ds_radix_fixture_state_tablefree_v1"
# Table-free (absorbed-latent) DS radix authorization is VALUE-equivalence, not
# bit-identity: recall@2048 radix-on-vs-off within +/-0.5pp overall and per length,
# cross-rank selection identity within each path, a clean eviction/partial-hit/
# page-boundary edge probe (no stale-slot reuse), no dense fallback, and the
# cold/warm selection flips documented as value-neutral near-cutoff reshuffling.
# The legacy label-capture schema cannot authorize table-free DS (the table is
# deleted); the v1 bit-identity schema above is superseded.
RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA = "ds_radix_fixture_state_tablefree_v2"


def _channel_mask_content_sha256(path: str) -> str:
    """Tensor-content SHA-256 of the channel mask (path/timestamp-independent).

    Hashes the canonical channel selection/weights via the same routine the mask
    writer embeds (``channel_mask.compute_content_sha256``), so a mask
    regenerated with identical content authorizes the same boot regardless of
    file path or the embedded ``created_at`` timestamp. The full-file SHA cannot
    serve this role: ``created_at`` makes it change on every calibration.
    """
    from sglang.srt.layers.attention.double_sparsity.channel_mask import (
        load_channel_mask,
    )

    # load_channel_mask recomputes the content hash over the canonical tensors
    # and raises on a tamper/corrupt mask, so this is fail-closed.
    return load_channel_mask(path).content_sha256


def radix_fixture_config_fingerprint(server_args: "ServerArgs") -> dict:
    """Fingerprint the serving config the radix flip is bound to.

    Pins the channel mask by its TENSOR-CONTENT SHA (not the file path or the
    full-file SHA): a flip recorded against one calibrated mask cannot authorize
    a boot that swaps in a different-content mask, while a mask regenerated with
    identical content authorizes regardless of path or the embedded
    ``created_at`` timestamp (which makes the full-file SHA non-reproducible).
    """
    from sglang.srt.layers.attention.double_sparsity.config import (
        parse_double_sparsity_config,
    )

    config = parse_double_sparsity_config(getattr(server_args, "double_sparsity_config"))
    return {
        "model_path": getattr(server_args, "model_path", None),
        "tp_size": getattr(server_args, "tp_size", None),
        "page_size": getattr(server_args, "page_size", None),
        "kv_cache_dtype": getattr(server_args, "kv_cache_dtype", None),
        "channel_mask_content_sha256": _channel_mask_content_sha256(
            config.channel_mask_path
        ),
        # DS selects table-free (the only mode); pin it so a fixture from any other
        # selector mode cannot authorize this boot.
        "selector_mode": "table_free",
    }


def write_radix_fixture_state(
    path: str,
    *,
    server_args: "ServerArgs",
    recall_equivalence_passed: bool,
    cross_rank_selection_identity_passed: bool,
    edge_probe_passed: bool,
    no_dense_fallback_passed: bool,
    cold_warm_flips_value_neutral_documented: bool,
) -> dict:
    """Write the table-free radix-fixture-passed state file (operator step after the
    radix correctness probes pass on hardware). The authorization criterion is
    VALUE-equivalence, not cold/warm selected-index bit-identity:

    * ``recall_equivalence_passed`` — recall@2048 radix-on vs radix-off within
      +/-0.5pp overall AND per length (measured on the same trial set);
    * ``cross_rank_selection_identity_passed`` — all TP ranks select identically
      within each path;
    * ``edge_probe_passed`` — clean eviction/partial-hit/page-boundary probe (no
      stale-slot reuse);
    * ``no_dense_fallback_passed`` — DS selection never degrades to dense attention;
    * ``cold_warm_flips_value_neutral_documented`` — the cold/warm top-k flips are
      characterized and documented as value-neutral near-cutoff reshuffling (a radix
      cache hit changes the decode query upstream of DS; this is NOT a bit-identity
      claim).

    Returns the written dict."""
    import time as _time

    state = {
        "schema": RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA,
        "recall_equivalence_passed": bool(recall_equivalence_passed),
        "cross_rank_selection_identity_passed": bool(
            cross_rank_selection_identity_passed
        ),
        "edge_probe_passed": bool(edge_probe_passed),
        "no_dense_fallback_passed": bool(no_dense_fallback_passed),
        "cold_warm_flips_value_neutral_documented": bool(
            cold_warm_flips_value_neutral_documented
        ),
        "config": radix_fixture_config_fingerprint(server_args),
        "recorded_at": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    return state


def apply_radix_fixture_artifact(server_args: "ServerArgs") -> None:
    """If `--double-sparsity-radix-fixture-artifact` is set and DS is radix-on,
    verify the state file authorizes THIS config and record the fixture-passed
    flag (so `validate_double_sparsity` accepts radix cache ON).

    No-op when the field is unset, DS is off, or radix is disabled. Fail-closed:
    a missing/malformed/mismatched/not-both-passed state file raises ValueError
    rather than silently authorizing the flip.
    """
    if not getattr(server_args, "enable_double_sparsity", False):
        return
    if getattr(server_args, "disable_radix_cache", True):
        return  # radix-off needs no authorization
    artifact = getattr(server_args, "double_sparsity_radix_fixture_artifact", None)
    if not artifact:
        return  # let validate_double_sparsity raise the radix-cache refusal

    if not os.path.isfile(artifact):
        raise ValueError(
            f"--double-sparsity-radix-fixture-artifact={artifact!r} does not exist; "
            "cannot authorize radix-cache ON."
        )
    try:
        with open(artifact, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"radix-fixture artifact {artifact!r} is unreadable/not JSON: {exc}."
        ) from exc

    schema = state.get("schema") if isinstance(state, dict) else None
    if schema == RADIX_FIXTURE_STATE_SCHEMA:
        # The legacy label-capture fixture proves bit-stability of the materialized
        # TokenLabelTable signatures — which no longer exist (the table is deleted;
        # DS selects table-free off the resident latent). It cannot authorize
        # table-free DS radix-on; the operator must regenerate the table-free fixture.
        raise ValueError(
            f"radix-fixture artifact {artifact!r} uses the legacy label-capture schema "
            f"{RADIX_FIXTURE_STATE_SCHEMA!r}, which cannot authorize table-free Double "
            "Sparsity radix-on (the TokenLabelTable is deleted). Regenerate the "
            f"table-free fixture (schema {RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA!r}: "
            "recall equivalence + cross-rank identity + edge probe + no-dense-fallback "
            "+ documented value-neutral flips)."
        )
    if schema == RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA_V1:
        # The v1 table-free schema gated cold/warm SELECTED-INDEX bit-identity. That
        # is not the radix authorization criterion: a radix cache hit changes the
        # decode query upstream of DS (the cached fp8 latent bytes are bit-identical),
        # so the cold/warm top-k reshuffles a small fraction of borderline near-cutoff
        # tokens — value-neutral (recall@2048 unchanged). Reject; regenerate under v2.
        raise ValueError(
            f"radix-fixture artifact {artifact!r} uses the superseded table-free schema "
            f"{RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA_V1!r}, which gated cold/warm "
            "SELECTED-INDEX bit-identity. That is not the radix authorization criterion "
            "(a radix cache hit changes the decode query upstream of DS, so the top-k "
            "reshuffles a small fraction of borderline tokens — value-neutral; "
            "recall@2048 is unchanged). Regenerate under "
            f"{RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA!r} (recall equivalence + cross-rank "
            "identity + edge probe + no-dense-fallback + documented value-neutral flips)."
        )
    if not isinstance(state, dict) or schema != RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA:
        raise ValueError(
            f"radix-fixture artifact {artifact!r} schema must be "
            f"{RADIX_FIXTURE_STATE_TABLEFREE_SCHEMA!r}, got "
            f"{schema if isinstance(state, dict) else type(state).__name__!r}."
        )
    required = (
        "recall_equivalence_passed",
        "cross_rank_selection_identity_passed",
        "edge_probe_passed",
        "no_dense_fallback_passed",
        "cold_warm_flips_value_neutral_documented",
    )
    # Fail-closed: each probe field must be the JSON boolean `true` EXACTLY. A
    # truthy non-bool (e.g. the string "false", a non-zero int) must NOT authorize
    # radix-on — `is True` rejects every non-bool and every False/missing value.
    bad = [k for k in required if state.get(k) is not True]
    if bad:
        raise ValueError(
            f"radix-fixture artifact {artifact!r} does not record every table-free "
            "correctness probe as the JSON boolean true; not satisfied for "
            + ", ".join(f"{k}={state.get(k)!r}" for k in bad)
            + " (radix-cache ON is not authorized)."
        )

    recorded = state.get("config") or {}
    current = radix_fixture_config_fingerprint(server_args)
    mismatches = [
        f"{k}: artifact={recorded.get(k)!r} current={current[k]!r}"
        for k in current
        if recorded.get(k) != current[k]
    ]
    if mismatches:
        raise ValueError(
            f"radix-fixture artifact {artifact!r} was recorded for a different "
            "serving config; it cannot authorize this boot. Mismatches: "
            + "; ".join(mismatches)
        )

    record_radix_fixture_passed(server_args, artifact_path=artifact)
