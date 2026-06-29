"""Channel-mask file format, loader, validator, and startup sanity probe.

The channel mask file is produced offline by :mod:`calibrate` and consumed at
server startup. It freezes the per-(layer, head) channel projection the runtime
selector applies to score each KV token for top-K selection.

Schema (``safetensors``):

* Tensors

  - ``channel_selection`` ``int32 [num_layers, num_heads, label_dim]`` — vocab
    of channel indices selected per (layer, head).
  - ``channel_weights`` ``float32 [num_layers, num_heads, label_dim]`` —
    normalized importance weights applied during projection.

* Metadata (``__metadata__`` dict in safetensors header)

  - ``schema_version`` — string, currently ``"1"``. Bumped on incompatible
    schema changes; old loaders refuse newer schema_versions.
  - ``dtype`` — string, the ``kv_cache_dtype`` the file was calibrated for
    (``"fp8_e4m3"`` or ``"bfloat16"``).
  - ``head_dim`` — string of int, sanity check against the model config.
  - ``page_size`` — string of int, must match the running server's page size.
  - ``label_dim`` — string of int, must match the selector buffer.
  - ``created_at`` — ISO-8601 timestamp at calibration time.
  - ``content_sha256`` — SHA-256 over the concatenated raw bytes of
    ``channel_selection`` (cast to ``torch.int32``) and ``channel_weights``
    (cast to ``torch.float32``), in that order. Recomputed at load.

The validator drops two fields the earlier draft carried: ``model_revision_sha``
(passed for LoRA fine-tunes that miscalibrate) and ``tp_world_size``
(derivable from runtime; embedding it invited the rank-divergence bug settles).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import torch

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


SCHEMA_VERSION = "1"
_REQUIRED_METADATA_FIELDS = (
    "schema_version",
    "dtype",
    "head_dim",
    "page_size",
    "label_dim",
    "content_sha256",
)
_SUPPORTED_DTYPES = ("fp8_e4m3", "bfloat16")


class DoubleSparsityChannelMaskMissing(FileNotFoundError):
    """Channel-mask file is absent (typed)."""


class DoubleSparsityChannelMaskCorrupt(ValueError):
    """Channel-mask file present but content is corrupt: schema drift,
    hash mismatch, dtype mismatch, out-of-range channel selection, OR
    value-domain corruption (NaN / Inf / all-zero per-row weights).
    """


@dataclass
class ChannelMask:
    """Loaded channel mask payload + metadata."""

    channel_selection: torch.Tensor  # int32 [L, H, label_dim]
    channel_weights: torch.Tensor  # float32 [L, H, label_dim]
    schema_version: str
    dtype: str
    head_dim: int
    page_size: int
    label_dim: int
    content_sha256: str
    created_at: Optional[str] = None
    raw_metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def num_layers(self) -> int:
        return int(self.channel_selection.shape[0])

    @property
    def num_heads(self) -> int:
        return int(self.channel_selection.shape[1])


def compute_content_sha256(
    channel_selection: torch.Tensor, channel_weights: torch.Tensor
) -> str:
    """Return SHA-256 over the canonicalized payload bytes."""

    selection_bytes = (
        channel_selection.detach().to(torch.int32).contiguous().cpu().numpy().tobytes()
    )
    weights_bytes = (
        channel_weights.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    )
    hasher = hashlib.sha256()
    hasher.update(selection_bytes)
    hasher.update(weights_bytes)
    return hasher.hexdigest()


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"channel mask metadata field {field_name!r} is not an integer: {value!r}."
        ) from exc


def load_channel_mask(path: str, *, map_location: str = "cpu") -> ChannelMask:
    """Load and content-verify a channel mask file.

    Recomputes ``content_sha256`` over the canonicalized tensors and raises if
    it does not match the metadata. Raises if any required metadata field is
    missing, if the schema version is newer than supported, or if the dtype
    is outside the supported set.
    """

    if not isinstance(path, str) or not path:
        raise ValueError("channel mask file path must be a non-empty string.")
    if not os.path.isfile(path):
        raise DoubleSparsityChannelMaskMissing(
            f"channel mask file not found at {path!r}. Set "
            "'channel_mask_path' in --double-sparsity-config to a readable file."
        )

    from safetensors import safe_open

    with safe_open(path, framework="pt", device=map_location) as f:
        tensor_keys = list(f.keys())
        if (
            "channel_selection" not in tensor_keys
            or "channel_weights" not in tensor_keys
        ):
            raise ValueError(
                f"channel mask file {path!r} is missing required tensors "
                "'channel_selection' and/or 'channel_weights'."
            )
        channel_selection = f.get_tensor("channel_selection")
        channel_weights = f.get_tensor("channel_weights")
        raw_metadata = dict(f.metadata() or {})

    if channel_selection.dim() != 3:
        raise ValueError(
            f"channel_selection must be 3-D [L, H, label_dim], got shape "
            f"{tuple(channel_selection.shape)}."
        )
    if channel_weights.shape != channel_selection.shape:
        raise ValueError(
            f"channel_weights shape {tuple(channel_weights.shape)} must match "
            f"channel_selection shape {tuple(channel_selection.shape)}."
        )

    missing = [k for k in _REQUIRED_METADATA_FIELDS if k not in raw_metadata]
    if missing:
        raise ValueError(
            f"channel mask file {path!r} is missing required metadata fields: {missing}. "
            "Re-run calibration with a current calibrate.py."
        )

    schema_version = raw_metadata["schema_version"]
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"channel mask schema_version {schema_version!r} is not supported by "
            f"this loader (expected {SCHEMA_VERSION!r}). Update SGLang or "
            "re-calibrate."
        )

    dtype = raw_metadata["dtype"]
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"channel mask dtype {dtype!r} is not in supported set {_SUPPORTED_DTYPES}."
        )

    head_dim = _coerce_int(raw_metadata["head_dim"], "head_dim")
    page_size = _coerce_int(raw_metadata["page_size"], "page_size")
    label_dim = _coerce_int(raw_metadata["label_dim"], "label_dim")
    if label_dim != int(channel_selection.shape[-1]):
        raise ValueError(
            f"channel mask metadata label_dim={label_dim} does not match "
            f"channel_selection.shape[-1]={int(channel_selection.shape[-1])}."
        )

    # channel_selection holds vocab indices into a head_dim-wide channel
    # axis; out-of-range values blow up later during the channel gather
    # with an opaque error, so reject up front.
    if channel_selection.numel() > 0:
        cs_int = channel_selection.to(torch.int64)
        cs_min = int(cs_int.min().item())
        cs_max = int(cs_int.max().item())
        if cs_min < 0 or cs_max >= head_dim:
            raise ValueError(
                f"channel mask channel_selection values are out of range "
                f"[0, head_dim={head_dim}): min={cs_min}, max={cs_max}. The "
                "file's content hash is valid but the selection indices "
                "cannot index into the model's head dimension. Recalibrate "
                "with a current calibrate.py."
            )

    # Value-domain validation: NaN, +/-Inf, or all-zero per-row weights
    # silently degrade compute_page_scores into NaN / flat top-K and yield
    # arbitrary selections. Reject at startup with a typed exception.
    if channel_weights.numel() > 0:
        cw_float = channel_weights.to(torch.float32)
        if bool(torch.isnan(cw_float).any().item()):
            raise DoubleSparsityChannelMaskCorrupt(
                f"channel mask file {path!r} channel_weights contain NaN. "
                "The file's hash may still match, but NaN weights produce "
                "NaN scores and arbitrary top-K. Recalibrate."
            )
        if bool(torch.isinf(cw_float).any().item()):
            raise DoubleSparsityChannelMaskCorrupt(
                f"channel mask file {path!r} channel_weights contain +/-Inf. "
                "Recalibrate with a current calibrate.py."
            )
        # All-zero per-(layer, head) row: collapses the projection to zero
        # for that row → all pages tie at score 0 → degenerate top-K.
        per_row_abs_sum = cw_float.abs().sum(dim=-1)
        if bool((per_row_abs_sum == 0).any().item()):
            raise DoubleSparsityChannelMaskCorrupt(
                f"channel mask file {path!r} contains at least one all-zero "
                "channel_weights row (per (layer, head)). Such rows degrade "
                "top-K into ties at score 0; recalibrate."
            )

    expected_hash = raw_metadata["content_sha256"]
    actual_hash = compute_content_sha256(channel_selection, channel_weights)
    if expected_hash != actual_hash:
        raise DoubleSparsityChannelMaskCorrupt(
            f"channel mask file {path!r} content hash mismatch: metadata "
            f"reports {expected_hash[:12]}... but recompute yields {actual_hash[:12]}.... "
            "The file is corrupted or was edited after calibration."
        )

    mask = ChannelMask(
        channel_selection=channel_selection.to(torch.int32).contiguous(),
        channel_weights=channel_weights.to(torch.float32).contiguous(),
        schema_version=schema_version,
        dtype=dtype,
        head_dim=head_dim,
        page_size=page_size,
        label_dim=label_dim,
        content_sha256=actual_hash,
        created_at=raw_metadata.get("created_at"),
        raw_metadata=raw_metadata,
    )

    logger.info(
        "Loaded channel mask file %s: content_sha256=%s dtype=%s head_dim=%d "
        "page_size=%d label_dim=%d created_at=%s L=%d H=%d",
        path,
        actual_hash[:12],
        dtype,
        head_dim,
        page_size,
        label_dim,
        mask.created_at,
        mask.num_layers,
        mask.num_heads,
    )
    return mask


def save_channel_mask(
    path: str,
    channel_selection: torch.Tensor,
    channel_weights: torch.Tensor,
    *,
    dtype: str,
    head_dim: int,
    page_size: int,
    label_dim: int,
    created_at: str,
    extra_metadata: Optional[Mapping[str, str]] = None,
) -> str:
    """Persist a channel mask file, returning the recomputed content_sha256.

    Used by :mod:`calibrate` to write the offline-produced artifact.
    """

    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"channel mask dtype {dtype!r} is not in supported set {_SUPPORTED_DTYPES}."
        )
    selection = channel_selection.detach().to(torch.int32).contiguous().cpu()
    weights = channel_weights.detach().to(torch.float32).contiguous().cpu()
    if selection.shape != weights.shape:
        raise ValueError(
            f"channel_selection shape {tuple(selection.shape)} must match "
            f"channel_weights shape {tuple(weights.shape)}."
        )
    if selection.shape[-1] != label_dim:
        raise ValueError(
            f"channel_selection.shape[-1]={int(selection.shape[-1])} must match "
            f"label_dim={label_dim}."
        )

    content_sha256 = compute_content_sha256(selection, weights)
    metadata: Dict[str, str] = {
        "schema_version": SCHEMA_VERSION,
        "dtype": dtype,
        "head_dim": str(int(head_dim)),
        "page_size": str(int(page_size)),
        "label_dim": str(int(label_dim)),
        "created_at": str(created_at),
        "content_sha256": content_sha256,
    }
    if extra_metadata:
        for k, v in extra_metadata.items():
            if k in metadata:
                raise ValueError(
                    f"extra_metadata key {k!r} collides with reserved metadata."
                )
            metadata[k] = str(v)

    from safetensors.torch import save_file

    save_file(
        {"channel_selection": selection, "channel_weights": weights},
        path,
        metadata=metadata,
    )
    return content_sha256


def slice_per_rank(
    mask: ChannelMask,
    *,
    num_local_heads: int,
    rank: int,
    tp_size: int,
) -> ChannelMask:
    """Return a per-rank head-sliced view of a TP-agnostic channel mask.

    The calibration artifact records ``channel_selection[L, H_full, label_dim]``
    where ``H_full = num_attention_heads``. At serving time each TP rank owns
    ``num_local_heads = H_full / tp_size`` consecutive heads, so the selector's
    binding requires ``[L, num_local_heads, label_dim]``. This helper performs
    that slice; the deploying team should call it on the loaded mask before
    passing it into ``DoubleSparsitySelector.bind_runtime_data``.

    The returned ``ChannelMask`` carries the SAME ``content_sha256`` as the
    source file (the on-disk hash is unchanged; this is an in-memory view).
    """

    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}.")
    if not (0 <= rank < tp_size):
        raise ValueError(f"rank={rank} must be in [0, {tp_size}).")
    if num_local_heads <= 0:
        raise ValueError(f"num_local_heads must be positive, got {num_local_heads}.")
    h_full = int(mask.channel_selection.shape[1])
    if h_full != num_local_heads * tp_size:
        raise ValueError(
            f"channel mask H_full={h_full} is not divisible into "
            f"num_local_heads={num_local_heads} × tp_size={tp_size}."
        )
    start = rank * num_local_heads
    stop = start + num_local_heads
    return ChannelMask(
        channel_selection=mask.channel_selection[:, start:stop, :].contiguous(),
        channel_weights=mask.channel_weights[:, start:stop, :].contiguous(),
        schema_version=mask.schema_version,
        dtype=mask.dtype,
        head_dim=mask.head_dim,
        page_size=mask.page_size,
        label_dim=mask.label_dim,
        content_sha256=mask.content_sha256,
        created_at=mask.created_at,
        raw_metadata=mask.raw_metadata,
    )


def validate_against_runtime(
    mask: ChannelMask,
    *,
    server_kv_cache_dtype: str,
    server_page_size: int,
    server_label_dim: int,
    model_head_dim: int,
) -> None:
    """Raise if the loaded mask is incompatible with the running configuration.

    Per the loader contract: shape correctness, content identity (covered by load_channel_mask
    hash check), and configuration agreement on dtype / head_dim / page_size /
    label_dim.
    """

    mismatches = []
    if mask.dtype != server_kv_cache_dtype:
        mismatches.append(
            f"dtype: mask={mask.dtype!r} server={server_kv_cache_dtype!r}"
        )
    if mask.head_dim != int(model_head_dim):
        mismatches.append(f"head_dim: mask={mask.head_dim} model={int(model_head_dim)}")
    if mask.page_size != int(server_page_size):
        mismatches.append(
            f"page_size: mask={mask.page_size} server={int(server_page_size)}"
        )
    if mask.label_dim != int(server_label_dim):
        mismatches.append(
            f"label_dim: mask={mask.label_dim} selector={int(server_label_dim)}"
        )
    if mismatches:
        raise ValueError("channel mask runtime mismatch:\n  " + "\n  ".join(mismatches))


def verify_bind_shapes(
    mask: ChannelMask,
    *,
    model_nope_head_dim: int,
    num_local_heads: int,
    tp_size: int,
    num_hidden_layers: int,
    server_page_size: int,
    server_label_dim: int,
    server_kv_cache_dtype: Optional[str] = None,
) -> None:
    """Hard-error (naming every offending field) if a calibrated channel mask is
    shape-incompatible with the running model, at the attention bind site.

    The startup validator can only best-effort the per-head no-PE width because
    the attention layer is not yet constructed when it runs; the projection width
    becomes authoritative only here. This is the gate that turns an explicit
    Double Sparsity request on an unsupported shape into a diagnostic failure
    instead of a silent wrong-channel selection — e.g. a mask calibrated for a
    narrower no-PE head (``head_dim``) loaded against a wider one keeps its
    indices in range, so the channel gather would not crash; it would quietly
    score the wrong channels.

    ``model_nope_head_dim`` is the per-head no-PE width the selection indices live
    in (``qk_nope_head_dim``). All checks aggregate so one error names every
    mismatch at once. The caller invokes this only when Double Sparsity is
    explicitly enabled, so the native attention path is unaffected when it is off.
    """

    problems: list[str] = []

    # dtype / head_dim / page_size / label_dim agreement with the running config.
    # Skip the dtype leg only when the server dtype is still unresolved (auto);
    # the startup validator already gates dtype, and head_dim is the new check.
    effective_dtype = (
        mask.dtype if server_kv_cache_dtype in (None, "auto") else server_kv_cache_dtype
    )
    try:
        validate_against_runtime(
            mask,
            server_kv_cache_dtype=effective_dtype,
            server_page_size=server_page_size,
            server_label_dim=server_label_dim,
            model_head_dim=model_nope_head_dim,
        )
    except ValueError as exc:
        problems.append(str(exc))

    # label_dim must agree across the mask metadata, the stored tensor, and the
    # selector buffer — a partial match silently truncates/over-reads labels.
    sel = mask.channel_selection
    tensor_label_dim = int(sel.shape[-1])
    if tensor_label_dim != int(server_label_dim):
        problems.append(
            f"channel_selection label_dim={tensor_label_dim} != selector "
            f"label_dim={int(server_label_dim)}"
        )

    # Per-head weights must line up with the per-head selection exactly.
    if tuple(mask.channel_weights.shape) != tuple(sel.shape):
        problems.append(
            f"channel_weights shape {tuple(mask.channel_weights.shape)} != "
            f"channel_selection shape {tuple(sel.shape)}"
        )

    # Layer count must match the model so layer_id indexing stays in range.
    mask_layers = int(sel.shape[0])
    if mask_layers != int(num_hidden_layers):
        problems.append(
            f"channel mask layers={mask_layers} != model "
            f"num_hidden_layers={int(num_hidden_layers)}"
        )

    # Head count must divide cleanly into this rank's local-head slice.
    h_full = int(sel.shape[1])
    if h_full != num_local_heads * tp_size:
        problems.append(
            f"channel mask num_heads={h_full} != num_local_heads={num_local_heads} "
            f"x tp_size={tp_size}"
        )

    # Selection indices must address the model's no-PE channel space. A mask from
    # a narrower model keeps small indices in range, so the equality check on
    # head_dim above is what catches it; this guards a too-wide mask.
    if sel.numel():
        cs_max = int(sel.max())
        if cs_max >= int(model_nope_head_dim):
            problems.append(
                f"channel_selection max index={cs_max} >= model no-PE "
                f"head_dim={int(model_nope_head_dim)}"
            )

    if problems:
        raise ValueError(
            "Double Sparsity channel mask is incompatible with the running model "
            "(Double Sparsity was explicitly requested, so this is a hard error "
            "rather than a silent fall back to native attention):\n  "
            + "\n  ".join(problems)
        )
