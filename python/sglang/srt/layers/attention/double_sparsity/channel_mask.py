"""Channel-mask file format, loader, validator, and startup sanity probe.

The channel mask file is produced offline by :mod:`calibrate` and consumed at
server startup. It freezes the per-(layer, head) projection that the
:mod:`page_signature_write` Triton kernel uses to compress each KV page into
a ``label_dim``-wide signature for runtime top-K selection.

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
(derivable from runtime; embedding it invited the rank-divergence bug DEC-9
settles).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.double_sparsity.config import (
        DoubleSparsityConfig,
    )

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
        raise FileNotFoundError(
            f"channel mask file not found at {path!r}. Set "
            "'channel_mask_path' in --double-sparsity-config to a readable file."
        )

    from safetensors import safe_open

    with safe_open(path, framework="pt", device=map_location) as f:
        tensor_keys = list(f.keys())
        if "channel_selection" not in tensor_keys or "channel_weights" not in tensor_keys:
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

    expected_hash = raw_metadata["content_sha256"]
    actual_hash = compute_content_sha256(channel_selection, channel_weights)
    if expected_hash != actual_hash:
        raise ValueError(
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
        raise ValueError(
            f"num_local_heads must be positive, got {num_local_heads}."
        )
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

    Per AC-4: shape correctness, content identity (covered by load_channel_mask
    hash check), and configuration agreement on dtype / head_dim / page_size /
    label_dim. Behavioural sanity is :func:`startup_sanity_probe`.
    """

    mismatches = []
    if mask.dtype != server_kv_cache_dtype:
        mismatches.append(
            f"dtype: mask={mask.dtype!r} server={server_kv_cache_dtype!r}"
        )
    if mask.head_dim != int(model_head_dim):
        mismatches.append(
            f"head_dim: mask={mask.head_dim} model={int(model_head_dim)}"
        )
    if mask.page_size != int(server_page_size):
        mismatches.append(
            f"page_size: mask={mask.page_size} server={int(server_page_size)}"
        )
    if mask.label_dim != int(server_label_dim):
        mismatches.append(
            f"label_dim: mask={mask.label_dim} selector={int(server_label_dim)}"
        )
    if mismatches:
        raise ValueError(
            "channel mask runtime mismatch:\n  " + "\n  ".join(mismatches)
        )


@dataclass
class SanityProbeResult:
    passed: bool
    score: float
    needle_position: int
    selected_indices: Optional[torch.Tensor] = None
    skipped_reason: Optional[str] = None


def startup_sanity_probe(
    mask: ChannelMask,
    selector,
    *,
    haystack_pages: int = 8,
    page_size: int = 64,
    needle_page: int = 4,
    abort_on_placeholder: bool = False,
) -> SanityProbeResult:
    """NIAH-min sanity probe: one needle, one short haystack.

    Constructs a deterministic 512-token haystack split into 8 pages of 64
    tokens each, plants a "needle" page-score signal at ``needle_page``, runs
    the selector, and asserts the needle page lands in
    ``selected_indices``. With the placeholder selector the probe is
    inconclusive (returns ``passed=False, skipped_reason=...``) — production
    serving is independently refused by the placeholder-guard.
    """

    if needle_page >= haystack_pages or needle_page < 0:
        raise ValueError(
            f"needle_page={needle_page} must be in [0, {haystack_pages})."
        )

    is_placeholder = bool(getattr(selector, "IS_PLACEHOLDER", False))
    if is_placeholder:
        msg = (
            "channel mask sanity probe is inconclusive with the placeholder "
            "selector: it returns deterministic ascending page IDs regardless "
            "of channel weights. Real selection kernels must land for the "
            "probe to be load-bearing."
        )
        if abort_on_placeholder:
            raise RuntimeError(msg)
        logger.warning(msg)
        return SanityProbeResult(
            passed=False,
            score=0.0,
            needle_position=needle_page,
            selected_indices=None,
            skipped_reason="placeholder_selector",
        )

    # Real-selector path: plant a needle directly in the page-signature
    # table (label-dim 0 set high at `needle_page`, low at the others), build
    # a query that projects onto label-dim 0 via the loaded channel mask, and
    # ask the selector to retrieve a SMALL top-K. The probe must discriminate
    # the needle from the haystack — a trivial top_k >= haystack_pages passes
    # by inclusion alone, so this overrides max_top_k to a sharp value.
    table = getattr(selector, "page_signature_table", None)
    if table is None:
        msg = (
            "channel mask sanity probe needs a bound page_signature_table "
            "on the selector; got None. Call bind_runtime_data first."
        )
        if abort_on_placeholder:
            raise RuntimeError(msg)
        logger.warning(msg)
        return SanityProbeResult(
            passed=False,
            score=0.0,
            needle_position=needle_page,
            selected_indices=None,
            skipped_reason="no_page_signature_table",
        )

    from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
        retrieve_topk_via_signatures,
    )

    device = table.signatures.device
    num_heads = int(getattr(selector, "num_local_heads", table.signatures.shape[2]))
    head_dim = int(getattr(selector, "head_dim", 128))
    layer_id = 0

    # Move per-layer mask slices onto the table device.
    sel_layer = mask.channel_selection[layer_id].to(device)  # [H, label_dim] int32
    w_layer = mask.channel_weights[layer_id].to(device)  # [H, label_dim] fp32

    # Snapshot the layer's signatures and valid_mask so we can restore on exit.
    sig_snapshot = table.signatures[layer_id, :haystack_pages].clone()
    valid_snapshot = table.valid_mask[layer_id, :haystack_pages].clone()

    try:
        # Plant: weak baseline along label-dim 0, strong signal at needle_page.
        table.signatures[layer_id, :haystack_pages].zero_()
        table.signatures[layer_id, :haystack_pages, :, 0] = 0.1
        table.signatures[layer_id, needle_page, :, 0] = 10.0
        table.valid_mask[layer_id, :haystack_pages] = True

        # Build a query that, when projected through (sel_layer, w_layer),
        # has its first label-dim slot equal to ~1.0 per head.
        queries = torch.zeros(1, num_heads, head_dim, device=device, dtype=torch.float32)
        for h in range(min(num_heads, sel_layer.shape[0])):
            ch_idx = int(sel_layer[h, 0].item())
            if ch_idx >= head_dim:
                continue
            weight = float(w_layer[h, 0].item())
            queries[0, h, ch_idx] = 1.0 / weight if abs(weight) > 1e-6 else 1.0

        # Sharp top-K so the probe must discriminate.
        probe_top_k = max(1, haystack_pages // 4)
        per_request_valid = torch.zeros(
            1, table.signatures.shape[1], dtype=torch.bool, device=device
        )
        per_request_valid[0, :haystack_pages] = True

        selected_indices, valid_lengths = retrieve_topk_via_signatures(
            queries=queries,
            page_signatures=table.signatures,
            valid_mask=table.valid_mask,
            channel_selection=mask.channel_selection.to(device),
            channel_weights=mask.channel_weights.to(device),
            layer_id=layer_id,
            max_top_k=probe_top_k,
            per_request_valid=per_request_valid,
        )
    finally:
        table.signatures[layer_id, :haystack_pages] = sig_snapshot
        table.valid_mask[layer_id, :haystack_pages] = valid_snapshot

    row = selected_indices[0]
    length = int(valid_lengths[0])
    unpadded = [int(v) for v in row[:length].tolist() if v >= 0]
    passed = needle_page in unpadded
    score = 1.0 if passed else 0.0
    return SanityProbeResult(
        passed=passed,
        score=score,
        needle_position=needle_page,
        selected_indices=selected_indices,
        skipped_reason=None,
    )
