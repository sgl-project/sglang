"""Double Sparsity calibration config — schema, validation, and TP slicing."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SUPPORTED_CHANNEL_TYPES = ("k",)
SUPPORTED_INDEXING = ("global_kv_head_id",)
SUPPORTED_GQA_REDUCTIONS = ("max_abs", "mean", "soq")


@dataclass(frozen=True)
class DoubleSparsityCalibration:
    """Parsed, validated calibration ready to be TP-sliced.

    `channels` is a dict layer_id (int) -> tensor of shape
    `[num_kv_heads_global, heavy_channels]` int32 on CPU, indices in `[0, head_dim)`.
    """

    schema_version: int
    model_arch: str
    model_name_or_path: str
    head_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads_global: int
    heavy_channels: int
    channel_type: str
    indexing: str
    channels: Dict[int, torch.Tensor]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid double-sparsity config: {message}")


def parse_calibration_file(path: str | Path) -> DoubleSparsityCalibration:
    """Load and validate a calibration JSON file. Pure CPU; no model required."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Calibration file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return parse_calibration_dict(raw)


def parse_calibration_dict(raw: dict) -> DoubleSparsityCalibration:
    """Validate a calibration dict (already-loaded JSON)."""
    schema_version = raw.get("schema_version")
    _require(
        schema_version == SCHEMA_VERSION,
        f"schema_version must be {SCHEMA_VERSION}, got {schema_version!r}",
    )

    head_dim = int(raw["head_dim"])
    num_layers = int(raw["num_layers"])
    num_heads = int(raw["num_heads"])
    num_kv_heads_global = int(raw["num_kv_heads"])
    heavy_channels = int(raw["heavy_channels"])
    channel_type = raw["channel_type"]
    indexing = raw.get("indexing", "global_kv_head_id")

    _require(
        channel_type in SUPPORTED_CHANNEL_TYPES,
        f"channel_type must be one of {SUPPORTED_CHANNEL_TYPES}, got {channel_type!r}",
    )
    _require(
        indexing in SUPPORTED_INDEXING,
        f"indexing must be one of {SUPPORTED_INDEXING}, got {indexing!r}",
    )
    _require(head_dim > 0, "head_dim must be positive")
    _require(num_layers > 0, "num_layers must be positive")
    _require(num_kv_heads_global > 0, "num_kv_heads must be positive")
    _require(
        num_heads >= num_kv_heads_global,
        f"num_heads {num_heads} must be >= num_kv_heads {num_kv_heads_global}",
    )
    _require(
        num_heads % num_kv_heads_global == 0,
        f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads_global}",
    )
    _require(
        0 < heavy_channels <= head_dim,
        f"heavy_channels {heavy_channels} must be in (0, head_dim={head_dim}]",
    )

    channels_raw = raw["channels"]
    _require(
        len(channels_raw) == num_layers,
        f"channels has {len(channels_raw)} layers, expected {num_layers}",
    )

    channels: Dict[int, torch.Tensor] = {}
    for layer_key, rows in channels_raw.items():
        layer_id = int(layer_key)
        _require(
            0 <= layer_id < num_layers,
            f"channel layer id {layer_id} out of range [0, {num_layers})",
        )
        _require(
            len(rows) == num_kv_heads_global,
            f"layer {layer_id} has {len(rows)} rows, expected {num_kv_heads_global}",
        )
        flat = []
        for h, row in enumerate(rows):
            _require(
                len(row) == heavy_channels,
                f"layer {layer_id} head {h}: row length {len(row)} != heavy_channels {heavy_channels}",
            )
            ints = [int(x) for x in row]
            bad = next((v for v in ints if not 0 <= v < head_dim), None)
            _require(
                bad is None,
                f"layer {layer_id} head {h}: channel index {bad} out of range [0, {head_dim})",
            )
            _require(
                len(set(ints)) == len(ints),
                f"layer {layer_id} head {h}: duplicate channel indices",
            )
            flat.append(ints)
        channels[layer_id] = torch.tensor(flat, dtype=torch.int32)

    return DoubleSparsityCalibration(
        schema_version=schema_version,
        model_arch=raw.get("model_arch", ""),
        model_name_or_path=raw.get("model_name_or_path", ""),
        head_dim=head_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads_global=num_kv_heads_global,
        heavy_channels=heavy_channels,
        channel_type=channel_type,
        indexing=indexing,
        channels=channels,
    )


def validate_against_model(
    calib: DoubleSparsityCalibration,
    *,
    head_dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads_global: int,
) -> None:
    """Compare a parsed calibration against the loaded model's geometry."""
    _require(
        calib.head_dim == head_dim,
        f"head_dim mismatch: calibration={calib.head_dim} model={head_dim}",
    )
    _require(
        calib.num_layers == num_layers,
        f"num_layers mismatch: calibration={calib.num_layers} model={num_layers}",
    )
    _require(
        calib.num_heads == num_heads,
        f"num_heads mismatch: calibration={calib.num_heads} model={num_heads}",
    )
    _require(
        calib.num_kv_heads_global == num_kv_heads_global,
        f"num_kv_heads mismatch: calibration={calib.num_kv_heads_global} "
        f"model={num_kv_heads_global}",
    )


def slice_for_tp(
    calib: DoubleSparsityCalibration,
    *,
    tp_size: int,
    tp_rank: int,
) -> Dict[int, torch.Tensor]:
    """Return per-layer channel-index tensors for this TP rank.

    KV heads are partitioned along the head axis: each rank owns a contiguous
    range of `num_kv_heads_global / tp_size` heads. Channel indices live in
    `[0, head_dim)` and are TP-independent, so we only slice rows.
    """
    _require(tp_size >= 1, f"tp_size must be >= 1, got {tp_size}")
    _require(0 <= tp_rank < tp_size, f"tp_rank {tp_rank} out of range [0, {tp_size})")
    _require(
        calib.num_kv_heads_global % tp_size == 0,
        f"num_kv_heads_global {calib.num_kv_heads_global} not divisible by tp_size {tp_size}",
    )
    num_kv_heads_local = calib.num_kv_heads_global // tp_size
    start = tp_rank * num_kv_heads_local
    end = start + num_kv_heads_local

    sliced: Dict[int, torch.Tensor] = {}
    for layer_id, tensor in calib.channels.items():
        sliced[layer_id] = tensor[start:end].clone()
    return sliced


def channel_indices_for_runtime(
    calib: DoubleSparsityCalibration,
    *,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """TP-slice and move to device, returning per-layer int32 tensors."""
    sliced = slice_for_tp(calib, tp_size=tp_size, tp_rank=tp_rank)
    return {
        layer_id: tensor.to(device=device, dtype=torch.int32)
        for layer_id, tensor in sliced.items()
    }


SUPPORTED_BLOCK_T = (256, 512, 1024, 2048)
SUPPORTED_K_BLOCK = (16, 32, 64, 128, 256)

# Single Triton program in-program-sort capacity. Below this, stage-2 merge
# and the union pass each fit in one program; above, the implementer must
# enable the chunked path. Empirical (v1.1 sweep tunable).
DEFAULT_MERGE_SAFE_THRESHOLD = 4096
DEFAULT_UNION_SAFE_THRESHOLD = 4096


@dataclass(frozen=True)
class DoubleSparsityRuntimeConfig:
    """Runtime knobs read from ServerArgs once at startup."""

    heavy_channels: int
    token_budget: int
    recent_tokens: int
    sink_tokens: int
    min_seq_len: int
    max_selected_per_request: int
    gqa_reduction: str
    klabel_dtype: str
    # v1.1 selection-kernel knobs (consumed by the two-stage Triton path).
    block_t: int = 1024
    k_block: int = 64
    # Worst-case decode batch size used to size DS selection scratch buffers
    # at algorithm init. Sourced from `server_args.max_running_requests` (or
    # `cuda_graph_max_bs`); default 1 keeps the existing test/CPU paths happy.
    scratch_max_bs: int = 1
    # Top-k selector backend used inside `ds_native_sparse_decode`.
    # Default `"torch"` keeps behavior identical to pre-flag builds.
    # See `selector_backends.SUPPORTED_SELECTOR_BACKENDS`.
    selector_backend: str = "torch"

    def validate(self) -> None:
        _require(self.heavy_channels > 0, "heavy_channels must be positive")
        _require(self.token_budget > 0, "token_budget must be positive")
        _require(
            self.recent_tokens >= 1,
            "recent_tokens must be >= 1 so the current decode position is always retained",
        )
        _require(self.sink_tokens >= 0, "sink_tokens must be non-negative")
        _require(self.min_seq_len >= 1, "min_seq_len must be >= 1")
        _require(
            self.max_selected_per_request >= 1, "max_selected_per_request must be >= 1"
        )
        _require(
            self.min_seq_len <= self.max_selected_per_request,
            f"min_seq_len ({self.min_seq_len}) must be <= "
            f"max_selected_per_request ({self.max_selected_per_request}); "
            f"otherwise dense-fallback rows would not fit the captured FA3 page-table.",
        )
        _require(
            self.gqa_reduction in SUPPORTED_GQA_REDUCTIONS,
            f"gqa_reduction must be one of {SUPPORTED_GQA_REDUCTIONS}, "
            f"got {self.gqa_reduction!r}",
        )
        _require(
            self.klabel_dtype in ("bf16", "fp32"),
            f"klabel_dtype must be 'bf16' or 'fp32', got {self.klabel_dtype!r}",
        )
        _require(
            self.block_t in SUPPORTED_BLOCK_T,
            f"block_t must be one of {SUPPORTED_BLOCK_T}, got {self.block_t}",
        )
        _require(
            self.k_block in SUPPORTED_K_BLOCK,
            f"k_block must be one of {SUPPORTED_K_BLOCK}, got {self.k_block}",
        )
        _require(
            self.scratch_max_bs >= 1,
            f"scratch_max_bs must be >= 1, got {self.scratch_max_bs}",
        )
        # Imported lazily to avoid a top-level cycle with selector_backends,
        # which imports kernels from layers/attention/triton_ops.
        from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
            FLASHINFER_TOPK_MAX,
            SUPPORTED_SELECTOR_BACKENDS,
        )

        _require(
            self.selector_backend in SUPPORTED_SELECTOR_BACKENDS,
            f"selector_backend must be one of {SUPPORTED_SELECTOR_BACKENDS}, "
            f"got {self.selector_backend!r}",
        )
        _require(
            self.selector_backend != "flashinfer_topk_page_table"
            or self.token_budget <= FLASHINFER_TOPK_MAX,
            f"selector_backend='flashinfer_topk_page_table' requires "
            f"token_budget <= {FLASHINFER_TOPK_MAX} (FlashInfer's "
            f"top_k_page_table_transform errors above that ceiling); "
            f"got token_budget={self.token_budget}. Lower the budget or "
            f"use selector_backend='torch'.",
        )

    def warn_capacity(
        self,
        max_seq_per_req: int,
        num_kv_heads_local: int,
        merge_safe_threshold: int = DEFAULT_MERGE_SAFE_THRESHOLD,
        union_safe_threshold: int = DEFAULT_UNION_SAFE_THRESHOLD,
    ) -> List[str]:
        """Return human-readable warnings for kernel-capacity overruns.

        Stage-2 merge: a single program sorts `num_blocks * k_block`
        candidates. Above `merge_safe_threshold` the implementer must enable
        the chunked merge path (or the user lowers k_block / raises block_t).

        Union: a single program processes `num_kv_heads_local * effective_budget`
        + recent + sink candidates. Same shape; same threshold.

        These are warnings, not hard errors — the implementer wires the
        chunked paths. We surface the thresholds at startup so misconfigs
        are visible.
        """
        warnings: List[str] = []
        num_blocks = (max_seq_per_req + self.block_t - 1) // self.block_t
        merge_candidates = num_blocks * self.k_block
        if merge_candidates > merge_safe_threshold:
            warnings.append(
                f"DS stage-2 merge candidates = num_blocks * k_block = "
                f"{num_blocks} * {self.k_block} = {merge_candidates} > "
                f"merge_safe_threshold={merge_safe_threshold}. The chunked "
                f"merge path must be enabled, or lower k_block / raise block_t."
            )
        effective_budget = min(self.token_budget, num_blocks * self.k_block)
        union_candidates = (
            num_kv_heads_local * effective_budget
            + self.recent_tokens
            + self.sink_tokens
        )
        if union_candidates > union_safe_threshold:
            warnings.append(
                f"DS union candidates = H_kv_local * effective_budget + "
                f"recent + sink = {num_kv_heads_local} * {effective_budget} + "
                f"{self.recent_tokens} + {self.sink_tokens} = "
                f"{union_candidates} > union_safe_threshold={union_safe_threshold}. "
                f"The chunked union path must be enabled, or lower token_budget."
            )
        return warnings


def build_runtime_config(server_args) -> DoubleSparsityRuntimeConfig:
    # Worst-case decode batch size for scratch sizing. Prefer the explicit
    # admission cap; fall back to the captured-graph cap; floor at 1.
    scratch_max_bs = (
        getattr(server_args, "max_running_requests", None)
        or getattr(server_args, "cuda_graph_max_bs", None)
        or 1
    )
    cfg = DoubleSparsityRuntimeConfig(
        heavy_channels=server_args.double_sparsity_heavy_channels,
        token_budget=server_args.double_sparsity_token_budget,
        recent_tokens=server_args.double_sparsity_recent_tokens,
        sink_tokens=server_args.double_sparsity_sink_tokens,
        min_seq_len=server_args.double_sparsity_min_seq_len,
        max_selected_per_request=server_args.double_sparsity_max_selected_per_request,
        gqa_reduction=server_args.double_sparsity_gqa_reduction,
        klabel_dtype=server_args.double_sparsity_klabel_dtype,
        block_t=server_args.double_sparsity_block_t,
        k_block=server_args.double_sparsity_k_block,
        scratch_max_bs=int(scratch_max_bs),
        selector_backend=getattr(
            server_args, "double_sparsity_selector_backend", "torch"
        ),
    )
    cfg.validate()
    return cfg


def torch_dtype_for_klabel(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported klabel_dtype: {name!r}")


def gqa_reduction_id(name: str) -> int:
    """Stable int code for the Triton kernel constexpr."""
    return {"max_abs": 0, "mean": 1, "soq": 2}[name]


__all__ = [
    "DoubleSparsityCalibration",
    "DoubleSparsityRuntimeConfig",
    "SCHEMA_VERSION",
    "SUPPORTED_CHANNEL_TYPES",
    "SUPPORTED_GQA_REDUCTIONS",
    "SUPPORTED_INDEXING",
    "build_runtime_config",
    "channel_indices_for_runtime",
    "gqa_reduction_id",
    "parse_calibration_dict",
    "parse_calibration_file",
    "slice_for_tp",
    "torch_dtype_for_klabel",
    "validate_against_model",
]
