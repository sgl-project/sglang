"""Checkpoint layer selection for the opt-in NPU INT8 DSA indexer cache."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

INT8_INDEXER_QUANT_TYPE = "INT8_DYNAMIC"
_FLAG = "--indexer-kv-cache-dtype int8"
GLOBAL_INDEXER_QUANT_TYPE_KEY = "indexer_quant_type"
_INDEXER_QUANT_TYPE_SUFFIX = ".indexer.quant_type"
# Weight schemes never select a cache dtype, but their keys indicate a
# per-layer description: absent INT8 cache entries must then stay dense.
_INDEXER_PER_LAYER_SUFFIXES = (_INDEXER_QUANT_TYPE_SUFFIX, ".indexer.wq_b.weight")
_LAYER_ID_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def parse_modelslim_int8_indexer_layers(
    quant_description: Mapping[str, Any],
) -> Optional[frozenset[int]]:
    """Layers whose indexer cache ModelSlim calibrated for INT8.

    Returns None when the description has no per-layer indexer keys, meaning
    every indexer layer is eligible.
    """
    layer_ids = set()
    has_per_layer_keys = False
    for key, value in quant_description.items():
        if not isinstance(key, str):
            continue
        suffix = next((s for s in _INDEXER_PER_LAYER_SUFFIXES if key.endswith(s)), None)
        if suffix is None:
            continue
        match = _LAYER_ID_RE.search(key)
        if match is None:
            continue
        has_per_layer_keys = True
        if suffix == _INDEXER_QUANT_TYPE_SUFFIX and value == INT8_INDEXER_QUANT_TYPE:
            layer_ids.add(int(match.group(1)))
    return frozenset(layer_ids) if has_per_layer_keys else None


def resolve_npu_int8_indexer_layer_ids(
    *,
    is_npu: bool,
    index_head_dim: Optional[int],
    kv_cache_is_fp8: bool,
    enable_hierarchical_cache: bool,
    enable_hisparse: bool,
    enable_sparse_kv_offload: bool,
    candidate_layer_ids: tuple[int, ...],
    modelslim_quant_description: Optional[Mapping[str, Any]],
    enable_speculative_decoding: bool = False,
    enable_prefill_cp: bool = False,
    enable_pp_disaggregation: bool = False,
    enable_decode_kv_offload: bool = False,
    enable_decode_host_backup: bool = False,
) -> frozenset[int]:
    """Resolve INT8 indexer layers after the caller checks the CLI option.

    ``candidate_layer_ids`` are the indexer layers the pool allocates on this
    rank. Speculative decoding, including MTP, is not supported yet.
    """
    if not is_npu:
        raise ValueError(f"{_FLAG} is only supported on Ascend NPU.")
    if index_head_dim is None:
        raise ValueError(f"{_FLAG} requires a DSA model.")
    if index_head_dim != 128:
        raise ValueError(
            f"{_FLAG} requires index_head_dim == 128, got {index_head_dim}."
        )
    # TODO: Support MTP by mapping draft pool IDs to checkpoint layer IDs and
    # accounting for the draft cache layout in memory sizing and transfers.
    unsupported = [
        name
        for name, enabled in (
            ("speculative decoding (including MTP)", enable_speculative_decoding),
            ("an FP8 KV cache", kv_cache_is_fp8),
            ("prefill context parallelism", enable_prefill_cp),
            ("pipeline-parallel PD disaggregation", enable_pp_disaggregation),
            ("--enable-hierarchical-cache", enable_hierarchical_cache),
            (
                "--disaggregation-decode-enable-offload-kvcache",
                enable_decode_kv_offload,
            ),
            (
                "--disaggregation-decode-retraction-backup=host_pool",
                enable_decode_host_backup,
            ),
            ("--enable-hisparse", enable_hisparse),
            ("SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD", enable_sparse_kv_offload),
        )
        if enabled
    ]
    if unsupported:
        raise ValueError(
            f"{_FLAG} is not supported together with " + ", ".join(unsupported) + "."
        )

    if modelslim_quant_description is None:
        logger.warning(
            "%s is set for a model without a ModelSlim quantization "
            "description. Every indexer layer is quantized without "
            "calibration; accuracy may drop.",
            _FLAG,
        )
        return frozenset(candidate_layer_ids)

    selected = parse_modelslim_int8_indexer_layers(modelslim_quant_description)
    global_quant_type = modelslim_quant_description.get(
        GLOBAL_INDEXER_QUANT_TYPE_KEY, ""
    )
    if selected is None:
        if global_quant_type != INT8_INDEXER_QUANT_TYPE:
            logger.warning(
                "%s is set, but this checkpoint describes no indexer "
                "quantization (%s=%r, no per-layer indexer keys). Every "
                "indexer layer is quantized without calibration; accuracy "
                "may drop.",
                _FLAG,
                GLOBAL_INDEXER_QUANT_TYPE_KEY,
                global_quant_type,
            )
        return frozenset(candidate_layer_ids)
    if not selected:
        # Checked against the whole filter, not this rank's candidates: with
        # pipeline parallelism a rank may legitimately own no INT8 layer.
        raise ValueError(
            f"{_FLAG} is set, but this checkpoint marks no indexer layer "
            f"{INT8_INDEXER_QUANT_TYPE} "
            f"({GLOBAL_INDEXER_QUANT_TYPE_KEY}={global_quant_type!r}). Remove "
            f"{_FLAG}, or use a checkpoint calibrated for an INT8 indexer cache."
        )
    return frozenset(i for i in candidate_layer_ids if i in selected)
