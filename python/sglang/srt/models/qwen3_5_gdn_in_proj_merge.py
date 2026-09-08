"""Qwen3.5 GDN ``in_proj_qkvz`` + ``in_proj_ba`` as one GEMM, on ROCm.

``in_proj_ba`` is a tiny projection reading the same hidden states as
``in_proj_qkvz``, so both can be issued as a single wider
``MergedColumnParallelLinear``: one GEMM launch and, on a quantized
checkpoint, one activation quant instead of two. :func:`split_output` hands
back the ``qkvz`` and ``ba`` halves the caller expects.

:func:`build` returns ``None`` whenever the merge does not apply -- off by
default, non-HIP, under LoRA, or on a checkpoint whose four shards disagree on
a quantization scheme (which :data:`PACKED_MODULES_MAPPING` lets quark catch)
-- and the caller then builds ``in_proj_qkvz`` and ``in_proj_ba`` as before.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.runtime_context import get_lora
from sglang.srt.utils import get_bool_env_var, is_hip

logger = logging.getLogger(__name__)

IS_HIP = is_hip()
ENABLED = get_bool_env_var("SGLANG_GDN_FUSE_QKVZBA", "False") and IS_HIP

# Pad each rank's N to the aiter a8w8 tile; correctness only needs N % 16 == 0.
_GEMM_N_ALIGN = 128

MERGED_PARAM = "in_proj_qkvzba"

# Absorbed in this order, which fixes the shard ids used below.
CHECKPOINT_SHARDS = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]

PACKED_MODULES_MAPPING = {MERGED_PARAM: CHECKPOINT_SHARDS} if IS_HIP else {}

# (param_name, shard_name, shard_id) triples for load_weights().
_SEPARATE_STACKED_MAPPING = [
    ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
    ("in_proj_qkvz.", "in_proj_z.", 3),
    ("in_proj_ba.", "in_proj_b.", 0),
    ("in_proj_ba.", "in_proj_a.", 1),
]
_MERGED_STACKED_MAPPING = [
    (f"{MERGED_PARAM}.", "in_proj_qkv.", (0, 1, 2)),
    (f"{MERGED_PARAM}.", "in_proj_z.", 3),
    (f"{MERGED_PARAM}.", "in_proj_b.", 4),
    (f"{MERGED_PARAM}.", "in_proj_a.", 5),
]


def stacked_params_mapping(model: torch.nn.Module) -> List[Tuple]:
    """Map the checkpoint's four input-projection tensors onto ``model``.

    ``build`` declines per layer as well as per platform, so the parameters the
    model actually holds decide this, not :data:`ENABLED` alone.
    """
    if not ENABLED:
        return _SEPARATE_STACKED_MAPPING
    if any(f"{MERGED_PARAM}." in name for name, _ in model.named_parameters()):
        return _MERGED_STACKED_MAPPING
    return _SEPARATE_STACKED_MAPPING


def _lora_needs_separate_projections() -> bool:
    # supported_lora_modules names in_proj_qkvz, which the merge removes.
    lora = get_lora()
    return bool(lora.lora_paths) or lora.enable_lora


def _padded_output_sizes(
    key_dim: int, value_dim: int, num_v_heads: int, shards: int
) -> Tuple[List[int], int]:
    """Shard sizes for the merged projection, and total padding rows."""
    output_sizes = [
        key_dim,  # q -> shard 0
        key_dim,  # k -> shard 1
        value_dim,  # v -> shard 2
        value_dim,  # z -> shard 3
        num_v_heads,  # b -> shard 4
        num_v_heads,  # a -> shard 5
    ]
    pad = (-(sum(output_sizes) // shards)) % _GEMM_N_ALIGN * shards
    if pad:
        output_sizes.append(pad)
    return output_sizes, pad


def build(
    hidden_size: int,
    key_dim: int,
    value_dim: int,
    num_v_heads: int,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
    tp_rank: Optional[int] = None,
    tp_size: Optional[int] = None,
) -> Optional[MergedColumnParallelLinear]:
    """The merged projection, or ``None`` to keep ``qkvz`` and ``ba`` separate."""
    if not ENABLED or _lora_needs_separate_projections():
        return None

    shards = tp_size if tp_size is not None else 1
    output_sizes, pad = _padded_output_sizes(key_dim, value_dim, num_v_heads, shards)
    try:
        merged = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
    except (ValueError, NotImplementedError) as e:
        logger.info_once(f"in_proj_qkvz and in_proj_ba kept separate: {e}")
        return None

    weight = getattr(merged, "weight", None)
    if pad and weight is not None:
        # No checkpoint tensor reaches these rows and split_output drops them.
        weight.data[-(pad // shards) :].zero_()
    return merged


def split_output(
    projected: torch.Tensor, qkvz_width: int, ba_width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The ``qkvz`` and ``ba`` halves of a merged projection's output.

    Column views rather than copies, so the alignment padding falls away
    without touching memory. Both carry the merged N as ``stride(0)`` and the
    ``ba`` half a nonzero storage offset, which the fused split/reshape/cat
    kernel and the ``fix_query_key_value_ordering`` fallback both accept.
    """
    return (
        projected[..., :qkvz_width],
        projected[..., qkvz_width : qkvz_width + ba_width],
    )
