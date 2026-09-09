"""Qwen3.5 GDN ``in_proj_qkvz`` + ``in_proj_ba`` as one GEMM, on ROCm.

``in_proj_ba`` is a tiny projection reading the same hidden states as
``in_proj_qkvz``, so both can be issued as a single wider
``MergedColumnParallelLinear``: one GEMM launch and, on a quantized
checkpoint, one activation quant instead of two. :func:`project` runs it and
hands back the ``qkvz`` and ``ba`` halves the caller expects.

:func:`build` returns ``None`` whenever the merge does not apply -- off by
default, non-HIP, under LoRA, or on a checkpoint whose four shards disagree on
a quantization scheme (which :data:`PACKED_MODULES_MAPPING` lets quark catch)
-- and the caller then builds ``in_proj_qkvz`` and ``in_proj_ba`` as before.

Everything the merge needs of ``Qwen3_5GatedDeltaNet`` lives here rather than
in ``qwen3_5``, so that shared model file only gates on :data:`ENABLED` and
delegates: :func:`build`, :func:`claim_input_proj`, :func:`project`,
:func:`qkvz_proj`, :func:`stacked_params_mapping` and :func:`owns_input_proj`
each replace a block that would otherwise sit inline there.

``Qwen3_5GatedDeltaNet.finalize_fused_in_proj`` merges the same two projections
by concatenating their weights once ``load_weights`` has run, and needs no
guard against this module: it returns on ``not _is_cuda``, and ``is_cuda()``
wants ``torch.version.cuda`` where :data:`IS_HIP` wants ``torch.version.hip``,
which a wheel has one of. The designs differ because the platforms do -- that
one rewrites weights the separate modules already own, which a quantized
checkpoint will not allow, while this one builds the merged module up front so
the four shards load straight into it.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.runtime_context import get_lora
from sglang.srt.utils import add_prefix, get_bool_env_var, is_cpu, is_hip

logger = logging.getLogger(__name__)

IS_HIP = is_hip()

# A ROCm build can still be told to run the CPU engine, which has its own
# fused input projection over in_proj_qkvz/in_proj_ba -- the two modules
# claim_input_proj releases. Declining here leaves that path exactly as it
# was, instead of teaching it about a merge it has no use for.
ENABLED = (
    get_bool_env_var("SGLANG_GDN_FUSE_QKVZBA", "False") and IS_HIP and not is_cpu()
)

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
    layer: torch.nn.Module,
    quant_config: Optional[QuantizationConfig],
    prefix: str,
) -> Optional[MergedColumnParallelLinear]:
    """``layer``'s merged projection, or ``None`` to keep qkvz and ba separate.

    Reads the geometry off ``layer``, which ``Qwen3_5GatedDeltaNet.__init__``
    has set by the point it builds its projections.
    """
    if not ENABLED or _lora_needs_separate_projections():
        return None

    shards = layer.attn_tp_size or 1
    output_sizes, pad = _padded_output_sizes(
        layer.key_dim, layer.value_dim, layer.num_v_heads, shards
    )
    try:
        merged = MergedColumnParallelLinear(
            input_size=layer.hidden_size,
            output_sizes=output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix(MERGED_PARAM, prefix),
            tp_rank=layer.attn_tp_rank,
            tp_size=layer.attn_tp_size,
        )
    except (ValueError, NotImplementedError) as e:
        logger.info_once(f"in_proj_qkvz and in_proj_ba kept separate: {e}")
        return None

    weight = getattr(merged, "weight", None)
    if pad and weight is not None:
        # No checkpoint tensor reaches these rows and split_output drops them.
        weight.data[-(pad // shards) :].zero_()

    # This rank's share of each half, for split_output. Any alignment padding
    # sits past both and so falls outside the views.
    merged.qkvz_width = sum(output_sizes[:4]) // shards
    merged.ba_width = sum(output_sizes[4:6]) // shards

    # qwen3_5 imports this module, so it can only be read back once loaded --
    # true by now, since layers are built long after both module bodies run.
    # Resolved here rather than in project() to keep it off the forward path.
    from sglang.srt.models.qwen3_5 import _select_fused_ar_input_for_linear

    merged.select_input = _select_fused_ar_input_for_linear
    return merged


def claim_input_proj(layer: torch.nn.Module) -> None:
    """Give ``layer``'s input projection to the merged GEMM, releasing the pair.

    The merged parameter absorbs all four checkpoint shards, so nothing in the
    checkpoint reaches ``in_proj_qkvz``/``in_proj_ba``. Dropping them frees the
    weights they allocated and takes them out of ``named_parameters()``, which
    is what :func:`stacked_params_mapping` reads to pick a mapping.
    """
    layer._bind_packed_weight_loaders(layer.in_proj_qkvzba)
    layer.in_proj_qkvz = layer.in_proj_ba = None


def qkvz_proj(layer: torch.nn.Module) -> torch.nn.Module:
    """Whichever of ``layer``'s modules produces ``qkvz``, merged or separate."""
    merged = getattr(layer, MERGED_PARAM, None)
    return layer.in_proj_qkvz if merged is None else merged


def project(
    layer: torch.nn.Module, hidden_states: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``layer``'s ``qkvz`` and ``ba``, from one GEMM over ``hidden_states``."""
    merged = layer.in_proj_qkvzba
    projected, _ = merged(merged.select_input(hidden_states, merged))
    return split_output(projected, merged.qkvz_width, merged.ba_width)


def owns_input_proj(layer: torch.nn.Module) -> bool:
    """Whether the merged projection serves ``layer``, rather than qkvz and ba.

    Per layer, not per platform: :func:`build` declines a layer whose
    checkpoint shards disagree, leaving the pair in place on a model whose
    other layers merged.
    """
    return getattr(layer, MERGED_PARAM, None) is not None


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
