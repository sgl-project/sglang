# SPDX-License-Identifier: Apache-2.0
"""Driver for the EP-to-TP weight transform used by ``load_tp_by_experts``.

With ``--model-loader-extra-config '{"load_tp_by_experts": true}'`` a pure-TP
run loads MoE weights as if it were running EP, so every rank reads a disjoint
set of whole experts instead of sharded slices of every expert. That is a much
friendlier disk IO pattern. Once loading finishes, each layer is redistributed
back into the normal TP layout by :meth:`FusedMoE.ep_to_tp_transform`.

This lives as a plain module-level function rather than on a weight-loader
mixin so that models which do not share a weight-loader base class (e.g. Kimi
K3) can drive the same transform.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import orjson
import torch

from sglang.srt.runtime_context import get_server_args

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)

# Architectures whose load_weights() drives maybe_ep_to_tp_transform_all_layers().
# load_tp_by_experts on anything else would leave the MoE layers stuck in the
# EP-load state, so ServerArgs rejects it up front rather than failing at the
# first forward. Keep in sync with the call sites of the transform.
EP_TO_TP_SUPPORTED_ARCHS = frozenset(
    {
        # DeepseekV2WeightLoaderMixin.load_weights()
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV3ForCausalLMNextN",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
        "Glm4MoeLiteForCausalLM",
        "MistralLarge3ForCausalLM",
        # KimiK3LinearForCausalLM.load_weights(), reached for the multimodal
        # wrapper too since it delegates to the language model.
        "KimiK3ForConditionalGeneration",
    }
)


def is_load_tp_by_experts_enabled() -> bool:
    """Whether ``--model-loader-extra-config`` requested EP-style TP loading."""
    extra_config = orjson.loads(get_server_args().model_loader_extra_config)
    return extra_config.get("load_tp_by_experts", False)


def maybe_ep_to_tp_transform_all_layers(model: nn.Module) -> None:
    """After EP-style weight loading, transform all MoE layers to TP layout.

    ``model`` is the inner decoder-stack module (the one holding ``layers``),
    not the ``*ForCausalLM`` wrapper. No-op unless ``load_tp_by_experts`` is
    set; layers that never entered the EP-load state are skipped individually,
    so mixed dense/MoE stacks are fine.

    Processes one layer at a time to keep peak memory overhead to ~1/num_layers.
    Reuses intermediate buffers across layers for parameters of the same shape.
    """
    if not is_load_tp_by_experts_enabled():
        return

    walk_started = time.perf_counter()

    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    buf_cache: dict[tuple[torch.Size, torch.dtype], torch.Tensor] = {}

    if hasattr(model, "layers"):
        layers = model.layers[model.start_layer : model.end_layer]
    elif hasattr(model, "decoder"):
        layers = [model.decoder]
    else:
        raise ValueError("Internal error: model has no layers or decoder attribute")

    transformed = 0
    for layer in layers:
        mlp = layer.mlp

        # Find the FusedMoE module (either mlp itself or mlp.experts)
        moe_layer = None
        if isinstance(mlp, FusedMoE):
            moe_layer = mlp
        elif hasattr(mlp, "experts") and isinstance(mlp.experts, FusedMoE):
            moe_layer = mlp.experts

        # FusedMoE always sets _ep_load_for_tp in __init__, so a plain read is
        # safe once the isinstance check above has narrowed the type.
        if moe_layer is None or not moe_layer._ep_load_for_tp:
            continue

        # Allocate or reuse buffers for each parameter. The layer owns the
        # target-shape rule because the shard dim is not a plain //tp_size once
        # the backend pads each partition; see ep_to_tp_target_shape().
        # Params sharing a (shape, dtype) within one layer must still get
        # distinct buffers — they are all live simultaneously — so the slot
        # index disambiguates. Across layers the same slots are reused.
        claimed: dict[tuple, int] = {}
        for name, param in moe_layer.named_parameters():
            target_shape = moe_layer.ep_to_tp_target_shape(name, param)

            shape_key = (target_shape, param.data.dtype)
            slot = claimed.get(shape_key, 0)
            claimed[shape_key] = slot + 1
            cache_key = shape_key + (slot,)

            if cache_key not in buf_cache:
                # Zero-filled: alignment padding beyond the real per-rank width
                # is never written by the transform and must read as zero.
                buf_cache[cache_key] = torch.zeros(
                    target_shape, dtype=param.data.dtype, device=param.data.device
                )
            else:
                # Reused across layers — clear the previous layer's padding.
                buf_cache[cache_key].zero_()

            param._ep_to_tp_buf = buf_cache[cache_key]

        started = time.perf_counter()
        moe_layer.ep_to_tp_transform()
        transformed += 1
        logger.debug(
            "EP-to-TP transformed layer %d in %.2fs",
            moe_layer.layer_id,
            time.perf_counter() - started,
        )

    # Free cached buffers
    buf_cache.clear()

    # Report the count, not just completion: every layer can legitimately be
    # skipped (none in the EP-load state), and a bare "complete" line would make
    # that silent no-op indistinguishable from real work.
    logger.info(
        "EP-to-TP weight transformation complete: %d/%d layer(s) transformed in %.1fs",
        transformed,
        len(layers),
        time.perf_counter() - walk_started,
    )
    if transformed == 0:
        logger.warning(
            "load_tp_by_experts is set but no MoE layer was in the EP-load "
            "state; weights were loaded as plain TP and the transform was a no-op."
        )
