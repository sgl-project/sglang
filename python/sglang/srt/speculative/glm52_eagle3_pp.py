"""GLM-5.2 EAGLE-3 auxiliary hidden-state propagation across PP stages.

GLM-5.2 EAGLE-3 target models capture multiple intermediate layer features
("auxiliary hidden states") and feed them to the draft model.  Under pipeline
parallelism, these capture layers are split across PP stages.

This module is GLM-5.2-specific. It provides:

* A deterministic **slot mapping** from global layer IDs to contiguous slots.
* **Packed buffer** creation / validation / pack / unpack utilities.
* **Static ownership validation** (no zero-tensor detection at runtime).
* A single ``PPProxyTensors`` key carrying a
  ``[num_token_rows, num_capture_layers, hidden_size]`` tensor.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# PP proxy key for the packed GLM-5.2 EAGLE-3 auxiliary hidden states.
GLM52_EAGLE3_AUX_PP_KEY = "glm52_eagle3_aux_hidden_states"

# Required PP proxy keys: their absence is always fatal (stale data risk).
REQUIRED_PP_PROXY_KEYS = frozenset({
    "hidden_states",
    "residual",
})

# Conditionally optional PP proxy keys: may be absent in specific paths.
OPTIONAL_PP_PROXY_KEYS = frozenset({
    "topk_indices",  # Only when DSA path doesn't require it.
})

# EAGLE3 aux is required on PP1 when remote capture layers exist (owned by PP0).
# It is optional on PP0 (PP0 produces it, not consumes it).


def classify_pp_proxy_key(
    key: str,
    pp_rank: int,
    remote_capture_layers_exist: bool,
) -> bool:
    """Return True if the key is required, False if optional.

    For PP1 (last stage), the EAGLE3 aux key is required when any capture
    layer is owned by an earlier PP stage.
    """
    if key in REQUIRED_PP_PROXY_KEYS:
        return True
    if key == GLM52_EAGLE3_AUX_PP_KEY:
        # Required on PP1 only when remote capture layers exist.
        return pp_rank > 0 and remote_capture_layers_exist
    if key in OPTIONAL_PP_PROXY_KEYS:
        return False
    # Unknown keys default to optional (forward-compatible).
    return False


def validate_pp_proxy_keys(
    available_keys: list,
    pp_rank: int,
    tp_rank: int,
    forward_mode: str,
    active_token_rows: int,
    remote_capture_layers_exist: bool,
    slot_ownership: Optional[Dict[int, int]] = None,
) -> None:
    """Validate that all required PP proxy keys are present.

    Raises RuntimeError with rich diagnostics if a required key is missing.
    """
    available_set = set(available_keys)

    for key in list(REQUIRED_PP_PROXY_KEYS):
        if key not in available_set:
            raise RuntimeError(
                f"[GLM52-E3-PP] Missing required PP proxy key '{key}' on "
                f"pp_rank={pp_rank}, tp_rank={tp_rank}, "
                f"forward_mode={forward_mode}, "
                f"active_token_rows={active_token_rows}. "
                f"Available keys: {sorted(available_set)}"
            )

    # Check EAGLE3 aux on PP1 when remote capture layers exist.
    if (
        pp_rank > 0
        and remote_capture_layers_exist
        and GLM52_EAGLE3_AUX_PP_KEY not in available_set
    ):
        remote_owners = {}
        if slot_ownership:
            remote_owners = {
                lid: owner
                for lid, owner in slot_ownership.items()
                if owner < pp_rank
            }
        raise RuntimeError(
            f"[GLM52-E3-PP] Missing required remote EAGLE3 aux key "
            f"'{GLM52_EAGLE3_AUX_PP_KEY}' on pp_rank={pp_rank}, "
            f"tp_rank={tp_rank}, forward_mode={forward_mode}, "
            f"active_token_rows={active_token_rows}. "
            f"remote_capture_layers={sorted(remote_owners.keys())}, "
            f"available_proxy_keys={sorted(available_set)}. "
            f"PP1 must not silently continue without required remote "
            f"EAGLE3 aux state."
        )


def build_layer_to_slot_map(
    global_capture_layers: List[int],
) -> Dict[int, int]:
    """Map each global capture layer ID to its slot index in the packed buffer.

    The slot ordering follows the globally sorted capture-layer list, so that
    ``packed[:, slot, :]`` is the feature for ``global_capture_layers[slot]``.
    """
    return {layer_id: slot for slot, layer_id in enumerate(global_capture_layers)}


def get_local_capture_layers(
    global_capture_layers: List[int],
    start_layer: int,
    end_layer: int,
) -> List[int]:
    """Return the subset of global capture layers owned by one PP stage.

    A layer ``i`` belongs to stage ``[start_layer, end_layer)`` when
    ``start_layer <= i < end_layer``.
    """
    return [
        layer_id
        for layer_id in global_capture_layers
        if start_layer <= layer_id < end_layer
    ]


def build_slot_ownership_map(
    global_capture_layers: List[int],
    pp_size: int,
    num_hidden_layers: int,
) -> Dict[int, int]:
    """Build a static mapping: capture_layer_id -> owning PP rank.

    Uses the same layer partition logic as get_pp_indices to determine
    which PP rank owns each layer. Uses only ``SGLANG_PP_LAYER_PARTITION``
    as the source of truth for partition overrides.

    Raises ValueError if any capture layer has no owner.
    """
    import os

    # P1-1: Single source of truth for PP layer partitioning.
    # SGLANG_PP_LAYER_PARTITION is the only env var for partition overrides.
    partition_list_str = os.getenv("SGLANG_PP_LAYER_PARTITION", None)
    if partition_list_str is not None:
        partitions = [int(x) for x in partition_list_str.split(",")]
        assert len(partitions) == pp_size, (
            f"SGLANG_PP_LAYER_PARTITION has {len(partitions)} entries, "
            f"but pp_size={pp_size}"
        )
        assert sum(partitions) == num_hidden_layers, (
            f"SGLANG_PP_LAYER_PARTITION sums to {sum(partitions)}, "
            f"but num_hidden_layers={num_hidden_layers}"
        )
    else:
        base = num_hidden_layers // pp_size
        remainder = num_hidden_layers % pp_size
        partitions = []
        for rank in range(pp_size):
            if rank >= pp_size - remainder:
                partitions.append(base + 1)
            else:
                partitions.append(base)

    # Build layer -> pp_rank mapping
    layer_to_rank: Dict[int, int] = {}
    start = 0
    for rank, count in enumerate(partitions):
        for i in range(start, start + count):
            layer_to_rank[i] = rank
        start += count

    # Build capture_layer -> owning rank
    ownership: Dict[int, int] = {}
    for layer_id in global_capture_layers:
        if layer_id not in layer_to_rank:
            raise ValueError(
                f"GLM-5.2 EAGLE-3 PP: capture layer {layer_id} has no PP owner. "
                f"num_hidden_layers={num_hidden_layers}, pp_size={pp_size}"
            )
        ownership[layer_id] = layer_to_rank[layer_id]
    return ownership


def allocate_packed_aux_buffer(
    num_tokens: int,
    num_capture_layers: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Allocate a zero-initialised packed auxiliary-hidden-state buffer.

    Shape: ``[num_tokens, num_capture_layers, hidden_size]``
    """
    return torch.zeros(
        (num_tokens, num_capture_layers, hidden_size),
        dtype=dtype,
        device=device,
    )


def pack_aux_into_buffer(
    packed_aux: torch.Tensor,
    aux_hidden_states: List[torch.Tensor],
    local_capture_layers: List[int],
    layer_to_slot: Dict[int, int],
) -> torch.Tensor:
    """Write locally captured features into their global slots in ``packed_aux``.

    ``packed_aux`` is modified in-place.  Slots not owned by this stage are
    left untouched (preserving values from earlier stages).
    """
    assert len(aux_hidden_states) == len(local_capture_layers), (
        f"Mismatch: {len(aux_hidden_states)} aux tensors vs "
        f"{len(local_capture_layers)} local capture layers"
    )
    for feature, layer_id in zip(aux_hidden_states, local_capture_layers):
        slot = layer_to_slot[layer_id]
        packed_aux[:, slot, :].copy_(feature)
    return packed_aux


def unpack_aux_from_buffer(
    packed_aux: torch.Tensor,
    global_capture_layers: List[int],
    layer_to_slot: Dict[int, int],
    slot_ownership: Dict[int, int],
    local_capture_layers: List[int],
    pp_rank: int,
    pp_size: int,
    available_proxy_keys: Optional[List[str]] = None,
) -> List[torch.Tensor]:
    """Reconstruct the ordered ``List[Tensor]`` expected by the logits processor.

    Uses static ownership validation instead of zero-tensor detection.

    Args:
        packed_aux: ``[num_tokens, num_capture_layers, hidden]`` tensor.
        global_capture_layers: Full sorted list of global capture layer IDs.
        layer_to_slot: ``{layer_id: slot_index}`` mapping.
        slot_ownership: ``{layer_id: owning_pp_rank}`` from build_slot_ownership_map.
        local_capture_layers: Layer IDs captured on *this* stage.
        pp_rank: Current PP rank.
        pp_size: Total PP size.
        available_proxy_keys: Keys present in the received PP proxy (for errors).
    """
    result: List[torch.Tensor] = []
    for layer_id in global_capture_layers:
        slot = layer_to_slot[layer_id]
        feature = packed_aux[:, slot, :]
        result.append(feature)

    # Static ownership validation on the last stage
    if pp_rank == pp_size - 1:
        all_owners = set(slot_ownership.values())
        expected_owners = set(range(pp_size))
        missing_owners = expected_owners - all_owners
        if missing_owners:
            logger.warning(
                "GLM-5.2 EAGLE-3 PP: PP stages %s do not own any capture layer. "
                "This is acceptable if those stages have no capture layers.",
                missing_owners,
            )
        # Verify every capture layer has an owner
        for layer_id in global_capture_layers:
            owner = slot_ownership.get(layer_id)
            if owner is None:
                raise RuntimeError(
                    f"GLM-5.2 EAGLE-3 PP: capture layer {layer_id} has no "
                    f"owning PP stage. slot_ownership={slot_ownership}, "
                    f"pp_rank={pp_rank}, pp_size={pp_size}, "
                    f"available_proxy_keys={available_proxy_keys}"
                )
    return result


def validate_capture_layers(
    global_capture_layers: List[int],
    num_hidden_layers: int,
    pp_size: int,
    start_layer: int,
    end_layer: int,
    hidden_size: int,
) -> Dict[int, int]:
    """Startup validation of GLM-5.2 EAGLE-3 capture-layer configuration.

    Returns the slot ownership map.
    Raises ``ValueError`` on any invalid configuration.
    """
    if len(global_capture_layers) == 0:
        raise ValueError(
            "GLM-5.2 EAGLE-3 PP: global_capture_layers is empty. "
            "EAGLE-3 requires at least one capture layer."
        )

    # P0-1: Capture-layer IDs must be strictly sorted, unique, and in range.
    # Do not silently sort user input — every checkpoint-weight slot and
    # EAGLE-3 draft expectation must match the exact ordering provided.
    seen = set()
    for idx, layer_id in enumerate(global_capture_layers):
        if layer_id < 0 or layer_id >= num_hidden_layers:
            raise ValueError(
                f"GLM-5.2 EAGLE-3 PP: layer ID {layer_id} is out of "
                f"range [0, {num_hidden_layers}). "
                f"Invalid IDs: {global_capture_layers}. "
                f"Constraint: 0 <= layer_id < num_hidden_layers."
            )
        if layer_id in seen:
            raise ValueError(
                f"GLM-5.2 EAGLE-3 PP: duplicate layer ID {layer_id} at "
                f"index {idx}. Capture-layer IDs must be unique. "
                f"Invalid IDs: {global_capture_layers}. "
                f"Constraint: strictly sorted, unique, non-empty."
            )
        seen.add(layer_id)
        if idx > 0 and layer_id <= global_capture_layers[idx - 1]:
            raise ValueError(
                f"GLM-5.2 EAGLE-3 PP: layer ID {layer_id} at index {idx} "
                f"is not strictly sorted (previous={global_capture_layers[idx - 1]}). "
                f"Invalid IDs: {global_capture_layers}. "
                f"Constraint: strictly sorted, unique, non-empty, "
                f"inside the target layer range [0, {num_hidden_layers})."
            )

    if hidden_size <= 0:
        raise ValueError(
            f"GLM-5.2 EAGLE-3 PP: invalid hidden_size={hidden_size}"
        )

    # Build and validate slot ownership
    ownership = build_slot_ownership_map(
        global_capture_layers, pp_size, num_hidden_layers
    )

    # Verify every PP stage that has capture layers owns at least one
    for layer_id, owner_rank in ownership.items():
        logger.debug(
            "GLM-5.2 EAGLE-3 PP: layer %d owned by PP rank %d",
            layer_id, owner_rank,
        )

    return ownership


def log_pp_aux_capture_info(
    global_capture_layers: List[int],
    start_layer: int,
    end_layer: int,
    hidden_size: int,
    pp_rank: int,
    pp_size: int,
    slot_ownership: Optional[Dict[int, int]] = None,
) -> None:
    """Log the PP auxiliary capture configuration once on rank 0."""
    local = [
        lid for lid in global_capture_layers
        if start_layer <= lid < end_layer
    ]
    logger.info(
        "GLM-5.2 EAGLE-3 PP auxiliary capture: "
        "global_layers=%s, "
        "PP%d local_layers=%s, "
        "packed_shape_per_token=[%d, %d], "
        "slot_ownership=%s",
        global_capture_layers,
        pp_rank,
        local,
        len(global_capture_layers),
        hidden_size,
        slot_ownership,
    )


def get_pp_split_layer(num_hidden_layers: int, pp_size: int) -> int:
    """Return the PP0/PP1 boundary layer index.

    P1-1: SGLANG_GLM52_PP_SPLIT is removed. Only SGLANG_PP_LAYER_PARTITION
    is used. This function now derives the split from the standard partition.
    """
    import os

    partition_list_str = os.getenv("SGLANG_PP_LAYER_PARTITION", None)
    if partition_list_str is not None:
        partitions = [int(x) for x in partition_list_str.split(",")]
        if pp_size == 2 and len(partitions) == 2:
            return partitions[0]
        # For other pp_size, return cumulative count for rank 0
        return partitions[0] if partitions else num_hidden_layers // pp_size
    return num_hidden_layers // pp_size


def validate_glm52_eagle3_tp4_pp2_configuration(
    server_args,
    spec_algorithm,
    is_draft_worker: bool,
    pp_rank: int,
    tp_rank: int,
) -> None:
    """Strict fail-fast validation for the only supported production config.

    This branch supports exactly:
      GLM-5.2 target + true EAGLE3 draft
      TP=4, PP=2
      topk=1
      non-overlap scheduler
      no adaptive speculation

    Called during startup after model/spec config is resolved.
    """
    from sglang.srt.environ import envs

    if not envs.SGLANG_ENABLE_PP_SPEC.get():
        return  # PP+spec not enabled; nothing to validate

    if server_args.pp_size <= 1:
        return  # No PP; nothing to validate

    errors = []

    if server_args.pp_size != 2:
        errors.append(
            f"pp_size={server_args.pp_size}, but only PP=2 is supported. "
            "Set --pp-size 2 or unset SGLANG_ENABLE_PP_SPEC."
        )

    if server_args.tp_size != 4:
        errors.append(
            f"tp_size={server_args.tp_size}, but only TP=4 is supported. "
            "Set --tp-size 4 or unset SGLANG_ENABLE_PP_SPEC."
        )

    if not spec_algorithm.is_eagle3():
        errors.append(
            f"speculative_algorithm={spec_algorithm}, but only EAGLE3 is supported. "
            "Use --speculative-algorithm EAGLE3 or unset SGLANG_ENABLE_PP_SPEC."
        )

    if server_args.speculative_eagle_topk != 1:
        errors.append(
            f"speculative_eagle_topk={server_args.speculative_eagle_topk}, "
            "but only topk=1 is supported."
        )

    if not server_args.disable_overlap_schedule:
        errors.append(
            "Overlap schedule is enabled, but GLM-5.2 EAGLE3 TP4xPP2 "
            "requires --disable-overlap-schedule."
        )

    if server_args.speculative_adaptive:
        errors.append(
            "Adaptive speculative decoding is enabled, but it is not "
            "compatible with PP+spec."
        )

    # P0-8: Hard-disable rejection sampling — the PP relay does not relay
    # draft_probs, so rejection sampling cannot work.
    if server_args.speculative_use_rejection_sampling:
        errors.append(
            "speculative_use_rejection_sampling is enabled, but the PP relay "
            "does not transport draft_probs. Rejection sampling is incompatible "
            "with PP+spec. Set --speculative-use-rejection-sampling false or "
            "unset SGLANG_ENABLE_PP_SPEC."
        )

    # P0-8: Reject all unreviewed runtime modes.
    if server_args.enable_disaggregation:
        errors.append(
            "PD/disaggregation is enabled, but it is not reviewed for PP+spec."
        )

    if getattr(server_args, "enable_context_parallel", False):
        errors.append(
            "Context parallelism is enabled, but it is not reviewed for PP+spec."
        )

    if server_args.enable_dp_attention:
        errors.append(
            "DP attention is enabled, but it is not reviewed for PP+spec."
        )

    if server_args.enable_ep_moe:
        errors.append(
            "Expert parallelism (EP) is enabled, but it is not reviewed for PP+spec."
        )

    if server_args.pp_async_batch_depth > 0:
        errors.append(
            f"pp_async_batch_depth={server_args.pp_async_batch_depth}, but only "
            "pp_async_batch_depth=0 is supported for PP+spec."
        )

    if server_args.speculative_token_map is not None:
        errors.append(
            "Token-map speculation is enabled, but it is not reviewed for PP+spec."
        )

    # Validate the topk=1 chain invariant
    if (
        server_args.speculative_num_steps is not None
        and server_args.speculative_num_draft_tokens is not None
    ):
        if server_args.speculative_num_draft_tokens != server_args.speculative_num_steps + 1:
            errors.append(
                f"speculative_num_draft_tokens={server_args.speculative_num_draft_tokens} "
                f"!= speculative_num_steps+1={server_args.speculative_num_steps + 1}. "
                "topk=1 requires num_draft_tokens == num_steps + 1."
            )

    # Require a separate draft model path (not MTP/NextN)
    if not server_args.speculative_draft_model_path:
        errors.append(
            "speculative_draft_model_path is not set. "
            "A separate trained EAGLE3 draft checkpoint is required. "
            "GLM-5.2 MTP/NextN is not supported under PP+spec."
        )

    if errors:
        msg = "\n".join(
            [
                "GLM-5.2 EAGLE3 TP4xPP2 configuration validation failed:",
                "",
            ]
            + [f"  - {e}" for e in errors]
            + [
                "",
                "Supported configuration:",
                "  --tp-size 4 --pp-size 2",
                "  --speculative-algorithm EAGLE3",
                "  --speculative-eagle-topk 1",
                "  --speculative-draft-model-path <path>",
                "  --disable-overlap-schedule",
                "  (no --speculative-adaptive)",
                "  (no --speculative-use-rejection-sampling)",
                "  (no --enable-disaggregation)",
                "  (no --enable-context-parallel)",
                "  (no --enable-dp-attention)",
                "  (no --enable-ep-moe)",
                "  pp_async_batch_depth=0",
                "  SGLANG_ENABLE_PP_SPEC=1",
            ]
        )
        raise ValueError(msg)

    # Log the validated configuration once on rank 0
    if pp_rank == 0 and tp_rank == 0 and not is_draft_worker:
        import os
        pp_partition = os.getenv("SGLANG_PP_LAYER_PARTITION", "default (even)")
        logger.info(
            "[GLM52-EAGLE3-PP] Configuration validated:\n"
            "  tp_size=%d, pp_size=%d\n"
            "  spec_algorithm=%s, is_eagle3=%s\n"
            "  topk=%d, num_steps=%d, num_draft_tokens=%d\n"
            "  overlap=%s, adaptive=%s\n"
            "  draft_model_path=%s\n"
            "  pp_layer_partition=%s",
            server_args.tp_size,
            server_args.pp_size,
            spec_algorithm,
            spec_algorithm.is_eagle3(),
            server_args.speculative_eagle_topk,
            server_args.speculative_num_steps,
            server_args.speculative_num_draft_tokens,
            not server_args.disable_overlap_schedule,
            server_args.speculative_adaptive,
            server_args.speculative_draft_model_path,
            pp_partition,
        )
        print("GLM52_EAGLE3_TP4_PP2_VALIDATED", flush=True)
