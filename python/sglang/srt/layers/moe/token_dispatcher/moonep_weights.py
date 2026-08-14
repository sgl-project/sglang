from __future__ import annotations

import logging
import math
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Sub-ranges of the pool, keyed by the attribute they end up backing. The
# scales are named apart from the parameters because what lives here is the
# post-transform runtime layout, not the checkpoint layout the loader fills.
W13_WEIGHT = "w13_weight"
W2_WEIGHT = "w2_weight"
W13_SCALE = "w13_weight_scale_runtime"
W2_SCALE = "w2_weight_scale_runtime"

_pool: Optional[MoonEPWeightPool] = None


class MoonEPWeightPool:
    """Process-global symmetric ranges, sliced one block per MoE layer."""

    def __init__(
        self,
        num_layers: int,
        num_local_experts: int,
        num_prefetch_slots: int,
        ep_rank: int,
        ep_size: int,
        group,
        specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
    ):
        from moonep.buffer import create_nvl_dist_tensor

        self.num_layers = num_layers
        self.num_local_experts = num_local_experts
        self.num_prefetch_slots = num_prefetch_slots
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.block_rows = num_local_experts + num_prefetch_slots
        self.chunk_rows = _resolve_chunk_rows(specs, num_layers * self.block_rows)
        self._layers: dict[int, int] = {}

        self.ranges = {
            kind: create_nvl_dist_tensor(
                [self.chunk_rows, *trailing],
                dtype,
                ep_rank,
                ep_size,
                group=group,
                # Rotated so this rank's chunk is at the base of the mapping.
                # DeepGEMM's tvm_ffi bindings resolve a tensor's device from
                # its base pointer, and the unrotated base is always rank 0's
                # memory -- every other rank then fails the device check.
                local_first=True,
            )
            for kind, (trailing, dtype) in specs.items()
        }
        resident = sum(t.numel() * t.element_size() for t in self.ranges.values())
        logger.info(
            "MoonEP: symmetric expert pool for %d layers, %d local experts + %d "
            "prefetch slots each, chunk_rows=%d (%.1f GB resident per rank)",
            num_layers,
            num_local_experts,
            num_prefetch_slots,
            self.chunk_rows,
            resident / ep_size / 1e9,
        )

    def layer_offset(self, layer_id: int) -> int:
        """Rows before this layer's block. Layers are numbered by the order
        they ask for storage, not by ``layer_id``: with pipeline parallelism a
        rank holds an arbitrary slice of the model's layer ids, and the pool is
        sized for what this rank actually builds."""
        index = self._layers.get(layer_id)
        if index is None:
            index = len(self._layers)
            if index >= self.num_layers:
                raise RuntimeError(
                    f"MoonEP expert pool was sized for {self.num_layers} layers "
                    f"but layer {layer_id} is the {index + 1}th to ask for "
                    "storage; the layer count derived from the model config is "
                    "wrong for this model"
                )
            self._layers[layer_id] = index
        return index * self.block_rows

    def chunk_start(self, owner_rank: int) -> int:
        """First row of ``owner_rank``'s chunk in this rank's mapping."""
        from moonep.buffer import local_first_chunk_index

        index = local_first_chunk_index(owner_rank, self.ep_rank, self.ep_size)
        return index * self.chunk_rows

    def local_view(self, kind: str, layer_id: int) -> torch.Tensor:
        """This rank's ``[num_local_experts, ...]`` slice: what the loader
        writes and what the parameter is bound to."""
        start = self.chunk_start(self.ep_rank) + self.layer_offset(layer_id)
        return self.ranges[kind][start : start + self.num_local_experts]

    def slot_view(self, kind: str, layer_id: int) -> torch.Tensor:
        """This layer's ``[num_prefetch_slots, ...]`` copy destinations."""
        start = (
            self.chunk_start(self.ep_rank)
            + self.layer_offset(layer_id)
            + self.num_local_experts
        )
        return self.ranges[kind][start : start + self.num_prefetch_slots]


def _minimum_aligned_rows(trailing_shape, dtype: torch.dtype) -> int:
    """Smallest row count whose bytes land on a VMM granularity boundary."""
    from moonep.buffer import pad_dim0_for_alignment

    return pad_dim0_for_alignment([1, *trailing_shape], dtype)


def _resolve_chunk_rows(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]], rows_in_use: int
) -> int:
    """One row count that is granularity-aligned for every range at once.

    A weight and its scale must share a row indexing, so they cannot each pick
    their own padding. A row count works for a tensor when
    ``rows * bytes_per_row`` is a multiple of the VMM granularity, so the
    common answer is the least common multiple of each one's minimum, rounded
    up past the rows actually in use.
    """
    step = 1
    for trailing, dtype in specs.values():
        step = math.lcm(step, _minimum_aligned_rows(trailing, dtype))
    return math.ceil(rows_in_use / step) * step


def _num_moe_layers() -> int:
    """How many layers will ask the pool for storage.

    Counted from the config rather than tracked dynamically because the ranges
    are fixed-size VMM mappings that cannot grow; ``layer_offset`` raises if
    the count turns out to be too small.
    """
    from sglang.srt.runtime_context import process_model_config

    config = process_model_config()
    num_layers = int(config.num_hidden_layers)
    first_dense = config.first_k_dense_replace or 0
    freq = getattr(config.hf_text_config, "moe_layer_freq", 1) or 1
    return sum(1 for i in range(num_layers) if i >= first_dense and i % freq == 0)


def get_pool() -> Optional[MoonEPWeightPool]:
    return _pool


def alloc_expert_tensors(
    layer: torch.nn.Module,
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> dict[str, torch.Tensor]:
    """Per-layer views of the symmetric pool, creating it on the first call.

    ``specs`` maps a sub-range name to ``(trailing_shape, dtype)`` for a single
    expert row. Every layer must pass the same specs -- they share one
    allocation. Collective over the EP group, so all ranks have to build their
    layers in the same order.
    """
    global _pool

    from sglang.srt.distributed import get_tp_group
    from sglang.srt.layers.moe.token_dispatcher.moonep import (
        get_moonep_num_prefetch_slots,
    )

    if _pool is None:
        group = get_tp_group().device_group
        ep_size = torch.distributed.get_world_size(group)
        _pool = MoonEPWeightPool(
            num_layers=_num_moe_layers(),
            num_local_experts=int(layer.num_local_experts),
            num_prefetch_slots=get_moonep_num_prefetch_slots(
                int(layer.num_experts), ep_size
            ),
            ep_rank=torch.distributed.get_rank(group),
            ep_size=ep_size,
            group=group,
            specs=specs,
        )
    elif set(specs) != set(_pool.ranges):
        raise NotImplementedError(
            "MoonEP places every layer's experts in one pool, so all layers "
            f"must request the same tensors; got {sorted(specs)} after "
            f"{sorted(_pool.ranges)}"
        )

    return {kind: _pool.local_view(kind, layer.layer_id) for kind in specs}


def expert_rows(layer_id: int, expert_ids: torch.Tensor) -> torch.Tensor:
    """Global expert ids -> rows of the symmetric range. Negative ids (unused
    prefetch slots) pass through so DeepGEMM still skips them."""
    assert _pool is not None, "MoonEP expert pool was never created"
    epn = _pool.num_local_experts
    # The mapping is local-first, so an owner's chunk index is relative to this
    # rank -- row numbers differ per rank, which is fine because every consumer
    # of them (m_indices, experts_to_copy) is computed locally.
    owner = expert_ids // epn
    chunk = (owner - _pool.ep_rank) % _pool.ep_size
    rows = chunk * _pool.chunk_rows + _pool.layer_offset(layer_id) + expert_ids % epn
    return torch.where(expert_ids < 0, expert_ids, rows).to(torch.int32)


def group_rows(
    layer_id: int, expert_ids: torch.Tensor, num_global_experts: int
) -> torch.Tensor:
    """MoonEP's per-VM-group expert ids -> rows of the symmetric range.

    The first ``num_global_experts`` groups are experts addressed by global id;
    the groups after them are this rank's prefetch slots, whose ids name the
    *source* expert but whose tokens must read the slot the copy landed in.
    Empty and unfilled groups keep their -1 and stay skipped.
    """
    assert _pool is not None, "MoonEP expert pool was never created"
    rows = expert_rows(layer_id, expert_ids)
    tail = expert_ids[num_global_experts:]
    slot_base = (
        _pool.chunk_start(_pool.ep_rank)
        + _pool.layer_offset(layer_id)
        + _pool.num_local_experts
    )
    slots = torch.arange(
        slot_base,
        slot_base + tail.numel(),
        device=expert_ids.device,
        dtype=torch.int32,
    )
    rows[num_global_experts:] = torch.where(tail < 0, tail.to(torch.int32), slots)
    return rows


def prefetch_pairs(
    layer_id: int,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]], list[tuple[torch.Tensor, torch.Tensor]]
]:
    """``(weight_pairs, scale_pairs)`` for ``Buffer.prefetch_weight``.

    Scales are re-tiled by the copy and so are kept apart from the weights.
    The scale views are the *storage* orientation, which is what a byte copy
    has to move -- see the MN-major note in the module docstring.
    """
    assert _pool is not None, "MoonEP expert pool was never created"
    weights = [
        (_pool.ranges[k], _pool.slot_view(k, layer_id)) for k in (W13_WEIGHT, W2_WEIGHT)
    ]
    scales = [
        (_pool.ranges[k], _pool.slot_view(k, layer_id))
        for k in (W13_SCALE, W2_SCALE)
        if k in _pool.ranges
    ]
    return weights, scales


def assert_resident(layer: torch.nn.Module, kind: str, tensor: torch.Tensor) -> None:
    """Fail loudly if a post-load step swapped a pooled tensor for a private one.

    SGLang's quant methods routinely rebind weights after loading, and a
    rebind that lands outside the pool silently costs MoonEP its remote
    readability -- prefetch would then copy from memory no peer can see.
    """
    assert _pool is not None, "MoonEP expert pool was never created"
    full = _pool.ranges[kind]
    start = full.data_ptr()
    end = start + full.numel() * full.element_size()
    if not (start <= tensor.data_ptr() < end):
        raise RuntimeError(
            f"MoonEP: layer {layer.layer_id} {kind} left the symmetric pool "
            "after loading. Some post-load step replaced the tensor instead of "
            "writing into it, which would make remote prefetch read private "
            "memory."
        )
