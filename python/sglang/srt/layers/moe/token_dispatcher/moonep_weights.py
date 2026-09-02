from __future__ import annotations

import logging
import math
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

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
        _require_moonep_local_first()

        from moonep.buffer import create_nvl_dist_tensor, local_first_chunk_index

        self.num_layers = num_layers
        self.num_local_experts = num_local_experts
        self.num_prefetch_slots = num_prefetch_slots
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.block_rows = num_local_experts + num_prefetch_slots
        self._chunk_index = [
            local_first_chunk_index(owner, ep_rank, ep_size) for owner in range(ep_size)
        ]
        if self._chunk_index != [(o - ep_rank) % ep_size for o in range(ep_size)]:
            raise RuntimeError(
                "MoonEP's local-first rotation is no longer "
                "(owner_rank - local_rank) % world_size, which expert_rows "
                f"assumes; local_first_chunk_index gives {self._chunk_index} "
                f"on rank {ep_rank} of {ep_size}. Update expert_rows to match."
            )
        _check_prefetch_tiling(specs)
        self.chunk_rows = _resolve_chunk_rows(specs, num_layers * self.block_rows)
        self._layers: dict[int, int] = {}

        self.ranges = {
            kind: create_nvl_dist_tensor(
                [self.chunk_rows, *trailing],
                dtype,
                ep_rank,
                ep_size,
                group=group,
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
        return self._chunk_index[owner_rank] * self.chunk_rows

    def local_view(self, kind: str, layer_id: int) -> torch.Tensor:
        start = self.chunk_start(self.ep_rank) + self.layer_offset(layer_id)
        return self.ranges[kind][start : start + self.num_local_experts]

    def slot_view(self, kind: str, layer_id: int) -> torch.Tensor:
        start = (
            self.chunk_start(self.ep_rank)
            + self.layer_offset(layer_id)
            + self.num_local_experts
        )
        return self.ranges[kind][start : start + self.num_prefetch_slots]


_MOONEP_LOCAL_FIRST_BRANCH = (
    "https://github.com/bytedance-iaas/MoonEP/tree/"
    "jxp/update_prefetch_api_and_support_local_first"
)


def _require_moonep_local_first() -> None:
    if envs.SGLANG_ENABLE_MOONEP_LOCAL_FIRST.get():
        return
    raise RuntimeError(
        "MoonEP with quantized experts needs a local-first symmetric mapping "
        "-- create_nvl_dist_tensor(local_first=...) and "
        "local_first_chunk_index() -- which upstream MoonEP does not carry "
        f"yet. Install a MoonEP built from {_MOONEP_LOCAL_FIRST_BRANCH} and "
        "set SGLANG_ENABLE_MOONEP_LOCAL_FIRST=1. BF16 experts are stored on "
        "every rank instead of pooled, and need none of this."
    )


def _check_prefetch_tiling(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> None:
    from moonep.prefetch import prefetch_retile_nbytes

    tile = prefetch_retile_nbytes(1)
    for kind, (trailing, dtype) in specs.items():
        per_expert = math.prod(trailing) * torch.empty(0, dtype=dtype).element_size()
        if per_expert % tile:
            raise ValueError(
                f"MoonEP cannot prefetch {kind} for this model: one expert is "
                f"{per_expert} bytes ({tuple(trailing)} x {dtype}), and the "
                f"prefetch copy needs a multiple of {tile} "
                f"(nearest is {prefetch_retile_nbytes(per_expert)}). This "
                "follows from the model's hidden and intermediate sizes, so "
                "it is not something the server can pad around."
            )


def _minimum_aligned_rows(trailing_shape, dtype: torch.dtype) -> int:
    from moonep.buffer import pad_dim0_for_alignment

    return pad_dim0_for_alignment([1, *trailing_shape], dtype)


def _resolve_chunk_rows(
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]], rows_in_use: int
) -> int:
    step = 1
    for trailing, dtype in specs.values():
        step = math.lcm(step, _minimum_aligned_rows(trailing, dtype))
    return math.ceil(rows_in_use / step) * step


def _num_moe_layers() -> int:
    from sglang.srt.distributed import get_pp_group
    from sglang.srt.distributed.utils import get_pp_indices
    from sglang.srt.runtime_context import process_model_config

    config = process_model_config()
    hf_config = config.hf_text_config

    if hasattr(hf_config, "moe_layer_freq") and hf_config.moe_layer_freq != 1:
        raise NotImplementedError(
            "MoonEP's expert pool assumes every layer from "
            "first_k_dense_replace onwards is an MoE layer, but this model "
            f"sets moe_layer_freq={hf_config.moe_layer_freq!r}. Counting its "
            "MoE layers is a per-model rule and getting it wrong undersizes a "
            "VMM mapping that cannot grow."
        )

    pp_group = get_pp_group()
    start, end = get_pp_indices(
        int(config.num_hidden_layers), pp_group.rank_in_group, pp_group.world_size
    )
    first_moe = max(start, config.first_k_dense_replace or 0)
    return max(0, end - first_moe)


def get_pool() -> Optional[MoonEPWeightPool]:
    return _pool


def alloc_expert_tensors(
    layer: torch.nn.Module,
    specs: dict[str, tuple[tuple[int, ...], torch.dtype]],
) -> dict[str, torch.Tensor]:
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
    assert _pool is not None, "MoonEP expert pool was never created"
    epn = _pool.num_local_experts
    owner = expert_ids // epn
    chunk = (owner - _pool.ep_rank) % _pool.ep_size
    rows = chunk * _pool.chunk_rows + _pool.layer_offset(layer_id) + expert_ids % epn
    return torch.where(expert_ids < 0, expert_ids, rows).to(torch.int32)


def group_rows(
    layer_id: int, expert_ids: torch.Tensor, num_global_experts: int
) -> torch.Tensor:
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


def prefetch_experts(layer_id: int, source_rows: torch.Tensor, num_sms: int) -> None:
    from moonep.prefetch import launch_prefetch, retile_for_prefetch

    assert _pool is not None, "MoonEP expert pool was never created"
    for kind in (W13_WEIGHT, W2_WEIGHT):
        launch_prefetch(
            _pool.ranges[kind],
            _pool.slot_view(kind, layer_id),
            source_rows,
            num_sms=num_sms,
        )
    for kind in (W13_SCALE, W2_SCALE):
        if kind not in _pool.ranges:
            continue
        launch_prefetch(
            retile_for_prefetch(_pool.ranges[kind]),
            retile_for_prefetch(_pool.slot_view(kind, layer_id)),
            source_rows,
            num_sms=num_sms,
        )


def assert_resident(layer: torch.nn.Module, kind: str, tensor: torch.Tensor) -> None:
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
