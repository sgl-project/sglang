"""DWDP for Kimi-K3 KDA projections and replicated shared experts (``--kda-dwdp-size N``).

Each group rank keeps 1/N of every sharded layer's weights; a layer's full weights are
gathered into a rotating slot by copy-engine peer pulls before it runs.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist

from sglang.srt.distributed.utils import get_group_rank_size
from sglang.srt.layers.communication.kda_dwdp.layout import (
    UnitLayout,
    shard_group_layout,
)
from sglang.srt.layers.communication.kda_dwdp.prefetch import SlotPrefetcher
from sglang.srt.layers.communication.kda_dwdp.store import ShardStore
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    eager_on_graph,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# The running layer's slot and the next sharded layer's, gathered while it runs.
NUM_SLOTS = 2

_ONLINE_UPDATE_ERROR = (
    "Online weight updates are not supported with --kda-dwdp-size: the sharded "
    "KDA projections and shared experts live in per-rank shards that are loaded once at "
    "startup and gathered into reused slots. Restart without --kda-dwdp-size to "
    "allow online updates."
)


class DwdpWeights:
    """One weight family: per layer, a merged column-parallel in-proj and a row-parallel out-proj."""

    def __init__(
        self,
        name: str,
        layers: Sequence[int],
        *,
        group,
        group_rank: int,
        group_size: int,
        device: torch.device,
        dummy_load: bool,
    ):
        self.name = name
        self.layers = list(layers)
        if not self.layers or self.layers != sorted(set(self.layers)):
            raise ValueError(f"DWDP {name}: layers must be non-empty and ascending")
        self._unit_index = {l: u for u, l in enumerate(self.layers)}
        self._group, self._group_rank, self._group_size = group, group_rank, group_size
        self._device, self._dummy_load = device, dummy_load
        self._store: Optional[ShardStore] = None
        self._prefetch: Optional[SlotPrefetcher] = None
        self._running: Optional[int] = None

    def bind_pair(self, layer_idx: int, *, owner, in_proj, out_proj) -> None:
        for proj in (in_proj, out_proj):
            if not isinstance(proj.quant_method, UnquantizedLinearMethod):
                raise ValueError(
                    f"--kda-dwdp-size shards unquantized weights only: layer "
                    f"{layer_idx} {self.name} uses {type(proj.quant_method).__name__}"
                )
        u = self._unit_index[layer_idx]
        in_w, out_w = in_proj.weight, out_proj.weight
        layout = UnitLayout(
            [(tuple(in_w.shape), len(in_proj.output_sizes)), (tuple(out_w.shape), 0)],
            self._group_size,
        )
        if self._store is None:
            self._store = ShardStore(
                unit_layers=self.layers,
                slot_numel=layout.slot_numel,
                num_slots=NUM_SLOTS,
                dtype=in_w.dtype,
                device=self._device,
                dummy_load=self._dummy_load,
            )
        store = self._store
        store.add_unit(u, layout)
        for i, w in enumerate((in_w, out_w)):
            if w.dtype != store.dtype:
                raise ValueError(f"DWDP {self.name}: mixed weight dtypes")
            w.data = store.param_view(u, i)
        owner._derived_weight_cache_error = _ONLINE_UPDATE_ERROR

        # Keep this rank's 1/N of its TP slice: the head-aligned (attention-)TP rows of
        # each in-proj block and columns of the out-proj.
        in_rank, _ = get_group_rank_size(in_proj.tp_group)
        out_rank, _ = get_group_rank_size(out_proj.tp_group)
        block_rows = in_proj.output_partition_sizes
        block_start = [sum(block_rows[:b]) for b in range(len(block_rows))]
        out_cols = out_proj.input_size_per_partition

        def load_in_proj(param, loaded_weight, loaded_shard_id):
            b = int(loaded_shard_id)
            rows = block_rows[b]
            store.load(
                u,
                0,
                loaded_weight[in_rank * rows : (in_rank + 1) * rows],
                row_offset=block_start[b],
            )

        def load_out_proj(param, loaded_weight, *args, **kwargs):
            store.load(
                u, 1, loaded_weight[:, out_rank * out_cols : (out_rank + 1) * out_cols]
            )

        in_w.weight_loader = load_in_proj
        out_w.weight_loader = load_out_proj

    def build(self) -> None:
        if self._store is None:
            raise RuntimeError(f"DWDP {self.name}: no layer was bound")
        self._store.map_shards(self._group, self._group_rank)
        self._prefetch = SlotPrefetcher(
            self._store.copy_unit, num_slots=NUM_SLOTS, device=self._device
        )
        shard_b, slot_b = self._store.nbytes()
        es = self._store.slots.element_size()
        replicated_b = es * sum(
            spec.shape[0] * spec.shape[1]
            for layout in self._store.layouts
            for spec in layout.specs
        )
        logger.info(
            "KDA DWDP %s: group rank %d of %d, %d layers: shard %.2f GB + %d slots "
            "%.2f GB per rank instead of %.2f GB replicated",
            self.name,
            self._group_rank,
            self._group_size,
            len(self.layers),
            shard_b / 1e9,
            NUM_SLOTS,
            slot_b / 1e9,
            replicated_b / 1e9,
        )

    def finish_load(self) -> None:
        self._store.finish_load()

    def needs_layer_hook(self, layer_idx: int) -> bool:
        return layer_idx in self._unit_index or layer_idx > self.layers[-1]

    def on_layer_start(self, layer_idx: int) -> None:
        # The previous sharded layer's kernels are all enqueued, so its slot is free.
        if self._running is not None:
            self._prefetch.release(self._running)
            self._running = None
        k = self._unit_index.get(layer_idx)
        if k is None:
            # Past the last sharded layer: start the next forward's first gather.
            self._prefetch.issue(0)
            return
        self._prefetch.wait(k)
        self._running = k
        nxt = (k + 1) % len(self.layers)
        # With an odd layer count the wrap to the first layer lands in this layer's slot.
        if nxt % NUM_SLOTS != k % NUM_SLOTS:
            self._prefetch.issue(nxt)

    def wait_issued(self) -> None:
        self._prefetch.wait_issued()


class KdaDwdp:
    """The DWDP weight families of one Kimi-K3 model, and its per-layer hooks."""

    def __init__(
        self,
        *,
        group_size: int,
        kda_layers: Sequence[int],
        shared_expert_layers: Sequence[int],
        dummy_load: bool,
    ):
        par = get_parallel()
        tp_group = par.tp_group
        group_members, my_group, group_rank = shard_group_layout(
            list(tp_group.ranks), par.attn_tp_size, group_size, tp_group.rank_in_group
        )
        # Every rank creates every subgroup, in the same order (torch.distributed contract).
        groups = [dist.new_group(m, backend="gloo") for m in group_members]
        common = dict(
            group=groups[my_group],
            group_rank=group_rank,
            group_size=group_size,
            device=torch.device("cuda", torch.cuda.current_device()),
            dummy_load=dummy_load,
        )
        self.shared = (
            DwdpWeights("shared experts", shared_expert_layers, **common)
            if shared_expert_layers
            else None
        )
        self.kda = DwdpWeights("KDA", kda_layers, **common)
        self._families: List[DwdpWeights] = [
            w for w in (self.shared, self.kda) if w is not None
        ]

    def build(self) -> None:
        for w in self._families:
            w.build()

    def finish_load(self) -> None:
        for w in self._families:
            w.finish_load()

    def needs_layer_hook(self, layer_idx: int) -> bool:
        return any(w.needs_layer_hook(layer_idx) for w in self._families)

    # Eager breaks, so a gather may stay in flight across captured prefill graph segments.
    @eager_on_graph
    def on_layer_start(self, layer_idx: int) -> None:
        for w in self._families:
            if w.needs_layer_hook(layer_idx):
                w.on_layer_start(layer_idx)

    # Keeps gather NVLink traffic off the MoE all-to-all.
    @eager_on_graph
    def wait_issued(self) -> None:
        for w in self._families:
            w.wait_issued()
