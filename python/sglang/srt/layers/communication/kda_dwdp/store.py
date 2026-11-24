"""DWDP weight memory: this rank's shard, every group peer's shard mapped over NVLink, and the gather slots."""

from __future__ import annotations

import logging
import os
import sys
import time
import weakref
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.communication.kda_dwdp.layout import UnitLayout
from sglang.srt.utils.cuda_vmm_utils import (
    VmmReservation,
    align_up,
    check_drv,
    exchange_shareable_handles,
    get_device_granularity,
    import_and_map_alloc,
    make_device_allocation_prop,
    release_mappings,
    tensor_from_pointer,
)

logger = logging.getLogger(__name__)


class ShardStore:
    def __init__(
        self,
        *,
        unit_layers: Sequence[int],
        slot_numel: int,
        num_slots: int,
        dtype: torch.dtype,
        device: torch.device,
        dummy_load: bool,
    ):
        self.unit_layers = list(unit_layers)
        self.layouts: List[Optional[UnitLayout]] = [None] * len(self.unit_layers)
        self.dtype, self.device = dtype, device
        self.slots = torch.empty((num_slots, slot_numel), dtype=dtype, device=device)
        self.shard: Optional[torch.Tensor] = None
        self.peers: List[torch.Tensor] = []
        self.dummy_load = dummy_load
        # Per unit: piece index -> disjoint row ranges [x, y) loaded so far.
        self._loaded: List[Dict[int, List[Tuple[int, int]]]] = [
            {} for _ in self.unit_layers
        ]

    def add_unit(self, u: int, layout: UnitLayout) -> None:
        if self.layouts[u] is not None:
            raise RuntimeError(f"DWDP: layer {self.unit_layers[u]} bound twice")
        if layout.slot_numel > self.slots.shape[1]:
            raise ValueError(
                f"DWDP: layer {self.unit_layers[u]} needs {layout.slot_numel} slot "
                f"elements, the slots hold {self.slots.shape[1]}"
            )
        self.layouts[u] = layout

    def param_view(self, u: int, i: int) -> torch.Tensor:
        spec = self.layouts[u].specs[i]
        rows, cols = spec.shape
        slot = self.slots[u % self.slots.shape[0]]
        return slot[spec.slot_offset : spec.slot_offset + rows * cols].view(rows, cols)

    def map_shards(self, group, group_rank: int) -> None:
        unbound = [l for l, lay in zip(self.unit_layers, self.layouts) if lay is None]
        if unbound:
            raise RuntimeError(f"DWDP: layers {unbound} were never bound")
        self.group, self.group_rank = group, group_rank
        self._unit_offset = []
        numel = 0
        for layout in self.layouts:
            self._unit_offset.append(numel)
            numel += layout.shard_numel
        device_id = self.device.index
        es = self.slots.element_size()
        gran = get_device_granularity(device_id)
        nbytes = align_up(numel * es, gran)
        local = VmmReservation(
            nbytes, make_device_allocation_prop(device_id), device_id, alignment=gran
        )
        handle = local.map(0, nbytes, retain_handle=True)
        peer_maps: list = []
        # The shard and peer views do not own their memory; it lives as long as the store.
        weakref.finalize(self, _release, device_id, local, peer_maps)

        def view(va: int) -> torch.Tensor:
            return tensor_from_pointer(
                va, numel * es, shape=(numel,), dtype=self.dtype, device_id=device_id
            )

        self.shard = view(local.base)
        n = dist.get_world_size(group)
        t0 = time.perf_counter()
        logger.info("KDA DWDP: mapping %d peer shards", n - 1)
        use_fabric, peer_handles = exchange_shareable_handles(
            [handle], group, group_rank, n
        )
        try:
            for p in range(n):
                if p == group_rank:
                    self.peers.append(self.shard)
                    continue
                peer = peer_handles[(p, 0)]
                va = import_and_map_alloc(
                    peer if use_fabric else None,
                    None if use_fabric else peer,
                    nbytes,
                    device_id,
                    use_fabric=use_fabric,
                    peer_rank=p,
                )
                peer_maps.append((va, nbytes, [(0, nbytes)]))
                self.peers.append(view(va))
        finally:
            if not use_fabric:
                for fd in peer_handles.values():
                    os.close(fd)
        dist.barrier(group=group)  # every peer mapped before anyone pulls
        logger.info(
            "KDA DWDP: peer shard mappings via %s done in %.2f s",
            "FABRIC" if use_fabric else "POSIX fd",
            time.perf_counter() - t0,
        )
        self._ops = [self._build_ops(u) for u in range(len(self.layouts))]

    def _piece(self, u: int, i: int) -> torch.Tensor:
        pc = self.layouts[u].pieces[i]
        start = self._unit_offset[u] + pc.shard_offset
        return self.shard[start : start + pc.rows * pc.cols].view(pc.rows, pc.cols)

    def load(self, u: int, i: int, loaded: torch.Tensor, row_offset: int = 0) -> None:
        layout = self.layouts[u]
        spec = layout.specs[i]
        rows, cols = spec.shape
        n = loaded.shape[0]
        if loaded.dim() != 2 or loaded.shape[1] != cols:
            raise ValueError(
                f"DWDP layer {self.unit_layers[u]} param {i}: loaded shape "
                f"{tuple(loaded.shape)} does not match {spec.shape}"
            )
        if row_offset < 0 or row_offset + n > rows:
            raise ValueError(
                f"DWDP layer {self.unit_layers[u]} param {i}: rows "
                f"[{row_offset}, +{n}) outside {spec.shape}"
            )
        r = self.group_rank
        for j, pc in enumerate(layout.pieces):
            if pc.param != i:
                continue
            piece = self._piece(u, j)
            if spec.row_blocks:
                lo = layout.first_row(pc, r)
                x, y = max(lo, row_offset), min(lo + pc.rows, row_offset + n)
                if x < y:
                    self._mark_loaded(u, j, x - lo, y - lo)
                    piece[x - lo : y - lo].copy_(
                        loaded[x - row_offset : y - row_offset]
                    )
            else:
                self._mark_loaded(u, j, row_offset, row_offset + n)
                piece[row_offset : row_offset + n].copy_(
                    loaded[:, r * pc.cols : (r + 1) * pc.cols]
                )

    def _mark_loaded(self, u: int, j: int, x: int, y: int) -> None:
        ranges = self._loaded[u].setdefault(j, [])
        for a, b in ranges:
            if a < y and x < b:
                pc = self.layouts[u].pieces[j]
                raise RuntimeError(
                    f"DWDP: layer {self.unit_layers[u]} param {pc.param} block "
                    f"{pc.block} rows [{max(a, x)}, {min(b, y)}) of this rank's slice "
                    "loaded twice"
                )
        ranges.append((x, y))

    def check_loaded(self) -> None:
        bad = []
        for u, layout in enumerate(self.layouts):
            for j, pc in enumerate(layout.pieces):
                got = sum(b - a for a, b in self._loaded[u].get(j, ()))
                if got != pc.rows:
                    bad.append((self.unit_layers[u], pc.param, pc.block, got, pc.rows))
        if bad:
            raise RuntimeError(
                "DWDP: slices not fully loaded, as (layer, param, block, rows loaded, "
                f"rows expected): {bad[:8]}"
            )

    def finish_load(self) -> None:
        if self.dummy_load and not any(self._loaded):
            g = torch.Generator(device=self.device).manual_seed(1234 + self.group_rank)
            self.shard.uniform_(-1e-3, 1e-3, generator=g)
            for u, layout in enumerate(self.layouts):
                self._loaded[u] = {
                    j: [(0, pc.rows)] for j, pc in enumerate(layout.pieces)
                }
        self.check_loaded()
        # A peer must not pull this shard before its host-to-device copies land.
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)

    def _build_ops(self, u: int) -> list:
        from cuda.bindings import driver as drv

        slot = self.slots[u % self.slots.shape[0]]
        base = self._unit_offset[u]
        es = self.slots.element_size()
        ops = []
        for op in self.layouts[u].copy_ops(self.group_rank):
            src = self.peers[op.src_rank][
                base + op.src_offset : base + op.src_offset + op.rows * op.cols
            ]
            if not op.dst_pitch:
                ops.append(
                    (slot[op.dst_offset : op.dst_offset + op.rows * op.cols], src)
                )
                continue
            # Copy-engine strided copy; a torch copy into the strided view would launch a kernel.
            args = drv.CUDA_MEMCPY2D()
            args.srcMemoryType = drv.CUmemorytype.CU_MEMORYTYPE_DEVICE
            args.srcDevice = src.data_ptr()
            args.srcPitch = op.cols * es
            args.dstMemoryType = drv.CUmemorytype.CU_MEMORYTYPE_DEVICE
            args.dstDevice = slot[op.dst_offset :].data_ptr()
            args.dstPitch = op.dst_pitch * es
            args.WidthInBytes = op.cols * es
            args.Height = op.rows
            ops.append((None, args))
        return ops

    def copy_unit(self, u: int, stream: torch.cuda.Stream) -> None:
        from cuda.bindings import driver as drv

        for dst, src in self._ops[u]:
            if dst is not None:
                dst.copy_(src, non_blocking=True)
            else:
                check_drv(
                    drv.cuMemcpy2DAsync(src, stream.cuda_stream), "cuMemcpy2DAsync"
                )

    def nbytes(self) -> Tuple[int, int]:
        es = self.slots.element_size()
        return self.shard.numel() * es, self.slots.numel() * es


def _release(device_id: int, local: VmmReservation, peer_maps: list) -> None:
    if sys.is_finalizing():
        # The driver tears the context down at exit anyway.
        return
    torch.cuda.synchronize(device_id)
    release_mappings(peer_maps)
    local.close()
