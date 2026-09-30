"""Gather selected target KV into request-owned, pinned Host storage."""

from __future__ import annotations

from typing import Mapping

import torch
from sglang.srt.training_capture.protocol import DTYPES, ContractError, KVSpec


class SelectedLayerKVExporter:
    def __init__(self, kv: KVSpec, buffers: Mapping[str, torch.Tensor]):
        self.kv = kv
        self.buffers = dict(buffers)
        expected = set()
        for layer in kv.layers:
            for component, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            ):
                name = f"target_{component}.{layer.layer_id}"
                expected.add(name)
                value = self.buffers[name]
                if (
                    value.ndim != 3
                    or list(value.shape[1:]) != [layer.num_kv_heads, dim]
                    or value.dtype != DTYPES[kv.dtype]
                ):
                    raise ContractError(
                        "KV exporter requires unquantized token/head/dim source buffers"
                    )
        if set(self.buffers) != expected:
            raise ContractError("KV source layers differ from the capture contract")

    @classmethod
    def from_pool(cls, kv: KVSpec, pool):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

        if (
            not isinstance(pool, MHATokenToKVPool)
            or pool.is_quantized_kv_cache
            or pool.use_hnd
        ):
            raise ContractError(
                "capture requires a dense, unquantized NHD MHA/GQA pool"
            )
        if pool.page_size != kv.source_page_size:
            raise ContractError("configured source page size disagrees with the pool")
        buffers = {}
        for layer in kv.layers:
            if not 0 <= layer.layer_id - pool.start_layer < pool.layer_num:
                raise ContractError("selected layer is not local to this worker")
            buffers[f"target_k.{layer.layer_id}"] = pool.get_key_buffer(layer.layer_id)
            buffers[f"target_v.{layer.layer_id}"] = pool.get_value_buffer(
                layer.layer_id
            )
        return cls(kv, buffers)

    @torch.no_grad()
    def export(
        self,
        slots: torch.Tensor,
        destinations: Mapping[str, torch.Tensor],
        start: int,
        end: int,
    ) -> None:
        """Enqueue on the producer stream, before any reuse of the source slots.

        Gathered temporaries own storage. Same-stream ordering protects the
        pool read, and record_stream protects temporary storage through D2H.
        The caller records completion and retains Host storage until it fires.
        """
        if slots.ndim != 1 or slots.numel() != end - start or not 0 <= start < end:
            raise ContractError("invalid KV export position mapping")
        for name, source in self.buffers.items():
            destination = destinations[name][start:end]
            if destination.shape[0] != end - start or destination.device.type != "cpu":
                raise ContractError("Host destination does not cover the KV range")
            if source.is_cuda and not destination.is_pinned():
                raise ContractError(
                    "asynchronous KV export requires pinned Host storage"
                )
            gathered = source.index_select(
                0, slots.to(dtype=torch.long, device=source.device)
            )
            destination.copy_(gathered, non_blocking=source.is_cuda)
            if source.is_cuda:
                gathered.record_stream(torch.cuda.current_stream(source.device))
