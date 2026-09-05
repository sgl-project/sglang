"""How a hybrid SWA allocator pairs a full-attention slot with its SWA slot."""

from __future__ import annotations

import abc

import torch

from sglang.srt.utils import is_npu

_is_npu = is_npu()

if _is_npu:
    import torch_npu


class BaseFullToSWAPairing(abc.ABC):
    """full slot id -> swa slot id. Ownership only: nothing here allocates or
    frees a slot."""

    @abc.abstractmethod
    def translate(self, full_indices: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def set(self, full_indices: torch.Tensor, swa_indices: torch.Tensor) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def clear(self, full_indices: torch.Tensor) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def transfer(self, kept_full: torch.Tensor, incoming_full: torch.Tensor) -> None:
        """Hand ``incoming_full``'s swa peers to ``kept_full`` (the same logical
        tokens, held by a locked node) and leave ``incoming_full`` unpaired."""
        raise NotImplementedError()

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()


class MappingTensorPairing(BaseFullToSWAPairing):
    """A dense full -> swa table. The KV pool and the attention kernels hold
    ``mapping`` by identity, so it is never reallocated."""

    def __init__(self, *, size_full: int, page_size: int, device: str):
        # Trailing -1: a last_loc of -1 (no prefix) indexes it, so alloc_extend and
        # alloc_decode see -1 on the SWA side as well.
        self.mapping = torch.cat(
            [
                torch.zeros(size_full + page_size, dtype=torch.int64, device=device),
                torch.tensor([-1], dtype=torch.int64, device=device),
            ]
        )

    def translate(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.mapping[full_indices]

    def set(self, full_indices: torch.Tensor, swa_indices: torch.Tensor) -> None:
        if full_indices.numel() == 0:
            return
        assert full_indices.numel() == swa_indices.numel()
        full_indices = full_indices.to(torch.int64)
        swa_indices = swa_indices.to(self.mapping.dtype)
        if _is_npu:
            torch_npu.npu_scatter_nd_update_(
                self.mapping, full_indices.unsqueeze(-1), swa_indices
            )
        else:
            self.mapping[full_indices] = swa_indices

    def clear(self, full_indices: torch.Tensor) -> None:
        if full_indices.numel() == 0:
            return
        full_indices = full_indices.to(torch.int64)
        if _is_npu:
            # NPU: aclnnIndexFill is unoptimized; direct assignment avoids the overhead.
            self.mapping[full_indices] = 0
        else:
            # CUDA: index_fill_ passes the 0 as a kernel argument; mapping[idx] = 0
            # copies a host-resident scalar and blocks until the stream drains.
            self.mapping.index_fill_(0, full_indices, 0)

    def transfer(self, kept_full: torch.Tensor, incoming_full: torch.Tensor) -> None:
        self.set(kept_full, self.mapping[incoming_full])
        self.clear(incoming_full)

    def reset(self) -> None:
        self.mapping[:-1].fill_(0)


def paired_pages(
    kept_full: torch.Tensor, incoming_full: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Page ids of two token ranges that address the same logical tokens,
    deduped by first occurrence with one shared mask. ``torch.unique`` would
    sort by id and pair page k of one range with an unrelated page of the other."""
    kept = kept_full.detach().to(torch.int64) // page_size
    incoming = incoming_full.detach().to(torch.int64) // page_size
    assert kept.numel() == incoming.numel(), (
        f"locked-full recovery needs a 1:1 token correspondence, got "
        f"{kept.numel()} kept vs {incoming.numel()} incoming"
    )
    starts = torch.ones_like(kept, dtype=torch.bool)
    starts[1:] = kept[1:] != kept[:-1]
    incoming_starts = torch.ones_like(incoming, dtype=torch.bool)
    incoming_starts[1:] = incoming[1:] != incoming[:-1]
    assert torch.equal(starts, incoming_starts), (
        "the two ranges break into pages at different offsets, so no "
        "page-granular ownership transfer expresses the token mapping"
    )
    return kept[starts], incoming[starts]


class VirtualIdPairing(BaseFullToSWAPairing):
    """Full and swa share one virtual id space: the swa pool's v2p table is the
    pairing, so there is nothing to record on alloc."""

    def __init__(self, swa_pool):
        self.swa_pool = swa_pool

    def translate(self, full_indices: torch.Tensor) -> torch.Tensor:
        return self.swa_pool.translate_kv_loc_for_kernel(full_indices)

    def set(self, full_indices: torch.Tensor, swa_indices: torch.Tensor) -> None:
        return

    def clear(self, full_indices: torch.Tensor) -> None:
        return

    def transfer(self, kept_full: torch.Tensor, incoming_full: torch.Tensor) -> None:
        """Rebind: the kept virtual pages take the incoming pages' physical
        pages, and the incoming virtual pages are tombstoned on the swa side."""
        swa = self.swa_pool
        kept_pages, incoming_pages = paired_pages(
            kept_full, incoming_full, swa.page_size
        )
        physical = swa.virtual_to_physical[incoming_pages]
        # The incoming ids were just allocated by the in-flight request, so every
        # page is live; the sink (0) or a tombstone (-1) here would serve zeros.
        assert bool((physical > 0).all()), (
            f"incoming swa pages must all be live, got {physical.tolist()}"
        )
        swa.bind(kept_pages, physical)
        swa.virtual_to_physical.index_fill_(0, incoming_pages, -1)
        swa.clear_inverse_history()

    def reset(self) -> None:
        return
