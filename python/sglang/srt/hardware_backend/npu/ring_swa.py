"""Per-request KV regions for Unlimited-OCR's prefill-aware sliding window.

Each request owns one page-aligned contiguous region sized for its own prompt:

    base .. base+R-1        prompt KV, written once by prefill
    base+R .. base+R+W-1    ring for the last W generated tokens

Decode writes generated token t to ``base + R + (t mod W)`` and attends over
``[base, base + R + min(t+1, W))``. Every ring slot is older than the single
decode query, so decode needs no attention mask; ring order after a wrap does
not matter because attention is permutation-invariant over keys. This mirrors
the reference implementation, which gives every sequence its own contiguous
``[prompt][ring]`` cache.

Generated tokens past the window never occupy KV, so the region is sized by the
prompt, not by the context length.
"""

from __future__ import annotations

import msgspec


def ring_swa_requested() -> bool:
    """Whether --uocr-ring-swa is on.

    Unit tests and helper processes run without published config namespaces;
    the flag only exists inside a server context, so treat that as off.
    """
    from sglang.srt.runtime_context import get_mm

    try:
        return bool(get_mm().uocr_ring_swa)
    except ValueError:
        return False


class RingRegionAllocator(msgspec.Struct):
    """First-fit allocator of contiguous KV page runs.

    The pool is one address space: free intervals live in a sorted list and a
    run is page-aligned, so ``logical position -> base + position`` holds inside
    it. Requests are long-lived and released whole, so the interval list stays
    short and merges back on release.
    """

    page_size: int
    window: int
    total_pages: int
    free: list[tuple[int, int]] = []
    runs: dict[int, int] = {}
    base_by_row: dict[int, int] = {}
    free_pages: int = 0

    def __post_init__(self) -> None:
        self.free = [(0, self.total_pages)]
        self.runs = {}
        self.base_by_row = {}
        self.free_pages = self.total_pages

    def pages_for(self, prompt_len: int) -> int:
        needed = prompt_len + self.window
        return (needed + self.page_size - 1) // self.page_size

    def alloc(self, prompt_len: int) -> int | None:
        """Reserve a run for one request; return its base token or None."""
        need = self.pages_for(prompt_len)
        for index, (start, end) in enumerate(self.free):
            if end - start < need:
                continue
            if end - start == need:
                self.free.pop(index)
            else:
                self.free[index] = (start + need, end)
            self.runs[start] = need
            self.free_pages -= need
            return start * self.page_size
        return None

    def free_run(self, token: int) -> None:
        """Release the run whose base is ``token`` (no-op if already released)."""
        base_page = token // self.page_size
        pages = self.runs.pop(base_page, None)
        if pages is None:
            return
        base_token = base_page * self.page_size
        for row in [r for r, b in self.base_by_row.items() if b == base_token]:
            del self.base_by_row[row]
        start, end = base_page, base_page + pages
        merged: list[tuple[int, int]] = []
        for free_start, free_end in self.free:
            if free_end < start:
                merged.append((free_start, free_end))
            elif end < free_start:
                if start >= 0:
                    merged.append((start, end))
                    start = -1
                merged.append((free_start, free_end))
            else:
                start = min(start, free_start)
                end = max(end, free_end)
        if start >= 0:
            merged.append((start, end))
        self.free = merged
        self.free_pages += pages

    def available_tokens(self) -> int:
        return self.free_pages * self.page_size


def extend_locs(*, bases, prefix_lens, seq_lens, page_size, device):
    """Physical slots for the newly extended positions, in request order.

    Prompt writes stay linear inside the region, so ``slot = base + position``.
    Only decode wraps into the ring.
    """
    import torch

    bases = bases.to(torch.int64)
    lens = (seq_lens - prefix_lens).to(torch.int64)
    total = int(lens.sum().item())
    if total == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    starts = prefix_lens.to(torch.int64)
    segment_starts = torch.cumsum(lens, 0) - lens
    owner = torch.repeat_interleave(torch.arange(bases.numel(), device=device), lens)
    within = torch.arange(total, device=device) - torch.repeat_interleave(
        segment_starts, lens
    )
    return bases[owner] + starts[owner] + within


def ring_write_slots(*, cache_loc, region_base, prompt_lens, window: int):
    """Map logical decode slots (``base + p``) to ring slots (``base + R + t%W``).

    ``cache_loc``, ``region_base`` and ``prompt_lens`` hold one entry per token
    and are aligned element-wise; a decode step writes one token per request.
    """
    import torch

    offset = cache_loc.to(torch.int64) - region_base.to(torch.int64)
    return region_base + prompt_lens + torch.remainder(offset - prompt_lens, window)


def attn_len(*, prompt_lens, seq_lens, window: int):
    """KV length visible to attention: prompt plus at most one window."""
    import torch

    return torch.minimum(seq_lens, prompt_lens + window)
