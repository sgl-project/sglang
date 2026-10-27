"""Bounded staging for pageable MLA page-first-direct host transfers.

Write-back packs on the GPU; read-back gathers pageable rows on the CPU.
Both preserve [page, layer, token, byte] layout and never dereference or
register host memory from a GPU kernel.
"""

import torch
import triton
import triton.language as tl


def transfer_pageable_dsv4_tokens(
    host, device_buffers, host_indices, device_indices, *, layer=None
):
    """Copy partial V4 rows through bounded, ordinary tensor copies.

    ``layer=None`` backs up all layers; an integer restores that layer. Only
    device tensors participate in GPU indexing. Blocking copies fence the CPU
    gather/scatter, including when host registration is unavailable.
    """
    page_size, value_bytes, scale_bytes = 64, 576, 8
    page_bytes = 37440
    if host.device.type != "cpu" or host.shape[-1] != page_bytes:
        raise ValueError("partial pageable transfers require the DSV4 V4 row layout")
    if host_indices.numel() != device_indices.numel():
        raise ValueError("partial pageable transfer index size mismatch")
    host_rows = host.view(host.shape[0], host.shape[1], page_bytes)
    hi = host_indices.to(device="cpu", dtype=torch.long)
    device_cpu = device_indices.to(device="cpu", dtype=torch.long)
    if hi.numel() and (int(hi.min()) < 0 or int(hi.max()) >= host.shape[0] * page_size):
        raise ValueError("partial pageable host index out of bounds")
    layers = range(len(device_buffers)) if layer is None else (layer,)
    for l in layers:
        buf = device_buffers[l]
        if not buf.is_contiguous():
            raise ValueError(
                "partial pageable transfers require contiguous device rows"
            )
        dev = buf.view(torch.uint8).view(-1, page_bytes)
        if device_cpu.numel() and (
            int(device_cpu.min()) < 0
            or int(device_cpu.max()) >= dev.shape[0] * page_size
        ):
            raise ValueError("partial pageable device index out of bounds")
        di = device_cpu.to(device=buf.device)
        # At most 1024 * 576 bytes of payload per layer/component per copy.
        for begin in range(0, hi.numel(), 1024):
            h = hi[begin : begin + 1024]
            d = di[begin : begin + 1024]
            for base, width in (
                (0, value_bytes),
                (page_size * value_bytes, scale_bytes),
            ):
                columns = torch.arange(width, device="cpu")
                hc = base + (h % page_size)[:, None] * width + columns
                dc = base + (d % page_size)[:, None] * width + columns.to(buf.device)
                hr, dr = (h // page_size)[:, None], (d // page_size)[:, None]
                if layer is None:
                    host_rows[hr, l, hc] = dev[dr, dc].to("cpu", non_blocking=False)
                else:
                    dev[dr, dc] = host_rows[hr, l, hc].to(
                        buf.device, non_blocking=False
                    )
        di.record_stream(torch.cuda.current_stream(buf.device))


@triton.jit
def _pack_pages(
    ptrs,
    indices,
    packed,
    PAGE: tl.constexpr,
    ROW: tl.constexpr,
    LAYERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    page = tl.program_id(0)
    layer = tl.program_id(1)
    x = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    mask = x < PAGE * ROW
    token = tl.load(indices + page * PAGE + x // ROW, mask, other=0)
    src = tl.load(ptrs + layer).to(tl.pointer_type(tl.uint8))
    data = tl.load(src + token.to(tl.int64) * ROW + x % ROW, mask, other=0)
    offset = ((page.to(tl.int64) * LAYERS + layer) * PAGE * ROW) + x
    tl.store(packed + offset, data, mask)


def contiguous_page_runs(pages):
    """Yield source offsets and destination spans without sorting page order."""
    begin = 0
    for end in range(1, len(pages) + 1):
        if end == len(pages) or pages[end] != pages[end - 1] + 1:
            yield begin, end, pages[begin]
            begin = end


class StagedDirectBackup:
    def __init__(self, host, page_size, device, max_bytes=64 * 1024 * 1024):
        assert host.device.type == "cpu" and host.is_contiguous()
        self.page_size = page_size
        self.layers = host.shape[1]
        self.row_bytes = host[0, 0, 0].numel() * host.element_size()
        self.page_bytes = self.layers * page_size * self.row_bytes
        self.capacity = max(1, min(64, max_bytes // self.page_bytes))
        self.host = host.view(torch.uint8).view(host.shape[0], self.page_bytes)
        self.packed = torch.empty(
            (self.capacity, self.page_bytes), dtype=torch.uint8, device=device
        )

    def backup(self, ptrs, host_indices, device_indices):
        # Host allocation supplies complete aligned pages. Check before issuing
        # any copy, so malformed destinations cannot overwrite adjacent pages.
        if host_indices.device.type != "cpu":
            raise ValueError("staged direct backup requires CPU host indices")
        n = host_indices.numel()
        if n != device_indices.numel() or n % self.page_size:
            raise ValueError("staged direct backup requires complete matching pages")
        if not n:
            return
        rows = host_indices.reshape(-1, self.page_size)
        starts = rows[:, 0]
        expected = starts[:, None] + torch.arange(self.page_size)
        if (
            not torch.equal(rows, expected)
            or bool((starts % self.page_size).any())
            or int(starts.min()) < 0
            or int(starts.max()) // self.page_size >= self.host.shape[0]
        ):
            raise ValueError("invalid staged direct destination pages")
        pages = (starts // self.page_size).tolist()
        if len(set(pages)) != len(pages):
            raise ValueError("duplicate staged direct destination pages")
        indices = device_indices.to(self.packed.device).contiguous()
        for begin in range(0, len(pages), self.capacity):
            chunk = pages[begin : begin + self.capacity]
            _pack_pages[
                (
                    len(chunk),
                    self.layers,
                    triton.cdiv(self.page_size * self.row_bytes, 1024),
                )
            ](
                ptrs,
                indices[begin * self.page_size :],
                self.packed,
                self.page_size,
                self.row_bytes,
                self.layers,
                1024,
            )
            for lo, hi, dst in contiguous_page_runs(chunk):
                self.host[dst : dst + hi - lo].copy_(
                    self.packed[lo:hi], non_blocking=True
                )
        # The caller owns a single D2H stream and fences pool reuse. Track the
        # temporary indices on that stream; packed storage persists with pool.
        indices.record_stream(torch.cuda.current_stream(self.packed.device))


class StagedDirectLoad:
    """Gather pageable host rows, then upload and scatter one layer at a time.

    All GPU accesses target device allocations. The blocking H2D copy is
    intentional: the CPU staging buffer must be reusable on the next layer,
    including on systems where host memory cannot be pinned or registered.
    The caller serializes calls on its load stream, including packed drafts.
    """

    def __init__(self, host, page_size, device, max_bytes=64 * 1024 * 1024):
        if host.device.type != "cpu" or not host.is_contiguous():
            raise ValueError("staged direct load requires a contiguous CPU pool")
        self.page_size = page_size
        self.row_bytes = host[0, 0].numel() * host.element_size() // page_size
        self.host = host.view(torch.uint8).reshape(
            host.shape[0], host.shape[1], page_size * self.row_bytes
        )
        self.capacity = max(1, max_bytes // (page_size * self.row_bytes))
        self.device = device
        self.cpu = self.gpu = None

    def load(self, dst, host_indices, device_indices, layer):
        n = host_indices.numel()
        if host_indices.device.type != "cpu":
            raise ValueError("staged direct load requires CPU host indices")
        if n != device_indices.numel() or n % self.page_size:
            raise ValueError("staged direct load requires complete matching pages")
        if not n:
            return
        rows = host_indices.reshape(-1, self.page_size)
        starts = rows[:, 0]
        if not torch.equal(
            rows, starts[:, None] + torch.arange(self.page_size, device="cpu")
        ) or bool((starts % self.page_size).any()):
            raise ValueError("invalid staged direct source pages")
        pages = (starts // self.page_size).long()
        capacity = min(pages.numel(), self.capacity)
        if self.cpu is None or self.cpu.shape[0] < capacity:
            self.cpu = torch.empty(
                (capacity, self.page_size * self.row_bytes),
                dtype=torch.uint8,
                device="cpu",
                pin_memory=False,
            )
            self.gpu = torch.empty_like(self.cpu, device=self.device)
        indices = device_indices.to(self.device, dtype=torch.long)
        dst_bytes = dst.view(torch.uint8).view(-1, self.row_bytes)
        for begin in range(0, pages.numel(), capacity):
            count = min(capacity, pages.numel() - begin)
            cpu = self.cpu[:count]
            gpu = self.gpu[:count]
            torch.index_select(
                self.host[:, layer], 0, pages[begin : begin + count], out=cpu
            )
            gpu.copy_(cpu, non_blocking=False)
            offset = begin * self.page_size
            dst_bytes.index_copy_(
                0,
                indices[offset : offset + count * self.page_size],
                gpu.view(-1, self.row_bytes),
            )
        indices.record_stream(torch.cuda.current_stream(self.device))
