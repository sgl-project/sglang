"""RowArena: a GPU cache of fixed-size rows whose physical memory can be handed back.

The arena reserves virtual address space for ``max_rows`` rows once (cuMemAddressReserve) and
backs a PREFIX of it with physical chunks on demand (cuMemCreate + cuMemMap). Rows are kept in
rank order (hottest first), so shrinking the cache is a tail unmap: the memory really returns to
the driver, in chunks, while every row address stays constant for the arena's lifetime. Kernels
that read rows through an address table, and CUDA graphs that captured such kernels, never need
to know.

    a = RowArena(row_bytes, max_rows, device_id)
    a.ensure_rows(n)      rows [0, n) are physically backed (may raise ArenaOOM)
    a.shrink_rows(n)      release every chunk that lies entirely beyond row n
    a.row_addr(i)         base + i * row_bytes (valid as an address even when unbacked)
    a.view(n, tail, dtype) non-owning torch tensor [n, *tail] over the backed prefix
    a.backed_rows         rows guaranteed backed right now

Chunk size is a multiple of the device granularity (2 MiB on current GPUs); with 4 MiB chunks
the cache is resizable in ~20-row steps for a 205 KB expert row.

Unit test: test/registered/unit/layers/moe/test_row_arena.py.
"""

import torch

from sglang.srt.cuda_vmm_utils import (  # the driver plumbing SGLang already ships
    _get_cuda_driver,
    align_up,
    check_drv,
    get_device_granularity,
    make_device_allocation_prop,
    make_rw_access_desc,
    tensor_from_pointer,
)


class ArenaOOM(RuntimeError):
    """cuMemCreate failed: the device has no free physical memory for another chunk."""


class RowArena:
    def __init__(
        self,
        row_bytes: int,
        max_rows: int,
        device_id: int = 0,
        chunk_bytes: int = 4 << 20,
        name: str = "",
    ):
        drv = _get_cuda_driver()
        self.row_bytes, self.max_rows, self.device_id, self.name = (
            int(row_bytes),
            int(max_rows),
            int(device_id),
            name,
        )
        self.gran = get_device_granularity(self.device_id)
        self.chunk = align_up(chunk_bytes, self.gran)
        self.prop = make_device_allocation_prop(self.device_id, handle_types=None)
        self.access = [make_rw_access_desc(self.device_id)]
        self.va_bytes = align_up(self.row_bytes * self.max_rows, self.chunk)
        self.base = int(
            check_drv(
                drv.cuMemAddressReserve(self.va_bytes, 0, 0, 0), "cuMemAddressReserve"
            )
        )
        self._handles = []  # chunk i backs [i*chunk, (i+1)*chunk)
        self._closed = False

    # ---------------------------------------------------------------- geometry
    @property
    def backed_bytes(self) -> int:
        return len(self._handles) * self.chunk

    @property
    def backed_rows(self) -> int:
        return min(self.max_rows, self.backed_bytes // self.row_bytes)

    def row_addr(self, i: int) -> int:
        return self.base + i * self.row_bytes

    def chunks_for_rows(self, n: int) -> int:
        return (
            -(-(min(int(n), self.max_rows) * self.row_bytes) // self.chunk)
            if n > 0
            else 0
        )

    def bytes_to_reach(self, n: int) -> int:
        """Bytes cuMemCreate would need to make rows [0, n) backed (chunk-granular)."""
        return max(0, self.chunks_for_rows(n) - len(self._handles)) * self.chunk

    def rows_for_bytes(self, nbytes: int) -> int:
        return min(self.max_rows, int(nbytes) // self.row_bytes)

    # ---------------------------------------------------------------- resize
    def ensure_rows(self, n: int) -> int:
        """Back rows [0, n). Returns the number of bytes newly mapped."""
        n = max(0, min(int(n), self.max_rows))
        want_chunks = -(-(n * self.row_bytes) // self.chunk) if n else 0
        drv = _get_cuda_driver()
        added = 0
        with torch.cuda.device(self.device_id):
            while len(self._handles) < want_chunks:
                off = len(self._handles) * self.chunk
                err, handle = drv.cuMemCreate(self.chunk, self.prop, 0)
                if err != drv.CUresult.CUDA_SUCCESS:
                    raise ArenaOOM(
                        f"{self.name}: cuMemCreate({self.chunk >> 20} MiB) -> {err}"
                    )
                try:
                    check_drv(
                        drv.cuMemMap(self.base + off, self.chunk, 0, handle, 0),
                        "cuMemMap",
                    )
                    check_drv(
                        drv.cuMemSetAccess(self.base + off, self.chunk, self.access, 1),
                        "cuMemSetAccess",
                    )
                except BaseException:
                    drv.cuMemRelease(handle)
                    raise
                self._handles.append(handle)
                added += self.chunk
        return added

    def shrink_rows(self, n: int) -> int:
        """Keep rows [0, n) backed, release every chunk entirely beyond. Returns bytes freed.
        The caller must make sure no kernel still reads the released rows (sync first).
        """
        n = max(0, min(int(n), self.max_rows))
        keep_chunks = -(-(n * self.row_bytes) // self.chunk) if n else 0
        drv = _get_cuda_driver()
        freed = 0
        while len(self._handles) > keep_chunks:
            off = (len(self._handles) - 1) * self.chunk
            handle = self._handles.pop()
            check_drv(drv.cuMemUnmap(self.base + off, self.chunk), "cuMemUnmap")
            check_drv(drv.cuMemRelease(handle), "cuMemRelease")
            freed += self.chunk
        return freed

    # ---------------------------------------------------------------- views
    def view(
        self, n_rows: int, tail=(), dtype: torch.dtype = torch.uint8
    ) -> torch.Tensor:
        """Non-owning tensor [n_rows, *tail] over the backed prefix. Re-create after a shrink."""
        assert (
            n_rows <= self.backed_rows
        ), f"{self.name}: view of {n_rows} rows, only {self.backed_rows} backed"
        return tensor_from_pointer(
            self.base,
            n_rows * self.row_bytes,
            shape=(n_rows,) + tuple(tail),
            dtype=dtype,
            device_id=self.device_id,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize()
        self.shrink_rows(0)
        check_drv(
            _get_cuda_driver().cuMemAddressFree(self.base, self.va_bytes),
            "cuMemAddressFree",
        )

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
