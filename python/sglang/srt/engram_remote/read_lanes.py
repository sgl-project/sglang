"""Optional independent READ lanes over one registered destination allocation."""

import torch
import triton
import triton.language as tl

from sglang.srt.engram_remote.engram_transport import Context


@triton.jit
def _merge_stats(src, dst, LANES: tl.constexpr):
    column = tl.arange(0, 8)
    value = tl.full((8,), 0, tl.uint64)
    for lane in range(LANES):
        item = tl.load(src + lane * 8 + column)
        value = tl.where(
            (column == 0) | (column == 3) | (column == 7),
            tl.maximum(value, item),
            value + item,
        )
    tl.store(dst + column, value)


class ReadLanes:
    def __init__(self, library, gpu, port, lanes):
        if lanes not in (2, 4):
            raise ValueError("READ lanes must be2or4")
        self.library, self.gpu, self.port, self.lanes = library, gpu, port, lanes
        self.contexts = [Context(library, gpu, port)]
        self.supports_row264 = self.contexts[0].supports_row264
        self.streams = [torch.cuda.Stream(device=gpu) for _ in range(lanes)]
        self.events = [torch.cuda.Event() for _ in range(lanes)]
        self.stats = torch.zeros((lanes, 8), dtype=torch.uint64, device="cuda")
        self.used = set()

    def register(self, tensors, host):
        if host:
            raise ValueError("READ lanes only register the GPU destination")
        self.contexts[0].register(tensors, False)
        for i in range(1, self.lanes):
            c = Context(self.library, self.gpu, self.port + i)
            self.contexts.append(c)
            c.register(tensors, False)

    def connect_raw(self, *args):
        for c in self.contexts:
            c.connect_raw(*args)

    def fetch(
        self,
        ids,
        count,
        payload_offset,
        scale_offset,
        dump_offset,
        window_rows,
        stats,
        **kw,
    ):
        if not kw.get("row264"):
            raise ValueError("READ lanes require row264")
        # Tiny lookups are dominated by launch/stream overhead, not READ service.
        active = self.lanes if count >= 4096 else 1
        chunk = (count + active - 1) // active
        main = torch.cuda.current_stream(self.gpu)
        for i in range(active):
            start = i * chunk
            end = min(count, start + chunk)
            self.streams[i].wait_stream(main)
            with torch.cuda.stream(self.streams[i]):
                part = ids[start:end]
                part.record_stream(self.streams[i])
                self.contexts[i].fetch(
                    part,
                    end - start,
                    payload_offset + start * 264,
                    scale_offset,
                    dump_offset + i,
                    window_rows,
                    self.stats[i],
                    **kw,
                )
                self.events[i].record()
            self.used.add(i)
        for i in range(active):
            main.wait_event(self.events[i])
        _merge_stats[(1,)](self.stats, stats, active, num_warps=1)

    def record_consumed(self):
        main = torch.cuda.current_stream(self.gpu)
        for i in sorted(self.used):
            self.streams[i].wait_stream(main)
            with torch.cuda.stream(self.streams[i]):
                self.contexts[i].record_consumed()

    def close(self):
        # A failed retirement raises before the owning ReaderRuntime lease release.
        for c in self.contexts:
            c.close()
