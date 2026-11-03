"""Opt-in model adapter: bounded scratch, live MR generation, GPU read plans."""

import torch

from sglang.srt.engram_remote import dedup_rows
from sglang.srt.engram_remote.engram_transport import Context
from sglang.srt.engram_remote.read_group import ReadGroup
from sglang.srt.engram_remote.read_lanes import _merge_stats


class ReadPlan:
    def __init__(
        self, library, gpu, port, lanes, max_queries=196608, graph_sizes=(49152,)
    ):
        if lanes not in (4, 8, 16):
            raise ValueError("planned READ supports4/8/16QP")
        self.gpu, self.lanes, self.maximum = gpu, lanes, max_queries
        self.graph_sizes = set(graph_sizes)
        self.graph_captures = 0
        self.graph_replays = 0
        self.direct_calls = 0
        self.contexts = []
        self.groups = {}
        self.plans = {}
        self.last_map = {}
        self.used = set()
        self.stream = None
        self.last_active = 0
        self.supports_row264 = True
        try:
            for i in range(lanes):
                self.contexts.append(Context(library, gpu, port + i))
        except BaseException:
            for c in self.contexts:
                c.close()
            raise

    def register(self, tensors, host):
        if host:
            raise ValueError("planned reader requires oneGPUarena")
        self.arena = tensors[0]
        for c in self.contexts:
            c.register(tensors, False)

    def connect_raw(self, *args):
        for c in self.contexts:
            c.connect_raw(*args)
        self.layers = len(args[1])
        self.span = self.arena.numel() // self.layers
        self.stats = torch.zeros((self.lanes, 8), dtype=torch.uint64, device="cuda")
        self.workspace = []
        for li in range(self.layers):
            scratch, reps = dedup_rows.workspace(self.maximum, "cuda")
            self.workspace.append(
                dict(
                    ids=torch.empty(self.maximum, dtype=torch.int64, device="cuda"),
                    scratch=scratch,
                    reps=reps,
                    work=torch.empty_like(reps),
                )
            )
            for q in (1, self.lanes):
                group = ReadGroup(self.contexts[:q])
                group.set_pipeline(2)
                self.groups[(li, q)] = group

    def _enqueue(self, li, q, ids, payload, dump, window, lo, hi, dedup):
        w = self.workspace[li]
        reps = w["reps"][: ids.numel()] if dedup else None
        if dedup:
            dedup_rows.prepare(
                ids, lo, hi, w["scratch"], reps, w["work"][: ids.numel()]
            )
        self.groups[(li, q)].fetch(
            ids,
            ids.numel(),
            payload,
            dump,
            li,
            window,
            self.stats[:q],
            lo,
            hi,
            False,
            reps,
            w["work"][: ids.numel()] if dedup else None,
            w["scratch"][-1:] if dedup else None,
        )

    def fetch(
        self,
        ids,
        count,
        payload_offset,
        scale_offset,
        dump_offset,
        window_rows,
        stats,
        weight_index=0,
        scale_index=0,
        row_lo=0,
        row_hi=2**63 - 1,
        row264=False,
    ):
        if not row264 or weight_index != scale_index:
            raise ValueError("planned reader needs row264")
        if count != ids.numel() or count > self.maximum:
            raise ValueError("planned query count")
        current = torch.cuda.current_stream(self.gpu)
        if self.stream is None:
            self.stream = current.cuda_stream
        if self.stream != current.cuda_stream:
            raise RuntimeError("all planned operations must stay on oneproducerstream")
        ids.record_stream(current)
        li = weight_index
        q = self.lanes if count >= 4096 else 1
        self.last_active = q
        dedup = count >= 4096
        w = self.workspace[li]
        self.last_map[li] = w["reps"][:count] if dedup else None
        self.used.add((li, q))
        key = (
            li,
            count,
            q,
            payload_offset,
            dump_offset,
            window_rows,
            row_lo,
            row_hi,
            stats.data_ptr(),
        )
        if count in self.graph_sizes:
            stable = w["ids"][:count]
            stable.copy_(ids)
            if key not in self.plans:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "prime the communication plan before outercapture"
                    )
                for _ in range(2):
                    self._enqueue(
                        li,
                        q,
                        stable,
                        payload_offset,
                        dump_offset,
                        window_rows,
                        row_lo,
                        row_hi,
                        dedup,
                    )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=current):
                    self._enqueue(
                        li,
                        q,
                        stable,
                        payload_offset,
                        dump_offset,
                        window_rows,
                        row_lo,
                        row_hi,
                        dedup,
                    )
                    _merge_stats[(1,)](self.stats, stats, q, num_warps=1)
                self.plans[key] = (graph, stats)
                self.graph_captures += 1
            self.plans[key][0].replay()
            self.graph_replays += 1
        else:
            self.direct_calls += 1
            self._enqueue(
                li,
                q,
                ids,
                payload_offset,
                dump_offset,
                window_rows,
                row_lo,
                row_hi,
                dedup,
            )
            _merge_stats[(1,)](self.stats, stats, q, num_warps=1)

    def representatives(self, li):
        return self.last_map.get(li)

    def record_consumed(self):
        # Runtime queues the main-consumer event onto this same producer stream first.
        for c in self.contexts[: self.last_active]:
            c.record_consumed()

    def prime(self, ranges, p_off, dump, output_stats):
        for li in range(self.layers):
            lo, hi = ranges[(1, 14)[li]]
            for n in self.graph_sizes:
                if n > self.maximum:
                    continue
                # All valid source rows, evenly spanning this rank's immutable range.
                ids = (
                    torch.arange(n, device="cuda", dtype=torch.int64) * 104729
                ).remainder(hi - lo) + lo
                self.fetch(
                    ids,
                    n,
                    li * self.span + p_off,
                    0,
                    li * self.span + dump,
                    512,
                    output_stats[li],
                    weight_index=li,
                    scale_index=li,
                    row_lo=lo,
                    row_hi=hi,
                    row264=True,
                )
                self.record_consumed()
        torch.cuda.synchronize()

    def close(self):
        torch.cuda.synchronize()
        self.plans.clear()
        for g in self.groups.values():
            g.close()
        for c in self.contexts:
            c.close()
