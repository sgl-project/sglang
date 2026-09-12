"""Bounded GPU-only repairs through conditional peer reads, experimental."""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as sm
import triton

from sglang.srt.speculative.compact_verify.exp_dag import (
    gather_peer_rows,
    patch_from_softmax,
)
from sglang.srt.speculative.compact_verify.queue import _queue
from sglang.srt.speculative.compact_verify.source_bound import SourceBoundVerify


class ConditionalPeerVerify(SourceBoundVerify):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared = sm.empty(
            *self.local.shape, dtype=self.local.dtype, device=self.local.device
        )
        self.shared.copy_(self.local)
        self.handle = sm.rendezvous(self.shared, dist.group.WORLD)
        self.peers = [
            self.handle.get_buffer(r, (self.r, self.lv), self.local.dtype)
            for r in range(4)
        ]
        self.scratch = torch.empty(
            (self.capacity, self.v), device=self.local.device, dtype=torch.float32
        )
        self.probs = torch.empty_like(self.scratch)

    def _body(self, graph):
        self.shared.copy_(self.local)
        # Stats collectives occur after publication of each rank's local copy.
        p = self.selected_probabilities()
        # Outside the interval domain: repair every row using the original
        # softmax rather than exposing an invalid acceptance count to consumers.
        flags = self.uncertainty_flags(p) | self.force_flags | self.domain_invalid_rows
        descriptor = torch.full((self.r + 1,), -1, dtype=torch.int32, device=p.device)
        if self.rank == 0:
            _queue[(1,)](
                flags, descriptor, self.r, self.r, triton.next_power_of_2(self.r)
            )
        dist.broadcast(descriptor, 0)

        def repair_leaf(start, stop):
            n = stop - start
            ids = descriptor[start:stop]
            gather_peer_rows[(n, triton.cdiv(self.v, 2048))](
                *self.peers, ids, self.scratch, self.lv, self.v, 2048
            )
            torch.softmax(self.scratch[:n], -1, out=self.probs[:n])
            patch_from_softmax[(1,)](
                p, self.probs, ids, self.targets, n, self.v, triton.next_power_of_2(n)
            )

        graph.begin_capture_to_if_node(descriptor[-1] > 0)
        first = min(self.capacity, self.r)
        repair_leaf(0, first)
        if self.r > first:
            graph.begin_capture_to_if_node(descriptor[-1] > first)
            # Rare overflow scans bounded chunks. Keeping a single overflow
            # body avoids one CUDA stream/conditional handle per chunk.
            for start in range(first, self.r, self.capacity):
                repair_leaf(start, min(start + self.capacity, self.r))
            graph.end_capture_to_conditional_node()
        graph.end_capture_to_conditional_node()
        output = self.finish(p)
        return (*output, descriptor[-1])

    def capture(self):
        # Compile kernels and initialize NCCL outside conditional capture.
        super().fixed()
        dist.barrier()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin()
            output = self._body(graph)
            graph.capture_end()
        stream.synchronize()
        return graph, output
