"""Exact-batch conditioner graphs; no padding or persistent request-state aliasing."""

import torch


class ConditionerGraphs:
    @torch.no_grad()
    def __init__(self, module, max_batch, owner=None):
        self.original = module.forward
        self.module = module
        self.owner = owner
        fusion = hasattr(module, "linear_fc")
        weight = module.linear_fc.weight if fusion else module.down.weight
        width = module.linear_fc.out_features if fusion else module.up.out_features
        self.weight_ptrs = tuple(p.data_ptr() for p in module.parameters())
        self.blocks = torch.zeros(
            max_batch,
            getattr(module, "draft_slots", 8),
            width,
            dtype=weight.dtype,
            device=weight.device,
        )
        self.hidden = torch.zeros(
            max_batch, width, dtype=weight.dtype, device=weight.device
        )
        self.graphs, self.outputs = {}, {}
        self.verified = set()
        self.hits = self.fallbacks = 0
        stream = torch.cuda.Stream(device=weight.device)
        stream.wait_stream(torch.cuda.current_stream())
        pool = torch.cuda.graph_pool_handle()
        for bs in range(max_batch, 0, -1):
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.original(self.blocks[:bs], self.hidden[:bs])
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = self.original(self.blocks[:bs], self.hidden[:bs])
            self.graphs[bs], self.outputs[bs] = graph, output
        torch.cuda.current_stream().wait_stream(stream)

    @torch.no_grad()
    def run(self, blocks, hidden):
        bs = blocks.shape[0]
        valid = (
            bs in self.graphs
            and blocks.shape[1:] == self.blocks.shape[1:]
            and hidden.shape == (bs, self.hidden.shape[1])
            and blocks.dtype == self.blocks.dtype
            and hidden.dtype == self.hidden.dtype
            and blocks.device == self.blocks.device
            and hidden.device == self.hidden.device
            and self.weight_ptrs
            == tuple(p.data_ptr() for p in self.module.parameters())
        )
        if self.owner is not None:
            self.owner.graph_used["conditioner"] = valid
        if not valid:
            self.fallbacks += 1
            return self.original(blocks, hidden)
        self.blocks[:bs].copy_(blocks)
        self.hidden[:bs].copy_(hidden)
        self.graphs[bs].replay()
        self.hits += 1
        if bs not in self.verified:
            reference = self.original(blocks, hidden)
            torch.testing.assert_close(self.outputs[bs], reference, rtol=0, atol=0)
            self.verified.add(bs)
        # The DFlash runner copies these embeddings before the next replay.
        return self.outputs[bs]
