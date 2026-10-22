"""Delayed H2D must stay ordered with captured KDA-style state reads."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    break_graph,
)
from sglang.srt.model_executor.runner_backend_utils.hicache_graph_events import (
    HiCacheGraphEvents,
    capture_layer_wait,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHiCacheGraphEvents(unittest.TestCase):
    def test_delayed_load_slot_rotation_and_overlap(self):
        counter = LayerDoneCounter(2)
        pool = SimpleNamespace(
            mamba_map={0: 0, 1: 1}, start_layer=0, layer_transfer_counter=counter
        )
        owner = HiCacheGraphEvents(pool)
        graph = BreakableCUDAGraph(hicache_graph_events=owner)
        states = [torch.zeros(1, device="cuda") for _ in range(2)]
        output = [torch.zeros_like(x) for x in states]
        early = torch.cuda.Event(external=True, enable_timing=True)
        late = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with owner.capture_scope(), BreakableCUDAGraphCapture(graph, stream=stream):
            capture_layer_wait(0)
            output[0].copy_(states[0])
            early.record()
            break_graph()
            capture_layer_wait(1)
            output[1].copy_(states[1])
        self.assertEqual(
            [[wait.layer for wait in segment] for segment in graph._hicache_wait_nodes],
            [[0], [1]],
        )
        self.assertEqual(len(graph._segments), 2)
        h2d = torch.cuda.Stream()
        slots = []
        for repeat in range(6):
            for tensor in states:
                tensor.zero_()
            torch.cuda.synchronize()
            slot = counter.update_producer()
            slots.append(slot)
            host = [
                torch.tensor([repeat + 1.0], pin_memory=True),
                torch.tensor([repeat + 101.0], pin_memory=True),
            ]
            with torch.cuda.stream(h2d):
                torch.cuda._sleep(20_000_000)
                states[0].copy_(host[0], non_blocking=True)
                counter.events[slot].complete(0)
                torch.cuda._sleep(100_000_000)
                states[1].copy_(host[1], non_blocking=True)
                late.record()
                counter.events[slot].complete(1)
            counter.set_consumer(slot)
            graph.replay()
            torch.cuda.synchronize()
            self.assertGreater(
                early.elapsed_time(late),
                0,
                "An early layer waited for the entire H2D load",
            )
            self.assertEqual(output[0].item(), repeat + 1)
            self.assertEqual(output[1].item(), repeat + 101)
        self.assertEqual(slots, [0, 1, 2, 0, 1, 2])
        counter.set_consumer(-1)
        states[0].fill_(777)
        states[1].fill_(888)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual([x.item() for x in output], [777, 888])


if __name__ == "__main__":
    unittest.main()
