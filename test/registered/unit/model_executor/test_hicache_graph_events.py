"""CPU coverage for live HiCache dependencies in captured KDA segments."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import sglang.srt.model_executor.runner_backend_utils.hicache_graph_events as events
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    cuda_utils,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiCacheGraphEvents(unittest.TestCase):
    def setUp(self):
        self.rt = SimpleNamespace(
            cudaError_t=SimpleNamespace(cudaSuccess=0),
            cudaGraphNodeType=SimpleNamespace(cudaGraphNodeTypeWaitEvent=1),
            cudaGraphExecEventWaitNodeSetEvent=Mock(return_value=(0,)),
            cudaGetErrorString=Mock(return_value=(0, b"invalid event")),
        )
        self.enterContext(patch.object(events, "rt", self.rt))
        self.enterContext(patch.object(cuda_utils, "rt", self.rt))
        self.ready = [
            SimpleNamespace(cuda_event=100 + i, record=Mock(), wait=Mock())
            for i in range(2)
        ]
        self.enterContext(
            patch.object(events.torch.cuda, "Event", side_effect=self.ready)
        )
        self.enterContext(patch.object(events.torch.cuda, "synchronize"))
        self.counter = SimpleNamespace(
            consumer_index=-1,
            num_layers=6,
            events=[
                SimpleNamespace(
                    load_events=[
                        SimpleNamespace(cuda_event=1000 + s * 10 + i) for i in range(6)
                    ]
                )
                for s in range(3)
            ],
        )
        self.pool = SimpleNamespace(
            mamba_map={4: 0, 8: 1}, start_layer=4, layer_transfer_counter=self.counter
        )
        self.owner = events.HiCacheGraphEvents(self.pool)
        self.graphs = [
            SimpleNamespace(raw_cuda_graph_exec=lambda: 20),
            SimpleNamespace(raw_cuda_graph_exec=lambda: 21),
        ]
        self.bindings = [
            [events.LayerEventWait(30, 4, 100)],
            [events.LayerEventWait(31, 8, 101)],
        ]

    def test_rotating_slots_and_no_load_reset(self):
        update = self.rt.cudaGraphExecEventWaitNodeSetEvent
        for slot in [0, 1, 2, 0]:
            self.counter.consumer_index = slot
            self.owner.bind(self.graphs, self.bindings)
            self.assertEqual(
                update.call_args_list[-2:],
                [call(20, 30, 1000 + 10 * slot), call(21, 31, 1004 + 10 * slot)],
            )
        count = update.call_count
        self.owner.bind(self.graphs, self.bindings)
        self.assertEqual(update.call_count, count)
        self.counter.consumer_index = -1
        self.owner.bind(self.graphs, self.bindings)
        self.assertEqual(
            update.call_args_list[-2:], [call(20, 30, 100), call(21, 31, 101)]
        )
        self.pool.layer_transfer_counter = None
        self.owner.bind(self.graphs, self.bindings)
        self.assertEqual(update.call_count, count + 2)

    def test_cpu_future_counter_waits_all_layers_and_resets_bindings(self):
        self.counter.consumer_index = 1
        self.owner.bind(self.graphs, self.bindings)
        cpu = SimpleNamespace(consumer_index=19, num_layers=6, wait_until=Mock())
        self.pool.layer_transfer_counter = cpu
        self.owner.bind(self.graphs, self.bindings)
        self.assertEqual(cpu.wait_until.call_args_list, [call(i) for i in range(6)])
        self.assertEqual(
            self.rt.cudaGraphExecEventWaitNodeSetEvent.call_args_list[-2:],
            [call(20, 30, 100), call(21, 31, 101)],
        )

    def test_wait_is_scoped_to_capture(self):
        with patch.object(
            events.torch.cuda, "is_current_stream_capturing", return_value=True
        ):
            events.capture_layer_wait(4)
            self.ready[0].wait.assert_not_called()
            with self.owner.capture_scope():
                events.capture_layer_wait(4)
                self.ready[0].wait.assert_called_once_with()
            events.capture_layer_wait(4)
            self.ready[0].wait.assert_called_once_with()

    def test_eager_variant_keeps_its_live_waits(self):
        cpu = SimpleNamespace(consumer_index=19, num_layers=6, wait_until=Mock())
        self.pool.layer_transfer_counter = cpu
        self.owner.bind(self.graphs, [[], []])
        cpu.wait_until.assert_not_called()
        self.rt.cudaGraphExecEventWaitNodeSetEvent.assert_not_called()

    def test_conservative_fallback_waits_each_layer(self):
        self.counter.consumer_index = 2
        self.counter.wait_until = Mock()
        events.wait_for_layer_loads(self.pool)
        self.assertEqual(
            self.counter.wait_until.call_args_list, [call(i) for i in range(6)]
        )

    def test_conservative_fallback_without_an_active_load(self):
        self.counter.wait_until = Mock()
        events.wait_for_layer_loads(self.pool)
        self.counter.wait_until.assert_not_called()
        self.pool.layer_transfer_counter = None
        events.wait_for_layer_loads(self.pool)

    def test_warmup_does_not_insert_a_wait(self):
        with (
            self.owner.capture_scope(),
            patch.object(
                events.torch.cuda, "is_current_stream_capturing", return_value=False
            ),
        ):
            events.capture_layer_wait(4)
        self.ready[0].wait.assert_not_called()

    def test_only_own_event_wait_nodes_are_bound(self):
        self.rt.cudaGraphGetNodes = Mock(side_effect=[(0, [], 4), (0, [1, 2, 3, 4], 4)])
        self.rt.cudaGraphNodeGetType = Mock(
            side_effect=[(0, 0), (0, 1), (0, 1), (0, 1)]
        )
        self.rt.cudaGraphEventWaitNodeGetEvent = Mock(
            side_effect=[(0, 100), (0, 999), (0, 101)]
        )
        graph = SimpleNamespace(raw_cuda_graph=lambda: 50)
        self.assertEqual(
            self.owner.find_wait_nodes(graph),
            [events.LayerEventWait(2, 4, 100), events.LayerEventWait(4, 8, 101)],
        )
        self.rt.cudaGraphGetNodes = Mock(return_value=(0, [], 0))
        self.assertEqual(self.owner.find_wait_nodes(graph), [])

    def test_failed_update_does_not_cache_handle(self):
        self.counter.consumer_index = 0
        self.rt.cudaGraphExecEventWaitNodeSetEvent.side_effect = [(1,), (0,), (0,)]
        with self.assertRaisesRegex(RuntimeError, "CUDA error"):
            self.owner.bind(self.graphs, self.bindings)
        self.assertEqual(self.bindings[0][0].event_handle, 100)
        self.owner.bind(self.graphs, self.bindings)
        self.assertEqual(self.bindings[0][0].event_handle, 1000)

    def test_segment_graphs_retain_raw_nodes(self):
        graph = bcg.BreakableCUDAGraph(hicache_graph_events=self.owner)
        capture = bcg.BreakableCUDAGraphCapture(graph)
        raw = Mock()
        with patch.object(bcg.torch.cuda, "CUDAGraph", return_value=raw) as ctor:
            capture._begin_new_segment()
            ctor.assert_called_once_with(keep_graph=True)
            self.assertTrue(capture._current_graph_needs_instantiate)
        with patch.object(self.owner, "find_wait_nodes", return_value=self.bindings[0]):
            graph._append_segment(raw, True)
        raw.instantiate.assert_called_once_with()
        self.assertEqual(graph._hicache_wait_nodes, self.bindings[:1])


if __name__ == "__main__":
    unittest.main()
