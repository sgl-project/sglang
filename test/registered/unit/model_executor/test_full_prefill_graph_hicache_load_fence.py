import unittest
from types import SimpleNamespace

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _counter(consumer_index):
    counter = object.__new__(LayerDoneCounter)
    counter.events = [SimpleNamespace(finish_event=f"finish-{i}") for i in range(3)]
    counter.producer_index = consumer_index
    counter.consumer_index = consumer_index
    return counter


def _runner(counter):
    waited = []
    stream = SimpleNamespace(wait_event=waited.append)
    runner = object.__new__(PrefillCudaGraphRunner)
    runner.device_module = SimpleNamespace(current_stream=lambda: stream)
    runner.model_runner = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(layer_transfer_counter=counter)
    )
    return runner, waited


class TestFullPrefillGraphHiCacheLoadFence(CustomTestCase):
    def test_pending_load_back_waits_on_final_event(self):
        runner, waited = _runner(_counter(consumer_index=1))
        runner._wait_for_hicache_load_back()
        self.assertEqual(waited, ["finish-1"])

    def test_no_pending_load_back_no_wait(self):
        runner, waited = _runner(_counter(consumer_index=-1))
        runner._wait_for_hicache_load_back()
        self.assertEqual(waited, [])

    def test_no_hicache_no_wait(self):
        runner, waited = _runner(None)
        runner._wait_for_hicache_load_back()
        self.assertEqual(waited, [])


if __name__ == "__main__":
    unittest.main()
