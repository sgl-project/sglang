"""Prefetch transfer accounting tests."""

from dataclasses import dataclass
from queue import Queue
from typing import Optional

import pytest
from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.sglang.cache_controller import C_HiCacheController
from sglang_simulator.simulation.sglang.hiradix_cache import C_HiRadixCacheHook
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager
from sglang_simulator.simulation.sglang.unified_radix_cache import (
    C_UnifiedRadixCacheHook,
)


@dataclass
class Operation:
    request_id: str
    host_indices: Optional[list[int]] = None
    completed_tokens: float = 0
    _terminated_flag: bool = False

    def mark_terminate(self):
        self._terminated_flag = True


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    StateManager.reset()
    request_stats_manager.reset()
    monkeypatch.setattr(C_HiCacheController, "KV_CACHE_BYTES", 1)
    monkeypatch.setattr(C_HiCacheController, "DISK_READ_BANDWIDTH_BYTES", 1000)
    yield
    StateManager.reset()
    request_stats_manager.reset()


def make_cache(hook):
    class Controller:
        def __init__(self):
            self.enable_storage = True
            self.page_size = 1
            self.prefetch_threshold = 1
            self.prefetch_queue = Queue()
            self.prefetch_hit_queue = Queue()
            self.backup_queue = Queue()
            self.ack_backup_queue = Queue()
            self.backup_skip = False

        def _storage_hit_query(self, operation):
            return [f"page{i}" for i in range(1024)], 1024

        def terminate_prefetch(self, operation):
            operation.mark_terminate()
            return operation.completed_tokens, operation.hash_value

        def append_host_mem_release(self, host_indices):
            pass

        def _page_backup(self, operation):
            pass

    C_HiCacheController.hook(Controller)

    class Cache:
        def __init__(self):
            self.cache_controller = Controller()

        def check_hicache_events(self):
            controller = self.cache_controller
            while not controller.prefetch_hit_queue.empty():
                operation = controller.prefetch_hit_queue.get_nowait()
                operation.host_indices = list(range(operation.storage_hit_count))
                controller.prefetch_buffer.put(operation)

    hook.hook(Cache)
    return Cache()


@pytest.mark.parametrize("hook", [C_HiRadixCacheHook, C_UnifiedRadixCacheHook])
def test_partial_transfer_advances_once_per_cache_poll(hook):
    cache = make_cache(hook)
    operation = Operation("request")
    cache.cache_controller.prefetch_queue.put(operation)

    StateManager.set_current_inference_dur(0.064)
    cache.check_hicache_events()
    assert operation.completed_tokens == 64

    StateManager.step_global_clock(0.010)
    StateManager.set_current_inference_dur(0.010)
    cache.check_hicache_events()

    assert operation.completed_tokens == pytest.approx(74)
