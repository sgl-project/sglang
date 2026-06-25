import unittest

from sglang.srt.mem_cache.common import alloc_req_slots
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


class FakeKVPool:
    def __init__(self):
        self.cleared = []

    def clear_c128_req_states(self, req_pool_indices):
        self.cleared.append(list(req_pool_indices))


class FakeAllocator:
    def __init__(self, kv_pool):
        self.kv_pool = kv_pool

    def get_kvcache(self):
        return self.kv_pool


class FakeTreeCache:
    def __init__(self, kv_pool):
        self.token_to_kv_pool_allocator = FakeAllocator(kv_pool)


class FakeReq:
    def __init__(self, req_pool_idx=None):
        self.req_pool_idx = req_pool_idx
        self.inflight_middle_chunks = 0
        self.kv_committed_len = 0


class TestDSV4C128ReqState(unittest.TestCase):
    def _make_pool(self):
        return ReqToTokenPool(
            size=4,
            max_context_len=16,
            device="cpu",
            enable_memory_saver=False,
        )

    def test_new_req_slots_clear_c128_state(self):
        req_to_token_pool = self._make_pool()
        kv_pool = FakeKVPool()
        reqs = [FakeReq(), FakeReq()]

        req_pool_indices = alloc_req_slots(
            req_to_token_pool, reqs, FakeTreeCache(kv_pool)
        )

        self.assertEqual(req_pool_indices, [1, 2])
        self.assertEqual(kv_pool.cleared, [[1, 2]])

    def test_reused_req_slot_skips_c128_state_clear(self):
        req_to_token_pool = self._make_pool()
        req_to_token_pool.free_slots.remove(3)
        kv_pool = FakeKVPool()
        reused_req = FakeReq(req_pool_idx=3)
        reused_req.kv_committed_len = 1
        new_req = FakeReq()

        req_pool_indices = alloc_req_slots(
            req_to_token_pool, [reused_req, new_req], FakeTreeCache(kv_pool)
        )

        self.assertEqual(req_pool_indices, [3, 1])
        self.assertEqual(kv_pool.cleared, [[1]])


if __name__ == "__main__":
    unittest.main()
