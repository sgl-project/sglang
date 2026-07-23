import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _mk_req():
    return SimpleNamespace(
        req_pool_idx=None,
        inflight_middle_chunks=0,
        kv_committed_len=0,
    )


class TestReqToTokenPoolAlloc(CustomTestCase):
    def _pool(self, size=8):
        return ReqToTokenPool(
            size=size,
            max_context_len=16,
            device=get_device(),
            enable_memory_saver=False,
        )

    def test_alloc_free_alloc_roundtrip_no_duplicates(self):
        # Derived property: tail-pop + tail-append must still yield disjoint
        # slots across an alloc/free/alloc cycle (this is the LIFO-reuse
        # behavior this change introduces relative to the old FIFO order).
        pool = self._pool(size=8)
        reqs = [_mk_req() for _ in range(8)]
        pool.alloc(reqs)
        self.assertEqual(pool.available_size(), 0)
        for r in reqs[:3]:
            pool.free(r)
        self.assertEqual(pool.available_size(), 3)
        new_reqs = [_mk_req() for _ in range(3)]
        new_indices = pool.alloc(new_reqs)
        self.assertEqual(len(set(new_indices)), 3)
        self.assertEqual(pool.available_size(), 0)

    def test_alloc_with_all_reqs_reusing_is_noop_on_free_slots(self):
        # Bug regression: need_size == 0 is the `-0` slicing footgun
        # (`free_slots[-0:]` is the *entire* list, not `[]`). A naive
        # `self.free_slots[-need_size:]` / `del self.free_slots[-need_size:]`
        # without this guard would silently drain the whole free pool
        # whenever every request in the batch is reusing its slot (pure
        # chunked-prefill continuation batches).
        pool = self._pool(size=4)
        reqs = [_mk_req() for _ in range(2)]
        pool.alloc(reqs)
        before = pool.available_size()
        for r in reqs:
            r.kv_committed_len = 1  # satisfy the "reusing" assertion
        indices = pool.alloc(reqs)  # all reqs already have req_pool_idx
        self.assertEqual(indices, [r.req_pool_idx for r in reqs])
        self.assertEqual(pool.available_size(), before)  # unchanged, not emptied

    def test_alloc_exhausts_pool_then_returns_none(self):
        # Derived property: the need_size > len(free_slots) boundary check
        # must still reject over-allocation after switching which end of
        # free_slots is consumed.
        pool = self._pool(size=4)
        reqs = [_mk_req() for _ in range(4)]
        self.assertIsNotNone(pool.alloc(reqs))
        self.assertEqual(pool.available_size(), 0)
        self.assertIsNone(pool.alloc([_mk_req()]))


if __name__ == "__main__":
    unittest.main()
