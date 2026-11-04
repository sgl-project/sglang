import sys
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, '/tmp/sglang-rfc-build/mem08-native-worktree/test/registered/unit/mem_cache')
import test_kv_control_transfers as fixtures
from sglang.srt.managers.kv_control.service import CacheLimits, ServiceConfig
from sglang.srt.managers.kv_control.contracts import Status
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle

class Repro(fixtures.TestKVControlStorageReads):
    def runTest(self):
        config = ServiceConfig(namespace='ns', engine_id='engine', cache_limits=CacheLimits(max_concurrent_storage_reads=1))
        with fixtures.hicache(self, config, namespace_keys=True) as owner:
            _, cache, service, _ = owner
            scopes = [fixtures.cache_fixtures.target(cache_salt=f'repro-{i}') for i in range(2)]
            self._cold_scopes(owner, scopes)
            cc = cache.cache_controller
            query, get = cc._storage_hit_query, cc.page_get_func
            first_get, ordinary_get, release_first, release_ordinary = (threading.Event() for _ in range(4))
            queries, gets = [], []
            demand = CacheRequestHandle('queued-demand-repro', 0)
            def counted_query(operation):
                queries.append(operation.request_id)
                return query(operation)
            def controlled_get(operation, *args, **kwargs):
                gets.append(operation.request_id)
                if len(gets) == 1:
                    first_get.set()
                    assert release_first.wait(5)
                if operation.request_id == demand.rid:
                    ordinary_get.set()
                    assert release_ordinary.wait(5)
                return get(operation, *args, **kwargs)
            try:
                with patch.object(cc, '_storage_hit_query', counted_query), patch.object(cc, 'page_get_func', controlled_get), patch.object(cache, 'prefetch_stop_policy', 'wait_complete'), patch.object(cache, 'prefetch_threshold', 1):
                    first = self._restore(service, scopes[0], 'first')
                    fixtures.wait_for(service, first_get.is_set)
                    second = self._restore(service, scopes[1], 'second')
                    reads = service.control.executor.storage_reads
                    self.assertIsNone(reads.interests[second.ref.operation_id].job.operation)
                    cache.prefetch_from_storage(demand, cache.tree_core.root_node.id, fixtures.cache_fixtures.array('q', scopes[1].token_ids), matched_prefix_tokens=[], cache_salt=service.cache_salt(scopes[1].cache_salt))
                    joined = demand in reads.demands
                    if not joined:
                        fixtures.wait_for(service, lambda: len(queries) == 2)
                        release_first.set()
                        fixtures.wait_for(service, ordinary_get.is_set)
                        fixtures.wait_for(service, lambda: len(queries) == 3)
                    release_first.set()
                    release_ordinary.set()
                    self.assertEqual(fixtures.wait(service, first).status, Status.COMPLETED)
                    self.assertEqual(fixtures.wait(service, second).status, Status.COMPLETED)
                    def ready():
                        cache.drain_storage_control_queues()
                        return cache.check_prefetch_progress(demand)
                    fixtures.wait_for(service, ready)
                    span = cache.pop_prefetch_loaded_span(demand)
                    cache.discard_storage_prefetch_accounting(demand)
                    service.progress()
                    print({'joined_while_queued': joined, 'physical_queries': len(queries), 'physical_gets': len(gets), 'demand_span': span}, flush=True)
                    fixtures.assert_retired(cache, service)
                    cache.sanity_check()
            finally:
                release_first.set()
                release_ordinary.set()
                cache.release_aborted_request(demand)

case = Repro()
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([case]))
raise SystemExit(not result.wasSuccessful())
