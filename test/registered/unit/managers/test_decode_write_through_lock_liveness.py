"""Regression coverage for PD decode HiCache write-through ack draining."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _TreeCache:
    def __init__(self, cache_controller, ready_acks):
        self.cache_controller = cache_controller
        self.ready_acks = ready_acks
        self.check_hicache_events_calls = 0

    def check_hicache_events(self):
        self.check_hicache_events_calls += 1
        self.ready_acks.clear()


def _scheduler(tree_cache, *, enable_decode_hicache):
    return SimpleNamespace(
        enable_decode_hicache=enable_decode_hicache,
        tree_cache=tree_cache,
        waiting_queue=[],
        disagg_decode_prealloc_queue=SimpleNamespace(
            retracted_queue=[object()],
            resume_retracted_reqs=lambda: [],
        ),
        disagg_decode_transfer_queue=SimpleNamespace(
            resolve_deferred_releases=lambda: None
        ),
    )


class TestDecodeWriteThroughLockLiveness(unittest.TestCase):
    def test_retraction_only_host_pool_drains_on_every_rank(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_decode_retraction_backup="host_pool",
            hicache_ratio=2.0,
        )
        set_global_server_args_for_scheduler(server_args)

        self.assertFalse(server_args.enable_hierarchical_cache)
        enable_decode_hicache = (
            server_args.disaggregation_decode_enable_radix_cache
            and server_args.enable_hierarchical_cache
        )
        self.assertFalse(enable_decode_hicache)

        # A host-pool retraction cache has a controller on every TP rank. Rank
        # local ack readiness must not affect whether ranks enter the drain.
        tree_caches = [
            _TreeCache(cache_controller=object(), ready_acks=[object()]),
            _TreeCache(cache_controller=object(), ready_acks=[]),
        ]
        self.assertTrue(
            all(cache.cache_controller is not None for cache in tree_caches)
        )

        for tree_cache in tree_caches:
            SchedulerDisaggregationDecodeMixin.process_decode_queue(
                _scheduler(
                    tree_cache,
                    enable_decode_hicache=enable_decode_hicache,
                )
            )

        self.assertEqual(
            [cache.check_hicache_events_calls for cache in tree_caches], [1, 1]
        )
        self.assertEqual([cache.ready_acks for cache in tree_caches], [[], []])

    def test_no_controller_skips_event_drain(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="decode")
        set_global_server_args_for_scheduler(server_args)
        tree_cache = _TreeCache(cache_controller=None, ready_acks=[])

        SchedulerDisaggregationDecodeMixin.process_decode_queue(
            _scheduler(tree_cache, enable_decode_hicache=False)
        )

        self.assertEqual(tree_cache.check_hicache_events_calls, 0)


if __name__ == "__main__":
    unittest.main()
