import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SwaCache(RadixCache):
    def __init__(self, *, auxiliary=False):
        self.sliding_window_size = 128
        self.page_size = 16
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.arange(1024).reshape(1, -1)
        )
        self.token_to_kv_pool_allocator = Mock(
            spec=["free_swa_segment", "free_group_begin", "free_group_end"]
        )
        self.trim_calls = 0
        self.auxiliary = auxiliary

    def supports_swa(self):
        return True

    def supports_auxiliary_swa(self):
        return self.auxiliary

    def evict_sliding_windows(self, req, pre_len, *, eviction_interval=1):
        self.trim_calls += 1
        super().evict_sliding_windows(req, pre_len, eviction_interval=eviction_interval)


class TestSWAEvictionInterval(unittest.TestCase):
    def test_scheduler_gates_on_request_cursor_before_protected_prefix(self):
        for (protected, shield), auxiliary in product(
            [(256, 0), (0, 249)], [False, True]
        ):
            with (
                self.subTest(protected=protected, shield=shield, auxiliary=auxiliary),
                envs.SGLANG_SWA_EVICTION_INTERVAL.override(128),
                envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.override(False),
            ):
                cache = _SwaCache(auxiliary=auxiliary)
                req = SimpleNamespace(
                    seqlen=401,
                    decode_batch_idx=1,
                    kv=ReqKvInfo(
                        req_pool_idx=0,
                        cache_protected_len=protected,
                        swa_evict_floor=shield,
                    ),
                )
                batch = ScheduleBatch(
                    reqs=[req],
                    tree_cache=cache,
                    req_to_token_pool=cache.req_to_token_pool,
                    token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
                    forward_mode=ForwardMode.DECODE,
                )
                free = cache.token_to_kv_pool_allocator.free_swa_segment

                # The raw cursor makes eviction due; the protected prefix only
                # limits which rows may be freed, not when the interval starts.
                batch.maybe_evict_swa()
                self.assertEqual(cache.trim_calls, 1)
                free.assert_called_once()
                self.assertTrue(
                    torch.equal(free.call_args.args[0], torch.arange(256, 272))
                )
                self.assertEqual(free.call_args.kwargs, {"start_pos": 256})
                self.assertEqual(req.kv.get_evicted_seqlen(ComponentType.SWA), 272)
                free.reset_mock()

                req.seqlen = 528
                batch.maybe_evict_swa()
                self.assertEqual(cache.trim_calls, 1 + auxiliary)
                free.assert_not_called()

                req.seqlen += 1
                batch.maybe_evict_swa()
                self.assertEqual(cache.trim_calls, 2 + auxiliary)
                free.assert_called_once()
                self.assertTrue(
                    torch.equal(free.call_args.args[0], torch.arange(272, 400))
                )

    def test_forward_cadence_is_not_gated_again_by_token_interval(self):
        cache = _SwaCache()
        req = SimpleNamespace(
            seqlen=201,
            decode_batch_idx=1,
            kv=ReqKvInfo(req_pool_idx=0),
        )
        batch = ScheduleBatch(
            reqs=[req],
            tree_cache=cache,
            req_to_token_pool=cache.req_to_token_pool,
            token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
            forward_mode=ForwardMode.DECODE,
            forward_iter=127,
        )
        free = cache.token_to_kv_pool_allocator.free_swa_segment
        with (
            patch("sglang.srt.managers.schedule_batch._is_hip", True),
            envs.SGLANG_AMD_USE_FLYDSL_MEGA_MOE.override(True),
            envs.SGLANG_SWA_EVICTION_INTERVAL.override(128),
            envs.SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW.override(False),
        ):
            batch.maybe_evict_swa()
            free.assert_not_called()
            batch.forward_iter += 1
            batch.maybe_evict_swa()

        free.assert_called_once()
        self.assertTrue(torch.equal(free.call_args.args[0], torch.arange(64)))
        self.assertEqual(req.kv.get_evicted_seqlen(ComponentType.SWA), 64)


if __name__ == "__main__":
    unittest.main()
