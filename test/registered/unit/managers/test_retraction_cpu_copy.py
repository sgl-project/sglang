import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.schedule_batch import FINISH_ABORT, ScheduleBatch
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MHATokenToKVPoolMXFP8,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    PageMajorMHATokenToKVPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(rid: str):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1],
        output_ids=[2],
        priority=None,
        to_finish=None,
    )


def _server_args():
    return SimpleNamespace(
        disaggregation_mode="decode",
        disaggregation_decode_enable_offload_kvcache=False,
    )


def _batch(cpu_copy_supported: bool, req_count: int = 2):
    batch = ScheduleBatch(reqs=[_req(str(i)) for i in range(req_count)])
    batch.token_to_kv_pool_allocator = SimpleNamespace(
        supports_cpu_copy=Mock(return_value=cpu_copy_supported)
    )
    batch._get_decode_retraction_order = Mock(return_value=list(range(req_count)))
    batch.check_decode_mem = Mock(return_value=True)
    batch.release_req = Mock()
    batch.filter_batch = Mock()
    return batch


class TestKVCacheCPUCopyCapability(unittest.TestCase):
    def test_mha_support_depends_on_layout(self):
        pool = object.__new__(MHATokenToKVPool)
        pool.kv_cache_layout = "nhd"
        pool.k_scale_buffer = None
        pool.v_scale_buffer = None
        self.assertTrue(pool.supports_cpu_copy())

        pool.kv_cache_layout = "hnd"
        self.assertFalse(pool.supports_cpu_copy())

        pool.kv_cache_layout = "vectorized_5d"
        self.assertFalse(pool.supports_cpu_copy())

    def test_mha_with_separate_scale_buffers_is_unsupported(self):
        pool = object.__new__(MHATokenToKVPool)
        pool.kv_cache_layout = "nhd"
        pool.k_scale_buffer = object()
        pool.v_scale_buffer = object()
        self.assertFalse(pool.supports_cpu_copy())

    def test_unsupported_mha_variants_report_false(self):
        self.assertFalse(object.__new__(PageMajorMHATokenToKVPool).supports_cpu_copy())
        self.assertFalse(object.__new__(MHATokenToKVPoolFP4).supports_cpu_copy())
        self.assertFalse(object.__new__(MHATokenToKVPoolMXFP8).supports_cpu_copy())

    def test_mla_variants_and_hisparse_report_distinct_capabilities(self):
        self.assertTrue(object.__new__(MLATokenToKVPool).supports_cpu_copy())
        self.assertFalse(object.__new__(MLATokenToKVPoolFP4).supports_cpu_copy())
        self.assertFalse(object.__new__(HiSparseDSATokenToKVPool).supports_cpu_copy())

    def test_swa_requires_both_subpools_to_support_cpu_copy(self):
        pool = object.__new__(SWAKVPool)
        pool.full_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: True)
        pool.swa_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: True)
        self.assertTrue(pool.supports_cpu_copy())

        pool.swa_kv_pool = SimpleNamespace(supports_cpu_copy=lambda: False)
        self.assertFalse(pool.supports_cpu_copy())


class TestPDDecodeRetractionCPUCopy(unittest.TestCase):
    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_supported_pool_retracts_and_resumes_with_async_offload_disabled(
        self, _estimate
    ):
        batch = _batch(cpu_copy_supported=True)

        retracted, ratio, aborted = batch.retract_decode(_server_args())

        self.assertEqual([req.rid for req in retracted], ["1"])
        self.assertEqual(aborted, [])
        self.assertEqual(ratio, 0.5)
        batch.release_req.assert_called_once_with(1, 1, _server_args(), offload_kv=True)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_unsupported_pool_aborts_only_retracted_request(self, _estimate):
        batch = _batch(cpu_copy_supported=False)
        args = _server_args()

        retracted, _, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual([req.rid for req in aborted], ["1"])
        self.assertIsInstance(aborted[0].to_finish, FINISH_ABORT)
        self.assertIn(
            "does not support synchronous CPU save and restore",
            aborted[0].to_finish.message,
        )
        batch.release_req.assert_called_once_with(1, 1, args, offload_kv=False)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_incomplete_round_trip_pools_abort_without_offload(self, _estimate):
        vectorized = object.__new__(MHATokenToKVPool)
        vectorized.kv_cache_layout = "vectorized_5d"
        vectorized.k_scale_buffer = None
        vectorized.v_scale_buffer = None

        for pool in (
            vectorized,
            object.__new__(MHATokenToKVPoolFP4),
            object.__new__(MLATokenToKVPoolFP4),
        ):
            with self.subTest(pool=type(pool).__name__):
                batch = _batch(cpu_copy_supported=pool.supports_cpu_copy())
                args = _server_args()

                retracted, _, aborted = batch.retract_decode(args)

                self.assertEqual(retracted, [])
                self.assertEqual([req.rid for req in aborted], ["1"])
                batch.release_req.assert_called_once_with(1, 1, args, offload_kv=False)

    @patch(
        "sglang.srt.managers.schedule_batch."
        "NewTokenRatioTracker.estimate_new_token_ratio_after_retract",
        return_value=0.5,
    )
    def test_last_request_abort_never_wastes_cpu_copy(self, _estimate):
        batch = _batch(cpu_copy_supported=True, req_count=1)
        batch.check_decode_mem = Mock(return_value=False)
        args = _server_args()

        retracted, _, aborted = batch.retract_decode(args)

        self.assertEqual(retracted, [])
        self.assertEqual([req.rid for req in aborted], ["0"])
        batch.release_req.assert_called_once_with(0, 0, args, offload_kv=False)


if __name__ == "__main__":
    unittest.main()
