"""A DeepSeek-V4 request retracted on a PD decode server is recomputed by the
prefill instead of backed up: the pool has no host copy of its KV."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import backup_kv_cache
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _allocator(pool_cls):
    # Buffers are never touched: the pool class alone decides the recovery path.
    pool = pool_cls.__new__(pool_cls)
    pool.device = "cpu"
    return SimpleNamespace(get_kvcache=lambda: pool)


def _req(output_ids, multimodal_inputs=None):
    return SimpleNamespace(
        rid="req",
        kv=ReqKvInfo(),
        seqlen=8,
        output_ids=list(output_ids),
        multimodal_inputs=multimodal_inputs,
        time_stats=MagicMock(),
        offload_kv_cache=MagicMock(),
    )


class TestDSV4RetractRebootstrap(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")

    def test_dsv4_backup_is_skipped(self):
        # Without DSpark the backend resolves to cpu_tensor, whose host copy the
        # DeepSeek-V4 pool does not implement.
        req = _req([1])
        saved = backup_kv_cache(
            req, None, None, _allocator(DeepSeekV4TokenToKVPool), "cpu_tensor"
        )
        self.assertTrue(saved)
        req.offload_kv_cache.assert_not_called()

    def test_dsv4_aborts_what_the_prefill_cannot_recompute(self):
        cases = {
            # An explicit none still means abort, not recompute.
            "none backend": (_req([1]), "none"),
            # The recompute payload carries token ids only.
            "image request": (_req([1], multimodal_inputs=object()), "cpu_tensor"),
        }
        for name, (req, backend) in cases.items():
            with self.subTest(case=name):
                saved = backup_kv_cache(
                    req, None, None, _allocator(DeepSeekV4TokenToKVPool), backend
                )
                self.assertFalse(saved)
                req.offload_kv_cache.assert_not_called()

    def test_other_pools_keep_the_cpu_tensor_backup(self):
        req = _req([1])
        backup_kv_cache(req, None, None, _allocator(MHATokenToKVPool), "cpu_tensor")
        req.offload_kv_cache.assert_called_once()

    def test_retracted_dsv4_request_rebootstraps(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.token_to_kv_pool_allocator = _allocator(DeepSeekV4TokenToKVPool)
        queue.retracted_queue = []
        queue.pending_reqs = []
        queue._check_if_req_exceed_kv_capacity = MagicMock(return_value=False)
        queue._create_receiver_and_enqueue = MagicMock()
        queue._resolve_prefill_dp_rank = MagicMock(return_value=None)
        req = _req([10, 11, 12])
        req.bootstrap_host = "prefill"

        queue.add(req, is_retracted=True)

        self.assertEqual(queue.retracted_queue, [])
        queue._create_receiver_and_enqueue.assert_called_once_with(
            req, is_rebootstrap=True
        )
        # The already-emitted boundary token is replayed, not re-sampled.
        self.assertEqual(req.output_ids, [10, 11])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 12)
        self.assertTrue(req.pd_rebootstrap_in_progress)


if __name__ == "__main__":
    unittest.main()
