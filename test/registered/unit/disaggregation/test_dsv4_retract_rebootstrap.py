import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.common import backup_kv_cache
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _allocator(pool_cls):
    # Buffers are never touched: the pool class alone decides the recovery path.
    pool = pool_cls.__new__(pool_cls)
    pool.device = "cpu"
    return SimpleNamespace(get_kvcache=lambda: pool)


def _req(multimodal_inputs=None):
    return SimpleNamespace(
        kv=ReqKvInfo(),
        seqlen=8,
        multimodal_inputs=multimodal_inputs,
        offload_kv_cache=MagicMock(),
    )


class TestDSV4RetractRebootstrap(CustomTestCase):
    def test_dsv4_backup_decision(self):
        cases = [
            # The default backend; the pool has no host copy, so the prefill recomputes.
            ("default backend", _req(), "cpu_tensor", True),
            # An explicit none still means abort, not recompute.
            ("none backend", _req(), "none", False),
            # The recompute payload carries token ids only.
            ("image request", _req(multimodal_inputs=object()), "cpu_tensor", False),
        ]
        for name, req, backend, saved in cases:
            with self.subTest(case=name):
                self.assertEqual(
                    backup_kv_cache(
                        req, None, None, _allocator(DeepSeekV4TokenToKVPool), backend
                    ),
                    saved,
                )
                req.offload_kv_cache.assert_not_called()

    def test_fake_transfer_skips_prefill_recompute(self):
        # Neither the fake manager nor a fake receiver can dispatch a recompute.
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        queue.kv_manager = MagicMock()
        req = MagicMock(bootstrap_host=FAKE_BOOTSTRAP_HOST)
        queue._submit_prefill_recompute(DecodeRequest(req=req, kv_receiver=MagicMock()))
        queue.kv_manager.submit_prefill_recompute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
