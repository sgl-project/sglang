"""NPU startup must initialize custom AiCPU before MemCache maps Host memory."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_SCHEDULER = "sglang.srt.managers.scheduler"
_NPU_UTILS = "sglang.srt.hardware_backend.npu.utils"
_DSV4_BACKEND = "sglang.srt.hardware_backend.npu.attention.ascend_dsv4_backend"


class TestSchedulerNpuInit(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.scheduler = Scheduler.__new__(Scheduler)
        self.scheduler.server_args = SimpleNamespace(
            hicache_storage_backend="npu_memcache"
        )
        self.scheduler.tp_group = SimpleNamespace(device="npu:0")
        self.init_zbal = Mock(side_effect=lambda *args: self.events.append("zbal"))
        self.prewarm_aicpu = Mock(side_effect=lambda *args: self.events.append("aicpu"))
        self.scheduler.maybe_init_hccl_dp_prewarm = Mock(
            side_effect=lambda: self.events.append("hccl")
        )
        self.is_dsv4 = Mock(return_value=True)
        patches = (
            patch(f"{_SCHEDULER}._is_npu", True),
            patch(f"{_SCHEDULER}.is_deepseek_v4", self.is_dsv4),
            patch(
                f"{_SCHEDULER}.get_parallel",
                return_value=SimpleNamespace(pp_size=1, tp_size=8, tp_rank=0),
            ),
            patch(f"{_SCHEDULER}.get_device", return_value=SimpleNamespace(gpu_id=0)),
            patch.dict(
                "sys.modules",
                {
                    _NPU_UTILS: SimpleNamespace(init_zbal=self.init_zbal),
                    _DSV4_BACKEND: SimpleNamespace(
                        prewarm_dsv4_aicpu=self.prewarm_aicpu
                    ),
                },
            ),
        )
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def _init_model_worker(self):
        self.scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=object()), attn_backend=object()
            )
        )
        self.events.append("model")

    def _startup(self):
        self.scheduler.init_npu(stage="allocator")
        self._init_model_worker()
        self.scheduler.init_npu(stage="pre_cache")
        self.events.append("cache")
        self.scheduler.init_npu(stage="post_cache")

    def test_memcache_prewarm_runs_after_model_and_before_cache(self):
        self._startup()
        self.assertEqual(self.events, ["zbal", "model", "hccl", "aicpu", "cache"])
        self.init_zbal.assert_called_once_with(8, 0, 0)
        self.prewarm_aicpu.assert_called_once_with(
            self.scheduler.tp_worker.model_runner.attn_backend, "npu:0"
        )
        self.scheduler.maybe_init_hccl_dp_prewarm.assert_called_once_with()

    def test_other_storage_keeps_hccl_after_cache(self):
        for storage_backend in (None, "file"):
            with self.subTest(storage_backend=storage_backend):
                self.events.clear()
                self.scheduler.server_args.hicache_storage_backend = storage_backend
                self._startup()
                self.assertEqual(self.events, ["zbal", "model", "cache", "hccl"])
        self.prewarm_aicpu.assert_not_called()

    def test_other_models_only_initialize_zbal(self):
        self.is_dsv4.return_value = False
        self._startup()
        self.assertEqual(self.events, ["zbal", "model", "cache"])

    def test_non_npu_needs_no_worker_or_npu_initialization(self):
        with patch(f"{_SCHEDULER}._is_npu", False):
            for stage in ("allocator", "pre_cache", "post_cache"):
                self.scheduler.init_npu(stage=stage)
        self.assertEqual(self.events, [])
        self.is_dsv4.assert_not_called()

    def test_hccl_failure_stops_before_aicpu_and_cache(self):
        self.scheduler.maybe_init_hccl_dp_prewarm.side_effect = RuntimeError("HCCL")
        with self.assertRaisesRegex(RuntimeError, "HCCL"):
            self._startup()
        self.assertEqual(self.events, ["zbal", "model"])
        self.prewarm_aicpu.assert_not_called()

    def test_aicpu_failure_stops_before_cache(self):
        self.prewarm_aicpu.side_effect = RuntimeError("AiCPU")
        with self.assertRaisesRegex(RuntimeError, "AiCPU"):
            self._startup()
        self.assertEqual(self.events, ["zbal", "model", "hccl"])


if __name__ == "__main__":
    unittest.main()
