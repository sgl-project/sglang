import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache.kv_cache_builder import resolve_decode_retraction_backup
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler


class TestDecodeGraphRoleSwitchState(unittest.TestCase):
    def _startup(self, decode_kind):
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = False
        runner.model = object()
        eager = object()
        decode = {
            "graph": SimpleNamespace(capture_bs=[1, 2]),
            "eager": eager,
            "disabled": None,
        }[decode_kind]
        capture = SimpleNamespace(
            eager_runner=eager,
            prefill=SimpleNamespace(runner=eager),
            decode=SimpleNamespace(runner=decode),
            memory_usage={"decode": 0.4},
            time_usage={"decode": 1.0},
        )
        module = "sglang.srt.model_executor.model_runner"
        with (
            patch(
                module + ".get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(is_ep_scale_joiner=False)
                ),
            ),
            patch(module + ".capture_cuda_graphs", return_value=capture),
            patch(module + ".prebuild_deepep_v2_buffer"),
        ):
            runner.init_cuda_graphs()
        return runner

    def test_startup_graph_is_reported_and_not_recaptured_on_flip(self):
        runner = self._startup("graph")
        self.assertTrue(runner.decode_cuda_graph_captured)
        self.assertEqual(runner.decode_cuda_graph_capture_bs, [1, 2])
        with patch.object(runner, "init_decode_cuda_graph") as capture:
            runner.ensure_decode_cuda_graphs([1, 2])
        capture.assert_not_called()

    def test_eager_and_disabled_startup_do_not_claim_a_decode_graph(self):
        for kind in ("eager", "disabled"):
            with self.subTest(kind=kind):
                runner = self._startup(kind)
                self.assertFalse(runner.decode_cuda_graph_captured)
                self.assertEqual(runner.decode_cuda_graph_capture_bs, [])


class TestAscendDecodeRetractionDefault(unittest.TestCase):
    def test_default_uses_cpu_tensor_for_ascend_blocked_kv_layout(self):
        pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
        worker = SimpleNamespace(
            is_hybrid_swa=False,
            get_memory_pool=lambda: (None, SimpleNamespace(get_kvcache=lambda: pool)),
            model_runner=SimpleNamespace(
                mtp_draft_device_pools=(), model_config=object()
            ),
        )
        module = "sglang.srt.mem_cache.kv_cache_builder"
        for npu, expected in ((False, "host_pool"), (True, "cpu_tensor")):
            with self.subTest(npu=npu):
                args = ServerArgs(
                    model_path="dummy",
                    device="cpu",
                    disaggregation_mode="decode",
                    hicache_ratio=1.0,
                )
                set_global_server_args_for_scheduler(args)
                with (
                    patch(module + ".is_npu", return_value=npu),
                    patch(module + ".is_hip", return_value=False),
                    patch(module + ".uses_ssm_state", return_value=False),
                ):
                    self.assertEqual(
                        resolve_decode_retraction_backup(tp_worker=worker), expected
                    )


if __name__ == "__main__":
    unittest.main()
