import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.model_executor.model_runner import ModelRunner


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


if __name__ == "__main__":
    unittest.main()
