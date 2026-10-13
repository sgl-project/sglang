import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.srt.model_executor.model_runner_components.cuda_graph_setup as graph_setup
from sglang.srt.model_executor.model_runner_components.cuda_graph_setup import (
    GraphCapture,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestElasticCudaGraphSetup(unittest.TestCase):
    def test_deferred_capture_skips_distributed_setup(self):
        model_runner = SimpleNamespace(is_draft_worker=False)
        eager_runner = Mock()
        prefill = GraphCapture(
            runner=eager_runner,
            memory_phase="prefill",
            memory_usage_gb=0,
            capture_time=0,
        )

        with (
            patch.object(graph_setup.GraphSharedOutput, "create_for_model_runner"),
            patch.object(
                graph_setup, "EagerRunner", return_value=eager_runner
            ) as eager_cls,
            patch.object(graph_setup, "capture_prefill_graph", return_value=prefill),
            patch.object(
                graph_setup, "refresh_deep_gemm_layout_memory_budget"
            ) as setup,
            patch.object(graph_setup, "finalize_cuda_graph_capture") as finalize,
        ):
            capture = graph_setup.capture_cuda_graphs(
                model_runner=model_runner,
                capture_decode_cuda_graph=False,
                finalize=False,
                defer_distributed_setup=True,
            )

        eager_cls.assert_called_once_with(model_runner, run_warmup=False)
        setup.assert_not_called()
        finalize.assert_not_called()
        self.assertIs(capture.decode.runner, eager_runner)

    def test_recapture_repeats_warmup_after_rendezvous(self):
        model_runner = SimpleNamespace(_kernel_warmed_up=True)
        eager_runner = Mock()
        decode_runner = Mock()
        replacement = Mock()
        capture = GraphCapture(
            runner=replacement,
            memory_phase="decode",
            memory_usage_gb=1,
            capture_time=2,
        )

        def capture_decode_graph(*, model_runner):
            self.assertFalse(model_runner._kernel_warmed_up)
            return capture

        with (
            patch.object(graph_setup.current_platform, "synchronize"),
            patch.object(graph_setup.current_platform, "empty_cache"),
            patch.object(graph_setup, "set_global_graph_memory_pool"),
            patch.object(graph_setup.gc, "collect"),
            patch.object(graph_setup, "refresh_deep_gemm_layout_memory_budget"),
            patch.object(
                graph_setup,
                "capture_decode_graph",
                side_effect=capture_decode_graph,
            ),
        ):
            graph_setup.drop_elastic_cuda_graph_state(
                decode_runner=decode_runner,
                eager_runner=eager_runner,
            )
            result = graph_setup.recapture_elastic_cuda_graph(
                model_runner=model_runner,
            )

        decode_runner.backend.cleanup.assert_called_once_with()
        self.assertIs(result, capture)

    def test_post_start_budget_refresh_skips_scale_joiner(self):
        model_runner = SimpleNamespace(device="cuda")
        exec_config = SimpleNamespace(moe=SimpleNamespace(ep_join_mode="scale"))

        with (
            patch.object(
                graph_setup.envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT,
                "get",
                return_value="auto",
            ),
            patch.object(graph_setup, "get_exec", return_value=exec_config),
            patch.object(graph_setup, "get_available_gpu_memory") as get_memory,
        ):
            graph_setup.refresh_deep_gemm_layout_memory_budget(
                model_runner, only_if_initialized=True
            )

        get_memory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
