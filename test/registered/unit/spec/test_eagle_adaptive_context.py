"""Adaptive target resources must not inherit draft TP or MoE contexts."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.runtime_context import get_context
from sglang.srt.speculative import eagle_worker_v2 as eagle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestEagleAdaptiveContext(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.active = []
        self.observed = []
        self.worker = object.__new__(eagle.EAGLEWorkerV2)
        worker = self.worker
        worker.speculative_num_steps = 5
        worker.speculative_num_draft_tokens = 6
        worker.device = "cpu"
        worker.gpu_id = 0
        worker._additional_graph_memory_usage = {}
        worker._additional_graph_time_usage = {}
        worker._override_worker_state = lambda *args, **kwargs: self.scope(
            "worker_state"
        )
        worker._draft_worker = SimpleNamespace(
            draft_owns_attention=True,
            init_attention_backend=Mock(
                side_effect=lambda: self.record("draft_backend")
            ),
            _capture_cuda_graphs=Mock(side_effect=lambda: self.record("draft_graph")),
            draft_attn_backend=object(),
            cuda_graph_runner=object(),
            draft_extend_attn_backend=object(),
            cuda_graph_runner_for_draft_extend=object(),
        )
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                init_new_workspace=False,
                attn_backend=object(),
                decode_cuda_graph_runner=object(),
                _get_attention_backend=Mock(
                    side_effect=lambda **kwargs: self.record("target_backend")
                ),
            )
        )
        worker.adaptive_controller = SimpleNamespace(
            register=Mock(),
            init_states=Mock(side_effect=self.build_states),
        )

    @contextlib.contextmanager
    def scope(self, name):
        self.active.append(name)
        try:
            yield
        finally:
            self.assertEqual(self.active.pop(), name)

    def record(self, phase):
        self.observed.append((phase, tuple(self.active)))
        return object()

    def build_states(self, **kwargs):
        self.record("controller")
        self.worker.build_adaptive_runtime_state(0, 1, cuda_graph_bs=[16, 32, 48, 64])

    @contextlib.contextmanager
    def mocked_runtime(self):
        with (
            get_context().override_server_args(
                cuda_graph_bs_decode=[16, 32, 48, 64], disable_cuda_graph=False
            ),
            patch.object(eagle.BaseSpecWorker, "init_cuda_graphs"),
            patch.object(
                eagle,
                "draft_tp_context",
                side_effect=self.draft_tp_scope,
            ),
            patch.object(
                eagle,
                "speculative_moe_backend_context",
                side_effect=lambda: self.scope("draft_moe"),
            ),
            patch.object(
                eagle,
                "speculative_moe_a2a_backend_context",
                side_effect=lambda: self.scope("draft_a2a"),
            ),
            patch.object(eagle, "get_available_gpu_memory", return_value=10.0),
            patch.object(eagle, "check_cuda_graph_backend", return_value=False),
            patch.object(eagle, "_is_npu", False),
            patch.object(
                eagle,
                "DecodeCudaGraphRunner",
                side_effect=lambda *args, **kwargs: self.record("target_graph"),
            ),
            patch.object(eagle, "log_info_on_rank0"),
        ):
            yield

    def draft_tp_scope(self, owns_attention):
        self.assertEqual(owns_attention, self.worker._draft_worker.draft_owns_attention)
        return self.scope("draft_tp") if owns_attention else contextlib.nullcontext()

    def test_only_draft_resources_use_draft_context(self):
        for owns_attention in (True, False):
            with self.subTest(owns_attention=owns_attention):
                self.observed.clear()
                self.worker._draft_worker.draft_owns_attention = owns_attention
                with self.mocked_runtime():
                    self.worker.init_cuda_graphs()
                draft_context = (
                    ("worker_state",)
                    + (("draft_tp",) if owns_attention else ())
                    + ("draft_moe", "draft_a2a")
                )
                self.assertEqual(
                    self.observed,
                    [
                        ("controller", ()),
                        ("draft_backend", draft_context),
                        ("draft_graph", draft_context),
                        ("target_backend", ("worker_state",)),
                        ("target_graph", ("worker_state",)),
                    ],
                )
                self.assertEqual(self.active, [])

    def test_failed_draft_capture_restores_context(self):
        self.worker._draft_worker._capture_cuda_graphs.side_effect = RuntimeError(
            "capture failed"
        )
        with (
            self.mocked_runtime(),
            self.assertRaisesRegex(RuntimeError, "capture failed"),
        ):
            self.worker.build_adaptive_runtime_state(0, 1)
        self.assertEqual(self.active, [])
        self.worker._target_worker.model_runner._get_attention_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
