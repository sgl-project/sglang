"""Adaptive capture must publish and restore the config used by graph runners."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    Phase,
    PhaseConfig,
    check_cuda_graph_backend,
)
from sglang.srt.runtime_context import get_context, get_exec, get_spec
from sglang.srt.speculative import eagle_worker_v2 as eagle
from sglang.srt.speculative.adaptive_runtime_state import SpecRuntimeState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAdaptiveGraphContract(unittest.TestCase):
    def setUp(self):
        self.cfg = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, bs=[1, 4, 8]),
            prefill=PhaseConfig(backend=Backend.BREAKABLE, bs=[16]),
        )
        self.scope = get_context().override_server_args(
            speculative_num_steps=1,
            speculative_num_draft_tokens=2,
            cuda_graph_bs_decode=[1, 4, 8],
            disable_cuda_graph=False,
            cuda_graph_config=self.cfg,
        )
        self.scope.install()
        self.addCleanup(self.scope.restore)
        self.worker = object.__new__(eagle.EAGLEWorkerV2)
        w = self.worker
        w.speculative_num_steps = 1
        w.speculative_num_draft_tokens = 2
        w.device = "cpu"
        w.gpu_id = 0
        w._additional_graph_memory_usage = {}
        w._additional_graph_time_usage = {}
        w._draft_worker = NS(
            speculative_num_steps=1,
            speculative_num_draft_tokens=2,
            draft_attn_backend=object(),
            draft_extend_attn_backend=object(),
            draft_runner=NS(draft_attn_backend=object(), attn_backend=object()),
            cuda_graph_runner=object(),
            cuda_graph_runner_for_draft_extend=object(),
            _rebuild_topk1_chain_buffers=Mock(),
            parallel_layout=NS(),
        )
        w._target_worker = NS(
            model_runner=NS(
                attn_backend=object(),
                decode_cuda_graph_runner=object(),
                init_new_workspace=False,
            )
        )

    def snapshot(self):
        w = self.worker
        dw = w._draft_worker
        return (
            w.speculative_num_steps,
            w.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.draft_runner.attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            get_spec().speculative_num_steps,
            get_spec().speculative_num_draft_tokens,
            get_exec().graph.cuda_graph_bs_decode,
            get_exec().graph.disable_cuda_graph,
            get_exec().graph.cuda_graph_config,
        )

    def test_pruned_bs_reaches_capture_consumer(self):
        before = self.snapshot()
        with self.worker._override_worker_state(3, 4, cuda_graph_bs=[1]):
            # Same leaf consumed by get_batch_sizes_to_capture, not the alias.
            self.assertEqual(
                get_exec().graph.cuda_graph_config.decode.bs,
                [1],
                "S1_PRUNED_BS_NOT_PROPAGATED",
            )
            self.assertEqual(self.cfg.decode.bs, [1, 4, 8])
            self.assertEqual(
                get_exec().graph.cuda_graph_config.prefill, self.cfg.prefill
            )
        self.assertEqual(self.snapshot(), before)

    def test_empty_tier_disables_actual_backend_gate_then_next_tier_restores(self):
        before = self.snapshot()
        with self.worker._override_worker_state(3, 4, cuda_graph_bs=[]):
            # This exact predicate gates draft and target runner construction.
            self.assertTrue(
                check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED),
                "S1_EMPTY_TIER_NOT_DISABLED",
            )
            self.assertEqual(
                get_exec().graph.cuda_graph_config.prefill, self.cfg.prefill
            )
        self.assertEqual(self.snapshot(), before)
        with self.worker._override_worker_state(5, 6, cuda_graph_bs=[4]):
            self.assertFalse(check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED))
            self.assertEqual(get_exec().graph.cuda_graph_config.decode.bs, [4])
        self.assertEqual(self.snapshot(), before)

    def test_capture_body_exception_restores_worker_and_config(self):
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            with self.worker._override_worker_state(3, 4, cuda_graph_bs=[]):
                dw = self.worker._draft_worker
                dw.draft_attn_backend = object()
                dw.draft_extend_attn_backend = object()
                dw.draft_runner.attn_backend = object()
                dw.draft_runner.draft_attn_backend = object()
                dw.cuda_graph_runner = None
                dw.cuda_graph_runner_for_draft_extend = None
                raise RuntimeError("capture failed")
        self.assertEqual(self.snapshot(), before)

    def test_low_high_low_reuses_all_runner_and_backend_identities(self):
        def state(steps):
            return SpecRuntimeState(steps, steps + 1, *[object() for _ in range(6)])

        low, high = state(1), state(5)
        # Force the initial application; same-step return is intentional.
        self.worker.speculative_num_steps = -1
        for s in (low, high, low):
            self.worker.apply_runtime_state(s)
            dw = self.worker._draft_worker
            mr = self.worker._target_worker.model_runner
            for actual, expected in (
                (dw.draft_attn_backend, s.draft_attn_backend),
                (dw.draft_runner.draft_attn_backend, s.draft_attn_backend),
                (dw.cuda_graph_runner, s.cuda_graph_runner),
                (dw.draft_extend_attn_backend, s.draft_extend_attn_backend),
                (dw.draft_runner.attn_backend, s.draft_extend_attn_backend),
                (
                    dw.cuda_graph_runner_for_draft_extend,
                    s.cuda_graph_runner_for_draft_extend,
                ),
                (mr.attn_backend, s.target_attn_backend),
                (mr.decode_cuda_graph_runner, s.target_graph_runner),
            ):
                self.assertIs(actual, expected)
            self.assertEqual(get_spec().speculative_num_steps, s.speculative_num_steps)
            self.assertEqual(
                get_spec().speculative_num_draft_tokens, s.speculative_num_draft_tokens
            )


if __name__ == "__main__":
    unittest.main()
