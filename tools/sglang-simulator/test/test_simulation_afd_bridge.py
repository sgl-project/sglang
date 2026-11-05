"""Real Scheduler hook on a test double: prefill, decode and request clocks."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from sglang_simulator.simulation.sglang import scheduler as hooks


class SchedulerAFDBridgeTests(unittest.TestCase):
    def test_prefill_decode_metadata_and_virtual_request_time(self):
        class GenerationBatchResult:
            pass

        class Scheduler:
            def __init__(self):
                pass

            def recv_requests(self):
                return []

            def _prefetch_kvcache(self):
                pass

            get_new_batch_prefill = event_loop_normal = init_request_dispatcher = (
                _prefetch_kvcache
            )

            def run_batch(self, batch):
                return GenerationBatchResult()

            def process_batch_result(self, batch, result):
                for req in batch.reqs:
                    req.output_ids.append(1)

        reqs = [
            NS(
                rid=r,
                origin_input_ids=[1, 2, 3, 4],
                prefix_indices=[],
                output_ids=[],
                extend_range=NS(length=4),
            )
            for r in ("r0", "r1")
        ]
        phase = ["prefill"]
        mode = NS(
            is_extend=lambda: phase[0] == "prefill",
            is_decode=lambda: phase[0] == "decode",
        )
        batch, seen = NS(reqs=reqs, forward_mode=mode), []

        def predict(b):
            seen.append(b)
            return 0.02

        predictor = NS(
            uses_virtual_event_time=True,
            predict_infer_time=predict,
            completion_offsets=lambda: {"r0": 0.01, "r1": 0.02},
        )
        hooks.C_SchedulerHook.hook(Scheduler)
        target = object.__new__(Scheduler)
        state, stats = hooks.StateManager, hooks.request_stats_manager
        state.reset()
        stats.reset()
        self.addCleanup(state.reset)
        self.addCleanup(stats.reset)
        with (
            patch.object(hooks.C_SchedulerHook, "INFERENCE_PREDICTOR", predictor),
            patch.object(
                hooks.C_SchedulerHook, "SIM_MODE", hooks.SimulationMode.OFFLINE
            ),
            patch.object(hooks.C_SchedulerHook, "ITERATION_STATS", []),
            patch.object(hooks, "get_obj_from_args", return_value=batch),
            patch.object(hooks.time, "time", return_value=100),
        ):
            for current in ("prefill", "decode"):
                phase[0] = current
                target.process_batch_result(batch, target.run_batch(batch))
            self.assertEqual([b.forward_phase for b in seen], ["prefill", "decode"])
            self.assertEqual(
                [(r.extend_length, r.past_kv_length) for r in seen[0].reqs],
                [(4, 0)] * 2,
            )
            self.assertEqual([r.past_kv_length for r in seen[1].reqs], [4, 4])
            self.assertAlmostEqual(state.get_global_clock(), 0.04)
            for rid, expected in (("r0", [0.01, 0.02]), ("r1", [0.02, 0.02])):
                actual = stats.get_req_stats(rid).gen_token_latencies
                self.assertEqual(len(actual), 2)
                for a, e in zip(actual, expected):
                    self.assertAlmostEqual(a, e)
            for row in hooks.C_SchedulerHook.ITERATION_STATS:
                self.assertEqual(row["cpu_overhead"], 0)
                self.assertTrue(row["afd_host_wall_time_excluded"])
