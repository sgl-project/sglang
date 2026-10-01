"""CPU ownership test for the context-parallel BCG KDA replay path."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.linear import kda_route_telemetry
from sglang.srt.layers.cp.bcg import execute_prefill_cp_bcg
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPrefillCPBCGKDATelemetryOwnership(unittest.TestCase):
    def test_direct_backend_replay_is_owned_by_route_transaction(self):
        backend = Mock()
        backend.replay.return_value = torch.tensor([[1.0], [2.0]])
        route_plans = object()
        runner = SimpleNamespace(
            prefill_cp_bcg_input=SimpleNamespace(live_local_tokens=1),
            model_runner=SimpleNamespace(
                model=SimpleNamespace(
                    capture_aux_hidden_states=False,
                    pp_group=SimpleNamespace(is_last_rank=False),
                )
            ),
            backend=backend,
            kda_cuda_graph_route_plans=route_plans,
            _prefill_forward_context=lambda *args, **kwargs: nullcontext(),
        )
        static_forward_batch = object()

        with patch.object(
            kda_route_telemetry,
            "replay_kda_route_plan",
            side_effect=lambda shape, mode, replay, **kwargs: replay(),
        ) as replay_transaction:
            output = execute_prefill_cp_bcg(
                runner,
                forward_batch=object(),
                static_forward_batch=static_forward_batch,
                static_num_tokens=4,
                raw_num_tokens=3,
            )

        self.assertEqual(output.tolist(), [[1.0]])
        replay_transaction.assert_called_once()
        args, kwargs = replay_transaction.call_args
        shape_key = args[0]
        self.assertEqual(shape_key.size, 4)
        self.assertEqual(args[1], "prefill")
        self.assertIs(kwargs["plans"], route_plans)
        backend.replay.assert_called_once_with(shape_key, static_forward_batch)


if __name__ == "__main__":
    unittest.main()
