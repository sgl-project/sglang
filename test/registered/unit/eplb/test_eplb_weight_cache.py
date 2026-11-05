import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE = "sglang.srt.eplb.eplb_manager"


class _Session:
    """Stands in for the engine's ExpertLayoutSession; records the moves."""

    def __init__(self, calls, mode, reason=None, begin_reasons=()):
        self._calls = calls
        self._mode = (mode, reason)
        self._begin_reasons = list(begin_reasons)

    def rebalance_mode(self):
        return self._mode

    def begin(self, layer_ids):
        self._calls.append(("begin", list(layer_ids)))
        return self._begin_reasons.pop(0) if self._begin_reasons else None

    def commit(self, layer_ids, physical_to_logical_map):
        self._calls.append(
            ("commit", list(layer_ids), physical_to_logical_map.tolist())
        )
        return None


class TestEPLBWithWeightCache(unittest.TestCase):
    def setUp(self):
        self.calls = []

    def _manager(
        self, weight_cache_mode, session=None, *, busy_gpu=False, elastic_ep=None
    ):
        moe = SimpleNamespace(
            eplb_rebalance_layers_per_chunk=1,
            eplb_rebalance_num_iterations=2,
            expert_distribution_recorder_buffer_size=2,
            eplb_min_rebalancing_utilization_threshold=0.5,
            ep_dispatch_algorithm="dynamic",
            elastic_ep_backend=elastic_ep,
        )
        self.recorder = Mock(recording=True)
        # A busy GPU makes a rebalance stop right after reading the counts.
        self.recorder.dump_record.return_value = {
            "logical_count": None,
            "average_utilization_rate_over_window": 1.0 if busy_gpu else 0.0,
        }
        self.metadata = SimpleNamespace(
            physical_to_logical_map_cpu=torch.zeros(2, 4, dtype=torch.int64)
        )

        def move(*, update_layer_ids, **_):
            # The engine's own map follows the moved layers.
            for layer_id in update_layer_ids:
                self.metadata.physical_to_logical_map_cpu[layer_id] = layer_id + 1
            self.calls.append(("move", list(update_layer_ids)))

        for name, replacement in (
            ("get_exec", lambda: SimpleNamespace(moe=moe)),
            ("get_model", lambda: SimpleNamespace(weight_cache_mode=weight_cache_mode)),
            ("get_parallel", lambda: SimpleNamespace(tp_rank=0)),
            ("get_global_expert_distribution_recorder", lambda: self.recorder),
            ("get_global_expert_location_metadata", lambda: self.metadata),
            ("get_expert_layout_session", lambda: session),
            ("update_expert_location_with_recovery", move),
        ):
            patcher = patch(f"{_MODULE}.{name}", replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        # A commit waits for the moves to land. CPU CI has a CUDA build of
        # torch and no driver, so a real device sync would raise there.
        device = Mock()
        device.synchronize.side_effect = lambda: self.calls.append(("sync",))
        patcher = patch.object(torch, "get_device_module", return_value=device)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(
            EPLBManager, "_compute_expert_location_metadata", return_value=None
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return EPLBManager(
            model_config=None,
            get_model=lambda: SimpleNamespace(
                routed_experts_weights_of_layer={0: [], 1: []}
            ),
            get_expert_location_updater=Mock(),
            get_expert_backup_client=Mock(),
            get_weight_updater=Mock(),
        )

    def test_rebalancing_stays_off_without_a_layout_record_to_follow(self):
        for session, elastic_ep, reason in (
            (None, None, "no expert layout session"),
            (
                _Session(self.calls, "off", "the daemon does not track it"),
                None,
                "the daemon does not track it",
            ),
            (_Session(self.calls, "coordinated"), "nixl", "elastic EP"),
        ):
            with self.subTest(reason=reason):
                manager = self._manager("client", session, elastic_ep=elastic_ep)
                with self.assertLogs(_MODULE, "WARNING") as logs:
                    self.assertEqual(list(manager.rebalance()), [])
                self.assertIn(
                    "rebalancing is off with the weight cache", logs.output[0]
                )
                self.assertIn(reason, logs.output[0])
                # Once per reason; elastic EP re-enabling rebalancing changes nothing.
                manager.enable_rebalance()
                with self.assertNoLogs(_MODULE, "WARNING"):
                    for _ in range(10):
                        manager.on_forward_pass_end()
                self.recorder.dump_record.assert_not_called()
                self.assertEqual(self.calls, [])

    def test_weights_loaded_from_disk_rebalance_as_usual(self):
        manager = self._manager("client", _Session(self.calls, "private"))
        self.assertEqual(list(manager.rebalance()), [None, None])
        self.assertEqual(self.calls, [("move", [0]), ("move", [1])])

    def test_shared_weights_move_through_the_daemon(self):
        manager = self._manager("daemon", _Session(self.calls, "coordinated"))
        list(manager.rebalance())
        self.assertEqual(
            self.calls,
            [
                ("begin", [0]),
                ("move", [0]),
                ("sync",),
                ("commit", [0], [[1] * 4, [0] * 4]),
                ("begin", [1]),
                ("move", [1]),
                ("sync",),
                ("commit", [1], [[1] * 4, [2] * 4]),
            ],
        )

    def test_a_busy_daemon_stops_the_rebalance_before_the_next_move(self):
        session = _Session(
            self.calls,
            "coordinated",
            begin_reasons=[None, "rank 1: 1 other engine(s) map these weights"],
        )
        manager = self._manager("client", session)
        list(manager.rebalance())
        self.assertEqual(
            self.calls,
            [
                ("begin", [0]),
                ("move", [0]),
                ("sync",),
                ("commit", [0], [[1] * 4, [0] * 4]),
                ("begin", [1]),
            ],
        )

    def test_rebalance_runs_without_the_weight_cache(self):
        manager = self._manager("off", busy_gpu=True)
        with self.assertNoLogs(_MODULE, "WARNING"):
            self.assertEqual(list(manager.rebalance()), [])
        self.recorder.dump_record.assert_called_once()


if __name__ == "__main__":
    unittest.main()
