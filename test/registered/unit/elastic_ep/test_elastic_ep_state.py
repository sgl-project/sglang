# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import unittest

import msgspec
import torch

from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPState,
    ElasticEPStateManager,
    ScaleCohort,
)


def _make_state(effective_ep_size=8, capacity=8):
    active = torch.ones(capacity, dtype=torch.int32)
    return ElasticEPState(
        active_ranks=active,
        last_active_ranks=active.clone(),
        active_ranks_cpu=active.clone(),
        effective_ep_size=effective_ep_size,
    )


class TestScaleCohort(unittest.TestCase):
    def test_json_roundtrip(self):
        cohort = ScaleCohort(target_ep_size=16, cuda_graph_enabled=True)
        decoded = msgspec.json.decode(msgspec.json.encode(cohort), type=ScaleCohort)
        self.assertEqual(decoded, cohort)


class TestElasticEPState(unittest.TestCase):
    def test_is_active_equal_last(self):
        state = _make_state()
        self.assertTrue(state.is_active_equal_last())
        state.active_ranks[0] = 0
        self.assertFalse(state.is_active_equal_last())

    def test_snapshot_clones(self):
        state = _make_state()
        state.snapshot_active_to_last()
        state.active_ranks[1] = 0
        self.assertTrue((state.last_active_ranks == 1).all())

    def test_sync_active_to_cpu_clones(self):
        state = _make_state()
        state.sync_active_to_cpu()
        state.active_ranks[2] = 0
        self.assertTrue((state.active_ranks_cpu == 1).all())

    def test_reset_fills_effective_prefix(self):
        state = _make_state(effective_ep_size=4, capacity=8)
        state.active_ranks.zero_()
        state.reset()
        expected = [1, 1, 1, 1, 0, 0, 0, 0]
        # Reserved slots stay inactive until their ranks join; reset also
        # snapshots to last and syncs the CPU copy.
        self.assertEqual(state.active_ranks.tolist(), expected)
        self.assertEqual(state.last_active_ranks.tolist(), expected)
        self.assertEqual(state.active_ranks_cpu.tolist(), expected)

    def test_reset_with_none_active_is_noop(self):
        state = ElasticEPState(
            active_ranks=None, last_active_ranks=None, active_ranks_cpu=None
        )
        state.reset()  # must not raise


class TestElasticEPStateManager(unittest.TestCase):
    """Tests for the scale state machine.

    ``init()`` requires a real torch.distributed world, so these tests inject
    a hand-built ``ElasticEPState`` into the manager's class-level instance
    slot and exercise the transitions directly.
    """

    def setUp(self):
        self.state = _make_state(effective_ep_size=8)
        ElasticEPStateManager._instance = self.state
        ElasticEPStateManager._on_scale = None

    def tearDown(self):
        ElasticEPStateManager._instance = None
        ElasticEPStateManager._on_scale = None

    def test_request_scale_happy_path(self):
        self.assertTrue(ElasticEPStateManager.request_scale(4))
        self.assertEqual(self.state.pending_ep_size, 4)
        self.assertEqual(self.state.scale_phase, "waiting_for_cohort")
        self.assertIsNone(self.state.last_error)
        self.assertIsNotNone(self.state.pending_since)

    def test_request_scale_rejected_while_pending(self):
        ElasticEPStateManager.request_scale(4)
        self.assertFalse(ElasticEPStateManager.request_scale(6))
        self.assertEqual(self.state.pending_ep_size, 4)

    def test_recovery_unsupported_blocks_scaling(self):
        ElasticEPStateManager.fail_recovery("no support")
        self.assertEqual(self.state.scale_phase, "recovery_unsupported")
        self.assertEqual(self.state.last_error, "no support")
        self.assertFalse(ElasticEPStateManager.request_scale(4))

    def test_disabled_accessors_without_instance(self):
        ElasticEPStateManager._instance = None
        self.assertFalse(ElasticEPStateManager.request_scale(4))
        self.assertFalse(ElasticEPStateManager.begin_scale())
        self.assertEqual(ElasticEPStateManager.get_scale_phase(), "disabled")
        self.assertIsNone(ElasticEPStateManager.get_last_error())
        self.assertIsNone(ElasticEPStateManager.get_pending_ep_size())
        self.assertEqual(ElasticEPStateManager.get_ep_join_rank_offset(), 0)

    def test_begin_scale_phase_gate(self):
        # Idle (no pending request): not allowed.
        self.assertFalse(ElasticEPStateManager.begin_scale())
        ElasticEPStateManager.request_scale(4)
        self.assertTrue(ElasticEPStateManager.begin_scale())
        self.assertEqual(self.state.scale_phase, "pending")
        # Already past waiting_for_cohort: not allowed again.
        self.assertFalse(ElasticEPStateManager.begin_scale())

    def test_mark_phases_require_pending(self):
        # Without a pending scale request, phase markers are ignored.
        ElasticEPStateManager.mark_joining()
        self.assertEqual(self.state.scale_phase, "idle")
        ElasticEPStateManager.request_scale(4)
        for marker, phase in (
            (ElasticEPStateManager.mark_joining, "joining"),
            (
                ElasticEPStateManager.mark_configuring_data_plane,
                "configuring_data_plane",
            ),
            (ElasticEPStateManager.mark_syncing_new_world, "syncing_new_world"),
        ):
            marker()
            self.assertEqual(self.state.scale_phase, phase)

    def test_commit_scale(self):
        ElasticEPStateManager.request_scale(4)
        ElasticEPStateManager.begin_scale()
        ElasticEPStateManager.commit_scale()
        self.assertEqual(self.state.effective_ep_size, 4)
        self.assertIsNone(self.state.pending_ep_size)
        self.assertTrue(self.state.has_scaled)
        self.assertEqual(self.state.scale_phase, "serving_expanded")
        # commit_scale resets rank state for the new effective size.
        self.assertEqual(self.state.active_ranks.tolist(), [1, 1, 1, 1, 0, 0, 0, 0])

    def test_commit_scale_without_pending_is_noop(self):
        ElasticEPStateManager.commit_scale()
        self.assertEqual(self.state.effective_ep_size, 8)
        self.assertEqual(self.state.scale_phase, "idle")
        self.assertFalse(self.state.has_scaled)

    def test_fail_scale_records_error(self):
        ElasticEPStateManager.request_scale(4)
        ElasticEPStateManager.fail_scale("boom")
        self.assertIsNone(self.state.pending_ep_size)
        self.assertEqual(self.state.scale_phase, "failed")
        self.assertEqual(self.state.last_error, "boom")
        # fail_scale resets rank state for the current effective size.
        self.assertEqual(self.state.active_ranks.tolist(), [1] * 8)

    def test_get_data_plane_ep_size_phase_dependent(self):
        self.state.pending_ep_size = 12
        self.state.scale_phase = "joining"
        self.assertEqual(ElasticEPStateManager.get_data_plane_ep_size(), 8)
        self.state.scale_phase = "configuring_data_plane"
        self.assertEqual(ElasticEPStateManager.get_data_plane_ep_size(), 12)
        self.state.scale_phase = "syncing_new_world"
        self.assertEqual(ElasticEPStateManager.get_data_plane_ep_size(), 12)
        self.state.scale_phase = "serving_expanded"
        self.assertEqual(ElasticEPStateManager.get_data_plane_ep_size(), 8)
        self.state.pending_ep_size = None
        self.assertEqual(ElasticEPStateManager.get_data_plane_ep_size(), 8)

    def test_is_scaling_matrix(self):
        # A pending scale request reports scaling in progress.
        ElasticEPStateManager.request_scale(4)
        self.assertTrue(ElasticEPStateManager.is_scaling())
        # Committed and healthy: not scaling.
        ElasticEPStateManager.commit_scale()
        self.assertFalse(ElasticEPStateManager.is_scaling())
        # A dead rank inside the effective window: scaling (recovery pending).
        self.state.active_ranks_cpu[2] = 0
        self.assertTrue(ElasticEPStateManager.is_scaling())
        # Healthy again, but recovery unsupported: never scaling.
        self.state.active_ranks_cpu[2] = 1
        ElasticEPStateManager.fail_recovery("unsupported")
        self.assertFalse(ElasticEPStateManager.is_scaling())

    def test_is_scaling_without_instance(self):
        ElasticEPStateManager._instance = None
        self.assertFalse(ElasticEPStateManager.is_scaling())


if __name__ == "__main__":
    unittest.main()
