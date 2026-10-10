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
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.elastic_ep import elastic_ep
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPState,
    ElasticEPStateManager,
    _iter_live_parallel_groups,
    _map_global_to_group_local_ranks,
    elastic_expanded_world_enabled,
    get_healthy_expert_location_src_rank,
    get_scale_cohort,
    register_scale_cohort,
)


class FakeStore:
    """Dict-backed stand-in for torch.distributed.TCPStore."""

    def __init__(self):
        self.data = {}

    def set(self, key, value):
        self.data[key] = value

    def check(self, keys):
        return [key in self.data for key in keys]

    def get(self, key):
        return self.data[key]


class TestScaleCohortStore(unittest.TestCase):
    def test_register_and_get_roundtrip(self):
        store = FakeStore()
        with mock.patch.object(elastic_ep, "get_global_tcp_store", return_value=store):
            register_scale_cohort(3, 16, True)
            self.assertIn("elastic_ep/scale_cohort/3", store.data)
            cohort = get_scale_cohort(3)
            self.assertIsNone(get_scale_cohort(99))
        self.assertEqual(cohort.target_ep_size, 16)
        self.assertTrue(cohort.cuda_graph_enabled)

    def test_absent_store(self):
        with mock.patch.object(elastic_ep, "get_global_tcp_store", return_value=None):
            self.assertIsNone(get_scale_cohort(0))
            with self.assertRaises(RuntimeError):
                register_scale_cohort(0, 8, False)


class TestMapGlobalToGroupLocalRanks(unittest.TestCase):
    def test_maps_and_filters(self):
        self.assertEqual(
            _map_global_to_group_local_ranks([10, 11, 12], [12, 10, 99, 11]),
            [2, 0, 1],
        )

    def test_empty_inputs(self):
        self.assertEqual(_map_global_to_group_local_ranks([], [1, 2]), [])
        self.assertEqual(_map_global_to_group_local_ranks([1, 2], []), [])

    def test_no_members(self):
        self.assertEqual(_map_global_to_group_local_ranks([1, 2], [3, 4]), [])


class TestIterLiveParallelGroups(unittest.TestCase):
    def test_sorted_and_dead_refs_skipped(self):
        class FakeGroup:
            def __init__(self, name):
                self.unique_name = name

        groups = [FakeGroup("b_group"), FakeGroup("a_group")]
        dead = weakref.ref(object())  # collected immediately on CPython
        self.assertIsNone(dead())
        fake_groups = {
            0: weakref.ref(groups[0]),
            1: dead,
            2: weakref.ref(groups[1]),
        }
        with mock.patch.object(elastic_ep.parallel_state, "_groups", fake_groups):
            live = list(_iter_live_parallel_groups())
        # Sorted by unique_name; the dead reference is dropped.
        self.assertEqual(live, [groups[1], groups[0]])


class TestElasticExpandedWorldEnabled(unittest.TestCase):
    def setUp(self):
        active = torch.ones(8, dtype=torch.int32)
        ElasticEPStateManager._instance = ElasticEPState(
            active_ranks=active,
            last_active_ranks=active.clone(),
            active_ranks_cpu=active.clone(),
            effective_ep_size=8,
            original_ep_size=8,
        )

    def tearDown(self):
        ElasticEPStateManager._instance = None

    def _enabled(self, max_ep_size):
        parallel = SimpleNamespace(max_ep_size=max_ep_size)
        with mock.patch.object(elastic_ep, "get_parallel", return_value=parallel):
            return elastic_expanded_world_enabled()

    def test_disabled_without_instance(self):
        ElasticEPStateManager._instance = None
        self.assertFalse(self._enabled(16))

    def test_disabled_without_max_ep_size(self):
        self.assertFalse(self._enabled(None))

    def test_enabled_only_when_data_plane_grew(self):
        # Data-plane size equals the original world: disabled.
        self.assertFalse(self._enabled(16))
        # Data-plane size grows past the original world: enabled.
        ElasticEPStateManager._instance.pending_ep_size = 12
        ElasticEPStateManager._instance.scale_phase = "configuring_data_plane"
        self.assertTrue(self._enabled(16))


class TestGetHealthyExpertLocationSrcRank(unittest.TestCase):
    class FakeWorldGroup:
        def __init__(self, gathered, ranks):
            self._gathered = gathered
            self.ranks = ranks

        def all_gather_object(self, flag):
            return self._gathered

    def _src_rank(self, gathered, ranks):
        world_group = self.FakeWorldGroup(gathered, ranks)
        parallel = SimpleNamespace(world_group=world_group)
        with mock.patch.object(elastic_ep, "get_parallel", return_value=parallel):
            return get_healthy_expert_location_src_rank(
                invoked_in_elastic_ep_rejoin_path=False
            )

    def test_first_healthy_rank_selected(self):
        # The first rank that did not rejoin broadcasts the metadata.
        self.assertEqual(
            self._src_rank([False, True, False, True], [10, 11, 12, 13]), 10
        )
        self.assertEqual(
            self._src_rank([True, False, True, True], [10, 11, 12, 13]), 11
        )

    def test_all_rejoin_raises(self):
        with self.assertRaises(RuntimeError):
            self._src_rank([True, True], [5, 6])


if __name__ == "__main__":
    unittest.main()
