"""Captured routing reaches the host cache sorted and in the narrowest dtype."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRoutedExpertsHostCache(CustomTestCase):
    def _host_dtype(self, num_physical_experts):
        metadata = (
            None
            if num_physical_experts is None
            else SimpleNamespace(num_physical_experts=num_physical_experts)
        )
        with mock.patch.object(
            re_mod, "get_global_expert_location_metadata", return_value=metadata
        ):
            return re_mod._host_dtype()

    def test_host_dtype_holds_every_physical_expert_id(self):
        self.assertEqual(self._host_dtype(128), torch.uint8)
        self.assertEqual(self._host_dtype(256), torch.uint8)
        self.assertEqual(self._host_dtype(288), torch.int16)
        self.assertEqual(self._host_dtype(1 << 16), torch.int32)
        self.assertEqual(self._host_dtype(None), torch.int32)

    def test_rows_are_sorted_and_narrowed_exactly(self):
        cap = object.__new__(re_mod.RoutedExpertsCapturer)
        cap.host_cache = SimpleNamespace(buffer=torch.zeros(1, dtype=torch.uint8))
        rows = torch.tensor([[[255, 3, 127, 0], [9, 8, 7, 6]]], dtype=torch.int32)
        out = cap._canonicalize_rows(rows)
        self.assertEqual(out.dtype, torch.uint8)
        torch.testing.assert_close(out.to(torch.int32), rows.sort(dim=-1).values)


if __name__ == "__main__":
    unittest.main()
