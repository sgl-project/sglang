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
    def _host_dtype(self, num_logical_experts):
        metadata = (
            None
            if num_logical_experts is None
            else SimpleNamespace(
                num_logical_experts=num_logical_experts,
                num_physical_experts=num_logical_experts + 32,
            )
        )
        with mock.patch.object(
            re_mod, "get_global_expert_location_metadata", return_value=metadata
        ):
            return re_mod._host_dtype()

    def test_host_dtype_follows_logical_experts_not_redundant_slots(self):
        # 256 logical experts stay uint8 even with 32 redundant physical slots
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

    def test_overlap_output_never_aliases_the_device_cache(self):
        """The next forward overwrites the device cache, so the overlap path's rows
        must own their storage, whether cloned (base) or freshly built (routed)."""
        from sglang.srt.state_capturer.base import BaseTopkCapturer

        for cls, host_dtype in (
            (BaseTopkCapturer, torch.int32),
            (re_mod.RoutedExpertsCapturer, torch.uint8),
        ):
            cap = object.__new__(cls)
            cap.topk_size = 4
            device = torch.tensor([[[9, 3, 7, 1]], [[2, 8, 4, 6]]], dtype=torch.int32)
            cap.device_cache = SimpleNamespace(buffer=device)
            cap.host_cache = SimpleNamespace(
                buffer=torch.zeros(4, 1, 4, dtype=host_dtype)
            )
            batch = SimpleNamespace(
                out_cache_loc=torch.tensor([0, 1]), out_cache_loc_virtual=None
            )
            out = cap.on_forward_end(batch, False, None, no_copy_to_cpu=True)
            self.assertNotEqual(out.topk.data_ptr(), device.data_ptr(), cls.__name__)
            device.fill_(0)  # the next forward reuses the buffer
            self.assertTrue(bool(out.topk.any()), cls.__name__)


if __name__ == "__main__":
    unittest.main()
