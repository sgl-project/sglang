import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    _aiter_logits_workspaces,
    _get_aiter_logits_workspace,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAiterLogitsWorkspace(unittest.TestCase):
    def setUp(self):
        _aiter_logits_workspaces.clear()

    def tearDown(self):
        _aiter_logits_workspaces.clear()

    def test_reuses_geometric_capacity_per_stream(self):
        device = torch.device("cpu")
        stream = object()
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.current_stream", return_value=stream),
        ):
            first = _get_aiter_logits_workspace(2, 3, device)
            same_capacity = _get_aiter_logits_workspace(2, 4, device)
            grown = _get_aiter_logits_workspace(2, 5, device)

        self.assertEqual(first.shape, (2, 3))
        self.assertEqual(same_capacity.shape, (2, 4))
        self.assertEqual(
            first.untyped_storage().data_ptr(),
            same_capacity.untyped_storage().data_ptr(),
        )
        self.assertEqual(grown.shape, (2, 5))
        self.assertNotEqual(
            grown.untyped_storage().data_ptr(), first.untyped_storage().data_ptr()
        )
        self.assertEqual(_aiter_logits_workspaces[("cpu", stream)].numel(), 16)

        other_stream = object()
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch.cuda.current_stream", return_value=other_stream),
        ):
            other = _get_aiter_logits_workspace(2, 3, device)

        self.assertNotEqual(
            other.untyped_storage().data_ptr(), grown.untyped_storage().data_ptr()
        )
        self.assertEqual(
            set(_aiter_logits_workspaces),
            {("cpu", stream), ("cpu", other_stream)},
        )

    def test_capture_uses_graph_owned_allocation(self):
        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch("torch.cuda.current_stream") as current_stream,
        ):
            result = _get_aiter_logits_workspace(2, 3, torch.device("cpu"))

        self.assertEqual(result.shape, (2, 3))
        self.assertFalse(_aiter_logits_workspaces)
        current_stream.assert_not_called()

    def test_rejects_empty_dimensions(self):
        with self.assertRaises(AssertionError):
            _get_aiter_logits_workspace(0, 3, torch.device("cpu"))
        with self.assertRaises(AssertionError):
            _get_aiter_logits_workspace(2, 0, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
