"""Unit tests for DSV4 (NPU) layer-split active-page staging math.

The staging read path serves a remote layer from only the pages the batch
references, so the mask/union/remap helpers must agree on page ids and keep
padding (-1) entries untouched through the rebase.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.dsv4 import dsv4_layer_split_staging as staging
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeRemoteRank:
    """All-reduce stand-in that marks one page active on a CP partner."""

    def __init__(self, remote_page: int):
        self.remote_page = remote_page

    def all_reduce(self, mask):
        mask[self.remote_page] += 1
        return mask


class TestDSV4LayerSplitStaging(CustomTestCase):
    def test_active_pages_include_remote_cp_ranks(self):
        indices = torch.tensor([0, 7, -1], dtype=torch.int32)

        with patch.object(
            staging,
            "all_reduce_active_pages_mask",
            side_effect=lambda mask, group: _FakeRemoteRank(3).all_reduce(mask),
        ):
            selected = staging.active_pages_for_indices(
                indices, page_size=4, max_pages=4, group=None
            )

        # Local pages {0, 1} plus the partner's page 3, in sorted order.
        self.assertEqual(selected.tolist(), [0, 1, 3])

    def test_indices_remap_preserves_padding(self):
        selected = torch.tensor([1, 3], dtype=torch.int64)
        remapped = staging.remap_indices_to_staging(
            torch.tensor([4, 5, 12, 15, -1], dtype=torch.int32),
            selected,
            page_size=4,
            max_pages=4,
        )
        self.assertEqual(remapped.tolist(), [0, 1, 4, 7, -1])

    def test_page_table_remap_preserves_padding(self):
        remapped = staging.remap_page_table_to_staging(
            torch.tensor([3, 1, -1], dtype=torch.int32),
            torch.tensor([1, 3], dtype=torch.int64),
            max_pages=4,
        )
        self.assertEqual(remapped.tolist(), [1, 0, -1])

    def test_mask_counts_hits_per_page(self):
        mask = staging.build_active_pages_mask(
            torch.tensor([0, 1, 2, 3, -1, 7], dtype=torch.int32),
            page_size=4,
            max_pages=3,
        )
        self.assertEqual(mask.tolist(), [4, 1, 0])


if __name__ == "__main__":
    unittest.main()
