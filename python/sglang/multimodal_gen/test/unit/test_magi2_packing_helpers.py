# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    MAGI2_MODALITY_AUDIO,
    MAGI2_MODALITY_TEXT,
    MAGI2_MODALITY_VIDEO,
    modality_runs,
    sharded_cu_seqlens,
)


class _Plan:
    """Stands in for SpShard, which needs a live process group to build."""

    def __init__(self, orig_len: int, local_len: int, num_pad: int, sp_size: int):
        self.orig_len = orig_len
        self.local_len = local_len
        self.num_pad = num_pad
        self.sp_size = sp_size


class TestModalityRuns(unittest.TestCase):
    def test_contiguous_segments_collapse_to_one_run_each(self):
        ids = torch.tensor(
            [MAGI2_MODALITY_VIDEO] * 5
            + [MAGI2_MODALITY_AUDIO] * 3
            + [MAGI2_MODALITY_TEXT] * 2
        )
        self.assertEqual(
            modality_runs(ids),
            [
                (0, 5, MAGI2_MODALITY_VIDEO),
                (5, 3, MAGI2_MODALITY_AUDIO),
                (8, 2, MAGI2_MODALITY_TEXT),
            ],
        )


class TestShardedCuSeqlens(unittest.TestCase):
    def test_padding_becomes_its_own_segment(self):
        # 62 real rows over 4 ranks pads to 64, and the 2 pad rows must not share
        # a segment with real tokens or they would attract attention mass.
        cu, max_seqlen = sharded_cu_seqlens(
            plan=_Plan(orig_len=62, local_len=16, num_pad=2, sp_size=4),
            device=torch.device("cpu"),
        )
        self.assertEqual(cu.tolist(), [0, 62, 64])
        self.assertEqual(max_seqlen, 62)


if __name__ == "__main__":
    unittest.main()
