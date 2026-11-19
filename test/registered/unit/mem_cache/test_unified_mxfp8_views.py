"""MXFP8 scale rows share the target's movable entries and BF16 draft bands."""

import unittest

import torch

from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec, MHAView
from sglang.srt.mem_cache.layout.fused_draft import (
    FusedDraftPlacement,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import arena_of

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _spec(heads=1, draft=None):
    return MHASubPoolSpec(
        name="full",
        layer_num=2,
        head_num=heads,
        head_dim=128,
        store_dtype=torch.uint8,
        kv_cache_dtype=torch.float8_e4m3fn,
        is_mxfp8=True,
        grow_direction="down",
        draft_region=draft,
    )


def _view(spec):
    placement = (
        None
        if spec.draft_region is None
        else FusedDraftPlacement.from_counts(
            counts={"full": (1,)}, regions={"full": spec.draft_region}
        )
    )
    arena = arena_of(
        [spec],
        total_bytes=16 * spec.entry_bytes(),
        device="cpu",
        page_size=2,
        fused_draft=placement,
    )
    return MHAView(
        arena=arena,
        spec=spec,
        page_size=2,
        model_layer_ids=(0, 1),
        enable_alt_stream=False,
    )


class TestUnifiedMXFP8Views(unittest.TestCase):
    def test_page_move_preserves_payload_and_scales(self):
        view = _view(_spec())
        for i, family in enumerate(view._row_buffers(), 1):
            for row in family:
                row.view(torch.uint8)[2:4] = i * 31
        view._arena.move_units("full", torch.tensor([3]), torch.tensor([1]))
        for i, family in enumerate(view._row_buffers(), 1):
            for row in family:
                self.assertTrue(torch.all(row.view(torch.uint8)[6:8] == i * 31))


if __name__ == "__main__":
    unittest.main()
