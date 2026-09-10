"""Attention metadata is built only for the compress ratios the model has.

DeepSeek-V4 carries ratios 4/128 and DeepSeek-V4.1 carries 1/2, and the CUDA
backend serves both from one metadata container. A ratio built unconditionally
costs a per-token c128 page-index table, a c4 top-k buffer, a DeepGEMM indexer
plan and two compressor plans on every forward of a model that has none of them
-- and none of that is reachable from any layer, so no correctness test would go
red. These cases pin the gate from both layouts, plus the copy_ path that has to
tolerate a field being None on both sides.
"""

import unittest

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
    DSV4_PAGE_SIZE,
    DSV4AttentionCase,
    build_dsv4_attention_fixture,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

V4_RATIOS = [0, 4, 128]
V41_RATIOS = [0, 2, 1]

# Fields the ratio gate decides, per ratio. Every one of them is a tensor or a
# planner object on a model that has the ratio, and None on one that does not.
C4_CORE_FIELDS = (
    "c4_out_loc",
    "c4_topk_lengths_raw",
    "c4_topk_lengths_clamp1",
    "c4_sparse_topk_lengths",
    "c4_sparse_page_indices",
)
C128_CORE_FIELDS = (
    "c128_out_loc",
    "c128_topk_lengths_clamp1",
    "c128_page_indices",
)
# Checked one-way only: _create_flashmla_metadata returns None on SM120 / XPU
# even for a ratio the model has, so only the gated-off direction is pinnable.
FLASHMLA_SCHED_FIELDS = ("c4_flashmla_metadata", "c128_flashmla_metadata")
LOW_RATIO_CORE_FIELDS = tuple(
    f"c{ratio}_{suffix}"
    for ratio in (1, 2)
    for suffix in (
        "out_loc",
        "topk_lengths_clamp1",
        "sparse_topk_lengths",
        "sparse_page_indices",
    )
)
C4_TOP_FIELDS = ("indexer_metadata", "c4_compress_metadata")
C128_TOP_FIELDS = ("c128_compress_metadata",)
LOW_RATIO_TOP_FIELDS = ("c1_indexer_metadata", "c2_indexer_metadata")


def _decode_case(name: str) -> DSV4AttentionCase:
    return DSV4AttentionCase(
        name=name,
        backend="dsv4",
        forward_mode=ForwardMode.DECODE,
        num_heads=64,
        page_size=DSV4_PAGE_SIZE,
        prefix_lens=(64, 130),
    )


@unittest.skipUnless(torch.cuda.is_available(), "builds the CUDA dsv4 backend")
class TestDSV4RatioMetadataGate(CustomTestCase):
    def _decode_metadata(self, compression_ratios):
        """One decode metadata build, upgraded from raw to full."""
        fixture = build_dsv4_attention_fixture(
            self,
            _decode_case(f"ratio_metadata_{'_'.join(map(str, compression_ratios))}"),
            compression_ratios=list(compression_ratios),
        )
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        return fixture, fixture.backend.forward_metadata

    def _assert_none(self, obj, field_names):
        for name in field_names:
            with self.subTest(field=name):
                self.assertIsNone(getattr(obj, name))

    def _assert_not_none(self, obj, field_names):
        for name in field_names:
            with self.subTest(field=name):
                self.assertIsNotNone(getattr(obj, name))

    def test_v41_builds_no_c4_or_c128_metadata(self):
        fixture, metadata = self._decode_metadata(V41_RATIOS)
        core = metadata.core_metadata

        self.assertEqual(fixture.backend.present_ratios, (1, 2))
        self.assertEqual(core.present_ratios, (1, 2))
        self._assert_none(
            core, C4_CORE_FIELDS + C128_CORE_FIELDS + FLASHMLA_SCHED_FIELDS
        )
        self._assert_none(metadata, C4_TOP_FIELDS + C128_TOP_FIELDS)
        self._assert_not_none(core, LOW_RATIO_CORE_FIELDS)
        self._assert_not_none(metadata, LOW_RATIO_TOP_FIELDS)

    def test_v4_builds_no_low_ratio_metadata(self):
        fixture, metadata = self._decode_metadata(V4_RATIOS)
        core = metadata.core_metadata

        self.assertEqual(fixture.backend.present_ratios, (4, 128))
        self.assertEqual(core.present_ratios, (4, 128))
        self.assertEqual(core.low_ratios, ())
        self._assert_not_none(core, C4_CORE_FIELDS + C128_CORE_FIELDS)
        self._assert_not_none(metadata, C4_TOP_FIELDS + C128_TOP_FIELDS)
        self._assert_none(core, LOW_RATIO_CORE_FIELDS)
        self._assert_none(metadata, LOW_RATIO_TOP_FIELDS)

    def test_copy_between_two_builds_of_the_same_layout(self):
        for compression_ratios in (V4_RATIOS, V41_RATIOS):
            with self.subTest(compression_ratios=compression_ratios):
                fixture, dst = self._decode_metadata(compression_ratios)
                fixture.backend.init_forward_metadata(fixture.forward_batch)
                src = fixture.backend.forward_metadata
                self.assertIsNot(dst, src)

                # copy_ enumerates every dataclass field, so a gated-off field
                # has to be None on both sides rather than absent on one.
                dst.copy_(src)
                self.assertTrue(
                    torch.equal(
                        dst.core_metadata.seq_lens_casual,
                        src.core_metadata.seq_lens_casual,
                    )
                )


if __name__ == "__main__":
    unittest.main()
