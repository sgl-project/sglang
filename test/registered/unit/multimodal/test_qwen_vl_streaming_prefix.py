"""Regression tests for prefix-stable Qwen3-VL streaming sessions.

These tests deliberately stay below the model runner.  They answer the first
two IngestFlow Gate-0 questions with deterministic processor geometry:

1. Can page-local token layouts, media offsets, and padded IDs be assembled
   into the same canonical tensors as a one-shot multi-page request?
2. Does appending future image pages leave every already-computed Qwen3-VL
   M-RoPE position unchanged?

The final test exercises the scheduler repair that rebuilds cumulative padded
IDs, fill IDs, and M-RoPE metadata after a multimodal session append.
"""

import copy
import random
import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


IMAGE_TOKEN_ID = 151655
VISION_START_ID = 151652
VISION_END_ID = 151653
VIDEO_TOKEN_ID = 151656
SPATIAL_MERGE_SIZE = 2


class _Qwen3VLGeometry:
    model_type = "qwen3_vl"
    _spatial_merge_size = SPATIAL_MERGE_SIZE
    _as_grid_batch = staticmethod(QwenVLImageProcessor._as_grid_batch)


PROCESSOR_GEOMETRY = _Qwen3VLGeometry()


def _make_page(page_index: int, grid: tuple[int, int, int]):
    t, h, w = grid
    image_token_count = t * (h // SPATIAL_MERGE_SIZE) * (w // SPATIAL_MERGE_SIZE)
    # Each page is separated by ordinary text plus vision boundary tokens.  In
    # the real prototype these boundaries are produced by a fixed page template.
    input_ids = [
        1000 + page_index,
        VISION_START_ID,
        *([IMAGE_TOKEN_ID] * image_token_count),
        VISION_END_ID,
        2000 + page_index,
    ]
    start = 2
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=page_index + 1,
        pad_value=300000 + page_index,
        offsets=[(start, start + image_token_count - 1)],
        model_specific_data={"image_grid_thw": torch.tensor([grid])},
    )
    return input_ids, item


def _shift_item(item: MultimodalDataItem, prefix_len: int):
    item = copy.deepcopy(item)
    item.offsets = [
        (start + prefix_len, end + prefix_len) for start, end in item.offsets
    ]
    return item


def _assemble_pages(grids: list[tuple[int, int, int]]):
    input_ids = []
    items = []
    local_pages = []
    for page_index, grid in enumerate(grids):
        page_ids, page_item = _make_page(page_index, grid)
        local_pages.append((page_ids, page_item))
        items.append(_shift_item(page_item, len(input_ids)))
        input_ids.extend(page_ids)
    return input_ids, items, local_pages


def _fast_mrope(input_ids, items):
    positions, delta = (
        QwenVLImageProcessor._compute_image_only_mrope_positions_from_offsets(
            PROCESSOR_GEOMETRY,
            input_len=len(input_ids),
            mm_items=items,
            dtype=torch.long,
            device=torch.device("cpu"),
        )
    )
    return positions.squeeze(1), delta


def _reference_mrope(input_ids, grids):
    positions, delta = MRotaryEmbedding.get_rope_index(
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        model_type="qwen3_vl",
        tokens_per_second=None,
        input_ids=torch.tensor(input_ids).unsqueeze(0),
        image_grid_thw=torch.tensor(grids),
        video_grid_thw=None,
    )
    return positions.squeeze(1), delta


def _page_inputs(page_ids, page_item):
    positions, delta = _fast_mrope(page_ids, [page_item])
    return MultimodalInputs(
        mm_items=[copy.deepcopy(page_item)],
        padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
            page_ids, [page_item]
        ),
        mrope_positions=positions,
        mrope_position_delta=delta,
        im_token_id=IMAGE_TOKEN_ID,
        im_start_id=VISION_START_ID,
        im_end_id=VISION_END_ID,
    )


class TestQwen3VLStreamingPrefix(CustomTestCase):
    def test_grid_extremes_preserve_prefix_positions(self):
        """Cover the smallest grid and a large document-page grid explicitly."""
        grids = [(1, 2, 2), (1, 64, 64)]
        full_ids, full_items, _ = _assemble_pages(grids)
        full_positions, full_delta = _fast_mrope(full_ids, full_items)
        ref_positions, ref_delta = _reference_mrope(full_ids, grids)

        prefix_ids, prefix_items, _ = _assemble_pages(grids[:1])
        prefix_positions, _ = _fast_mrope(prefix_ids, prefix_items)

        self.assertTrue(torch.equal(full_positions, ref_positions))
        self.assertTrue(torch.equal(full_delta, ref_delta))
        self.assertTrue(
            torch.equal(prefix_positions, full_positions[:, : len(prefix_ids)])
        )

    def test_random_page_assembly_and_prefix_stability(self):
        """Exercise 1,000 deterministic documents with 1-20 image pages."""
        rng = random.Random(0x51A6)
        grid_values = (2, 4, 6, 8)

        for case_index in range(1000):
            page_count = rng.randint(1, 20)
            grids = [
                (1, rng.choice(grid_values), rng.choice(grid_values))
                for _ in range(page_count)
            ]
            full_ids, full_items, local_pages = _assemble_pages(grids)
            full_positions, full_delta = _fast_mrope(full_ids, full_items)
            ref_positions, ref_delta = _reference_mrope(full_ids, grids)

            with self.subTest(case=case_index, pages=page_count):
                self.assertTrue(torch.equal(full_positions, ref_positions))
                self.assertTrue(torch.equal(full_delta, ref_delta))

                # Page-local padded IDs and shifted offsets compose exactly.
                streamed_padded_ids = []
                streamed_offsets = []
                prefix_len = 0
                for page_ids, page_item in local_pages:
                    page_padded = MultimodalProcessorOutput.build_padded_input_ids(
                        page_ids, [page_item]
                    )
                    streamed_padded_ids.extend(page_padded)
                    streamed_offsets.extend(_shift_item(page_item, prefix_len).offsets)
                    prefix_len += len(page_ids)

                canonical_padded_ids = MultimodalProcessorOutput.build_padded_input_ids(
                    full_ids, full_items
                )
                self.assertEqual(streamed_padded_ids, canonical_padded_ids)
                self.assertEqual(
                    streamed_offsets,
                    [offset for item in full_items for offset in item.offsets],
                )

                # A future append must not change positions already used by KV.
                if page_count > 1:
                    prefix_pages = rng.randint(1, page_count - 1)
                    prefix_ids, prefix_items, _ = _assemble_pages(grids[:prefix_pages])
                    prefix_positions, _ = _fast_mrope(prefix_ids, prefix_items)
                    self.assertTrue(
                        torch.equal(
                            prefix_positions,
                            full_positions[:, : len(prefix_ids)],
                        )
                    )

    def test_streaming_append_rebuilds_cumulative_multimodal_state(self):
        grids = [(1, 4, 6), (1, 8, 4)]
        full_ids, full_items, local_pages = _assemble_pages(grids)
        canonical_positions, canonical_delta = _fast_mrope(full_ids, full_items)
        canonical_padded_ids = MultimodalProcessorOutput.build_padded_input_ids(
            full_ids, full_items
        )

        first_ids, first_item = local_pages[0]
        second_ids, second_item = local_pages[1]
        first_inputs = _page_inputs(first_ids, first_item)
        first_inputs.mrope_position_delta_repeated_cache = torch.tensor([99])
        second_inputs = _page_inputs(second_ids, second_item)

        class _Req:
            def __init__(self):
                first_padded_ids = MultimodalProcessorOutput.build_padded_input_ids(
                    first_ids, [first_item]
                )
                self.session = SimpleNamespace(streaming=True)
                self.origin_input_ids_unpadded = array("q", first_ids + second_ids)
                self.origin_input_ids = array("q", first_padded_ids + second_ids)
                # This is the stale carry produced before scheduler-side MM
                # padding rewrites the second page's token values.
                self.full_untruncated_fill_ids = self.origin_input_ids[:]
                self.multimodal_inputs = first_inputs

            def extend_image_inputs(self, image_inputs):
                self.multimodal_inputs.merge(image_inputs)

        class _Processor:
            def __init__(self):
                self.seen_input_ids = None

            def compute_mrope_positions(self, input_ids, mm_items):
                self.seen_input_ids = list(input_ids)
                return _fast_mrope(self.seen_input_ids, mm_items)

        processor = _Processor()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._mm_processor = processor
        scheduler.pad_input_ids_func = lambda input_ids, mm_inputs: (
            MultimodalProcessorOutput.build_padded_input_ids(
                list(input_ids), mm_inputs.mm_items
            )
        )

        req = _Req()
        recv_req = SimpleNamespace(input_ids=array("q", second_ids))
        scheduler._prepare_multimodal_inputs_for_generate(recv_req, req, second_inputs)

        self.assertEqual(processor.seen_input_ids, full_ids)
        self.assertEqual(list(req.origin_input_ids), canonical_padded_ids)
        self.assertEqual(list(req.full_untruncated_fill_ids), canonical_padded_ids)
        self.assertEqual(
            req.multimodal_inputs.mm_items[1].offsets, full_items[1].offsets
        )
        self.assertTrue(
            torch.equal(req.multimodal_inputs.mrope_positions, canonical_positions)
        )
        self.assertTrue(
            torch.equal(req.multimodal_inputs.mrope_position_delta, canonical_delta)
        )
        self.assertIsNone(req.multimodal_inputs.mrope_position_delta_repeated_cache)


if __name__ == "__main__":
    unittest.main(verbosity=2)
