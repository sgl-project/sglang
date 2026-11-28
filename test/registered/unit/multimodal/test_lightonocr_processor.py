"""Unit tests for LightOnOCRProcessor break/end token removal and offset remapping."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.multimodal.processors.lightonocr import LightOnOCRProcessor
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_processor(break_token_ids=None):
    processor = object.__new__(LightOnOCRProcessor)
    processor._break_token_ids = (
        {101, 102} if break_token_ids is None else set(break_token_ids)
    )
    return processor


def test_lightonocr_removes_break_tokens_and_remaps_multi_row_offsets():
    processor = _make_processor({101, 102})
    # Image 1: [99, 99, 101, 99, 99, 102] at old indices 2..7
    # Image 2: [99, 101, 99, 102] at old indices 10..13
    input_ids = [1, 2, 99, 99, 101, 99, 99, 102, 3, 4, 99, 101, 99, 102, 5]
    mm_item_1 = SimpleNamespace(offsets=[(2, 3), (5, 6)])
    mm_item_2 = SimpleNamespace(offsets=[(10, 13)])
    fake_result = SimpleNamespace(input_ids=input_ids, mm_items=[mm_item_1, mm_item_2])

    with patch.object(
        PixtralProcessor,
        "process_mm_data_async",
        new=AsyncMock(return_value=fake_result),
    ):
        out = asyncio.run(
            processor.process_mm_data_async(["img1", "img2"], "prompt", None)
        )

    assert out.input_ids == [1, 2, 99, 99, 99, 99, 3, 4, 99, 99, 5]
    assert mm_item_1.offsets == [(2, 5)]
    assert mm_item_2.offsets == [(8, 9)]


def test_lightonocr_clears_offsets_when_all_span_tokens_removed():
    processor = _make_processor({101, 102})
    input_ids = [1, 101, 102, 2]
    mm_item = SimpleNamespace(offsets=[(1, 2)])
    fake_result = SimpleNamespace(input_ids=input_ids, mm_items=[mm_item])

    with patch.object(
        PixtralProcessor,
        "process_mm_data_async",
        new=AsyncMock(return_value=fake_result),
    ):
        out = asyncio.run(processor.process_mm_data_async(["img1"], "prompt", None))

    assert out.input_ids == [1, 2]
    assert mm_item.offsets == []
