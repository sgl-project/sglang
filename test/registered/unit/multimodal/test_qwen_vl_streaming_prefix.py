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
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
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


VOCAB = 1 << 20
RAW_MM_TOKEN = 555


def _no_pad_page_item(offset_in_page: int, span: int, pad_value: int, hash_id: int):
    """A synthetic MM item without any Qwen grid geometry."""
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=hash_id,
        pad_value=pad_value,
        offsets=[(offset_in_page, offset_in_page + span - 1)],
    )


def _expand_pad_spans(input_ids, mm_items):
    """Length-changing synthetic padding: each raw MM span becomes three pad
    tokens.  Offsets address the UNPADDED sequence, mirroring the contract of
    Scheduler.pad_input_ids_func(origin_input_ids_unpadded, mm_inputs)."""
    result = []
    cursor = 0
    for item in sorted(mm_items, key=lambda it: it.offsets[0][0]):
        for start, end in item.offsets:
            result.extend(input_ids[cursor:start])
            result.extend([item.pad_value] * 3)
            cursor = end + 1
    result.extend(input_ids[cursor:])
    return result


def _session_recv(rid, input_ids, max_new_tokens=8):
    """A TokenizedGenerateReqInput stand-in accepted by Session.create_req."""
    return SimpleNamespace(
        rid=rid,
        input_ids=array("q", input_ids),
        mm_inputs=None,
        session_params=SimpleNamespace(
            id="s", rid=None, offset=None, replace=False, drop_previous_output=False
        ),
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


class _CarryReq:
    """Minimal streaming-session carry: previous padded prefix plus raw token
    IDs of the new turn, exactly as Session._share_token_arrays produces."""

    def __init__(self, unpadded_ids, origin_ids, multimodal_inputs):
        self.session = SimpleNamespace(streaming=True)
        self.origin_input_ids_unpadded = array("q", unpadded_ids)
        self.origin_input_ids = array("q", origin_ids)
        # Stale carry: padding of the new turn has not been applied yet.
        self.full_untruncated_fill_ids = self.origin_input_ids[:]
        self.multimodal_inputs = multimodal_inputs

    def extend_image_inputs(self, image_inputs):
        if self.multimodal_inputs is None:
            self.multimodal_inputs = image_inputs
        else:
            self.multimodal_inputs.merge(image_inputs)


class _RecordingMropeProcessor:
    def __init__(self):
        self.calls = 0
        self.seen_input_ids = None

    def compute_mrope_positions(self, input_ids, mm_items):
        self.calls += 1
        self.seen_input_ids = list(input_ids)
        return _fast_mrope(self.seen_input_ids, mm_items)


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


class TestStreamingSessionMMEdgeCases(CustomTestCase):
    """Contract-matrix tests for the streaming-session MM rebuild: each test
    pins one input/model contract instead of assuming Qwen3-VL geometry."""

    def test_streaming_append_without_pad_function_uses_precomputed_ids(self):
        """No pad func and no M-RoPE: splice precomputed padded IDs of the
        new turn, keep the old padded prefix byte-identical, resync fill IDs."""
        old_raw = [101, RAW_MM_TOKEN, 102]
        old_padded = [101, 900, 102]
        new_raw = [201, RAW_MM_TOKEN, 203]
        new_padded = [201, 901, 203]

        req = _CarryReq(
            unpadded_ids=old_raw + new_raw,
            origin_ids=old_padded + new_raw,
            multimodal_inputs=MultimodalInputs(
                mm_items=[_no_pad_page_item(1, 1, pad_value=900, hash_id=11)]
            ),
        )
        recv_req = SimpleNamespace(input_ids=array("q", new_raw))
        new_inputs = MultimodalInputs(
            mm_items=[_no_pad_page_item(1, 1, pad_value=901, hash_id=12)],
            padded_input_ids=list(new_padded),
        )

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pad_input_ids_func = None
        scheduler._mm_processor = None
        scheduler._prepare_multimodal_inputs_for_generate(recv_req, req, new_inputs)

        expected_origin = old_padded + new_padded
        self.assertEqual(list(req.origin_input_ids), expected_origin)
        # The old padded prefix (including its pad value) is untouched.
        self.assertEqual(list(req.origin_input_ids)[: len(old_padded)], old_padded)
        # The new turn used precomputed padded IDs, not the raw MM token.
        self.assertNotIn(RAW_MM_TOKEN, list(req.origin_input_ids))
        # Pre-fix, the carried fill array kept the raw MM token; it must now
        # be resynced to the rebuilt origin.
        self.assertEqual(list(req.full_untruncated_fill_ids), expected_origin)
        # The new item offset shifted by the cumulative UNPADDED prefix length.
        self.assertEqual(
            req.multimodal_inputs.mm_items[1].offsets,
            [(len(old_raw) + 1, len(old_raw) + 1)],
        )
        # No M-RoPE model: fields stay absent and nothing raises.
        self.assertIsNone(req.multimodal_inputs.mrope_positions)
        self.assertIsNone(req.multimodal_inputs.mrope_position_delta)

    def test_streaming_multimodal_append_after_generated_tokens(self):
        """Generated tokens between two MM turns are retained exactly once and
        every rebuilt structure is based on the cumulative sequence including
        them.  Uses real Session.create_req/finish_req token-array carry."""
        grids = [(1, 4, 6), (1, 8, 4)]
        _, _, local_pages = _assemble_pages(grids)
        first_ids, first_item = local_pages[0]
        second_ids, second_item = local_pages[1]
        generated_ids = [7001, 7002]

        canonical_raw = first_ids + generated_ids + second_ids
        canonical_items = [
            copy.deepcopy(first_item),
            _shift_item(second_item, len(first_ids) + len(generated_ids)),
        ]
        canonical_padded = MultimodalProcessorOutput.build_padded_input_ids(
            canonical_raw, canonical_items
        )
        canonical_positions, canonical_delta = _fast_mrope(
            canonical_raw, canonical_items
        )

        processor = _RecordingMropeProcessor()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._mm_processor = processor
        scheduler.pad_input_ids_func = lambda input_ids, mm_inputs: (
            MultimodalProcessorOutput.build_padded_input_ids(
                list(input_ids), mm_inputs.mm_items
            )
        )
        session = Session(capacity_of_str_len=0, session_id="s", streaming=True)

        # Turn 1: real session request + scheduler MM preparation.
        recv1 = _session_recv("r1", first_ids)
        req1 = session.create_req(recv1, tokenizer=None, vocab_size=VOCAB)
        scheduler._prepare_multimodal_inputs_for_generate(
            recv1, req1, _page_inputs(first_ids, first_item)
        )
        turn1_positions = req1.multimodal_inputs.mrope_positions.clone()
        # Deterministic decode, then a successful finish.
        req1.output_ids.extend(generated_ids)
        req1._refresh_fill_ids()
        session.finish_req(req1)

        # Turn 2: the session carries page1 + generated + page2.
        recv2 = _session_recv("r2", second_ids)
        req2 = session.create_req(recv2, tokenizer=None, vocab_size=VOCAB)
        req2.multimodal_inputs.mrope_position_delta_repeated_cache = torch.tensor([99])
        scheduler._prepare_multimodal_inputs_for_generate(
            recv2, req2, _page_inputs(second_ids, second_item)
        )

        self.assertEqual(list(req2.origin_input_ids_unpadded), canonical_raw)
        # Generated IDs appear exactly once in every rebuilt array.
        for token in generated_ids:
            self.assertEqual(list(req2.origin_input_ids_unpadded).count(token), 1)
            self.assertEqual(list(req2.origin_input_ids).count(token), 1)
        self.assertEqual(list(req2.origin_input_ids), canonical_padded)
        self.assertEqual(list(req2.full_untruncated_fill_ids), canonical_padded)
        # The second page MM offset includes the generated-token length.
        self.assertEqual(
            req2.multimodal_inputs.mm_items[1].offsets, canonical_items[1].offsets
        )
        # M-RoPE recomputed from the cumulative UNPADDED sequence.
        self.assertEqual(processor.seen_input_ids, canonical_raw)
        self.assertTrue(
            torch.equal(req2.multimodal_inputs.mrope_positions, canonical_positions)
        )
        self.assertTrue(
            torch.equal(req2.multimodal_inputs.mrope_position_delta, canonical_delta)
        )
        self.assertIsNone(req2.multimodal_inputs.mrope_position_delta_repeated_cache)
        # Turn 1 positions stay prefix-exact after the append.
        self.assertTrue(
            torch.equal(
                req2.multimodal_inputs.mrope_positions[:, : len(first_ids)],
                turn1_positions,
            )
        )

    def test_streaming_append_rebuilds_length_changing_padding_from_unpadded_ids(
        self,
    ):
        """Length-changing padding plus a generated token between turns: the
        cumulative repad from unpadded IDs must equal a one-shot pad of the
        full raw sequence."""
        page1_raw = [10, RAW_MM_TOKEN, 11]
        page2_raw = [20, RAW_MM_TOKEN, 21]
        generated_ids = [7001]
        item1 = _no_pad_page_item(1, 1, pad_value=910, hash_id=21)
        item2 = _no_pad_page_item(1, 1, pad_value=920, hash_id=22)

        canonical_raw = page1_raw + generated_ids + page2_raw
        canonical_items = [
            copy.deepcopy(item1),
            _shift_item(item2, len(page1_raw) + len(generated_ids)),
        ]
        canonical_padded = _expand_pad_spans(canonical_raw, canonical_items)

        req = _CarryReq(
            unpadded_ids=canonical_raw,
            # Previous padded prefix of turn 1 (5 tokens) + generated + new raw.
            origin_ids=(
                _expand_pad_spans(page1_raw, [item1]) + generated_ids + page2_raw
            ),
            multimodal_inputs=MultimodalInputs(mm_items=[copy.deepcopy(item1)]),
        )
        recv_req = SimpleNamespace(input_ids=array("q", page2_raw))
        new_inputs = MultimodalInputs(mm_items=[copy.deepcopy(item2)])

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.pad_input_ids_func = lambda input_ids, mm_inputs: (
            _expand_pad_spans(list(input_ids), mm_inputs.mm_items)
        )
        scheduler._mm_processor = None
        scheduler._prepare_multimodal_inputs_for_generate(recv_req, req, new_inputs)

        # The shift uses the unpadded prefix (4 tokens); the padded prefix is
        # 6 tokens, so a padded-base shift would wrongly produce offset 7.
        shifted_offset = len(page1_raw) + len(generated_ids) + 1
        self.assertEqual(
            req.multimodal_inputs.mm_items[1].offsets,
            [(shifted_offset, shifted_offset)],
        )
        self.assertEqual(list(req.origin_input_ids_unpadded), canonical_raw)
        self.assertEqual(list(req.origin_input_ids), canonical_padded)
        self.assertEqual(list(req.full_untruncated_fill_ids), canonical_padded)
        # The generated token is neither duplicated nor dropped.
        self.assertEqual(list(req.origin_input_ids).count(7001), 1)

    def test_non_streaming_multimodal_preparation_keeps_existing_fast_path(self):
        """Ordinary requests keep the pre-existing path: precomputed padded
        IDs splice first, no cumulative repad, no forced M-RoPE recompute."""
        prefix = [7, 8]
        new_raw = [1, RAW_MM_TOKEN, 2]
        new_padded = [1, 900, 2]
        positions = torch.arange(5).unsqueeze(0).expand(3, -1).clone()
        delta = torch.tensor([3])

        class _Req:
            def __init__(self, session, origin_ids, unpadded_ids):
                self.session = session
                self.origin_input_ids = array("q", origin_ids)
                self.origin_input_ids_unpadded = array("q", unpadded_ids)
                self.full_untruncated_fill_ids = array("q")
                self.multimodal_inputs = None

            def extend_image_inputs(self, image_inputs):
                self.multimodal_inputs = image_inputs

        # Case A: precomputed padded IDs win; pad func / processor untouched.
        pad_calls = []
        processor = _RecordingMropeProcessor()
        scheduler = Scheduler.__new__(Scheduler)
        scheduler._mm_processor = processor

        def _recording_pad(input_ids, mm_inputs):
            pad_calls.append(list(input_ids))
            return MultimodalProcessorOutput.build_padded_input_ids(
                list(input_ids), mm_inputs.mm_items
            )

        scheduler.pad_input_ids_func = _recording_pad

        for session in (None, SimpleNamespace(streaming=False)):
            with self.subTest(session=session):
                req = _Req(session, prefix + new_raw, prefix + new_raw)
                recv_req = SimpleNamespace(input_ids=array("q", new_raw))
                image_inputs = MultimodalInputs(
                    mm_items=[_no_pad_page_item(1, 1, pad_value=900, hash_id=31)],
                    padded_input_ids=list(new_padded),
                    mrope_positions=positions,
                    mrope_position_delta=delta,
                )
                scheduler._prepare_multimodal_inputs_for_generate(
                    recv_req, req, image_inputs
                )

                self.assertEqual(list(req.origin_input_ids), prefix + new_padded)
                # No cumulative repad on the ordinary path.
                self.assertEqual(pad_calls, [])
                # Existing M-RoPE is kept; no forced recomputation.
                self.assertEqual(processor.calls, 0)
                self.assertIs(req.multimodal_inputs.mrope_positions, positions)
                self.assertIs(req.multimodal_inputs.mrope_position_delta, delta)
                # The non-streaming path does not rebuild fill IDs here.
                self.assertEqual(list(req.full_untruncated_fill_ids), [])

        # Case B: no precomputed padded IDs -> page-local pad on
        # origin_input_ids (NOT a cumulative repad from unpadded IDs).
        req = _Req(
            None,
            origin_ids=[77, 88] + new_raw,  # carried/origin view
            unpadded_ids=prefix + new_raw,
        )
        recv_req = SimpleNamespace(input_ids=array("q", new_raw))
        image_inputs = MultimodalInputs(
            mm_items=[_no_pad_page_item(1, 1, pad_value=900, hash_id=32)],
            padded_input_ids=None,
        )
        # A model without M-RoPE: no processor, so the pad fallback runs
        # without position recomputation (same as the pre-PR behavior).
        scheduler._mm_processor = None
        scheduler._prepare_multimodal_inputs_for_generate(recv_req, req, image_inputs)

        # The pad func saw the ORIGIN array (old behavior), not the unpadded
        # cumulative array the streaming path would have used.
        self.assertEqual(pad_calls, [[77, 88] + new_raw])
        self.assertEqual(list(req.origin_input_ids), [77, 88] + new_padded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
