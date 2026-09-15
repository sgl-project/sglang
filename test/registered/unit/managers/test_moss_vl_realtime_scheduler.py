import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.managers.scheduler import Scheduler

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_scheduler(
    *,
    architectures=("MossVLForConditionalGeneration",),
    vision_seq_pad_multiple=1,
    page_size=1,
    allocator_page_size=1,
    chunked_prefill_size=None,
):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=list(architectures),
            vision_seq_pad_multiple=vision_seq_pad_multiple,
        )
    )
    scheduler.page_size = page_size
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        page_size=allocator_page_size
    )
    scheduler.tree_cache = SimpleNamespace(
        page_size=1, has_slot=MagicMock(return_value=True)
    )
    scheduler.chunked_prefill_size = chunked_prefill_size
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    scheduler._mm_processor = SimpleNamespace(merge_realtime_inputs=MagicMock())
    scheduler.pad_input_ids_func = MagicMock()
    return scheduler


def _make_req(*, streaming=True, multimodal_inputs=None):
    return SimpleNamespace(
        session=SimpleNamespace(streaming=streaming, session_id="session-a"),
        origin_input_ids=array("q", [10, 20]),
        origin_input_ids_unpadded=array("q", [10, 20]),
        multimodal_inputs=multimodal_inputs,
        incremental_encoder_cache_prefix_len=None,
    )


class TestMossVLRealtimeScheduler(unittest.TestCase):
    def test_gate_requires_exact_architecture_and_streaming_session(self):
        scheduler = _make_scheduler()
        self.assertTrue(
            scheduler._is_moss_vl_realtime_request(_make_req(streaming=True))
        )

        cases = (
            ((), _make_req(streaming=True)),
            (("MossVLForConditionalGenerationExtra",), _make_req(streaming=True)),
            (("Qwen2VLForConditionalGeneration",), _make_req(streaming=True)),
            (("MossVLForConditionalGeneration",), _make_req(streaming=False)),
            (
                ("MossVLForConditionalGeneration",),
                SimpleNamespace(session=None),
            ),
        )
        for architectures, req in cases:
            with self.subTest(architectures=architectures, session=req.session):
                scheduler.model_config.hf_config.architectures = list(architectures)
                self.assertFalse(scheduler._is_moss_vl_realtime_request(req))

    def test_gate_excludes_same_architecture_instruct_checkpoint(self):
        scheduler = _make_scheduler(vision_seq_pad_multiple=8)

        self.assertFalse(
            scheduler._is_moss_vl_realtime_request(_make_req(streaming=True))
        )

    def test_realtime_text_only_placeholder_is_discarded(self):
        scheduler = _make_scheduler()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            model_specific_data={"moss_vl_text_only_placeholder": True},
        )
        current = MultimodalInputs(mm_items=[item])
        current.release_features = MagicMock()

        resolved = scheduler._resolve_moss_vl_text_only_placeholder(
            _make_req(streaming=True), current
        )

        current.release_features.assert_called_once_with()
        self.assertIsNot(resolved, current)
        self.assertEqual(resolved.mm_items, [])

    def test_regular_session_keeps_text_only_placeholder_error(self):
        scheduler = _make_scheduler()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            model_specific_data={"moss_vl_text_only_placeholder": True},
        )
        current = MultimodalInputs(mm_items=[item])

        with self.assertRaisesRegex(
            ValueError, "found 1 frame\\(s\\) and 0 token\\(s\\)"
        ):
            scheduler._resolve_moss_vl_text_only_placeholder(
                _make_req(streaming=False), current
            )

    def test_unmarked_multimodal_input_is_unchanged(self):
        scheduler = _make_scheduler()
        current = MultimodalInputs(
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE)]
        )

        resolved = scheduler._resolve_moss_vl_text_only_placeholder(
            _make_req(streaming=True), current
        )

        self.assertIs(resolved, current)

    def test_page_size_is_rejected_before_merge_or_padding(self):
        cases = (
            (16, 1, "scheduler page size"),
            (1, 16, "allocator page size with a page-one tree"),
        )
        for page_size, allocator_page_size, label in cases:
            with self.subTest(label=label):
                scheduler = _make_scheduler(
                    page_size=page_size,
                    allocator_page_size=allocator_page_size,
                )

                with self.assertRaisesRegex(ValueError, "page size of 1"):
                    scheduler._prepare_moss_vl_realtime_inputs(_make_req(), None)

                scheduler._mm_processor.merge_realtime_inputs.assert_not_called()
                scheduler.pad_input_ids_func.assert_not_called()

    def test_chunked_prefill_is_rejected_before_merge_or_padding(self):
        scheduler = _make_scheduler(chunked_prefill_size=4096)

        with self.assertRaisesRegex(ValueError, "chunked prefill"):
            scheduler._prepare_moss_vl_realtime_inputs(_make_req(), None)

        scheduler._mm_processor.merge_realtime_inputs.assert_not_called()
        scheduler.pad_input_ids_func.assert_not_called()

    def test_unsupported_serving_modes_are_rejected_before_merge_or_padding(self):
        cases = (
            (DisaggregationMode.PREFILL, True, "disaggregated serving"),
            (DisaggregationMode.DECODE, True, "disaggregated serving"),
            (DisaggregationMode.NULL, False, "speculative decoding"),
        )
        for disaggregation_mode, spec_is_none, error in cases:
            with self.subTest(
                disaggregation_mode=disaggregation_mode,
                spec_is_none=spec_is_none,
            ):
                scheduler = _make_scheduler()
                scheduler.disaggregation_mode = disaggregation_mode
                scheduler.spec_algorithm = SimpleNamespace(
                    is_none=lambda value=spec_is_none: value
                )

                with self.assertRaisesRegex(ValueError, error):
                    scheduler._prepare_moss_vl_realtime_inputs(_make_req(), None)

                scheduler._mm_processor.merge_realtime_inputs.assert_not_called()
                scheduler.pad_input_ids_func.assert_not_called()

    def test_merge_and_padding_use_cumulative_unpadded_ids(self):
        scheduler = _make_scheduler()
        previous = MultimodalInputs(
            mm_items=[],
            incremental_encoder_cache=True,
            encoder_cached_len=5,
        )
        current = MultimodalInputs(mm_items=[])
        req = _make_req(multimodal_inputs=previous)
        cumulative_unpadded_ids = array("q", [10, 20, 30, 40])
        req.origin_input_ids_unpadded = cumulative_unpadded_ids
        req.origin_input_ids = array("q", [10, 99, 99, 20])
        merged = MultimodalInputs(
            mm_items=[],
            incremental_encoder_cache=True,
            encoder_cached_len=5,
            encoder_append_len=7,
        )
        scheduler._mm_processor.merge_realtime_inputs.return_value = merged
        scheduler.pad_input_ids_func.return_value = [10, 88, 88, 20, 30, 40]

        scheduler._prepare_moss_vl_realtime_inputs(req, current)

        merge_args = scheduler._mm_processor.merge_realtime_inputs.call_args.args
        self.assertIs(merge_args[0], cumulative_unpadded_ids)
        self.assertIs(merge_args[1], previous)
        self.assertIs(merge_args[2], current)
        pad_args = scheduler.pad_input_ids_func.call_args.args
        self.assertIs(pad_args[0], cumulative_unpadded_ids)
        self.assertIs(pad_args[1], merged)
        self.assertIs(req.multimodal_inputs, merged)
        self.assertTrue(req.multimodal_inputs.incremental_encoder_cache)
        self.assertEqual(req.incremental_encoder_cache_prefix_len, 5)
        self.assertEqual(req.origin_input_ids, array("q", [10, 88, 88, 20, 30, 40]))

    def test_text_only_first_turn_still_enters_incremental_mode(self):
        scheduler = _make_scheduler()
        req = _make_req(multimodal_inputs=None)
        merged = MultimodalInputs(
            mm_items=[],
            incremental_encoder_cache=True,
            encoder_cached_len=0,
            encoder_append_len=0,
        )
        scheduler._mm_processor.merge_realtime_inputs.return_value = merged
        scheduler.pad_input_ids_func.return_value = list(req.origin_input_ids_unpadded)

        scheduler._prepare_moss_vl_realtime_inputs(req, None)

        merge_args = scheduler._mm_processor.merge_realtime_inputs.call_args.args
        self.assertIsNone(merge_args[1])
        self.assertIsInstance(merge_args[2], MultimodalInputs)
        self.assertEqual(merge_args[2].mm_items, [])
        self.assertTrue(req.multimodal_inputs.incremental_encoder_cache)
        self.assertEqual(req.incremental_encoder_cache_prefix_len, 0)
        self.assertEqual(req.origin_input_ids, req.origin_input_ids_unpadded)

    def test_previous_turn_requires_a_live_session_cache_slot(self):
        scheduler = _make_scheduler()
        scheduler.tree_cache.has_slot.return_value = False
        previous = MultimodalInputs(
            mm_items=[],
            incremental_encoder_cache=True,
            encoder_cached_len=5,
        )
        req = _make_req(multimodal_inputs=previous)

        with self.assertRaisesRegex(ValueError, "create a new streaming session"):
            scheduler._prepare_moss_vl_realtime_inputs(req, None)

        scheduler.tree_cache.has_slot.assert_called_once_with(req.session.session_id)
        scheduler._mm_processor.merge_realtime_inputs.assert_not_called()

    def test_queue_limit_aborts_incremental_cache_state(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_queued_requests = 0
        scheduler.waiting_queue = []
        scheduler.enable_priority_scheduling = False
        scheduler.tree_cache = SimpleNamespace(abort_queued_req=MagicMock())
        scheduler.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=MagicMock())
        )
        req = SimpleNamespace(
            rid="realtime-request",
            weight_version_events=[],
            output_ids=[],
            time_stats=SimpleNamespace(trace_ctx=SimpleNamespace(abort=MagicMock())),
        )

        with patch(
            "sglang.srt.managers.scheduler._make_abort_req",
            return_value=SimpleNamespace(),
        ):
            self.assertTrue(scheduler._abort_on_queued_limit(req))

        scheduler.tree_cache.abort_queued_req.assert_called_once_with(req)
        scheduler.ipc_channels.send_to_tokenizer.send_output.assert_called_once()


if __name__ == "__main__":
    unittest.main()
