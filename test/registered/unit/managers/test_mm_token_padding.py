"""Multimodal token-padding values, ownership, and buffer conversion coverage."""

import random
import unittest
from array import array
from collections import defaultdict
from itertools import product
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    MultiModalityDataPaddingPatternTokenPairs,
    pad_mm_input_ids,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    Req,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def mm_item(modality, pad_value, offsets):
    return MultimodalDataItem(modality=modality, pad_value=pad_value, offsets=offsets)


def legacy_pad_tokens(input_ids, mm_inputs):
    """List-based reference; production padding must never take this path."""
    values = list(input_ids)
    if not values or not mm_inputs.mm_items:
        return values
    tensor = torch.as_tensor(values)
    items_by_modality = defaultdict(list)
    for item in mm_inputs.mm_items:
        items_by_modality[item.modality].append(item)
    token_ids = {
        Modality.IMAGE: mm_inputs.im_token_id,
        Modality.AUDIO: mm_inputs.audio_token_id,
        Modality.VIDEO: mm_inputs.video_token_id,
    }
    for modality, items in items_by_modality.items():
        if token_ids.get(modality) is None:
            continue
        for item in items:
            for start, end in item.offsets:
                tensor[start : end + 1] = item.pad_value
    return tensor.tolist()


class TestMultimodalTokenPadding(CustomTestCase):
    def setUp(self):
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()

    def test_array_matches_legacy_and_preserves_input(self):
        mm = MultimodalInputs(
            mm_items=[
                mm_item(Modality.IMAGE, 2**40 + 17, [(2, 4), (9, 10)]),
                mm_item(Modality.AUDIO, 2**40 + 29, [(6, 7)]),
                mm_item(Modality.VIDEO, 2**40 + 41, [(12, 14)]),
            ],
            im_token_id=0,
            audio_token_id=1,
            video_token_id=2,
        )
        values = list(range(257, 274))
        expected = legacy_pad_tokens(values, mm)
        source = array("q", values)
        actual = self.pattern.pad_input_tokens(source, mm)
        self.assertIsInstance(actual, array)
        self.assertEqual(actual.typecode, "q")
        self.assertEqual(list(actual), expected)
        self.assertEqual(list(source), values)
        self.assertEqual(values, list(range(257, 274)))

    def test_empty_inputs_and_no_items_keep_identity(self):
        with patch.object(
            torch, "frombuffer", side_effect=AssertionError("empty buffer")
        ):
            values = array("q")
            mm = MultimodalInputs(
                mm_items=[mm_item(Modality.IMAGE, 1000, [(0, 0)])], im_token_id=1
            )
            self.assertIs(self.pattern.pad_input_tokens(values, mm), values)
            values = array("q", [300, 301])
            self.assertIs(
                self.pattern.pad_input_tokens(values, MultimodalInputs(mm_items=[])),
                values,
            )

    def test_missing_modality_token_id_is_skipped(self):
        source = array("q", range(300, 306))
        mm = MultimodalInputs(
            mm_items=[
                mm_item(Modality.AUDIO, 1001, [(0, 5)]),
                mm_item(Modality.IMAGE, 1002, [(2, 3)]),
                mm_item(Modality.VIDEO, 1003, [(0, 5)]),
            ],
            im_token_id=0,
        )
        self.assertEqual(
            list(self.pattern.pad_input_tokens(source, mm)),
            [300, 301, 1002, 1002, 304, 305],
        )

    def test_overlap_preserves_modality_group_and_item_order(self):
        # IMAGE is processed as a group before AUDIO, including the second
        # IMAGE item even though it appears after AUDIO in the input list.
        mm = MultimodalInputs(
            mm_items=[
                mm_item(Modality.IMAGE, 1001, [(1, 4)]),
                mm_item(Modality.AUDIO, 1002, [(3, 5)]),
                mm_item(Modality.IMAGE, 1003, [(2, 3)]),
            ],
            im_token_id=1,
            audio_token_id=2,
        )
        expected = [300, 1001, 1003, 1002, 1002, 1002, 306]
        source = array("q", range(300, 307))
        self.assertEqual(list(self.pattern.pad_input_tokens(source, mm)), expected)

    def test_slice_boundaries_match_legacy_tensor_behavior(self):
        offsets = [(0, 0), (7, 7), (-3, -2), (-50, 2), (5, 50), (5, 3), (50, 60)]
        for offset in offsets:
            with self.subTest(offset=offset):
                values = list(range(300, 308))
                mm = MultimodalInputs(
                    mm_items=[mm_item(Modality.IMAGE, 1001, [offset])], im_token_id=1
                )
                expected = legacy_pad_tokens(values, mm)
                actual = self.pattern.pad_input_tokens(array("q", values), mm)
                self.assertEqual(list(actual), expected)
                self.assertEqual(len(actual), len(values))

    def test_int64_values_do_not_lose_precision(self):
        values = [-(2**63), 2**53 + 17, 2**63 - 1, -1]
        for pad_value in (-(2**63), 2**53 + 29, 2**63 - 1):
            with self.subTest(pad_value=pad_value):
                mm = MultimodalInputs(
                    mm_items=[mm_item(Modality.IMAGE, pad_value, [(1, 1)])],
                    im_token_id=1,
                )
                actual = self.pattern.pad_input_tokens(array("q", values), mm)
                self.assertEqual(list(actual), [values[0], pad_value, values[2], -1])

        # A failed later assignment must not expose earlier mutations through
        # the original request/unpadded-input alias.
        source = array("q", values)
        mm = MultimodalInputs(
            mm_items=[
                mm_item(Modality.IMAGE, 1001, [(0, 0)]),
                mm_item(Modality.IMAGE, 2**63, [(1, 1)]),
            ],
            im_token_id=1,
        )
        with self.assertRaises((RuntimeError, ValueError)) as baseline_error:
            legacy_pad_tokens(values, mm)
        with self.assertRaises(type(baseline_error.exception)) as buffer_error:
            self.pattern.pad_input_tokens(source, mm)
        self.assertEqual(str(buffer_error.exception), str(baseline_error.exception))
        self.assertEqual(list(source), values)

    def test_helpers_reject_non_native_int64_arrays_before_early_returns(self):
        class WrongItemSizeArray(array):
            @property
            def itemsize(self):
                return 4

        patterns = (self.pattern, MultiModalityDataPaddingPatternTokenPairs([(1, 2)]))
        invalid = [
            None,
            [],
            [300, 301],
            (),
            (300, 301),
            torch.tensor([300, 301]),
            WrongItemSizeArray("q", [300, 301]),
        ]
        invalid.extend(
            array(code, values)
            for code in ("i", "I", "Q", "l", "d")
            for values in ([], [300, 301])
        )
        for pattern in patterns:
            for source in invalid:
                for items in ([], [mm_item(Modality.IMAGE, 1001, [(0, 0)])]):
                    with self.subTest(
                        pattern=type(pattern).__name__, source=source, items=bool(items)
                    ):
                        mm = MultimodalInputs(mm_items=items, im_token_id=1)
                        with self.assertRaisesRegex(AssertionError, "array"):
                            pattern.pad_input_tokens(source, mm)

    def test_scheduler_round_trip_keeps_unpadded_request_alias(self):
        source = array("q", [300, 301, 302, 303])
        req = Req(
            rid="mm-padding",
            origin_input_text=None,
            origin_input_ids=source,
            sampling_params=SamplingParams(),
        )
        self.assertIs(req.origin_input_ids_unpadded, source)
        mm = MultimodalInputs(
            mm_items=[mm_item(Modality.IMAGE, 1001, [(1, 2)])], im_token_id=1
        )
        req.origin_input_ids = pad_mm_input_ids(
            req.origin_input_ids, mm, self.pattern.pad_input_tokens
        )
        self.assertEqual(list(req.origin_input_ids), [300, 1001, 1001, 303])
        self.assertEqual(list(req.origin_input_ids_unpadded), [300, 301, 302, 303])

    def test_array_path_does_not_box_tokens_through_tensor_conversion(self):
        class PackedOnlyArray(array):
            def __iter__(self):
                raise AssertionError("per-token iteration")

            def __getitem__(self, key):
                if not isinstance(key, slice):
                    raise AssertionError("per-token indexing")
                return super().__getitem__(key)

        source = PackedOnlyArray("q", range(257, 257 + 16384))
        original_bytes = source.tobytes()
        mm = MultimodalInputs(
            mm_items=[mm_item(Modality.IMAGE, 2**40, [(100, 200)])], im_token_id=1
        )
        with (
            patch.object(
                torch, "as_tensor", side_effect=AssertionError("sequence walk")
            ),
            patch.object(
                torch.Tensor, "tolist", side_effect=AssertionError("token boxing")
            ),
        ):
            actual = pad_mm_input_ids(source, mm, self.pattern.pad_input_tokens)
        self.assertEqual(len(actual), len(source))
        self.assertEqual(actual[99], 257 + 99)
        self.assertEqual(actual[100], 2**40)
        self.assertEqual(actual[201], 257 + 201)
        self.assertEqual(source.tobytes(), original_bytes)

    def test_result_owns_its_buffer_after_tensor_view_is_released(self):
        source = array("q", [300, 301, 302])
        mm = MultimodalInputs(
            mm_items=[mm_item(Modality.IMAGE, 1001, [(1, 1)])], im_token_id=1
        )
        actual = self.pattern.pad_input_tokens(source, mm)
        self.assertIsInstance(actual, array)
        actual.append(400)
        actual[0] = 500
        source[2] = 600
        self.assertEqual(list(actual), [500, 1001, 302, 400])
        self.assertEqual(list(source), [300, 301, 600])

    def test_randomized_array_legacy_equivalence(self):
        rng = random.Random(0)
        for case in range(100):
            size = rng.randint(1, 512)
            values = [rng.randrange(257, 2**40) for _ in range(size)]
            items = []
            for _ in range(rng.randint(1, 12)):
                offsets = [
                    (rng.randrange(-size, 2 * size), rng.randrange(-size, 2 * size))
                    for _ in range(rng.randint(0, 3))
                ]
                items.append(
                    mm_item(rng.choice(list(Modality)), rng.randrange(2**40), offsets)
                )
            mm = MultimodalInputs(
                mm_items=items,
                im_token_id=rng.choice([None, 0]),
                audio_token_id=rng.choice([None, 1]),
                video_token_id=rng.choice([None, 2]),
            )
            with self.subTest(case=case):
                source = array("q", values)
                actual = self.pattern.pad_input_tokens(source, mm)
                self.assertEqual(list(actual), legacy_pad_tokens(values, mm))
                self.assertEqual(list(source), values)


class TestTokenPairPadding(CustomTestCase):
    def test_empty_span_does_not_convert_an_unused_pad_value(self):
        source = array("q", [300, 11, 12, 301])
        mm = MultimodalInputs(mm_items=[mm_item(Modality.IMAGE, 2**63, [])])
        actual = MultiModalityDataPaddingPatternTokenPairs([(11, 12)]).pad_input_tokens(
            source, mm
        )
        self.assertEqual(actual, source)
        self.assertEqual(mm.data_offsets, [1])

    def test_array_output_preserves_markers_offsets_and_input(self):
        source = array("q", [300, 11, 1, 2, 12, 301, 21, 3, 22, 302, 11, 4, 12])
        original = source[:]
        mm = MultimodalInputs(
            mm_items=[
                mm_item(Modality.IMAGE, 2**53 + 17, []),
                mm_item(Modality.IMAGE, 2**53 + 19, []),
            ]
        )
        pattern = MultiModalityDataPaddingPatternTokenPairs([(11, 12), (21, 22)], [11])
        actual = pattern.pad_input_tokens(source, mm)
        self.assertIsInstance(actual, array)
        self.assertEqual(actual.typecode, "q")
        self.assertEqual(
            list(actual),
            [
                300,
                11,
                2**53 + 17,
                2**53 + 17,
                12,
                301,
                21,
                2**53 + 17,
                22,
                302,
                11,
                2**53 + 19,
                12,
            ],
        )
        self.assertEqual(mm.data_offsets, [1, 10])
        self.assertEqual(source, original)

    def test_chained_visual_audio_padding_stays_packed(self):
        source = array("q", [300, 11, 1, 12, 301, 21, 2, 22, 302])
        visual = MultimodalInputs(mm_items=[mm_item(Modality.IMAGE, 1001, [])])
        audio = MultimodalInputs(mm_items=[mm_item(Modality.AUDIO, 1002, [])])
        intermediate = MultiModalityDataPaddingPatternTokenPairs(
            [(11, 12)]
        ).pad_input_tokens(source, visual)
        self.assertIsInstance(intermediate, array)
        actual = MultiModalityDataPaddingPatternTokenPairs([(21, 22)]).pad_input_tokens(
            intermediate, audio
        )
        self.assertEqual(
            actual, array("q", [300, 11, 1001, 12, 301, 21, 1002, 22, 302])
        )
        self.assertEqual(
            intermediate, array("q", [300, 11, 1001, 12, 301, 21, 2, 22, 302])
        )
        self.assertEqual(source, array("q", [300, 11, 1, 12, 301, 21, 2, 22, 302]))

    def test_empty_unmatched_and_insufficient_item_paths_stay_packed(self):
        pattern = MultiModalityDataPaddingPatternTokenPairs([(11, 12)])
        for values, expected, offsets in (
            ([], [], []),
            ([300, 301], [300, 301], []),
            ([11, 1], [11, 1], []),
            ([11, 1, 12, 11, 2, 12], [11, 1001, 12, 11, 1001, 12], [0, 3]),
        ):
            with self.subTest(values=values):
                mm = MultimodalInputs(mm_items=[mm_item(Modality.IMAGE, 1001, [])])
                source = array("q", values)
                actual = pattern.pad_input_tokens(source, mm)
                self.assertEqual(actual, array("q", expected))
                self.assertEqual(mm.data_offsets, offsets)
                self.assertEqual(source, array("q", values))


class TestModelPaddingContract(CustomTestCase):
    def test_invalid_input_is_rejected_by_shared_helper(self):
        source = array("d", [300, 301])
        mm = MultimodalInputs(
            mm_items=[mm_item(Modality.IMAGE, 1001, [(0, 0)])], im_token_id=1
        )
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        callback = Mock(wraps=pattern.pad_input_tokens)
        with self.assertRaisesRegex(AssertionError, "input_ids must be array"):
            pad_mm_input_ids(source, mm, callback)
        callback.assert_called_once_with(source, mm)
        self.assertEqual(source, array("d", [300, 301]))

    def test_model_cannot_return_a_list_or_wrong_array_format(self):
        source = array("q", [300, 301])
        mm = MultimodalInputs(mm_items=[])
        for output in (None, [], [300, 301], array("i", [300, 301]), array("Q")):
            with self.subTest(output=output):
                callback = Mock(return_value=output)
                with self.assertRaisesRegex(
                    AssertionError, "pad_input_ids result must be array"
                ):
                    pad_mm_input_ids(source, mm, callback)
                callback.assert_called_once_with(source, mm)

    def test_scheduler_owns_result_even_when_model_retains_or_returns_input(self):
        source = array("q", [300, 301])
        # No-op models return input_ids; NanoNemotron can also retain its
        # newly padded output in pre_chunked_input_ids metadata.
        for model_output in (source, array("q", [300, 1001])):
            with self.subTest(model_output=model_output):
                original = model_output[:]
                callback = Mock(return_value=model_output)
                actual = pad_mm_input_ids(
                    source, MultimodalInputs(mm_items=[]), callback
                )
                self.assertIsNot(actual, model_output)
                self.assertEqual(actual, original)
                actual[0] = 999
                self.assertEqual(model_output, original)
                self.assertEqual(source, array("q", [300, 301]))

    def test_tokenizer_ipc_and_request_padding_keep_native_arrays(self):
        import zmq

        from sglang.srt.managers.io_struct import (
            EmbeddingReqInput,
            GenerateReqInput,
            sock_recv,
            sock_send,
        )
        from sglang.srt.managers.tokenizer_manager import TokenizerManager
        from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

        values = [300, 301, 302, 303]
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        for request_type, use_pickle in product(
            (GenerateReqInput, EmbeddingReqInput), (False, True)
        ):
            with (
                self.subTest(request_type=request_type.__name__, pickle=use_pickle),
                patch("sglang.srt.managers.io_struct._USE_PICKLE_IPC", use_pickle),
                zmq.Context() as context,
                context.socket(zmq.PAIR) as sender,
                context.socket(zmq.PAIR) as receiver,
            ):
                sender.setsockopt(zmq.LINGER, 0)
                receiver.setsockopt(zmq.LINGER, 0)
                sender.setsockopt(zmq.SNDTIMEO, 1000)
                receiver.setsockopt(zmq.RCVTIMEO, 1000)
                sender.bind("inproc://packed-input")
                receiver.connect("inproc://packed-input")
                request = request_type(
                    rid="packed-input", input_ids=values, sampling_params={}
                )
                if isinstance(request, GenerateReqInput):
                    request.bootstrap_room = 1
                request.normalize_batch_and_arguments()
                manager = SimpleNamespace(
                    preferred_sampling_params=None,
                    sampling_params_class=SamplingParams,
                    tokenizer=None,
                    model_config=SimpleNamespace(vocab_size=1 << 20),
                    rid_to_state={
                        request.rid: SimpleNamespace(time_stats=APIServerReqTimeStats())
                    },
                )
                received = TokenizerManager._create_tokenized_object(
                    manager, request, None, values
                )
                self.assertEqual(received.input_ids, array("q", values))
                received.wrap_pickle_fields()
                # The controller/router may decode and re-encode before the
                # scheduler receives the tokenized request.
                for _ in range(2):
                    sock_send(sender, received)
                    received = sock_recv(receiver)
                    self.assertIsInstance(received.input_ids, array)
                    self.assertEqual(received.input_ids.typecode, "q")
                    self.assertEqual(received.input_ids.itemsize, 8)
                received.unwrap_pickle_fields()
                req = Req(
                    received.rid,
                    received.input_text,
                    received.input_ids,
                    received.sampling_params,
                )
                mm = MultimodalInputs(
                    mm_items=[mm_item(Modality.IMAGE, 1001, [(1, 2)])], im_token_id=1
                )
                callback = Mock(wraps=pattern.pad_input_tokens)
                req.origin_input_ids = pad_mm_input_ids(
                    req.origin_input_ids, mm, callback
                )
                self.assertIs(callback.call_args.args[0], received.input_ids)
                self.assertEqual(
                    req.origin_input_ids, array("q", [300, 1001, 1001, 303])
                )
                self.assertEqual(req.origin_input_ids_unpadded, array("q", values))
                self.assertEqual(values, [300, 301, 302, 303])


if __name__ == "__main__":
    unittest.main()
