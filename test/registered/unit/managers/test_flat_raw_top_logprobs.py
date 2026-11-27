"""Unit tests for the flat raw top logprob response format (prompt and output),
and for an input candidate selector's weights from the forward's output to the
response."""

import asyncio
import base64
import itertools
import json
import pickle
import unittest
from array import array
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt import runtime_context as rc
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.incremental_detokenize import (
    DetokenizeHeader,
    DetokenizeStateTable,
)
from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    BatchTokenIDOutput,
    GenerateReqInput,
    RequestOutputStats,
    build_flat_top_logprobs_arrays,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.multi_tokenizer_mixin import (
    _send_output_to_tokenizer_workers,
)
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.scheduler_components.output_streamer import (
    OutputSink,
    _GenerationStreamAccumulator,
)
from sglang.srt.managers.tokenizer_manager import (
    ReqState,
    TokenizerManager,
    _build_flat_top_logprobs_fields,
    _build_flat_top_logprobs_fields_from_arrays,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.json_response import dumps_json

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

# Synthetic per-position top-k rows (k=2). The leading None mirrors the first
# prompt position, which has no top logprobs.
_VAL_ROWS = [None, [-0.1, -2.5], [-0.3, -1.5], [-0.05, -4.0]]
_IDX_ROWS = [None, [11, 22], [33, 44], [55, 66]]

# Float32-exact values for scheduler-flat equivalence tests: the scheduler
# ships float32 arrays, so equality against the python-float rows needs values
# that survive the float64 -> float32 round trip (production logprobs do, being
# computed in float32).
_EXACT_VAL_ROWS = [None, [-0.5, -2.5], [-0.25, -1.5], [-0.125, -4.0]]


class _TokenizerManagerStub:
    """Borrow the real logprob meta_info methods without a full manager."""

    convert_logprob_style = TokenizerManager.convert_logprob_style
    add_logprob_to_meta_info = TokenizerManager.add_logprob_to_meta_info
    detokenize_logprob_tokens = TokenizerManager.detokenize_logprob_tokens
    detokenize_top_logprobs_tokens = TokenizerManager.detokenize_top_logprobs_tokens


def _make_state(**obj_kwargs) -> ReqState:
    obj = GenerateReqInput(text="hello", **obj_kwargs)
    obj.normalize_batch_and_arguments()
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _add_logprob_meta_info(state: ReqState, top_logprobs_num: int = 2) -> dict:
    meta_info = {}
    _TokenizerManagerStub().add_logprob_to_meta_info(
        meta_info,
        state,
        top_logprobs_num=top_logprobs_num,
        token_ids_logprob=None,
        return_text_in_logprobs=False,
        finished=False,
    )
    return meta_info


class TestFlatRawTopLogprobsValidation(CustomTestCase):
    def test_flag_defaults_off_and_valid(self):
        for kwargs in (
            {},
            {"return_flat_raw_top_logprobs": True},
        ):
            req = GenerateReqInput(text="hello", **kwargs)
            req.normalize_batch_and_arguments()

    def test_flat_rejects_multi_item_scoring(self):
        req = GenerateReqInput(
            text="a<sep>b",
            return_flat_raw_top_logprobs=True,
            multi_item_delimiter_indices=[1],
        )
        with self.assertRaisesRegex(ValueError, "multi-item"):
            req.normalize_batch_and_arguments()

    def test_flag_propagates_to_batch_items(self):
        req = GenerateReqInput(
            text=["a", "b"],
            return_flat_raw_top_logprobs=True,
        )
        req.normalize_batch_and_arguments()
        for i in range(2):
            self.assertTrue(req[i].return_flat_raw_top_logprobs)

    def test_b64_requires_flat_flag(self):
        req = GenerateReqInput(text="hello", return_flat_raw_top_logprobs_b64=True)
        with self.assertRaisesRegex(ValueError, "return_flat_raw_top_logprobs"):
            req.normalize_batch_and_arguments()

    def test_b64_flag_propagates_to_batch_items(self):
        req = GenerateReqInput(
            text=["a", "b"],
            return_flat_raw_top_logprobs=True,
            return_flat_raw_top_logprobs_b64=True,
        )
        req.normalize_batch_and_arguments()
        for i in range(2):
            self.assertTrue(req[i].return_flat_raw_top_logprobs_b64)

    def test_output_flag_enables_b64_and_propagates_to_batch_items(self):
        req = GenerateReqInput(
            text=["a", "b"],
            return_flat_raw_output_top_logprobs=True,
            return_flat_raw_top_logprobs_b64=True,
        )
        req.normalize_batch_and_arguments()
        for i in range(2):
            self.assertTrue(req[i].return_flat_raw_output_top_logprobs)
            self.assertTrue(req[i].return_flat_raw_top_logprobs_b64)

    def test_output_flag_rejects_stream(self):
        req = GenerateReqInput(
            text="hello", stream=True, return_flat_raw_output_top_logprobs=True
        )
        with self.assertRaisesRegex(ValueError, "stream"):
            req.normalize_batch_and_arguments()


class TestFlatAssembly(CustomTestCase):
    def test_flat_matches_nested_rows(self):
        fields = _build_flat_top_logprobs_fields(
            "input", _VAL_ROWS, _IDX_ROWS, top_logprobs_num=2
        )
        self.assertEqual(fields["input_top_logprobs_shape"], [3, 2])
        self.assertEqual(fields["input_top_logprobs_null_prefix"], 1)
        self.assertEqual(
            fields["input_top_logprobs_val_flat"],
            [v for row in _VAL_ROWS[1:] for v in row],
        )
        self.assertEqual(
            fields["input_top_logprobs_idx_flat"],
            [i for row in _IDX_ROWS[1:] for i in row],
        )
        # Reconstruct the nested rows: covered position i, entry j lives at
        # flat[(i - null_prefix) * k + j].
        rows, k = fields["input_top_logprobs_shape"]
        null_prefix = fields["input_top_logprobs_null_prefix"]
        flat_val = fields["input_top_logprobs_val_flat"]
        self.assertEqual(len(flat_val), rows * k)
        for i in range(null_prefix, null_prefix + rows):
            start = (i - null_prefix) * k
            self.assertEqual(flat_val[start : start + k], _VAL_ROWS[i])

    def test_b64_roundtrip(self):
        fields = _build_flat_top_logprobs_fields(
            "input", _VAL_ROWS, _IDX_ROWS, top_logprobs_num=2, return_b64=True
        )
        self.assertEqual(fields["input_top_logprobs_shape"], [3, 2])
        self.assertEqual(fields["input_top_logprobs_null_prefix"], 1)
        self.assertEqual(fields["input_top_logprobs_val_flat_b64_dtype"], "float32")
        self.assertEqual(fields["input_top_logprobs_idx_flat_b64_dtype"], "int32")
        # shape is the literal array shape, so the b64 buffer reshapes with it.
        val = np.frombuffer(
            base64.b64decode(fields["input_top_logprobs_val_flat_b64"]),
            dtype=np.dtype(fields["input_top_logprobs_val_flat_b64_dtype"]),
        ).reshape(fields["input_top_logprobs_shape"])
        idx = np.frombuffer(
            base64.b64decode(fields["input_top_logprobs_idx_flat_b64"]),
            dtype=np.dtype(fields["input_top_logprobs_idx_flat_b64_dtype"]),
        ).reshape(fields["input_top_logprobs_shape"])
        np.testing.assert_array_equal(val, np.asarray(_VAL_ROWS[1:], dtype=np.float32))
        np.testing.assert_array_equal(idx, np.asarray(_IDX_ROWS[1:], dtype=np.int32))

    def test_all_null_rows(self):
        fields = _build_flat_top_logprobs_fields(
            "input", [None], [None], top_logprobs_num=2
        )
        self.assertEqual(fields["input_top_logprobs_shape"], [0, 2])
        self.assertEqual(fields["input_top_logprobs_null_prefix"], 1)
        self.assertEqual(fields["input_top_logprobs_val_flat"], [])
        self.assertEqual(fields["input_top_logprobs_idx_flat"], [])

    def test_rejects_null_row_after_prefix(self):
        with self.assertRaisesRegex(ValueError, "leading prefix"):
            _build_flat_top_logprobs_fields(
                "input",
                [None, [-0.1, -2.5], None, [-0.3, -1.5]],
                [None, [11, 22], None, [33, 44]],
                top_logprobs_num=2,
            )

    def test_rejects_ragged_rows(self):
        with self.assertRaisesRegex(ValueError, "rectangular"):
            _build_flat_top_logprobs_fields(
                "input",
                [None, [-0.1, -2.5], [-0.3]],
                [None, [11, 22], [33]],
                top_logprobs_num=2,
            )


class TestAddLogprobToMetaInfo(CustomTestCase):
    def _extend_input_top(self, state: ReqState, val_rows, idx_rows):
        state.input_top_logprobs_val.extend(val_rows)
        state.input_top_logprobs_idx.extend(idx_rows)

    def test_nested_path_unchanged_when_flags_unset(self):
        state = _make_state(return_logprob=True, top_logprobs_num=2)
        self._extend_input_top(state, _VAL_ROWS, _IDX_ROWS)
        meta_info = _add_logprob_meta_info(state)
        self.assertEqual(
            meta_info["input_top_logprobs"],
            [
                None if row is None else [(v, i, None) for v, i in zip(row, idx_row)]
                for row, idx_row in zip(_VAL_ROWS, _IDX_ROWS)
            ],
        )
        self.assertIn("output_top_logprobs", meta_info)
        for key in meta_info:
            self.assertNotIn("_flat", key)
            self.assertNotIn("weights", key)
        self.assertNotIn("input_top_logprobs_shape", meta_info)
        self.assertNotIn("input_top_logprobs_null_prefix", meta_info)

    def test_flat_path_replaces_nested_input(self):
        state = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=True,
        )
        self._extend_input_top(state, _VAL_ROWS, _IDX_ROWS)
        meta_info = _add_logprob_meta_info(state)
        self.assertNotIn("input_top_logprobs", meta_info)
        self.assertIn("output_top_logprobs", meta_info)
        self.assertEqual(meta_info["input_top_logprobs_shape"], [3, 2])
        self.assertEqual(meta_info["input_top_logprobs_null_prefix"], 1)
        self.assertEqual(
            meta_info["input_top_logprobs_val_flat"],
            [v for row in _VAL_ROWS[1:] for v in row],
        )

    def test_unrepresentable_rows_fall_back_to_nested(self):
        # The shared batch-output loop must not raise on unrepresentable
        # rows; the request degrades to the nested format.
        state = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=True,
        )
        val_rows = [_VAL_ROWS[1], None, _VAL_ROWS[2]]
        idx_rows = [_IDX_ROWS[1], None, _IDX_ROWS[2]]
        self._extend_input_top(state, val_rows, idx_rows)
        meta_info = _add_logprob_meta_info(state)
        self.assertIn("input_top_logprobs", meta_info)
        self.assertNotIn("input_top_logprobs_val_flat", meta_info)
        self.assertNotIn("input_top_logprobs_shape", meta_info)
        self.assertEqual(len(meta_info["input_top_logprobs"]), 3)

    def test_chunked_accumulation_matches_one_shot(self):
        # One shot.
        one_shot = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=True,
        )
        self._extend_input_top(one_shot, _VAL_ROWS, _IDX_ROWS)
        expected = _add_logprob_meta_info(one_shot)

        # Rows arriving across two chunks, with meta_info assembled after each
        # (as happens for streaming requests).
        chunked = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=True,
        )
        self._extend_input_top(chunked, _VAL_ROWS[:2], _IDX_ROWS[:2])
        _add_logprob_meta_info(chunked)
        self._extend_input_top(chunked, _VAL_ROWS[2:], _IDX_ROWS[2:])
        got = _add_logprob_meta_info(chunked)

        flat_keys = [
            "input_top_logprobs_val_flat",
            "input_top_logprobs_idx_flat",
            "input_top_logprobs_shape",
            "input_top_logprobs_null_prefix",
        ]
        for key in flat_keys:
            self.assertEqual(got[key], expected[key])

        # No new rows -> the encoded payload is reused, not rebuilt.
        again = _add_logprob_meta_info(chunked)
        self.assertIs(
            again["input_top_logprobs_val_flat"],
            got["input_top_logprobs_val_flat"],
        )


class TestB64MetaInfo(CustomTestCase):
    def test_b64_fields_replace_flat_and_cache_reused(self):
        state = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=True,
            return_flat_raw_top_logprobs_b64=True,
        )
        state.input_top_logprobs_val.extend(_VAL_ROWS)
        state.input_top_logprobs_idx.extend(_IDX_ROWS)
        meta_info = _add_logprob_meta_info(state)
        self.assertNotIn("input_top_logprobs", meta_info)
        self.assertNotIn("input_top_logprobs_val_flat", meta_info)
        self.assertIn("input_top_logprobs_val_flat_b64", meta_info)
        self.assertEqual(meta_info["input_top_logprobs_shape"], [3, 2])
        # No new rows -> the encoded payload is reused, not rebuilt.
        again = _add_logprob_meta_info(state)
        self.assertIs(
            again["input_top_logprobs_val_flat_b64"],
            meta_info["input_top_logprobs_val_flat_b64"],
        )


class TestFlatOutputTopLogprobs(CustomTestCase):
    def _state(self, val_rows=None, **obj_kwargs) -> ReqState:
        state = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_output_top_logprobs=True,
            **obj_kwargs,
        )
        state.output_top_logprobs_val.extend(val_rows or _EXACT_VAL_ROWS[1:])
        state.output_top_logprobs_idx.extend(_IDX_ROWS[1:])
        return state

    def _meta_info(self, state: ReqState, finished: bool) -> dict:
        meta_info = {}
        _TokenizerManagerStub().add_logprob_to_meta_info(
            meta_info,
            state,
            top_logprobs_num=2,
            token_ids_logprob=None,
            return_text_in_logprobs=False,
            finished=finished,
        )
        return meta_info

    def test_unfinished_output_carries_no_top_logprobs(self):
        state = self._state()

        meta_info = self._meta_info(state, finished=False)

        self.assertFalse(any(k.startswith("output_top_logprobs") for k in meta_info))
        self.assertEqual(state.output_top_logprobs, [])

    def test_finished_output_replaces_nested(self):
        meta_info = self._meta_info(self._state(), finished=True)

        self.assertNotIn("output_top_logprobs", meta_info)
        self.assertEqual(meta_info["output_top_logprobs_shape"], [3, 2])
        self.assertEqual(meta_info["output_top_logprobs_null_prefix"], 0)
        self.assertEqual(
            meta_info["output_top_logprobs_val_flat"],
            [v for row in _EXACT_VAL_ROWS[1:] for v in row],
        )
        self.assertEqual(
            meta_info["output_top_logprobs_idx_flat"],
            [i for row in _IDX_ROWS[1:] for i in row],
        )
        # The prompt side keeps its own format.
        self.assertIn("input_top_logprobs", meta_info)

    def test_plain_json_flattens_the_original_rows(self):
        val_rows = [None, [-0.5, -2.5], [-0.25, -1.5]]
        state = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_output_top_logprobs=True,
        )
        state.output_top_logprobs_val.extend(val_rows)
        state.output_top_logprobs_idx.extend([None, [11, 22], [33, 44]])

        meta_info = self._meta_info(state, finished=True)

        self.assertEqual(meta_info["output_top_logprobs_shape"], [2, 2])
        self.assertEqual(meta_info["output_top_logprobs_null_prefix"], 1)
        self.assertEqual(meta_info["output_top_logprobs_idx_flat"], [11, 22, 33, 44])
        # Plain JSON reuses the engine's floats; no NumPy round trip.
        self.assertIs(meta_info["output_top_logprobs_val_flat"][0], val_rows[1][0])

    def test_b64_roundtrip_keeps_non_finite_logprobs(self):
        val_rows = [[-0.5, float("-inf")], [-0.25, -1.5], [-0.125, -4.0]]
        state = self._state(val_rows, return_flat_raw_top_logprobs_b64=True)

        meta_info = self._meta_info(state, finished=True)

        self.assertNotIn("output_top_logprobs_val_flat", meta_info)
        shape = meta_info["output_top_logprobs_shape"]
        val = np.frombuffer(
            base64.b64decode(meta_info["output_top_logprobs_val_flat_b64"]),
            dtype=meta_info["output_top_logprobs_val_flat_b64_dtype"],
        ).reshape(shape)
        idx = np.frombuffer(
            base64.b64decode(meta_info["output_top_logprobs_idx_flat_b64"]),
            dtype=meta_info["output_top_logprobs_idx_flat_b64_dtype"],
        ).reshape(shape)
        np.testing.assert_array_equal(val, np.asarray(val_rows, dtype=np.float32))
        np.testing.assert_array_equal(idx, np.asarray(_IDX_ROWS[1:], dtype=np.int32))

    def test_unrepresentable_rows_fall_back_to_nested(self):
        state = self._state([[-0.5, -2.5], [-0.25], [-0.125, -4.0]])

        meta_info = self._meta_info(state, finished=True)

        self.assertNotIn("output_top_logprobs_shape", meta_info)
        self.assertEqual(len(meta_info["output_top_logprobs"]), 3)


def _make_logprob_processor() -> SchedulerLogprobResultProcessor:
    # enable_mis comes from the published exec bag, not from here; see setUp.
    return SchedulerLogprobResultProcessor(
        model_config=SimpleNamespace(
            vocab_size=1_000_000, token_id_vocab_size=1_000_000
        ),
    )


# Per-position rows as computed during prefill: one row per prompt position
# from logprob_start_len on, the last row being the sampling position that
# scheduler-side assembly pops.
_SCHED_VAL_ROWS = [
    [-0.5, -2.5],
    [-0.25, -1.5],
    [-0.125, -4.0],
    [-1.0, -3.0],
    [-2.0, -5.0],
]
_SCHED_IDX_ROWS = [[11, 22], [33, 44], [55, 66], [77, 88], [99, 100]]
# The rows' weights, for a request with an input candidate selector.
_SCHED_WEIGHTS = torch.arange(10, dtype=torch.float32).reshape(5, 2)


class TestSchedulerFlatAssembly(CustomTestCase):
    """Scheduler-side flat assembly in the logprob result processor."""

    def setUp(self):
        super().setUp()
        self._server_args_override = rc.get_context().override_server_args()
        self._server_args_override.install()

    def tearDown(self):
        self._server_args_override.restore()

    def _make_req(self, flat: bool, num_tokens: int = 5) -> Req:
        return Req(
            "r0",
            "",
            array("q", range(1, num_tokens + 1)),
            SamplingParams(),
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=flat,
        )

    def _run_prefill(self, req: Req, chunk_sizes, finish: bool = True) -> None:
        processor = _make_logprob_processor()
        token_logprobs = [row[0] for row in _SCHED_VAL_ROWS]
        pt = 0
        for chunk_idx, size in enumerate(chunk_sizes):
            output = SimpleNamespace(
                input_token_logprobs=tuple(token_logprobs[pt : pt + size]),
                input_top_logprobs_val=[_SCHED_VAL_ROWS[pt : pt + size]],
                input_top_logprobs_idx=[_SCHED_IDX_ROWS[pt : pt + size]],
            )
            if req.logprob.input_candidate_selector is not None:
                output.input_top_logprobs_weights = [[_SCHED_WEIGHTS[pt : pt + size]]]
            processor.add_input_logprob_return_values(
                0,
                req,
                output,
                0,
                size,
                last_prefill_chunk=finish and chunk_idx == len(chunk_sizes) - 1,
            )
            pt += size

    def test_flat_arrays_replace_nested_rows(self):
        flag_off = self._make_req(flat=False)
        self._run_prefill(flag_off, [3, 2])
        flag_on = self._make_req(flat=True)
        self._run_prefill(flag_on, [3, 2])

        # Flag off: nested rows as today, no arrays.
        self.assertIsNone(flag_off.logprob.input_top_logprobs_val_flat)
        self.assertIsNone(flag_off.logprob.input_top_logprobs_flat_null_prefix)
        self.assertEqual(
            flag_off.logprob.input_top_logprobs_val, [None] + _SCHED_VAL_ROWS[:-1]
        )

        # Flag on: arrays carrying the nested rows' content, nested emptied.
        val_arr = flag_on.logprob.input_top_logprobs_val_flat
        idx_arr = flag_on.logprob.input_top_logprobs_idx_flat
        self.assertEqual(val_arr.dtype, np.float32)
        self.assertEqual(idx_arr.dtype, np.int32)
        self.assertEqual(flag_on.logprob.input_top_logprobs_flat_null_prefix, 1)
        np.testing.assert_array_equal(
            val_arr, np.asarray(_SCHED_VAL_ROWS[:-1], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            idx_arr, np.asarray(_SCHED_IDX_ROWS[:-1], dtype=np.int32)
        )
        self.assertEqual(flag_on.logprob.input_top_logprobs_val, [])
        self.assertEqual(flag_on.logprob.input_top_logprobs_idx, [])
        # The non-top logprob results are untouched.
        self.assertEqual(
            flag_on.logprob.input_token_logprobs_val,
            flag_off.logprob.input_token_logprobs_val,
        )
        self.assertEqual(
            flag_on.logprob.input_token_logprobs_idx,
            flag_off.logprob.input_token_logprobs_idx,
        )

    def test_selector_weights_follow_the_rows_across_prefill_chunks(self):
        req = self._make_req(flat=False)
        req.logprob.input_candidate_selector = object()
        # A request retracted mid-prefill starts over without its old rows.
        self._run_prefill(req, [3], finish=False)
        req.reset_for_retract()
        self._run_prefill(req, [3, 2])
        # Like the top logprobs, without the sampling position's row.
        weights = req.logprob.input_top_logprobs_weights
        self.assertEqual(weights.dtype, np.float32)
        np.testing.assert_array_equal(weights, _SCHED_WEIGHTS[:-1].numpy())

    def test_chunked_matches_one_shot(self):
        one_shot = self._make_req(flat=True)
        self._run_prefill(one_shot, [5])
        chunked = self._make_req(flat=True)
        self._run_prefill(chunked, [2, 2, 1])
        np.testing.assert_array_equal(
            one_shot.logprob.input_top_logprobs_val_flat,
            chunked.logprob.input_top_logprobs_val_flat,
        )
        np.testing.assert_array_equal(
            one_shot.logprob.input_top_logprobs_idx_flat,
            chunked.logprob.input_top_logprobs_idx_flat,
        )
        self.assertEqual(
            one_shot.logprob.input_top_logprobs_flat_null_prefix,
            chunked.logprob.input_top_logprobs_flat_null_prefix,
        )

    def test_unrepresentable_rows_fall_back_to_nested(self):
        req = self._make_req(flat=True, num_tokens=3)
        processor = _make_logprob_processor()
        val_rows = [[-0.5, -2.5], [-0.25], [-0.125, -4.0]]
        idx_rows = [[11, 22], [33], [55, 66]]
        output = SimpleNamespace(
            input_token_logprobs=(-0.5, -0.25, -0.125),
            input_top_logprobs_val=[val_rows],
            input_top_logprobs_idx=[idx_rows],
        )
        with self.assertLogs(
            "sglang.srt.managers.scheduler_components.logprob_result_processor",
            level="WARNING",
        ):
            processor.add_input_logprob_return_values(
                0, req, output, 0, 3, last_prefill_chunk=True
            )
        self.assertIsNone(req.logprob.input_top_logprobs_val_flat)
        self.assertIsNone(req.logprob.input_top_logprobs_flat_null_prefix)
        self.assertEqual(req.logprob.input_top_logprobs_val, [None] + val_rows[:-1])
        self.assertEqual(req.logprob.input_top_logprobs_idx, [None] + idx_rows[:-1])


class TestFromArraysMatchesFromRows(CustomTestCase):
    """The tokenizer-manager from-arrays builder must produce the same
    response fields as the rows-based builder."""

    def _both(self, return_b64: bool):
        from_rows = _build_flat_top_logprobs_fields(
            "input",
            _EXACT_VAL_ROWS,
            _IDX_ROWS,
            top_logprobs_num=2,
            return_b64=return_b64,
        )
        val_arr, idx_arr, null_prefix = build_flat_top_logprobs_arrays(
            _EXACT_VAL_ROWS, _IDX_ROWS, top_logprobs_num=2
        )
        from_arrays = _build_flat_top_logprobs_fields_from_arrays(
            "input", val_arr, idx_arr, null_prefix, return_b64=return_b64
        )
        return from_rows, from_arrays

    def test_non_b64(self):
        from_rows, from_arrays = self._both(return_b64=False)
        self.assertEqual(from_rows, from_arrays)

    def test_b64(self):
        from_rows, from_arrays = self._both(return_b64=True)
        self.assertEqual(from_rows, from_arrays)

    def test_all_null_rows(self):
        val_arr, idx_arr, null_prefix = build_flat_top_logprobs_arrays(
            [None], [None], top_logprobs_num=2
        )
        self.assertEqual(val_arr.shape, (0, 2))
        self.assertEqual(null_prefix, 1)
        fields = _build_flat_top_logprobs_fields_from_arrays(
            "input", val_arr, idx_arr, null_prefix
        )
        self.assertEqual(
            fields,
            _build_flat_top_logprobs_fields(
                "input", [None], [None], top_logprobs_num=2
            ),
        )


class TestMetaInfoFromSchedulerArrays(CustomTestCase):
    """add_logprob_to_meta_info consumes scheduler-flat arrays directly."""

    def _rows_meta(self, **state_kwargs) -> dict:
        state = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=True,
            **state_kwargs,
        )
        state.input_top_logprobs_val.extend(_EXACT_VAL_ROWS)
        state.input_top_logprobs_idx.extend(_IDX_ROWS)
        return _add_logprob_meta_info(state)

    def _arrays_state(self, **state_kwargs) -> ReqState:
        state = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=True,
            **state_kwargs,
        )
        # Scheduler-flat requests arrive with empty nested rows and the arrays.
        state.input_top_logprobs_scheduler_flat = build_flat_top_logprobs_arrays(
            _EXACT_VAL_ROWS, _IDX_ROWS, top_logprobs_num=2
        )
        return state

    def test_matches_rows_path_field_for_field(self):
        got = _add_logprob_meta_info(self._arrays_state())
        self.assertEqual(got, self._rows_meta())

    def test_b64_matches_rows_path_field_for_field(self):
        got = _add_logprob_meta_info(
            self._arrays_state(return_flat_raw_top_logprobs_b64=True)
        )
        self.assertEqual(got, self._rows_meta(return_flat_raw_top_logprobs_b64=True))

    def test_fields_cached_across_chunks(self):
        state = self._arrays_state()
        first = _add_logprob_meta_info(state)
        again = _add_logprob_meta_info(state)
        self.assertIs(
            again["input_top_logprobs_val_flat"], first["input_top_logprobs_val_flat"]
        )


class TestTokenizerManagerLogprobs(CustomTestCase):
    def test_output_logprobs_without_input_logprobs(self):
        state = _make_state(return_logprob=True, top_logprobs_num=0)
        recv_obj = SimpleNamespace(
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            output_token_logprobs_val=[[-0.25]],
            output_token_logprobs_idx=[[42]],
        )
        meta_info = {}

        _TokenizerManagerStub().convert_logprob_style(
            meta_info,
            state,
            top_logprobs_num=0,
            token_ids_logprob=None,
            return_text_in_logprobs=False,
            recv_obj=recv_obj,
            recv_obj_index=0,
            finished=False,
        )

        self.assertEqual(meta_info["input_token_logprobs"], [])
        self.assertEqual(meta_info["output_token_logprobs"], [(-0.25, 42, None)])
        self.assertEqual(meta_info["output_token_logprobs_length"], 1)

    def test_top_and_token_ids_logprobs_without_logprob_lists(self):
        """An output without logprob lists, as sent for a request aborted before
        prefill, must not crash a request that asked for top or token id logprobs."""
        state = _make_state(
            return_logprob=True, top_logprobs_num=2, token_ids_logprob=[42]
        )
        recv_obj = SimpleNamespace(
            input_token_logprobs_val=None,
            output_token_logprobs_val=None,
            input_top_logprobs_val=None,
            input_top_logprobs_val_flat=None,
            input_top_logprobs_weights=None,
            output_top_logprobs_val=None,
            input_token_ids_logprobs_val=None,
            output_token_ids_logprobs_val=None,
        )
        meta_info = {}

        _TokenizerManagerStub().convert_logprob_style(
            meta_info,
            state,
            top_logprobs_num=2,
            token_ids_logprob=[42],
            return_text_in_logprobs=False,
            recv_obj=recv_obj,
            recv_obj_index=0,
            finished=True,
        )

        for key in (
            "input_top_logprobs",
            "output_top_logprobs",
            "input_token_ids_logprobs",
            "output_token_ids_logprobs",
        ):
            self.assertEqual(meta_info[key], [])


def _make_batch_token_id_output(**overrides) -> BatchTokenIDOutput:
    """A two-request BatchTokenIDOutput with the required fields stubbed."""
    n = 2
    fields = dict(
        rids=["r0", "r1"],
        finished_reasons=[None] * n,
        output_ids_flat=array("q"),
        output_ids_offsets=[0] * (n + 1),
        reasoning_tokens=[0] * n,
        completion_tokens=[1] * n,
        input_token_logprobs_val=[[], []],
        input_token_logprobs_idx=[[], []],
        output_token_logprobs_val=[[], []],
        output_token_logprobs_idx=[[], []],
        input_top_logprobs_val=[[], []],
        input_top_logprobs_idx=[[], []],
        output_top_logprobs_val=[[], []],
        output_top_logprobs_idx=[[], []],
        input_token_ids_logprobs_val=[[], []],
        input_token_ids_logprobs_idx=[[], []],
        output_token_ids_logprobs_val=[[], []],
        output_token_ids_logprobs_idx=[[], []],
        output_token_entropy_val=None,
        output_token_sampling_mask=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
    )
    fields.update(overrides)
    return BatchTokenIDOutput(**fields)


class TestBatchOutputTransport(CustomTestCase):
    """The flat array fields must survive both IPC transports: pickle
    (SGLANG_USE_PICKLE_IPC, the default) and msgpack (enc/dec hooks)."""

    def _flat_output(self) -> BatchTokenIDOutput:
        val_arr, idx_arr, null_prefix = build_flat_top_logprobs_arrays(
            _EXACT_VAL_ROWS, _IDX_ROWS, top_logprobs_num=2
        )
        return _make_batch_token_id_output(
            input_top_logprobs_val_flat=[None, val_arr],
            input_top_logprobs_idx_flat=[None, idx_arr],
            input_top_logprobs_flat_null_prefix=[None, null_prefix],
            input_top_logprobs_weights=[None, -val_arr],
        )

    def _check_roundtrip(self, decoded, original):
        self.assertIsNone(decoded.input_top_logprobs_val_flat[0])
        self.assertIsNone(decoded.input_top_logprobs_idx_flat[0])
        self.assertIsNone(decoded.input_top_logprobs_flat_null_prefix[0])
        self.assertIsNone(decoded.input_top_logprobs_weights[0])
        for got, sent in (
            (
                decoded.input_top_logprobs_val_flat[1],
                original.input_top_logprobs_val_flat[1],
            ),
            (
                decoded.input_top_logprobs_idx_flat[1],
                original.input_top_logprobs_idx_flat[1],
            ),
            (
                decoded.input_top_logprobs_weights[1],
                original.input_top_logprobs_weights[1],
            ),
        ):
            self.assertIsInstance(got, np.ndarray)
            self.assertEqual(got.dtype, sent.dtype)
            np.testing.assert_array_equal(got, sent)
        self.assertEqual(decoded.input_top_logprobs_flat_null_prefix[1], 1)

    def test_pickle_roundtrip(self):
        output = self._flat_output()
        decoded = pickle.loads(pickle.dumps(output, protocol=pickle.HIGHEST_PROTOCOL))
        self._check_roundtrip(decoded, output)

    def test_msgpack_roundtrip(self):
        output = self._flat_output()
        decoded = msgpack_decode(msgpack_encode(output))
        self._check_roundtrip(decoded, output)


_REQUEST_STATS = RequestOutputStats(
    prompt_tokens=0,
    cached_tokens=0,
    cached_tokens_details=None,
    retraction_count=0,
    image_tokens=0,
    audio_tokens=0,
    video_tokens=0,
    time_stats=None,
)

# What every forward below returns for the position after the prompt: the
# sampled token, its logprob and its top logprobs.
_NEXT_TOKEN = 7
_NEXT_TOKEN_LOGPROB = -0.5
_NEXT_TOKEN_TOP_VAL = [-0.5, -1.25]
_NEXT_TOKEN_TOP_IDX = [7, 8]


def _make_candidate_rows(tag: int, num_rows: int):
    """A request's logprob rows as its prefill computes them, the last being
    the sampling position's: top logprob values, token ids and float32
    [rows, 2] weights, each distinct by request, row and candidate."""
    val = [[-tag - row / 8, -tag - row / 8 - 0.5] for row in range(num_rows)]
    idx = [[100 * tag + row, 100 * tag + row + 50] for row in range(num_rows)]
    # Below 1 and not dyadic: neither a narrower float nor a decimal rounding
    # returns them.
    weights = (64 * tag + torch.arange(1, 2 * num_rows + 1, dtype=torch.float32)) / 1021
    return val, idx, weights.reshape(num_rows, 2)


def _make_logprob_req(rid: str, num_tokens: int, selector=None, **req_kwargs) -> Req:
    req = Req(
        rid,
        "",
        array("q", range(1, num_tokens + 1)),
        SamplingParams(),
        return_logprob=True,
        top_logprobs_num=2,
        **req_kwargs,
    )
    req.logprob.input_candidate_selector = selector
    return req


def _run_prefill_forward(processor, parts, carries_selectors=True) -> None:
    """One prefill forward over `parts`, in batch order: (req, rows, lo, hi)
    puts the request's logprob rows [lo, hi) in the forward, which ends its
    prefill when hi is its last row. A selector request's weights come as one
    host span per logprob chunk; here a chunk ends after its first row. A
    forward that does not carry the selectors answers without them."""
    token_logprobs, top_val, top_idx, weight_spans = [], [], [], []
    has_selector = False
    for req, (val, idx, weights), lo, hi in parts:
        token_logprobs += [row[0] for row in val[lo:hi]]
        top_val.append(val[lo:hi])
        top_idx.append(idx[lo:hi])
        spans = []
        if carries_selectors and req.logprob.input_candidate_selector is not None:
            has_selector = True
            spans = [weights[lo : lo + 1], weights[lo + 1 : hi]]
        weight_spans.append([span for span in spans if len(span)])
    output = LogitsProcessorOutput(
        next_token_logits=None,
        next_token_logprobs=[_NEXT_TOKEN_LOGPROB] * len(parts),
        next_token_top_logprobs_val=[_NEXT_TOKEN_TOP_VAL] * len(parts),
        next_token_top_logprobs_idx=[_NEXT_TOKEN_TOP_IDX] * len(parts),
        input_token_logprobs=tuple(token_logprobs),
        input_top_logprobs_val=top_val,
        input_top_logprobs_idx=top_idx,
        input_top_logprobs_weights=weight_spans if has_selector else None,
    )
    pt = 0
    for i, (req, (val, _, _), lo, hi) in enumerate(parts):
        if hi == len(val):
            req.output_ids.append(_NEXT_TOKEN)
            processor.add_logprob_return_values(
                i, req, pt, [_NEXT_TOKEN] * len(parts), hi - lo, output
            )
        else:
            processor.add_input_logprob_return_values(
                i, req, output, pt, hi - lo, last_prefill_chunk=False
            )
        pt += hi - lo


def _make_output_batch(
    reqs, output_sink: OutputSink = OutputSink.TOKENIZER
) -> BatchTokenIDOutput:
    """The batch output of one scheduler step for `reqs`."""
    accumulator = _GenerationStreamAccumulator(
        return_logprob=True,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=1,
        build_request_stats=lambda req: _REQUEST_STATS,
        output_sink=output_sink,
        is_request_level_customized_info_key=lambda key: False,
        current_weight_version="default",
    )
    for req in reqs:
        accumulator.accept(req=req)
    return accumulator.to_payload(dp_rank=0, is_idle_batch=False)


class _Tokenizer:
    is_fast = True

    def decode(self, ids, **kwargs):
        return "".join(f"{token}." for token in ids)

    def batch_decode(self, ids_list, **kwargs):
        return [self.decode(ids) for ids in ids_list]


def _make_detokenizer() -> DetokenizerManager:
    detokenizer = DetokenizerManager.__new__(DetokenizerManager)
    detokenizer.tokenizer = _Tokenizer()
    detokenizer.vocab_size = None
    detokenizer.detokenize_state_table = DetokenizeStateTable(capacity=16)
    detokenizer.disable_tokenizer_batch_decode = False
    detokenizer.is_tool_call_parser_gpt_oss = False
    detokenizer.tool_close_token_ids = frozenset()
    return detokenizer


_IPC_ROUND_TRIPS = {
    "pickle": lambda obj: pickle.loads(
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    ),
    "msgpack": lambda obj: msgpack_decode(msgpack_encode(obj)),
}


def _receive_batch(states, recv_obj) -> dict:
    """The logprob meta_info of each row of a batch output, as the tokenizer
    manager's batch loop builds it for the request states by rid."""
    meta_infos = {}
    for index, rid in enumerate(recv_obj.rids):
        if rid not in states:
            continue
        meta_infos[rid] = {}
        _TokenizerManagerStub().convert_logprob_style(
            meta_infos[rid],
            states[rid],
            top_logprobs_num=2,
            token_ids_logprob=None,
            return_text_in_logprobs=False,
            recv_obj=recv_obj,
            recv_obj_index=index,
            finished=recv_obj.finished_reasons[index] is not None,
        )
    return meta_infos


def _reject_json_constant(name: str):
    raise ValueError(f"{name} is not JSON")


def _read_response(meta_info: dict):
    """The prompt top logprobs a client reads from the HTTP response of a
    request, in the layout it asked for: float32 [rows, 2] values, int32
    [rows, 2] token ids, and the float32 [rows, 2] weights or None. A JSON
    null reads as NaN."""
    body = json.loads(dumps_json(meta_info), parse_constant=_reject_json_constant)

    def as_array(values, dtype, shape):
        values = [np.nan if value is None else value for value in values]
        return np.array(values, dtype).reshape(shape)

    weights = None
    if "input_top_logprobs" in body:
        rows = body["input_top_logprobs"]
        assert rows[0] is None and None not in rows[1:], rows
        shape = [len(rows) - 1, 2]
        val = [value for row in rows[1:] for value, _, _ in row]
        idx = [token for row in rows[1:] for _, token, _ in row]
        if "input_top_logprobs_weights" in body:
            weights = body["input_top_logprobs_weights"]
            # Null where input_top_logprobs is.
            assert len(weights) == len(rows), (len(weights), len(rows))
            assert weights[0] is None and None not in weights[1:], weights
            weights = [weight for row in weights[1:] for weight in row]
    else:
        shape = body["input_top_logprobs_shape"]
        assert body["input_top_logprobs_null_prefix"] == 1, body
        if "input_top_logprobs_val_flat" in body:
            val = body["input_top_logprobs_val_flat"]
            idx = body["input_top_logprobs_idx_flat"]
            weights = body.get("input_top_logprobs_weights_flat")
            # A list of numbers, not of rows.
            assert weights is None or len(weights) == shape[0] * shape[1], weights
        else:
            val, idx, weights = (
                (
                    np.frombuffer(
                        base64.b64decode(body[f"{key}_b64"]), body[f"{key}_b64_dtype"]
                    )
                    if f"{key}_b64" in body
                    else None
                )
                for key in (
                    "input_top_logprobs_val_flat",
                    "input_top_logprobs_idx_flat",
                    "input_top_logprobs_weights_flat",
                )
            )
    if weights is not None:
        weights = as_array(weights, np.float32, shape)
    return as_array(val, np.float32, shape), as_array(idx, np.int32, shape), weights


def _canonical_nan(values: np.ndarray) -> np.ndarray:
    """`values` with every NaN replaced by the one a JSON null reads as."""
    values = values.copy()
    values[np.isnan(values)] = np.nan
    return values


_LAYOUTS = {
    "nested": {},
    "flat": dict(return_flat_raw_top_logprobs=True),
    "b64": dict(
        return_flat_raw_top_logprobs=True, return_flat_raw_top_logprobs_b64=True
    ),
}


class _WithServerArgs(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(rc.get_context().override_server_args())

    def assertSameBits(self, got: np.ndarray, want: np.ndarray, msg=None):
        self.assertIsInstance(got, np.ndarray, msg)
        self.assertEqual(got.dtype, want.dtype, msg)
        self.assertEqual(got.shape, want.shape, msg)
        self.assertEqual(got.tobytes(), want.tobytes(), msg)


class TestCandidateWeightsSchedulerAssembly(_WithServerArgs):
    """The weights of a request's rows, from the forwards' outputs to the
    request, in the logprob result processor."""

    def test_requests_sharing_forwards_get_their_own_rows(self):
        processor = _make_logprob_processor()
        plain = _make_logprob_req("plain", 4)
        whole = _make_logprob_req("whole", 5, selector=object())
        chunked = _make_logprob_req(
            "chunked", 7, selector=object(), return_flat_raw_top_logprobs=True
        )
        late = _make_logprob_req("late", 3, selector=object())
        rows = {
            req.rid: _make_candidate_rows(tag, len(req.origin_input_ids))
            for tag, req in enumerate((plain, whole, chunked, late), start=1)
        }
        # The chunked request moves from batch index 2 to 0.
        _run_prefill_forward(
            processor,
            [
                (plain, rows["plain"], 0, 4),
                (whole, rows["whole"], 0, 5),
                (chunked, rows["chunked"], 0, 3),
            ],
        )
        self.assertIsNone(chunked.logprob.input_top_logprobs_weights)
        _run_prefill_forward(
            processor,
            [(chunked, rows["chunked"], 3, 7), (late, rows["late"], 0, 3)],
        )

        for req in (whole, chunked, late):
            val, idx, weights = rows[req.rid]
            with self.subTest(rid=req.rid):
                # Row r of the weights belongs to position r + 1, as the top
                # logprobs do: the sampling position's row is gone.
                self.assertSameBits(
                    req.logprob.input_top_logprobs_weights, weights[:-1].numpy()
                )
                if req.return_flat_raw_top_logprobs:
                    self.assertSameBits(
                        req.logprob.input_top_logprobs_idx_flat,
                        np.asarray(idx[:-1], np.int32),
                    )
                    self.assertSameBits(
                        req.logprob.input_top_logprobs_val_flat,
                        np.asarray(val[:-1], np.float32),
                    )
                else:
                    self.assertEqual(
                        req.logprob.input_top_logprobs_idx, [None] + idx[:-1]
                    )
                    self.assertEqual(
                        req.logprob.input_top_logprobs_val, [None] + val[:-1]
                    )
        self.assertIsNone(plain.logprob.input_top_logprobs_weights)
        # The host spans of the forwards are released.
        for req in (plain, whole, chunked, late):
            self.assertIsNone(req.temp_input_top_logprobs_weights, req.rid)

    def test_weights_cover_the_logprob_window(self):
        # (prompt tokens, logprob_start_len): a prompt of one token has the
        # sampling position's row alone, so its weights have no row.
        for num_tokens, start in ((1, 0), (2, 0), (6, 0), (6, 4), (6, 5)):
            with self.subTest(num_tokens=num_tokens, logprob_start_len=start):
                req = _make_logprob_req("r0", num_tokens, selector=object())
                req.logprob_start_len = start
                val, idx, weights = _make_candidate_rows(1, num_tokens - start)
                _run_prefill_forward(
                    _make_logprob_processor(), [(req, (val, idx, weights), 0, len(val))]
                )
                self.assertSameBits(
                    req.logprob.input_top_logprobs_weights, weights[:-1].numpy()
                )
                self.assertEqual(
                    req.logprob.input_top_logprobs_weights.shape,
                    (num_tokens - start - 1, 2),
                )
                self.assertEqual(req.logprob.input_top_logprobs_val, [None] + val[:-1])

    def test_requests_without_weight_rows_get_no_weights(self):
        """A request scored at chosen positions, whose forward selects no
        candidates, and a selector request without input logprobs beside one
        that has them."""
        processor = _make_logprob_processor()
        scored = Req(
            "scored",
            "",
            array("q", range(1, 7)),
            SamplingParams(max_new_tokens=0),
            return_logprob=True,
            top_logprobs_num=2,
            token_indices_to_pool=[1, 3, 4],
        )
        scored_rows = _make_candidate_rows(1, 3)
        _run_prefill_forward(processor, [(scored, scored_rows, 0, 3)])
        # Every scoring position keeps its row: none is the sampling position.
        self.assertEqual(scored.logprob.input_top_logprobs_val, scored_rows[0])
        self.assertIsNone(scored.logprob.input_top_logprobs_weights)

        no_rows = _make_logprob_req("no-rows", 3, selector=object())
        no_rows.logprob_start_len = -1
        selected = _make_logprob_req("selected", 4, selector=object())
        selected_rows = _make_candidate_rows(3, 4)
        _run_prefill_forward(
            processor,
            [
                (no_rows, _make_candidate_rows(2, 0), 0, 0),
                (selected, selected_rows, 0, 4),
            ],
        )
        self.assertEqual(no_rows.logprob.input_top_logprobs_val, [])
        self.assertIsNone(no_rows.logprob.input_top_logprobs_weights)
        self.assertSameBits(
            selected.logprob.input_top_logprobs_weights, selected_rows[2][:-1].numpy()
        )

    def test_weights_short_of_the_top_logprob_rows_fail_the_assembly(self):
        """A weight span lost on the way, as at a logprob chunk boundary, would
        leave every later row on the wrong position."""
        req = _make_logprob_req("r0", 4, selector=object())
        val, idx, weights = _make_candidate_rows(1, 4)
        with self.assertRaises(AssertionError):
            _run_prefill_forward(
                _make_logprob_processor(), [(req, (val, idx, weights[1:]), 0, 4)]
            )

    def test_a_forward_that_lost_the_selectors_fails_the_assembly(self):
        """Its rows are the top-k: the request must not take them for its
        selector's candidates."""
        for hi in (2, 4):
            with self.subTest(last_prefill_chunk=hi == 4):
                req = _make_logprob_req("r0", 4, selector=object())
                with self.assertRaisesRegex(RuntimeError, "no candidate weights"):
                    _run_prefill_forward(
                        _make_logprob_processor(),
                        [(req, _make_candidate_rows(1, 4), 0, hi)],
                        carries_selectors=False,
                    )

    def test_retraction_after_the_prefill_keeps_the_weights(self):
        processor = _make_logprob_processor()
        req = _make_logprob_req("r0", 4, selector=object())
        rows = _make_candidate_rows(1, 4)
        _run_prefill_forward(processor, [(req, rows, 0, 4)])
        weights = req.logprob.input_top_logprobs_weights
        self.assertSameBits(weights, rows[2][:-1].numpy())

        # The request prefills again; its input logprobs are not assembled a
        # second time and the new forward's spans are not held.
        req.reset_for_retract()
        _run_prefill_forward(processor, [(req, _make_candidate_rows(2, 4), 0, 4)])
        self.assertIs(req.logprob.input_top_logprobs_weights, weights)
        self.assertSameBits(weights, rows[2][:-1].numpy())
        self.assertEqual(req.temp_input_top_logprobs_weights, [])


class TestCandidateWeightsOutputBatch(_WithServerArgs):
    """The weights list of a scheduler batch output: one entry per request of
    the batch, None for a request that sends no weights."""

    def _make_mixed_batch(self, output_sink, flat: bool = False) -> SimpleNamespace:
        """A batch output of two selector requests between a selector request
        whose input logprobs went out with an `earlier` batch and a request
        without logprobs, with a plain top-k request at the end."""
        processor = _make_logprob_processor()
        selector_kwargs = dict(selector=object(), return_flat_raw_top_logprobs=flat)
        decoding = _make_logprob_req(
            "decoding", 3, http_worker_ipc="worker-a", **selector_kwargs
        )
        first = _make_logprob_req(
            "first", 5, http_worker_ipc="worker-b", **selector_kwargs
        )
        silent = Req(
            "silent",
            "",
            array("q", [1, 2]),
            SamplingParams(),
            http_worker_ipc="worker-a",
        )
        second = _make_logprob_req(
            "second", 4, http_worker_ipc="worker-b", **selector_kwargs
        )
        plain = _make_logprob_req("plain", 4, http_worker_ipc="worker-a")
        rows = {
            req.rid: _make_candidate_rows(tag, len(req.origin_input_ids))
            for tag, req in enumerate((decoding, first, second, plain), start=1)
        }
        _run_prefill_forward(processor, [(decoding, rows["decoding"], 0, 3)])
        earlier = _make_output_batch([decoding], output_sink)
        _run_prefill_forward(
            processor,
            [
                (first, rows["first"], 0, 5),
                (second, rows["second"], 0, 4),
                (plain, rows["plain"], 0, 4),
            ],
        )
        reqs = [decoding, first, silent, second, plain]
        return SimpleNamespace(
            reqs=reqs,
            earlier=earlier,
            output=_make_output_batch(reqs, output_sink),
            weights={
                "decoding": None,
                "first": rows["first"][2][:-1].numpy(),
                "silent": None,
                "second": rows["second"][2][:-1].numpy(),
                "plain": None,
            },
        )

    def assertRowsCarryTheirWeights(self, output, weights_by_rid):
        self.assertEqual(len(output.input_top_logprobs_weights), len(output.rids))
        for rid, got in zip(output.rids, output.input_top_logprobs_weights):
            if weights_by_rid[rid] is None:
                self.assertIsNone(got, rid)
            else:
                self.assertSameBits(got, weights_by_rid[rid], rid)

    def test_weights_stay_in_step_with_rids(self):
        for output_sink, flat in itertools.product(
            (OutputSink.TOKENIZER, OutputSink.DETOKENIZER), (False, True)
        ):
            with self.subTest(output_sink=output_sink, flat=flat):
                batch = self._make_mixed_batch(output_sink, flat)
                self.assertEqual(batch.output.rids, [req.rid for req in batch.reqs])
                self.assertRowsCarryTheirWeights(batch.output, batch.weights)
                # The earlier batch took the first request's weights.
                self.assertEqual(batch.earlier.rids, ["decoding"])
                self.assertSameBits(
                    batch.earlier.input_top_logprobs_weights[0],
                    batch.reqs[0].logprob.input_top_logprobs_weights,
                )

    def test_weights_out_of_step_with_rids_fail_the_batch(self):
        """A list shorter than rids hands a request the weights of a later one:
        no request reads any from such a batch."""
        processor = _make_logprob_processor()
        plain = _make_logprob_req("plain", 4)
        selected = _make_logprob_req("selected", 4, selector=object())
        _run_prefill_forward(
            processor,
            [
                (plain, _make_candidate_rows(1, 4), 0, 4),
                (selected, _make_candidate_rows(2, 4), 0, 4),
            ],
        )
        output = _make_output_batch([plain, selected])
        self.assertIsNone(output.input_top_logprobs_weights[0])
        # As if the plain request had left no entry.
        del output.input_top_logprobs_weights[0]
        states = {
            rid: _make_state(return_logprob=True, top_logprobs_num=2)
            for rid in output.rids
        }
        with self.assertRaises(AssertionError):
            _receive_batch(states, output)
        for rid, state in states.items():
            self.assertIsNone(state.input_top_logprobs_weights, rid)

    def test_batch_without_selector_weights_has_no_list(self):
        processor = _make_logprob_processor()
        plain = _make_logprob_req("plain", 4)
        _run_prefill_forward(processor, [(plain, _make_candidate_rows(1, 4), 0, 4)])
        output = _make_output_batch([plain])
        self.assertEqual(output.rids, ["plain"])
        self.assertIsNone(output.input_top_logprobs_weights)

    def test_worker_splits_keep_each_rows_weights(self):
        """--tokenizer-worker-num > 1: every tokenizer worker is sent the rows
        of its own requests, of the scheduler's batch or the detokenizer's."""
        token_ids = self._make_mixed_batch(OutputSink.TOKENIZER)
        detokenized = self._make_mixed_batch(OutputSink.DETOKENIZER)
        detokenizer = _make_detokenizer()
        detokenizer.handle_batch_token_id_out(detokenized.earlier)
        detokenized.output = detokenizer.handle_batch_token_id_out(detokenized.output)
        self.assertIsInstance(detokenized.output, BatchStrOutput)
        for batch, batching in itertools.product(
            (token_ids, detokenized), (True, False)
        ):
            ipcs = batch.output.http_worker_ipcs
            self.assertEqual(ipcs, [req.http_worker_ipc for req in batch.reqs])
            sent = []
            mapping = SimpleNamespace(
                send_output=lambda ipc, obj, *, is_tokenizer: sent.append((ipc, obj))
            )
            with (
                self.subTest(carrier=type(batch.output).__name__, batching=batching),
                envs.SGLANG_BATCH_TOKENIZER_OUTPUTS.override(batching),
            ):
                _send_output_to_tokenizer_workers(batch.output, ipcs, mapping)
                rids_by_worker = {}
                for ipc, obj in sent:
                    rids_by_worker.setdefault(ipc, []).extend(obj.rids)
                    self.assertRowsCarryTheirWeights(obj, batch.weights)
                self.assertEqual(
                    rids_by_worker,
                    {
                        "worker-a": ["decoding", "silent", "plain"],
                        "worker-b": ["first", "second"],
                    },
                )
                self.assertEqual(len(sent), 2 if batching else len(batch.reqs))


class TestCandidateWeights(_WithServerArgs):
    """Weights of an input candidate selector's candidates: one float32
    [rows, k] array per request, from the scheduler's output to the response."""

    def test_response_layouts(self):
        weights = np.arange(6, dtype=np.float32).reshape(3, 2) / 8
        recv_obj = _make_detokenizer().handle_batch_token_id_out(
            _make_batch_token_id_output(
                rids=["r0"],
                detokenize_headers=[
                    DetokenizeHeader(
                        prompt_tail_ids=array("q"),
                        skip_special_tokens=True,
                        spaces_between_special_tokens=True,
                        no_stop_trim=False,
                        http_worker_ipc=None,
                    )
                ],
                finished_reasons=[None],
                output_ids_offsets=[0, 0],
                reasoning_tokens=[0],
                completion_tokens=[0],
                input_top_logprobs_val=[_EXACT_VAL_ROWS],
                input_top_logprobs_idx=[_IDX_ROWS],
                input_top_logprobs_weights=[weights],
            )
        )
        self.assertEqual(recv_obj.rids, ["r0"])
        flat = dict(return_flat_raw_top_logprobs=True)
        for obj_kwargs, want in (
            ({}, {"input_top_logprobs_weights": [None] + weights.tolist()}),
            (flat, {"input_top_logprobs_weights_flat": weights.reshape(-1).tolist()}),
            (
                dict(flat, return_flat_raw_top_logprobs_b64=True),
                {
                    "input_top_logprobs_weights_flat_b64": base64.b64encode(
                        weights.tobytes()
                    ).decode(),
                    "input_top_logprobs_weights_flat_b64_dtype": "float32",
                },
            ),
        ):
            state = _make_state(return_logprob=True, top_logprobs_num=2, **obj_kwargs)
            meta_info = {}
            _TokenizerManagerStub().convert_logprob_style(
                meta_info,
                state,
                top_logprobs_num=2,
                token_ids_logprob=None,
                return_text_in_logprobs=False,
                recv_obj=recv_obj,
                recv_obj_index=0,
                finished=True,
            )
            self.assertEqual(
                {key: meta_info[key] for key in meta_info if "weights" in key}, want
            )

    def test_each_request_reads_its_own_weights_until_its_last_response(self):
        """A selector request decodes on while two more finish their prefill:
        the response of each carries its own candidates and weights, whatever
        its layout, the output sink and the IPC encoding."""
        for output_sink, encoding in itertools.product(
            (OutputSink.TOKENIZER, OutputSink.DETOKENIZER), _IPC_ROUND_TRIPS
        ):
            with self.subTest(output_sink=output_sink, encoding=encoding):
                self._check_stream(output_sink, _IPC_ROUND_TRIPS[encoding])

    def _check_stream(self, output_sink, round_trip):
        processor = _make_logprob_processor()
        layouts = {
            "plain": "nested",
            "nested": "nested",
            "b64": "b64",
            "flat": "flat",
            "last": "nested",
        }
        reqs = {
            "plain": _make_logprob_req("plain", 4),
            "nested": _make_logprob_req("nested", 5, selector=object()),
            "b64": _make_logprob_req(
                "b64", 7, selector=object(), return_flat_raw_top_logprobs=True
            ),
            "flat": _make_logprob_req(
                "flat", 3, selector=object(), return_flat_raw_top_logprobs=True
            ),
            "last": _make_logprob_req("last", 4),
        }
        silent = Req("silent", "", array("q", [1, 2]), SamplingParams())
        rows = {
            rid: _make_candidate_rows(tag, len(req.origin_input_ids))
            for tag, (rid, req) in enumerate(reqs.items(), start=1)
        }
        states = {
            rid: _make_state(
                return_logprob=True, top_logprobs_num=2, **_LAYOUTS[layout]
            )
            for rid, layout in layouts.items()
        }
        detokenizer = _make_detokenizer()

        def respond(batch_reqs) -> dict:
            output = round_trip(_make_output_batch(batch_reqs, output_sink))
            if output_sink is OutputSink.DETOKENIZER:
                output = round_trip(detokenizer.handle_batch_token_id_out(output))
                self.assertIsInstance(output, BatchStrOutput)
            self.assertEqual(output.rids, [req.rid for req in batch_reqs])
            return _receive_batch(states, output)

        def part(rid, lo, hi):
            return reqs[rid], rows[rid], lo, hi

        # "b64" is in the middle of its prefill: it has no output yet.
        _run_prefill_forward(
            processor, [part("plain", 0, 4), part("nested", 0, 5), part("b64", 0, 3)]
        )
        responses = respond([reqs["plain"], reqs["nested"]])
        _run_prefill_forward(
            processor, [part("b64", 3, 7), part("flat", 0, 3), part("last", 0, 4)]
        )
        # "nested" decodes its second and last token in the same step.
        nested = reqs["nested"]
        nested.output_ids.append(9)
        processor.append_output_token_logprobs(
            nested,
            9,
            -0.75,
            top_logprobs_val=[[-0.75, -2.0]],
            top_logprobs_idx=[[9, 10]],
        )
        nested.finished_reason = FINISH_LENGTH(length=2)
        first_nested = responses["nested"]
        responses.update(
            respond([nested, silent, reqs["b64"], reqs["flat"], reqs["last"]])
        )

        for rid in ("nested", "b64", "flat"):
            val, idx, weights = rows[rid]
            got_val, got_idx, got_weights = _read_response(responses[rid])
            self.assertSameBits(got_idx, np.asarray(idx[:-1], np.int32), rid)
            self.assertSameBits(got_val, np.asarray(val[:-1], np.float32), rid)
            self.assertSameBits(got_weights, weights[:-1].numpy(), rid)
        for rid in ("plain", "last"):
            val, idx, _ = rows[rid]
            got_val, got_idx, got_weights = _read_response(responses[rid])
            self.assertSameBits(got_idx, np.asarray(idx[:-1], np.int32), rid)
            self.assertSameBits(got_val, np.asarray(val[:-1], np.float32), rid)
            self.assertIsNone(got_weights, rid)
            self.assertEqual([key for key in responses[rid] if "weights" in key], [])
        # The weights were in the request's first response and stay in the
        # ones after it; its output positions keep their top logprobs.
        self.assertSameBits(
            _read_response(first_nested)[2], rows["nested"][2][:-1].numpy()
        )
        self.assertEqual(
            responses["nested"]["output_top_logprobs"],
            [
                [(-0.5, 7, None), (-1.25, 8, None)],
                [(-0.75, 9, None), (-2.0, 10, None)],
            ],
        )

    def _respond(self, rows, **layout) -> dict:
        """The meta_info of a selector request with these logprob rows, from
        the forward's output to the response."""
        num_rows = len(rows[0])
        req = _make_logprob_req(
            "r0",
            num_rows,
            selector=object(),
            return_flat_raw_top_logprobs="return_flat_raw_top_logprobs" in layout,
        )
        _run_prefill_forward(_make_logprob_processor(), [(req, rows, 0, num_rows)])
        state = _make_state(return_logprob=True, top_logprobs_num=2, **layout)
        return _receive_batch({"r0": state}, _make_output_batch([req]))["r0"]

    def test_every_layout_returns_the_weights_bit_for_bit(self):
        val, idx, _ = _make_candidate_rows(1, 6)
        # A quiet NaN with a payload, a negative NaN, the smallest float32,
        # 1e-9 and thirds; the last row is the sampling position's.
        weights = torch.from_numpy(
            np.array(
                [
                    [0x7FC00001, 0xFFC00000],
                    [0x00000001, 0x3F800000],
                    [0x3089705F, 0x3F7FFFFF],
                    [0x3EAAAAAB, 0x3F2AAAAB],
                    [0x00000000, 0x3F800000],
                    [0x3F000000, 0x3F000000],
                ],
                np.uint32,
            ).view(np.float32)
        )
        want = weights[:-1].numpy()
        self.assertEqual(want[2, 0], np.float32(1e-9))
        self.assertEqual(want[3, 0], np.float32(1 / 3))
        for name, layout in _LAYOUTS.items():
            with self.subTest(layout=name):
                _, _, got = _read_response(self._respond((val, idx, weights), **layout))
                # JSON has no NaN: null. Every other value reads back exactly.
                self.assertSameBits(
                    got, want if name == "b64" else _canonical_nan(want)
                )

    def test_non_finite_log_p_and_weights_in_every_layout(self):
        """A selector may fill a slot it has no candidate for with log p -inf
        and weight 0, and a position it cannot choose from with NaN log p and
        weights. JSON has neither value and the response carries null for
        both, so there only the weight of a slot, 0.0 or null, tells the two
        apart; base64 carries the float32 bits."""
        inf, nan = float("inf"), float("nan")
        # The last row is the sampling position's.
        val = [[-0.25, -1.5], [0.0, -inf], [nan, nan], [-1.0, -3.0]]
        idx = [[11, 22], [33, 44], [0, 0], [55, 66]]
        weights = torch.tensor([[0.75, 0.25], [1.0, 0.0], [nan, nan], [0.5, 0.5]])
        want_json = {
            "nested": {
                "input_top_logprobs": b"[null,[[-0.25,11,null],[-1.5,22,null]],"
                b"[[0.0,33,null],[null,44,null]],[[null,0,null],[null,0,null]]]",
                "input_top_logprobs_weights": (
                    b"[null,[0.75,0.25],[1.0,0.0],[null,null]]"
                ),
            },
            "flat": {
                "input_top_logprobs_val_flat": b"[-0.25,-1.5,0.0,null,null,null]",
                "input_top_logprobs_idx_flat": b"[11,22,33,44,0,0]",
                "input_top_logprobs_weights_flat": b"[0.75,0.25,1.0,0.0,null,null]",
            },
        }
        for name, want in want_json.items():
            meta_info = self._respond((val, idx, weights), **_LAYOUTS[name])
            for key, json_bytes in want.items():
                with self.subTest(layout=name, key=key):
                    # What the /generate route sends for the field.
                    self.assertEqual(dumps_json(meta_info[key]), json_bytes)
        with self.subTest(layout="b64"):
            got_val, got_idx, got_weights = _read_response(
                self._respond((val, idx, weights), **_LAYOUTS["b64"])
            )
            self.assertSameBits(got_val, np.asarray(val[:-1], np.float32))
            self.assertSameBits(got_idx, np.asarray(idx[:-1], np.int32))
            self.assertSameBits(got_weights, weights[:-1].numpy())
            self.assertEqual(got_val[1, 1], -inf)
            self.assertEqual(got_weights[1].tobytes(), np.float32([1, 0]).tobytes())
            self.assertTrue(np.isnan(got_val[2]).all())
            self.assertTrue(np.isnan(got_weights[2]).all())

    def test_nested_weights_out_of_step_with_the_positions_fail_the_response(self):
        """Rows are not padded with nulls up to the number of positions."""
        req = _make_logprob_req("r0", 4, selector=object())
        _run_prefill_forward(
            _make_logprob_processor(), [(req, _make_candidate_rows(1, 4), 0, 4)]
        )
        output = _make_output_batch([req])
        output.input_top_logprobs_weights[0] = output.input_top_logprobs_weights[0][1:]
        state = _make_state(return_logprob=True, top_logprobs_num=2)
        with self.assertRaises(AssertionError):
            _receive_batch({"r0": state}, output)

    def test_selector_request_without_input_logprobs_has_no_weights_keys(self):
        for name, layout in _LAYOUTS.items():
            with self.subTest(layout=name):
                req = _make_logprob_req(
                    "r0",
                    3,
                    selector=object(),
                    return_flat_raw_top_logprobs=name != "nested",
                )
                req.logprob_start_len = -1
                _run_prefill_forward(
                    _make_logprob_processor(), [(req, _make_candidate_rows(1, 0), 0, 0)]
                )
                output = _make_output_batch([req])
                self.assertIsNone(output.input_top_logprobs_weights)
                state = _make_state(return_logprob=True, top_logprobs_num=2, **layout)
                meta_info = _receive_batch({"r0": state}, output)["r0"]
                self.assertEqual([key for key in meta_info if "weights" in key], [])
                self.assertEqual(
                    meta_info["output_top_logprobs"],
                    [[(-0.5, 7, None), (-1.25, 8, None)]],
                )

    def test_weights_without_rows_still_have_their_keys(self):
        """A prompt of one token has no candidate row; the keys a client checks
        for are in the response all the same."""
        want = {
            "nested": {"input_top_logprobs_weights": [None]},
            "flat": {"input_top_logprobs_weights_flat": []},
            "b64": {
                "input_top_logprobs_weights_flat_b64": "",
                "input_top_logprobs_weights_flat_b64_dtype": "float32",
            },
        }
        for name, layout in _LAYOUTS.items():
            with self.subTest(layout=name):
                meta_info = self._respond(_make_candidate_rows(1, 1), **layout)
                self.assertEqual(
                    {key: meta_info[key] for key in meta_info if "weights" in key},
                    want[name],
                )
                if name == "nested":
                    self.assertEqual(meta_info["input_top_logprobs"], [None])
                else:
                    self.assertEqual(meta_info["input_top_logprobs_shape"], [0, 2])
                    self.assertEqual(meta_info["input_top_logprobs_null_prefix"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
