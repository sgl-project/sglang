"""Unit tests for the flat raw prompt top logprob response format
(`return_flat_raw_top_logprobs` / `return_flat_raw_top_logprobs_b64`).
"""

import asyncio
import base64
import pickle
import unittest
from array import array
from types import SimpleNamespace

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt import runtime_context as rc
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    GenerateReqInput,
    build_flat_input_top_logprobs_arrays,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.logprob_result_processor import (
    SchedulerLogprobResultProcessor,
)
from sglang.srt.managers.tokenizer_manager import (
    ReqState,
    TokenizerManager,
    _build_flat_input_top_logprobs_fields,
    _build_flat_input_top_logprobs_fields_from_arrays,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

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


class TestFlatAssembly(CustomTestCase):
    def test_flat_matches_nested_rows(self):
        fields = _build_flat_input_top_logprobs_fields(
            _VAL_ROWS, _IDX_ROWS, top_logprobs_num=2
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
        fields = _build_flat_input_top_logprobs_fields(
            _VAL_ROWS, _IDX_ROWS, top_logprobs_num=2, return_b64=True
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
        fields = _build_flat_input_top_logprobs_fields(
            [None], [None], top_logprobs_num=2
        )
        self.assertEqual(fields["input_top_logprobs_shape"], [0, 2])
        self.assertEqual(fields["input_top_logprobs_null_prefix"], 1)
        self.assertEqual(fields["input_top_logprobs_val_flat"], [])
        self.assertEqual(fields["input_top_logprobs_idx_flat"], [])

    def test_rejects_null_row_after_prefix(self):
        with self.assertRaisesRegex(ValueError, "leading prefix"):
            _build_flat_input_top_logprobs_fields(
                [None, [-0.1, -2.5], None, [-0.3, -1.5]],
                [None, [11, 22], None, [33, 44]],
                top_logprobs_num=2,
            )

    def test_rejects_ragged_rows(self):
        with self.assertRaisesRegex(ValueError, "rectangular"):
            _build_flat_input_top_logprobs_fields(
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


def _make_logprob_processor() -> SchedulerLogprobResultProcessor:
    # enable_mis comes from the published exec bag, not from here; see setUp.
    return SchedulerLogprobResultProcessor(
        model_config=SimpleNamespace(vocab_size=1_000_000),
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

    def _run_prefill(self, req: Req, chunk_sizes) -> None:
        processor = _make_logprob_processor()
        token_logprobs = [row[0] for row in _SCHED_VAL_ROWS]
        pt = 0
        for chunk_idx, size in enumerate(chunk_sizes):
            output = SimpleNamespace(
                input_token_logprobs=tuple(token_logprobs[pt : pt + size]),
                input_top_logprobs_val=[_SCHED_VAL_ROWS[pt : pt + size]],
                input_top_logprobs_idx=[_SCHED_IDX_ROWS[pt : pt + size]],
            )
            processor.add_input_logprob_return_values(
                0,
                req,
                output,
                0,
                size,
                last_prefill_chunk=chunk_idx == len(chunk_sizes) - 1,
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
        from_rows = _build_flat_input_top_logprobs_fields(
            _EXACT_VAL_ROWS, _IDX_ROWS, top_logprobs_num=2, return_b64=return_b64
        )
        val_arr, idx_arr, null_prefix = build_flat_input_top_logprobs_arrays(
            _EXACT_VAL_ROWS, _IDX_ROWS, top_logprobs_num=2
        )
        from_arrays = _build_flat_input_top_logprobs_fields_from_arrays(
            val_arr, idx_arr, null_prefix, return_b64=return_b64
        )
        return from_rows, from_arrays

    def test_non_b64(self):
        from_rows, from_arrays = self._both(return_b64=False)
        self.assertEqual(from_rows, from_arrays)

    def test_b64(self):
        from_rows, from_arrays = self._both(return_b64=True)
        self.assertEqual(from_rows, from_arrays)

    def test_all_null_rows(self):
        val_arr, idx_arr, null_prefix = build_flat_input_top_logprobs_arrays(
            [None], [None], top_logprobs_num=2
        )
        self.assertEqual(val_arr.shape, (0, 2))
        self.assertEqual(null_prefix, 1)
        fields = _build_flat_input_top_logprobs_fields_from_arrays(
            val_arr, idx_arr, null_prefix
        )
        self.assertEqual(
            fields,
            _build_flat_input_top_logprobs_fields([None], [None], top_logprobs_num=2),
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
        state.input_top_logprobs_scheduler_flat = build_flat_input_top_logprobs_arrays(
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
        decoded_texts=["", ""],
        decode_ids=[array("q", [1]), array("q", [2])],
        read_offsets=[0] * n,
        output_ids=None,
        skip_special_tokens=[True] * n,
        spaces_between_special_tokens=[True] * n,
        no_stop_trim=[False] * n,
        prompt_tokens=[5] * n,
        reasoning_tokens=[0] * n,
        completion_tokens=[1] * n,
        cached_tokens=[0] * n,
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
        val_arr, idx_arr, null_prefix = build_flat_input_top_logprobs_arrays(
            _EXACT_VAL_ROWS, _IDX_ROWS, top_logprobs_num=2
        )
        return _make_batch_token_id_output(
            input_top_logprobs_val_flat=[None, val_arr],
            input_top_logprobs_idx_flat=[None, idx_arr],
            input_top_logprobs_flat_null_prefix=[None, null_prefix],
        )

    def _check_roundtrip(self, decoded, original):
        self.assertIsNone(decoded.input_top_logprobs_val_flat[0])
        self.assertIsNone(decoded.input_top_logprobs_idx_flat[0])
        self.assertIsNone(decoded.input_top_logprobs_flat_null_prefix[0])
        for got, sent in (
            (
                decoded.input_top_logprobs_val_flat[1],
                original.input_top_logprobs_val_flat[1],
            ),
            (
                decoded.input_top_logprobs_idx_flat[1],
                original.input_top_logprobs_idx_flat[1],
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

    def test_fields_default_none(self):
        output = _make_batch_token_id_output()
        self.assertIsNone(output.input_top_logprobs_val_flat)
        self.assertIsNone(output.input_top_logprobs_idx_flat)
        self.assertIsNone(output.input_top_logprobs_flat_null_prefix)


if __name__ == "__main__":
    unittest.main(verbosity=2)
