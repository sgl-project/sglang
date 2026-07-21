"""Unit tests for the flat raw prompt top logprob response format
(`return_flat_raw_top_logprobs` / `return_flat_raw_top_logprobs_b64`).
"""

import asyncio
import base64
import json
import os
import time
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    ReqState,
    TokenizerManager,
    _build_flat_input_top_logprobs_fields,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Synthetic per-position top-k rows (k=2). The leading None mirrors the first
# prompt position, which has no top logprobs.
_VAL_ROWS = [None, [-0.1, -2.5], [-0.3, -1.5], [-0.05, -4.0]]
_IDX_ROWS = [None, [11, 22], [33, 44], [55, 66]]


class _TokenizerManagerStub:
    """Borrow the real logprob meta_info methods without a full manager."""

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
    def test_b64_requires_flat_flag(self):
        req = GenerateReqInput(text="hello", return_flat_raw_top_logprobs_b64=True)
        with self.assertRaisesRegex(ValueError, "return_flat_raw_top_logprobs"):
            req.normalize_batch_and_arguments()

    def test_flags_default_off_and_valid_combinations(self):
        for kwargs in (
            {},
            {"return_flat_raw_top_logprobs": True},
            {
                "return_flat_raw_top_logprobs": True,
                "return_flat_raw_top_logprobs_b64": True,
            },
        ):
            req = GenerateReqInput(text="hello", **kwargs)
            req.normalize_batch_and_arguments()

    def test_flags_propagate_to_batch_items(self):
        req = GenerateReqInput(
            text=["a", "b"],
            return_flat_raw_top_logprobs=True,
            return_flat_raw_top_logprobs_b64=True,
        )
        req.normalize_batch_and_arguments()
        for i in range(2):
            self.assertTrue(req[i].return_flat_raw_top_logprobs)
            self.assertTrue(req[i].return_flat_raw_top_logprobs_b64)


class TestFlatAssembly(CustomTestCase):
    def test_flat_matches_nested_rows(self):
        fields = _build_flat_input_top_logprobs_fields(
            _VAL_ROWS, _IDX_ROWS, top_logprobs_num=2, return_b64=False
        )
        self.assertEqual(fields["input_top_logprobs_shape"], [4, 2])
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
        self.assertEqual(len(flat_val), (rows - null_prefix) * k)
        for i in range(null_prefix, rows):
            start = (i - null_prefix) * k
            self.assertEqual(flat_val[start : start + k], _VAL_ROWS[i])

    def test_b64_roundtrip(self):
        fields = _build_flat_input_top_logprobs_fields(
            _VAL_ROWS, _IDX_ROWS, top_logprobs_num=2, return_b64=True
        )
        self.assertEqual(fields["input_top_logprobs_shape"], [4, 2])
        self.assertEqual(fields["input_top_logprobs_null_prefix"], 1)
        self.assertEqual(fields["input_top_logprobs_val_flat_b64_dtype"], "float32")
        self.assertEqual(fields["input_top_logprobs_idx_flat_b64_dtype"], "int32")
        rows, k = fields["input_top_logprobs_shape"]
        null_prefix = fields["input_top_logprobs_null_prefix"]
        val = np.frombuffer(
            base64.b64decode(fields["input_top_logprobs_val_flat_b64"]),
            dtype=np.dtype(fields["input_top_logprobs_val_flat_b64_dtype"]),
        ).reshape(rows - null_prefix, k)
        idx = np.frombuffer(
            base64.b64decode(fields["input_top_logprobs_idx_flat_b64"]),
            dtype=np.dtype(fields["input_top_logprobs_idx_flat_b64_dtype"]),
        ).reshape(rows - null_prefix, k)
        np.testing.assert_array_equal(val, np.asarray(_VAL_ROWS[1:], dtype=np.float32))
        np.testing.assert_array_equal(idx, np.asarray(_IDX_ROWS[1:], dtype=np.int32))

    def test_all_null_rows(self):
        fields = _build_flat_input_top_logprobs_fields(
            [None], [None], top_logprobs_num=2, return_b64=False
        )
        self.assertEqual(fields["input_top_logprobs_shape"], [1, 2])
        self.assertEqual(fields["input_top_logprobs_null_prefix"], 1)
        self.assertEqual(fields["input_top_logprobs_val_flat"], [])
        self.assertEqual(fields["input_top_logprobs_idx_flat"], [])

    def test_rejects_null_row_after_prefix(self):
        with self.assertRaisesRegex(ValueError, "leading prefix"):
            _build_flat_input_top_logprobs_fields(
                [None, [-0.1, -2.5], None, [-0.3, -1.5]],
                [None, [11, 22], None, [33, 44]],
                top_logprobs_num=2,
                return_b64=False,
            )

    def test_rejects_ragged_rows(self):
        with self.assertRaisesRegex(ValueError, "rectangular"):
            _build_flat_input_top_logprobs_fields(
                [None, [-0.1, -2.5], [-0.3]],
                [None, [11, 22], [33]],
                top_logprobs_num=2,
                return_b64=False,
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
        self.assertEqual(meta_info["input_top_logprobs_shape"], [4, 2])
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
            return_flat_raw_top_logprobs_b64=True,
        )
        self._extend_input_top(one_shot, _VAL_ROWS, _IDX_ROWS)
        expected = _add_logprob_meta_info(one_shot)

        # Rows arriving across two chunks, with meta_info assembled after each
        # (as happens for streaming requests).
        chunked = _make_state(
            return_logprob=True,
            top_logprobs_num=2,
            return_flat_raw_top_logprobs=True,
            return_flat_raw_top_logprobs_b64=True,
        )
        self._extend_input_top(chunked, _VAL_ROWS[:2], _IDX_ROWS[:2])
        _add_logprob_meta_info(chunked)
        self._extend_input_top(chunked, _VAL_ROWS[2:], _IDX_ROWS[2:])
        got = _add_logprob_meta_info(chunked)

        flat_keys = [
            "input_top_logprobs_val_flat_b64",
            "input_top_logprobs_idx_flat_b64",
            "input_top_logprobs_val_flat_b64_dtype",
            "input_top_logprobs_idx_flat_b64_dtype",
            "input_top_logprobs_shape",
            "input_top_logprobs_null_prefix",
        ]
        for key in flat_keys:
            self.assertEqual(got[key], expected[key])

        # No new rows -> the encoded payload is reused, not rebuilt.
        again = _add_logprob_meta_info(chunked)
        self.assertIs(
            again["input_top_logprobs_val_flat_b64"],
            got["input_top_logprobs_val_flat_b64"],
        )


@unittest.skipUnless(
    os.environ.get("SGLANG_BENCH_FLAT_RAW_TOP_LOGPROBS"),
    "Serialization microbenchmark; set SGLANG_BENCH_FLAT_RAW_TOP_LOGPROBS=1 to run.",
)
class BenchFlatRawTopLogprobsSerialization(CustomTestCase):
    """Compares JSON-encode time and payload size of the three formats."""

    def test_bench(self):
        num_positions, k = 32768, 2
        rng = np.random.default_rng(0)
        vals = rng.standard_normal((num_positions, k)).astype(np.float32)
        idxs = rng.integers(0, 150000, size=(num_positions, k), dtype=np.int32)
        val_rows = [None] + vals[1:].tolist()
        idx_rows = [None] + idxs[1:].tolist()

        def bench(name, build):
            start = time.perf_counter()
            payload = json.dumps(build())
            elapsed = time.perf_counter() - start
            print(f"{name}: {elapsed * 1e3:.1f} ms, {len(payload)} bytes")

        bench(
            "nested triples",
            lambda: [
                (None if row is None else [(v, i, None) for v, i in zip(row, idx_row)])
                for row, idx_row in zip(val_rows, idx_rows)
            ],
        )
        bench(
            "flat lists",
            lambda: _build_flat_input_top_logprobs_fields(
                val_rows, idx_rows, top_logprobs_num=k, return_b64=False
            ),
        )
        bench(
            "flat b64",
            lambda: _build_flat_input_top_logprobs_fields(
                val_rows, idx_rows, top_logprobs_num=k, return_b64=True
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
