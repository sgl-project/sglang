"""Unit tests for dense decode top logprobs.

Uniform decode batches compute, copy and store top logprobs as [batch, k]
matrices; requests keep numpy row views and the output streamer sends one
array per request chunk. These tests cover that transport and its equivalence
with the nested and flat (`return_flat_raw_output_top_logprobs`) responses.
"""

import asyncio
import base64
import pickle
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.logprob_processor import (
    OutputLogprobProcessor,
    get_top_logprobs,
)
from sglang.srt.managers.io_struct import GenerateReqInput, flat_top_logprobs_layout
from sglang.srt.managers.scheduler_components import output_streamer
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    _top_logprobs_to_lists,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _TokenizerManagerStub:
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


def _flat_state(k=2):
    return _make_state(
        return_logprob=True,
        top_logprobs_num=k,
        return_flat_raw_output_top_logprobs=True,
        return_flat_raw_top_logprobs_b64=True,
    )


def _recv(values, indices):
    return SimpleNamespace(
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=[[-0.5] * len(values)],
        output_token_logprobs_idx=[[1] * len(values)],
        input_top_logprobs_val=None,
        input_top_logprobs_val_flat=None,
        input_top_logprobs_weights=None,
        output_top_logprobs_val=[values],
        output_top_logprobs_idx=[indices],
        input_token_ids_logprobs_val=None,
        output_token_ids_logprobs_val=None,
    )


def _convert(state, recv_obj, k, finished=True):
    meta_info = {}
    _TokenizerManagerStub().convert_logprob_style(
        meta_info,
        state,
        top_logprobs_num=k,
        token_ids_logprob=None,
        return_text_in_logprobs=False,
        recv_obj=recv_obj,
        recv_obj_index=0,
        finished=finished,
    )
    return meta_info


def _decode(meta_info):
    rows, k = meta_info["output_top_logprobs_shape"]
    values = np.frombuffer(
        base64.b64decode(meta_info["output_top_logprobs_val_flat_b64"]), dtype="<f4"
    ).reshape(rows, k)
    indices = np.frombuffer(
        base64.b64decode(meta_info["output_top_logprobs_idx_flat_b64"]), dtype="<i4"
    ).reshape(rows, k)
    return values, indices


# Float32-exact values so the nested and dense paths compare exactly.
_VALUES = np.array([[-0.5, -1.25], [-0.125, -2.5], [-0.25, -3.0]], dtype=np.float32)
_INDICES = np.array([[7, 9], [3, 4], [11, 2]], dtype=np.int64)


class TestDenseDecodeTopk(CustomTestCase):
    def test_uniform_batch_returns_dense_matrices(self):
        torch.manual_seed(0)
        logprobs = torch.log_softmax(torch.randn(4, 50), dim=-1)
        expected_values, expected_indices = get_top_logprobs(
            logprobs.clone(), [3] * 4, no_copy_to_cpu=True
        )
        result = OutputLogprobProcessor().compute_logprobs(
            logprobs, [3] * 4, [None] * 4, torch.zeros(4, dtype=torch.long)
        )
        self.assertIsInstance(result.top_logprobs_val, torch.Tensor)
        self.assertEqual(tuple(result.top_logprobs_val.shape), (4, 3))
        for row in range(4):
            torch.testing.assert_close(
                result.top_logprobs_val[row], expected_values[row], rtol=0, atol=0
            )
            torch.testing.assert_close(
                result.top_logprobs_idx[row], expected_indices[row], rtol=0, atol=0
            )

    def test_mixed_widths_keep_row_lists(self):
        logprobs = torch.log_softmax(torch.randn(3, 20), dim=-1)
        result = OutputLogprobProcessor().compute_logprobs(
            logprobs, [2, 0, 3], [None] * 3, torch.zeros(3, dtype=torch.long)
        )
        self.assertIsInstance(result.top_logprobs_val, list)
        self.assertEqual([len(row) for row in result.top_logprobs_val], [2, 0, 3])

    def test_dense_rows_become_numpy_matrices(self):
        values, indices = _top_logprobs_to_lists(
            torch.from_numpy(_VALUES.copy()), torch.from_numpy(_INDICES.copy())
        )
        self.assertIsInstance(values, np.ndarray)
        np.testing.assert_array_equal(values, _VALUES)
        self.assertEqual(indices[1].tolist(), [3, 4])

    def test_row_lists_become_python_lists(self):
        values, indices = _top_logprobs_to_lists(
            [torch.tensor([-0.5]), torch.tensor([])],
            [torch.tensor([4]), torch.tensor([])],
        )
        self.assertEqual(values, [[-0.5], []])
        self.assertEqual(indices, [[4], []])


class TestStreamerPacking(CustomTestCase):
    def test_numpy_rows_pack_into_one_array(self):
        packed = output_streamer._pack_top_logprob_rows([_VALUES[i] for i in range(3)])
        self.assertIsInstance(packed, np.ndarray)
        self.assertEqual(packed.dtype, np.float32)
        np.testing.assert_array_equal(packed, _VALUES)
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(packed)), _VALUES)

    def test_prefill_list_row_then_numpy_rows(self):
        rows = [_VALUES[0].tolist(), _VALUES[1], _VALUES[2]]
        np.testing.assert_array_equal(
            output_streamer._pack_top_logprob_rows(rows), _VALUES
        )

    def test_list_rows_unchanged(self):
        rows = [[-0.5, -1.0], [-0.25, -2.0]]
        self.assertIs(output_streamer._pack_top_logprob_rows(rows), rows)
        self.assertEqual(output_streamer._pack_top_logprob_rows([]), [])

    def test_ragged_rows_fall_back_to_lists(self):
        rows = [[], np.array([-0.5, -1.0], dtype=np.float32)]
        self.assertEqual(
            output_streamer._pack_top_logprob_rows(rows), [[], [-0.5, -1.0]]
        )

    def test_msgpack_ipc_sends_lists(self):
        original = output_streamer._USE_PICKLE_IPC
        output_streamer._USE_PICKLE_IPC = False
        try:
            packed = output_streamer._pack_top_logprob_rows([_VALUES[0], _VALUES[1]])
        finally:
            output_streamer._USE_PICKLE_IPC = original
        self.assertEqual(packed, _VALUES[:2].tolist())


class TestTokenizerManagerDenseChunks(CustomTestCase):
    def test_nested_from_dense_chunk_matches_list_chunk(self):
        dense = _convert(
            _make_state(return_logprob=True, top_logprobs_num=2),
            _recv(_VALUES, _INDICES),
            2,
        )
        nested = _convert(
            _make_state(return_logprob=True, top_logprobs_num=2),
            _recv(_VALUES.tolist(), _INDICES.tolist()),
            2,
        )
        self.assertEqual(dense["output_top_logprobs"], nested["output_top_logprobs"])
        self.assertIsInstance(dense["output_top_logprobs"][0][0][0], float)
        self.assertIsInstance(dense["output_top_logprobs"][0][0][1], int)

    def test_flat_from_dense_chunks_matches_list_chunks(self):
        state = _flat_state()
        first = _convert(state, _recv(_VALUES[:2], _INDICES[:2]), 2, finished=False)
        self.assertNotIn("output_top_logprobs_val_flat_b64", first)
        self.assertNotIn("output_top_logprobs", first)
        dense = _convert(state, _recv(_VALUES[2:], _INDICES[2:]), 2)
        lists = _convert(_flat_state(), _recv(_VALUES.tolist(), _INDICES.tolist()), 2)
        self.assertNotIn("output_top_logprobs", dense)
        for key in (
            "output_top_logprobs_val_flat_b64",
            "output_top_logprobs_idx_flat_b64",
            "output_top_logprobs_shape",
            "output_top_logprobs_null_prefix",
        ):
            self.assertEqual(dense[key], lists[key])
        values, indices = _decode(dense)
        np.testing.assert_array_equal(values, _VALUES)
        np.testing.assert_array_equal(indices, _INDICES)

    def test_flat_with_prefill_list_row_then_dense_chunk(self):
        state = _flat_state()
        _convert(state, _recv([_VALUES[0].tolist()], [_INDICES[0].tolist()]), 2, False)
        values, indices = _decode(_convert(state, _recv(_VALUES[1:], _INDICES[1:]), 2))
        np.testing.assert_array_equal(values, _VALUES)
        np.testing.assert_array_equal(indices, _INDICES)

    def test_ragged_flat_falls_back_to_nested_from_numpy_rows(self):
        state = _flat_state()
        _convert(state, _recv([[-0.5]], [[4]]), 2, finished=False)
        meta = _convert(state, _recv(_VALUES[:1], _INDICES[:1]), 2)
        self.assertNotIn("output_top_logprobs_val_flat_b64", meta)
        self.assertEqual(
            meta["output_top_logprobs"],
            [[(-0.5, 4, None)], [(-0.5, 7, None), (-1.25, 9, None)]],
        )

    def test_layout_accepts_numpy_rows(self):
        rows = [None, [], _VALUES[0], _VALUES[1]]
        self.assertEqual(flat_top_logprobs_layout(rows, 2), (2, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
