import asyncio
import base64
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    ReqState,
    _add_sampling_output_to_meta_info,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_mask import SamplingMaskChunk

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_FLAT_KEYS = (
    "output_sampling_mask_logprobs_val_flat",
    "output_sampling_mask_logprobs_idx_flat",
    "output_sampling_mask_logprobs_row_lengths",
)
# The output sampling masks of three batches, one row of token ids per output token.
_ROWS = [[[5, 7, 9], [2]], [[4, 1], [8, 6, 3, 0]], [[11]]]
_NUM_TOKENS = sum(len(rows) for rows in _ROWS)


def _chunk(rows) -> SamplingMaskChunk:
    """Float32-exact logprobs: one per row for the sampled token, one per token id."""
    token_ids = [t for row in rows for t in row]
    return SamplingMaskChunk(
        selected_logprobs=-np.arange(1, len(rows) + 1, dtype=np.float32) / 4,
        lengths=np.asarray([len(row) for row in rows], dtype=np.int32),
        token_ids=np.asarray(token_ids, dtype=np.int32),
        support_logprobs=-np.arange(1, len(token_ids) + 1, dtype=np.float32) / 8,
    )


def _make_state(filters_sampling: bool = True, **obj_kwargs) -> ReqState:
    obj = GenerateReqInput(
        text="hello",
        return_logprob=True,
        return_sampling_logprobs=True,
        return_sampling_mask=True,
        **obj_kwargs,
    )
    obj.normalize_batch_and_arguments()
    obj.filters_sampling = filters_sampling
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _meta_info(state: ReqState, chunks, finished: bool, num_output_tokens=0) -> dict:
    meta_info = {}
    for i, chunk in enumerate(chunks):
        _add_sampling_output_to_meta_info(
            meta_info,
            state,
            chunk,
            num_output_tokens=num_output_tokens,
            finished=finished and i == len(chunks) - 1,
            detokenize_logprob_tokens=lambda logprobs, token_ids: [
                (logprob, token_id, None)
                for logprob, token_id in zip(logprobs, token_ids)
            ],
        )
    return meta_info


def _flat_values(fields: dict, key: str) -> np.ndarray:
    if f"{key}_b64" in fields:
        return np.frombuffer(
            base64.b64decode(fields[f"{key}_b64"]), dtype=fields[f"{key}_b64_dtype"]
        )
    return np.asarray(fields[key])


class TestFlatRawSamplingMaskValidation(CustomTestCase):
    def test_invalid_flag_combinations_are_rejected(self):
        for kwargs, error in (
            ({"stream": True, "return_flat_raw_sampling_mask": True}, "stream"),
            ({"return_flat_raw_sampling_mask_b64": True}, "requires return_flat_raw"),
        ):
            with self.subTest(kwargs=kwargs):
                req = GenerateReqInput(text="hello", **kwargs)
                with self.assertRaisesRegex(ValueError, error):
                    req.normalize_batch_and_arguments()

    def test_flags_propagate_to_batch_items(self):
        req = GenerateReqInput(
            text=["a", "b"],
            return_flat_raw_sampling_mask=True,
            return_flat_raw_sampling_mask_b64=True,
        )
        req.normalize_batch_and_arguments()
        for i in range(2):
            self.assertTrue(req[i].return_flat_raw_sampling_mask)
            self.assertTrue(req[i].return_flat_raw_sampling_mask_b64)


class TestSamplingOutputMetaInfo(CustomTestCase):
    def test_flat_matches_nested_rows(self):
        chunks = [_chunk(rows) for rows in _ROWS]
        nested = _meta_info(_make_state(), chunks, finished=True)
        nested_rows = nested["output_sampling_mask_logprobs"]
        self.assertEqual(
            [[token_id for _, token_id, _ in row] for row in nested_rows],
            [row for rows in _ROWS for row in rows],
        )
        self.assertEqual(len(nested["output_token_sampling_logprobs"]), _NUM_TOKENS)
        for b64 in (False, True):
            with self.subTest(b64=b64):
                state = _make_state(
                    return_flat_raw_sampling_mask=True,
                    return_flat_raw_sampling_mask_b64=b64,
                )
                unfinished = _meta_info(state, chunks[:-1], finished=False)
                fields = _meta_info(state, chunks[-1:], finished=True)
                logprobs, ids, lengths = (
                    _flat_values(fields, key) for key in _FLAT_KEYS
                )

                self.assertNotIn("output_sampling_mask_logprobs", unfinished)
                self.assertNotIn("output_sampling_mask_logprobs", fields)
                self.assertEqual(state.output_sampling_mask_logprobs, [])
                self.assertEqual(
                    fields["output_token_sampling_logprobs"],
                    nested["output_token_sampling_logprobs"],
                )
                ends = np.cumsum(lengths)[:-1]
                self.assertEqual(
                    [
                        list(zip(row_logprobs.tolist(), row_ids.tolist()))
                        for row_logprobs, row_ids in zip(
                            np.split(logprobs, ends), np.split(ids, ends)
                        )
                    ],
                    [
                        [(logprob, token_id) for logprob, token_id, _ in row]
                        for row in nested_rows
                    ],
                )

    def test_a_request_finished_without_rows_gets_empty_arrays(self):
        state = _make_state(return_flat_raw_sampling_mask=True)

        fields = _meta_info(state, [None], finished=True)

        self.assertEqual(fields["output_token_sampling_logprobs"], [])
        for key in _FLAT_KEYS:
            self.assertEqual(fields[key], [])

    def test_unfiltered_request_returns_output_logprobs_and_empty_rows(self):
        """Without a sampling filter nothing is captured: the sampling logprobs are
        the output token logprobs and every mask row is empty, one per output token."""
        output_logprobs = [-0.5, -1.25, -0.125]
        for flat in (False, True):
            with self.subTest(flat=flat):
                state = _make_state(
                    filters_sampling=False, return_flat_raw_sampling_mask=flat
                )
                state.output_token_logprobs_val.extend(output_logprobs[:2])
                _meta_info(state, [None], finished=False, num_output_tokens=2)
                state.output_token_logprobs_val.extend(output_logprobs[2:])
                fields = _meta_info(state, [None], finished=True, num_output_tokens=3)

                self.assertEqual(
                    fields["output_token_sampling_logprobs"], output_logprobs
                )
                if flat:
                    self.assertEqual(fields[_FLAT_KEYS[0]], [])
                    self.assertEqual(fields[_FLAT_KEYS[1]], [])
                    self.assertEqual(fields[_FLAT_KEYS[2]], [0, 0, 0])
                else:
                    self.assertEqual(
                        fields["output_sampling_mask_logprobs"], [[], [], []]
                    )


if __name__ == "__main__":
    unittest.main()
