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
    _add_sampling_mask_to_meta_info,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_mask import SamplingMaskChunk

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_FLAT_KEYS = (
    "output_token_sampling_mask_flat",
    "output_token_sampling_mask_row_lengths",
    "output_token_sampling_logprobs_flat",
)
# The output sampling masks of three batches, one row of token ids per output token.
_ROWS = [[[5, 7, 9], [2]], [[4, 1], [8, 6, 3, 0]], [[11]]]


def _chunk(rows, support_logprobs: bool) -> SamplingMaskChunk:
    """Float32-exact logprobs: one per token id in support mode, one per row in selected mode."""
    token_ids = [t for row in rows for t in row]
    count = len(token_ids) if support_logprobs else len(rows)
    return SamplingMaskChunk(
        lengths=np.asarray([len(row) for row in rows], dtype=np.int32),
        token_ids=np.asarray(token_ids, dtype=np.int32),
        logprobs=-np.arange(1, count + 1, dtype=np.float32) / 8,
    )


def _make_state(**obj_kwargs) -> ReqState:
    obj = GenerateReqInput(text="hello", return_sampling_mask=True, **obj_kwargs)
    obj.normalize_batch_and_arguments()
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _meta_info(state: ReqState, chunks, finished: bool) -> dict:
    meta_info = {}
    for i, chunk in enumerate(chunks):
        _add_sampling_mask_to_meta_info(
            meta_info, state, chunk, finished=finished and i == len(chunks) - 1
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


class TestFlatRawSamplingMask(CustomTestCase):
    def test_flat_matches_nested_rows(self):
        for mode in ("selected", "support"):
            chunks = [_chunk(rows, mode == "support") for rows in _ROWS]
            nested = _meta_info(
                _make_state(sampling_logprobs_mode=mode), chunks, finished=True
            )
            for b64 in (False, True):
                with self.subTest(mode=mode, b64=b64):
                    state = _make_state(
                        sampling_logprobs_mode=mode,
                        return_flat_raw_sampling_mask=True,
                        return_flat_raw_sampling_mask_b64=b64,
                    )
                    unfinished = _meta_info(state, chunks[:-1], finished=False)
                    fields = _meta_info(state, chunks[-1:], finished=True)
                    ids, lengths, logprobs = (
                        _flat_values(fields, key) for key in _FLAT_KEYS
                    )

                    self.assertEqual(unfinished, {})
                    self.assertEqual(state.output_token_sampling_mask, [])
                    self.assertNotIn("output_token_sampling_mask", fields)
                    self.assertEqual(
                        [
                            row.tolist()
                            for row in np.split(ids, np.cumsum(lengths)[:-1])
                        ],
                        nested["output_token_sampling_mask"],
                    )
                    self.assertEqual(
                        fields["output_token_sampling_mask_length"],
                        nested["output_token_sampling_mask_length"],
                    )
                    expected = nested["output_token_sampling_logprobs"]
                    if mode == "support":
                        expected = [v for row in expected for v in row]
                    self.assertEqual(logprobs.tolist(), expected)

    def test_a_request_finished_without_rows_gets_empty_arrays(self):
        state = _make_state(return_flat_raw_sampling_mask=True)

        fields = _meta_info(state, [None], finished=True)

        self.assertEqual(fields["output_token_sampling_mask_length"], 0)
        for key in _FLAT_KEYS:
            self.assertEqual(fields[key], [])


if __name__ == "__main__":
    unittest.main()
