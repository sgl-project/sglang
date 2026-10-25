"""Behavior guards for the incremental streaming coalescer's performance changes."""

import asyncio
import copy
import unittest

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestStreamingCoalescing(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tm = TokenizerManager.__new__(TokenizerManager)
        self.tm.incremental_streaming_output = True
        obj = GenerateReqInput(text="hello", rid="coalesce", stream=True)
        self.state = ReqState([], False, asyncio.Event(), obj, APIServerReqTimeStats())
        self.state.time_stats.set_response_sent_to_client_time()
        self.tm.rid_to_state = {obj.rid: self.state}

    async def consume(self, chunks):
        self.state.out_list = chunks
        self.state.event.set()
        stream = self.tm._wait_one_response(self.state.obj)
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    async def test_optional_metadata_keeps_absence_empty_lists_and_token_order(self):
        for key in (
            "output_token_logprobs",
            "output_top_logprobs",
            "output_token_ids_logprobs",
            "output_token_sampling_mask",
            "output_token_sampling_logprobs",
        ):
            for values, expected in (
                ((None, None, None), None),
                (([], None, None), []),
                (([None, [1]], None, [[2]]), [None, [1], [2]]),
            ):
                with self.subTest(key=key, values=values):
                    chunks = [
                        {"text": text, "output_ids": [token], "meta_info": {}}
                        for text, token in (("a", 1), ("b", 2), ("c", 3))
                    ]
                    for chunk, value in zip(chunks, values):
                        if value is not None:
                            chunk["meta_info"][key] = value
                    original = copy.deepcopy(chunks)
                    out = await self.consume(chunks)
                    self.assertEqual(
                        out,
                        {
                            "text": "abc",
                            "output_ids": [1, 2, 3],
                            "meta_info": {} if expected is None else {key: expected},
                        },
                    )
                    out["output_ids"].append(99)
                    if expected is not None:
                        out["meta_info"][key].append("later")
                    self.assertEqual(chunks, original)

    async def test_sparse_custom_metadata_keeps_only_reported_incremental_fields(self):
        self.state.customized_info_accumulated = {
            "scores": [1, 2],
            "hidden": [99],
        }
        chunks = [
            {"meta_info": {"scores": [1], "earlier_only": True, "count": 1}},
            {"meta_info": {}},
            {"meta_info": {"scores": [2], "count": 3}},
        ]
        original = copy.deepcopy(chunks)
        out = await self.consume(chunks)
        self.assertEqual(out, {"meta_info": {"scores": [1, 2], "count": 3}})
        out["meta_info"]["scores"].append(4)
        self.assertEqual(chunks, original)

    async def test_single_chunk_and_cumulative_streams_return_latest_output(self):
        for incremental, count in ((True, 1), (False, 2)):
            with self.subTest(incremental=incremental):
                self.tm.incremental_streaming_output = incremental
                chunks = [
                    {"text": "hello", "output_ids": [1], "meta_info": {}},
                    {"text": "hello world", "output_ids": [1, 2], "meta_info": {}},
                ][-count:]
                self.assertEqual(await self.consume(chunks), chunks[-1])


if __name__ == "__main__":
    unittest.main()
