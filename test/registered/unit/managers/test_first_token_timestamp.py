import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from test_output_streamer_customized_info import _FakeReq

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.scheduler_components.output_streamer import (
    _GenerationStreamAccumulator,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def accumulator():
    return _GenerationStreamAccumulator(
        return_logprob=False,
        return_hidden_states=False,
        return_routed_experts=False,
        return_indexer_topk=False,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        disaggregation_mode=DisaggregationMode.NULL,
        default_stream_interval=1,
        default_force_stream_interval=50,
        get_cached_tokens_details=lambda req: None,
        current_weight_version=None,
    )


class TestFirstTokenFlush(unittest.TestCase):
    def emit(self, req):
        acc = accumulator()
        acc.accept(req=req)
        return acc.to_payload(dp_rank=0, is_idle_batch=False)

    def test_nonstream_first_token_and_speculative_block_flush_immediately(self):
        for count in (1, 8, 50):
            with self.subTest(count=count):
                req = _FakeReq("test", list(range(count)))
                payload = self.emit(req)
                self.assertEqual(payload.output_ids, [list(range(count))])
                self.assertEqual(req.send_token_offset, count)
                self.assertEqual(payload.finished_reasons, [None])
                self.assertFalse(req.stream)

    def test_subsequent_outputs_keep_batching_without_duplicate_tokens(self):
        req = _FakeReq("test", list(range(8)))
        self.assertEqual(self.emit(req).output_ids, [list(range(8))])
        req.output_ids = req.output_ids_through_stop = list(range(9))
        self.assertIsNone(self.emit(req))
        req.output_ids = req.output_ids_through_stop = list(range(50))
        self.assertEqual(self.emit(req).output_ids, [list(range(8, 50))])
        req.output_ids = req.output_ids_through_stop = list(range(53))
        req._finished = True
        req.finished_reason = SimpleNamespace(to_json=lambda: {"type": "length"})
        self.assertEqual(self.emit(req).output_ids, [list(range(50, 53))])
        self.assertTrue(req.finished_output)

    def test_first_output_waits_for_stop_prefix_to_clear(self):
        req = _FakeReq("test", [10])
        req.check_match_stop_str_prefix = Mock(return_value=True)
        self.assertIsNone(self.emit(req))
        self.assertEqual(req.send_token_offset, 0)
        req.output_ids = req.output_ids_through_stop = [10, 11]
        req.check_match_stop_str_prefix.return_value = False
        self.assertEqual(self.emit(req).output_ids, [[10, 11]])

    def test_stop_prefix_blocks_first_output_even_at_batch_boundary(self):
        req = _FakeReq("test", list(range(50)))
        req.check_match_stop_str_prefix = Mock(return_value=True)
        self.assertIsNone(self.emit(req))

    def test_finished_request_flushes_even_with_stop_prefix(self):
        req = _FakeReq("test", [10], finished=True)
        req.check_match_stop_str_prefix = Mock(return_value=True)
        self.assertEqual(self.emit(req).output_ids, [[10]])
        req.check_match_stop_str_prefix.assert_not_called()

    def test_streaming_interval_and_stop_prefix_are_preserved(self):
        req = _FakeReq("test", [10])
        req.stream = True
        req.sampling_params.stream_interval = 3
        req.check_match_stop_str_prefix = Mock(return_value=True)
        self.assertIsNone(self.emit(req))
        req.check_match_stop_str_prefix.return_value = False
        self.assertEqual(self.emit(req).output_ids, [[10]])
        req.output_ids = req.output_ids_through_stop = [10, 11]
        self.assertIsNone(self.emit(req))
        req.output_ids = req.output_ids_through_stop = [10, 11, 12, 13]
        self.assertEqual(self.emit(req).output_ids, [[11, 12, 13]])

    def test_beam_candidates_remain_buffered(self):
        req = _FakeReq("test", [10])
        req.beam_group = object()
        for is_leader in (False, True):
            req.is_beam_leader = is_leader
            self.assertIsNone(self.emit(req))
            self.assertEqual(req.send_token_offset, 0)

    def test_ttft_includes_post_generation_delivery_time(self):
        # Generation at 12, followed by IPC and detok until arrival at 15.
        api = APIServerReqTimeStats(
            created_time=10, first_token_time=15, last_time=15, finished_time=30
        )
        self.assertEqual(api.get_first_token_latency(), 5)
        self.assertEqual(api.get_decode_latency(), 15)
        self.assertEqual(api.get_e2e_latency(), 20)
        self.assertEqual(
            api.get_first_token_latency() + api.get_decode_latency(),
            api.get_e2e_latency(),
        )


class TestNonstreamResponse(unittest.IsolatedAsyncioTestCase):
    async def test_first_internal_output_does_not_yield_to_client(self):
        manager = SimpleNamespace(
            incremental_streaming_output=False,
            request_logger=Mock(),
            request_metrics_exporter_manager=SimpleNamespace(
                exporter_enabled=lambda: False
            ),
        )
        obj = SimpleNamespace(rid="test", stream=False)
        state = SimpleNamespace(
            event=asyncio.Event(),
            out_list=[{"text": None, "meta_info": {}}],
            finished=False,
            time_stats=SimpleNamespace(response_sent_to_client_time=1),
        )
        state.event.set()
        response = TokenizerManager._stream_one_response(manager, obj, state)
        pending = asyncio.create_task(response.__anext__())
        try:

            async def wait_until_consumed():
                while state.event.is_set():
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_until_consumed(), timeout=1)
            self.assertFalse(pending.done())
            final = {"text": "complete output", "meta_info": {}}
            state.out_list.append(final)
            state.finished = True
            state.event.set()
            self.assertEqual(await asyncio.wait_for(pending, timeout=1), final)
            with self.assertRaises(StopAsyncIteration):
                await response.__anext__()
        finally:
            if not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await response.aclose()


if __name__ == "__main__":
    unittest.main()
