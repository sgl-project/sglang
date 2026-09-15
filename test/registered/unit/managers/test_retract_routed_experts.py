"""R3 must return sampling-time routes, not routes from a later re-prefill."""

import unittest
from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import schedule_batch
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, release_req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    extract_routed_experts_from_meta_info,
    set_global_experts_capturer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestRetractRoutedExperts(CustomTestCase):
    def setUp(self) -> None:
        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        self.capturer = object.__new__(RoutedExpertsCapturer)
        self.capturer.host_cache = SimpleNamespace(
            buffer=torch.full((64, 2, 2), -1, dtype=torch.int32)
        )
        self.pool = SimpleNamespace(
            req_to_token=torch.full((1, 16), 999, dtype=torch.int64)
        )
        set_global_experts_capturer(self.capturer)
        self.processor = SimpleNamespace(req_to_token_pool=self.pool)

    def _req(self, *, start: int = 0, opted_in: bool = True) -> Req:
        req = Req(
            rid="r3-retract",
            origin_input_text=None,
            origin_input_ids=array("q", [1, 2, 3, 4]),
            sampling_params=SamplingParams(max_new_tokens=8),
            return_logprob=True,
            return_routed_experts=opted_in,
            routed_experts_start_len=start,
        )
        req.output_ids.extend([10, 11])
        req.logprob.output_token_logprobs_val = [-0.1, -0.2]
        return req

    def _prefill(
        self, req: Req, version: int, *, rows: int | None = None
    ) -> torch.Tensor:
        """Publish distinct routes, using different pool slots after each retract."""
        if rows is None:
            rows = req.seqlen - 1
        indices = torch.arange(version * 16, version * 16 + rows)
        self.pool.req_to_token.fill_(999)
        self.pool.req_to_token[0, :rows] = indices
        req.kv = ReqKvInfo(req_pool_idx=0, kv_committed_len=rows, kv_allocated_len=rows)
        routes = (
            torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 2, 2) + version * 32
        )
        TopkCaptureOutput(
            out_cache_loc=indices,
            topk=routes,
            host_cache=self.capturer.host_cache,
        ).finalize()
        return routes

    def _release_cache(self, request: Req, *_args, **_kwargs) -> None:
        # Simulate immediate reuse of every freed slot. The snapshot must
        # happen before release and own its data, not alias the pool.
        self.capturer.host_cache.buffer.fill_(-1)
        self.pool.req_to_token.fill_(999)
        request.kv = ReqKvInfo()

    def _retract(self, req: Req) -> None:
        with (
            patch.object(
                schedule_batch, "release_kv_cache", side_effect=self._release_cache
            ),
            patch.object(schedule_batch, "evict_from_tree_cache"),
        ):
            self.assertTrue(
                release_req(
                    req=req,
                    remaing_req_count=0,
                    req_to_token_pool=self.pool,
                    token_to_kv_pool_allocator=None,
                    tree_cache=None,
                    hisparse_coordinator=None,
                    offload_kv=False,
                )
            )

    def _collect(self, req: Req) -> torch.Tensor | None:
        SchedulerBatchResultProcessor._maybe_collect_routed_experts(self.processor, req)
        return req.routed_experts

    def test_sampling_routes_survive_reprefill_and_wire_encoding(self) -> None:
        for start in (0, 3, 4, 5, 10):
            with self.subTest(start=start):
                req = self._req(start=start)
                original = self._prefill(req, 0)
                self._retract(req)
                self.assertIsNone(req.routed_experts)  # No partial stream payload.
                self.assertEqual(req.output_ids, array("q", [10, 11]))
                self.assertEqual(req.logprob.output_token_logprobs_val, [-0.1, -0.2])

                req.output_ids.append(12)
                recomputed = self._prefill(req, 1)
                expected = torch.cat((original, recomputed[len(original) :]))[start:]
                actual = self._collect(req)
                torch.testing.assert_close(actual, expected)

                encoded = DetokenizerManager._b64_encode_per_request([actual])[0]
                decoded = extract_routed_experts_from_meta_info(
                    {"meta_info": {"routed_experts": encoded}}
                ).reshape(expected.shape)
                torch.testing.assert_close(torch.from_numpy(decoded.copy()), expected)

    def test_pause_retract_preserves_the_drained_overlap_result(self) -> None:
        req = self._req()
        self._prefill(req, 0)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE, reqs=[req])

        def finish_pending_result(_batch, _result) -> None:
            req.output_ids.append(12)
            req.logprob.output_token_logprobs_val.append(-0.3)
            self._prefill(req, 0)

        scheduler = SimpleNamespace(
            enable_overlap=True,
            last_batch=batch,
            running_batch=SimpleNamespace(reqs=[req], batch_is_full=False),
            result_queue=deque([(batch, None)]),
            process_batch_result=finish_pending_result,
            chunked_req=None,
            disaggregation_mode=DisaggregationMode.NULL,
            req_to_token_pool=self.pool,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            hisparse_coordinator=None,
            _add_request_to_queue=Mock(),
            metrics_reporter=SimpleNamespace(current_scheduler_metrics_enabled=False),
            kv_events_publisher=Mock(),
        )
        with (
            patch.object(
                schedule_batch, "release_kv_cache", side_effect=self._release_cache
            ),
            patch.object(schedule_batch, "evict_from_tree_cache"),
        ):
            Scheduler.pause_generation(
                scheduler, PauseGenerationReqInput(mode="retract")
            )

        scheduler._add_request_to_queue.assert_called_once_with(req)
        self.assertEqual(req.logprob.output_token_logprobs_val, [-0.1, -0.2, -0.3])
        self.assertEqual(len(scheduler.result_queue), 0)
        req.output_ids.append(13)
        recomputed = self._prefill(req, 1)
        original = torch.arange(6 * 4, dtype=torch.int32).reshape(6, 2, 2)
        expected = torch.cat((original, recomputed[6:]))
        torch.testing.assert_close(self._collect(req), expected)

    def test_repeated_retracts_keep_each_sampling_version(self) -> None:
        req = self._req()
        original = self._prefill(req, 0)
        self._retract(req)
        req.output_ids.extend([12, 13])
        second = self._prefill(req, 1)
        self._retract(req)
        req.output_ids.append(14)
        third = self._prefill(req, 2)
        expected = torch.cat((original, second[len(original) :], third[len(second) :]))
        torch.testing.assert_close(self._collect(req), expected)

    def test_retract_during_reprefill_does_not_read_uncomputed_rows(self) -> None:
        req = self._req()
        original = self._prefill(req, 0)
        self._retract(req)
        self._prefill(req, 1, rows=2)  # The remaining mapping is deliberately invalid.
        self._retract(req)
        req.output_ids.append(12)
        final = self._prefill(req, 2)
        expected = torch.cat((original, final[len(original) :]))
        torch.testing.assert_close(self._collect(req), expected)

    def test_initial_chunked_prefill_has_no_sampling_routes_to_preserve(self) -> None:
        req = self._req()
        req.output_ids = array("q")
        self._prefill(req, 0, rows=2)
        self._retract(req)
        req.output_ids.append(10)
        expected = self._prefill(req, 1)
        torch.testing.assert_close(self._collect(req), expected)

    def test_stop_before_saved_prefix_end_trims_routes(self) -> None:
        req = self._req(start=3)
        original = self._prefill(req, 0)
        self._retract(req)
        self._prefill(req, 1)
        req.finished_len = 1
        torch.testing.assert_close(self._collect(req), original[3:4])

    def test_discarded_input_embeds_output_discards_its_routes(self) -> None:
        req = self._req()
        req.input_embeds = [[0.0]] * len(req.origin_input_ids)
        self._prefill(req, 0)
        self._retract(req)
        self.assertEqual(len(req.output_ids), 0)
        req.output_ids.append(10)
        expected = self._prefill(req, 1)
        torch.testing.assert_close(self._collect(req), expected)

    def test_non_retracted_request_is_unchanged(self) -> None:
        req = self._req(start=3)
        expected = self._prefill(req, 0)[3:]
        torch.testing.assert_close(self._collect(req), expected)

    def test_non_opted_in_request_never_gathers_routes(self) -> None:
        req = self._req(opted_in=False)
        self._prefill(req, 0)
        with patch.object(self.capturer, "get_topk") as gather:
            self._retract(req)
            self._prefill(req, 1)
            self.assertIsNone(self._collect(req))
            gather.assert_not_called()

    def test_no_capturer_is_a_noop(self) -> None:
        req = self._req()
        self._prefill(req, 0)
        set_global_experts_capturer(None)
        self._retract(req)
        self.assertIsNone(self._collect(req))


if __name__ == "__main__":
    unittest.main()
