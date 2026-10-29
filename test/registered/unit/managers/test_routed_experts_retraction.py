"""Routing provenance must outlive the KV slots used during generation."""

from array import array
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel, published_topology

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers import schedule_batch
from sglang.srt.managers.io_struct import PauseGenerationReqInput
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.state_capturer.base import BaseTopkCapturer, TopkCaptureOutput
from sglang.srt.state_capturer.routed_experts import set_global_experts_capturer

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture
def capture():
    # Exercise the real get_topk and finalize without allocating CUDA buffers.
    capturer = BaseTopkCapturer.__new__(BaseTopkCapturer)
    capturer.num_layers = 2
    capturer.topk_size = 2
    capturer.host_cache = SimpleNamespace(
        buffer=torch.arange(64 * 4, dtype=torch.int32).reshape(64, 2, 2)
    )
    # Request positions deliberately differ from their KV slots.
    pool = SimpleNamespace(req_to_token=torch.arange(63, -1, -1).reshape(1, -1))
    with published_topology():
        set_global_experts_capturer(capturer)
        yield capturer, pool


def initial_routes():
    # Slots 63 through 58, in request order, before any cache reuse.
    return torch.tensor(
        [
            [[252, 253], [254, 255]],
            [[248, 249], [250, 251]],
            [[244, 245], [246, 247]],
            [[240, 241], [242, 243]],
            [[236, 237], [238, 239]],
            [[232, 233], [234, 235]],
        ],
        dtype=torch.int32,
    )


def make_req(*, start=0, outputs=3, **kwargs):
    req = Req(
        rid="routing-history",
        origin_input_text="",
        origin_input_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_new_tokens=100),
        return_logprob=True,
        return_routed_experts=kwargs.pop("return_routed_experts", True),
        routed_experts_start_len=start,
        **kwargs,
    )
    req.kv.req_pool_idx = 0
    req.output_ids.extend(range(20, 20 + outputs))
    req.logprob.output_token_logprobs_val.extend([-0.25] * outputs)
    return req


def collect(req, pool):
    processor = SchedulerBatchResultProcessor.__new__(SchedulerBatchResultProcessor)
    object.__setattr__(processor, "req_to_token_pool", pool)
    processor._maybe_collect_routed_experts(req)
    return req.routed_experts


def free_slots(req, capturer):
    # Another request immediately reuses both the KV slots and the request slot.
    capturer.host_cache.buffer.fill_(-777)
    req.kv.req_pool_idx = None


def retract(req, capturer, pool):
    with (
        patch.object(
            schedule_batch,
            "release_kv_cache",
            side_effect=lambda req, *args, **kwargs: free_slots(req, capturer),
        ),
        patch.object(schedule_batch, "evict_from_tree_cache"),
    ):
        schedule_batch.release_req(
            req=req,
            remaing_req_count=0,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            hisparse_coordinator=None,
            offload_kv=False,
        )


@pytest.mark.parametrize("start", [0, 2, 4])
def test_reprefill_cannot_overwrite_routes_for_accepted_tokens(capture, start):
    capturer, pool = capture
    req = make_req(start=start)
    output_ids = list(req.output_ids)
    logprobs = list(req.logprob.output_token_logprobs_val)

    retract(req, capturer, pool)
    assert list(req.output_ids) == output_ids
    assert req.logprob.output_token_logprobs_val == logprobs
    assert req.retraction_count == 1
    assert req.routed_experts is None

    # Re-prefill under a new weight version recomputes every prefix route.
    req.kv.req_pool_idx = 0
    capturer.host_cache.buffer.fill_(999)
    req.output_ids.append(30)
    req.logprob.output_token_logprobs_val.append(-0.5)
    expected = torch.cat(
        [initial_routes()[start:], torch.full((1, 2, 2), 999, dtype=torch.int32)]
    )
    torch.testing.assert_close(collect(req, pool), expected)


def test_multiple_retractions_append_only_new_rows(capture):
    capturer, pool = capture
    req = make_req()
    retract(req, capturer, pool)
    req.kv.req_pool_idx = 0
    capturer.host_cache.buffer.fill_(500)
    req.output_ids.extend([31, 32])
    retract(req, capturer, pool)

    # Retract again halfway through rebuilding the prefix. Only the first slot
    # has been computed; gathering the remaining slots would read stale data.
    req.kv.req_pool_idx = 0
    req.kv.kv_committed_len = 1
    with patch.object(capturer, "get_topk", side_effect=AssertionError("bad read")):
        retract(req, capturer, pool)

    req.kv.req_pool_idx = 0
    capturer.host_cache.buffer.fill_(600)
    req.output_ids.append(33)
    expected = torch.cat(
        [
            initial_routes(),
            torch.full((2, 2, 2), 500, dtype=torch.int32),
            torch.full((1, 2, 2), 600, dtype=torch.int32),
        ]
    )
    torch.testing.assert_close(collect(req, pool), expected)


@pytest.mark.parametrize("start,outputs", [(0, 0), (2, 0), (4, 1)])
def test_no_uncomputed_or_empty_prefix_gather(capture, start, outputs):
    capturer, pool = capture
    req = make_req(start=start, outputs=outputs)
    req.kv.kv_committed_len = 1
    with patch.object(capturer, "get_topk", side_effect=AssertionError("bad read")):
        retract(req, capturer, pool)
    req.kv.req_pool_idx = 0
    capturer.host_cache.buffer.fill_(888)
    if outputs == 0:
        req.output_ids.append(20)
    torch.testing.assert_close(
        collect(req, pool), torch.full((4 - start, 2, 2), 888, dtype=torch.int32)
    )


@pytest.mark.parametrize("finished_len", [0, 1, 2])
def test_stop_truncation_crops_saved_history(capture, finished_len):
    capturer, pool = capture
    req = make_req()
    retract(req, capturer, pool)
    req.kv.req_pool_idx = 0
    req.finished_len = finished_len
    torch.testing.assert_close(
        collect(req, pool), initial_routes()[: 4 + finished_len - 1]
    )


def test_non_routing_requests_do_not_snapshot(capture):
    capturer, pool = capture
    req = make_req(return_routed_experts=False)
    with patch.object(capturer, "get_topk", side_effect=AssertionError("bad read")):
        retract(req, capturer, pool)
        assert collect(req, pool) is None


def test_no_retraction_preserves_existing_response_path(capture):
    _, pool = capture
    req = make_req()
    torch.testing.assert_close(collect(req, pool), initial_routes())


def test_embedding_restart_discards_routes_with_output_ids(capture):
    capturer, pool = capture
    req = make_req(input_embeds=[[0.5]])
    with patch.object(capturer, "get_topk", side_effect=AssertionError("bad read")):
        retract(req, capturer, pool)
    assert req.output_ids == array("q")
    req.kv.req_pool_idx = 0
    capturer.host_cache.buffer.fill_(888)
    req.output_ids.append(30)
    torch.testing.assert_close(
        collect(req, pool), torch.full((4, 2, 2), 888, dtype=torch.int32)
    )


def test_short_capture_fails_before_freeing_slots(capture):
    capturer, pool = capture
    req = make_req()
    pool.req_to_token = pool.req_to_token[:, :1]
    with pytest.raises(RuntimeError, match="expected 6 rows, got 1"):
        retract(req, capturer, pool)
    assert req.kv.req_pool_idx == 0
    assert req.retraction_count == 0


def test_overlap_weight_pause_finalizes_routes_before_snapshot(capture):
    capturer, pool = capture
    req = make_req(outputs=1)
    result = TopkCaptureOutput(
        out_cache_loc=pool.req_to_token[0, 4:5],
        topk=torch.full((1, 2, 2), 700, dtype=torch.int32),
        host_cache=capturer.host_cache,
    )

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._engine_paused = False
    scheduler.enable_overlap = True
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.running_batch = ScheduleBatch(reqs=[req])
    scheduler.last_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    scheduler.chunked_req = None
    scheduler.result_queue = deque([(scheduler.last_batch, result)])
    scheduler.req_to_token_pool = pool
    scheduler.token_to_kv_pool_allocator = Mock()
    scheduler.tree_cache = Mock()
    scheduler.hisparse_coordinator = None
    scheduler.decode_offload_manager = None
    scheduler.metrics_reporter = Mock(current_scheduler_metrics_enabled=False)
    scheduler.kv_events_publisher = Mock()
    scheduler._add_request_to_queue = Mock()

    def process_pending(batch, capture_result):
        # Pending result completion publishes routes and accepts the next token.
        capture_result.finalize()
        req.output_ids.append(21)

    scheduler.process_batch_result = Mock(side_effect=process_pending)
    with (
        patch.object(
            schedule_batch,
            "release_kv_cache",
            side_effect=lambda req, *args, **kwargs: free_slots(req, capturer),
        ),
        patch.object(schedule_batch, "evict_from_tree_cache"),
    ):
        scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))
    assert not scheduler.result_queue
    scheduler._add_request_to_queue.assert_called_once_with(req)
    req.kv.req_pool_idx = 0
    capturer.host_cache.buffer.fill_(999)
    expected = torch.cat(
        [initial_routes()[:4], torch.full((1, 2, 2), 700, dtype=torch.int32)]
    )
    torch.testing.assert_close(collect(req, pool), expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
