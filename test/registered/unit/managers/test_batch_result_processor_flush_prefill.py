"""SchedulerBatchResultProcessor._flush_prefill: a generation prefill's per-request
KV calls are queued and run in order, before free_group_end and the send. A subclass
that sends first gets the same KV calls, in the same order and one free group.
A decode step sends first and then releases its finished requests, in the same free
group."""

import unittest
from array import array
from dataclasses import MISSING, fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    SamplingMaskStatus,
)
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BRP = "sglang.srt.managers.scheduler_components.batch_result_processor"
RETIRER = "sglang.srt.managers.scheduler_components.request_retirer"
PROMPT = 12


class _StreamFirst(SchedulerBatchResultProcessor):
    __slots__ = ()

    def _flush_prefill(self, batch, stream_reqs, kv_ops, skip_stream_req):
        self.output_streamer.stream_output(
            stream_reqs, batch.return_logprob, skip_stream_req
        )
        for op in kv_ops:
            op()
        self.token_to_kv_pool_allocator.free_group_end()


def _req(rid, max_new_tokens=8, *, mask=False):
    params = SamplingParams(max_new_tokens=max_new_tokens)
    params.normalize(None)
    req = Req(
        rid,
        "",
        array("q", range(100, 100 + PROMPT)),
        params,
        return_sampling_mask=mask,
    )
    req.kv.req_pool_idx = 0
    req.kv.kv_committed_len = req.kv.kv_allocated_len = PROMPT
    return req


def _batch():
    """Finished (released), running (checkpointed), middle chunk (neither), and a
    sampling-mask abort (released without the checkpoint)."""
    reqs = [
        _req("r0", 1),
        _req("r1"),
        _req("r2"),
        _req("r3", mask=True),
    ]
    return ScheduleBatch(
        reqs=reqs, spec_algorithm=SpeculativeAlgorithm.NONE, chunked_req=reqs[2]
    )


def _decode_req(rid, max_new_tokens, to_finish=None):
    req = _req(rid, max_new_tokens)
    req.output_ids = [7]
    req.to_finish = to_finish
    return req


def _run(cls, *, hisparse=False, decode_reqs=None, auxiliary=False):
    """One prefill result (or a decode step over ``decode_reqs``) through ``cls``;
    returns the log of KV calls, free group, auxiliary output and send."""
    log = []
    allocator = Mock()
    allocator.free_group_begin.side_effect = lambda: log.append("free_group_begin")
    allocator.free_group_end.side_effect = lambda: log.append("free_group_end")
    streamer = Mock()
    streamer.stream_output.side_effect = lambda *args: log.append("send")
    hisparse_coordinator = (
        SimpleNamespace(
            release_request=lambda req: log.append(("hisparse_finished", req.rid)),
            admit_request_into_staging=lambda req: log.append(("admit", req.rid)),
        )
        if hisparse
        else None
    )
    kwargs = dict.fromkeys(
        f.name for f in fields(SchedulerBatchResultProcessor) if f.default is MISSING
    )
    kwargs.update(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        model_config=SimpleNamespace(
            self_opened_think_start_ids=None, think_end_ids=None
        ),
        token_to_kv_pool_allocator=allocator,
        hisparse_coordinator=hisparse_coordinator,
        request_retirer=release_only_retirer(
            tree_cache=Mock(), hisparse_coordinator=hisparse_coordinator
        ),
        output_streamer=streamer,
        still_in_batch=lambda req: True,
        beam_coordinator=Mock(),
        metrics_reporter=Mock(num_generated_tokens=0, forward_ct_decode=0),
    )
    if decode_reqs is None:
        batch = _batch()
        statuses = [SamplingMaskStatus.OVERFLOW] * len(batch.reqs)
    else:
        batch = ScheduleBatch(
            reqs=decode_reqs, spec_algorithm=SpeculativeAlgorithm.NONE
        )
        statuses = None
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None, next_token_sampling_mask_status=statuses
        ),
        next_token_ids=torch.tensor([7 + i for i in range(len(batch.reqs))]),
    )
    if auxiliary:
        result.auxiliary_host_output = Mock()
        result.auxiliary_host_output.consume.side_effect = lambda *_: log.append("aux")
    memory = SimpleNamespace(enable_hisparse=hisparse)
    with (
        published_topology(),
        patch(
            f"{RETIRER}.release_kv_cache",
            lambda req, tree, *, checkpoint: log.append(
                ("release", req.rid, checkpoint)
            ),
        ),
        patch(
            f"{BRP}.checkpoint_kv_cache",
            lambda req, tree: log.append(("checkpoint", req.rid)),
        ),
        patch(f"{BRP}.get_memory", lambda: memory),
        patch.object(cls, "_mamba_prefix_cache_update"),
    ):
        if decode_reqs is None:
            cls(**kwargs).process_batch_result_prefill(batch, result)
        else:
            cls(**kwargs).process_batch_result_decode(batch, result)
    return log


SEND = "send"
KV_CALLS = [("release", "r0", True), ("checkpoint", "r1"), ("release", "r3", False)]


class TestFlushPrefill(CustomTestCase):
    def test_kv_calls_run_before_the_send(self):
        log = _run(SchedulerBatchResultProcessor)
        self.assertEqual(log, ["free_group_begin", *KV_CALLS, "free_group_end", SEND])

    def test_override_sends_before_the_kv_calls(self):
        log = _run(_StreamFirst)
        self.assertEqual(log, ["free_group_begin", SEND, *KV_CALLS, "free_group_end"])

    def test_hisparse_calls_are_queued_in_row_order(self):
        # The order the loop used to run them in, each release_request just before
        # its row's release.
        kv_calls = [
            ("hisparse_finished", "r0"),
            KV_CALLS[0],
            KV_CALLS[1],
            ("admit", "r1"),
            ("hisparse_finished", "r3"),
            KV_CALLS[2],
        ]
        self.assertEqual(
            _run(SchedulerBatchResultProcessor, hisparse=True),
            ["free_group_begin", *kv_calls, "free_group_end", SEND],
        )
        self.assertEqual(
            _run(_StreamFirst, hisparse=True),
            ["free_group_begin", SEND, *kv_calls, "free_group_end"],
        )

    def test_decode_sends_before_the_finished_release(self):
        """The final output does not wait for the prefix-cache insert, which still
        runs in the step's free group, before the scheduler's next iteration."""
        reqs = [_decode_req("r0", 2), _decode_req("r1", 8)]
        self.assertEqual(
            _run(SchedulerBatchResultProcessor, decode_reqs=reqs),
            ["free_group_begin", SEND, ("release", "r0", True), "free_group_end"],
        )

    def test_decode_keeps_release_first_where_the_send_depends_on_it(self):
        # A requested finish (here an abort) leaves finished_len for the send to
        # fill, and the release reads it; a consumer of auxiliary output may
        # defer_response only a released request.
        for to_finish, auxiliary in ((FINISH_ABORT(), False), (None, True)):
            with self.subTest(auxiliary=auxiliary):
                log = _run(
                    SchedulerBatchResultProcessor,
                    decode_reqs=[_decode_req("r0", 2, to_finish)],
                    auxiliary=auxiliary,
                )
                self.assertEqual(
                    log,
                    ["free_group_begin", ("release", "r0", True)]
                    + ["aux"] * auxiliary
                    + [SEND, "free_group_end"],
                )


if __name__ == "__main__":
    unittest.main()
