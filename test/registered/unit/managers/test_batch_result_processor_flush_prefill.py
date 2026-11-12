"""SchedulerBatchResultProcessor._flush_prefill: a generation prefill's per-request
KV calls are queued and run in order, before free_group_end and the send. A subclass
that sends first gets the same KV calls, in the same order and one free group."""

import unittest
from array import array
from contextlib import nullcontext
from dataclasses import MISSING, fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.layers.logits_processor import (
    LogitsProcessorOutput,
    SamplingMaskStatus,
)
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase, published_topology
from sglang.test.unified_allocator_fixtures import (
    build_mamba_cache,
    setup_allocator_context,
)

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


def _processor_kwargs(**kwargs):
    """The processor's required fields, None but for ``kwargs``."""
    fixed = dict.fromkeys(
        f.name for f in fields(SchedulerBatchResultProcessor) if f.default is MISSING
    )
    fixed.update(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        model_config=SimpleNamespace(
            self_opened_think_start_ids=None, think_end_ids=None
        ),
        still_in_batch=lambda req: True,
    )
    fixed.update(kwargs)
    return fixed


def _result(batch):
    return GenerationBatchResult(
        logits_output=LogitsProcessorOutput(
            next_token_logits=None,
            next_token_sampling_mask_status=[SamplingMaskStatus.OVERFLOW]
            * len(batch.reqs),
        ),
        next_token_ids=torch.tensor([7 + i for i in range(len(batch.reqs))]),
    )


def _run(cls, *, hisparse=False):
    """One prefill result through ``cls``; returns the log of KV calls, free group
    and send."""
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
    kwargs = _processor_kwargs(
        token_to_kv_pool_allocator=allocator,
        tree_cache=SimpleNamespace(
            lease_checkpoint_state_slots=lambda reqs: nullcontext()
        ),
        hisparse_coordinator=hisparse_coordinator,
        request_retirer=release_only_retirer(
            tree_cache=Mock(), hisparse_coordinator=hisparse_coordinator
        ),
        output_streamer=streamer,
    )
    batch = _batch()
    result = _result(batch)
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
    ):
        cls(**kwargs).process_batch_result_prefill(batch, result)
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


def _running_req(cache, start):
    """A request whose prefill of ``PROMPT`` tokens ran and that decodes on."""
    tokens = array("q", range(start, start + PROMPT))
    params = SamplingParams(max_new_tokens=8)
    params.normalize(None)
    req = Req(f"r{start}", "", tokens, params)
    pool = cache.req_to_token_pool
    allocator = cache.token_to_kv_pool_allocator
    acquire_rows(pool, allocator.state, [req])
    pool.write((req.kv.req_pool_idx, slice(0, PROMPT)), allocator.alloc(PROMPT))
    req.kv.kv_committed_len = req.kv.kv_allocated_len = PROMPT
    req.full_untruncated_fill_ids = array("q", tokens)
    req.extend_end = PROMPT
    req.last_node = cache.root_node_handle()
    return req


def _checkpoint_batch(*, lease, spare_slots=None):
    """Process one prefill result of two requests that decode on over a
    unified state pool, so each is checkpointed into the tree; ``spare_slots``
    leaves only that many state slots free. Returns the state band's takes and,
    per request, what the tree holds for it."""
    setup_allocator_context()
    _, allocator, cache = build_mamba_cache(page_size=1)
    reqs = [_running_req(cache, 100 * (i + 1)) for i in range(2)]
    slots = allocator.state_slots
    if spare_slots is not None:
        held = []
        while (slot := slots.alloc(1)) is not None:
            held.append(slot)
        slots.free(torch.cat(held[:spare_slots]))
    processor = SchedulerBatchResultProcessor(
        **_processor_kwargs(
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            request_retirer=release_only_retirer(tree_cache=cache),
            output_streamer=Mock(),
        )
    )
    batch = ScheduleBatch(reqs=reqs, spec_algorithm=SpeculativeAlgorithm.NONE)
    per_request = (
        nullcontext()
        if lease
        else patch.object(
            UnifiedRadixCache,
            "lease_checkpoint_state_slots",
            lambda self, reqs: nullcontext(),
        )
    )
    with (
        per_request,
        patch.object(slots, "_take", wraps=slots._take) as take,
        patch(f"{BRP}.get_memory", lambda: SimpleNamespace(enable_hisparse=False)),
    ):
        processor.process_batch_result_prefill(batch, _result(batch))
    tree = []
    for req in reqs:
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(req.origin_input_ids))
        )
        value = cache.tree_core.get_component_device_value(
            match.last_device_node, ComponentType.MAMBA
        )
        tree.append(
            (
                req.kv.cache_protected_len,
                match.device_prefix_len,
                None if value is None else value.tolist(),
            )
        )
    return [c.args for c in take.call_args_list], tree, slots.available_size()


class TestCheckpointStateSlots(CustomTestCase):
    def test_a_batch_of_checkpoints_takes_its_state_slots_once(self):
        """Each request a prefill checkpoints takes a state slot for the tree's
        copy of its state; a batch's slots come from one take of the state band,
        and the tree ends as with one take per request, the same slots
        included."""
        takes, tree, available = _checkpoint_batch(lease=True)
        self.assertEqual(takes, [(2,)])
        per_request = _checkpoint_batch(lease=False)
        self.assertEqual(per_request[0], [(1,)] * 2)
        self.assertEqual((tree, available), per_request[1:])
        self.assertTrue(all(value is not None for _, _, value in tree))

    def test_short_state_slots_take_one_by_one_as_before(self):
        """With a slot for one of the two, nothing is taken ahead: each
        request takes its own, and the one left short is checkpointed without
        a state, as with one take per request."""
        takes, tree, available = _checkpoint_batch(lease=True, spare_slots=1)
        self.assertEqual(
            (takes, tree, available),
            _checkpoint_batch(lease=False, spare_slots=1),
        )
        self.assertEqual(sum(value is not None for _, _, value in tree), 1)


if __name__ == "__main__":
    unittest.main()
