"""The PD owners' request lists: each move from list to list, and an abort
from each list."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.req_state import DisaggReqState
from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.request_lists import CoreLists
from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.disagg_owner_shells import (
    decode_owner_shell,
    prefill_owner_shell,
    result_processor_shell,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PREFILL_MODULE = "sglang.srt.disaggregation.prefill"


def _names(owner):
    return {lst.name: [req.rid for req in lst.reqs] for lst in owner.lists()}


def _prefill_req(rid):
    return SimpleNamespace(
        rid=rid,
        bootstrap_host=FAKE_BOOTSTRAP_HOST,
        bootstrap_room=1,
        cache_request_handle=object(),
        disagg=DisaggReqState(metadata_buffer_index=3),
        kv=SimpleNamespace(holds_kv=False, holds_mamba=False),
        to_finish=None,
        finished_reason=None,
        discard_output_reason=None,
        return_logprob=False,
        return_sampling_mask=False,
        return_routed_experts=False,
        grammar=None,
        output_ids=[],
        time_stats=MagicMock(),
    )


def _sender(poll=KVPoll.Transferring):
    return Mock(poll=Mock(return_value=poll))


class TestDisaggPrefillLists(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_allocation_policy="early"
        )
        override.install()
        self.addCleanup(override.restore)
        self.owner = prefill_owner_shell(
            batch_result_processor=result_processor_shell(),
            spec_algorithm=SimpleNamespace(is_eagle=lambda: False),
            tree_cache=Mock(),
            req_to_metadata_buffer_idx_allocator=Mock(),
            checkpoint=Mock(),
            send_kv_chunk=Mock(),
            output_streamer=Mock(),
            metrics_sink=SimpleNamespace(enable_metrics=False),
            request_retirer=Mock(),
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
        )
        for target in ("checkpoint_kv_cache",):
            patcher = patch(f"{PREFILL_MODULE}.{target}")
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch(
            f"{PREFILL_MODULE}.poll_and_all_reduce_attn_cp_tp_group",
            side_effect=lambda senders, *groups: [s.poll() for s in senders],
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def bootstrapped(self, rid):
        """A request through the bootstrap list into the core waiting list."""
        owner = self.owner
        req = _prefill_req(rid)
        owner.kv_senders[rid] = _sender()
        owner.bootstrap_queue.queue.append(req)
        self.assertEqual(_names(owner)["bootstrap"], [rid])

        def pop_bootstrapped():
            reqs, owner.bootstrap_queue.queue = owner.bootstrap_queue.queue, []
            return reqs

        owner.bootstrap_queue.pop_bootstrapped = pop_bootstrapped
        self.assertEqual(owner.intake(), [req])
        self.assertEqual(_names(owner)["bootstrap"], [])
        return req

    def committed(self, req):
        """A launched request, prefilling until its commit puts it to transfer."""
        owner = self.owner
        owner.on_launch([req])
        self.assertEqual(_names(owner)["prefilling"], [req.rid])
        batch = SimpleNamespace(
            reqs=[req],
            chunked_req=None,
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
            extend_lens=None,
            extend_logprob_start_lens=None,
            return_logprob=False,
        )
        result = GenerationBatchResult(next_token_ids=torch.tensor([11]))
        self.assertEqual(owner.commit_extend(batch, result), [])
        self.assertEqual(_names(owner)["prefilling"], [])
        self.assertEqual(_names(owner)["transfer"], [req.rid])
        owner.send_kv_chunk.assert_called_with(req, last_chunk=True)

    def test_bootstrap_to_transfer_to_release(self):
        owner = self.owner
        req = self.bootstrapped("r")
        self.committed(req)

        owner.after_commit([])
        self.assertEqual(_names(owner)["transfer"], [req.rid])
        self.assertIn(req.rid, owner.kv_senders)

        owner.kv_senders[req.rid].poll.return_value = KVPoll.Success
        owner.after_commit([])
        self.assertEqual(
            _names(owner), {"bootstrap": [], "prefilling": [], "transfer": []}
        )
        self.assertNotIn(req.rid, owner.kv_senders)
        owner.output_streamer.stream_output.assert_called_with([req], False, None)
        owner.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)

    @patch(f"{PREFILL_MODULE}.get_parallel", return_value=SimpleNamespace(pp_size=1))
    def test_abort_from_bootstrap_notifies_the_sender(self, _):
        owner = self.owner
        req = _prefill_req("r")
        other = _prefill_req("other")
        owner.bootstrap_queue.queue.extend([req, other])
        owner.kv_senders.update({"r": _sender(), "other": _sender()})
        (bootstrap,) = [lst for lst in owner.lists() if lst.name == "bootstrap"]

        bootstrap.abort(AbortReq(rid="r"))

        owner.request_retirer.abort_queued_attempt.assert_called_once_with(req)
        owner.kv_senders["r"].abort.assert_called_once_with()
        owner.kv_senders["other"].abort.assert_not_called()

    @patch(
        f"{PREFILL_MODULE}.get_parallel",
        return_value=SimpleNamespace(pp_size=1, tp_rank=0),
    )
    def test_abort_from_transfer_ends_in_a_failed_release(self, _):
        owner = self.owner
        req = self.bootstrapped("r")
        self.committed(req)
        (transfer,) = [lst for lst in owner.lists() if lst.name == "transfer"]

        transfer.abort(AbortReq(rid="r"))
        sender = owner.kv_senders["r"]
        sender.abort.assert_called_once_with()
        # The aborted sender polls Failed, and the commit releases the request.
        sender.poll.return_value = KVPoll.Failed
        sender.failure_exception.side_effect = RuntimeError("aborted")
        owner.after_commit([])

        self.assertEqual(_names(owner)["transfer"], [])
        self.assertNotIn("r", owner.kv_senders)

    def test_abort_from_waiting_drops_the_sender(self):
        owner = self.owner
        req = self.bootstrapped("r")
        sender = owner.kv_senders["r"]

        owner.release_aborted_waiting_req(req)

        sender.abort.assert_called_once_with()
        self.assertNotIn("r", owner.kv_senders)

    def test_abort_of_the_chunked_request_drops_the_sender(self):
        owner = self.owner
        req = self.bootstrapped("r")
        owner.pending_chunk_rids.add("r")
        sender = owner.kv_senders["r"]

        owner.release_aborted_chunked_req(req)

        sender.abort.assert_called_once_with()
        self.assertNotIn("r", owner.kv_senders)
        self.assertNotIn("r", owner.pending_chunk_rids)


def _decode_req(rid):
    return SimpleNamespace(
        rid=rid,
        finished_reason=None,
        last_node=object(),
        kv=SimpleNamespace(kv_committed_len=None),
        init_next_round_input=Mock(),
        time_stats=MagicMock(),
    )


class _PreallocQueue:
    """Preallocation pops the requests whose receiver polled WaitingForInput."""

    def __init__(self):
        self.queue = []
        self.retracted_queue = []
        self.held_rebootstrap_reqs = []
        self.resumable = []

    def resume_retracted_reqs(self, decoding):
        resumed = [r for r in self.retracted_queue if r.rid in self.resumable]
        self.retracted_queue = [r for r in self.retracted_queue if r not in resumed]
        return resumed

    def pop_preallocated(self, decoding, *, lora_ids_in_use):
        ready = [
            r for r in self.queue if r.kv_receiver.poll() == KVPoll.WaitingForInput
        ]
        self.queue = [r for r in self.queue if r not in ready]
        return ready, []


class _TransferQueue:
    """The transfer pops the requests whose receiver polled Success."""

    def __init__(self):
        self.queue = []
        self.resolve_deferred_releases = Mock(return_value=[])

    def extend(self, decode_reqs):
        self.queue.extend(decode_reqs)

    def pop_transferred(self, decoding):
        done = [r for r in self.queue if r.kv_receiver.poll() == KVPoll.Success]
        self.queue = [r for r in self.queue if r not in done]
        return [r.req for r in done]


class TestDisaggDecodeLists(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            disaggregation_decode_polling_interval=1,
            disaggregation_decode_host_receive_threshold=1,
            disaggregation_decode_enable_offload_kvcache=False,
            disaggregation_decode_enable_radix_cache=True,
        )
        override.install()
        self.addCleanup(override.restore)
        self.owner = decode_owner_shell(
            prealloc_queue=_PreallocQueue(),
            transfer_queue=_TransferQueue(),
            enable_decode_hicache=False,
            enable_hisparse=False,
            enable_priority_scheduling=False,
            scheduler_stage_metrics=None,
            polling_count=0,
            polling_interval=1,
            req_to_token_pool=SimpleNamespace(size=8),
            max_running_requests=8,
            core_lists=CoreLists(),
            tree_cache=Mock(),
            request_retirer=Mock(),
        )

    def preallocating(self, rid):
        req = _decode_req(rid)
        decode_req = SimpleNamespace(
            req=req, kv_receiver=Mock(poll=Mock(return_value=KVPoll.Bootstrapping))
        )
        self.owner.prealloc_queue.queue.append(decode_req)
        return decode_req

    def test_prealloc_to_transfer_to_ready_to_decoding(self):
        owner = self.owner
        decode_req = self.preallocating("r")
        req = decode_req.req
        self.assertEqual(_names(owner)["prealloc"], ["r"])

        owner.intake(lora_ids_in_use=set)
        self.assertEqual(_names(owner)["prealloc"], ["r"])

        decode_req.kv_receiver.poll.return_value = KVPoll.WaitingForInput
        owner.intake(lora_ids_in_use=set)
        self.assertEqual(_names(owner)["prealloc"], [])
        self.assertEqual(_names(owner)["transfer"], ["r"])

        decode_req.kv_receiver.poll.return_value = KVPoll.Success
        owner.intake(lora_ids_in_use=set)
        self.assertEqual(_names(owner)["transfer"], [])
        self.assertEqual(_names(owner)["ready"], ["r"])

        self.assertEqual(owner.pop_reqs_to_admit(ScheduleSlot()), [req])
        self.assertEqual(_names(owner)["ready"], [])
        req.init_next_round_input.assert_called_once_with(None)

    def test_admission_takes_only_the_free_rows(self):
        owner = self.owner
        owner.ready = [_decode_req("a"), _decode_req("b")]
        owner.max_running_requests = 2
        owner.core_lists.decoding = [object()]

        admitted = owner.pop_reqs_to_admit(ScheduleSlot())
        self.assertEqual([r.rid for r in admitted], ["a"])
        self.assertEqual(_names(owner)["ready"], ["b"])

    def test_retracted_resume_to_ready(self):
        owner = self.owner
        owner.prealloc_queue.retracted_queue = [_decode_req("a"), _decode_req("b")]
        owner.prealloc_queue.resumable = ["a"]

        owner.intake(lora_ids_in_use=set)

        self.assertEqual(_names(owner)["retracted"], ["b"])
        self.assertEqual(_names(owner)["ready"], ["a"])

    def abort(self, name, rid):
        (lst,) = [lst for lst in self.owner.lists() if lst.name == name]
        lst.abort(AbortReq(rid=rid))

    @patch(
        "sglang.srt.disaggregation.decode.get_parallel",
        return_value=SimpleNamespace(pp_size=1),
    )
    def test_abort_from_prealloc_aborts_the_receiver(self, _):
        decode_req = self.preallocating("r")
        other = self.preallocating("other")

        self.abort("prealloc", "r")

        decode_req.kv_receiver.abort.assert_called_once_with()
        other.kv_receiver.abort.assert_not_called()

    @patch("sglang.srt.disaggregation.decode.discard_kv_cache_backup")
    def test_abort_from_retracted(self, discard):
        owner = self.owner
        owner.prealloc_queue.retracted_queue = [_decode_req("r"), _decode_req("x")]

        self.abort("retracted", "r")

        self.assertEqual(_names(owner)["retracted"], ["x"])
        discard.assert_called_once()
        owner.request_retirer.send_abort_output.assert_called_once()

    def test_abort_from_held(self):
        owner = self.owner
        req = _decode_req("r")
        owner.prealloc_queue.held_rebootstrap_reqs = [req, _decode_req("x")]

        self.abort("held", "r")

        self.assertEqual(_names(owner)["held"], ["x"])
        owner.request_retirer.retire_aborted.assert_called_once_with(req)

    def test_abort_from_transfer_aborts_the_receiver(self):
        owner = self.owner
        decode_req = self.preallocating("r")
        decode_req.host_staged = False
        owner.prealloc_queue.queue = []
        owner.transfer_queue.queue = [decode_req]

        self.abort("transfer", "r")

        decode_req.kv_receiver.abort.assert_called_once_with()

    def test_abort_from_ready_releases_the_kv(self):
        owner = self.owner
        req = _decode_req("r")
        owner.ready = [req, _decode_req("x")]

        self.abort("ready", "r")

        self.assertEqual(_names(owner)["ready"], ["x"])
        owner.request_retirer.retire_aborted.assert_called_once_with(
            req, checkpoint=True
        )
        self.assertIsNotNone(req.finished_reason)


if __name__ == "__main__":
    unittest.main()
