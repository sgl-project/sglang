"""Finite VPP waves against queue transport and CPU model boundaries."""

import unittest
from array import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed.vpp_activation import VPPReceivedActivation
from sglang.srt.distributed.vpp_batch import VPPBatchLedger, VPPBatchState
from sglang.srt.distributed.vpp_layout import VPPLayout
from sglang.srt.distributed.vpp_schedule import VPPFixedSchedule
from sglang.srt.managers import scheduler_vpp_wave_mixin as vpp_mod
from sglang.srt.managers.scheduler_vpp_wave_mixin import SchedulerVPPMixin
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.runner.vpp_stage import execute_vpp_stage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _Stream:
    def wait_stream(self, stream):
        pass

    def wait_event(self, event):
        pass


class _Event:
    def __init__(self):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True


class _Receiver:
    def __init__(self, queue):
        self.queue = queue
        self.posted = False

    def start(self):
        if self.posted:
            raise RuntimeError("Receive leaked from the preceding wave")
        self.posted = True

    def take(self, *, post_next=True):
        if not self.posted:
            raise RuntimeError("No receive posted")
        result = self.queue.get(timeout=10)
        self.posted = post_next
        return VPPReceivedActivation(result)


class _Sends:
    def __init__(self, queue):
        self.queue = queue

    def acquire(self):
        return None

    def send(self, tensors, forward_event):
        if not forward_event.recorded:
            raise RuntimeError("Activation published before the forward fence")
        self.queue.put(tensors)

    def wait_for_reuse(self):
        return None


class _Worker:
    def __init__(self, rank, calls):
        self.rank = rank
        self.calls = calls
        self.model_runner = object()
        # Deliberately reused output storage; the scheduler must snapshot views.
        self.workspace = torch.empty(1)

    def set_hicache_consumer(self, index):
        pass

    def forward_batch_generation(
        self, batch, *, forward_batch, pp_proxy_tensors, vpp_stage_id
    ):
        self.calls.append((forward_batch.batch_seq, vpp_stage_id))
        value = (
            forward_batch.batch_seq
            if pp_proxy_tensors is None
            else pp_proxy_tensors["hidden_states"].item()
        )
        self.workspace.fill_(value + 1)
        if vpp_stage_id == forward_batch.final_stage:
            return SimpleNamespace(
                next_token_ids=self.workspace,
                pp_hidden_states_proxy_tensors=None,
            )
        return SimpleNamespace(
            pp_hidden_states_proxy_tensors=PPProxyTensors(
                {"hidden_states": self.workspace}
            )
        )


class TestSchedulerVPP(CustomTestCase):
    def test_prefix_match_respects_vpp_frontier_and_existing_caps(self):
        """A local cache hit cannot advance past the agreed pipeline frontier."""
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid="prefix",
            origin_input_text="",
            origin_input_ids=array("q", range(16)),
            sampling_params=SamplingParams(max_new_tokens=2),
        )
        self.assertEqual(req._compute_max_prefix_len(16), 15)
        for limit, expected in ((8, 8), (0, 0), (32, 15)):
            with self.subTest(limit=limit):
                req.vpp_prefix_limit = limit
                self.assertEqual(req._compute_max_prefix_len(16), expected)
        req.vpp_prefix_limit = 8
        req.return_logprob = True
        req.logprob_start_len = 4
        self.assertEqual(req._compute_max_prefix_len(16), 4)
        req.vpp_prefix_limit = None
        self.assertEqual(req._compute_max_prefix_len(16), 4)

    def test_bootstrap_consensus_holds_unconfirmed_requests_and_failure_wins(self):
        """Consensus success still needs local readiness; missing RIDs stay queued."""
        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.managers.scheduler_components.request_list import (
            IndexedReqList,
            ListKind,
            move,
        )
        from sglang.srt.sampling.sampling_params import SamplingParams

        for outcomes in (
            {},
            {"ready": False, "pending": False, "failed": True},
        ):
            with self.subTest(outcomes=outcomes):
                queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
                queue.scheduler_stage_metrics = None
                queue.pp_size = 2
                failed = []
                queue.scheduler = SimpleNamespace(
                    attn_cp_cpu_group=None,
                    attn_tp_cpu_group=None,
                    processed_tokens_counter=7,
                    handle_bootstrap_failure=failed.append,
                    token_to_kv_pool_allocator=SimpleNamespace(page_size=4),
                )
                queue.ensure_metadata_buffer = lambda req: True
                reqs = [
                    Req(
                        rid=rid,
                        origin_input_text="",
                        origin_input_ids=array("q", range(12)),
                        sampling_params=SamplingParams(max_new_tokens=2),
                    )
                    for rid in ("ready", "unconfirmed", "pending", "failed")
                ]
                for req in reqs:
                    req.pending_bootstrap = True
                    req.metadata_buffer_index = 0
                    req.disagg_kv_sender = SimpleNamespace(
                        pop_decode_prefix_len=lambda: 4,
                        init=lambda num_pages, metadata_index: None,
                    )
                queue.queue = IndexedReqList(ListKind.PD_BOOTSTRAP)
                move(reqs, queue.queue)
                dst = IndexedReqList(ListKind.WAITING)
                with (
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                        return_value=[
                            KVPoll.WaitingForInput,
                            KVPoll.WaitingForInput,
                            KVPoll.Bootstrapping,
                            KVPoll.WaitingForInput,
                        ],
                    ),
                    patch(
                        "sglang.srt.disaggregation.prefill.poll_and_all_reduce_prefill_pp",
                        side_effect=AssertionError("Consensus must not poll PP again"),
                    ),
                    patch(
                        "sglang.srt.disaggregation.prefill.should_force_retry",
                        return_value=False,
                    ),
                ):
                    admitted = queue.move_bootstrapped(dst, outcomes=outcomes)
                expected = [reqs[0]] if outcomes else []
                self.assertEqual(admitted, expected)
                self.assertEqual(list(dst), expected)
                self.assertEqual(failed, [reqs[3]] if outcomes else [])
                self.assertEqual(list(queue.queue), reqs[1:] if outcomes else reqs)
                self.assertTrue(all(req.pending_bootstrap for req in reqs[1:]))
                if outcomes:
                    self.assertFalse(reqs[0].pending_bootstrap)
                    self.assertEqual(reqs[0].start_send_idx, 4)
                    self.assertEqual(reqs[0].arrival_processed_tokens, 7)

    def test_manifest_tracks_real_request_chunk_boundaries(self):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        req = Req(
            rid="chunked",
            origin_input_text="",
            origin_input_ids=array("q", range(12)),
            sampling_params=SamplingParams(max_new_tokens=2),
        )
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(value=1),
            reqs=[req],
            seq_lens_cpu=torch.tensor([8]),
            extend_lens=[4],
        )
        self.assertEqual(
            vpp_mod._vpp_manifest(batch)[1], (("chunked", None, None, None),)
        )
        req.prefix_len, req.extend_end = 4, 8
        first = vpp_mod._vpp_manifest(batch)
        self.assertEqual(first[1], (("chunked", None, 4, 8),))
        req.prefix_len, req.extend_end = 8, 12
        self.assertNotEqual(first, vpp_mod._vpp_manifest(batch))

    def test_pd_terminal_consensus_waits_for_all_layers_and_failure_wins(self):
        """A stage that finished early cannot admit or free a partial KV request."""
        from threading import Barrier

        from sglang.srt.disaggregation.base.conn import KVPoll
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

        class TransferQueue(dict):
            # IndexedReqList.items() exposes queue entries, rather than RID pairs.
            def items(self):
                return list(self.values())

        size = 3
        channels = [Queue() for _ in range(size - 1)]
        barrier = Barrier(size)
        broadcast = [None]
        schedulers = []
        for rank in range(size):
            scheduler = object.__new__(SchedulerVPPMixin)

            def recv(rank=rank):
                return channels[rank - 1].get(timeout=10)

            def send(value, async_send=True, rank=rank):
                channels[rank].put(value)
                return []

            def bcast(value, src, rank=rank):
                if rank == src:
                    broadcast[0] = value
                barrier.wait(timeout=10)
                result = broadcast[0]
                barrier.wait(timeout=10)
                return result

            polls = {
                "request": (KVPoll.Success, KVPoll.Success, KVPoll.Transferring)[rank]
            }
            queue = TransferQueue(
                request=SimpleNamespace(req=SimpleNamespace(rid="request"))
            )
            retired = {}

            def get_rids(reqs, is_send, *groups, polls=polls):
                return tuple(
                    [req.req.rid for req in reqs if KVPoll.Success in group]
                    for group in groups
                )

            def transfer(waiting, outcomes, queue=queue, retired=retired):
                for rid, failed in outcomes.items():
                    queue.pop(rid)
                    retired[rid] = failed
                return [], outcomes

            scheduler.__dict__.update(
                disaggregation_mode=vpp_mod.DisaggregationMode.DECODE,
                _vpp_next_slot=0,
                pp_group=SimpleNamespace(
                    is_first_rank=rank == 0,
                    is_last_rank=rank == size - 1,
                    world_size=size,
                    broadcast_object=bcast,
                ),
                _pp_recv_pyobj_from_prev_stage=recv,
                _pp_send_pyobj_to_next_stage=send,
                _pp_commit_comm_work=lambda work: None,
                pause_state=SimpleNamespace(may_launch=lambda: True),
                waiting_queue=[],
                disagg_decode_prealloc_queue=SimpleNamespace(
                    queue={},
                    retracted_queue=[],
                    publish_pending_destinations=lambda: None,
                ),
                disagg_decode_transfer_queue=SimpleNamespace(
                    queue=queue,
                    resolve_deferred_releases=lambda: None,
                    publish_pending_metadata=lambda: None,
                    enable_staging=True,
                    _poll_with_staging=lambda polls=polls: list(polls.values()),
                    move_transferred_with_outcomes=transfer,
                ),
                get_rids=get_rids,
                enable_hisparse=False,
                process_retract_queue=lambda rids: rids,
                process_prealloc_queue=lambda rids: rids,
                retired=retired,
                polls=polls,
            )
            for name in (
                "_pp_pd_get_retract_ids",
                "_pp_pd_get_prealloc_ids",
                "_route_aborts_to_bad",
                "process_decode_transfer_queue",
            ):
                method = getattr(SchedulerPPMixin, name)
                setattr(
                    scheduler,
                    name,
                    method
                    if name == "_route_aborts_to_bad"
                    else method.__get__(scheduler),
                )
            schedulers.append(scheduler)

        def advance():
            with ThreadPoolExecutor(max_workers=size) as pool:
                results = [pool.submit(s._vpp_pd_advance) for s in schedulers]
                for result in results:
                    result.result(timeout=10)

        advance()
        self.assertTrue(all(s.disagg_decode_transfer_queue.queue for s in schedulers))
        self.assertTrue(all(not s.retired for s in schedulers))
        schedulers[1].polls["request"] = KVPoll.Failed
        schedulers[2].polls["request"] = KVPoll.Success
        advance()
        self.assertTrue(
            all(not s.disagg_decode_transfer_queue.queue for s in schedulers)
        )
        self.assertEqual([s.retired for s in schedulers], [{"request": True}] * size)
        self.assertTrue(all(channel.empty() for channel in channels))

    def test_abort_inputs_are_not_dispatched_between_retained_admissions(self):
        """A later input poll could release the preceding unlaunched batch's KV."""
        layout = VPPLayout(12, 2, virtual_stages=3)
        scheduler = object.__new__(SchedulerVPPMixin)
        polls = []
        batches = [
            SimpleNamespace(
                reqs=[
                    SimpleNamespace(
                        rid=f"req-{i}",
                        session_generation=0,
                        prefix_len=0,
                        extend_end=None,
                        beam_group=None,
                        finished=False,
                    )
                ],
                seq_lens_cpu=torch.tensor([10]),
                extend_lens=[10],
                forward_mode=SimpleNamespace(value=1),
                return_hidden_states=False,
                chunked_req=None,
            )
            for i in range(2)
        ]

        def ingest():
            polls.append(True)
            if len(polls) > 1:
                batches[0].reqs[0].finished = True
            return []

        pending = iter(batches)
        scheduler.__dict__.update(
            disaggregation_mode=vpp_mod.DisaggregationMode.NULL,
            pp_loop_size=2,
            _vpp_next_slot=0,
            _vpp_next_batch_seq=0,
            running_mbs=[object(), object()],
            last_mbs=[None, None],
            mbs=[None, None],
            chunked_req=None,
            vpp_batches=VPPBatchLedger(layout, 0, 2),
            vpp_schedule=VPPFixedSchedule(layout, 0, 2),
            pp_group=SimpleNamespace(
                is_first_rank=True,
                is_last_rank=False,
                all_gather_object=lambda manifest: [manifest, manifest],
            ),
            send_req_work=[],
            increment_scheduler_iterations=lambda: None,
            _pp_mark_capacity_digest=lambda: None,
            _pp_check_capacity_digest=lambda: None,
            _pp_ingest_requests=ingest,
            _pp_commit_comm_work=lambda works: None,
            _pp_send_reqs_to_next_stage=lambda requests: [],
            _pp_send_capacity_digest=lambda: None,
            pause_state=SimpleNamespace(may_launch=lambda: True),
            get_next_batch_to_run=lambda **kwargs: SimpleNamespace(
                running_batch=kwargs["running_batch"], batch_to_run=next(pending)
            ),
        )
        wave = scheduler._vpp_admit_wave()
        self.assertEqual(
            [state.batch.reqs[0].rid for state in wave], ["req-0", "req-1"]
        )
        self.assertFalse(batches[0].reqs[0].finished)
        self.assertEqual([state.slot_id for state in wave], [0, 1])

    def test_physical_last_rank_samples_only_at_global_final_stage(self):
        """The last physical rank's earlier passes must return activations."""
        samples = []
        layout = VPPLayout(12, 2, virtual_stages=3)

        def forward(batch, *, pp_proxy_tensors, vpp_stage_id):
            return SimpleNamespace(
                logits_output=(
                    SimpleNamespace(next_token_logits=torch.ones(1, 4))
                    if vpp_stage_id == 5
                    else PPProxyTensors({"hidden_states": torch.ones(1, 2)})
                ),
                can_run_graph=False,
                expert_distribution_metrics=None,
                routed_experts_output=None,
                indexer_topk_output=None,
                kv_weight_version_record=None,
            )

        worker = object.__new__(TpModelWorker)
        worker.__dict__.update(
            pp_group=SimpleNamespace(is_last_rank=True),
            enable_overlap=True,  # VPP must also bypass delayed sampling.
            enable_spec=False,
            is_dllm=lambda: False,
            _maybe_finalize_elastic_cuda_graph_scale=lambda: None,
            _model_runner=SimpleNamespace(
                model=SimpleNamespace(vpp_layout=layout),
                forward=forward,
                sample=lambda logits, batch: samples.append(batch) or torch.tensor([2]),
            ),
        )
        batch = SimpleNamespace(
            apply_deprecated_skip_attn_backend_init=lambda value: None,
            is_prefill_only=False,
            sampling_info=SimpleNamespace(grammars=[object()]),
        )
        for stage_id in (1, 3, 5):
            result = worker.forward_batch_generation(
                None, forward_batch=batch, vpp_stage_id=stage_id
            )
            if stage_id < 5:
                self.assertIsNotNone(result.pp_hidden_states_proxy_tensors)
                self.assertIsNone(result.next_token_ids)
                self.assertEqual(samples, [])
            else:
                self.assertEqual(result.next_token_ids.tolist(), [2])
                self.assertIsNone(result.delay_sample_func)
                self.assertEqual(samples, [batch])

    def test_stage_replans_metadata_without_replacing_retained_inputs(self):
        """Each pass must rebuild shared metadata after another batch used it."""
        seen = []
        backend = SimpleNamespace(current=None)

        def plan(batch):
            backend.current = batch.input_ids.clone()

        def forward(stage_id, input_ids, positions, batch, **kwargs):
            torch.testing.assert_close(backend.current, input_ids)
            seen.append((stage_id, input_ids.data_ptr(), input_ids.tolist()))
            return input_ids + stage_id

        backend.init_forward_metadata = plan
        layout = VPPLayout(12, 2, virtual_stages=3)
        model = SimpleNamespace(
            vpp_layout=layout,
            vpp_stages=layout.stages_for_rank(0),
            forward_vpp_stage=forward,
        )
        batches = [
            SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_decode=lambda: True,
                    is_extend=lambda: False,
                    is_mixed=lambda: False,
                ),
                global_num_tokens_cpu=None,
                input_ids=torch.tensor([i, i + 1]),
                positions=torch.tensor([7, 8]),
                input_embeds=None,
                replace_embeds=None,
                attn_tp_sequence_sharded=False,
            )
            for i in (10, 20)
        ]
        for stage_id in (0, 2, 4):
            for batch in batches:
                execute_vpp_stage(
                    model=model,
                    attn_backend=backend,
                    forward_batch=batch,
                    stage_id=stage_id,
                    pp_proxy_tensors=None,
                    device_timer=None,
                )
        self.assertEqual(
            seen,
            [
                (stage_id, batch.input_ids.data_ptr(), batch.input_ids.tolist())
                for stage_id in (0, 2, 4)
                for batch in batches
            ],
        )

    def test_multiple_waves_keep_inputs_outputs_and_completion_independent(self):
        """A partial wave must not leave receives or recycle workspace views."""
        from threading import local

        topology = local()
        layout = VPPLayout(12, physical_size=2, virtual_stages=3)
        queues = [Queue(), Queue()]
        schedulers = []
        calls = [[], []]
        launches = [[], []]
        completions = [[], []]
        for rank in range(2):
            scheduler = object.__new__(SchedulerVPPMixin)
            scheduler.__dict__.update(
                pp_group=SimpleNamespace(is_first_rank=rank == 0),
                vpp_schedule=VPPFixedSchedule(layout, rank, 2),
                vpp_batches=VPPBatchLedger(layout, rank, 2),
                vpp_receiver=_Receiver(queues[rank]),
                vpp_sends=_Sends(queues[1 - rank]),
                forward_stream=_Stream(),
                schedule_stream=_Stream(),
                forward_stream_ctx=nullcontext(),
                device_module=SimpleNamespace(Event=_Event),
                tp_worker=_Worker(rank, calls[rank]),
                future_map=None,
                enable_unified_memory=False,
                resolve_forward_inputs=lambda batch, future_map: None,
                _pp_prepare_tensor_dict=lambda result, batch: {
                    "next_token_ids": result.next_token_ids
                },
                _on_step_launch=lambda batch, r=rank: (
                    launches[r].append(batch.batch_seq) or batch.batch_seq
                ),
                _on_step_forward_recorded=lambda event, r=rank: completions[r].append(
                    event
                ),
            )
            schedulers.append(scheduler)

        def init_forward_batch(batch, runner, *, return_hidden_states_before_norm):
            self.assertFalse(return_hidden_states_before_norm)
            return SimpleNamespace(
                batch_seq=batch.batch_seq, final_stage=layout.logical_size - 1
            )

        def execute(rank, wave):
            topology.rank = rank
            schedulers[rank]._vpp_execute_wave(wave)

        with (
            patch.object(
                vpp_mod,
                "get_parallel",
                lambda: SimpleNamespace(pp_rank=topology.rank),
            ),
            patch.object(vpp_mod.ForwardBatch, "init_new", init_forward_batch),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            for sequences in ((0, 1), (2,), (3, 4)):
                waves = [[], []]
                for rank, scheduler in enumerate(schedulers):
                    for slot_id, seq in enumerate(sequences):
                        batch = SimpleNamespace(
                            batch_seq=seq,
                            hicache_consumer_index=-1,
                            return_logprob=False,
                            extend_lens=[1],
                            forward_mode=SimpleNamespace(is_extend=lambda: True),
                        )
                        state = VPPBatchState(seq, slot_id, ("req", seq), batch)
                        scheduler.vpp_batches.admit(state)
                        waves[rank].append(state)
                        if rank == 0:
                            scheduler.vpp_schedule.enqueue(vpp_mod.VPPTask(seq, 0))
                futures = [
                    executor.submit(execute, rank, waves[rank]) for rank in range(2)
                ]
                for future in futures:
                    future.result(timeout=15)
                for rank, scheduler in enumerate(schedulers):
                    self.assertFalse(scheduler.vpp_receiver.posted)
                    for state in waves[rank]:
                        self.assertTrue(state.metadata.forward_done.recorded)
                        if rank == 1:
                            self.assertEqual(
                                state.output_tensors["next_token_ids"].item(),
                                state.batch_seq + layout.logical_size,
                            )
                        scheduler.vpp_batches.complete(state.batch_seq)
        for rank in range(2):
            self.assertEqual(launches[rank], list(range(5)))
            self.assertEqual(len(completions[rank]), 5)
            self.assertEqual(
                calls[rank],
                [
                    (seq, stage_id)
                    for sequences in ((0, 1), (2,), (3, 4))
                    for stage_id in range(rank, 6, 2)
                    for seq in sequences
                ],
            )


if __name__ == "__main__":
    unittest.main()
