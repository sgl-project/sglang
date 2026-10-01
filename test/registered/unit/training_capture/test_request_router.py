"""Capture identity follows real request IPC and scheduler cancellation paths."""

import copy
import dataclasses
import json
import multiprocessing as mp
import pickle
import tempfile
import time
import unittest
from array import array
from collections import Counter
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch
import torch.distributed as dist
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenizedGenerateReqInput,
    TokenizedGenerateReqInput,
    msgpack_decode,
    msgpack_encode,
)
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.cohort import CaptureCohortAllocator
from sglang.srt.training_capture.cohort_service import (
    CaptureCohortService,
    CaptureTicket,
)
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.protocol import canonical_bytes
from sglang.srt.training_capture.request_router import (
    CaptureRequestRouter,
    CaptureRequestTicket,
)
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import Registrar, make_snapshot

register_cpu_ci(est_time=35, suite="base-a-test-cpu")


def incoming(rid="request"):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", [3, 4]),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(
            max_new_tokens=3,
            sampling_seed=17,
            stop_token_ids={7, 9},
            logit_bias={"8": 0.2, "9": -0.1},
            is_normalized=True,
        ),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


def local_request(raw):
    return Req(
        raw.rid,
        raw.input_text,
        raw.input_ids,
        raw.sampling_params,
        vocab_size=256,
        require_reasoning=raw.require_reasoning,
        extra_key=raw.extra_key,
        token_type_ids=raw.token_type_ids,
        training_capture_ticket=raw.training_capture_ticket,
    )


def receiver(router, *, pp_rank=0, tp_rank=0):
    group = SimpleNamespace(rank=tp_rank, ranks=[0, 1], cpu_group=None)
    return SchedulerRequestReceiver(
        recv_from_tokenizer=None,
        recv_from_rpc=None,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        ps=SimpleNamespace(pp_rank=pp_rank, attn_tp_rank=tp_rank, attn_cp_rank=0),
        tp_group=group,
        tp_cpu_group=None,
        attn_tp_group=group,
        attn_tp_cpu_group=None,
        attn_cp_group=group,
        attn_cp_cpu_group=None,
        world_group=group,
        server_args=None,
        model_config=SimpleNamespace(is_multimodal=False),
        max_recv_per_poll=-1,
        stream_output=lambda *_: None,
        get_last_batch=lambda: None,
        training_capture_router=router,
    )


def scheduler(router):
    obj = Scheduler.__new__(Scheduler)
    obj.request_receiver = SimpleNamespace(training_capture_router=router)
    obj.enable_session_radix_cache = False
    obj.disaggregation_mode = DisaggregationMode.NULL
    obj.model_config = SimpleNamespace(hf_eos_token_id={9}, vocab_size=256)
    obj.metrics_reporter = SimpleNamespace(enable_metrics=False)
    obj.tokenizer = obj.dllm_config = None
    obj._maybe_namespace_elastic_radix_cache = MagicMock()
    obj.spec_algorithm = MagicMock()
    obj.spec_algorithm.is_dflash_family.return_value = False
    obj.max_new_tokens_limit = 2
    obj.max_req_len = obj.max_total_num_tokens = obj.max_req_input_len = 64
    obj.page_size = 1
    obj.grammar_manager = MagicMock()
    obj.grammar_manager.process_req_with_grammar.return_value = False
    obj.enable_priority_scheduling = False
    obj.abort_on_priority_when_disabled = True
    obj.max_queued_requests = None
    obj.waiting_queue = []
    obj._prefetch_kvcache = MagicMock()
    obj.ipc_channels = MagicMock()
    obj.enable_hicache_storage = obj.enable_hierarchical_cache = False
    obj.chunked_req = obj.running_batch = obj.last_batch = None
    obj.ps = SimpleNamespace(pp_size=1)
    return obj


def distributed_request_worker(rank, root, endpoint):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=(Path(root) / "rendezvous").as_uri(),
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    base, _ = make_snapshot()
    layout = plan_capture_layout(
        base.kv, tp_size=2, pp_layer_ranges=[(0, 1), (1, 4)], aux_tp_rank=1
    )
    partition = layout.partitions[rank]
    rows = []
    try:
        with (
            get_context().override_server_args(enable_dp_attention=False),
            get_parallel().override(tp_rank=rank % 2),
        ):
            tp_groups = [dist.new_group(ranks=[0, 1]), dist.new_group(ranks=[2, 3])]
            for case in ("bound", "queued"):
                config = CaptureConfig(
                    dataset_id=f"request-{case}",
                    model_id="fixture",
                    producer_revision="test",
                    selected_layer_ids=base.kv.selected_layer_ids,
                    catalog_endpoint=endpoint,
                    journal_directory=str(Path(root) / f"journal-{rank}"),
                    store=StoreSetup(
                        local_hostname=f"rank-{rank}", master_server_addr="localhost:1"
                    ),
                    sample_ratio=1.0,
                    max_sample_tokens=8,
                    max_inflight_samples=1,
                    max_host_bytes=2 << 20,
                )
                resources = CaptureResources()
                resources.catalog = (
                    HTTPCaptureCatalog(endpoint) if partition.include_aux else None
                )
                if partition.active:
                    resources.pool = HostBufferPool(
                        kv=base.kv,
                        max_tokens=8,
                        slots=1,
                        max_bytes=config.max_host_bytes,
                        registrar=Registrar(),
                        pin_memory=False,
                        partition=partition,
                    )
                control = dist.new_group(backend="gloo", timeout=timedelta(seconds=20))
                service = CaptureCohortService(
                    CaptureCohortAllocator(
                        group=control,
                        layout=layout,
                        config=config,
                        teacher=base.teacher,
                        kv=base.kv,
                        resources=resources,
                        timeout_seconds=15,
                    )
                )
                assert not service._cycle()
                assert not service._cycle()
                service.start()
                router = CaptureRequestRouter(service)
                raw = msgpack_decode(msgpack_encode(incoming("same-rid")))
                ps = SimpleNamespace(
                    pp_rank=rank // 2,
                    pp_size=2,
                    tp_size=2,
                    attn_tp_rank=rank % 2,
                    attn_cp_rank=0,
                    attn_dp_rank=0,
                    attn_cp_size=1,
                    attn_tp_size=2,
                )
                rec = dataclasses.replace(
                    receiver(router),
                    ps=ps,
                    tp_group=SimpleNamespace(
                        rank=rank, ranks=[rank // 2 * 2, rank // 2 * 2 + 1]
                    ),
                    tp_cpu_group=tp_groups[rank // 2],
                    world_group=SimpleNamespace(cpu_group=dist.group.WORLD),
                )
                pull = SchedulerRequestReceiver._pull_raw_reqs

                def source(this, raw=raw, pull=pull):
                    if this.ps.pp_rank == 0:
                        return [raw] if this.ps.attn_tp_rank == 0 else None
                    return pull(this)

                with patch.object(SchedulerRequestReceiver, "_pull_raw_reqs", source):
                    requests = rec.recv_requests()
                obj = scheduler(router)
                obj.request_receiver, obj.ps, obj.world_group = rec, ps, rec.world_group
                obj.running_mbs, obj.mbs = [], []
                for request in requests:
                    obj.handle_generate_request(request)
                if rank // 2 == 0:
                    obj._pp_send_pyobj_to_next_stage(requests)
                req = obj.waiting_queue[0]
                assert req.training_capture_route is not None
                assert requests[0].sampling_params.max_new_tokens == 3
                assert req.sampling_params.max_new_tokens == 2
                route = req.training_capture_route
                if case == "bound":
                    route = router.bind(req)
                    assert route is not None
                    obj.waiting_queue = []
                    obj.running_mbs = [SimpleNamespace(reqs=[req])]
                dist.barrier()
                row = {
                    "case": case,
                    "capture_id": route.ticket.cohort.capture_id,
                    "fingerprint": route.ticket.cohort.request_sha256,
                    "execution": route.execution_sha256,
                    "selected": router.counters["selected"],
                    "active": partition.active,
                }
                obj.abort_request(AbortReq(rid=req.rid))
                assert router.bind(req) is None
                if case == "bound":
                    assert not service.close(timeout=0)
                    assert (
                        resources.pool is None or resources.pool.stats()["filling"] == 1
                    )
                    dist.barrier()
                    service.finish(
                        route.handle, outcome="failed", transfer_complete=True
                    )
                assert service.close(timeout=15)
                assert service.error is None
                rows.append(row)
                if resources.pool is not None:
                    assert resources.pool.stats()["free"] == 1
                    resources.pool.close()
                dist.destroy_process_group(control)
                dist.barrier()
            dist.destroy_process_group(tp_groups[rank // 2])
        (Path(root) / f"rank-{rank}.json").write_bytes(canonical_bytes(rows))
    finally:
        dist.destroy_process_group()


class TestCaptureRequestRouter(CustomTestCase):
    def setUp(self):
        config_scope = get_context().override_server_args(enable_dp_attention=False)
        config_scope.__enter__()
        self.addCleanup(config_scope.__exit__, None, None, None)
        parallel_scope = get_parallel().override(tp_rank=0)
        parallel_scope.__enter__()
        self.addCleanup(parallel_scope.__exit__, None, None, None)
        config = CaptureConfig(
            dataset_id="router",
            model_id="fixture",
            producer_revision="test",
            selected_layer_ids=[0],
            catalog_endpoint="http://localhost:1",
            journal_directory="unused",
            store=StoreSetup(
                local_hostname="localhost", master_server_addr="localhost:1"
            ),
            sample_ratio=1.0,
            max_sample_tokens=8,
        )
        self.service = SimpleNamespace(
            allocator=SimpleNamespace(config=config, rank=0),
            claim=MagicMock(
                side_effect=lambda fingerprint: CaptureTicket(
                    capture_id=f"capture-{self.service.claim.call_count}",
                    fencing_token=1,
                    request_sha256=fingerprint,
                )
            ),
            bind=MagicMock(return_value=object()),
            cancel=MagicMock(return_value=True),
            finish=MagicMock(),
        )
        self.router = CaptureRequestRouter(self.service)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(), "Gloo required"
    )
    def test_real_tp_pp_request_path_binds_one_cohort_and_drains_cancelled_actors(self):
        """PP forwards original ingress while each stage clips its private Req."""
        catalog = TestCaptureCatalog()
        self.addCleanup(catalog.close)
        with tempfile.TemporaryDirectory() as root:
            context = mp.get_context("spawn")
            workers = [
                context.Process(
                    target=distributed_request_worker,
                    args=(rank, root, catalog.endpoint),
                )
                for rank in range(4)
            ]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 100
                for worker in workers:
                    worker.join(timeout=max(0, deadline - time.monotonic()))
                self.assertEqual([worker.exitcode for worker in workers], [0] * 4)
                rows = [
                    json.loads((Path(root) / f"rank-{rank}.json").read_bytes())
                    for rank in range(4)
                ]
            finally:
                for worker in workers:
                    if worker.pid is not None and worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)
                        if worker.is_alive():
                            worker.kill()
                            worker.join(timeout=5)
        for case_rows in zip(*rows, strict=True):
            with self.subTest(case=case_rows[0]["case"]):
                self.assertEqual([row["selected"] for row in case_rows], [1, 0, 0, 0])
                self.assertEqual(
                    [row["active"] for row in case_rows], [False, False, True, True]
                )
                self.assertEqual(len({row["capture_id"] for row in case_rows}), 1)
                self.assertEqual(len({row["fingerprint"] for row in case_rows}), 1)
                self.assertEqual(len({row["execution"] for row in case_rows}), 1)
                self.assertEqual(
                    catalog.captures[case_rows[0]["capture_id"]]["state"], "FAILED"
                )
        self.assertFalse(catalog.publications)
        self.assertFalse(catalog.errors)

    def selected(self, rid="request"):
        raw = incoming(rid)
        self.router.prepare([raw])
        self.assertIsNotNone(raw.training_capture_ticket)
        return raw

    def attached(self, rid="request"):
        raw = self.selected(rid)
        req = local_request(raw)
        self.router.attach(raw, req)
        self.assertIsNotNone(req.training_capture_route)
        return req

    def test_entry_decision_precedes_broadcast_and_survives_pp_normalization(self):
        raw = incoming()
        rec = receiver(self.router)

        def broadcast(_receiver, payload):
            self.assertIsNotNone(payload[0].training_capture_ticket)
            return msgpack_decode(
                msgpack_encode(BatchTokenizedGenerateReqInput(batch=payload))
            ).batch

        with (
            patch.object(
                SchedulerRequestReceiver, "_pull_raw_reqs", return_value=[raw]
            ),
            patch.object(
                SchedulerRequestReceiver, "_broadcast_reqs_across_ranks", broadcast
            ),
        ):
            received = rec.recv_requests()[0]
        first = scheduler(self.router)
        first.handle_generate_request(received)
        req = first.waiting_queue[0]
        self.assertEqual(req.sampling_params.max_new_tokens, 2)
        self.assertEqual(received.sampling_params.max_new_tokens, 3)
        bound = self.router.bind(req)
        self.assertIsNotNone(bound)
        self.assertIsNone(self.router.bind(req))
        self.assertNotEqual(bound.execution_sha256, bound.ticket.cohort.request_sha256)
        following = pickle.loads(pickle.dumps(received))
        second = scheduler(self.router)
        second.handle_generate_request(following)
        route = self.router.bind(second.waiting_queue[0])
        self.assertEqual(route.execution_sha256, bound.execution_sha256)
        self.assertEqual(route.ticket, bound.ticket)
        self.service.claim.assert_called_once()
        self.assertEqual(self.service.bind.call_count, 2)
        self.service.cancel.assert_not_called()

    def test_pp_and_tp_followers_never_select_again(self):
        raw = self.selected()
        for pp_rank, tp_rank in ((0, 1), (1, 0), (1, 1)):
            with self.subTest(pp=pp_rank, tp=tp_rank):
                rec = receiver(self.router, pp_rank=pp_rank, tp_rank=tp_rank)
                with (
                    patch.object(
                        SchedulerRequestReceiver, "_pull_raw_reqs", return_value=[raw]
                    ),
                    patch.object(
                        SchedulerRequestReceiver,
                        "_broadcast_reqs_across_ranks",
                        return_value=[raw],
                    ),
                ):
                    self.assertIs(rec.recv_requests()[0], raw)
        self.service.claim.assert_called_once()

    def test_old_array_ipc_without_trailing_ticket_remains_decodable(self):
        raw = incoming()
        wire = msgspec.msgpack.decode(msgpack_encode(raw))
        self.assertIsNone(wire[-1])
        decoded = msgpack_decode(msgspec.msgpack.encode(wire[:-1]))
        self.assertIsInstance(decoded, TokenizedGenerateReqInput)
        self.assertIsNone(decoded.training_capture_ticket)
        self.assertEqual(decoded.input_ids, raw.input_ids)

    def test_changed_request_cancels_ticket_without_handing_out_storage(self):
        changes = {
            "rid": lambda raw: setattr(raw, "rid", "other"),
            "prompt": lambda raw: raw.input_ids.append(5),
            "seed": lambda raw: setattr(raw.sampling_params, "sampling_seed", 99),
            "bias": lambda raw: raw.sampling_params.logit_bias.update({"8": 0.7}),
            "grammar": lambda raw: setattr(raw.sampling_params, "regex", "[ab]+"),
            "cache_salt": lambda raw: setattr(raw, "extra_key", "other"),
            "reasoning": lambda raw: setattr(raw, "require_reasoning", True),
            "token_types": lambda raw: setattr(raw, "token_type_ids", [0, 1]),
        }
        for name, mutate in changes.items():
            with self.subTest(field=name):
                raw = self.selected()
                mutate(raw)
                req = local_request(raw)
                self.router.attach(raw, req)
                self.assertIsNone(req.training_capture_route)
                self.assertIsNone(self.router.bind(req))
        self.assertEqual(self.service.cancel.call_count, len(changes))
        self.service.bind.assert_not_called()

    def test_same_rid_gets_a_distinct_incarnation_and_client_ticket_is_replaced(self):
        first = self.selected()
        second = incoming()
        second.training_capture_ticket = first.training_capture_ticket
        self.router.prepare([BatchTokenizedGenerateReqInput(batch=[second])])
        left = msgspec.json.decode(
            first.training_capture_ticket, type=CaptureRequestTicket
        )
        right = msgspec.json.decode(
            second.training_capture_ticket, type=CaptureRequestTicket
        )
        self.assertNotEqual(left.nonce, right.nonce)
        self.assertNotEqual(left.cohort.request_sha256, right.cohort.request_sha256)
        self.assertNotEqual(left.cohort.capture_id, right.cohort.capture_id)

    def test_unsupported_or_private_requests_never_consume_cohorts(self):
        modifications = (
            {"no_logs": True},
            {"lora_id": "lora"},
            {"input_embeds": [[1.0]]},
            {"session_id": "session"},
            {"custom_logit_processor": "custom"},
            {"input_ids": array("q", range(9))},
        )
        raws = [
            msgspec.structs.replace(incoming(), **values) for values in modifications
        ]
        self.router.prepare(
            [BatchTokenizedGenerateReqInput(batch=raws), AbortReq(rid="other")]
        )
        self.assertTrue(all(raw.training_capture_ticket is None for raw in raws))
        self.service.claim.assert_not_called()

    def test_sampling_backpressure_and_encoding_failure_do_not_block_requests(self):
        self.router.sample_ratio = lambda: 0
        self.router.prepare([incoming()])
        self.service.claim.assert_not_called()
        self.router.sample_ratio = lambda: 1
        self.service.claim.side_effect = None
        self.service.claim.return_value = None
        raw = incoming()
        self.router.prepare([raw])
        self.assertIsNone(raw.training_capture_ticket)
        self.service.claim.return_value = CaptureTicket(
            capture_id="huge-" + "a" * 3000, fencing_token=1, request_sha256="a" * 64
        )
        self.router.prepare([raw])
        self.assertIsNone(raw.training_capture_ticket)
        self.service.cancel.assert_called_once()

    def test_malformed_wire_never_binds_a_cohort(self):
        for wire in (b"{", b"x" * 2049, b'{"version":2}'):
            with self.subTest(wire=wire[:20]):
                raw = incoming()
                raw.training_capture_ticket = wire
                req = local_request(raw)
                self.router.attach(raw, req)
                self.assertIsNone(req.training_capture_route)
        self.service.bind.assert_not_called()

    def test_normalization_does_not_alias_nested_ingress_sampling_values(self):
        raw = self.selected()
        req = local_request(raw)
        self.router.attach(raw, req)
        req.sampling_params.logit_bias["8"] = 99.0
        req.sampling_params.stop_token_ids.add(55)
        self.assertEqual(raw.sampling_params.logit_bias["8"], 0.2)
        self.assertEqual(raw.sampling_params.stop_token_ids, {7, 9})
        req.sampling_params = copy.deepcopy(raw.sampling_params)
        req.sampling_params.logit_bias = {"9": -0.1, "8": 0.2}
        route = self.router.bind(req)
        self.assertEqual(route.execution_sha256, route.ticket.cohort.request_sha256)

    def test_scheduler_queue_rejection_cancels_only_the_removed_request(self):
        obj = scheduler(self.router)
        old, new = self.attached("old"), self.attached("new")
        old.priority, new.priority = 2, 1
        obj.enable_priority_scheduling = True
        obj.schedule_low_priority_values_first = True
        obj.waiting_queue = [old]
        obj.max_queued_requests = 1
        obj._add_request_to_queue(new)
        self.assertEqual(obj.waiting_queue, [new])
        self.service.cancel.assert_called_once_with(
            old.training_capture_route.ticket.cohort, "queue_rejected"
        )
        self.assertIsNone(self.router.bind(old))
        self.assertIsNotNone(self.router.bind(new))

    def test_priority_rejection_and_grammar_abort_cancel_before_binding(self):
        obj = scheduler(self.router)
        req = self.attached()
        req.priority = 3
        obj._add_request_to_queue(req)
        self.assertFalse(obj.waiting_queue)
        self.assertIsNone(self.router.bind(req))
        other = self.attached("grammar")
        other.set_finish_with_abort("invalid grammar")
        self.assertIsNone(self.router.bind(other))
        self.assertEqual(self.service.cancel.call_count, 2)

    def test_waiting_timeout_and_explicit_abort_invalidate_pending_routes(self):
        obj = scheduler(self.router)
        req = self.attached("timeout")
        req.time_stats.wait_queue_entry_time = time.perf_counter() - 10
        obj.waiting_queue = [req]
        with envs.SGLANG_REQ_WAITING_TIMEOUT.override(1):
            obj._abort_on_waiting_timeout()
        self.assertFalse(obj.waiting_queue)
        other = self.attached("abort")
        obj.waiting_queue = [other]
        obj.abort_request(AbortReq(rid="abort"))
        self.assertFalse(obj.waiting_queue)
        self.assertEqual(self.service.cancel.call_count, 2)
        self.service.bind.assert_not_called()

    def test_running_abort_invalidates_but_does_not_finish_transfer_ownership(self):
        obj = scheduler(self.router)
        req = self.attached()
        route = self.router.bind(req)
        obj.running_batch = SimpleNamespace(reqs=[req])
        obj.abort_request(AbortReq(rid=req.rid))
        self.assertIsNotNone(req.to_finish)
        self.assertIs(route.handle, self.service.bind.return_value)
        self.service.cancel.assert_called_once_with(
            route.ticket.cohort, "running_request_aborted"
        )
        self.service.finish.assert_not_called()

    def test_binding_bookkeeping_failure_returns_unused_actor_ownership(self):
        """A failure after local bind must not orphan a handle before any transfer."""
        req = self.attached()

        class FaultCounter(Counter):
            def __setitem__(self, key, value):
                if key == "bound":
                    raise MemoryError("injected admission bookkeeping failure")
                super().__setitem__(key, value)

        self.router.counters = FaultCounter()
        self.assertIsNone(self.router.bind(req))
        self.service.cancel.assert_called_once()
        self.service.finish.assert_called_once_with(
            self.service.bind.return_value, outcome="failed", transfer_complete=True
        )
        self.assertIsNone(req.training_capture_route.handle)


if __name__ == "__main__":
    unittest.main()
