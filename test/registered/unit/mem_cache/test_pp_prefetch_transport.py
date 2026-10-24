"""Real four-process Gloo test of the native PP P2P + TP broadcast path.

The shared harness loads production helpers without the optional model stack.
Transport executes real torch.distributed operations, including envelope pickle.
"""

import pickle
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.pp_cache_test_support import (
    PrefillControl,
    RowPool,
    functions,
)
from sglang.test.pp_cache_test_support import m as runtime
from sglang.test.pp_cache_test_support import (
    prefetch_wait_type,
    request,
    select,
)
from sglang.test.pp_cache_test_support import stage as make_stage
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


def _transport_worker(rank, init_path, tp_size):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    pp_size = 4 // tp_size
    groups = [dist.new_group(list(range(i, i + tp_size))) for i in range(0, 4, tp_size)]
    namespace = dict(
        torch=torch,
        dist=dist,
        np=np,
        pickle=pickle,
        is_musa=lambda: False,
        PrefillControl=PrefillControl,
    )
    functions("utils/common.py", {"point_to_point_pyobj", "broadcast_pyobj"}, namespace)
    functions(
        "managers/scheduler_pp_mixin.py",
        {
            "_pp_send_pyobj_to_next_stage",
            "_pp_recv_pyobj_from_prev_stage",
            "_pp_send_prefill_control",
            "_pp_recv_prefill_control",
        },
        namespace,
        cls="SchedulerPPMixin",
    )
    stage = rank // tp_size
    scheduler = NS(
        ps=NS(
            attn_tp_rank=rank % tp_size,
            attn_cp_rank=0,
            attn_dp_rank=0,
            attn_cp_size=1,
            attn_tp_size=tp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            pp_rank=stage,
        ),
        world_group=NS(cpu_group=dist.group.WORLD),
        attn_tp_group=NS(
            rank=rank, ranks=list(range(stage * tp_size, (stage + 1) * tp_size))
        ),
        attn_tp_cpu_group=groups[stage],
        pp_prefetch_admission=None,
    )
    namespace["get_parallel"] = lambda: scheduler.ps
    scheduler.attn_tp_group.world_size = tp_size
    scheduler.attn_tp_group.rank_in_group = rank % tp_size
    scheduler.attn_tp_group.cpu_group = groups[stage]
    namespace["get_attn_tp_group"] = lambda: scheduler.attn_tp_group
    namespace["get_attn_cp_group"] = lambda: NS(world_size=1)
    functions(
        "distributed/communication_op.py", {"attn_cp_tp_broadcast_pyobj"}, namespace
    )
    send = namespace["_pp_send_pyobj_to_next_stage"]
    recv = namespace["_pp_recv_pyobj_from_prev_stage"]
    scheduler._pp_send_pyobj_to_next_stage = lambda data, **kw: send(
        scheduler, data, **kw
    )
    scheduler._pp_recv_pyobj_from_prev_stage = lambda: recv(scheduler)

    def send_control(control):
        return namespace["_pp_send_prefill_control"](scheduler, control)

    def recv_control():
        return namespace["_pp_recv_prefill_control"](scheduler)

    work_container = NS(P2PWork=lambda work, payload: NS(work=work, payload=payload))
    try:
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.dict(
                sys.modules, {"sglang.srt.distributed.parallel_state": work_container}
            ),
        ):
            # Existing request/transfer messages, including empty traffic,
            # stay on the native ordered PP path and TP broadcast helpers.
            messages = [
                [],
                ["legacy-rid"],
                [[], []],
            ]
            if stage == 0:
                works = []
                for message in messages:
                    works.extend(send(scheduler, message, async_send=True))
                for work in works:
                    work.work.wait()
                for message in messages:
                    assert recv(scheduler) == message
            else:
                received = [recv(scheduler) for _ in messages]
                assert received == messages
                works = []
                for message in received:
                    works.extend(send(scheduler, message, async_send=True))
                for work in works:
                    work.work.wait()

            # The actual boundary codec preserves native A/B framing and
            # carries typed C fields through the same P2P/TP relay.
            for enabled in (False, True):
                scheduler.pp_prefetch_admission = object() if enabled else None
                expected = PrefillControl(
                    ["transferred"],
                    (
                        runtime.AdmissionUpdate(layout=(("r", 256, 1025),))
                        if enabled
                        else None
                    ),
                )
                control = expected if stage == 0 else recv_control()
                assert control == expected
                for work in send_control(control):
                    work.work.wait()
                if stage == 0:
                    assert recv_control() == expected

            # Exercise the real gate, not just isolated message dataclasses.
            for case in ("ready", "pending", "capacity", "abort"):
                gate = make_stage(stage, pp_size, {"r": 512 - 64 * stage})
                scheduler.pp_prefetch_admission = gate
                computed = []
                for beat in range(12):
                    inputs = (
                        (["abort-r"] if case == "abort" and beat == 2 else [])
                        if stage == 0
                        else recv(scheduler)
                    )
                    if inputs:
                        for req in list(gate.s.waiting_queue):
                            gate.release(req)
                            gate.s.waiting_queue.remove(req)
                    if stage == pp_size - 1:
                        gate.s.tree_cache.waiting = (
                            {"r"} if case == "pending" and beat < 3 else set()
                        )
                        if case == "capacity" and beat == 0:
                            gate.s.req_to_token_pool = RowPool(0)
                        if case == "capacity" and beat == 3:
                            gate.s.req_to_token_pool = RowPool(1)
                    if stage:
                        gate.receive_forward(recv_control().admission)
                    selection = gate.prepare_selection()
                    batch = select(gate, selection)
                    computed.extend(row[0] for row in runtime.batch_layout(batch))
                    update = gate.finish_selection(batch)
                    transferred = [f"done-{beat}"]
                    control = PrefillControl(transferred, update)
                    works = (
                        []
                        if stage == pp_size - 1
                        else send(scheduler, inputs, async_send=True)
                    )
                    works.extend(send_control(control))
                    for work in works:
                        work.work.wait()
                    control = recv_control()
                    assert control.transferred_rids == transferred
                    gate.consume_return(control.admission)
                    works = send_control(control)
                    for work in works:
                        work.work.wait()
                    if stage == 0:
                        assert recv_control() == control
                    if batch is not None:
                        assert batch.reqs[0].prefix_len == 512 - 64 * (pp_size - 1)
                assert computed == ([] if case == "abort" else ["r"]), (
                    case,
                    stage,
                    computed,
                )
                assert not gate.entries

            # One batch of timeout permissions uses only the stage TP group.
            handles = [
                request("clock0").cache_request_handle,
                request("clock1").cache_request_handle,
            ]
            cache = NS(
                prefetch_stop_policy="timeout",
                prefetch_timeout_base=1.0,
                prefetch_timeout_per_page=0.0,
                page_size=64,
                ongoing_prefetch={
                    handle: NS(
                        operation=NS(
                            start_time=0.0 if i == 0 else 100.0,
                            token_ids=[0] * 64,
                            hash_value=["hash"],
                            handle=handle,
                            stats_requested_tokens=64,
                            host_indices=None,
                            completed_tokens=0,
                            is_terminated=lambda: False,
                        )
                    )
                    for i, handle in enumerate(handles)
                },
                attn_cp_group=None,
                attn_tp_group=groups[stage],
                tp_group=groups[stage],
                tp_world_size=tp_size,
            )
            now = 10.0 if rank % tp_size == 0 else 0.5
            Wait = prefetch_wait_type(lambda: now)
            cache._storage_prefetch_wait = {}
            for i, handle in enumerate(handles):
                wait = Wait(timeout_s=1.0)
                wait.timer.started_at = 0.0 if i == 0 else 100.0
                cache._storage_prefetch_wait[handle] = wait
            names = (
                "_all_reduce_attn_groups",
                "prefetch_ready_to_finalize",
                "_prefetch_stop_allowed",
                "_start_storage_prefetch_wait",
                "_prefetch_deadline_expired",
            )
            native = functions(
                "mem_cache/unified_radix_cache.py",
                names,
                dict(
                    torch=torch,
                    _StoragePrefetchWait=Wait,
                    pp_cache_admission_enabled=lambda: True,
                ),
                cls="UnifiedRadixCache",
            )
            for name in names:
                setattr(cache, name, lambda *args, fn=native[name]: fn(cache, *args))
            assert cache.prefetch_ready_to_finalize(handles) == [True, False]
            dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
class TransportTests(CustomTestCase):
    def test_tp2pp2_hop_votes_and_tp_clock_agreement(self):
        self.run_transport(2)

    def test_tp1pp4_hop_votes_and_native_control_order(self):
        self.run_transport(1)

    def run_transport(self, tp_size):
        with tempfile.TemporaryDirectory(prefix="pp-admission-gloo-") as directory:
            context = mp.start_processes(
                _transport_worker,
                args=(str(Path(directory) / "init"), tp_size),
                nprocs=4,
                join=False,
                start_method="spawn",
            )
            try:
                deadline = time.monotonic() + 60
                while not context.join(timeout=5):
                    if time.monotonic() >= deadline:
                        self.fail("PP/TP control transport timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
