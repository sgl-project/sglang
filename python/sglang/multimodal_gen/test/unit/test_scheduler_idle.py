# SPDX-License-Identifier: Apache-2.0
"""Idle receive regressions; real ZMQ/Gloo coverage needs no GPUs or model weights."""

import math
import multiprocessing as mp
import pickle
import time
from collections import deque
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import zmq

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.control_requests import (
    GetDisaggStatsReq,
    ShutdownReq,
)
from sglang.multimodal_gen.runtime.managers import scheduler as scheduler_module
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req


def _make_scheduler(receiver=None):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.context = Mock()
    scheduler.receiver = receiver
    scheduler._poller = Mock()
    scheduler.dp_replica = 0
    scheduler.gpu_id = 0
    scheduler.server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(),
        sp_degree=1,
        enable_cfg_parallel=False,
        tp_size=1,
        comfyui_mode=False,
    )
    scheduler._disagg_role = RoleType.MONOLITHIC
    scheduler._disagg_metrics = None
    scheduler.metrics = None
    scheduler.waiting_queue = deque()
    scheduler._running = True
    scheduler._consecutive_error_count = 0
    scheduler._max_consecutive_errors = 3
    scheduler._batch_admission = SimpleNamespace(enabled=False)
    scheduler._batch_metrics_enabled = False
    scheduler._log_batch_metrics_summary = Mock()
    scheduler._cleanup_disagg = Mock()
    scheduler._return_item_result = Mock()
    scheduler.process_received_reqs_with_req_based_warmup = lambda reqs: reqs
    scheduler.request_handlers = {
        ShutdownReq: scheduler._handle_shutdown,
        GetDisaggStatsReq: scheduler._handle_get_disagg_stats,
        Req: lambda reqs: OutputBatch(output=reqs[0].request_id),
    }
    return scheduler


@pytest.mark.parametrize("gpu_id", [0, 2])
def test_idle_ingress_polls_before_each_receive(gpu_id):
    scheduler = _make_scheduler(receiver=Mock())
    scheduler.gpu_id = gpu_id
    scheduler.dp_replica = gpu_id // 2
    events = []
    scheduler._poller.poll.side_effect = lambda **kw: events.append(("poll", kw))

    def receive():
        events.append(("receive", {}))
        if sum(event[0] == "receive" for event in events) == 3:
            scheduler._running = False
        return []

    scheduler.recv_reqs = receive
    scheduler.event_loop()

    assert events == [("poll", {"timeout": 100}), ("receive", {})] * 3


def test_idle_peer_keeps_receiving_without_polling():
    scheduler = _make_scheduler()
    scheduler._poller.poll.side_effect = AssertionError("A peer has no ZMQ ingress")
    calls = 0

    def receive():
        nonlocal calls
        calls += 1
        if calls == 3:
            scheduler._running = False
        return []

    scheduler.recv_reqs = receive
    scheduler.event_loop()
    assert calls == 3
    scheduler._poller.poll.assert_not_called()


def test_queued_work_does_not_wait_for_another_message():
    scheduler = _make_scheduler(receiver=Mock())
    request = ShutdownReq()
    scheduler.waiting_queue.append((b"client", request, time.monotonic()))
    scheduler.recv_reqs = Mock(return_value=[])

    scheduler.event_loop()

    scheduler._poller.poll.assert_not_called()
    scheduler._return_item_result.assert_called_once()
    assert scheduler._return_item_result.call_args.args[0] == (b"client", request)
    assert not scheduler.waiting_queue
    assert not scheduler._running


def test_shutdown_received_after_idle_wait_is_dispatched():
    scheduler = _make_scheduler(receiver=Mock())
    request = ShutdownReq()
    scheduler.recv_reqs = Mock(return_value=[(b"client", request)])

    scheduler.event_loop()

    scheduler._poller.poll.assert_called_once_with(timeout=100)
    scheduler.recv_reqs.assert_called_once_with()
    scheduler._return_item_result.assert_called_once()
    assert not scheduler._running


def test_idle_poll_errors_use_receive_error_handling():
    scheduler = _make_scheduler(receiver=Mock())
    scheduler._poller.poll.side_effect = zmq.ZMQError(zmq.ETERM)
    scheduler.recv_reqs = Mock(return_value=[(b"client", ShutdownReq())])

    with pytest.raises(RuntimeError, match="3 consecutive errors"):
        scheduler.event_loop()

    assert scheduler._poller.poll.call_count == 3
    scheduler.recv_reqs.assert_not_called()


@pytest.mark.parametrize("ingress", [False, True])
def test_batching_wait_keeps_its_remaining_deadline(ingress, monkeypatch):
    scheduler = _make_scheduler(receiver=Mock() if ingress else None)
    request = Req(sampling_params=SamplingParams(prompt="test"))
    scheduler.waiting_queue.append((b"client", request, 9.997))
    scheduler._batching_delay_s = 0.010
    scheduler._dynamic_batching_enabled = Mock(return_value=True)
    scheduler.recv_reqs = Mock(return_value=[])
    scheduler.get_next_batch_to_run = Mock(
        side_effect=[None, [(b"client", ShutdownReq())]]
    )
    monkeypatch.setattr(scheduler_module.time, "monotonic", lambda: 10.0)
    sleep = Mock()
    monkeypatch.setattr(scheduler_module.time, "sleep", sleep)

    scheduler.event_loop()

    if ingress:
        scheduler._poller.poll.assert_called_once_with(timeout=pytest.approx(7))
        sleep.assert_not_called()
    else:
        scheduler._poller.poll.assert_not_called()
        sleep.assert_called_once_with(pytest.approx(0.007))


def test_disaggregated_loop_is_unchanged():
    scheduler = _make_scheduler(receiver=Mock())
    scheduler._disagg_role = RoleType.DENOISER
    scheduler._disagg_event_loop = Mock()
    scheduler.recv_reqs = Mock()

    scheduler.event_loop()

    scheduler._disagg_event_loop.assert_called_once_with()
    scheduler._poller.poll.assert_not_called()
    scheduler.recv_reqs.assert_not_called()


def _distributed_worker(rank, rendezvous, pipe, parallelism):
    torch.set_num_threads(1)
    context = zmq.Context()
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=20),
        )
        receiver = context.socket(zmq.ROUTER) if rank == 0 else None
        endpoint = None
        if receiver is not None:
            port = receiver.bind_to_random_port("tcp://127.0.0.1")
            endpoint = f"tcp://127.0.0.1:{port}"
        scheduler = _make_scheduler(receiver)
        scheduler.context = context
        scheduler.gpu_id = rank
        scheduler._poller = zmq.Poller()
        if receiver is not None:
            scheduler._poller.register(receiver, zmq.POLLIN)
        scheduler.server_args.sp_degree = 2 if parallelism == "sp" else 1
        scheduler.server_args.enable_cfg_parallel = parallelism == "cfg"
        scheduler.server_args.tp_size = 2 if parallelism == "tp" else 1
        group = SimpleNamespace(rank=rank, ranks=[0, 1])
        scheduler.worker = SimpleNamespace(
            sp_group=group,
            sp_cpu_group=dist.group.WORLD,
            cfg_group=group,
            cfg_cpu_group=dist.group.WORLD,
            tp_group=group,
            tp_cpu_group=dist.group.WORLD,
        )
        empty_receives = 0
        observed = []
        receive = scheduler.recv_reqs

        def counted_receive():
            nonlocal empty_receives
            requests = receive()
            if not requests:
                empty_receives += 1
            observed.extend(
                req.request_id if isinstance(req, Req) else type(req).__name__
                for _, req in requests
            )
            return requests

        scheduler.recv_reqs = counted_receive

        def reply(item, output):
            if receiver is not None:
                receiver.send_multipart([item[0], b"", pickle.dumps(output.output)])

        scheduler._return_item_result = reply
        dist.barrier()
        started = time.monotonic()
        pipe.send(endpoint)
        scheduler.event_loop()
        pipe.send((observed, empty_receives, time.monotonic() - started))
    finally:
        context.destroy(linger=0)
        if dist.is_initialized():
            dist.destroy_process_group()
        pipe.close()


def _receive_worker(pipe):
    assert pipe.poll(30), "The CPU-only scheduler worker timed out"
    return pipe.recv()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("parallelism", ["sp", "cfg", "tp"])
def test_two_idle_ranks_wake_and_shutdown_in_order(tmp_path, monkeypatch, parallelism):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    spawn = mp.get_context("spawn")
    context = zmq.Context()
    client = context.socket(zmq.REQ)
    client.setsockopt(zmq.RCVTIMEO, 10000)
    client.setsockopt(zmq.SNDTIMEO, 10000)
    processes = []
    pipes = []
    try:
        for rank in range(2):
            parent, child = spawn.Pipe()
            process = spawn.Process(
                target=_distributed_worker,
                args=(rank, str(tmp_path / "gloo"), child, parallelism),
            )
            process.start()
            child.close()
            processes.append(process)
            pipes.append(parent)
        endpoint = _receive_worker(pipes[0])
        assert _receive_worker(pipes[1]) is None
        client.connect(endpoint)

        for cycle in range(2):
            time.sleep(0.25)
            client.send_pyobj(GetDisaggStatsReq())
            assert client.recv_pyobj()["role"] == "monolithic"
            for index in range(3):
                request_id = f"{cycle}-{index}"
                client.send_pyobj(
                    [
                        Req(
                            request_id=request_id,
                            sampling_params=SamplingParams(prompt="test"),
                        )
                    ]
                )
                assert client.recv_pyobj() == request_id

        time.sleep(0.25)
        client.send_pyobj(ShutdownReq())
        assert client.recv_pyobj() is None
        results = [_receive_worker(pipe) for pipe in pipes]
        expected = [
            "GetDisaggStatsReq",
            "0-0",
            "0-1",
            "0-2",
            "GetDisaggStatsReq",
            "1-0",
            "1-1",
            "1-2",
            "ShutdownReq",
        ]
        for observed, empty_receives, elapsed in results:
            assert observed == expected
            assert empty_receives <= math.ceil(elapsed / 0.1) + 2
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        client.close(linger=0)
        context.term()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
        for pipe in pipes:
            pipe.close()
