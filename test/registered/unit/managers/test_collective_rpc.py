"""RPC ingress ownership and two-rank control-plane regression coverage."""

import json
import multiprocessing
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.distributed as dist
import zmq

from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers.io_struct import RpcReqInput, sock_send
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="stage-b-test-cpu-intel")

IPC_MODULE = "sglang.srt.managers.scheduler_components.ipc_channels"
RECEIVER_MODULE = "sglang.srt.managers.scheduler_components.request_receiver"
ENGINE_MODULE = "sglang.srt.entrypoints.engine"


def _parallel(rank=0, local=False, enabled=True):
    return SimpleNamespace(
        pp_rank=0,
        attn_tp_rank=0,
        attn_cp_rank=0,
        attn_dp_enabled=enabled,
        enable_dp_attention_local_control_broadcast=local,
        tp_rank=rank,
        tp_size=2,
        num_dp_ranks=2,
    )


def _ports(directory, rank):
    return SimpleNamespace(
        scheduler_input_ipc_name=f"ipc://{directory}/input-{rank}",
        tokenizer_ipc_name=f"ipc://{directory}/tokenizer",
        detokenizer_ipc_name=f"ipc://{directory}/detokenizer",
        rpc_ipc_name=f"ipc://{directory}/rpc",
        metrics_ipc_name=f"ipc://{directory}/metrics",
    )


def _channels(ports):
    return SchedulerIpcChannels.create(
        port_args=ports,
        is_rank_zero=True,
        skip_tokenizer_init=False,
        metrics_enabled=False,
        enable_scripted_runtime=False,
    )


def _close_channels(channels):
    sockets = [
        channels.recv_from_tokenizer,
        channels.recv_from_rpc,
        channels.send_to_tokenizer.socket,
        channels.send_to_detokenizer.socket,
        channels.send_metrics_from_scheduler,
    ]
    contexts = {socket.context for socket in sockets if socket is not None}
    for socket in sockets:
        if socket is not None:
            socket.close(linger=0)
    for context in contexts:
        context.term()


def _receiver(channels):
    return SchedulerRequestReceiver(
        recv_from_tokenizer=channels.recv_from_tokenizer,
        recv_from_rpc=channels.recv_from_rpc,
        recv_skipper=None,
        input_blocker=None,
        mm_receiver=None,
        server_args=SimpleNamespace(),
        model_config=None,
        max_recv_per_poll=-1,
        request_retirer=None,
        get_last_batch=lambda: None,
        rust_server_mode=False,
    )


def _rpc_worker(rank, directory):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    group = SimpleNamespace(rank=rank, ranks=[0, 1], cpu_group=dist.group.WORLD)
    parallel = _parallel(rank)
    parallel.tp_group = group
    channels = None

    class Target:
        stopped = False

        def record(self, sequence, fail_rank=-1):
            with Path(directory, f"events-{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(sequence) + "\n")
            if rank == fail_rank:
                raise ValueError(f"rank {rank}, sequence {sequence}，执行失败")

        def raise_empty(self):
            raise ValueError()

        def stop(self):
            self.stopped = True

    target = Target()
    target.tp_group = group
    try:
        with (
            patch(f"{IPC_MODULE}.get_parallel", return_value=parallel, create=True),
            patch(f"{RECEIVER_MODULE}.get_parallel", return_value=parallel),
            patch(
                f"{RECEIVER_MODULE}.get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(is_ep_scale_joiner=False)
                ),
            ),
            # Attention TP/CP groups each have one rank in this topology.
            patch(
                f"{RECEIVER_MODULE}.attn_cp_tp_broadcast_pyobj", side_effect=lambda x: x
            ),
        ):
            channels = _channels(_ports(directory, rank))
            receiver = _receiver(channels)
            Path(directory, f"ready-{rank}").touch()
            while not target.stopped:
                requests = receiver._broadcast_reqs_across_ranks(
                    receiver._pull_raw_reqs()
                )
                for request in requests:
                    output = Scheduler.handle_rpc_request(target, request)
                    if channels.recv_from_rpc is not None:
                        sock_send(channels.recv_from_rpc, output)
            # Wait until rank 0 has queued the final response before closing.
            dist.barrier()
            time.sleep(0.1)
    finally:
        if channels is not None:
            _close_channels(channels)
        dist.destroy_process_group()


class TestRpcTransport(CustomTestCase):
    def test_only_global_source_connects_rpc(self):
        with tempfile.TemporaryDirectory() as directory:
            for rank in (0, 1):
                with (
                    self.subTest(rank=rank),
                    patch(
                        f"{IPC_MODULE}.get_parallel",
                        return_value=_parallel(rank),
                        create=True,
                    ),
                ):
                    channels = _channels(_ports(directory, rank))
                    try:
                        self.assertEqual(channels.recv_from_rpc is not None, rank == 0)
                        self.assertIsNotNone(channels.recv_from_tokenizer)
                        self.assertIsNotNone(channels.send_to_tokenizer.socket)
                    finally:
                        _close_channels(channels)

    def test_local_control_has_no_direct_rpc(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                f"{IPC_MODULE}.get_parallel",
                return_value=_parallel(local=True),
                create=True,
            ),
        ):
            channels = _channels(_ports(directory, 0))
            try:
                self.assertIsNone(channels.recv_from_rpc)
            finally:
                _close_channels(channels)

    def test_non_dp_transport_is_preserved(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                f"{IPC_MODULE}.get_parallel",
                return_value=_parallel(enabled=False),
                create=True,
            ),
        ):
            channels = _channels(_ports(directory, 0))
            try:
                self.assertIsNotNone(channels.recv_from_rpc)
            finally:
                _close_channels(channels)

    def test_receiver_without_rpc_still_pulls_tokenizer(self):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind("inproc://rpc-optional")
        sender = context.socket(zmq.PUSH)
        sender.connect("inproc://rpc-optional")
        request = RpcReqInput(method="flush_cache")
        sock_send(sender, request)
        self.assertTrue(socket.poll(1000))
        try:
            receiver = _receiver(
                SimpleNamespace(recv_from_tokenizer=socket, recv_from_rpc=None)
            )
            with patch(f"{RECEIVER_MODULE}.get_parallel", return_value=_parallel(1)):
                self.assertEqual(receiver._pull_raw_reqs(), [request])
        finally:
            sender.close(linger=0)
            socket.close(linger=0)
            context.term()

    def test_local_control_rejected_before_sending(self):
        engine = SimpleNamespace(send_to_rpc=MagicMock())
        with patch(f"{ENGINE_MODULE}.get_parallel", return_value=_parallel(local=True)):
            with self.assertRaisesRegex(ValueError, "local.control.broadcast"):
                Engine.collective_rpc(engine, "flush_cache")
        self.assertEqual(engine.send_to_rpc.mock_calls, [])

    def test_idle_sleeper_without_rpc_socket(self):
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        socket.bind("inproc://rpc-idle")
        scheduler = SimpleNamespace(
            ipc_channels=SimpleNamespace(
                recv_from_tokenizer=socket, recv_from_rpc=None
            ),
            pause_state=SimpleNamespace(paused=False),
        )
        try:
            with (
                patch(
                    "sglang.srt.managers.scheduler.get_parallel",
                    return_value=_parallel(1),
                ),
                patch(
                    "sglang.srt.managers.scheduler.get_device",
                    return_value=SimpleNamespace(sleep_on_idle=True),
                ),
            ):
                Scheduler.init_idle_sleeper(scheduler)
            self.assertEqual(scheduler.idle_sleeper.poller.poll(0), [])
        finally:
            socket.close(linger=0)
            context.term()


class TestRpcGloo(CustomTestCase):
    def test_repeated_calls_failure_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            context = zmq.Context()
            socket = context.socket(zmq.DEALER)
            socket.setsockopt(zmq.RCVTIMEO, 5000)
            socket.bind(_ports(directory, 0).rpc_ipc_name)
            engine = SimpleNamespace(send_to_rpc=socket)
            processes = []
            try:
                ctx = multiprocessing.get_context("spawn")
                for rank in (1, 0):
                    process = ctx.Process(target=_rpc_worker, args=(rank, directory))
                    process.start()
                    processes.append(process)
                deadline = time.monotonic() + 45
                while not all(Path(directory, f"ready-{r}").exists() for r in (0, 1)):
                    self.assertTrue(all(p.is_alive() for p in processes))
                    self.assertLess(
                        time.monotonic(), deadline, "workers did not initialize"
                    )
                    time.sleep(0.05)
                with patch(f"{ENGINE_MODULE}.get_parallel", return_value=_parallel()):
                    for sequence in range(12):
                        fail_rank = 1 if sequence % 3 == 1 else -1
                        if fail_rank == 1:
                            with self.assertRaisesRegex(
                                AssertionError, f"sequence {sequence}"
                            ):
                                Engine.collective_rpc(
                                    engine, "record", sequence=sequence, fail_rank=1
                                )
                        else:
                            Engine.collective_rpc(engine, "record", sequence=sequence)
                        self.assertFalse(
                            socket.poll(20), "unexpected extra RPC response"
                        )
                    with self.assertRaisesRegex(AssertionError, "missing_method"):
                        Engine.collective_rpc(engine, "missing_method")
                    with self.assertRaises(AssertionError):
                        Engine.collective_rpc(engine, "raise_empty")
                    Engine.collective_rpc(engine, "stop")
                for process in processes:
                    process.join(10)
                    self.assertEqual(process.exitcode, 0)
                for rank in (0, 1):
                    events = (
                        Path(directory, f"events-{rank}.jsonl").read_text().splitlines()
                    )
                    self.assertEqual([json.loads(x) for x in events], list(range(12)))
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)
                    if process.is_alive():
                        process.kill()
                        process.join(5)
                socket.close(linger=0)
                context.term()


if __name__ == "__main__":
    unittest.main()
