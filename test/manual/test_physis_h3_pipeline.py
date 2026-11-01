# SPDX-License-Identifier: Apache-2.0
"""Four-GPU Physis H3 request distribution, failure propagation and recovery.

Run: python test/manual/test_physis_h3_pipeline.py
"""

import os
import pickle
import socket
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import zmq

from sglang.multimodal_gen.runtime.distributed.causal_pipeline import CausalPipelinePlan
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_world_group,
    init_distributed_environment,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.pipelines.physis_h3_pipeline import (
    PhysisH3DecodingStage,
    _distribute_tensor,
    _run_wavefront,
    _synchronized_component_scope,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.test.test_utils import CustomTestCase


class Runner:
    shape = (1, 1, 1, 2, 2)
    padded_tokens = 5
    model = SimpleNamespace(arch=SimpleNamespace(hidden_size=8))

    def __init__(self, rank, fail=False):
        self.rank = rank
        self.fail = fail
        self.device = torch.device("cuda", rank)

    def run_chunk(self, chunk, slices, latent=None, hidden=None):
        if self.fail and self.rank == 1 and chunk == 6:
            raise ValueError("injected chunk failure")
        if self.rank:
            torch.testing.assert_close(
                hidden,
                torch.full_like(hidden, chunk + self.rank - 1),
                rtol=0,
                atol=0,
            )
        else:
            latent = torch.full(
                self.shape, chunk, device=self.device, dtype=torch.bfloat16
            )
        hidden = (
            torch.full(
                (self.padded_tokens, 8),
                chunk + self.rank,
                device=self.device,
                dtype=torch.bfloat16,
            )
            if self.rank < 2
            else None
        )
        return latent + self.rank + 1, hidden


def check_request_broadcast(rank):
    """Every pipeline rank receives the same requests and idle polls."""
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(
        causal_pipeline_mode="block",
        comfyui_mode=False,
        sp_degree=1,
        enable_cfg_parallel=False,
        tp_size=1,
    )
    scheduler.receiver = None
    scheduler._socket_lock = threading.Lock()
    payload = {"request_id": "physis-h3"}
    with zmq.Context() as context:
        with (
            context.socket(zmq.ROUTER) as receiver,
            context.socket(zmq.DEALER) as client,
        ):
            if rank == 0:
                receiver.bind("inproc://physis-h3")
                client.setsockopt(zmq.IDENTITY, b"client")
                client.connect("inproc://physis-h3")
                client.send_multipart([b"", pickle.dumps(payload)])
                assert receiver.poll(1000), "Scheduler did not receive the request"
                scheduler.receiver = receiver
            assert scheduler.recv_reqs() == [(b"client", payload)]
            assert scheduler.recv_reqs() == []


def rank_main(rank, port, folder):
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=4,
        rank=rank,
        local_rank=rank,
        backend="nccl",
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        device_id=torch.device("cuda", rank),
    )
    world = get_world_group()
    dist.barrier(group=world.device_group)
    check_request_broadcast(rank)
    root = Path(folder)
    plan = CausalPipelinePlan(4, 3, 3, "block")

    def failed(call, message):
        try:
            call()
        except RuntimeError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError("Failure was not propagated to every rank")

    def healthy_wave():
        result = _run_wavefront(Runner(rank) if rank < 3 else None, plan, rank, 8, 5)
        if rank == 2:
            assert len(result) == 3
            for chunk, tensor in zip(range(5, 8), result):
                torch.testing.assert_close(tensor, torch.full_like(tensor, chunk + 6))
        else:
            assert result == []

    failed(
        lambda: _run_wavefront(
            Runner(rank, fail=True) if rank < 3 else None, plan, rank, 8, 5
        ),
        "injected chunk failure",
    )
    healthy_wave()

    real_empty = torch.empty

    def failed_allocation(*args, **kwargs):
        device = kwargs.get("device")
        if rank == 2 and device is not None and torch.device(device).type == "cuda":
            raise torch.cuda.OutOfMemoryError("injected receive allocation failure")
        return real_empty(*args, **kwargs)

    with patch.object(torch, "empty", side_effect=failed_allocation):
        failed(
            lambda: _run_wavefront(
                Runner(rank) if rank < 3 else None, plan, rank, 8, 5
            ),
            "injected receive allocation failure",
        )
    healthy_wave()

    for destination in (None, 0):
        payload = (
            torch.arange(16, device="cuda", dtype=torch.float32) if rank == 3 else None
        )

        def failed_destination(*args, **kwargs):
            if rank == 0 and kwargs.get("device") == "cuda":
                raise torch.cuda.OutOfMemoryError(
                    "injected destination allocation failure"
                )
            return real_empty(*args, **kwargs)

        with patch.object(torch, "empty", side_effect=failed_destination):
            failed(
                lambda: _distribute_tensor(payload, src=3, dst=destination),
                "injected destination allocation failure",
            )
        output = _distribute_tensor(payload, src=3, dst=destination)
        if destination is None or rank == destination:
            torch.testing.assert_close(output, torch.arange(16, device="cuda").float())
        else:
            assert output is None

    for phase in ("enter", "exit"):

        @contextmanager
        def component_scope():
            if rank == 1 and phase == "enter":
                raise RuntimeError("injected component enter failure")
            try:
                yield
            finally:
                if rank == 2 and phase == "exit":
                    raise RuntimeError("injected component exit failure")

        def enter_scope():
            with _synchronized_component_scope(component_scope()):
                pass

        failed(enter_scope, f"injected component {phase} failure")
        healthy_wave()

    rolled_back = []

    @contextmanager
    def rollback_scope():
        try:
            yield
        except RuntimeError:
            rolled_back.append(True)
            raise

    def failed_body():
        with _synchronized_component_scope(rollback_scope()):
            raise RuntimeError("coordinated body failure")

    failed(failed_body, "coordinated body failure")
    assert rolled_back == [True], "Component scope lost the forward exception"

    class VAE:
        config = SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])

        def __init__(self):
            self.fail = True

        def decode(self, latent):
            assert not torch.is_autocast_enabled("cuda")
            if self.fail:
                raise ValueError("injected VAE failure")
            return SimpleNamespace(sample=torch.zeros(1, 3, 2, 4, 4, device="cuda"))

    set_global_server_args(SimpleNamespace(comfyui_mode=False))
    vae = VAE()
    pipeline = SimpleNamespace(
        decoder_rank=3,
        get_module=lambda name: vae if rank == 3 else None,
    )
    stage = PhysisH3DecodingStage(pipeline)
    args = SimpleNamespace(
        disable_autocast=True,
        component_precisions={},
        pipeline_config=SimpleNamespace(vae_decode_precision="fp16"),
    )
    batch = Req(num_frames=2, fps=24)
    if rank == 3:
        batch.latents = torch.zeros(1, 1, 1, 1, 1, device="cuda")
    failed(lambda: stage.forward(batch, args), "injected VAE failure")
    vae.fail = False
    result = stage.forward(batch, args)
    if rank == 0:
        assert tuple(result.output.shape) == (1, 3, 2, 4, 4)
    else:
        assert result.output is None
    (root / f"finished-{rank}").touch()
    dist.destroy_process_group()


class TestPhysisPipelineFailures(CustomTestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA GPUs")
    def test_rank_local_failure_then_success(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory(prefix="physis-failure-") as folder:
            context = mp.spawn(rank_main, args=(port, folder), nprocs=4, join=False)
            try:
                deadline = time.monotonic() + 120
                while not context.join(timeout=1):
                    self.assertLess(
                        time.monotonic(), deadline, "rank failure recovery hung"
                    )
                self.assertEqual(len(list(Path(folder).glob("finished-*"))), 4)
                print("PHYSIS_FAILURE_RECOVERY_PASS", flush=True)
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
