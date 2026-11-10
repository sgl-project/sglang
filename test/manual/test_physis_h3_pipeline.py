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
    initialize_model_parallel,
)
from sglang.multimodal_gen.runtime.managers import request_cancel
from sglang.multimodal_gen.runtime.managers.request_cancel_error import (
    RequestCancelledError,
)
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3Rope
from sglang.multimodal_gen.runtime.pipelines.physis_h3_pipeline import (
    PhysisH3DecodingStage,
    PhysisH3DenoisingStage,
    PhysisH3Pipeline,
    _distribute_tensor,
    _run_wavefront,
    _synchronized_component_scope,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
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

    pipeline = object.__new__(PhysisH3Pipeline)

    def load_components(*args, **kwargs):
        if rank == 0:
            raise ValueError("injected VAE loading failure")
        return {"rank": rank}

    with patch.object(
        ComposedPipelineBase, "load_modules", side_effect=load_components
    ):
        failed(lambda: pipeline.load_modules(None), "injected VAE loading failure")
    with patch.object(
        ComposedPipelineBase, "load_modules", return_value={"rank": rank}
    ):
        assert pipeline.load_modules(None) == {"rank": rank}

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
        def __init__(self):
            self.fail = True

        def decode_base(self, latent):
            assert not torch.is_autocast_enabled("cuda")
            if self.fail:
                raise ValueError("injected VAE failure")
            return torch.zeros(1, 3, 2, 4, 4, device="cuda")

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
        pipeline_config=SimpleNamespace(
            vae_decode_precision="fp16",
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(latents_mean=[0.0], latents_std=[1.0])
            ),
        ),
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


class TinyModel:
    arch = SimpleNamespace(latents_dim=1, hidden_size=8, patch_size=(1, 2, 2))

    def __init__(self):
        self.rope = MiniMaxH3Rope(1).cuda()
        self.blocks = [lambda hidden, **kwargs: hidden for _ in range(3)]

    def refine_prompt_embeds(self, embeds, *args, **kwargs):
        return embeds

    def _time_embedding(self, timestep):
        return timestep.new_zeros((1, 8))

    def video_patch_proj(self, rows):
        return rows.new_zeros((rows.shape[0], 8)), None

    def final_layer(self, hidden, **kwargs):
        return hidden.new_zeros((hidden.shape[0], 4)), None


def rank_cancel_main(rank, port, folder, mode):
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
    parallel = mode != "off"
    initialize_model_parallel(
        sequence_parallel_degree=1 if parallel else 4,
        ulysses_degree=1 if parallel else 4,
        pipeline_parallel_degree=4 if parallel else 1,
        separate_decode_ranks=parallel,
    )
    args = SimpleNamespace(
        comfyui_mode=False,
        causal_pipeline_mode=mode,
        causal_pipeline_parallel_size=3 if parallel else 1,
        pipeline_config=SimpleNamespace(
            physis_bootstrap_chunks=5, physis_history_chunks=18
        ),
    )
    set_global_server_args(args)
    model = TinyModel() if not parallel or rank < 3 else None
    pipeline = SimpleNamespace(
        decoder_rank=3 if parallel else 0,
        get_module=lambda name: model,
    )
    stage = PhysisH3DenoisingStage(pipeline)
    original_check = stage.check_cancelled
    checks = 0

    def check(batch):
        nonlocal checks
        checks += 1
        original_check(batch)

    stage.check_cancelled = check
    for cancel_at in (3, 5, 8, None):
        checks = 0
        batch = Req(
            num_frames=22,
            height=32,
            width=32,
            seed=42,
            request_id="cancel-test",
            cancellable=True,
        )
        batch.prompt_embeds = [torch.ones(2, 8, device="cuda", dtype=torch.bfloat16)]
        listener = SimpleNamespace(
            is_cancelled=lambda ids: cancel_at is not None and checks >= cancel_at
        )
        with (
            patch.object(request_cancel, "_enabled", True),
            patch.object(request_cancel, "_listener", listener if rank == 0 else None),
        ):
            if cancel_at is None:
                result = stage.forward(batch, args)
                if rank == pipeline.decoder_rank:
                    assert tuple(result.latents.shape) == (1, 1, 7, 2, 2)
            else:
                try:
                    stage.forward(batch, args)
                except RequestCancelledError:
                    assert checks == cancel_at, (rank, checks, cancel_at)
                else:
                    raise AssertionError("Physis denoising ignored cancellation")
        get_world_group().barrier()
    Path(folder, f"finished-{rank}").touch()
    dist.destroy_process_group()


def rank_oom_main(rank, port, entered, escaped):
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        backend="nccl",
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        device_id=torch.device("cuda", rank),
    )
    initialize_model_parallel(sequence_parallel_degree=2, ulysses_degree=2)
    args = SimpleNamespace(
        comfyui_mode=False,
        causal_pipeline_mode="off",
        causal_pipeline_parallel_size=1,
        pipeline_config=SimpleNamespace(
            physis_bootstrap_chunks=5, physis_history_chunks=18
        ),
    )
    set_global_server_args(args)
    model = TinyModel()
    original_final_layer = model.final_layer

    def final_layer(hidden, **kwargs):
        dist.barrier(group=get_world_group().cpu_group)
        if rank == 0:
            entered.set()
            raise torch.cuda.OutOfMemoryError("injected Ulysses allocation failure")
        return original_final_layer(hidden, **kwargs)

    model.final_layer = final_layer
    stage = PhysisH3DenoisingStage(SimpleNamespace(get_module=lambda name: model))
    batch = Req(num_frames=22, height=32, width=32, seed=42)
    batch.prompt_embeds = [torch.ones(2, 8, device="cuda", dtype=torch.bfloat16)]
    try:
        stage.forward(batch, args)
    except torch.cuda.OutOfMemoryError:
        escaped.set()
        os._exit(23)
    raise AssertionError("The injected OOM did not reach worker failure handling")


class TestPhysisPipelineFailures(CustomTestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA GPUs")
    def test_ulysses_oom_reaches_worker_failure_handling(self):
        """A failed rank escapes promptly while peers wait in its forward collective."""
        spawn = mp.get_context("spawn")
        entered, escaped = spawn.Event(), spawn.Event()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        context = mp.spawn(
            rank_oom_main, args=(port, entered, escaped), nprocs=2, join=False
        )
        try:
            self.assertTrue(entered.wait(90), "workers did not reach the injected OOM")
            self.assertTrue(escaped.wait(10), "Ulysses OOM waited in component cleanup")
            context.processes[0].join(timeout=5)
            self.assertEqual(context.processes[0].exitcode, 23)
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)

    @unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA GPUs")
    def test_cancellation_at_sampling_boundaries(self):
        """Every rank stops together during bootstrap, sink setup and later sampling."""
        for mode in ("off", "block"):
            with self.subTest(mode=mode), socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            with tempfile.TemporaryDirectory(prefix="physis-cancel-") as folder:
                context = mp.spawn(
                    rank_cancel_main, args=(port, folder, mode), nprocs=4, join=False
                )
                try:
                    deadline = time.monotonic() + 120
                    while not context.join(timeout=1):
                        self.assertLess(
                            time.monotonic(), deadline, "cancellation hung a rank"
                        )
                    self.assertEqual(len(list(Path(folder).glob("finished-*"))), 4)
                finally:
                    for process in context.processes:
                        if process.is_alive():
                            process.terminate()
                    for process in context.processes:
                        process.join(timeout=5)
                        if process.is_alive():
                            process.kill()
                            process.join(timeout=5)

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
