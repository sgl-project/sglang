# SPDX-License-Identifier: Apache-2.0
"""CPU upload ownership, preparation lifecycle, and TP readiness consensus."""

import multiprocessing
import tempfile
import threading
import unittest
from contextlib import nullcontext
from contextvars import ContextVar
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.srt.lora.cpu_preparation import LoRAPreparationQueue
from sglang.srt.lora.cpu_upload import build_upload_plan
from sglang.srt.lora.lora_overlap_loader import LoRAOverlapLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def request(uid):
    return SimpleNamespace(lora_id=uid, to_ref=lambda: uid)


class Event:
    def __init__(self, ready):
        self.ready = ready

    def query(self):
        return self.ready


def loader_for(events, slots):
    loader = object.__new__(LoRAOverlapLoader)
    loader.completed_loras = set()
    loader.wanted_loras = None
    loader.lora_manager = SimpleNamespace(
        memory_pool=SimpleNamespace(uid_to_buffer_id=slots)
    )
    loader.lora_to_overlap_load_event = events
    loader.device_module = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(wait_event=lambda event: None)
    )
    return loader


def _consensus_rank(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    release = threading.Event()
    queue = None
    try:
        torch.set_default_device("meta")
        events = {None: Event(True), "a": Event(True), "b": Event(rank == 0)}
        if rank:
            events = dict(reversed(list(events.items())))
        loader = loader_for(events, {None: 0, "a": 1, "b": 2})
        with patch(
            "sglang.srt.lora.lora_overlap_loader.get_parallel",
            return_value=SimpleNamespace(tp_group=SimpleNamespace(cpu_group=None)),
        ):
            loader._drain_completed_overlap_loads()
            assert set(events) == {"b"}
            assert loader.completed_loras == {None, "a"}
            events["b"].ready = True
            loader._drain_completed_overlap_loads()
            assert not events

        def prepare(ref):
            if rank:
                assert release.wait(10)
            return "config", "weights"

        queue = LoRAPreparationQueue(prepare)
        queue.submit(request("adapter"))
        if rank == 0:
            queue.pending[0][1].result(timeout=5)
        dist.barrier()
        assert queue.poll(lambda req, value: None, None) is None
        release.set()
        queue.pending[0][1].result(timeout=5)

        def validate(req, prepared):
            if rank:
                raise ValueError("incompatible rank")

        completed = queue.poll(validate, None)
        assert completed[2] == "ValueError: incompatible rank"
        assert not queue
    finally:
        release.set()
        if queue is not None:
            queue.close()
        torch.set_default_device("cpu")
        dist.destroy_process_group()


class TestLoRACPUPreparation(CustomTestCase):
    def test_preparation_keeps_context_and_commits_only_after_completion(self):
        context = ContextVar("adapter-context", default="missing")
        context.set("request-context")
        release, entered = threading.Event(), threading.Event()
        main_thread = threading.get_ident()

        def prepare(ref):
            self.assertNotEqual(threading.get_ident(), main_thread)
            entered.set()
            self.assertTrue(release.wait(5))
            return ref, context.get()

        queue = LoRAPreparationQueue(prepare)
        try:
            queue.submit(request("adapter"))
            self.assertTrue(entered.wait(5))
            self.assertIsNone(queue.poll(lambda req, value: None, None))
            release.set()
            queue.pending[0][1].result(timeout=5)

            def validate(req, prepared):
                self.assertEqual(threading.get_ident(), main_thread)
                self.assertEqual(prepared, ("adapter", "request-context"))

            completed = queue.poll(validate, None)
            self.assertIsNone(completed[2])
            self.assertFalse(queue)
        finally:
            release.set()
            queue.close()

    def test_failed_or_unloaded_preparation_does_not_prevent_next_load(self):
        for failure in ("read", "validate", "unload"):
            with self.subTest(failure=failure):

                def prepare(ref):
                    if failure == "read" and ref == "bad":
                        raise FileNotFoundError("adapter weights")
                    return ref

                def validate(req, value):
                    if failure == "validate":
                        raise ValueError("adapter rank")

                queue = LoRAPreparationQueue(prepare)
                try:
                    queue.submit(request("bad"))
                    try:
                        queue.pending[0][1].result(timeout=5)
                    except FileNotFoundError:
                        pass
                    if failure == "unload":
                        queue.cancel("bad")
                    completed = queue.poll(validate, None)
                    self.assertIsNotNone(completed[2])
                    self.assertFalse(queue)
                    self.assertFalse(queue.cancelled)
                    queue.submit(request("good"))
                    queue.pending[0][1].result(timeout=5)
                    recovered = queue.poll(lambda req, value: None, None)
                    self.assertEqual(recovered[1:], ("good", None))
                finally:
                    queue.close()

    def test_completed_upload_is_reserved_until_admission_or_cancellation(self):
        loader = loader_for({"a": Event(True), "b": Event(True)}, {"a": 0, "b": 1})
        loader.prepare_admission({"a", "b"})
        loader._drain_completed_overlap_loads()
        self.assertEqual(loader.completed_loras, {"a", "b"})
        self.assertTrue(loader.try_overlap_load_lora("b", set()))
        self.assertEqual(loader.completed_loras, {"a"})
        loader.prepare_admission({"b"})
        self.assertFalse(loader.completed_loras)

    def test_unload_cancels_queued_and_running_preparation(self):
        for cancelled in ("running", "queued"):
            with self.subTest(cancelled=cancelled):
                entered, release = threading.Event(), threading.Event()

                def prepare(ref):
                    if ref == "running":
                        entered.set()
                        self.assertTrue(release.wait(5))
                    return ref

                queue = LoRAPreparationQueue(prepare)
                try:
                    queue.submit(request("running"))
                    queue.submit(request("queued"))
                    self.assertTrue(entered.wait(5))
                    queue.cancel(cancelled)
                    self.assertIsNone(queue.poll(lambda req, value: None, None))
                    release.set()
                    for name in ("running", "queued"):
                        future = queue.pending[0][1]
                        if not future.cancelled():
                            future.result(timeout=5)
                        result = queue.poll(lambda req, value: None, None)
                        self.assertEqual(result[0].lora_id, name)
                        self.assertEqual(result[2] is not None, name == cancelled)
                    self.assertFalse(queue)
                finally:
                    release.set()
                    queue.close()

    def test_slot_pressure_admits_each_adapter_without_repeated_uploads(self):
        class DelayedEvent:
            def __init__(self):
                self.polls = 0

            def query(self):
                self.polls += 1
                return self.polls > 1

            def record(self, stream):
                pass

        for queued in (4, 8):
            with self.subTest(queued=queued):
                slots, uploads = {None: 0}, []

                def fetch(new, protected):
                    for uid in new:
                        if len(slots) == 4:
                            victims = [v for v in slots if v not in protected]
                            del slots[
                                next((v for v in victims if v is not None), victims[0])
                            ]
                        slots[uid] = 1
                        uploads.append(uid)

                device = SimpleNamespace(
                    Stream=lambda: None,
                    stream=lambda stream: nullcontext(),
                    Event=DelayedEvent,
                    current_stream=lambda: SimpleNamespace(
                        wait_event=lambda event: None
                    ),
                )
                manager = SimpleNamespace(
                    device="cuda",
                    pending_lora_load_events={},
                    memory_pool=SimpleNamespace(uid_to_buffer_id=slots),
                    validate_lora_batch=lambda ids: len(ids) <= 4,
                    fetch_new_loras=fetch,
                )
                with patch("torch.get_device_module", return_value=device):
                    loader = LoRAOverlapLoader(manager)
                waiting, running, admitted = list("abcdefgh"[:queued]), set(), set()
                for _ in range(12):
                    loader.prepare_admission(set(waiting) | running)
                    for uid in list(waiting):
                        if loader.try_overlap_load_lora(uid, running):
                            admitted.add(uid)
                            running.add(uid)
                            waiting.remove(uid)
                    if len(running) == 4:
                        running.clear()
                self.assertFalse(waiting)
                self.assertEqual(admitted, set("abcdefgh"[:queued]))
                self.assertEqual(len(uploads), queued)

    def test_upload_plan_owns_weights_and_validates_all_buffers_before_copy(self):
        class Pool:
            def __init__(self):
                self.A_buffer = {"q": [torch.full((2, 4, 3), 37.0)]}
                self.B_buffer = {"q": [torch.full((2, 5, 4), 37.0)]}
                self.embedding_A_buffer = {}
                self.embedding_B_buffer = {}
                self.lm_head_A_buffer = {}
                self.lm_head_B_buffer = {}
                self.new_embeddings_buffer = {}

        def load(pool, uid, slot, adapter, modules, embed, head):
            pool.A_buffer["q"][0][slot, :2].copy_(adapter.a)
            pool.B_buffer["q"][0][slot, :, :2].copy_(adapter.b)

        pool = Pool()
        manager = SimpleNamespace(
            memory_pool=pool,
            device="cuda:0",
            lora_modules=[],
            embed_tokens_module=None,
            lm_head_module=None,
        )
        adapter = SimpleNamespace(uid="a", a=torch.ones(2, 3), b=torch.ones(5, 2))
        zeros = torch.zeros

        def allocate(*args, **kwargs):
            kwargs.pop("pin_memory")
            return zeros(*args, **kwargs)

        with (
            patch("torch.zeros", side_effect=allocate),
            patch("torch.cuda.device", return_value=nullcontext()),
        ):
            plan = build_upload_plan(manager, adapter, load)
        self.assertTrue(torch.all(pool.A_buffer["q"][0] == 37))
        adapter.a.fill_(-1)
        adapter.b.fill_(-1)
        self.assertTrue(plan.copy_to(pool, 1))
        self.assertTrue(torch.all(pool.A_buffer["q"][0][1, :2] == 1))
        self.assertFalse(pool.A_buffer["q"][0][1, 2:].count_nonzero())
        self.assertTrue(torch.all(pool.B_buffer["q"][0][1, :, :2] == 1))
        self.assertFalse(pool.B_buffer["q"][0][1, :, 2:].count_nonzero())
        self.assertFalse(plan.copy_to(Pool(), 0))
        pool.B_buffer["q"][0] = torch.full((2, 6, 4), 47.0)
        self.assertFalse(plan.copy_to(pool, 0))
        self.assertTrue(torch.all(pool.A_buffer["q"][0][0] == 37))

    def test_two_rank_readiness_and_failure_consensus(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            processes = [
                context.Process(
                    target=_consensus_rank, args=(rank, str(Path(directory) / "gloo"))
                )
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=25)
                self.assertEqual([p.exitcode for p in processes], [0, 0])
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
