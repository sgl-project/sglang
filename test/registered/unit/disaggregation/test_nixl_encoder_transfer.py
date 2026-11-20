"""Validate staging lifetime and cross-process UCX writes for encoder embeddings."""

import asyncio
import importlib.util
import multiprocessing
import threading
import time
import unittest
from unittest.mock import patch

import msgspec
import torch
from sglang.srt.disaggregation.encoder import nixl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _source():
    return torch.arange(65536, dtype=torch.int32).to(torch.bfloat16).reshape(32, 2048)


def _send(descriptor):
    asyncio.run(nixl.write_embedding(_source(), descriptor, 15))


class TestNixlEncoderTransfer(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_transfer_never_registers_source(self):
        source = _source()
        with patch.object(
            nixl, "_agent", side_effect=AssertionError("unexpected agent")
        ):
            for sample, address, size in [
                (source, 8, 8),
                (source, 0, source.nbytes),
                (source.T, 8, source.nbytes),
            ]:
                descriptor = msgspec.msgpack.encode(
                    nixl._Destination(address, size, b"1" * 16, b"test")
                )
                with (
                    self.subTest(address=address, size=size),
                    self.assertRaises(ValueError),
                ):
                    await nixl.write_embedding(sample, descriptor, 1)

    async def test_cancel_waits_for_writer_fence(self):
        started, allowed_to_finish = threading.Event(), threading.Event()
        source = torch.ones(4, dtype=torch.uint8)
        descriptor = msgspec.msgpack.encode(nixl._Destination(8, 4, b"1" * 16, b"test"))

        def in_flight(source, destination, stop, deadline):
            started.set()
            if not (stop.wait(5) and allowed_to_finish.wait(5)):
                raise TimeoutError("test failed to advance simulated write")

        with patch.object(nixl, "_write", in_flight):
            writer = asyncio.create_task(nixl.write_embedding(source, descriptor, 3))
            self.assertTrue(await asyncio.to_thread(started.wait, 5))
            writer.cancel()
            await asyncio.sleep(0)
            writer.cancel()
            await asyncio.sleep(0)
            self.assertFalse(writer.done())
            allowed_to_finish.set()
            with self.assertRaises(asyncio.CancelledError):
                await writer

    @unittest.skipUnless(importlib.util.find_spec("nixl"), "requires NIXL with UCX")
    async def test_reuses_writer_across_distinct_targets_on_one_receiver(self):
        receiver = nixl.NixlReceiver(capacity=4)
        samples = [torch.full((8192,), i, dtype=torch.int32) for i in range(1, 5)]
        targets = [receiver.allocate(sample.nbytes) for sample in samples]

        def write(i):
            return nixl.write_embedding(samples[i], targets[i].descriptor, 15)

        with patch.object(nixl, "_agent", wraps=nixl._agent) as factory:
            await write(0)
            await write(1)
            self.assertEqual(factory.call_count, 1)
            await asyncio.gather(write(2), write(3))
        for sample, target in zip(samples, targets):
            self.assertTrue(torch.equal(target.tensor.view(sample.dtype), sample))
            target.close()

    @unittest.skipUnless(importlib.util.find_spec("nixl"), "requires NIXL with UCX")
    async def test_real_cross_process_write_reclaims_abandoned_slot(self):
        receiver = nixl.NixlReceiver(capacity=1)
        source = _source()
        target = receiver.allocate(source.nbytes)
        target.tensor.zero_()
        target.abandon()
        with self.assertRaisesRegex(RuntimeError, "no free transfer slots"):
            receiver.allocate(1)

        # Only the descriptor crosses this channel. A spawned interpreter has
        # its own address space and creates the full BF16 tensor independently.
        child = multiprocessing.get_context("spawn").Process(
            target=_send, args=(target.descriptor,)
        )
        child.start()
        try:
            await asyncio.to_thread(child.join, 40)
            self.assertEqual(child.exitcode, 0)
        finally:
            if child.is_alive():
                child.kill()
                child.join()

        deadline = time.monotonic() + 3
        while True:
            try:
                replacement = receiver.allocate(1)
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.01)
            else:
                replacement.close()
                break
        self.assertTrue(
            torch.equal(target.tensor.view(torch.bfloat16).reshape_as(source), source)
        )
        target.close()


if __name__ == "__main__":
    unittest.main()
