"""Validate staging lifetime and cross-process UCX writes for encoder embeddings."""

import asyncio
import importlib.util
import multiprocessing
import threading
import time
import unittest
from unittest.mock import Mock, call, patch

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
    def test_registered_region_is_reused_after_the_last_tensor_alias(self):
        agent = Mock()
        agent.get_partial_agent_metadata.return_value = b"pool"
        agent.get_new_notifs.return_value = {}
        with patch.object(nixl, "_agent", return_value=agent):
            receiver = nixl.NixlReceiver(capacity=2, pool_bytes=512)

        first = receiver.allocate(256)
        first.tensor.fill_(7)
        first_addr = first.tensor.data_ptr()
        alias = first.tensor.view(16, 16).detach()[3]
        first.close()
        second = receiver.allocate(256)
        self.assertNotEqual(first_addr, second.tensor.data_ptr())
        second.tensor.fill_(9)
        self.assertEqual(alias.tolist(), [7] * 16)
        overflow = receiver.allocate(128)
        self.assertNotEqual(overflow.tensor.data_ptr(), first_addr)
        overflow.tensor.fill_(10)
        self.assertEqual(alias.tolist(), [7] * 16)
        with self.assertRaisesRegex(RuntimeError, "no free transfer slots"):
            receiver.allocate(1)
        overflow.close()

        del alias
        reused = receiver.allocate(256)
        self.assertEqual(reused.tensor.data_ptr(), first_addr)
        reused.tensor.fill_(11)
        self.assertTrue(torch.all(second.tensor == 9).item())
        second.close()
        reused.close()
        self.assertEqual(agent.register_memory.call_count, 2)
        self.assertEqual(agent.get_partial_agent_metadata.call_count, 2)
        agent.deregister_memory.assert_called_once()

    def test_unavailable_host_pool_falls_back_to_per_transfer_registration(self):
        for fail_metadata in (False, True):
            with self.subTest(fail_metadata=fail_metadata):
                agent = Mock()
                agent.get_new_notifs.return_value = {}
                if fail_metadata:
                    agent.register_memory.side_effect = ["pool", "temporary"]
                    agent.get_partial_agent_metadata.side_effect = [
                        RuntimeError("pool metadata unavailable"),
                        b"temporary",
                    ]
                else:
                    agent.register_memory.side_effect = [MemoryError(), "temporary"]
                    agent.get_partial_agent_metadata.return_value = b"temporary"
                with (
                    patch.object(nixl, "_agent", return_value=agent),
                    self.assertLogs(nixl.logger, level="WARNING") as logs,
                ):
                    receiver = nixl.NixlReceiver(capacity=1, pool_bytes=512)
                self.assertIn("per-transfer registration", logs.output[0])
                receiver.allocate(256).close()
                retired = [call("pool")] if fail_metadata else []
                self.assertEqual(
                    agent.deregister_memory.call_args_list,
                    retired + [call("temporary")],
                )

    def test_disabled_pool_and_oversized_writes_use_one_shot_registrations(self):
        for pool_bytes in (0, 256):
            with self.subTest(pool_bytes=pool_bytes):
                agent = Mock()
                agent.get_partial_agent_metadata.return_value = b"region"
                agent.get_new_notifs.return_value = {}
                with patch.object(nixl, "_agent", return_value=agent):
                    receiver = nixl.NixlReceiver(capacity=1, pool_bytes=pool_bytes)
                for _ in range(2):
                    target = receiver.allocate(512)
                    self.assertEqual(target.tensor.numel(), 512)
                    target.close()
                self.assertEqual(agent.register_memory.call_count, 2 + bool(pool_bytes))
                self.assertEqual(agent.deregister_memory.call_count, 2)

    def test_late_notification_cannot_release_a_reused_address(self):
        agent = Mock()
        agent.get_partial_agent_metadata.return_value = b"pool"
        agent.get_new_notifs.return_value = {}
        with patch.object(nixl, "_agent", return_value=agent):
            receiver = nixl.NixlReceiver(capacity=1, pool_bytes=256)

        first = receiver.allocate(128)
        old = msgspec.msgpack.decode(first.descriptor, type=nixl._Destination)
        first.close()
        current = receiver.allocate(128)
        new = msgspec.msgpack.decode(current.descriptor, type=nixl._Destination)
        self.assertEqual(new.address, old.address)
        self.assertEqual(new.agent_metadata, old.agent_metadata)
        self.assertNotEqual(new.notification, old.notification)

        current.abandon()
        agent.get_new_notifs.return_value = {"encoder": [old.notification]}
        self.assertFalse(current.reaped())
        with self.assertRaisesRegex(RuntimeError, "no free transfer slots"):
            receiver.allocate(1)
        agent.get_new_notifs.return_value = {"encoder": [new.notification]}
        self.assertTrue(current.reaped())
        receiver.allocate(256).close()

    def test_quarantine_preserves_memory_and_bounds_replacement_slots(self):
        agent = Mock()
        agent.get_partial_agent_metadata.return_value = b"pool"
        agent.get_new_notifs.return_value = {}
        with patch.object(nixl, "_agent", return_value=agent):
            receiver = nixl.NixlReceiver(capacity=1, pool_bytes=512)

        first = receiver.allocate(256)
        first_desc = msgspec.msgpack.decode(first.descriptor, type=nixl._Destination)
        first.abandon()
        first.quarantine()
        second = receiver.allocate(256)
        self.assertNotEqual(first_desc.address, second.tensor.data_ptr())
        second.abandon()
        second.quarantine()
        with self.assertRaisesRegex(RuntimeError, "no free transfer slots"):
            receiver.allocate(1)
        agent.get_new_notifs.return_value = {"encoder": [first_desc.notification]}
        self.assertTrue(first.reaped())
        replacement = receiver.allocate(256)
        self.assertEqual(replacement.tensor.data_ptr(), first_desc.address)
        replacement.close()
        second.close()  # A positive fence from its original issuer.

    def test_writer_reuses_peers_and_bounds_retired_receiver_instances(self):
        agent = Mock()
        agent.add_remote_agent.side_effect = bytes.decode
        agent.transfer.return_value = "DONE"
        sample = torch.zeros(8, dtype=torch.uint8)
        peers = (b"a", b"a", b"b", b"a", b"c", b"b")
        with (
            patch.object(nixl, "_writer", threading.local()),
            patch.object(nixl, "_agent", return_value=agent),
            patch.object(nixl, "_MAX_CACHED_PEERS", 2),
        ):
            for index, peer in enumerate(peers):
                destination = nixl._Destination(8 + 8 * index, 8, b"1" * 16, peer)
                nixl._write(
                    sample, destination, threading.Event(), time.monotonic() + 2
                )
        self.assertEqual(
            agent.add_remote_agent.call_args_list,
            [call(b"a"), call(b"b"), call(b"c"), call(b"b")],
        )
        self.assertEqual(
            agent.remove_remote_agent.call_args_list, [call("b"), call("a")]
        )
        self.assertEqual(agent.deregister_memory.call_count, len(peers))

    def test_writer_refreshes_changed_registration_for_same_peer(self):
        agent = Mock()
        agent.add_remote_agent.return_value = "same-prefill"
        with patch.object(nixl, "_agent", return_value=agent):
            writer = nixl._NixlWriter()
            for metadata in (b"first", b"second", b"first"):
                self.assertEqual(writer.peer_for(metadata), "same-prefill")
        self.assertGreaterEqual(
            agent.add_remote_agent.call_args_list.count(call(b"first")), 2
        )
        agent.remove_remote_agent.assert_any_call("same-prefill")

    def test_writer_recovers_when_nixl_rejects_overlapping_peer_metadata(self):
        old_agent, fresh_agent = Mock(), Mock()
        old_agent.add_remote_agent.side_effect = ["prefill", RuntimeError("overlap")]
        fresh_agent.add_remote_agent.return_value = "prefill"
        with patch.object(nixl, "_agent", side_effect=[old_agent, fresh_agent]):
            writer = nixl._NixlWriter()
            self.assertEqual(writer.peer_for(b"old registration"), "prefill")
            self.assertEqual(writer.peer_for(b"new registration"), "prefill")
            self.assertEqual(writer.peer_for(b"new registration"), "prefill")
        fresh_agent.add_remote_agent.assert_called_once_with(b"new registration")

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
        samples = [torch.full((8192,), i, dtype=torch.int32) for i in range(1, 5)]
        receiver = nixl.NixlReceiver(
            capacity=4, pool_bytes=sum(sample.nbytes for sample in samples)
        )
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
    async def test_real_writes_alternate_pooled_and_one_shot_destinations(self):
        receiver = nixl.NixlReceiver(capacity=1, pool_bytes=1024)
        for value, size in enumerate((512, 2048, 512), start=1):
            sample = torch.full((size,), value, dtype=torch.uint8)
            target = receiver.allocate(size)
            await nixl.write_embedding(sample, target.descriptor, 15)
            torch.testing.assert_close(target.tensor, sample)
            target.close()

    @unittest.skipUnless(importlib.util.find_spec("nixl"), "requires NIXL with UCX")
    async def test_real_cross_process_write_reclaims_abandoned_slot(self):
        source = _source()
        receiver = nixl.NixlReceiver(capacity=1, pool_bytes=source.nbytes)
        target = receiver.allocate(source.nbytes)
        received = target.tensor.view(torch.bfloat16).reshape_as(source)
        received.zero_()
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

        self.assertTrue(torch.equal(received, source))
        del received
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
        target.close()


if __name__ == "__main__":
    unittest.main()
