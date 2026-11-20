"""Exercise the producer's real gRPC service with an in-memory encode backend."""

import asyncio
import hashlib
import unittest
from unittest.mock import AsyncMock, patch

import grpc
import zmq
import zmq.asyncio

from sglang.encoder.v1 import encoder_pb2 as pb
from sglang.encoder.v1 import encoder_pb2_grpc as rpc
from sglang.srt.disaggregation.encoder.staged import StagedImageService
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class FakeEncoder(StagedImageService, rpc.SglangEncoderServicer):
    def __init__(self, sockets):
        self.encoded = []
        self.parts = {}
        self.released = []
        self.sockets = sockets
        self.sending = asyncio.Event()
        self.pause_send = asyncio.Event()
        self.encoding = asyncio.Event()
        self.release_done = asyncio.Event()
        super().__init__(
            encode=self.encode_images,
            send=self.send_images,
            release=self.release_images,
            make_part_id=lambda rid, idx: f"{rid}_local_part_{idx}",
            info=pb.GetEncoderInfoResponse(model_path="fake"),
            stage_timeout=3,
            ttl=5,
            capacity=2,
        )

    async def encode_images(self, rid, images, videos):
        self.encoded.append((rid, images))
        self.parts[rid] = [image.encode() for image in images]
        self.encoding.set()
        if images == ["wait"]:
            await asyncio.Event().wait()
        if images == ["fail"]:
            raise RuntimeError("encoder forward failed")
        return [
            pb.ImageLayout(t=1, h=2, w=2, identity=hashlib.sha256(data).hexdigest())
            for data in self.parts[rid]
        ], []

    async def release_images(self, rid):
        self.parts.pop(rid, None)
        self.released.append(rid)
        self.release_done.set()

    async def send_images(self, rid, url):
        if url == "fail":
            raise OSError("receiver unavailable")
        if url == "wait":
            self.sending.set()
            await self.pause_send.wait()
            return
        with self.sockets.socket(zmq.PUSH) as socket:
            socket.connect("tcp://" + url)
            await socket.send_multipart([rid.encode(), *self.parts[rid]])


class TestStagedImages(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.sockets = zmq.asyncio.Context()
        self.addCleanup(self.sockets.destroy, 0)
        self.backends = []
        self.peers = []
        for _ in range(2):
            backend = FakeEncoder(self.sockets)
            self.backends.append(backend)
            server = grpc.aio.server()
            rpc.add_SglangEncoderServicer_to_server(backend, server)
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            self.addAsyncCleanup(server.stop, 0)
            channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
            self.addAsyncCleanup(channel.close)
            self.peers.append(rpc.SglangEncoderStub(channel))

    async def stage(self, replica, images=("a", "b"), request_id="chat-id"):
        reply = await self.peers[replica].Stage(
            pb.StageRequest(request_id=request_id, mm_items=images), timeout=2
        )
        return reply.transfer

    async def error(self, code, awaitable):
        with self.assertRaises(grpc.aio.AioRpcError) as raised:
            await awaitable
        self.assertEqual(raised.exception.code(), code)

    async def test_delayed_receivers_choose_exact_replica_and_part(self):
        transfers = [await self.stage(i) for i in range(2)]
        self.assertNotEqual(transfers[0].transfer_id, transfers[1].transfer_id)
        for i, transfer in enumerate(transfers):
            self.assertEqual(transfer.part_id, f"{transfer.transfer_id}_local_part_0")
            self.assertEqual(
                (transfer.num_parts, transfer.part_idx, transfer.image_count), (1, 0, 2)
            )
            self.assertEqual(transfer.request_id, "chat-id")
            self.assertEqual(
                [layout.identity for layout in transfer.images],
                [hashlib.sha256(b).hexdigest() for b in (b"a", b"b")],
            )
            with self.sockets.socket(zmq.PULL) as pull:
                port = pull.bind_to_random_port("tcp://127.0.0.1")
                request = pb.DeliverRequest(
                    transfer=transfer, receive_url=f"127.0.0.1:{port}"
                )
                await self.error(
                    grpc.StatusCode.NOT_FOUND, self.peers[1 - i].Deliver(request)
                )
                await self.peers[i].Deliver(request, timeout=2)
                self.assertEqual(
                    await asyncio.wait_for(pull.recv_multipart(), 2),
                    [transfer.part_id.encode(), b"a", b"b"],
                )
                await self.error(
                    grpc.StatusCode.NOT_FOUND, self.peers[i].Deliver(request)
                )
            self.assertEqual(self.backends[i].encoded, [(transfer.part_id, ["a", "b"])])
            self.assertEqual(self.backends[i].released, [transfer.part_id])
            self.assertEqual(self.backends[i].parts, {})

    async def test_nixl_description_and_one_shot_destination(self):
        backend, peer = self.backends[0], self.peers[0]
        transfer = await self.stage(0)
        describe = pb.DescribeRequest(transfer=transfer)
        deliver = pb.DeliverRequest(transfer=transfer, nixl_target=b"registered buffer")
        await self.error(grpc.StatusCode.FAILED_PRECONDITION, peer.Describe(describe))
        await self.error(grpc.StatusCode.INVALID_ARGUMENT, peer.Deliver(deliver))
        backend._send = None
        backend._write = AsyncMock()
        descriptor = pb.DescribeResponse(shape=[2, 2048], dtype="bfloat16")
        backend._describe = lambda part_id: descriptor
        self.assertEqual(descriptor, await peer.Describe(describe))
        invalid = pb.DescribeRequest(transfer=transfer)
        invalid.transfer.images[0].identity = "another image"
        await self.error(grpc.StatusCode.INVALID_ARGUMENT, peer.Describe(invalid))
        deliver.receive_url = "127.0.0.1:12345"
        await self.error(grpc.StatusCode.INVALID_ARGUMENT, peer.Deliver(deliver))
        deliver.ClearField("receive_url")
        await peer.Deliver(deliver)
        backend._write.assert_awaited_once_with(transfer.part_id, b"registered buffer")
        await self.error(grpc.StatusCode.NOT_FOUND, peer.Deliver(deliver))
        await self.error(grpc.StatusCode.NOT_FOUND, peer.Describe(describe))

    async def test_release_fences_cancelled_nixl_write(self):
        backend, peer = self.backends[0], self.peers[0]
        backend._capacity = 1
        posted, draining, complete = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def write(rid, descriptor):
            posted.set()
            try:
                await asyncio.Event().wait()
            finally:
                draining.set()
                await complete.wait()

        backend._send, backend._write = None, write
        transfer = await self.stage(0)
        deliver = peer.Deliver(pb.DeliverRequest(transfer=transfer, nixl_target=b"dst"))
        await asyncio.wait_for(posted.wait(), 2)
        deliver.cancel()
        await asyncio.wait_for(draining.wait(), 2)
        release = peer.Release(pb.ReleaseRequest(transfer_id=transfer.transfer_id))
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(release), 0.05)
        await self.error(grpc.StatusCode.RESOURCE_EXHAUSTED, self.stage(0))
        complete.set()
        await asyncio.wait_for(release, 2)
        await asyncio.wait_for(backend.release_done.wait(), 2)
        self.assertFalse(backend._writing)
        self.assertEqual(backend.released, [transfer.part_id])

    async def test_repeated_cancel_cannot_detach_an_active_write(self):
        backend = self.backends[0]
        transfer = await self.stage(0)
        posted, draining, second_cancel, done = (asyncio.Event() for _ in range(4))

        async def write(rid, descriptor):
            posted.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                draining.set()
                try:
                    await done.wait()
                except asyncio.CancelledError:
                    second_cancel.set()
                    await done.wait()
                raise

        backend._write = write
        delivery = asyncio.create_task(
            backend.Deliver(
                pb.DeliverRequest(transfer=transfer, nixl_target=b"dst"), None
            )
        )
        await asyncio.wait_for(posted.wait(), 2)
        delivery.cancel()
        await asyncio.wait_for(draining.wait(), 2)
        delivery.cancel()
        await asyncio.wait_for(second_cancel.wait(), 2)
        release = asyncio.create_task(
            backend.Release(pb.ReleaseRequest(transfer_id=transfer.transfer_id), None)
        )
        await asyncio.sleep(0)
        self.assertFalse(release.done())
        self.assertIn(transfer.part_id, backend.parts)
        done.set()
        with self.assertRaises(asyncio.CancelledError):
            await delivery
        await release
        self.assertEqual(backend.released, [transfer.part_id])

    async def test_expiry_and_capacity(self):
        backend = self.backends[0]
        backend._capacity = 1
        first = await self.stage(0)
        await self.error(grpc.StatusCode.RESOURCE_EXHAUSTED, self.stage(0))
        for _ in range(2):
            await self.peers[0].Release(
                pb.ReleaseRequest(transfer_id=first.transfer_id)
            )
        backend.release_done.clear()
        backend._ttl = 0.02
        transfer = await self.stage(0)
        await asyncio.wait_for(backend.release_done.wait(), 2)
        self.assertEqual(backend.parts, {})
        self.assertEqual(backend.released, [first.part_id, transfer.part_id])
        await self.error(
            grpc.StatusCode.NOT_FOUND,
            self.peers[0].Deliver(
                pb.DeliverRequest(transfer=transfer, receive_url="late")
            ),
        )

    async def test_video_requires_capability_and_nonempty_inputs(self):
        peer = self.peers[0]
        for videos, status in (
            (["video.mp4"], grpc.StatusCode.FAILED_PRECONDITION),
            ([""], grpc.StatusCode.INVALID_ARGUMENT),
        ):
            request = pb.StageRequest(request_id="bad-video", video_urls=videos)
            await self.error(status, peer.Stage(request))
        self.assertEqual(self.backends[0].encoded, [])

    async def test_failure_abort_and_invalid_handle(self):
        backend = self.backends[0]
        req = pb.StageRequest(request_id="broken", mm_items=["wait"])
        call = self.peers[0].Stage(req, timeout=5)
        await asyncio.wait_for(backend.encoding.wait(), 2)
        call.cancel()
        await asyncio.wait_for(backend.release_done.wait(), 2)
        self.assertEqual(backend.parts, {})
        self.assertEqual(len(backend.released), 1)
        await self.error(grpc.StatusCode.INTERNAL, self.stage(0, ["fail"]))
        transfer = await self.stage(0)
        corrupt = pb.DeliverRequest(transfer=transfer, receive_url="wait")
        corrupt.transfer.images[0].h += 1
        await self.error(
            grpc.StatusCode.INVALID_ARGUMENT, self.peers[0].Deliver(corrupt)
        )
        self.assertIn(transfer.part_id, backend.parts)
        req = pb.DeliverRequest(transfer=transfer, receive_url="fail")
        await self.error(grpc.StatusCode.UNAVAILABLE, self.peers[0].Deliver(req))
        self.assertEqual(backend.parts, {})
        self.assertEqual(len(backend.released), 3)

    async def test_one_shot_claim_during_send(self):
        backend = self.backends[0]
        transfer = await self.stage(0)
        req = pb.DeliverRequest(transfer=transfer, receive_url="wait")
        send = self.peers[0].Deliver(req, timeout=2)
        await asyncio.wait_for(backend.sending.wait(), 2)
        await self.error(grpc.StatusCode.NOT_FOUND, self.peers[0].Deliver(req))
        await self.peers[0].Release(pb.ReleaseRequest(transfer_id=transfer.transfer_id))
        self.assertIn(transfer.part_id, backend.parts)
        backend.pause_send.set()
        await send
        self.assertEqual(backend.parts, {})
        self.assertEqual(backend.released, [transfer.part_id])

    async def test_aborted_release_finishes_native_cleanup(self):
        backend = self.backends[0]
        transfer = await self.stage(0)
        started, resume = asyncio.Event(), asyncio.Event()

        async def delayed_release(rid):
            started.set()
            await resume.wait()
            await backend.release_images(rid)

        backend._release = delayed_release
        release = self.peers[0].Release(
            pb.ReleaseRequest(transfer_id=transfer.transfer_id)
        )
        await asyncio.wait_for(started.wait(), 2)
        release.cancel()
        resume.set()
        await asyncio.wait_for(backend.release_done.wait(), 2)
        self.assertEqual(backend.released, [transfer.part_id])
        self.assertEqual(backend._active, 0)

    async def test_claimed_send_has_deadline_without_client_timeout(self):
        backend = self.backends[0]
        transfer = await self.stage(0)
        backend._ttl = 0.02
        req = pb.DeliverRequest(transfer=transfer, receive_url="wait")
        await asyncio.wait_for(
            self.error(grpc.StatusCode.UNAVAILABLE, self.peers[0].Deliver(req)),
            timeout=2,
        )
        self.assertTrue(backend.sending.is_set())
        self.assertEqual(backend.parts, {})
        self.assertEqual(backend.released, [transfer.part_id])

    async def test_expired_claim_before_reaper_callback(self):
        backend = self.backends[0]
        transfer = await self.stage(0)
        deadline = backend._leases[transfer.transfer_id].expiry.when()
        ctx = AsyncMock()
        ctx.abort.side_effect = RuntimeError("expired")
        loop = asyncio.get_running_loop()
        with patch.object(loop, "time", return_value=deadline):
            with self.assertRaisesRegex(RuntimeError, "expired"):
                await backend.Deliver(
                    pb.DeliverRequest(transfer=transfer, receive_url="fail"), ctx
                )
        ctx.abort.assert_awaited_once_with(
            grpc.StatusCode.NOT_FOUND, "transfer expired or consumed"
        )
        await asyncio.wait_for(backend.release_done.wait(), 2)
        self.assertEqual(backend.parts, {})


if __name__ == "__main__":
    unittest.main()
