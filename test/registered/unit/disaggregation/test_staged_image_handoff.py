"""Contracts shared by staged image delivery, prefill and decode."""

import asyncio
import unittest

import grpc
import msgspec
from sglang.encoder.v1 import encoder_pb2 as pb
from sglang.encoder.v1 import encoder_pb2_grpc as rpc
from sglang.srt.disaggregation.encoder.handoff import EncoderHandoff
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def descriptor():
    transfer = "ba" * 16
    return {
        "version": 1,
        "request_id": "client",
        "transfer_id": transfer,
        "part_id": transfer + "_local_part_0",
        "part_idx": 0,
        "num_parts": 1,
        "image_count": 2,
        "peer": "grpc://127.0.0.1:1",
        "images": [
            {"t": 1, "h": 4, "w": 4, "identity": "01" * 32},
            {"t": 1, "h": 4, "w": 8, "identity": "02" * 32},
        ],
    }


class TestStagedImageHandoff(unittest.TestCase):
    def test_invalid_route_part_version_and_layout(self):
        for edit in [
            {"version": 2},
            {"transfer_id": "bad"},
            {"part_idx": 1},
            {"part_id": "another"},
            {"num_parts": 2},
            {"image_count": 1},
            {"request_id": ""},
            {"peer": "https://e:1"},
            {"peer": "grpc://e"},
            {"peer": "grpc://e:42/encode"},
            {"peer": "grpc://user@e:42"},
            {"images": []},
        ]:
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                EncoderHandoff.parse(msgspec.json.encode(descriptor() | edit), 2)
        with self.assertRaisesRegex(ValueError, "placeholders"):
            EncoderHandoff.parse(msgspec.json.encode(descriptor()), 1)
        handoff = EncoderHandoff.parse(msgspec.json.encode(descriptor()), 2)
        handoff.validate_layout(2, 10, 21)
        with self.assertRaisesRegex(ValueError, "input length"):
            handoff.validate_layout(2, 10, 20)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            handoff.validate_layout(3, 10, 128)

    def test_caller_hash_does_not_replace_content_identity(self):
        class Item:
            def __init__(self, embedding, key):
                self.precomputed_embeddings = embedding
                self.cache_key = key

            def set_identity(self, value):
                self.identity = value
                self.hash = int.from_bytes(value[:8], "big")

        handoff = EncoderHandoff.parse(msgspec.json.encode(descriptor()), 2)
        ids = [bytes.fromhex(image.identity) for image in handoff.images]
        # P verifies actual embeddings; D starts with metadata only.
        for embedding in (object(), None):
            items = [Item(embedding, key) for key in ids]
            handoff.bind_identities(items, [42, None])
            self.assertEqual([item.identity for item in items], ids)
            self.assertEqual(items[0].hash, 42)
            self.assertEqual(items[1].hash, int.from_bytes(ids[1][:8], "big"))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            handoff.bind_identities([Item(object(), k) for k in reversed(ids)], [])

    def test_delivery_reaches_only_selected_grpc_peer(self):
        class Encoder(rpc.SglangEncoderServicer):
            def __init__(self):
                self.deliveries, self.releases = [], []

            async def Deliver(self, req, _):
                self.deliveries.append((req.transfer, req.receive_url))
                return pb.DeliverResponse()

            async def Release(self, req, _):
                self.releases.append(req.transfer_id)
                return pb.ReleaseResponse()

        async def run():
            encoders = []
            try:
                for _ in range(2):
                    server = grpc.aio.server()
                    encoder = Encoder()
                    rpc.add_SglangEncoderServicer_to_server(encoder, server)
                    port = server.add_insecure_port("127.0.0.1:0")
                    await server.start()
                    encoders.append((encoder, server, port))
                data = descriptor() | {"peer": f"grpc://127.0.0.1:{encoders[1][2]}"}
                handoff = EncoderHandoff.parse(msgspec.json.encode(data), 2)
                await handoff.deliver("127.0.0.1:4422", 2)
                await handoff.release(2)
                selected = encoders[1][0]
                self.assertEqual(
                    selected.deliveries, [(handoff.native_transfer(), "127.0.0.1:4422")]
                )
                self.assertEqual(selected.releases, [data["transfer_id"]])
                self.assertEqual(encoders[0][0].deliveries, [])
                self.assertEqual(encoders[0][0].releases, [])
            finally:
                for _, server, _ in encoders:
                    await server.stop(0)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
