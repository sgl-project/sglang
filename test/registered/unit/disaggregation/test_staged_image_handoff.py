"""Contracts shared by staged image delivery, prefill and decode."""

import asyncio
import unittest
from types import SimpleNamespace

import grpc
import msgspec
import torch

from sglang.encoder.v1 import encoder_pb2 as pb
from sglang.encoder.v1 import encoder_pb2_grpc as rpc
from sglang.srt.disaggregation.encoder.handoff import (
    EncoderHandoff,
    verify_staged_embeddings,
)
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
        from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

        processor = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
        processor.model_type = "qwen2_vl"
        processor._spatial_merge_size = 2
        processor.get_staged_media_layout = lambda prompt, **kw: SimpleNamespace(
            input_ids=prompt
        )
        prompt = list(range(10))
        handoff.build_layout(processor, prompt, 21)
        with self.assertRaisesRegex(ValueError, "input length"):
            handoff.build_layout(processor, prompt, 20)
        processor._spatial_merge_size = 3
        with self.assertRaisesRegex(ValueError, "incompatible"):
            handoff.build_layout(processor, prompt, 128)

    def test_staged_content_identity_is_independent_of_cache_settings_and_routing_hints(
        self,
    ):
        from sglang.srt.disaggregation.encoder.handoff import staged_embedding_digest
        from sglang.srt.environ import envs
        from sglang.srt.managers.req_types import MultimodalDataItem
        from sglang.srt.multimodal.modality import Modality

        def item(embedding):
            return MultimodalDataItem(
                modality=Modality.IMAGE, precomputed_embeddings=embedding
            )

        data = descriptor()
        embeddings = [torch.full((1, 4), 1.0), torch.full((2, 4), 2.0)]
        with envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(False):
            expected = [item(t).cache_key for t in embeddings]
        for encoder_skip in (False, True):
            with envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(encoder_skip):
                for layout, tensor in zip(data["images"], embeddings):
                    layout["identity"] = staged_embedding_digest(
                        Modality.IMAGE, tensor
                    ).hex()
            handoff = EncoderHandoff.parse(msgspec.json.encode(data), 2)
            for receiver_skip in (False, True):
                with (
                    self.subTest(encoder=encoder_skip, receiver=receiver_skip),
                    envs.SGLANG_MM_SKIP_COMPUTE_HASH.override(receiver_skip),
                ):
                    # P/PD verifies bytes. D reconstructs the same cache identity from the receipt.
                    for tensors in (embeddings, (None, None)):
                        items = [item(t) for t in tensors]
                        identities = handoff.bind_identities(items, [42, None])
                        if tensors is embeddings:
                            verify_staged_embeddings(items, tensors, identities)
                        self.assertEqual([i.identity for i in items], expected)
                        self.assertEqual(items[0].hash, 42)
                        self.assertEqual(
                            items[1].hash, int.from_bytes(expected[1][:8], "big")
                        )
                    changed = item(embeddings[0] + 1)
                    changed.set_identity(expected[0])
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        verify_staged_embeddings(
                            [changed, item(embeddings[1])],
                            [embeddings[0] + 1, embeddings[1]],
                            expected,
                        )

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
