"""A native Qwen request rebuilt on CPU at P and D after a staged encode."""

import asyncio
import pickle
import unittest
from array import array
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import grpc
import msgspec
import torch
import zmq
from google.protobuf.json_format import MessageToDict
from sglang.encoder.v1 import encoder_pb2 as pb
from sglang.encoder.v1 import encoder_pb2_grpc as rpc
from sglang.srt.disaggregation.encoder import receiver
from sglang.srt.disaggregation.encoder.handoff import EncoderHandoff
from sglang.srt.disaggregation.encoder.staged import StagedImageService
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.rust_server.server import NoEncoderError, RustServer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def processor():
    q = QwenVLImageProcessor.__new__(QwenVLImageProcessor)
    q.model_type = "qwen3_vl"
    q.hf_config = SimpleNamespace()
    q.IM_TOKEN_ID, q.IM_START_TOKEN_ID, q.IM_END_TOKEN_ID = 8, 7, 9
    q.vision_start_token_id, q.vision_end_token_id = 7, 9
    q.VIDEO_TOKEN_ID = 10
    q.audio_token_id = q.audio_start_token_id = None
    q._tokens_per_second = None
    q._spatial_merge_size = 2
    q.mm_tokens = SimpleNamespace(
        image_token_id=8, video_token_id=10, audio_token_id=None
    )
    return q


def request(rid):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=None,
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=None,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
    )


def bridge(external=True):
    rust = RustServer(Mock(), http_port=0)
    rust._external_image_token_id = 8 if external else None
    rust._encoder_dispatcher = Mock()
    return rust


def scheduler(context):
    rx = receiver.MMReceiverHTTP.__new__(receiver.MMReceiverHTTP)
    rx.encoder_transfer_backend = "zmq_to_scheduler"
    rx.encode_urls = ["http://never-used"]
    rx.waiting_by_rid, rx.waiting_list = {}, []
    rx.wait_timeout = 2
    rx.scheduler = SimpleNamespace(max_req_input_len=100)
    rx.scheduler_context = context
    rx.scheduler_recv_socket = None
    rx.scheduler_embedding_port = None
    rx.mm_processor, rx.model_type = processor(), "qwen3_vl"
    rx.hostname = "127.0.0.1"
    rx.tp_size, rx.tp_group = 1, SimpleNamespace(cpu_group=None)
    rx.embedding_pool = None
    rx.registration_runner = SimpleNamespace(submit=asyncio.create_task)
    return rx


@asynccontextmanager
async def staged_encoder(sock, grids, tensor, identities):
    """Only the model forward is represented by a deterministic CPU tensor."""
    calls, held = [], set()

    async def encode(part_id, images):
        calls.append(("encode", images))
        held.add(part_id)
        return [
            pb.ImageLayout(t=t, h=h, w=w, identity=identity.hex())
            for (t, h, w), identity in zip(grids, identities)
        ]

    async def send(part_id, receive_url):
        assert part_id in held
        calls.append(("send", receive_url))
        sock.connect("tcp://" + receive_url)
        frame = receiver.EmbeddingData(
            part_id, 1, 0, grids, Modality.IMAGE, embedding=tensor
        ).copy_without_embedding()
        sock.send_multipart([pickle.dumps(frame), tensor.numpy().tobytes()])

    async def release(part_id):
        held.remove(part_id)

    class Service(StagedImageService, rpc.SglangEncoderServicer):
        pass

    service = Service(
        encode=encode,
        send=send,
        release=release,
        make_part_id=receiver.create_part_req_id,
        info=pb.GetEncoderInfoResponse(),
        ttl=10,
        stage_timeout=5,
        capacity=1,
    )
    server = grpc.aio.server()
    rpc.add_SglangEncoderServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    try:
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            response = await rpc.SglangEncoderStub(channel).Stage(
                pb.StageRequest(request_id="public", mm_items=["first", "second"]),
                timeout=3,
            )
        metadata = MessageToDict(
            response.transfer,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )
        metadata["peer"] = f"grpc://127.0.0.1:{port}"
        yield metadata, calls, held
    finally:
        await server.stop(0)


class TestStagedImageReceiver(unittest.TestCase):
    def test_external_ingress_rejects_missing_handoff_and_bypasses_text(self):
        with self.assertRaisesRegex(ValueError, "requires the private"):
            GenerateReqInput(
                input_ids=[7, 8, 9], encoder_handoff={}
            ).normalize_batch_and_arguments()
        rust = bridge()
        for buffers in [
            {"input_ids": array("q", [7, 8, 9])},
            {"input_ids": array("q", [1]), "epd.media": msgspec.msgpack.encode(["x"])},
        ]:
            with self.assertRaisesRegex(NoEncoderError, "requires a staged"):
                rust._attach_buffers(request("raw"), buffers)
        text = request("text")
        rust._attach_buffers(text, {"input_ids": array("q", [1, 2])})
        self.assertIsNone(text.need_wait_for_mm_inputs)
        rust._encoder_dispatcher.send_encode_request.assert_not_called()
        with self.assertRaisesRegex(ValueError, "requires SGLANG_EXTERNAL"):
            bridge(False)._attach_buffers(
                text, {"input_ids": array("q", [8]), "epd.handoff": b"{}"}
            )

    def test_abort_and_unavailable_encoder_retire_receiver_once(self):
        async def run(fail):
            reached, released = asyncio.Event(), asyncio.Event()

            async def deliver(handoff, receive_url, timeout):
                reached.set()
                if fail:
                    raise RuntimeError("expired transfer")
                await asyncio.Future()

            release_ids = []

            async def release(handoff, timeout):
                release_ids.append(handoff.transfer_id)
                released.set()

            transfer = "00" * 16
            handoff = EncoderHandoff.parse(
                msgspec.json.encode(
                    {
                        "version": 1,
                        "transfer_id": transfer,
                        "part_id": transfer + "_local_part_0",
                        "request_id": "public",
                        "peer": "grpc://127.0.0.1:1",
                        "num_parts": 1,
                        "part_idx": 0,
                        "image_count": 1,
                        "images": [{"t": 1, "h": 2, "w": 2, "identity": "01" * 32}],
                    }
                ),
                1,
            )
            obj = request("local")
            obj.encoder_handoff = handoff
            obj.need_wait_for_mm_inputs = True
            obj.input_ids = array("q", [7, 8, 9])
            with (
                zmq.Context() as ctx,
                (
                    patch.object(
                        receiver,
                        "get_disagg",
                        return_value=SimpleNamespace(disaggregation_mode="null"),
                    )
                ),
                patch.object(EncoderHandoff, "deliver", deliver),
                patch.object(EncoderHandoff, "release", release),
                patch.object(torch.distributed, "all_reduce"),
            ):
                rx = scheduler(ctx)
                rx.create_req = lambda req: req
                self.assertEqual(rx.process_waiting_requests([obj]), ([], []))
                waiting = rx.waiting_by_rid["local"]
                await asyncio.wait_for(reached.wait(), 2)
                if fail:
                    await asyncio.sleep(0)  # apply the registration error callback
                else:
                    rx.abort_waiting_requests(
                        SimpleNamespace(matches=lambda rid: rid == "local")
                    )
                admitted, errors = rx.process_waiting_requests([])
                self.assertFalse(admitted)
                self.assertEqual(len(errors), 1)
                self.assertIs(errors[0][0], obj)
                self.assertEqual(errors[0][2], 502 if fail else 400)
                await asyncio.wait_for(released.wait(), 2)
                self.assertEqual(release_ids, [transfer])
                self.assertIsNone(waiting.recv_socket)
                self.assertFalse(rx.waiting_list or rx.waiting_by_rid)

        for fail in (False, True):
            with self.subTest(fail=fail):
                asyncio.run(run(fail))

    def test_native_part_to_prefill_and_decode_layout_only(self):
        async def run():
            prompt = [42, 7, 8, 9, 43, 7, 8, 9, 44]
            grids = [[1, 2, 2], [1, 4, 2]]
            tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            identities = [
                MultimodalDataItem(
                    modality=Modality.IMAGE, precomputed_embeddings=e
                ).cache_key
                for e in [tensor[:1], tensor[1:]]
            ]
            rust = bridge()
            prefill, decode = request("p-incarnation"), request("d-incarnation")
            with zmq.Context() as ctx, ctx.socket(zmq.PUSH) as sock:
                sock.linger = 0
                rx = scheduler(ctx)
                async with staged_encoder(sock, grids, tensor, identities) as (
                    metadata,
                    calls,
                    held,
                ):
                    self.assertEqual(held, {metadata["part_id"]})
                    self.assertEqual(calls, [("encode", ["first", "second"])])
                    buffers = {
                        "input_ids": array("q", prompt),
                        "epd.handoff": msgspec.json.encode(metadata),
                        "epd.hashes": msgspec.msgpack.encode([22, 33]),
                    }
                    for obj in (prefill, decode):
                        rust._attach_buffers(obj, buffers)
                    with (
                        patch.object(
                            receiver,
                            "get_disagg",
                            return_value=SimpleNamespace(disaggregation_mode="prefill"),
                        ),
                        patch.object(torch.distributed, "all_reduce"),
                    ):
                        self.assertEqual(
                            rx.process_waiting_requests([prefill]), ([], [])
                        )
                        waiting = rx.waiting_by_rid[prefill.rid]
                        self.assertIsNotNone(waiting.recv_socket)
                        await asyncio.wait_for(waiting.registration_future, 3)
                        self.assertEqual(
                            calls,
                            [
                                ("encode", ["first", "second"]),
                                ("send", f"127.0.0.1:{waiting.embedding_port}"),
                            ],
                        )
                        self.assertFalse(held)
                        self.assertTrue(waiting.recv_socket.poll(2000))
                        self.assertEqual(
                            rx.process_waiting_requests([]), ([prefill], [])
                        )
                        self.assertIsNone(waiting.recv_socket)
                        self.assertFalse(rx.waiting_by_rid)
                        self.assertFalse(rx.waiting_list)
                # The staged lease and entire E process can be gone before D starts.
                with patch.object(
                    receiver,
                    "get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode="decode"),
                ):
                    self.assertEqual(
                        rx.process_waiting_requests([decode]), ([decode], [])
                    )

            self.assertEqual(list(prefill.input_ids), [42, 7, 8, 9, 43, 7, 8, 8, 9, 44])
            self.assertEqual(prefill.input_ids, decode.input_ids)
            p, d = prefill.mm_inputs, decode.mm_inputs
            torch.testing.assert_close(p.mrope_positions, d.mrope_positions)
            torch.testing.assert_close(p.mrope_position_delta, d.mrope_position_delta)
            for mm in (p, d):
                self.assertEqual(
                    [item.offsets for item in mm.mm_items], [[(2, 2)], [(6, 7)]]
                )
                self.assertEqual([item.identity for item in mm.mm_items], identities)
                self.assertEqual([item.hash for item in mm.mm_items], [22, 33])
            self.assertEqual(
                [item.pad_values for item in p.mm_items],
                [item.pad_values for item in d.mm_items],
            )
            torch.testing.assert_close(
                torch.cat([item.precomputed_embeddings for item in p.mm_items]), tensor
            )
            self.assertTrue(
                all(item.precomputed_embeddings is None for item in d.mm_items)
            )
            rust._encoder_dispatcher.send_encode_request.assert_not_called()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
