"""A native Qwen request rebuilt on CPU at P and D after a staged encode."""

import asyncio
import importlib.util
import pickle
import unittest
from array import array
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
    q.hf_config = SimpleNamespace(vision_config=SimpleNamespace(out_hidden_size=2))
    q._processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text, **_: [51, 52])
    )
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
    if external:
        rust.bind_external_encoder_processor(processor())
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
async def staged_encoder(
    sock, grids, tensor, identities, writer=None, video_layouts=()
):
    """Only the model forward is represented by a deterministic CPU tensor."""
    calls, held = [], set()

    async def encode(part_id, images, videos):
        calls.append(("encode", images, videos))
        held.add(part_id)
        return [
            pb.ImageLayout(t=t, h=h, w=w, identity=identity.hex())
            for (t, h, w), identity in zip(grids, identities)
        ], video_layouts

    async def send(part_id, receive_url):
        assert part_id in held
        calls.append(("send", receive_url))
        sock.connect("tcp://" + receive_url)
        frame = receiver.EmbeddingData(
            part_id,
            1,
            0,
            grids if grids else [[v.t, v.h, v.w] for v in video_layouts],
            Modality.IMAGE if grids else Modality.VIDEO,
            embedding=tensor,
        ).copy_without_embedding()
        sock.send_multipart([pickle.dumps(frame), tensor.numpy().tobytes()])

    def describe(part_id):
        assert part_id in held
        return pb.DescribeResponse(shape=tensor.shape, dtype="float32")

    async def write(part_id, destination):
        assert part_id in held
        calls.append(("write", part_id))
        await writer(tensor, destination, 3)

    async def release(part_id):
        held.remove(part_id)

    class Service(StagedImageService, rpc.SglangEncoderServicer):
        pass

    service = Service(
        encode=encode,
        send=send if writer is None else None,
        describe=describe if writer is not None else None,
        write=write if writer is not None else None,
        release=release,
        make_part_id=receiver.create_part_req_id,
        info=pb.GetEncoderInfoResponse(supports_staged_video=True),
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
                pb.StageRequest(
                    request_id="public",
                    mm_items=["first", "second"] if grids else [],
                    video_urls=["clip"] if video_layouts else [],
                ),
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
            {"input_ids": array("q", [7, 10, 9])},
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

    def test_external_ingress_uses_each_processors_raw_markers(self):
        rust = bridge()
        glm = SimpleNamespace(
            IM_TOKEN_ID=10,
            VIDEO_TOKEN_ID=11,
            mm_tokens=SimpleNamespace(image_token_id=10, video_token_id=10),
        )
        rust.bind_external_encoder_processor(glm)
        self.assertEqual(rust._external_media_markers.counts([10, 11, 11]), (1, 2))
        glm.accepts_staged_expanded_spans = True
        rust.bind_external_encoder_processor(glm)
        self.assertEqual(rust._external_media_markers.counts([10, 10]), (None, None))
        kimi = SimpleNamespace(mm_tokens=SimpleNamespace(image_token_id=163605))
        rust.bind_external_encoder_processor(kimi)
        self.assertEqual(rust._external_media_markers.counts([163605]), (1, 0))
        muse = SimpleNamespace(
            patch_id=30,
            video_id=31,
            image_placeholder_ids=[7, 8],
            video_placeholder_ids=[7, 9],
        )
        rust.bind_external_encoder_processor(muse)
        for prompt in ([7, 8], [42, 7, 9], [30], [31]):
            with self.assertRaises(NoEncoderError):
                rust._attach_buffers(request("raw"), {"input_ids": array("q", prompt)})
        rust._attach_buffers(request("text"), {"input_ids": array("q", [7, 42, 9])})

    def test_video_only_limits_timing_and_cache_identity(self):
        rid = "09" * 16
        meta = {
            "version": 2,
            "peer": "grpc://127.0.0.1:1234",
            "request_id": "v",
            "transfer_id": rid,
            "part_id": rid + "_local_part_0",
            "part_idx": 0,
            "num_parts": 1,
            "image_count": 0,
            "images": [],
        }
        video = {
            "t": 2,
            "h": 2,
            "w": 2,
            "identities": ["ab" * 32],
            "second_per_grid_ts": 0.5,
        }

        def descriptor(**change):
            return msgspec.json.encode(meta | {"videos": [video | change]})

        q = processor()
        q.model_type = "qwen2_5_vl"
        q._tokens_per_second = 4
        prompt = [42, 7, 10, 9, 44]
        with self.assertRaisesRegex(ValueError, "placeholders"):
            EncoderHandoff.parse(descriptor(), 0, 2)
        with self.assertRaisesRegex(ValueError, "sampling interval"):
            EncoderHandoff.parse(descriptor(second_per_grid_ts=0), 0, 1)
        rust = bridge()
        rust._external_media_limits = {"video": 0}
        with patch(
            "sglang.srt.rust_server.server.msgpack_decode_explained",
            side_effect=lambda _: request("v"),
        ):
            for bad, error in ((descriptor(), ValueError), (b"{", msgspec.DecodeError)):
                with self.assertRaises(error):
                    rust._admit(
                        b"", {"input_ids": array("q", prompt), "epd.handoff": bad}
                    )
                wire, _ = rust.server.push_decode_result_batch.call_args.args
                finish = msgspec.msgpack.decode(wire)[1][0]
                self.assertEqual(finish["status_code"], 400)
                self.assertEqual(finish["type"], "abort")
            rust.server.push_error.assert_not_called()

        layouts = []
        for interval in (0.5, 1.5):
            handoff = EncoderHandoff.parse(
                descriptor(second_per_grid_ts=interval), 0, 1
            )
            mm = handoff.build_layout(q, prompt, 100)
            handoff.bind_identities(mm.mm_items, [])
            layouts.append(mm)
        self.assertNotEqual(
            layouts[0].mm_items[0].identity, layouts[1].mm_items[0].identity
        )
        self.assertFalse(
            torch.equal(layouts[0].mrope_positions, layouts[1].mrope_positions)
        )

        q.model_type = "qwen3_omni_moe"
        q.audio_start_token_id, q.hf_config.audio_token_id = 11, 12
        q.hf_config.position_id_per_seconds = 4
        omni_clip = handoff.build_layout(q, prompt, 100)
        self.assertEqual([i.modality for i in omni_clip.mm_items], [Modality.VIDEO])
        q.model_type = "qwen3_vl"
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            handoff.build_layout(q, prompt, 100)
        timed_video = descriptor(
            timestamps=[0.0, 1.0],
            identities=["ab" * 32, "cd" * 32],
            second_per_grid_ts=None,
        )
        handoff = EncoderHandoff.parse(timed_video, 0, 1)
        for model_type in ("qwen3_5", "qwen3_5_moe", "intern_s2_preview"):
            with self.subTest(model_type=model_type):
                q.model_type = model_type
                mm = handoff.build_layout(q, prompt, 100)
                self.assertEqual(
                    [i.modality for i in mm.mm_items], [Modality.VIDEO] * 2
                )
        q.model_type = "unsupported_vl"
        with self.assertRaisesRegex(ValueError, "unsupported for unsupported_vl"):
            handoff.build_layout(q, prompt, 100)
        q.model_type = "qwen3_vl"
        # The two embedding rows fit; the per-frame timestamp and marker tokens do not.
        with self.assertRaisesRegex(ValueError, "input length"):
            handoff.build_layout(q, prompt, 8)
        self.assertEqual(handoff.native_transfer().videos[0].timestamps[:], [0, 1])
        leading_video = handoff.build_layout(q, [10, 44], 100)
        self.assertEqual(
            leading_video.input_ids, [51, 52, 7, 10, 9, 51, 52, 7, 10, 9, 44]
        )
        self.assertEqual(
            [i.offsets for i in leading_video.mm_items], [[(3, 3)], [(8, 8)]]
        )

        obj = request("v")
        bridge()._attach_buffers(
            obj, {"input_ids": array("q", prompt), "epd.handoff": timed_video}
        )
        rx = scheduler(None)
        rx.scheduler.max_req_input_len = 8
        rx.registration_runner = None  # Synchronous admission-only fixture.
        rx.create_req = lambda req: req
        admitted, errors = rx.process_waiting_requests([obj])
        self.assertFalse(admitted)
        self.assertEqual(len(errors), 1)
        self.assertIs(errors[0][0], obj)
        self.assertIn("input length", errors[0][1])
        self.assertEqual(errors[0][2], 400)
        self.assertFalse(rx.waiting_list or rx.waiting_by_rid)

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

    def test_nixl_uncertain_delivery_keeps_registration_until_fenced(self):
        async def run(fence_ok):
            transfer_id = "11" * 16
            handoff = EncoderHandoff.parse(
                msgspec.json.encode(
                    {
                        "version": 1,
                        "peer": "grpc://127.0.0.1:1",
                        "request_id": "r",
                        "transfer_id": transfer_id,
                        "part_id": transfer_id + "_local_part_0",
                        "part_idx": 0,
                        "num_parts": 1,
                        "image_count": 1,
                        "images": [{"t": 1, "h": 2, "w": 2, "identity": "aa" * 32}],
                    }
                ),
                1,
            )
            stub = SimpleNamespace(
                Describe=AsyncMock(
                    return_value=pb.DescribeResponse(shape=[1, 4], dtype="bfloat16")
                ),
                Deliver=AsyncMock(side_effect=RuntimeError("lost Deliver response")),
                Release=AsyncMock(
                    side_effect=[None]
                    if fence_ok
                    else [grpc.RpcError("lost fence"), None]
                ),
            )
            target = Mock(
                tensor=torch.empty(8, dtype=torch.uint8), descriptor=b"destination"
            )
            target.reaped.return_value = False
            released = asyncio.Event()
            target.close.side_effect = released.set
            rx = Mock()
            rx.allocate.return_value = target
            with patch.object(rpc, "SglangEncoderStub", return_value=stub):
                with self.assertRaisesRegex(ValueError, "shape differs"):
                    await handoff.receive_nixl(rx, (2, 4), 1)
                rx.allocate.assert_not_called()
                with self.assertRaises((RuntimeError, grpc.RpcError)):
                    await handoff.receive_nixl(rx, (1, 4), 1)
                if fence_ok:
                    target.abandon.assert_not_called()
                    stub.Release.assert_awaited_once()
                else:
                    target.close.assert_not_called()
                    target.abandon.assert_called_once()
                    await asyncio.wait_for(released.wait(), 2)
                    self.assertEqual(stub.Release.await_count, 2)
                target.close.assert_called_once()

        for fenced in (True, False):
            with self.subTest(fenced=fenced):
                asyncio.run(run(fenced))

    def test_native_part_to_prefill_and_decode_layout_only(self):
        async def run(backend, video_model=None, video_only=False):
            prompt = [42, 7, 8, 9, 43, 7, 8, 9, 44]
            grids = [[1, 2, 2], [1, 4, 2]]
            tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            identities = [
                MultimodalDataItem(
                    modality=Modality.IMAGE, precomputed_embeddings=e
                ).cache_key
                for e in [tensor[:1], tensor[1:]]
            ]
            video_layouts = []
            requested_images = ["first", "second"]
            requested_videos = []
            hints = [22, 33]
            if video_model:
                # Packing is [image 1, image 2, video], prompt order is [image 1, video, image 2].
                prompt[5:5] = [7, 10, 9]
                requested_videos = ["clip"]
                hints = [22, 101, 33]
                video = torch.tensor([[7.0, 8.0], [9.0, 10.0]])
                if video_only:
                    grids, identities, requested_images = [], [], []
                    prompt, hints, tensor = [10, 44], [101], video
                else:
                    tensor = torch.cat((tensor, video))
                timed_frames = video_model == "qwen3_vl"
                segments = video.split(1) if timed_frames else [video]
                vid_keys = [
                    MultimodalDataItem(
                        modality=Modality.VIDEO, precomputed_embeddings=t
                    ).cache_key.hex()
                    for t in segments
                ]
                video_layouts = [
                    pb.VideoLayout(
                        t=2,
                        h=2,
                        w=2,
                        identities=vid_keys,
                        timestamps=[0.25, 1.25] if timed_frames else [],
                        second_per_grid_ts=0.5 if video_model == "qwen2_5_vl" else None,
                    )
                ]
            rust = bridge()
            prefill, decode = request("p-incarnation"), request("d-incarnation")
            with zmq.Context() as ctx, ctx.socket(zmq.PUSH) as sock:
                sock.linger = 0
                rx = scheduler(ctx)
                if video_model:
                    rx.mm_processor.model_type = rx.model_type = video_model
                    rx.mm_processor._tokens_per_second = 4
                use_nixl = backend != "zmq_to_scheduler"
                rx.encoder_transfer_backend = "nixl" if use_nixl else backend
                writer = None
                if use_nixl:
                    rx.mm_processor.hf_config.vision_config = SimpleNamespace(
                        out_hidden_size=2, deepstack_visual_indexes=None
                    )
                if backend == "nixl-native":
                    from sglang.srt.disaggregation.encoder.nixl import (
                        NixlReceiver,
                        write_embedding,
                    )

                    rx.nixl_receiver, writer = NixlReceiver(capacity=1), write_embedding
                elif backend == "nixl-mock":
                    targets = {}

                    async def writer(source, destination, timeout):
                        targets[destination].tensor.copy_(
                            source.view(torch.uint8).flatten()
                        )

                    def allocate(size):
                        self.assertEqual(size, tensor.nbytes)
                        target = SimpleNamespace(
                            tensor=torch.empty(size, dtype=torch.uint8),
                            descriptor=b"target",
                            close=lambda: targets.pop(b"target"),
                        )
                        targets[target.descriptor] = target
                        return target

                    rx.nixl_receiver = SimpleNamespace(allocate=allocate)
                async with staged_encoder(
                    sock, grids, tensor, identities, writer, video_layouts
                ) as (
                    metadata,
                    calls,
                    held,
                ):
                    self.assertEqual(held, {metadata["part_id"]})
                    self.assertEqual(
                        calls, [("encode", requested_images, requested_videos)]
                    )
                    buffers = {
                        "input_ids": array("q", prompt),
                        "epd.handoff": msgspec.json.encode(metadata),
                        "epd.hashes": msgspec.msgpack.encode(hints),
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
                        await asyncio.wait_for(waiting.registration_future, 3)
                        if use_nixl:
                            self.assertIsNone(waiting.recv_socket)
                            delivery = ("write", metadata["part_id"])
                            if backend == "nixl-mock":
                                self.assertFalse(targets)
                        else:
                            self.assertTrue(waiting.recv_socket.poll(2000))
                            delivery = ("send", f"127.0.0.1:{waiting.embedding_port}")
                        self.assertEqual(
                            calls,
                            [
                                ("encode", requested_images, requested_videos),
                                delivery,
                            ],
                        )
                        self.assertFalse(held)
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

            self.assertEqual(prefill.input_ids, decode.input_ids)
            p, d = prefill.mm_inputs, decode.mm_inputs
            torch.testing.assert_close(p.mrope_positions, d.mrope_positions)
            torch.testing.assert_close(p.mrope_position_delta, d.mrope_position_delta)
            self.assertEqual(
                [i.pad_values for i in p.mm_items], [i.pad_values for i in d.mm_items]
            )
            if video_model:
                # Use ordinary Qwen EPD reconstruction as an independent reference.
                kwargs = {"video_grid_thw": torch.tensor([[2, 2, 2]])}
                native_embeddings = {Modality.VIDEO: video}
                if grids:
                    kwargs["img_grid_thw"] = torch.tensor(grids)
                    native_embeddings[Modality.IMAGE] = tensor[:3]
                if video_model == "qwen3_vl":
                    kwargs["video_timestamps"] = [[0.25, 1.25]]
                elif video_model == "qwen2_5_vl":
                    kwargs["second_per_grid_ts"] = torch.tensor([0.5])
                reference = rx.mm_processor.get_validated_mm_data(
                    prompt,
                    native_embeddings,
                    **kwargs,
                )
                expected_hashes = (
                    [22, 101, 101, 33] if video_model == "qwen3_vl" else [22, 101, 33]
                )
                if video_only:
                    expected_hashes = [101, 101]
                self.assertEqual(list(prefill.input_ids), reference.input_ids)
                for mm in (p, d):
                    self.assertEqual([i.hash for i in mm.mm_items], expected_hashes)
                    self.assertEqual(
                        [i.offsets for i in mm.mm_items],
                        [i.offsets for i in reference.mm_items],
                    )
                    torch.testing.assert_close(
                        mm.mrope_positions, reference.mrope_positions
                    )
                for staged, expected in zip(
                    p.mm_items, reference.mm_items, strict=True
                ):
                    torch.testing.assert_close(
                        staged.precomputed_embeddings, expected.precomputed_embeddings
                    )
            else:
                self.assertEqual(
                    list(prefill.input_ids), [42, 7, 8, 9, 43, 7, 8, 8, 9, 44]
                )
                for mm in (p, d):
                    self.assertEqual(
                        [item.offsets for item in mm.mm_items], [[(2, 2)], [(6, 7)]]
                    )
                    self.assertEqual(
                        [item.identity for item in mm.mm_items], identities
                    )
                    self.assertEqual([item.hash for item in mm.mm_items], [22, 33])
                torch.testing.assert_close(
                    torch.cat([item.precomputed_embeddings for item in p.mm_items]),
                    tensor,
                )
            self.assertTrue(
                all(item.precomputed_embeddings is None for item in d.mm_items)
            )
            rust._encoder_dispatcher.send_encode_request.assert_not_called()

        backends = ["zmq_to_scheduler", "nixl-mock"]
        if importlib.util.find_spec("nixl") is not None:
            backends.append("nixl-native")
        for backend in backends:
            for model, video_only in (
                (None, False),
                ("qwen3_vl", False),
                ("qwen2_5_vl", False),
                ("qwen3_vl", True),
            ):
                with self.subTest(backend=backend, model=model, video_only=video_only):
                    asyncio.run(run(backend, model, video_only))


if __name__ == "__main__":
    unittest.main()
