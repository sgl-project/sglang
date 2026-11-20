"""Two Gloo ranks must agree on staged media while only rank zero contacts E."""

import asyncio
import importlib.util
import pickle
from contextlib import AsyncExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch
import torch.distributed as dist
import zmq

from sglang.srt.disaggregation.encoder import receiver
from sglang.srt.disaggregation.encoder.handoff import EncoderHandoff
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.req_types import MultimodalDataItem
from sglang.srt.multimodal.modality import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from test_staged_image_receiver import request, scheduler, staged_encoder

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


def _group(rank):
    # The normal coordinator creates accelerator communicators. Exercise its
    # production metadata and tensor collectives using a real CPU process group.
    group = GroupCoordinator.__new__(GroupCoordinator)
    group.ranks, group.world_size, group.rank_in_group = [0, 1], 2, rank
    group.cpu_group = group.device_group = dist.group.WORLD
    group.mq_broadcaster = None
    return group


async def _exercise(rank, group, backend, result):
    counts = {"receive": 0, "release": 0}
    rows = torch.arange(8, dtype=torch.float32).reshape(4, 2)

    def digest(mod, data):
        return MultimodalDataItem(
            modality=mod, precomputed_embeddings=data
        ).cache_key.hex()

    transfer = "db" * 16
    wire = dict(
        version=2,
        peer="grpc://127.0.0.1:1",
        request_id="stage",
        transfer_id=transfer,
        part_id=transfer + "_local_part_0",
        part_idx=0,
        num_parts=1,
        image_count=2,
        images=[
            dict(t=1, h=2, w=2, identity=digest(Modality.IMAGE, rows[0:1])),
            dict(t=1, h=2, w=2, identity=digest(Modality.IMAGE, rows[1:2])),
        ],
        videos=[
            dict(
                t=2,
                h=2,
                w=2,
                timestamps=[0.0, 1.0],
                identities=[digest(Modality.VIDEO, part) for part in rows[2:].split(1)],
            )
        ],
    )
    if result == "layout-error":
        wire["videos"][0]["timestamps"] = []
    handoff = EncoderHandoff.parse(msgspec.json.encode(wire), 2, 1)

    def req(rid):
        r = request(rid)
        r.input_ids = [7, 8, 9, 7, 10, 9, 7, 8, 9]
        r.encoder_handoff = handoff
        r.need_wait_for_mm_inputs = True
        return r

    with zmq.Context() as ctx, ctx.socket(zmq.PUSH) as sender:
        rx = scheduler(ctx)
        rx.tp_size, rx.tp_rank, rx.tp_group = 2, rank, group
        rx.encoder_transfer_backend = backend
        rx.nixl_receiver = object()
        rx.create_req = lambda r: r
        rx.wait_timeout = 30
        if result == "peer-layout-error" and rank == 1:
            rx.scheduler.max_req_input_len = 1

        async def receive(*_):
            counts["receive"] += 1
            if result == "abort":
                await asyncio.Event().wait()
            if result == "error":
                raise ValueError("test encoder failure")
            return rows

        async def deliver(h, url, _timeout):
            await receive()
            sender.connect("tcp://" + url)
            frame = receiver.EmbeddingData(
                h.part_id, 1, 0, [[1, 2, 2], [1, 2, 2]], Modality.IMAGE, embedding=rows
            ).copy_without_embedding()
            sender.send_multipart([pickle.dumps(frame), rows.numpy().tobytes()])

        async def release(*_):
            counts["release"] += 1

        with (
            patch.object(EncoderHandoff, "receive_nixl", receive),
            patch.object(EncoderHandoff, "deliver", deliver),
            patch.object(EncoderHandoff, "release", release),
            patch.object(
                receiver,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ),
        ):
            p = req("p")
            ready, aborted = rx.process_waiting_requests([p])
            if result not in ("layout-error", "peer-layout-error"):
                assert not ready and not aborted
                waiting = rx.waiting_by_rid["p"]
                if rank == 1:
                    assert (
                        waiting.recv_socket is None
                        and waiting.registration_future is None
                    )
                if result == "abort":
                    await asyncio.sleep(0)
                    rx.abort_waiting_requests(
                        SimpleNamespace(matches=lambda rid: rid == "p")
                    )
                for _ in range(200):
                    await asyncio.sleep(0.015)
                    ready, aborted = rx.process_waiting_requests([])
                    if ready or aborted:
                        break
            await asyncio.sleep(0)
            if result == "ok":
                assert ready == [p] and not aborted
                items = p.mm_inputs.mm_items
                assert [i.modality for i in items] == [
                    Modality.IMAGE,
                    Modality.VIDEO,
                    Modality.VIDEO,
                    Modality.IMAGE,
                ]
                for item, row in zip(items, [0, 2, 3, 1]):
                    torch.testing.assert_close(
                        item.precomputed_embeddings, rows[row : row + 1]
                    )
            else:
                assert not ready and len(aborted) == 1
                reason = {
                    "layout-error": "metadata differs",
                    "peer-layout-error": "input length limit",
                    "abort": "Aborted by user",
                    "error": "test encoder failure",
                }[result]
                assert reason in aborted[0][1]
                if result in ("layout-error", "peer-layout-error", "abort"):
                    assert aborted[0][2] == 400
            assert not rx.waiting_list and not rx.waiting_by_rid
        if result == "ok":
            with patch.object(
                receiver,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
            ):
                d = req("d")
                ready, aborted = rx.process_waiting_requests([d])
                assert ready == [d] and not aborted
                assert d.input_ids == p.input_ids
                assert [i.cache_key for i in d.mm_inputs.mm_items] == [
                    i.cache_key for i in p.mm_inputs.mm_items
                ]
                assert all(
                    i.precomputed_embeddings is None for i in d.mm_inputs.mm_items
                )
    assert group.all_gather_object(counts) == [
        {
            "receive": int(result not in ("layout-error", "peer-layout-error")),
            "release": int(result != "ok"),
        },
        {"receive": 0, "release": 0},
    ]


async def _native_nixl(rank, group):
    from sglang.srt.disaggregation.encoder.nixl import NixlReceiver, write_embedding

    tensor = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    with zmq.Context() as ctx, ctx.socket(zmq.PUSH) as sender:
        async with AsyncExitStack() as stack:
            metadata = None
            if rank == 0:
                grids = [[1, 2, 2], [1, 4, 2]]
                identities = [
                    MultimodalDataItem(
                        modality=Modality.IMAGE, precomputed_embeddings=part
                    ).cache_key
                    for part in (tensor[:1], tensor[1:])
                ]
                metadata, calls, held = await stack.enter_async_context(
                    staged_encoder(sender, grids, tensor, identities, write_embedding)
                )
            metadata = group.broadcast_object(metadata)
            rx = scheduler(ctx)
            rx.tp_size, rx.tp_rank, rx.tp_group = 2, rank, group
            rx.encoder_transfer_backend = "nixl"
            rx.nixl_receiver = NixlReceiver(capacity=1) if rank == 0 else None
            rx.wait_timeout = 30
            p = request("native-p")
            p.input_ids = [7, 8, 9, 23, 7, 8, 9]
            p.encoder_handoff = EncoderHandoff.parse(msgspec.json.encode(metadata), 2)
            p.need_wait_for_mm_inputs = True
            with patch.object(
                receiver,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ):
                ready, aborted = rx.process_waiting_requests([p])
                for _ in range(300):
                    if ready or aborted:
                        break
                    await asyncio.sleep(0.03)
                    ready, aborted = rx.process_waiting_requests([])
            assert ready == [p] and not aborted
            torch.testing.assert_close(
                torch.cat([i.precomputed_embeddings for i in p.mm_inputs.mm_items]),
                tensor,
            )
            if rank == 0:
                assert [call[0] for call in calls] == ["encode", "write"]
                assert not held


def _tp_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        group = _group(rank)
        for backend in ("zmq_to_scheduler", "nixl"):
            for result in ("ok", "error", "layout-error", "peer-layout-error", "abort"):
                asyncio.run(_exercise(rank, group, backend, result))
        if importlib.util.find_spec("nixl") is not None:
            asyncio.run(_native_nixl(rank, group))
    finally:
        dist.destroy_process_group()


def test_staged_media_with_two_tensor_parallel_ranks(tmp_path):
    torch.multiprocessing.spawn(
        _tp_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2
    )
