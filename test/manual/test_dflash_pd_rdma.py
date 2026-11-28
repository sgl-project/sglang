"""Actual GPU RDMA of a bounded source slab into mixed full/window rows."""

import json
import multiprocessing as mp
import os
import socket
import time
from dataclasses import asdict
from pathlib import Path

import torch

from sglang.srt.disaggregation.draft_lane import (
    DraftLease,
    DraftRegion,
    plan_region_copy,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.speculative.dflash_independent_cache import DraftSlotLayout

ROOT = Path(os.environ.get("DFLASH_TEST_OUTPUT_DIR", "dflash-pd-rdma-results"))


def regions_for(tensors):
    return [
        DraftRegion(
            layer,
            kind,
            "FULL" if layer == 4 else "WINDOW",
            tensor.data_ptr(),
            tensor.numel() * tensor.element_size(),
            tensor.stride(0) * tensor.element_size(),
            tensor.shape[0],
        )
        for (layer, kind), tensor in tensors.items()
    ]


def receiver(pipe):
    torch.cuda.set_device(1)
    host = socket.gethostbyname(socket.gethostname())
    engine = MooncakeTransferEngine(
        host, gpu_id=1, ib_device=os.environ.get("DFLASH_TEST_RECEIVER_IB_DEVICE")
    )
    layout = DraftSlotLayout(2, 8192, 64, 12, 2048)
    tensors = {
        (layer, kind): torch.full(
            (
                64 + (layout.full_capacity if layer == 4 else layout.window_capacity),
                8,
                128,
            ),
            -99,
            dtype=torch.bfloat16,
            device="cuda:1",
        )
        for layer in range(5)
        for kind in ("K", "V")
    }
    regions = regions_for(tensors)
    engine.batch_register(
        [r.base for r in regions], [r.registered_bytes for r in regions]
    )
    torch.cuda.synchronize()
    lease = DraftLease(
        "real-rdma",
        99,
        "request",
        1,
        1,
        2,
        6000,
        8192,
        layout.full_loc(2, 0),
        layout.stride,
        layout.window_loc(2, 0),
        layout.window_stride,
        "real-epoch",
    )
    pipe.send((engine.session_id, [asdict(r) for r in regions], asdict(lease)))
    if not pipe.poll(100):
        raise RuntimeError("No source transfer completion")
    assert pipe.recv() == "DONE"
    torch.cuda.synchronize()
    for (layer, kind), tensor in tensors.items():
        positions = torch.arange(
            0 if layer == 4 else lease.window_floor, lease.logical_end, device="cuda:1"
        )
        rows = (
            lease.full_row0 + positions
            if layer == 4
            else lease.window_row0 + positions.remainder(lease.window_stride)
        )
        expected = (
            positions.remainder(997) + layer * 10 + (1000 if kind == "V" else 0)
        ).to(torch.bfloat16)
        actual = tensor[rows]
        torch.testing.assert_close(
            actual, expected[:, None, None].expand_as(actual), rtol=0, atol=0
        )
        # Unrelated owner1/slot1 must remain untouched.
        assert torch.all(
            tensor[64 : 64 + (layout.stride if layer == 4 else layout.window_stride)]
            == -99
        ).item()
    pipe.send("PASS")
    if not pipe.poll(30):
        raise RuntimeError("No cleanup acknowledgment")
    assert pipe.recv() == "CLEANUP"
    assert engine.batch_deregister([r.base for r in regions]) == 0


def sender(pipe):
    torch.cuda.set_device(0)
    host = socket.gethostbyname(socket.gethostname())
    engine = MooncakeTransferEngine(
        host, gpu_id=0, ib_device=os.environ.get("DFLASH_TEST_SENDER_IB_DEVICE")
    )
    layout = DraftSlotLayout(2, 4096, 64, 12)
    tensors = {
        (layer, kind): torch.zeros(
            (64 + layout.full_capacity, 8, 128), dtype=torch.bfloat16, device="cuda:0"
        )
        for layer in range(5)
        for kind in ("K", "V")
    }
    regions = regions_for(tensors)
    engine.batch_register(
        [r.base for r in regions], [r.registered_bytes for r in regions]
    )
    if not pipe.poll(100):
        raise RuntimeError("No destination registration")
    session, raw_regions, raw_lease = pipe.recv()
    dest = {r.key: r for r in map(lambda x: DraftRegion(**x), raw_regions)}
    lease = DraftLease(**raw_lease).validate()
    byte_count = 0
    block_count = 0
    for start, end in [(0, 4000), (4000, 6000)]:
        positions = torch.arange(start, end, device="cuda:0")
        source_rows = 64 + layout.stride + positions.remainder(layout.stride)
        for (layer, kind), tensor in tensors.items():
            values = (
                positions.remainder(997) + layer * 10 + (1000 if kind == "V" else 0)
            ).to(torch.bfloat16)
            tensor[source_rows] = values[:, None, None]
        torch.cuda.synchronize()
        blocks = []
        for region in regions:
            blocks += plan_region_copy(
                region,
                dest[region.key],
                source_rows.cpu().tolist(),
                positions.cpu().tolist(),
                lease,
            )
        src, dst, sizes = zip(*blocks)
        assert (
            engine.batch_transfer_sync(session, list(src), list(dst), list(sizes)) == 0
        )
        byte_count += sum(sizes)
        block_count += len(blocks)
    pipe.send("DONE")
    if not pipe.poll(100):
        raise RuntimeError("No destination verification")
    assert pipe.recv() == "PASS"
    pipe.send("CLEANUP")
    assert engine.batch_deregister([r.base for r in regions]) == 0
    result = dict(
        state="PASS",
        transport="Mooncake actual GPU RDMA",
        source_gpu=0,
        destination_gpu=1,
        owner=1,
        private_slot=2,
        prompt_tokens=6000,
        window_stride=2176,
        source_chunk_tokens=4096,
        bytes=byte_count,
        rdma_blocks=block_count,
        checks=[
            "all full-history rows",
            "all window-tail rows",
            "source slab wrap",
            "destination window wrap",
            "unrelated owner isolation",
        ],
    )
    (ROOT / "typed-draft-real-rdma.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    a, b = context.Pipe()
    children = [
        context.Process(target=receiver, args=(a,)),
        context.Process(target=sender, args=(b,)),
    ]
    for child in children:
        child.start()
    deadline = time.monotonic() + 160
    for child in children:
        child.join(max(1, deadline - time.monotonic()))
    for child in children:
        if child.is_alive():
            child.terminate()
            child.join(10)
    assert all(child.exitcode == 0 for child in children), [
        child.exitcode for child in children
    ]
