"""GPU conformance for the prebuilt nvCOMP ABI; no extension compilation.

Manual-only: DE cases require SM100/SM103; CUDA decode cases do not require DE.
All cases use the paired Miles image, nvCOMP 6, zstandard, python-snappy, and lz4.
Missing hardware or dependencies fail this suite. Both decoder paths preserve known
bytes. Tests intentionally never feed malformed compressed streams to the GPU decoder.
"""

import sys
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.srt.weight_sync.gpu_delta.apply import prepare_status_check
from sglang.srt.weight_sync.gpu_delta.codec import NvcompDecoder
from sglang.srt.weight_sync.gpu_delta.host_worker import TENSOR_DTYPE
from sglang.srt.weight_sync.gpu_delta.memory import HostAllocation, device_empty
from sglang.srt.weight_sync.gpu_delta.receiver import (
    PreparedDelta,
    _plan_decode,
    _PreparedBatch,
)


def _frame_table(batches):
    rows = [row for batch in batches for row in batch]
    table = np.asarray(rows, dtype=np.int64).reshape(-1, 4).T.copy()
    return table, list(map(len, batches))


def _encode(values, codec, offsets=None):
    import lz4.block
    import snappy

    compress = (
        snappy.compress
        if codec == "snappy-zstd"
        else lambda value: lz4.block.compress(value, store_size=False)
    )
    payload = bytearray()
    frames = []
    offset = 0
    for index, value in enumerate(values):
        if offsets is not None:
            offset = offsets[index]
        payload.extend(bytes((-len(payload)) % 16))
        encoded = compress(value)
        frames.append((len(payload), len(encoded), len(value), offset))
        payload.extend(encoded)
        offset += len(value)
    return payload, frames


# Outer Zstd unwrapping is covered by the CPU payload suite; this exercises the two
# distinct inner decoders with two output slots and both sorting settings.
@pytest.mark.parametrize(
    "codec,sort_chunks",
    [
        ("snappy-zstd", "0"),
        ("snappy-zstd", "1"),
        ("lz4", "0"),
        ("lz4", "1"),
    ],
)
def test_batched_plans_share_metadata_and_reuse_tensor_scratch(
    codec, sort_chunks, monkeypatch
):
    monkeypatch.setenv("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS", sort_chunks)
    device = torch.device("cuda", 0)
    decoder = NvcompDecoder(device, codec.removesuffix("-zstd"))
    # Full 4 MiB nonzero frame, 1 MiB zero frame, and a partial final frame.
    values = [
        [bytes(1 << 20), bytes(range(256)) * 16384, bytes(range(253)) * 3],
        [bytes([7]) * (64 << 10), bytes(range(251)) * 3],
    ]
    payload_a, frames_a = _encode(values[0], codec)
    assert frames_a[1][2] == 4 << 20
    assert decoder.backend == "hardware" and decoder._options.backend == 1
    # A different frame count, partial final frame and sparse output offsets
    # exercise views whose row stride is the complete metadata slab's width.
    payload_b, frames_b = _encode(values[1], codec, [128, 128 + (64 << 10) + 256])
    input_b = (len(payload_a) + 15) // 16 * 16
    allocation = HostAllocation(input_b + len(payload_b), device.index)
    host = torch.frombuffer(allocation.view, dtype=torch.uint8)
    host[: len(payload_a)].numpy()[:] = memoryview(payload_a)
    host[input_b : input_b + len(payload_b)].numpy()[:] = memoryview(payload_b)
    batches = [
        frames_a,
        [
            (input_b + offset, encoded, decoded, output)
            for offset, encoded, decoded, output in frames_b
        ],
    ]
    batches = [batches[index % 2] for index in range(5)]
    batches[-1] = []  # A fully omitted final batch still zeroes/reuses its slot.
    stream = torch.cuda.Stream(device=device)
    de_stream = torch.cuda.Stream(device=device)
    table, counts = _frame_table(batches)
    plan = decoder.prepare_batches(table, counts, host, de_stream)
    assert plan.stream is de_stream
    workspace = plan.workspace
    # Large outputs are allocated only at the paused binding boundary.
    decoded = [
        device_empty(sum(map(len, values[0])), dtype=torch.uint8, device=device)
        for _ in range(2)
    ]
    plans = plan.bind_outputs(decoded)
    # Native status/size views exercise both typed allocation paths, including
    # nonzero-offset ring slots; successful decode below verifies their contents.
    assert workspace.statuses.dtype == torch.int32
    assert workspace.actual_sizes.dtype == plan.metadata.dtype == torch.int64
    assert workspace.statuses.device == workspace.actual_sizes.device == device
    plan.record_stream(stream)
    del plan
    assert plans[0].metadata.untyped_storage().data_ptr() == (
        plans[1].metadata.untyped_storage().data_ptr()
    )
    assert plans[0].host_metadata.untyped_storage().data_ptr() == (
        plans[1].host_metadata.untyped_storage().data_ptr()
    )
    assert all(plan.metadata.stride(0) == sum(map(len, batches)) for plan in plans)
    assert plans[0].statuses.data_ptr() != plans[1].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() != plans[1].actual_sizes.data_ptr()
    assert plans[0].statuses.data_ptr() == plans[2].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() == plans[2].actual_sizes.data_ptr()
    assert all(plan.host_input is host for plan in plans)
    for batch in plans:
        assert batch.stream is de_stream
        assert batch._arguments[-1] == batch.stream.cuda_stream
        temporary = workspace.temporary
        assert batch._arguments[5:7] == (temporary.data_ptr(), temporary.numel())
    prepared = PreparedDelta.__new__(PreparedDelta)
    prepared.device = device
    prepared.stream, prepared.io_stream = stream, de_stream
    prepared.hardware = True
    prepared.decoded_ready = [torch.cuda.Event() for _ in range(2)]
    prepared.decoded_free = [torch.cuda.Event() for _ in range(2)]
    # B leaves a prefix, an interior hole and a short tail; the unused part of
    # larger scratch retains its prior bytes and is not read by B's apply plan.
    output_sizes = [
        decoded[0].numel() if index % 2 == 0 else frames_b[-1][3] + frames_b[-1][2] + 64
        for index in range(len(batches))
    ]
    output_sizes[-1] = 64
    prepared_batches = []
    # Output/workspace allocation happened on the current stream. The setup
    # event below carries this dependency onward to the DE stream as well.
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.device(device), torch.cuda.stream(stream):
        prepared.error = torch.zeros(1, dtype=torch.int32, device=device)
        for index, (plan, frames, size) in enumerate(zip(plans, batches, output_sizes)):
            tensors = np.zeros(1, dtype=TENSOR_DTYPE)
            tensors["frame_count"] = len(frames)
            holes, cursor = [], 0
            for _, _, count, offset in frames:
                if offset > cursor:
                    holes.append((cursor, offset - cursor))
                cursor = offset + count
            if cursor < size:
                holes.append((cursor, size - cursor))
            tensors["gap_count"] = len(holes)
            publication = SimpleNamespace(
                tensors=tensors,
                frames=_frame_table([frames])[0],
                gaps=np.asarray(holes, dtype="<i8").reshape(-1, 2),
            )
            mapped, mapped_counts, gaps = _plan_decode(
                [(np.array([[0], [0]], dtype="<i8"), size, None)],
                publication,
                np.zeros(1, dtype=np.int64),
            )
            np.testing.assert_array_equal(mapped, _frame_table([frames])[0])
            assert mapped_counts == [len(frames)]
            prepared_batches.append(
                _PreparedBatch(
                    decoder=plan,
                    apply=None,
                    zero_ranges=[
                        decoded[index % 2][offset : offset + length]
                        for offset, length in gaps[0]
                    ],
                    check_status=prepare_status_check(plan, prepared.error),
                )
            )
        for output in decoded:
            output.fill_(0xA5)
        ready = torch.cuda.Event()
        ready.record(stream)
    de_stream.wait_event(ready)
    assert not prepared_batches[0].zero_ranges
    assert [view.numel() for view in prepared_batches[1].zero_ranges] == [128, 256, 64]
    observations = []

    # Run the production DE/apply loop, retaining snapshots as an ordered
    # apply consumer. Prepared metadata/output binding above is already real.
    # Distinct A/B bytes cross two ring boundaries, including the final empty
    # batch reusing the first output/status slot after both earlier decodes.
    def observe(batch):
        index = len(observations)
        PreparedDelta._apply_batch(prepared, batch)
        observations.append(
            (
                prepared.decoded[index % 2].clone(),
                batch.decoder.statuses.clone(),
                batch.decoder.actual_sizes.clone(),
            )
        )

    prepared.backend = SimpleNamespace(
        layout=SimpleNamespace(check_identity=lambda: None)
    )
    prepared._allocate_paused = lambda: None
    prepared._apply_batch = observe
    prepared.timings, prepared.events = {}, {}
    prepared.raw_copies, prepared.derived = {}, []
    prepared.batches = list(prepared_batches)
    prepared.status_checks, prepared.apply_bindings, prepared.apply_source_patches = (
        [],
        [],
        [],
    )
    prepared.apply_fallback_groups = set()
    prepared.decoded, prepared.workspace, prepared.decode_plan = (
        decoded,
        workspace,
        None,
    )
    prepared.target_version, prepared.matrix_tensor_count, prepared.raw_tensor_count = (
        1,
        1,
        0,
    )
    prepared.h2d_bytes = 0
    error = prepared.error
    assert prepared.apply()["applied"]
    prepared.error = error
    del prepared._apply_batch, prepared._allocate_paused
    expected_slots = [bytearray([0xA5]) * output.numel() for output in decoded]
    for index, (output, statuses, sizes) in enumerate(observations):
        expected = expected_slots[index % 2]
        expected[: output_sizes[index]] = bytes(output_sizes[index])
        source_values = values[index % 2] if batches[index] else []
        for frame, value in zip(batches[index], source_values):
            expected[frame[3] : frame[3] + len(value)] = value
        assert statuses.tolist() == [0] * len(batches[index])
        assert sizes.tolist() == list(map(len, source_values))
        assert plans[index].expected_sizes.tolist() == sizes.tolist()
        assert bytes(output.cpu().numpy()) == expected
    assert prepared.error.item() == 0
    assert bytes(host[: len(payload_a)].numpy()) == payload_a
    assert bytes(host[input_b : input_b + len(payload_b)].numpy()) == payload_b
    # Each slot's callback reads its own status/size row. Mutate metadata only
    # after the real DE work completes.
    for index, batch in enumerate(prepared_batches[:2]):
        field = batch.decoder.statuses if index % 2 == 0 else batch.decoder.actual_sizes
        prepared.error.zero_()
        before = field[0].clone()
        field[0].add_(1)
        batch.check_status()
        assert prepared.error.item() == 1
        field[0].copy_(before)
        batch.check_status()
        assert prepared.error.item() == 1
    # Inject status metadata only, never a malformed GPU-compressed stream.
    # Failure in either CTA and a size mismatch all set the same sticky flag;
    # a later successful batch must not clear it.
    status = SimpleNamespace(
        statuses=torch.zeros(1025, dtype=torch.int32, device=device),
        actual_sizes=torch.ones(1025, dtype=torch.int64, device=device),
        expected_sizes=torch.ones(1025, dtype=torch.int64, device=device),
    )
    check = prepare_status_check(status, prepared.error)
    for field, index in (
        (status.statuses, 0),
        (status.statuses, 1024),
        (status.actual_sizes, 1024),
    ):
        prepared.error.zero_()
        before = field[index].clone()
        field[index].add_(1)
        check()
        assert prepared.error.item() == 1
        field[index].copy_(before)
        check()
        assert prepared.error.item() == 1
    assert decoder.backend == "hardware"
    # Exercise the same final-release path after all DE/apply/status work drains.
    # Per-batch leases keep every decoded slot alive after the plan is discarded.
    output_refs = [weakref.ref(output) for output in decoded]
    prepared.batches = prepared_batches
    prepared.status_checks = []
    prepared.apply_bindings, prepared.apply_source_patches = [], []
    prepared.apply_fallback_groups = set()
    prepared.raw_copies = {}
    prepared.decoded = decoded
    prepared.workspace = workspace
    prepared.decode_plan = None
    del decoded
    assert all(ref() is not None for ref in output_refs)
    prepared._release_gpu()
    plans.clear()
    del plan, batch
    assert all(ref() is None for ref in output_refs)
    # Original host input is released only after all streams' readers finish.
    del host
    allocation.close()


@pytest.mark.parametrize("algorithm", ["bitcomp", "snappy", "lz4"])
def test_cuda_decode_overlaps_h2d_and_reuses_double_buffer(algorithm, monkeypatch):
    from miles.utils.gpu_delta.nvcomp import NvcompCompressor

    from sglang.srt.weight_sync.gpu_delta import codec

    # Force the CUDA path for Snappy/LZ4 even on the Blackwell validation node.
    monkeypatch.setattr(codec, "decoder_backend", lambda device, inner_codec: "cuda")
    device = torch.device("cuda", 0)
    compute, h2d = [torch.cuda.Stream(device=device) for _ in range(2)]
    values = [bytes(1 << 20), bytes(range(251)) * 3, bytes([17]) * (64 << 10)]
    with torch.cuda.stream(compute):
        tensors = [
            torch.frombuffer(bytearray(value), dtype=torch.uint8).to(device)
            for value in values
        ]
        encoded = NvcompCompressor(algorithm, device).compress(tensors, compute)
    compute.synchronize()
    assert encoded.statuses.tolist() == [0] * len(values)
    payload, frames = bytearray(), []
    for value, output, size in zip(values, encoded.outputs, encoded.sizes.tolist()):
        payload.extend(bytes((-len(payload)) % 16))
        frames.append((len(payload), size, len(value), 0))
        payload.extend(output[:size].cpu().numpy().tobytes())
    host = torch.empty(len(payload), dtype=torch.uint8, pin_memory=True)
    host.numpy()[:] = np.frombuffer(payload, dtype=np.uint8)
    groups = [[0, 1], [2], [1, 2], [0], [2, 0, 1], []]
    batches, references = [], []
    for group in groups:
        rows, reference = [], bytearray()
        for index in group:
            reference.extend(bytes([0xA5]) * ((-len(reference)) % 16))
            source, encoded_size, decoded_size, _ = frames[index]
            rows.append((source, encoded_size, decoded_size, len(reference)))
            reference.extend(values[index])
        batches.append(rows)
        references.append(bytes(reference))
    decoder = NvcompDecoder(device, algorithm)
    assert decoder.backend == "cuda" and decoder._options.backend == 2
    table, counts = _frame_table(batches)
    plan = decoder.prepare_batches(table, counts, host, compute)
    assert not plan.input_slots  # No compressed device arena before the pause.
    with torch.cuda.stream(compute):
        outputs = [
            torch.empty(max(map(len, references)), dtype=torch.uint8, device=device)
            for _ in range(2)
        ]
        plans = plan.bind_outputs(outputs)
        error = torch.zeros(1, dtype=torch.int32, device=device)
        checks = [prepare_status_check(batch, error) for batch in plans]
        initialized = torch.cuda.Event()
        initialized.record(compute)
    h2d.wait_event(initialized)
    ready, free = [[torch.cuda.Event() for _ in range(2)] for _ in range(2)]
    snapshots = []
    plans[0].copy_input(h2d)
    ready[0].record(h2d)
    for index, batch in enumerate(plans):
        slot = index % 2
        compute.wait_event(ready[slot])
        if index + 1 < len(plans):
            next_slot = (index + 1) % 2
            if index >= 1:
                h2d.wait_event(free[next_slot])
            plans[index + 1].copy_input(h2d)
            ready[next_slot].record(h2d)
        with torch.cuda.stream(compute):
            outputs[slot].fill_(0xA5)
            batch.enqueue()
            free[slot].record(compute)
            checks[index]()
            if batches[index]:
                snapshots.append(outputs[slot][: len(references[index])].clone())
    compute.synchronize()
    h2d.synchronize()
    assert error.item() == 0
    assert len(plan.input_slots) == 2
    assert plan.input_capacity_bytes <= 2 * len(payload)
    for index, actual in enumerate(snapshots):
        assert actual.cpu().numpy().tobytes() == references[index]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
