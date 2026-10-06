"""GPU conformance for the prebuilt nvCOMP ABI; no extension compilation.

Manual-only: requires Blackwell, the paired Miles image, nvCOMP 5.3,
zstandard, python-snappy, and lz4. Missing hardware or dependencies fail this suite.
The hardware decoder preserves known bytes. Tests intentionally never feed malformed
compressed streams to the GPU decoder.
"""

import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.weight_sync.gpu_delta_apply import prepare_status_check
from sglang.srt.weight_sync.gpu_delta_codec import DecodeFrame, NvcompDecoder
from sglang.srt.weight_sync.gpu_delta_layout import (
    PreparedDelta,
    _plan_decode,
    _PreparedBatch,
)
from sglang.srt.weight_sync.gpu_delta_memory import HostAllocation


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
        frames.append(DecodeFrame(len(payload), len(encoded), offset, len(value)))
        payload.extend(encoded)
        offset += len(value)
    return payload, frames


@pytest.mark.parametrize("codec", ["snappy-zstd", "lz4-zstd"])
@pytest.mark.parametrize("sort_chunks", ["0", "1"])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_batched_plans_share_metadata_and_reuse_tensor_scratch(
    codec, sort_chunks, stages, monkeypatch
):
    monkeypatch.setenv("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS", sort_chunks)
    device = torch.device("cuda", 0)
    decoder = NvcompDecoder(device, codec)
    # Partial final frame, many zero bytes, and valid nonzero XOR values.
    values = [
        [bytes(1 << 20), bytes(range(256)) * 4096, bytes(range(253)) * 3],
        [bytes([7]) * (64 << 10), bytes(range(251)) * 3],
    ]
    payload_a, frames_a = _encode(values[0], codec)
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
            DecodeFrame(
                input_b + f.input_offset,
                f.encoded_bytes,
                f.output_offset,
                f.decoded_bytes,
            )
            for f in frames_b
        ],
    ]
    batches = [batches[index % 2] for index in range(2 * stages + 1)]
    stream = torch.cuda.Stream(device=device)
    de_stream = torch.cuda.Stream(device=device)
    workspace = decoder.allocate_workspace(batches, slot_count=stages)
    plan = decoder.prepare_batches(batches, host, workspace, de_stream)
    # Large outputs are allocated only at the paused binding boundary.
    decoded = [
        torch.empty(sum(map(len, values[0])), dtype=torch.uint8, device=device)
        for _ in range(stages)
    ]
    plans = plan.bind_outputs(decoded)
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
    assert plans[0].statuses.data_ptr() == plans[stages].statuses.data_ptr()
    assert plans[0].actual_sizes.data_ptr() == plans[stages].actual_sizes.data_ptr()
    assert all(plan.host_input is host for plan in plans)
    prepared = PreparedDelta.__new__(PreparedDelta)
    prepared.timing_enabled = False
    prepared.decode_stages = stages
    prepared.stream, prepared.de_stream = stream, de_stream
    prepared.decoded_ready = [torch.cuda.Event() for _ in range(stages)]
    prepared.decoded_free = [torch.cuda.Event() for _ in range(stages)]
    # B leaves a prefix, an interior hole and a short tail; the unused part of
    # larger scratch retains its prior bytes and is not read by B's apply plan.
    output_sizes = [
        decoded[0].numel()
        if index % 2 == 0
        else frames_b[-1].output_offset + frames_b[-1].decoded_bytes + 64
        for index in range(len(batches))
    ]
    prepared_batches = []
    # Output/workspace allocation happened on the current stream. The setup
    # event below carries this dependency onward to the DE stream as well.
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.device(device), torch.cuda.stream(stream):
        prepared.error = torch.zeros(1, dtype=torch.int32, device=device)
        for index, (plan, frames, size) in enumerate(zip(plans, batches, output_sizes)):
            binding = SimpleNamespace(name="tensor")
            mapped, gaps = _plan_decode(
                [(binding, 0, size)],
                {
                    "tensor": {
                        "frames": [
                            {
                                "encoded_offset": f.input_offset,
                                "encoded_bytes": f.encoded_bytes,
                                "decoded_offset": f.output_offset,
                                "decoded_bytes": f.decoded_bytes,
                            }
                            for f in frames
                        ]
                    }
                },
                {"tensor": {"offset": 0}},
            )
            assert mapped == frames
            prepared_batches.append(
                _PreparedBatch(
                    plan,
                    None,
                    [],
                    [
                        decoded[index % stages][offset : offset + length]
                        for offset, length in gaps
                    ],
                    prepare_status_check(plan, prepared.error),
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
    # Alternating A/B batches cross two ring boundaries and reuse every slot.
    # DE overlaps prior apply; reuse waits until output/status readers complete.
    # One final apply fence joins all work, including metadata upload on DE.
    with torch.cuda.device(device), torch.cuda.stream(stream):
        prepared._decode_batch(prepared_batches[0], 0)
        for index, (plan, batch) in enumerate(zip(plans, prepared_batches)):
            stream.wait_event(prepared.decoded_ready[index % stages])
            prepared._apply_batch(batch)
            observations.append(
                (
                    decoded[index % stages].clone(),
                    plan.statuses.clone(),
                    plan.actual_sizes.clone(),
                )
            )
            prepared.decoded_free[index % stages].record(stream)
            if index + 1 < len(plans):
                prepared._decode_batch(prepared_batches[index + 1], index + 1)
        complete = torch.cuda.Event()
        complete.record(stream)
    complete.synchronize()
    expected_slots = [bytearray([0xA5]) * output.numel() for output in decoded]
    for index, (output, statuses, sizes) in enumerate(observations):
        expected = expected_slots[index % stages]
        expected[: output_sizes[index]] = bytes(output_sizes[index])
        source_values = values[index % 2]
        for frame, value in zip(batches[index], source_values):
            expected[frame.output_offset : frame.output_offset + len(value)] = value
        assert statuses.tolist() == [0] * len(batches[index])
        assert sizes.tolist() == list(map(len, source_values))
        assert plans[index].expected_sizes.tolist() == sizes.tolist()
        assert bytes(output.cpu().numpy()) == expected
    assert prepared.error.item() == 0
    assert bytes(host[: len(payload_a)].numpy()) == payload_a
    assert bytes(host[input_b : input_b + len(payload_b)].numpy()) == payload_b
    # Each actual slot's callback reads that slot's status/size row, including
    # newly added slots. Mutate metadata only after the real DE work completes.
    for index, batch in enumerate(prepared_batches[:stages]):
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
    # Original host input is released only after both streams' readers finish.
    del host
    allocation.close()


def test_rejects_frame_range_before_decode():
    device = torch.device("cuda", 0)
    decoder = NvcompDecoder(device, "snappy-zstd")
    allocation = HostAllocation(256, device.index)
    host = torch.frombuffer(allocation.view, dtype=torch.uint8)[:256]
    decoded = tuple(
        torch.empty(256, dtype=torch.uint8, device=device) for _ in range(2)
    )
    stream = torch.cuda.Stream(device=device)
    good = [DecodeFrame(0, 32, 0, 128)]
    workspace = decoder.allocate_workspace([good])
    with pytest.raises(ValueError, match="outside input"):
        decoder.prepare_batches(
            [good, [DecodeFrame(240, 32, 0, 128)]],
            host,
            workspace,
            stream,
        )
    plan = decoder.prepare_batches(
        [good, [DecodeFrame(0, 32, 250, 128)]], host, workspace, stream
    )
    with pytest.raises(ValueError, match="Out-of-bounds"):
        plan.bind_outputs(decoded)
    stream.synchronize()
    del plan
    del host
    allocation.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
