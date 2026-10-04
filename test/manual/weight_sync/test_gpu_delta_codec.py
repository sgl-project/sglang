"""GPU conformance for the prebuilt nvCOMP ABI; no extension compilation.

Manual-only: requires Blackwell, the paired Miles image, nvCOMP 5.3,
zstandard, and python-snappy. Missing hardware or dependencies fail this suite.
The hardware decoder preserves known bytes. Tests intentionally never feed malformed
compressed streams to the GPU decoder.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.weight_sync.gpu_delta_apply import prepare_status_check
from sglang.srt.weight_sync.gpu_delta_codec import DecodeFrame, NvcompDecoder
from sglang.srt.weight_sync.gpu_delta_layout import (
    PreparedDelta,
    _decoded_gaps,
    _PreparedBatch,
)


def _encode(values, offsets=None):
    import snappy

    compress = snappy.compress
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


def test_batched_plans_share_metadata_and_reuse_tensor_scratch():
    device = torch.device("cuda", 0)
    decoder = NvcompDecoder(device)
    # Partial final frame, many zero bytes, and valid nonzero XOR values.
    values = [
        [bytes(1 << 20), bytes(range(256)) * 4096, bytes(range(253)) * 3],
        [bytes([7]) * (64 << 10), bytes(range(251)) * 3],
    ]
    payload_a, frames_a = _encode(values[0])
    # A different frame count, partial final frame and sparse output offsets
    # exercise views whose row stride is the complete metadata slab's width.
    payload_b, frames_b = _encode(values[1], [128, 128 + (64 << 10) + 256])
    batches = [frames_a, frames_b]
    hosts = []
    for payload in (payload_a, payload_b):
        host = torch.empty(
            len(payload), dtype=torch.uint8, device="cpu", pin_memory=True
        )
        host.numpy()[:] = memoryview(payload)
        hosts.append(host)
    stream = torch.cuda.Stream(device=device)
    encoded = torch.empty(
        max(host.numel() for host in hosts), dtype=torch.uint8, device=device
    )
    decoded = torch.empty(sum(map(len, values[0])), dtype=torch.uint8, device=device)
    workspace = decoder.allocate_workspace(batches)
    plans = decoder.prepare_batches(batches, encoded, decoded, workspace, stream)
    assert plans[0].metadata.untyped_storage().data_ptr() == (
        plans[1].metadata.untyped_storage().data_ptr()
    )
    assert plans[0].host_metadata.untyped_storage().data_ptr() == (
        plans[1].host_metadata.untyped_storage().data_ptr()
    )
    assert all(plan.metadata.stride(0) == 5 for plan in plans)
    assert plans[0].statuses.data_ptr() == plans[1].statuses.data_ptr()
    prepared = PreparedDelta.__new__(PreparedDelta)
    prepared.timing_enabled = False
    # B leaves a prefix, an interior hole and a short tail; the rest of the
    # larger scratch still belongs to A and is not read by B's apply plan.
    output_sizes = [
        decoded.numel(),
        frames_b[-1].output_offset + frames_b[-1].decoded_bytes + 64,
    ]
    prepared_batches = []
    with torch.cuda.device(device), torch.cuda.stream(stream):
        prepared.error = torch.zeros(1, dtype=torch.int32, device=device)
        for plan, size in zip(plans, output_sizes):
            gaps = _decoded_gaps(
                [(SimpleNamespace(name="weight"), 0, size)],
                {
                    "weight": {
                        "frames": [
                            {
                                "decoded_offset": f.output_offset,
                                "decoded_bytes": f.decoded_bytes,
                            }
                            for f in plan.frames
                        ]
                    }
                },
            )
            prepared_batches.append(
                _PreparedBatch(
                    [],
                    plan,
                    [],
                    [],
                    [decoded[offset : offset + length] for offset, length in gaps],
                    prepare_status_check(plan, prepared.error),
                )
            )
    assert not prepared_batches[0].zero_ranges
    assert [view.numel() for view in prepared_batches[1].zero_ranges] == [128, 256, 64]
    observations = []
    # A -> B -> A reuses both the encoded slot and shared status/output scratch
    # on one stream, including the initial metadata upload, with one final fence.
    with torch.cuda.device(device), torch.cuda.stream(stream):
        decoded.fill_(0xA5)
        for index in (0, 1, 0):
            plan, host = plans[index], hosts[index]
            encoded[: host.numel()].copy_(host, non_blocking=True)
            prepared._decode_batch(prepared_batches[index])
            observations.append(
                (decoded.clone(), plan.statuses.clone(), plan.actual_sizes.clone())
            )
        complete = torch.cuda.Event()
        complete.record(stream)
    complete.synchronize()
    expected = bytearray([0xA5]) * decoded.numel()
    for index, (output, statuses, sizes) in zip((0, 1, 0), observations):
        expected[: output_sizes[index]] = bytes(output_sizes[index])
        for frame, value in zip(batches[index], values[index]):
            expected[frame.output_offset : frame.output_offset + len(value)] = value
        assert statuses.tolist() == [0] * len(batches[index])
        assert sizes.tolist() == list(map(len, values[index]))
        assert plans[index].expected_sizes.tolist() == sizes.tolist()
        assert bytes(output.cpu().numpy()) == expected
    assert prepared.error.item() == 0

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


def test_rejects_frame_range_before_decode():
    device = torch.device("cuda", 0)
    decoder = NvcompDecoder(device)
    encoded = torch.empty(256, dtype=torch.uint8, device=device)
    decoded = torch.empty(256, dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)
    good = [DecodeFrame(0, 32, 0, 128)]
    workspace = decoder.allocate_workspace([good])
    with pytest.raises(ValueError, match="outside input"):
        decoder.prepare_batches(
            [good, [DecodeFrame(240, 32, 0, 128)]],
            encoded,
            decoded,
            workspace,
            stream,
        )
    with pytest.raises(ValueError, match="Overlapping"):
        decoder.prepare_batches(
            [good, [DecodeFrame(0, 32, 250, 128)]],
            encoded,
            decoded,
            workspace,
            stream,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
