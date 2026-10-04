"""GPU conformance for the prebuilt nvCOMP ABI; no extension compilation.

Manual-only: requires Blackwell, the paired Miles image, nvCOMP 5.3,
zstandard, and python-snappy. Missing hardware or dependencies fail this suite.
The hardware decoder preserves known bytes. Tests intentionally never feed malformed
compressed streams to the GPU decoder.
"""

import sys

import pytest
import torch

from sglang.srt.weight_sync.gpu_delta_codec import DecodeFrame, NvcompDecoder


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
    observations = []
    # A -> B -> A reuses both the encoded slot and shared status/output scratch
    # on one stream, including the initial metadata upload, with one final fence.
    with torch.cuda.device(device), torch.cuda.stream(stream):
        for index in (0, 1, 0):
            plan, host = plans[index], hosts[index]
            decoded.fill_(0xA5)
            encoded[: host.numel()].copy_(host, non_blocking=True)
            plan.enqueue()
            observations.append(
                (decoded.clone(), plan.statuses.clone(), plan.actual_sizes.clone())
            )
        complete = torch.cuda.Event()
        complete.record(stream)
    complete.synchronize()
    for index, (output, statuses, sizes) in zip((0, 1, 0), observations):
        expected = bytearray([0xA5]) * decoded.numel()
        for frame, value in zip(batches[index], values[index]):
            expected[frame.output_offset : frame.output_offset + len(value)] = value
        assert statuses.tolist() == [0] * len(batches[index])
        assert sizes.tolist() == list(map(len, values[index]))
        assert plans[index].expected_sizes.tolist() == sizes.tolist()
        assert bytes(output.cpu().numpy()) == expected
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
