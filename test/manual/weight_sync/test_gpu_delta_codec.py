"""GPU conformance for the prebuilt nvCOMP ABI; no extension compilation.

Manual-only: requires Blackwell, the paired Miles image, nvCOMP 5.3,
zstandard, and python-snappy. Missing hardware or dependencies fail this suite.
Both codecs decode the same bytes. Tests intentionally never feed malformed
compressed streams to the GPU decoder.
"""

import sys

import pytest
import torch

from sglang.srt.weight_sync.gpu_delta_codec import DecodeFrame, NvcompDecoder


def _encode(codec, values):
    if codec == "zstd":
        import zstandard

        compress = zstandard.ZstdCompressor(level=1).compress
    else:
        import snappy

        compress = snappy.compress
    payload = bytearray()
    frames = []
    offset = 0
    for value in values:
        payload.extend(bytes((-len(payload)) % 16))
        encoded = compress(value)
        frames.append(DecodeFrame(len(payload), len(encoded), offset, len(value)))
        payload.extend(encoded)
        offset += len(value)
    return payload, frames


@pytest.mark.parametrize("codec", ["zstd", "snappy"])
def test_same_delta_decode_and_reuse(codec):
    device = torch.device("cuda", 0)
    decoder = NvcompDecoder(codec, device)
    # Partial final frame, many zero bytes, and valid nonzero XOR values.
    values = [bytes(1 << 20), bytes(range(256)) * 4096, bytes(range(253)) * 3]
    payload, frames = _encode(codec, values)
    host = torch.empty(len(payload), dtype=torch.uint8, pin_memory=True)
    host.numpy()[:] = memoryview(payload)
    stream = torch.cuda.Stream(device=device)
    encoded = torch.empty_like(host, device=device)
    decoded = torch.empty(sum(map(len, values)), dtype=torch.uint8, device=device)
    workspace = decoder.allocate_workspace([frames])
    plan = decoder.prepare(frames, encoded, decoded, workspace, stream)
    # Reuse the pointer plan and overwrite the encoded slot on the consumer
    # stream for each successive tensor.
    with torch.cuda.stream(stream):
        for _ in range(2):
            decoded.fill_(0xA5)
            encoded.copy_(host, non_blocking=True)
            plan.enqueue(stream)
            complete = torch.cuda.Event()
            complete.record(stream)
            complete.synchronize()
            assert plan.statuses.tolist() == [0] * len(frames)
            assert plan.actual_sizes.tolist() == list(map(len, values))
            assert bytes(decoded.cpu().numpy()) == b"".join(values)
    assert decoder.backend == ("hardware" if codec == "snappy" else "cuda")


def test_rejects_frame_range_before_decode():
    device = torch.device("cuda", 0)
    decoder = NvcompDecoder("zstd", device)
    encoded = torch.empty(256, dtype=torch.uint8, device=device)
    decoded = torch.empty(256, dtype=torch.uint8, device=device)
    stream = torch.cuda.Stream(device=device)
    good = [DecodeFrame(0, 32, 0, 128)]
    workspace = decoder.allocate_workspace([good])
    with pytest.raises(ValueError, match="outside input"):
        decoder.prepare(
            [DecodeFrame(240, 32, 0, 128)], encoded, decoded, workspace, stream
        )
    with pytest.raises(ValueError, match="Overlapping"):
        decoder.prepare(
            [DecodeFrame(0, 32, 250, 128)], encoded, decoded, workspace, stream
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
