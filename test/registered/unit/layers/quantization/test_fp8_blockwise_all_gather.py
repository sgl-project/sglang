import pytest
import torch
from sglang.srt.layers.quantization.fp8_blockwise_all_gather import (
    Fp8BlockwisePacket,
    all_gather_fp8_blockwise,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize("rows", [0, 1, 17])
@pytest.mark.parametrize("hidden", [256, 4096])
def test_one_collective_and_encoded_scale_layout(rows, hidden):
    packets = [
        Fp8BlockwisePacket.allocate(rows, hidden, torch.float8_e4m3fnuz, "cpu")
        for _ in range(4)
    ]
    for rank, packet in enumerate(packets):
        packet.quantized.view(torch.uint8).fill_(rank + 1)
        packet.scales.copy_(
            torch.arange(rows * (hidden // 128), dtype=torch.float32).view(rows, -1)
            if rows
            else torch.empty((0, hidden // 128))
        )
        packet.scales.add_(rank * 1000)
    calls = []

    def gather(out, inp):
        assert inp.dtype == out.dtype == torch.float32
        assert inp.is_contiguous() and out.is_contiguous()
        assert inp.data_ptr() == packets[0].data.data_ptr()
        out.copy_(torch.cat([p.data for p in packets]))
        calls.append(out)

    q, scale = all_gather_fp8_blockwise(packets[0], 4, gather)
    assert len(calls) == 1
    assert q.shape == (rows * 4, hidden)
    assert q.untyped_storage().data_ptr() == calls[0].untyped_storage().data_ptr()
    assert torch.equal(
        q.view(torch.uint8), torch.cat([p.quantized.view(torch.uint8) for p in packets])
    )
    expected_scale = torch.cat([p.scales for p in packets])
    torch.testing.assert_close(scale.view(hidden // 128, rows * 4).t(), expected_scale)


def test_packet_rejects_incompatible_shapes_and_dtype():
    with pytest.raises(ValueError):
        Fp8BlockwisePacket.allocate(2, 130, torch.float8_e4m3fnuz, "cpu")
    with pytest.raises(ValueError):
        Fp8BlockwisePacket.allocate(2, 256, torch.bfloat16, "cpu")


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
