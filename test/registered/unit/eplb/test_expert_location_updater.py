import torch

from sglang.srt.eplb.expert_location_updater import _get_p2p_transport_tensor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_e8m0_p2p_transport_uses_bit_preserving_uint8_view():
    raw = torch.tensor([0x00, 0x7F, 0x80, 0xFF], dtype=torch.uint8)
    tensor = raw.view(torch.float8_e8m0fnu)

    transport = _get_p2p_transport_tensor(tensor)

    assert transport.dtype == torch.uint8
    assert transport.data_ptr() == tensor.data_ptr()
    assert torch.equal(transport, raw)

    transport[1] = 0x42
    assert tensor.view(torch.uint8)[1].item() == 0x42


def test_supported_p2p_transport_dtype_is_unchanged():
    tensor = torch.arange(4, dtype=torch.float32)

    transport = _get_p2p_transport_tensor(tensor)

    assert transport is tensor
