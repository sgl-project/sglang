"""CPU test for the RoPE padding clamp used by speculative verify."""

import pathlib
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(
    0,
    str(
        _ROOT
        / "python"
        / "sglang"
        / "srt"
        / "hardware_backend"
        / "npu"
        / "dsv4"
    ),
)

from dsv4_rope import Dsv4NpuRoPE  # noqa: E402


def test_negative_padding_position_clamps_to_zero():
    freqs = torch.randn(8, 4, dtype=torch.complex64)
    rope = Dsv4NpuRoPE(freqs)
    cos0, sin0 = rope.get_cos_sin(
        torch.tensor([0], dtype=torch.long), torch.float32, allow_build=True
    )
    cos_pad, sin_pad = rope.get_cos_sin(
        torch.tensor([-1], dtype=torch.long), torch.float32, allow_build=True
    )
    assert torch.equal(cos_pad[0], cos0[0])
    assert torch.equal(sin_pad[0], sin0[0])
