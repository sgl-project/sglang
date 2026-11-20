"""Shared oracle for the swizzled MXFP8 producer tests."""

import torch


def assert_matches_flashinfer_mxfp8(
    got: tuple[torch.Tensor, torch.Tensor], bf16: torch.Tensor
) -> None:
    """Payload and swizzled E8M0 scales, padding included, byte for byte."""
    import flashinfer

    q, s = flashinfer.mxfp8_quantize(bf16.contiguous(), True)
    assert torch.equal(got[0].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(got[1].view(torch.uint8), s.view(torch.uint8))
