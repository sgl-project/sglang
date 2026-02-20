"""
Tests for the JIT segment_packbits kernel.

Correctness is validated by:
1. Known-answer tests against numpy/CPU reference for little-endian packing.
2. Single-segment and multi-segment cases.
3. JIT vs AOT cross-validation (when sgl_kernel is available).
"""

import pytest
import torch

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# CPU reference
# ---------------------------------------------------------------------------


def cpu_segment_packbits(bits: torch.Tensor, input_indptr: torch.Tensor, output_indptr: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation: pack bool bits into uint8 with little-endian bit order.
    Bit i within a byte is placed at position i % 8 (LSB first).
    """
    batch_size = input_indptr.numel() - 1
    total_out = output_indptr[-1].item()
    out = torch.zeros(total_out, dtype=torch.uint8)
    bits_cpu = bits.cpu().bool()
    for b in range(batch_size):
        seg_bits = bits_cpu[input_indptr[b].item() : input_indptr[b + 1].item()]
        out_start = output_indptr[b].item()
        for i, bit in enumerate(seg_bits):
            if bit:
                byte_idx = i // 8
                bit_idx = i % 8  # little-endian
                out[out_start + byte_idx] |= 1 << bit_idx
    return out


def make_inputs(segments, device=DEVICE):
    """
    Build tensors for segment_packbits from a list of bool lists (one per segment).

    Returns x, input_indptr, output_indptr, y (zeroed output).
    """
    input_lengths = [len(s) for s in segments]
    output_lengths = [(l + 7) // 8 for l in input_lengths]
    batch_size = len(segments)

    input_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    output_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    for i, (il, ol) in enumerate(zip(input_lengths, output_lengths)):
        input_indptr[i + 1] = input_indptr[i] + il
        output_indptr[i + 1] = output_indptr[i] + ol

    x = torch.zeros(input_indptr[-1].item(), dtype=torch.bool)
    offset = 0
    for seg in segments:
        for b in seg:
            x[offset] = b
            offset += 1

    y = torch.zeros(output_indptr[-1].item(), dtype=torch.uint8)

    return (
        x.to(device),
        input_indptr.to(device),
        output_indptr.to(device),
        y.to(device),
    )


# ---------------------------------------------------------------------------
# Known-answer tests
# ---------------------------------------------------------------------------


def test_single_byte_all_ones():
    """8 True bits → 0xFF in little-endian."""
    from sglang.jit_kernel.packbit import segment_packbits

    segments = [[True] * 8]
    x, input_indptr, output_indptr, y = make_inputs(segments)
    segment_packbits(x, input_indptr, output_indptr, y, batch_size=1)
    assert y[0].item() == 0xFF


def test_single_byte_all_zeros():
    """8 False bits → 0x00."""
    from sglang.jit_kernel.packbit import segment_packbits

    segments = [[False] * 8]
    x, input_indptr, output_indptr, y = make_inputs(segments)
    segment_packbits(x, input_indptr, output_indptr, y, batch_size=1)
    assert y[0].item() == 0x00


def test_little_endian_order():
    """[True, False, False, False, False, False, False, False] → 0x01 (LSB first)."""
    from sglang.jit_kernel.packbit import segment_packbits

    segments = [[True, False, False, False, False, False, False, False]]
    x, input_indptr, output_indptr, y = make_inputs(segments)
    segment_packbits(x, input_indptr, output_indptr, y, batch_size=1)
    assert y[0].item() == 0x01


def test_known_pattern():
    """[1,0,1,0,1,0,1,0] → 0x55 in little-endian."""
    from sglang.jit_kernel.packbit import segment_packbits

    bits = [True, False, True, False, True, False, True, False]
    segments = [bits]
    x, input_indptr, output_indptr, y = make_inputs(segments)
    segment_packbits(x, input_indptr, output_indptr, y, batch_size=1)
    assert y[0].item() == 0x55


def test_partial_byte():
    """5 bits → 1 output byte; unused bits should be 0."""
    from sglang.jit_kernel.packbit import segment_packbits

    segments = [[True, False, True, True, False]]
    x, input_indptr, output_indptr, y = make_inputs(segments)
    segment_packbits(x, input_indptr, output_indptr, y, batch_size=1)
    # little-endian: bit0=1, bit1=0, bit2=1, bit3=1, bit4=0 → 0b00001101 = 0x0D
    assert y[0].item() == 0x0D


@pytest.mark.parametrize("bs,seg_len", [(1, 64), (4, 32), (8, 128), (2, 7)])
def test_vs_cpu_reference(bs, seg_len):
    """JIT output matches CPU reference for random inputs."""
    import random

    from sglang.jit_kernel.packbit import segment_packbits

    random.seed(42)
    segments = [[random.random() > 0.5 for _ in range(seg_len)] for _ in range(bs)]
    x, input_indptr, output_indptr, y = make_inputs(segments)

    segment_packbits(x, input_indptr, output_indptr, y, batch_size=bs)

    expected = cpu_segment_packbits(x.cpu(), input_indptr.cpu(), output_indptr.cpu())
    assert torch.equal(y.cpu(), expected), f"mismatch for bs={bs} seg_len={seg_len}"


# ---------------------------------------------------------------------------
# JIT vs AOT cross-validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs,seg_len", [(1, 32), (4, 64), (2, 7)])
def test_vs_aot(bs, seg_len):
    try:
        from sgl_kernel import segment_packbits as segment_packbits_aot
    except ImportError:
        pytest.skip("sgl_kernel not available")

    import random

    from sglang.jit_kernel.packbit import segment_packbits as segment_packbits_jit

    random.seed(0)
    segments = [[random.random() > 0.5 for _ in range(seg_len)] for _ in range(bs)]
    x, input_indptr, output_indptr, y_jit = make_inputs(segments)
    y_aot = y_jit.clone()

    segment_packbits_jit(x, input_indptr, output_indptr, y_jit, batch_size=bs)
    segment_packbits_aot(x, input_indptr, output_indptr, y_aot, batch_size=bs)

    assert torch.equal(y_jit, y_aot), f"JIT vs AOT mismatch for bs={bs} seg_len={seg_len}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
