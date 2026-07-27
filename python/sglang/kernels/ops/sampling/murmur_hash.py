import torch
import triton
import triton.language as tl


@triton.jit
def rotl32(x, r: tl.constexpr) -> tl.uint32:
    """
    rotate left 32-bit integer x by r bits
    e.g. x = 01110001, r = 2 -> 11000101
    """
    x = x.to(tl.uint64)
    return ((x << r) | (x >> (32 - r))) & 0xFFFFFFFF


@triton.jit
def fmix32(h: tl.uint32) -> tl.uint32:
    """
    final mix of 32-bit hash value for MurmurHash
    """
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h


@triton.jit
def murmur3_mix(h: tl.uint32, k: tl.uint32) -> tl.uint32:
    """
    Mixes a 32-bit key into the hash state.
    """
    c1: tl.uint32 = 0xCC9E2D51
    c2: tl.uint32 = 0x1B873593
    r1: tl.constexpr = 15
    r2: tl.constexpr = 13
    mm: tl.uint32 = 5
    nn: tl.uint32 = 0xE6546B64

    k = (k * c1) & 0xFFFFFFFF
    k = rotl32(k, r1)
    k = (k * c2) & 0xFFFFFFFF
    h ^= k
    h = rotl32(h, r2)
    h = (h * mm + nn) & 0xFFFFFFFF
    return h


@triton.jit
def murmur_hash32_kernel(
    seed_ptr,
    positions_ptr,
    col_indices_ptr,
    output_ptr,
    num_rows,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    MurmurHash 32-bit implementation for Triton.
    Reference:
    - https://medium.com/@thealonemusk/murmurhash-the-scrappy-algorithm-that-secretly-powers-half-the-internet-2d3f79b4509b
    - https://en.wikipedia.org/wiki/MurmurHash

    We treat 64-bit seed, 32-bit position, and 32-bit col_index as 4 4-byte blocks, and bit-blend them together.
    """
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_idx = pid_row
    col_offsets = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < num_cols

    # Load inputs
    seed = tl.load(seed_ptr + row_idx).to(tl.uint64)
    pos = tl.load(positions_ptr + row_idx).to(tl.uint32)
    col = tl.load(col_indices_ptr + col_offsets, mask=mask, other=0).to(tl.uint32)

    h: tl.uint32 = 0  # hash accumulator

    # Process seed_low
    k: tl.uint32 = (seed & 0xFFFFFFFF).to(tl.uint32)
    h = murmur3_mix(h, k)

    # Process seed_high
    k = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    h = murmur3_mix(h, k)

    # Process position block starting from seed32
    h = murmur3_mix(h, pos)

    # Process col block
    h = murmur3_mix(h, col)

    # Finalize (len=16 for seed + pos + col)
    h ^= 16
    h = fmix32(h)

    # Store result as uint32
    tl.store(output_ptr + row_idx * num_cols + col_offsets, h, mask=mask)


def murmur_hash32(seed, positions, col_indices):
    assert (
        seed.shape == positions.shape
    ), "Seed and positions must have the same shape (n,)"
    assert (
        len(seed.shape) == 1 and len(col_indices.shape) == 1
    ), f"Inputs must be 1D tensors {seed.shape=} {col_indices.shape=}"
    n = seed.shape[0]
    m = col_indices.shape[0]
    device = seed.device
    hashed = torch.empty((n, m), dtype=torch.uint32, device=device)

    BLOCK_SIZE = 1024
    grid = (n, triton.cdiv(m, BLOCK_SIZE))
    murmur_hash32_kernel[grid](
        seed, positions, col_indices, hashed, n, m, BLOCK_SIZE=BLOCK_SIZE
    )
    return hashed
