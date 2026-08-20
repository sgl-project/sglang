import torch
import triton
import triton.language as tl

_UINT32_MASK = 0xFFFFFFFF
def _rotl32_torch(x: torch.Tensor, r: int) -> torch.Tensor:
    """Rotate left with uint32 semantics using int64 tensors."""
    x = x & _UINT32_MASK
    return ((x << r) | (x >> (32 - r))) & _UINT32_MASK

def _fmix32_torch(h: torch.Tensor) -> torch.Tensor:
    """Torch implementation matching fmix32()."""
    h = (h ^ (h >> 16)) & _UINT32_MASK
    h = (h * 0x85EBCA6B) & _UINT32_MASK
    h = (h ^ (h >> 13)) & _UINT32_MASK
    h = (h * 0xC2B2AE35) & _UINT32_MASK
    h = (h ^ (h >> 16)) & _UINT32_MASK

    return h


def _murmur3_mix_torch(
    h: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Torch implementation matching murmur3_mix()."""
    k = (k * 0xCC9E2D51) & _UINT32_MASK
    k = _rotl32_torch(k, 15)
    k = (k * 0x1B873593) & _UINT32_MASK
    h = (h ^ k) & _UINT32_MASK
    h = _rotl32_torch(h, 13)
    h = (h * 5 + 0xE6546B64) & _UINT32_MASK

    return h

def _murmur_hash32_torch(
    seed: torch.Tensor,
    positions: torch.Tensor,
    col_indices: torch.Tensor,
) -> torch.Tensor:
    """CPU implementation matching murmur_hash32_kernel bit-for-bit."""
    # Use int64 because uint32 arithmetic support is limited in PyTorch.
    # Masking after every operation preserves uint32 wraparound semantics.
    seed = seed.to(torch.int64)
    positions = positions.to(torch.int64)
    col_indices = col_indices.to(torch.int64)

    n = seed.shape[0]
    m = col_indices.shape[0]

    h = torch.zeros((n, m), dtype=torch.int64, device=seed.device)

    # Process seed_low
    seed_low = (seed & _UINT32_MASK).view(n, 1)
    h = _murmur3_mix_torch(h, seed_low)

    # Process seed_high
    seed_high = ((seed >> 32) & _UINT32_MASK).view(n, 1)
    h = _murmur3_mix_torch(h, seed_high)

    # position block
    pos = (positions & _UINT32_MASK).view(n, 1)
    h = _murmur3_mix_torch(h, pos)

    # column block
    col = (col_indices & _UINT32_MASK).view(1, m)
    h = _murmur3_mix_torch(h, col)


    h = (h ^ 16) & _UINT32_MASK
    h = _fmix32_torch(h)

    return h.to(torch.uint32)

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
    if seed.device.type == "cpu":
        return _murmur_hash32_torch(seed, positions, col_indices)
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
