# SPDX-License-Identifier: Apache-2.0
"""Keyed NH hashing of CUDA tensors, so cache keys copy 32 B per 4 KiB block to host.

Each 4 KiB block is reduced on the GPU by NH (the UMAC inner hash) under four
Toeplitz-shifted lanes of a per-process random key. For two distinct blocks a lane
collides with probability at most 2^-32, so a block collides with at most 2^-128.
The key never leaves the process, and cache keys never leave their rank (ranks
agree on hits with a collective), so no rank needs another's key.
"""

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None

NH_BLOCK_WORDS = 1024
NH_LANES = 4
_keys: dict[torch.device, torch.Tensor] = {}


def gpu_hash_available(tensor):
    return triton is not None and tensor.is_cuda


if triton is not None:

    @triton.jit
    def _nh_block_kernel(
        words_ptr,
        key_ptr,
        out_ptr,
        num_words,
        BLOCK_WORDS: tl.constexpr,
        LANES: tl.constexpr,
    ):
        block = tl.program_id(0).to(tl.int64)
        pair = tl.arange(0, BLOCK_WORDS // 2)
        index = block * BLOCK_WORDS + 2 * pair
        # The last block is zero-padded; the caller hashes the length beside it.
        m0 = tl.load(words_ptr + index, mask=index < num_words, other=0)
        m1 = tl.load(words_ptr + index + 1, mask=index + 1 < num_words, other=0)
        m0 = m0.to(tl.uint32, bitcast=True)
        m1 = m1.to(tl.uint32, bitcast=True)
        for lane in tl.static_range(LANES):
            k0 = tl.load(key_ptr + 4 * lane + 2 * pair).to(tl.uint32, bitcast=True)
            k1 = tl.load(key_ptr + 4 * lane + 2 * pair + 1).to(tl.uint32, bitcast=True)
            # uint32 adds and uint64 products and sums all wrap, as NH specifies.
            product = (m0 + k0).to(tl.uint64) * (m1 + k1).to(tl.uint64)
            total = tl.sum(product, axis=0)
            tl.store(out_ptr + block * LANES + lane, total.to(tl.int64, bitcast=True))


def _key(device):
    key = _keys.get(device)
    if key is None:
        words = NH_BLOCK_WORDS + 4 * (NH_LANES - 1)
        secret = bytearray(os.urandom(4 * words))
        key = torch.frombuffer(secret, dtype=torch.int32).to(device)
        _keys[device] = key
    return key


# data is a flat CUDA uint8 tensor; returns the per-block sums and the bytes past the last word.
def nh_block_sums(data):
    num_words = data.numel() // 4
    tail = data[4 * num_words :]
    tail = tail.cpu().numpy().tobytes() if tail.numel() else b""
    words = data[: 4 * num_words]
    if words.data_ptr() % 4:
        words = words.clone()
    words = words.view(torch.int32)
    num_blocks = triton.cdiv(num_words, NH_BLOCK_WORDS)
    sums = torch.empty((num_blocks, NH_LANES), dtype=torch.int64, device=data.device)
    if num_blocks:
        _nh_block_kernel[(num_blocks,)](
            words,
            _key(data.device),
            sums,
            num_words,
            BLOCK_WORDS=NH_BLOCK_WORDS,
            LANES=NH_LANES,
        )
    return sums.cpu().numpy().tobytes(), tail
