# SPDX-License-Identifier: Apache-2.0
"""Standard unkeyed BLAKE3-256 of a CUDA tensor's contiguous raw bytes.

Implements https://c2sp.org/BLAKE3, including short blocks, incomplete trees,
and the ROOT domain flag. No input-sized padding or CPU staging is needed.
"""

import torch
import triton
import triton.language as tl

_IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)


@triton.jit
def _rotr(x, n: tl.constexpr):
    return (x >> n) | (x << (32 - n))


@triton.jit
def _g(a, b, c, d, x, y):
    a = a + b + x
    d = _rotr(d ^ a, 16)
    c = c + d
    b = _rotr(b ^ c, 12)
    a = a + b + y
    d = _rotr(d ^ a, 8)
    c = c + d
    b = _rotr(b ^ c, 7)
    return a, b, c, d


@triton.jit
def _compress(h, m, counter, block_len, flags, IV: tl.constexpr):
    v0, v1, v2, v3, v4, v5, v6, v7 = h
    v8 = tl.full(h[0].shape, IV[0], tl.uint32)
    v9 = tl.full(h[0].shape, IV[1], tl.uint32)
    v10 = tl.full(h[0].shape, IV[2], tl.uint32)
    v11 = tl.full(h[0].shape, IV[3], tl.uint32)
    v12 = tl.broadcast_to(counter.to(tl.uint32), h[0].shape)
    v13 = tl.broadcast_to((counter.to(tl.uint64) >> 32).to(tl.uint32), h[0].shape)
    v14 = tl.broadcast_to(block_len.to(tl.uint32), h[0].shape)
    v15 = tl.broadcast_to(flags.to(tl.uint32), h[0].shape)
    # Keep the seven rounds as a loop to bound JIT time and generated code.
    for _ in range(7):
        v0, v4, v8, v12 = _g(v0, v4, v8, v12, m[0], m[1])
        v1, v5, v9, v13 = _g(v1, v5, v9, v13, m[2], m[3])
        v2, v6, v10, v14 = _g(v2, v6, v10, v14, m[4], m[5])
        v3, v7, v11, v15 = _g(v3, v7, v11, v15, m[6], m[7])
        v0, v5, v10, v15 = _g(v0, v5, v10, v15, m[8], m[9])
        v1, v6, v11, v12 = _g(v1, v6, v11, v12, m[10], m[11])
        v2, v7, v8, v13 = _g(v2, v7, v8, v13, m[12], m[13])
        v3, v4, v9, v14 = _g(v3, v4, v9, v14, m[14], m[15])
        m = (
            m[2],
            m[6],
            m[3],
            m[10],
            m[7],
            m[0],
            m[4],
            m[13],
            m[1],
            m[11],
            m[12],
            m[5],
            m[9],
            m[14],
            m[15],
            m[8],
        )
    return v0 ^ v8, v1 ^ v9, v2 ^ v10, v3 ^ v11, v4 ^ v12, v5 ^ v13, v6 ^ v14, v7 ^ v15


@triton.jit
def _initial_cv(SIZE: tl.constexpr, IV: tl.constexpr):
    return (
        tl.full((SIZE,), IV[0], tl.uint32),
        tl.full((SIZE,), IV[1], tl.uint32),
        tl.full((SIZE,), IV[2], tl.uint32),
        tl.full((SIZE,), IV[3], tl.uint32),
        tl.full((SIZE,), IV[4], tl.uint32),
        tl.full((SIZE,), IV[5], tl.uint32),
        tl.full((SIZE,), IV[6], tl.uint32),
        tl.full((SIZE,), IV[7], tl.uint32),
    )


@triton.jit
def _reduce_cv(h, count, GROUP: tl.constexpr, ROOT: tl.constexpr, IV: tl.constexpr):
    # Each group spans a power-of-two number of consecutive leaves. Carrying
    # an odd final node preserves BLAKE3's left-full, left-big tree topology.
    for _ in tl.static_range(GROUP.bit_length() - 1):
        l0, r0 = tl.split(tl.reshape(h[0], (h[0].shape[0] // 2, 2)))
        l1, r1 = tl.split(tl.reshape(h[1], (h[0].shape[0] // 2, 2)))
        l2, r2 = tl.split(tl.reshape(h[2], (h[0].shape[0] // 2, 2)))
        l3, r3 = tl.split(tl.reshape(h[3], (h[0].shape[0] // 2, 2)))
        l4, r4 = tl.split(tl.reshape(h[4], (h[0].shape[0] // 2, 2)))
        l5, r5 = tl.split(tl.reshape(h[5], (h[0].shape[0] // 2, 2)))
        l6, r6 = tl.split(tl.reshape(h[6], (h[0].shape[0] // 2, 2)))
        l7, r7 = tl.split(tl.reshape(h[7], (h[0].shape[0] // 2, 2)))
        flags = tl.full((), 4, tl.uint32)
        if ROOT:
            flags = flags | tl.where(count == 2, 8, 0).to(tl.uint32)
        parent = _compress(
            _initial_cv(h[0].shape[0] // 2, IV),
            (l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7),
            tl.full((), 0, tl.uint64),
            tl.full((), 64, tl.uint32),
            flags,
            IV,
        )
        paired = 2 * tl.arange(0, h[0].shape[0] // 2) + 1 < count
        h = (
            tl.where(paired, parent[0], l0),
            tl.where(paired, parent[1], l1),
            tl.where(paired, parent[2], l2),
            tl.where(paired, parent[3], l3),
            tl.where(paired, parent[4], l4),
            tl.where(paired, parent[5], l5),
            tl.where(paired, parent[6], l6),
            tl.where(paired, parent[7], l7),
        )
        count = (count + 1) // 2
    return h


@triton.jit
def _load_word(data, offset, nbytes, ALIGNED: tl.constexpr):
    if ALIGNED:
        return tl.load(
            data.to(tl.pointer_type(tl.uint32)) + offset // 4,
            mask=offset < nbytes,
            other=0,
        )
    else:
        b0 = tl.load(data + offset, mask=offset < nbytes, other=0).to(tl.uint32)
        b1 = tl.load(data + offset + 1, mask=offset + 1 < nbytes, other=0).to(tl.uint32)
        b2 = tl.load(data + offset + 2, mask=offset + 2 < nbytes, other=0).to(tl.uint32)
        b3 = tl.load(data + offset + 3, mask=offset + 3 < nbytes, other=0).to(tl.uint32)
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)


@triton.jit(do_not_specialize=["nbytes"])
def _chunk_kernel(
    data,
    output,
    nbytes,
    GROUP: tl.constexpr,
    ROOT: tl.constexpr,
    ALIGNED: tl.constexpr,
    IV: tl.constexpr,
):
    nbytes = nbytes.to(tl.int64)
    chunk = tl.program_id(0).to(tl.int64) * GROUP + tl.arange(0, GROUP)
    n_chunks = tl.maximum(tl.cdiv(nbytes, 1024), 1)
    count = tl.minimum(n_chunks - tl.program_id(0) * GROUP, GROUP)
    h = _initial_cv(GROUP, IV)
    n_blocks = tl.minimum(tl.maximum(tl.cdiv(nbytes, 64), 1), 16).to(tl.int32)
    for block in range(n_blocks):
        offset = chunk * 1024 + block * 64
        m = (
            _load_word(data, offset + 0, nbytes, ALIGNED),
            _load_word(data, offset + 4, nbytes, ALIGNED),
            _load_word(data, offset + 8, nbytes, ALIGNED),
            _load_word(data, offset + 12, nbytes, ALIGNED),
            _load_word(data, offset + 16, nbytes, ALIGNED),
            _load_word(data, offset + 20, nbytes, ALIGNED),
            _load_word(data, offset + 24, nbytes, ALIGNED),
            _load_word(data, offset + 28, nbytes, ALIGNED),
            _load_word(data, offset + 32, nbytes, ALIGNED),
            _load_word(data, offset + 36, nbytes, ALIGNED),
            _load_word(data, offset + 40, nbytes, ALIGNED),
            _load_word(data, offset + 44, nbytes, ALIGNED),
            _load_word(data, offset + 48, nbytes, ALIGNED),
            _load_word(data, offset + 52, nbytes, ALIGNED),
            _load_word(data, offset + 56, nbytes, ALIGNED),
            _load_word(data, offset + 60, nbytes, ALIGNED),
        )
        flags = tl.where(block == 0, 1, 0) | tl.where(
            (block == 15) | (offset + 64 >= nbytes), 2, 0
        )
        if ROOT:
            flags = flags | tl.where((n_chunks == 1) & (block == n_blocks - 1), 8, 0)
        block_len = tl.minimum(tl.maximum(nbytes - offset, 0), 64)
        next_h = _compress(h, m, chunk, block_len, flags, IV)
        active = (chunk < n_chunks) & ((offset < nbytes) | (nbytes == 0))
        h = (
            tl.where(active, next_h[0], h[0]),
            tl.where(active, next_h[1], h[1]),
            tl.where(active, next_h[2], h[2]),
            tl.where(active, next_h[3], h[3]),
            tl.where(active, next_h[4], h[4]),
            tl.where(active, next_h[5], h[5]),
            tl.where(active, next_h[6], h[6]),
            tl.where(active, next_h[7], h[7]),
        )
    h = _reduce_cv(h, count, GROUP, ROOT, IV)
    for i in tl.static_range(8):
        tl.store(output + tl.program_id(0) * 8 + i, tl.sum(h[i], 0))


@triton.jit(do_not_specialize=["n_nodes"])
def _parent_kernel(
    data, output, n_nodes, GROUP: tl.constexpr, ROOT: tl.constexpr, IV: tl.constexpr
):
    node = tl.program_id(0).to(tl.int64) * GROUP + tl.arange(0, GROUP)
    count = tl.minimum(n_nodes - tl.program_id(0) * GROUP, GROUP)
    h = (
        tl.load(data + node * 8 + 0, node < n_nodes, other=0),
        tl.load(data + node * 8 + 1, node < n_nodes, other=0),
        tl.load(data + node * 8 + 2, node < n_nodes, other=0),
        tl.load(data + node * 8 + 3, node < n_nodes, other=0),
        tl.load(data + node * 8 + 4, node < n_nodes, other=0),
        tl.load(data + node * 8 + 5, node < n_nodes, other=0),
        tl.load(data + node * 8 + 6, node < n_nodes, other=0),
        tl.load(data + node * 8 + 7, node < n_nodes, other=0),
    )
    h = _reduce_cv(h, count, GROUP, ROOT, IV)
    for i in tl.static_range(8):
        tl.store(output + tl.program_id(0) * 8 + i, tl.sum(h[i], 0))


def _blake3_device(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda or tensor.layout != torch.strided:
        raise ValueError("BLAKE3 requires a dense CUDA tensor")
    with torch.cuda.device(tensor.device):
        data = (
            tensor.detach()
            .resolve_conj()
            .resolve_neg()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
        )
        n_chunks = max(triton.cdiv(data.numel(), 1024), 1)
        group = min(triton.next_power_of_2(n_chunks), 128)
        n_nodes = triton.cdiv(n_chunks, group)
        nodes = torch.empty((n_nodes, 8), dtype=torch.uint32, device=data.device)
        _chunk_kernel[(n_nodes,)](
            data,
            nodes,
            data.numel(),
            GROUP=group,
            ROOT=n_nodes == 1,
            ALIGNED=data.data_ptr() % 4 == 0 and data.numel() % 4 == 0,
            IV=_IV,
            num_warps=4,
        )
        while n_nodes > 1:
            group = min(triton.next_power_of_2(n_nodes), 128)
            next_count = triton.cdiv(n_nodes, group)
            next_nodes = torch.empty(
                (next_count, 8), dtype=torch.uint32, device=data.device
            )
            _parent_kernel[(next_count,)](
                nodes,
                next_nodes,
                n_nodes,
                GROUP=group,
                ROOT=next_count == 1,
                IV=_IV,
                num_warps=4,
            )
            nodes = next_nodes
            n_nodes = next_count
        return nodes.view(torch.uint8).reshape(32)


def gpu_tensor_blake3(tensor: torch.Tensor) -> bytes:
    """Hash contiguous raw tensor bytes using standard unkeyed BLAKE3-256.

    Noncontiguous tensors are made contiguous on the GPU. Shape and dtype are
    deliberately not hashed; callers can include them in their cache key.
    Returning bytes synchronizes a 32-byte device-to-host copy.
    """
    return _blake3_device(tensor).cpu().numpy().tobytes()


def gpu_tensors_blake3(tensors: list[torch.Tensor]) -> list[bytes]:
    """Hash a same-device batch with one digest-only host synchronization.

    Inputs remain independent: concatenating their bytes would lose boundaries
    and allocate another feature-sized buffer. Work runs on the caller's current
    stream; callers must establish producer readiness before hashing.
    """
    if not tensors:
        return []
    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("BLAKE3 batch requires tensors on the same CUDA device")
    digests = torch.stack([_blake3_device(t) for t in tensors])
    host = digests.cpu().numpy()
    return [row.tobytes() for row in host]
