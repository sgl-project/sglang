"""Benchmark: QSA sparse decode attention on gfx950.

``fused_flydsl`` (``aiter.ops.flydsl.qsa_sparse_decode``) reads the selected KV
straight from the pool and splits each row across CTAs. ``gather_triton`` is the ROCm path it replaces in
``QwenSparseAttnBackend._forward_paged_attention``: a Triton gather into packed
scratch, then ``sparse_gqa_packed_decode_triton`` with one program per row and
KV head.

Cold cache: each captured call reads a different random slice of a pool that is
several times larger than MI355X's 256 MB last-level cache.
"""

import itertools

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="jit-kernel-benchmark", runner_config="amd")

DIM = 256
TOPK = 2051  # Qwen3.8-Flash-Next: 512 compressed blocks x ratio 4 + 3
MAX_CTX = 8192
POOL_SLOTS = 1 << 21  # 2 GiB of bf16 K+V at Hkv=2
# Touch at least 4x the last-level cache across the rotated calls.
COLD_BYTES = 4 << 30


def _fused_flydsl(q, k, v, req_to_token, row_req, indices, seq_lens, splits, ws):
    from aiter.ops.flydsl.qsa_sparse_decode import flydsl_qsa_sparse_decode

    return flydsl_qsa_sparse_decode(
        q,
        k,
        v,
        req_to_token,
        row_req,
        indices,
        seq_lens,
        DIM**-0.5,
        num_splits=splits,
        workspace=ws,
    )


def _gather_triton(q, k, v, req_to_token, row_req, indices, seq_lens, splits, ws):
    from sglang.srt.layers.attention.qsa.sparse_attn import (
        qwen_sparse_fa2_cu_seqlens_triton,
        qwen_sparse_kv_extraction_compact_triton,
        sparse_gqa_packed_decode_triton,
    )

    batch, topk = indices.shape
    counts = torch.empty(batch, dtype=torch.int32, device=q.device)
    cu_k = torch.empty(batch + 1, dtype=torch.int32, device=q.device)
    qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, counts, cu_k, batch, topk)
    packed_k = torch.empty(
        batch * topk, k.shape[1], DIM, dtype=q.dtype, device=q.device
    )
    packed_v = torch.empty_like(packed_k)
    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        req_to_token,
        row_req,
        indices,
        seq_lens,
        cu_k,
        packed_k,
        packed_v,
        batch,
        topk,
    )
    relative = torch.arange(topk, dtype=torch.int32, device=q.device).expand(batch, -1)
    relative = relative.masked_fill(relative >= counts[:, None], -1).contiguous()
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
    return sparse_gqa_packed_decode_triton(
        q, packed_k, packed_v, relative, cu_q, cu_k, counts, DIM**-0.5
    )


FN_MAP = {"fused_flydsl": _fused_flydsl, "gather_triton": _gather_triton}


@marker.parametrize(
    "num_kv_heads,group", [(2, 12), (1, 3)], [(2, 12)]
)  # TP1 and the cookbook's TP8 per-rank shape
@marker.parametrize("batch", [1, 4, 16, 64, 128], [1, 64])
@marker.benchmark("impl", ["fused_flydsl", "gather_triton"])
def benchmark(num_kv_heads: int, group: int, batch: int, impl: str):
    from aiter.ops.flydsl.qsa_sparse_decode import recommended_num_splits

    gen = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(batch, num_kv_heads * group, DIM, generator=gen, device="cuda")
    q = q.bfloat16()
    k = torch.randn(POOL_SLOTS, num_kv_heads, DIM, generator=gen, device="cuda")
    k = k.bfloat16()
    v = torch.randn_like(k)
    touched = batch * num_kv_heads * TOPK * DIM * 2 * 2
    rotations = max(2, min(256, -(-COLD_BYTES // touched)))
    req_to_token = torch.randint(
        1,
        POOL_SLOTS,
        (rotations * batch, MAX_CTX),
        generator=gen,
        device="cuda",
        dtype=torch.int32,
    )
    # Each rotation maps the same positions through different requests' slots.
    row_reqs = [
        torch.arange(r * batch, (r + 1) * batch, device="cuda")
        for r in range(rotations)
    ]
    seq_lens = torch.full((batch,), MAX_CTX, dtype=torch.int32, device="cuda")
    indices = torch.stack(
        [
            torch.randperm(MAX_CTX, generator=gen, device="cuda")[:TOPK].sort().values
            for _ in range(batch)
        ]
    ).int()
    splits = recommended_num_splits(batch, num_kv_heads, TOPK, q.device)
    partials = batch * num_kv_heads * group * splits
    workspace = (
        torch.empty(partials, device="cuda"),
        torch.empty(partials, device="cuda"),
        torch.empty(partials * DIM, device="cuda"),
    )
    fn = FN_MAP[impl]
    rotation = itertools.cycle(row_reqs)

    def call():
        row_req = next(rotation)
        return fn(q, k, v, req_to_token, row_req, indices, seq_lens, splits, workspace)

    return marker.do_bench(
        call,
        graph_clone_args=None,
        memory_args=None,
        memory_output=None,
        extra_memory_footprint=touched,
    )


if __name__ == "__main__":
    benchmark.run()
