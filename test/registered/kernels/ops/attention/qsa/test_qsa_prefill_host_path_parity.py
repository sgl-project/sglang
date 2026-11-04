"""QSA prefill host path on the GPU: host-staged lengths and a host max_q give the
bytes the device-read path gives."""

import random
import sys

import pytest
import torch

from sglang.srt.layers.attention.qsa.sparse_attn import (
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import _last_rows_of_runs
from sglang.srt.utils.common import async_h2d
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.gpu


def _require_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("SM100+ CUDA GPU required")


def _poisoned_like(shape, dtype):
    """Leave a NaN block in the caching allocator for the kernel's empty_like to reuse."""
    block = torch.full(shape, float("nan"), dtype=dtype, device="cuda")
    del block


def _chunk_case(seed, num_requests, heads=4, head_dim=128, topk=64):
    rng = random.Random(seed)
    extend = [rng.randint(1, 700) for _ in range(num_requests)]
    prefix = [rng.choice([0, 16, 4096, 16384 - e]) for e in extend]
    seq = [p + e for p, e in zip(prefix, extend)]
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        sum(extend), heads, head_dim, device="cuda", dtype=torch.bfloat16, generator=g
    )
    k = torch.randn(
        sum(seq), 1, head_dim, device="cuda", dtype=torch.bfloat16, generator=g
    )
    v = torch.randn_like(k)
    rows = []
    for p, e in zip(prefix, extend):
        visible = torch.arange(e, device="cuda") + p + 1
        idx = (
            torch.arange(topk, device="cuda")[None, :] * 13
            + torch.arange(e, device="cuda")[:, None] * 7
        ) % visible[:, None]
        rows.append(idx)
    indices = torch.cat(rows).to(torch.int32)
    cu_q = torch.tensor([0] + extend, device="cuda", dtype=torch.int32).cumsum(0)
    return q, k, v, indices, cu_q.to(torch.int32), extend, seq, head_dim**-0.5


@pytest.mark.parametrize("seed,num_requests", [(0, 1), (1, 3), (2, 8), (3, 17)])
def test_host_lengths_and_max_q_are_bitwise_equal(seed, num_requests):
    _require_sm100()
    q, k, v, indices, cu_q, extend, seq, scale = _chunk_case(seed, num_requests)

    def run(lens_tensor, max_q):
        cu_k = torch.nn.functional.pad(lens_tensor.cumsum(0), (1, 0)).contiguous()
        _poisoned_like(q.shape, q.dtype)
        return sparse_gqa_fwd_interface_triton_ck(
            q, k, v, indices, cu_q, cu_k, lens_tensor, scale, max_q=max_q
        )

    device_path = run(torch.tensor(seq, dtype=torch.int32, device="cuda"), None)
    host_lens = async_h2d(seq, dtype=torch.int32, device=q.device)
    host_path = run(host_lens, max(extend))
    assert not device_path.isnan().any()
    assert torch.equal(host_path, device_path)
    if max(extend) > 1:
        # Positive control: a max_q one short leaves the longest request's last row unwritten.
        short = run(host_lens, max(extend) - 1)
        assert not torch.equal(short, device_path)


def test_host_output_size_rowmap_is_bitwise_equal():
    _require_sm100()
    rng = random.Random(4)
    for _ in range(50):
        lens = [rng.randint(0, 900) for _ in range(rng.randint(1, 40))]
        device_lens = async_h2d(lens, dtype=torch.long, device=torch.device("cuda"))
        arange = torch.arange(len(lens), device="cuda", dtype=torch.int32)
        assert torch.equal(
            torch.repeat_interleave(arange, device_lens, output_size=sum(lens)),
            torch.repeat_interleave(arange, device_lens),
        )


def test_host_seed_rows_match_device_nonzero():
    _require_sm100()
    rng = random.Random(5)
    for _ in range(200):
        bs = rng.randint(1, 16)
        extend = [rng.choice([0, 1, 3, 64, 700]) for _ in range(bs)]
        if not sum(extend):
            extend[0] = 1
        pool = (
            rng.sample(range(1, 512), bs)
            if rng.random() < 0.7
            else [rng.randint(0, 3) for _ in range(bs)]
        )
        num_rows = sum(extend) - rng.randint(0, sum(extend) - 1)
        row_to_req = torch.repeat_interleave(
            torch.arange(bs, device="cuda"), torch.tensor(extend, device="cuda")
        )
        rows = torch.tensor(pool, device="cuda")[row_to_req[:num_rows]]
        is_last = torch.ones_like(rows, dtype=torch.bool)
        if rows.numel() > 1:
            is_last[:-1] = rows[:-1] != rows[1:]
        host = _last_rows_of_runs(extend_lens=extend, pool_ids=pool, num_rows=num_rows)
        assert torch.equal(
            async_h2d(host, dtype=torch.long, device=rows.device),
            is_last.nonzero().flatten(),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
