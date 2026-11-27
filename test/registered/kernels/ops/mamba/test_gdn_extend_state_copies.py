"""The eager GDN extend's one-launch state copies against the index ops they replace:
the state-pool gather / write-back (gather_pool_states / scatter_pool_states, padding
rows included), the prefix-track state copies (_track_mamba_state_extend) and the
conv-window track. They are pure copies, so every byte must equal the index ops'."""

import sys
import types

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or torch.cuda.get_device_capability()[0] < 10,
    reason="the eager extend takes the copy kernels on SM100+ CUDA only",
)

from sglang.kernels.ops.mamba import gdn_extend_state_copies as sc  # noqa: E402
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (  # noqa: E402
    MambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.kernels.state_pool_io import (  # noqa: E402
    gather_pool_states,
    scatter_pool_states,
)

DEV = "cuda"
HV, D = 64, 128  # 64 heads of a 128 x 128 state per slot
QKV, L = 12288, 3


def _pool(slots, seed, shape=(HV, D, D), dtype=torch.bfloat16):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn((slots, *shape), generator=g, device=DEV).to(dtype)


def _same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize(
    "indices",
    ([5, 9, 1, 30], [7, 0, 3, -1, 0], [0], [12, 2, 25, 4, 18, 6, 31]),
    ids=("live", "padding", "only-padding", "seven-rows"),
)
def test_gather_scatter_equal_index_ops(indices):
    pool = _pool(32, 1)
    idx = torch.tensor(indices, dtype=torch.int32, device=DEV)
    _, _, stock = gather_pool_states(pool, idx)
    assert sc.gather_pool_rows_ok(states=pool, indices=idx, dtype=pool.dtype)
    fused = sc.gather_pool_rows(states=pool, indices=idx)
    assert fused.shape == stock.shape and _same(fused, stock)

    upd = _pool(len(indices), 2)
    a, b = pool.clone(), pool.clone()
    safe, live, _ = gather_pool_states(a, idx)
    scatter_pool_states(a, safe, live, upd)
    assert sc.scatter_pool_rows_ok(states=b, indices=idx, updates=upd)
    sc.scatter_pool_rows(states=b, indices=idx, updates=upd)
    assert _same(a, b)


def test_gather_scatter_strided_pool():
    # Per-slot contiguous rows whose slot stride exceeds the row size.
    big = _pool(16, 3, shape=(2, HV, D, D))
    pool = big[:, 0]
    idx = torch.tensor([3, 0, 11], dtype=torch.int32, device=DEV)
    assert sc.gather_pool_rows_ok(states=pool, indices=idx, dtype=pool.dtype)
    fused = sc.gather_pool_rows(states=pool, indices=idx)
    assert _same(fused, gather_pool_states(pool, idx)[2])
    upd = _pool(3, 4)
    a, b = big.clone(), big.clone()
    scatter_pool_states(a[:, 0], *gather_pool_states(a[:, 0], idx)[:2], upd)
    sc.scatter_pool_rows(states=b[:, 0], indices=idx, updates=upd)
    assert _same(a, b), "the other half of each slot must stay untouched"


@pytest.mark.parametrize("n_h,n_final", ((3, 1), (0, 2), (4, 0), (0, 0)))
def test_track_state_rows_equal_index_ops(n_h, n_final):
    pool = _pool(40, 7)
    h = _pool(50, 8).unsqueeze(0)  # [1, checkpoints, ...] in the pool dtype
    gen = torch.Generator().manual_seed(n_h * 10 + n_final)
    slots = (torch.randperm(39, generator=gen) + 1)[: 2 * (n_h + n_final)]
    md = types.SimpleNamespace(
        has_mamba_track_mask=True,
        track_ssm_h_src=torch.randint(0, 50, (n_h,), generator=gen).to(DEV),
        track_ssm_h_dst=slots[:n_h].to(DEV),
        track_ssm_final_src=slots[n_h : n_h + n_final].to(DEV),
        track_ssm_final_dst=slots[n_h + n_final : n_h + 2 * n_final].to(DEV),
        track_ssm_recompute_dst=None,
    )
    a, b = pool.clone(), pool.clone()
    MambaAttnBackendBase._track_mamba_state_extend(None, None, h, a, md)
    assert sc.track_state_rows_ok(
        states=b, h=h, h_rows=n_h, snapshot=False, recompute_rows=0
    )
    sc.track_state_rows(
        states=b,
        h=h,
        h_src=md.track_ssm_h_src,
        h_dst=md.track_ssm_h_dst,
        final_src=md.track_ssm_final_src,
        final_dst=md.track_ssm_final_dst,
    )
    assert _same(a, b)


@pytest.mark.parametrize("dim_major", (True, False), ids=("dim-contig", "pos-contig"))
def test_track_conv_rows_equal_index_put(dim_major):
    g = torch.Generator(device=DEV).manual_seed(11)
    if dim_major:  # [slots, dim, L] view of a [slots, L, dim] buffer
        conv = torch.randn((24, L, QKV), generator=g, device=DEV)
        conv = conv.to(torch.bfloat16).transpose(1, 2)
    else:
        conv = torch.randn((24, QKV, L), generator=g, device=DEV).to(torch.bfloat16)
    proj = torch.randn((700, 20480), generator=g, device=DEV).to(torch.bfloat16)
    x = proj[:, :QKV]  # the raw pre-conv input is a column slice of the projection
    starts = torch.tensor([-2, 61, 637, 695])
    tokens = torch.minimum(
        (starts[:, None] + torch.arange(L)).clamp(min=0), torch.tensor(699)
    ).to(DEV)
    dst = torch.tensor([3, 17, 9, 22], device=DEV)
    a, b = conv.clone(), conv.clone()
    a[dst] = x[tokens].transpose(1, 2)
    assert sc.track_conv_rows_ok(conv=b, x=x, tokens=tokens)
    sc.track_conv_rows(conv=b, x=x, tokens=tokens, dst=dst)
    assert _same(a.contiguous(), b.contiguous())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
