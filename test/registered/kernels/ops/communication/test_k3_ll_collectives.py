"""Kimi-K3's gfx950 TP8 LL all-gather read must match torch.distributed and attn_res_hip."""

import unittest

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase, run_distributed_test

# backend-specific: the LL kernels are gfx950 wave64 HIP over aiter's IPC buffers.
register_amd_ci(est_time=45, suite="base-c-test-large-8-gpu-amd-mi35x")

WORLD = 8
# Kimi-K3: hidden 7168, latent 3584, 8 bank rows read and one written (9 slots).
H, L, NB, NVB, EPS = 7168, 3584, 9, 8, 1e-5
# the column tail's largest batch, which the read finishes from its split-K partials
ROWS = 64


def _beyond_ulp(got, want) -> int:
    want = want.float()
    return ((got.float() - want).abs() > want.abs() * 2**-7 + 1e-3).sum().item()


def _check_gather_tail_read(comm, rank):
    from sglang.kernels.ops.attention import k3_attn_res_read_hip as hip
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
    from sglang.kernels.ops.gemm.kimi_k3_latent_tail_hip import k3_latent_tail_partials

    g = torch.Generator(device="cuda").manual_seed(2)
    up = (torch.randn(H, L, generator=g, device="cuda") * 0.02).bfloat16()
    norm_w = (torch.rand(L, generator=g, device="cuda") + 0.5).bfloat16()
    cw = torch.randn(H, generator=g, device="cuda") * 0.05
    ow = (torch.rand(H, generator=g, device="cuda") + 0.5).bfloat16().float()
    latent = (torch.randn(ROWS, L, generator=g, device="cuda") * 3).bfloat16()
    shared = torch.randn(ROWS, H, generator=g, device="cuda").bfloat16()
    prefix = (torch.randn(ROWS, H, generator=g, device="cuda") * 4).bfloat16()
    bank = (torch.randn(ROWS, NB, H, generator=g, device="cuda") * 4).bfloat16()
    cols = slice(rank * H // WORLD, (rank + 1) * H // WORLD)

    part, sq = k3_latent_tail_partials(latent, norm_w, up[cols])
    proj = (part.sum(0) * torch.rsqrt(sq.sum(0) / L + EPS)[:, None]).bfloat16()
    mine = proj.float() + shared[:, cols].float() + prefix[:, cols].float()
    slices = [torch.empty_like(mine, dtype=torch.bfloat16) for _ in range(WORLD)]
    dist.all_gather(slices, mine.bfloat16())
    want_stream = torch.cat(slices, dim=1)
    want = torch.empty_like(want_stream)
    attn_res_hip(want_stream, bank, cw, ow, want, NVB, EPS, EPS)

    got_bank = bank.clone()
    out, stream = hip.gather_tail_attn_res_read(
        part, sq, shared, prefix, EPS, got_bank, NVB, cw, ow, EPS, EPS, comm,
        write_bank=True,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert _beyond_ulp(stream, want_stream) <= ROWS * 8, "gathered stream"
    assert torch.equal(got_bank[:, NVB], stream), "bank snapshot of the stream"
    assert _beyond_ulp(out, want) <= ROWS * 16, "attention-residual read"


def _check_rank(rank):
    from sglang.kernels.ops.communication.k3_ll_comm_hip import create_ll_comm

    comm = create_ll_comm(dist.new_group(backend="gloo"), torch.device(f"cuda:{rank}"))
    assert comm is not None, "create_ll_comm returned None"
    with torch.inference_mode():
        _check_gather_tail_read(comm, rank)


@unittest.skipUnless(
    is_gfx95_supported() and torch.cuda.device_count() >= WORLD, "8 gfx950 GPUs"
)
class TestK3LLCollectives(CustomTestCase):
    def test_gather_tail_read_matches_torch_distributed(self):
        run_distributed_test(_check_rank, world_size=WORLD)


if __name__ == "__main__":
    unittest.main()
