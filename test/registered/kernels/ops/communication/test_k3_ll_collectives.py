"""Kimi-K3's gfx950 TP8 LL collectives against torch.distributed references."""

import unittest

import torch
import torch.distributed as dist

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase, run_distributed_test

# backend-specific: the LL kernels are gfx950 wave64 HIP over aiter's IPC buffers.
register_amd_ci(est_time=45, suite="stage-c-test-large-8-gpu-amd-mi35x")

WORLD = 8
# Kimi-K3: hidden 7168, latent 3584, up to 8 bank rows (9 slots).
H, L, NB, EPS = 7168, 3584, 9, 1e-5


def _beyond_ulp(got, want) -> int:
    want = want.float()
    return ((got.float() - want).abs() > want.abs() * 2**-7 + 1e-3).sum().item()


def _check_gather_tail_read(comm, rank, failures):
    """Each rank finishes its column slice from the column tail's partials; the
    gathered stream (and the bank row it may write) and the read match torch."""
    from sglang.kernels.ops.attention import k3_attn_res_read_hip as hip
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
    from sglang.kernels.ops.gemm.kimi_k3_latent_tail_hip import k3_latent_tail_partials

    g = torch.Generator(device="cuda").manual_seed(2)
    up = (torch.randn(H, L, generator=g, device="cuda") * 0.02).bfloat16()
    norm_w = (torch.rand(L, generator=g, device="cuda") + 0.5).bfloat16()
    cw = torch.randn(H, generator=g, device="cuda") * 0.05
    ow = (torch.rand(H, generator=g, device="cuda") + 0.5).bfloat16().float()
    cols = slice(rank * H // WORLD, (rank + 1) * H // WORLD)
    # Up to 64 rows the slice is finished from the split-K partials; above, it is
    # the bf16 GEMM of the RMSNorm'd latent (projected), up to 384 rows.
    cases = ((64, True, 8, True), (384, True, 8, True))
    for rows, with_prefix, nvb, write in cases:
        latent = (torch.randn(rows, L, generator=g, device="cuda") * 3).bfloat16()
        shared = torch.randn(rows, H, generator=g, device="cuda").bfloat16()
        prefix = (torch.randn(rows, H, generator=g, device="cuda") * 4).bfloat16()
        bank = (torch.randn(rows, NB, H, generator=g, device="cuda") * 4).bfloat16()
        if rows <= 64:
            part, sq = k3_latent_tail_partials(latent, norm_w, up[cols])
            proj = (part.sum(0) * torch.rsqrt(sq.sum(0) / L + EPS)[:, None]).bfloat16()
        else:
            normed = latent.float() * torch.rsqrt(
                latent.float().pow(2).mean(-1, keepdim=True) + EPS
            )
            normed = (normed * norm_w.float()).bfloat16()
            part, sq = normed @ up[cols].T, None
            proj = part
        pre = prefix if with_prefix else None
        mine = proj.float() + shared[:, cols].float()
        if with_prefix:
            mine = mine + prefix[:, cols].float()
        parts = [torch.empty_like(mine, dtype=torch.bfloat16) for _ in range(WORLD)]
        dist.all_gather(parts, mine.bfloat16())
        want_stream = torch.cat(parts, dim=1)
        want = torch.empty_like(want_stream)
        attn_res_hip(want_stream, bank, cw, ow, want, nvb, EPS, EPS)
        got_bank = bank.clone()
        out, stream = hip.gather_tail_attn_res_read(
            part, sq, shared, pre, EPS, got_bank, nvb, cw, ow, EPS, EPS, comm,
            write_bank=write,
        )  # fmt: skip
        stream_ok = _beyond_ulp(stream, want_stream) <= rows * 8
        bank_ok = not write or torch.equal(got_bank[:, nvb], stream)
        bad = _beyond_ulp(out, want)
        if not (stream_ok and bank_ok and bad <= rows * 16):
            failures.append(
                f"AG rows={rows} prefix={with_prefix} nvb={nvb} write={write}: "
                f"stream_ok={stream_ok} bank_ok={bank_ok} beyond_ulp={bad}"
            )


def _check_rank(rank):
    from sglang.kernels.ops.communication.k3_ll_comm_hip import create_ll_comm

    comm = create_ll_comm(dist.new_group(backend="gloo"), torch.device(f"cuda:{rank}"))
    assert comm is not None, "create_ll_comm returned None"
    failures = []
    with torch.inference_mode():
        _check_gather_tail_read(comm, rank, failures)
    torch.cuda.synchronize()
    assert not failures, failures


@unittest.skipUnless(
    is_gfx95_supported() and torch.cuda.device_count() >= WORLD, "8 gfx950 GPUs"
)
class TestK3LLCollectives(CustomTestCase):
    def test_fused_reads_match_torch_distributed(self):
        run_distributed_test(_check_rank, world_size=WORLD)


if __name__ == "__main__":
    unittest.main()
