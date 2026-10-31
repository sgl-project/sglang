"""Kimi-K3's gfx950 TP8 LL collectives against torch.distributed references."""

import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the LL kernels are gfx950 wave64 HIP over aiter's IPC buffers.
register_amd_ci(est_time=150, suite="stage-c-test-large-8-gpu-amd-mi35x")

WORLD = 8
# Kimi-K3: hidden 7168, latent 3584, up to 8 bank rows (9 slots).
H, L, NB, EPS = 7168, 3584, 9, 1e-5


def _open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _beyond_ulp(got, want) -> int:
    want = want.float()
    return ((got.float() - want).abs() > want.abs() * 2**-7 + 1e-3).sum().item()


def _gather(x):
    parts = [torch.empty_like(x) for _ in range(WORLD)]
    dist.all_gather(parts, x.contiguous())
    return parts


def _read_reference(stream, bank, nvb, cw, ow):
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip

    out = torch.empty_like(stream)
    attn_res_hip(stream, bank, cw, ow, out, nvb, EPS, EPS)
    return out


def _check_all_reduce_read(comm, rank, failures):
    """bf16(residual + bf16(rank-order sum of the partials)), then the FFN read, on
    the one-shot kernel (4 rows); every rank agrees."""
    from sglang.kernels.ops.attention import k3_attn_res_read_hip as hip

    g = torch.Generator(device="cuda").manual_seed(1)
    gr = torch.Generator(device="cuda").manual_seed(100 + rank)
    cw = torch.randn(H, generator=g, device="cuda") * 0.05
    ow = (torch.rand(H, generator=g, device="cuda") + 0.5).bfloat16().float()
    for rows in (4,):
        residual = (torch.randn(rows, H, generator=g, device="cuda") * 4).bfloat16()
        bank = (torch.randn(rows, NB, H, generator=g, device="cuda") * 4).bfloat16()
        partial = torch.randn(rows, H, generator=gr, device="cuda").bfloat16()
        delta = torch.stack(_gather(partial)).float().sum(0).bfloat16()
        for with_residual in (True, False):
            res = residual if with_residual else None
            want_stream = (
                (res.float() + delta.float()).bfloat16() if with_residual else delta
            )
            for nvb in (1, 3, 8):
                want = _read_reference(want_stream, bank, nvb, cw, ow)
                if not hip.all_reduce_covered(partial, res, bank, nvb):
                    failures.append(f"AR rows={rows}: not covered")
                    continue
                for _ in range(3):  # the flags advance across launches
                    out, stream = hip.all_reduce_attn_res_read(
                        partial, res, bank, nvb, cw, ow, EPS, EPS, comm
                    )
                same = all(torch.equal(x, out) for x in _gather(out))
                bad = _beyond_ulp(out, want)
                if not (same and torch.equal(stream, want_stream) and bad <= rows * 4):
                    failures.append(
                        f"AR rows={rows} residual={with_residual} nvb={nvb}: "
                        f"ranks_equal={same} stream_exact="
                        f"{torch.equal(stream, want_stream)} beyond_ulp={bad}"
                    )


def _check_gather_tail_read(comm, rank, failures):
    """Each rank finishes its column slice from the column tail's partials; the
    gathered stream (and the bank row it may write) and the read match torch."""
    from sglang.kernels.ops.attention import k3_attn_res_read_hip as hip
    from sglang.kernels.ops.gemm.kimi_k3_latent_tail_hip import k3_latent_tail_partials

    g = torch.Generator(device="cuda").manual_seed(2)
    up = (torch.randn(H, L, generator=g, device="cuda") * 0.02).bfloat16()
    norm_w = (torch.rand(L, generator=g, device="cuda") + 0.5).bfloat16()
    cw = torch.randn(H, generator=g, device="cuda") * 0.05
    ow = (torch.rand(H, generator=g, device="cuda") + 0.5).bfloat16().float()
    cols = slice(rank * H // WORLD, (rank + 1) * H // WORLD)
    for rows in (1, 4, 64):
        latent = (torch.randn(rows, L, generator=g, device="cuda") * 3).bfloat16()
        shared = torch.randn(rows, H, generator=g, device="cuda").bfloat16()
        prefix = (torch.randn(rows, H, generator=g, device="cuda") * 4).bfloat16()
        bank = (torch.randn(rows, NB, H, generator=g, device="cuda") * 4).bfloat16()
        part, sq = k3_latent_tail_partials(latent, norm_w, up[cols])
        proj = (part.sum(0) * torch.rsqrt(sq.sum(0) / L + EPS)[:, None]).bfloat16()
        for with_prefix in (True, False):
            pre = prefix if with_prefix else None
            mine = proj.float() + shared[:, cols].float()
            if with_prefix:
                mine = mine + prefix[:, cols].float()
            want_stream = torch.cat(_gather(mine.bfloat16()), dim=1)
            for nvb, write in ((1, False), (4, True), (8, True)):
                want = _read_reference(want_stream, bank, nvb, cw, ow)
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


def _worker(rank, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=WORLD, device_id=torch.device(f"cuda:{rank}")
    )
    failures = []
    try:
        from sglang.kernels.ops.communication.k3_ll_comm_hip import create_ll_comm

        comm = create_ll_comm(
            dist.new_group(backend="gloo"), torch.device(f"cuda:{rank}")
        )
        if comm is None:
            failures.append("create_ll_comm returned None")
        else:
            with torch.inference_mode():
                _check_all_reduce_read(comm, rank, failures)
                _check_gather_tail_read(comm, rank, failures)
        torch.cuda.synchronize()
    finally:
        results[rank] = failures
        dist.destroy_process_group()


@unittest.skipUnless(
    is_gfx95_supported() and torch.cuda.device_count() >= WORLD, "8 gfx950 GPUs"
)
class TestK3LLCollectives(CustomTestCase):
    def test_fused_reads_match_torch_distributed(self):
        ctx = mp.get_context("spawn")
        with ctx.Manager() as manager:
            results = manager.dict()
            port = _open_port()
            procs = [
                ctx.Process(target=_worker, args=(r, port, results))
                for r in range(WORLD)
            ]
            for p in procs:
                p.start()
            for p in procs:
                p.join(timeout=900)
            self.assertTrue(
                all(p.exitcode == 0 for p in procs), "a rank crashed or hung"
            )
            failures = [
                f"rank {r}: {msg}"
                for r in range(WORLD)
                for msg in results.get(r, ["no result"])
            ]
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
