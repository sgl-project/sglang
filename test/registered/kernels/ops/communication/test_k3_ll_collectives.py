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


def _check_front_gather(comm, rank, failures):
    """The column-parallel front's gather equals dist.all_gather of the slices; its
    fused MXFP8 latent, top-k and shared-expert SiTU equal aiter's quant-sort,
    kimi_k3_route_sort and situ_and_mul on the gathered values."""
    from aiter.fused_moe import moe_sorting
    from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort, mxfp4_moe_sort_fwd

    from sglang.kernels.ops.activation import situ_and_mul
    from sglang.kernels.ops.moe import k3_front_gather_hip as fg
    from sglang.kernels.ops.moe.kimi_k3_route_sort import route_sort

    g = torch.Generator(device="cuda").manual_seed(3 + rank)
    bias = (torch.randn(fg.NUM_EXPERTS, generator=g, device="cuda") * 0.01).bfloat16()
    for rows in (1, 16, 64):
        # the slices sit inside a wider front output, as in the model
        front = torch.randn(rows, 2096, generator=g, device="cuda").bfloat16()
        gate_up, slices = front[:, :1536], front[:, 1536:]
        gathered = _gather(slices)
        want_logits = torch.cat([x[:, :112] for x in gathered], dim=1)
        want_latent = torch.cat([x[:, 112:] for x in gathered], dim=1)
        got = fg.front_gather(
            slices, comm, quant=True, bias=bias, gate_up=gate_up,
            situ_beta=4.0, situ_linear_beta=25.0,
        )  # fmt: skip
        routes = route_sort(
            got.logits, bias, topk=16, block_m=32, model_dim=fg.LATENT_SIZE,
            moe_buf=None, routed_scaling_factor=1.0, renormalize=True,
        )  # fmt: skip
        sorted_ids, sorted_w, _, num_valid, _ = moe_sorting(
            routes.topk_ids, routes.topk_weights, fg.NUM_EXPERTS, fg.LATENT_SIZE,
            torch.bfloat16, 32,
        )  # fmt: skip
        want_a1, want_scale = fused_dynamic_mxfp8_quant_moe_sort(
            want_latent, sorted_ids=sorted_ids, num_valid_ids=num_valid,
            token_num=rows, topk=16, block_size=32, sorted_weights=sorted_w,
            num_experts_upper_bound=fg.NUM_EXPERTS,
        )  # fmt: skip
        got_scale = mxfp4_moe_sort_fwd(
            got.a1_scale, sorted_ids=sorted_ids, num_valid_ids=num_valid,
            token_num=rows, cols=fg.LATENT_SIZE,
        )  # fmt: skip
        # the scales of routed rows only; padding rows of expert blocks are unspecified
        sid = sorted_ids[: int(num_valid[0])].long()
        x = torch.nonzero((sid & 0xFFFFFF) < rows).flatten()[:, None]
        y = torch.arange(112, device="cuda")[None, :]
        idx = (x // 32 * 112) * 32 + (y // 8) * 256 + (y % 4) * 64 + (x % 16) * 4
        idx = idx + (y % 8) // 4 * 2 + (x % 32) // 16
        checks = dict(
            logits=torch.equal(got.logits, want_logits),
            latent=torch.equal(got.latent, want_latent),
            a1=torch.equal(got.a1.view(torch.uint8), want_a1.view(torch.uint8)[:rows]),
            scales=torch.equal(
                got_scale.view(torch.uint8).flatten()[idx],
                want_scale.view(torch.uint8).flatten()[idx],
            ),
            topk=torch.equal(got.topk_ids, routes.topk_ids)
            and torch.equal(got.topk_weights, routes.topk_weights),
            situ=torch.equal(got.situ, situ_and_mul(gate_up, None, 4.0, 25.0)),
        )
        if not all(checks.values()):
            failures.append(f"front gather rows={rows}: {checks}")


def _check_bank_partial_reads(comm, rank, failures):
    """Reads that take the bank partials the route sort's spare workgroups compute
    give the reads over the bank rows: the same stream, the output within rounding."""
    from sglang.kernels.ops.attention import k3_attn_res_read_hip as hip
    from sglang.kernels.ops.gemm.kimi_k3_latent_tail_hip import k3_latent_tail_partials
    from sglang.kernels.ops.moe import kimi_k3_route_sort as rs

    g = torch.Generator(device="cuda").manual_seed(4)
    gr = torch.Generator(device="cuda").manual_seed(200 + rank)
    up = (torch.randn(H, L, generator=g, device="cuda") * 0.02).bfloat16()
    norm_w = (torch.rand(L, generator=g, device="cuda") + 0.5).bfloat16()
    cw_attn = torch.randn(H, generator=g, device="cuda") * 0.05
    cw_ffn = torch.randn(H, generator=g, device="cuda") * 0.05
    ow = (torch.rand(H, generator=g, device="cuda") + 0.5).bfloat16().float()
    bias = (torch.randn(896, generator=g, device="cuda") * 0.1).float()
    cols = slice(rank * H // WORLD, (rank + 1) * H // WORLD)
    for rows in (1, 4, 16):
        latent = (torch.randn(rows, L, generator=g, device="cuda") * 3).bfloat16()
        shared = torch.randn(rows, H, generator=g, device="cuda").bfloat16()
        prefix = (torch.randn(rows, H, generator=g, device="cuda") * 4).bfloat16()
        bank = (torch.randn(rows, NB, H, generator=g, device="cuda") * 4).bfloat16()
        logits = torch.randn(rows, 896, generator=g, device="cuda").bfloat16()
        attn_partial = torch.randn(rows, H, generator=gr, device="cuda").bfloat16()
        part, sq = k3_latent_tail_partials(latent, norm_w, up[cols])
        for nvb in (2, 5, 8):
            rs.route_sort(
                logits, bias, topk=16, block_m=32, model_dim=L, moe_buf=None,
                routed_scaling_factor=1.0, renormalize=True,
                partials=rs.BankPartials(
                    bank=bank, nvb=nvb, score_weights=(cw_attn, cw_ffn),
                    score_eps=EPS, acc=comm.bank_acc, ms=comm.bank_ms,
                ),
            )  # fmt: skip
            attn_side = (comm.bank_acc[0], comm.bank_ms[0])
            ffn_side = (comm.bank_acc[1], comm.bank_ms[1])
            pairs = {}
            for side in ("bank", "partial"):
                pairs[side] = hip.gather_tail_attn_res_read(
                    part, sq, shared, prefix, EPS, bank.clone(), nvb, cw_attn, ow,
                    EPS, EPS, comm, bank_partial=attn_side if side == "partial" else None,
                )  # fmt: skip
            if rows in hip.AR_READ_ROWS:
                for side in ("ar_bank", "ar_partial"):
                    pairs[side] = hip.all_reduce_attn_res_read(
                        attn_partial, prefix, bank, nvb, cw_ffn, ow, EPS, EPS, comm,
                        bank_partial=ffn_side if side == "ar_partial" else None,
                    )  # fmt: skip
            for read, against in (("partial", "bank"), ("ar_partial", "ar_bank")):
                if read not in pairs:
                    continue
                (out, stream), (want, want_stream) = pairs[read], pairs[against]
                bad = _beyond_ulp(out, want)
                if not (torch.equal(stream, want_stream) and bad <= rows * 8):
                    failures.append(
                        f"{read} rows={rows} nvb={nvb}: stream_exact="
                        f"{torch.equal(stream, want_stream)} beyond_ulp={bad}"
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
                _check_front_gather(comm, rank, failures)
                _check_bank_partial_reads(comm, rank, failures)
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
