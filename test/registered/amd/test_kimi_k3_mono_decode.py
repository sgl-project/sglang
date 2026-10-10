"""TP8 correctness tests for the Kimi-K3 mono launches (ROCm / FlyDSL).

For the MoE launch, two things are checked, and they answer different
questions:

* against AITER's ``fused_moe`` stack, that the kernel computes the MoE;
* that the two AITER expert layouts -- gate/up separated, which the kernel was
  written for, and gate/up interleaved, which SGLang's ``Mxfp4MoEMethod``
  produces -- give the same answer from the same logical weights.

For the whole-layer launch, the same MoE check plus the part the wider launch
adds: o_proj, its all-reduce and the MLP attention-residual seam. Those two
halves are checked separately, against the kernel's own seam output, so that a
rounding difference in the seam cannot flip a top-16 choice and turn a correct
MoE into a failure that reads like a MoE bug. See ``_worker_k2``.

Run directly with ``python test/registered/amd/test_kimi_k3_mono_decode.py``;
it needs 8 GPUs.
"""

import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-c-test-large-8-gpu-amd-mi35x")

TP_SIZE = 8
ROWS = 8
COS_MIN = 0.999
BF = torch.bfloat16
# Attention-residual bank: rows the launch is built for, and the live rows the
# layer test uses. Keeping them apart exercises the production bank stride.
MAX_BLOCKS = 8
NVB = 3


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _randn(gen, *shape, scale=1.0, dtype=BF):
    return (
        torch.randn(*shape, generator=gen, device=gen.device, dtype=torch.float32)
        * scale
    ).to(dtype)


def _cos(a, b):
    return float(
        F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0, eps=1e-12)
    )


def _all_reduce_fp32(t):
    """The ranks' BF16 partials summed in FP32, as the custom all-reduce does."""
    t = t.float()
    dist.all_reduce(t)
    return t.to(BF)


def _situ(g, u):
    g, u = g.float(), u.float()
    a = (2.0 * torch.tanh(g * 0.25)) * (1.0 + torch.tanh(g * 0.5))
    return (a * (25.0 * torch.tanh(u * 0.04))).to(BF)


def _expert_bytes(gen, device, *, gate_up_interleaved):
    """The routed experts in one of AITER's two A16W4 layouts.

    The logical bytes are drawn from a fixed generator, so the two layouts hold
    the same weights and must give the same MoE output. Scale bytes stay near
    E8M0 1.0: the exponent is raw with bias 127, and uniform bytes would reach
    2^128 and make the layer non-finite.
    """
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    from sglang.kernels.ops.moe.k3_mono_flydsl.stages.moe import LAT, RI, E

    def u8(*shape, lo=0, hi=256):
        return torch.randint(
            lo, hi, shape, generator=gen, device=device, dtype=torch.int32
        ).to(torch.uint8)

    gu = gate_up_interleaved
    return {
        "w13": shuffle_weight_a16w4(u8(E, 2 * RI, LAT // 2), 16, gu),
        "w13s": shuffle_scale_a16w4(
            u8(E, 2 * RI, LAT // 32, lo=118, hi=123).view(-1, LAT // 32), E, gu
        ),
        "w2": shuffle_weight_a16w4(u8(E, LAT, RI // 2), 16, False),
        "w2s": shuffle_scale_a16w4(
            u8(E, LAT, RI // 32, lo=118, hi=123).view(-1, RI // 32), E, False
        ),
    }


def _weights(rank, device, *, gate_up_interleaved):
    from sglang.kernels.ops.moe.k3_mono_flydsl.stages.moe import (
        HIDDEN,
        LAT,
        SI,
        UP_N,
        E,
    )

    # The router, the latent down and the latent up are replicated across
    # ranks; the shared and routed experts are TP shards.
    gc = torch.Generator(device=device).manual_seed(7)
    gr = torch.Generator(device=device).manual_seed(100 + rank)
    ge = torch.Generator(device=device).manual_seed(100 + rank)

    w_up = _randn(gc, HIDDEN, LAT, scale=0.02)
    out = {
        "w_gate": _randn(gc, E, HIDDEN, scale=0.02),
        "bias": _randn(gc, E, scale=0.02, dtype=torch.float32),
        "w_ld": _randn(gc, LAT, HIDDEN, scale=0.02),
        "ln_w": (1 + _randn(gc, LAT, scale=0.1, dtype=torch.float32)).to(BF),
        "w_up_shard": w_up[rank * UP_N : (rank + 1) * UP_N].contiguous(),
        "w_sgu": _randn(gr, 2 * SI, HIDDEN, scale=0.02),
        "w_sd": _randn(gr, HIDDEN, SI, scale=0.03),
    }
    out.update(_expert_bytes(ge, device, gate_up_interleaved=gate_up_interleaved))
    return out


def _reference(x, w, rank):
    """The MoE as SGLang's unfused decode path computes it."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    from sglang.kernels.ops.moe.k3_mono_flydsl.stages.moe import SI, TOPK, UP_N

    sig = torch.sigmoid(torch.mm(x.float(), w["w_gate"].float().t()))
    ids = torch.topk(sig + w["bias"], TOPK, dim=-1, sorted=True).indices
    wts = sig.gather(-1, ids)
    wts = wts / wts.sum(-1, keepdim=True)
    gu = F.linear(x, w["w_sgu"])
    shared = F.linear(_situ(gu[:, :SI], gu[:, SI:]), w["w_sd"])
    # The kernel takes the experts as raw bytes, which is how SGLang holds
    # them; fused_moe dispatches on the dtype and on an is_shuffled marker
    # that a view drops, so tag the same storage for the reference only.
    fp4 = torch.float4_e2m1fn_x2
    w13, w2 = w["w13"].view(fp4), w["w2"].view(fp4)
    w13.is_shuffled = w2.is_shuffled = True
    y = fused_moe(
        F.linear(x, w["w_ld"]),
        w13,
        w2,
        wts.float(),
        ids.to(torch.int32),
        None,
        ActivationType.Situv2,
        QuantType.per_1x32,
        False,
        w["w13s"],
        w["w2s"],
        None,
        None,
        dtype=BF,
        gate_mode="separated",
        beta=4.0,
        linear_beta=25.0,
    )
    lat = _all_reduce_fp32(y).float()
    ln = lat * torch.rsqrt(lat.pow(2).mean(-1, keepdim=True) + 1e-5) * w["ln_w"]
    lo = rank * UP_N
    up = ln.to(BF).float() @ w["w_up_shard"].float().t()
    shared[:, lo : lo + UP_N] = (shared[:, lo : lo + UP_N].float() + up).to(BF)
    return _all_reduce_fp32(shared)


def _run_moe(mod, w, x, *, peers, epoch, rank, layer, guint):
    out = torch.empty_like(x)
    key = (
        mod.MoeBuild(tokens=x.size(0), guint=guint)
        if guint
        else mod.MoeBuild(tokens=x.size(0))
    )
    mod.moe(
        key,
        x=x,
        w_gate=w["w_gate"],
        bias=w["bias"],
        w_ld=w["w_ld"],
        w_sgu=w["w_sgu"],
        w_sd=w["w_sd"],
        w13=w["w13"],
        w13s=w["w13s"],
        w2=w["w2"],
        w2s=w["w2s"],
        ln_w=w["ln_w"],
        w_up=w["w_up_shard"],
        out=out,
        scratch=torch.zeros(mod.scratch_bytes(key), dtype=torch.uint8, device=x.device),
        queue=torch.zeros(mod.QUEUE_BYTES // 4, dtype=torch.int32, device=x.device),
        peers=peers.addresses,
        rank=rank,
        epoch=epoch,
        layer=layer,
    )
    return out


def _worker(rank, port, results, check_guint):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP_SIZE
    )
    peers = None
    try:
        from sglang.kernels.ops.moe.k3_mono_flydsl.common.peer_memory import PeerBuffer
        from sglang.kernels.ops.moe.k3_mono_flydsl.stages import moe

        peers = PeerBuffer(
            moe.peer_bytes(ROWS), dist.group.WORLD, rank, TP_SIZE, device
        )
        peers.bytes.zero_()
        epoch = torch.zeros(1, dtype=torch.int32, device=device)

        gx = torch.Generator(device=device).manual_seed(11)
        x = _randn(gx, ROWS, moe.HIDDEN)

        w = _weights(rank, device, gate_up_interleaved=False)
        epoch.add_(1)
        out = _run_moe(
            moe, w, x, peers=peers, epoch=epoch, rank=rank, layer=1, guint=False
        ).clone()
        torch.cuda.synchronize()

        gathered = [torch.empty_like(out.cpu()) for _ in range(TP_SIZE)]
        dist.all_gather(gathered, out.cpu().contiguous())
        result = {
            "finite": bool(torch.isfinite(out).all()),
            "nonzero": bool(torch.count_nonzero(out) > 0),
            "ranks_agree": all(torch.equal(gathered[0], g) for g in gathered[1:]),
            "cos_vs_reference": _cos(out, _reference(x, w, rank)),
            "error": "",
        }

        if check_guint:
            # Control: the same launch again, same weights. Any difference here
            # is the harness or the kernel's step state, not the layout.
            epoch.add_(1)
            out_again = _run_moe(
                moe, w, x, peers=peers, epoch=epoch, rank=rank, layer=1, guint=False
            ).clone()
            torch.cuda.synchronize()
            result["cos_rerun"] = _cos(out_again, out)

            # Same logical weights, SGLang's gate/up interleaved layout.
            w_gu = _weights(rank, device, gate_up_interleaved=True)
            epoch.add_(1)
            out_gu = _run_moe(
                moe, w_gu, x, peers=peers, epoch=epoch, rank=rank, layer=1, guint=True
            ).clone()
            torch.cuda.synchronize()
            result["cos_layouts"] = _cos(out_gu, out)
            result["layouts_equal"] = bool(torch.equal(out_gu, out))

        results[rank] = result
    except Exception as error:  # surfaced by the parent as a test failure
        import traceback

        results[rank] = {
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        }
    finally:
        if peers is not None:
            peers.close()
        dist.destroy_process_group()


def _rms(x, weight, eps):
    """RMSNorm over the last dimension, in FP32 like the kernel."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def _seam_reference(core, w_o, prefix_in, blocks, nvb, w, *, reset):
    """o_proj + its all-reduce + the MLP attention-residual seam, in torch.

    Mirrors AttnResidual's eager path: score every source (the ``nvb`` banked
    rows and the updated prefix) through the score RMSNorm and the 1-row
    projection, softmax over the sources, mix, then the output RMSNorm.
    Returns (updated prefix, the MoE input).
    """
    attn = _all_reduce_fp32(F.linear(core, w_o)).float()
    # A block-write layer restarts the prefix at the attention output.
    prefix = (attn if reset else prefix_in.float() + attn).to(BF)
    rows = torch.cat([blocks[:, :nvb, :], prefix.unsqueeze(1)], dim=1)
    scores = (_rms(rows, w["ares_nw"], w["eps"]) * w["ares_qk"].float()).sum(-1)
    probs = torch.softmax(scores, dim=-1)
    mixed = (probs.unsqueeze(-1) * rows.float()).sum(dim=1)
    return prefix, _rms(mixed, w["in_nw"], w["out_eps"]).to(BF)


def _layer_weights(rank, device, nvb):
    """The layer launch's own weights, on top of the MoE's.

    o_proj is a TP shard, so it is drawn per rank; the attention-residual
    stream and its two norms are replicated and must match on every rank.
    """
    from sglang.kernels.ops.moe.k3_mono_flydsl.attention.back import OK
    from sglang.kernels.ops.moe.k3_mono_flydsl.attention.kda import HIDDEN

    gc = torch.Generator(device=device).manual_seed(23)
    gr = torch.Generator(device=device).manual_seed(200 + rank)
    return {
        "w_o": _randn(gr, HIDDEN, OK, scale=0.02),
        "core": _randn(gr, ROWS, OK, scale=0.3),
        # The bank is allocated for every block of the model, so only its
        # first nvb rows are live -- that is the production stride.
        "blocks": _randn(gc, ROWS, MAX_BLOCKS, HIDDEN, scale=0.3),
        "prefix": _randn(gc, ROWS, HIDDEN, scale=0.3),
        "ares_nw": (1 + _randn(gc, HIDDEN, scale=0.1)).to(BF),
        "ares_qk": _randn(gc, HIDDEN, scale=0.02),
        "in_nw": (1 + _randn(gc, HIDDEN, scale=0.1)).to(BF),
        "eps": 1e-5,
        "out_eps": 1e-5,
        "nvb": nvb,
    }


def _run_k2(mod, w, scratch, *, nvb, reset, peers, epoch, rank, layer):
    """One layer launch. Returns (out, the updated prefix, the kernel's own
    MoE input read back out of its scratch)."""
    from sglang.kernels.ops.moe.k3_mono_flydsl.attention.kda import HIDDEN

    key = mod.K2Build(
        tokens=ROWS,
        nblocks=nvb,
        eps=w["eps"],
        out_eps=w["out_eps"],
        reset=reset,
    )
    # The kernel updates the prefix in place, so hand it a copy and keep the
    # input for the reference.
    prefix = w["prefix"].clone()
    out = torch.empty_like(prefix)
    mod.k2_launch(
        key,
        core=w["core"],
        w_o=w["w_o"],
        prefix=prefix,
        blocks=w["blocks"],
        ares_nw=w["ares_nw"],
        ares_qk=w["ares_qk"],
        in_nw=w["in_nw"],
        w_gate=w["w_gate"],
        bias=w["bias"],
        w_ld=w["w_ld"],
        w_sgu=w["w_sgu"],
        w_sd=w["w_sd"],
        w13=w["w13"],
        w13s=w["w13s"],
        w2=w["w2"],
        w2s=w["w2s"],
        ln_w=w["ln_w"],
        w_up=w["w_up_shard"],
        out=out,
        scratch=scratch,
        peers=peers.addresses,
        rank=rank,
        epoch=epoch,
        layer=layer,
    )
    torch.cuda.synchronize()
    off, nbytes = mod.scratch_layout(key)["xrow"]
    xrow = scratch[off : off + nbytes].view(BF).view(ROWS, HIDDEN)
    return out.clone(), prefix, xrow.clone()


def _worker_k2(rank, port, results):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP_SIZE
    )
    peers = None
    try:
        from sglang.kernels.ops.moe.k3_mono_flydsl import layer as k2
        from sglang.kernels.ops.moe.k3_mono_flydsl.common.peer_memory import PeerBuffer

        peers = PeerBuffer(k2.peer_bytes(ROWS), dist.group.WORLD, rank, TP_SIZE, device)
        peers.bytes.zero_()
        epoch = torch.zeros(1, dtype=torch.int32, device=device)
        scratch = torch.zeros(
            k2.scratch_bytes(k2.K2Build(tokens=ROWS, nblocks=MAX_BLOCKS)),
            dtype=torch.uint8,
            device=device,
        )

        w = _weights(rank, device, gate_up_interleaved=False)
        w.update(_layer_weights(rank, device, NVB))

        result = {"error": ""}
        seen = {}
        # Both prefix modes: a plain layer accumulates into the prefix, a
        # block-write layer restarts it at the attention output.
        for tag, reset in (("accumulate", False), ("reset", True)):
            epoch.add_(1)
            out, prefix, xrow = _run_k2(
                k2,
                w,
                scratch,
                nvb=NVB,
                reset=reset,
                peers=peers,
                epoch=epoch,
                rank=rank,
                layer=1,
            )
            ref_prefix, ref_x = _seam_reference(
                w["core"], w["w_o"], w["prefix"], w["blocks"], NVB, w, reset=reset
            )
            gathered = [torch.empty_like(out.cpu()) for _ in range(TP_SIZE)]
            dist.all_gather(gathered, out.cpu().contiguous())
            result[tag] = {
                "finite": bool(torch.isfinite(out).all()),
                "nonzero": bool(torch.count_nonzero(out) > 0),
                "ranks_agree": all(torch.equal(gathered[0], g) for g in gathered[1:]),
                # The seam: o_proj, its reduce and the aggregation.
                "cos_prefix": _cos(prefix, ref_prefix),
                "cos_x": _cos(xrow, ref_x),
                # The MoE, given the seam output the kernel itself produced.
                "cos_moe": _cos(out, _reference(xrow, w, rank)),
            }
            seen[tag] = xrow

        # Control: the two modes read the input prefix differently, so they
        # must disagree. Without this a kernel that ignored the prefix
        # entirely -- or a harness reading a stale scratch -- would still
        # match both references and the test above would mean nothing.
        result["modes_differ"] = _cos(seen["accumulate"], seen["reset"]) < COS_MIN
        results[rank] = result
    except Exception as error:  # surfaced by the parent as a test failure
        import traceback

        results[rank] = {
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        }
    finally:
        if peers is not None:
            peers.close()
        dist.destroy_process_group()


class TestKimiK3MonoDecode(CustomTestCase):
    check_guint = False

    def _run(self):
        if torch.cuda.device_count() < TP_SIZE:
            self.skipTest(f"needs {TP_SIZE} GPUs")
        if not torch.version.hip:
            self.skipTest("ROCm only")
        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(
            _worker,
            args=(_free_port(), results, self.check_guint),
            nprocs=TP_SIZE,
            join=True,
        )
        return results

    def test_moe_matches_aiter_reference(self):
        results = self._run()
        r0 = results.get(0) or {}
        if not r0.get("error"):
            print(
                f"rank0: cos vs reference {r0['cos_vs_reference']:.6f}"
                + (
                    f", rerun {r0['cos_rerun']:.6f}, layouts {r0['cos_layouts']:.6f}"
                    f" (equal={r0['layouts_equal']})"
                    if self.check_guint
                    else ""
                )
            )
        for rank in range(TP_SIZE):
            r = results.get(rank)
            self.assertIsNotNone(r, f"rank {rank} produced no result")
            self.assertEqual(r["error"], "", f"rank {rank} failed")
            self.assertTrue(r["finite"], f"rank {rank} output is not finite")
            self.assertTrue(r["nonzero"], f"rank {rank} output is all zero")
            self.assertTrue(r["ranks_agree"], "TP ranks disagree after the reduce")
            self.assertGreater(
                r["cos_vs_reference"],
                COS_MIN,
                f"rank {rank}: cos {r['cos_vs_reference']:.6f} vs the AITER stack",
            )
            if self.check_guint:
                self.assertGreater(
                    r["cos_rerun"],
                    COS_MIN,
                    f"rank {rank}: re-running the same launch differs "
                    f"(cos {r['cos_rerun']:.6f}) -- not a layout problem",
                )
                self.assertGreater(
                    r["cos_layouts"],
                    COS_MIN,
                    f"rank {rank}: the two expert layouts disagree "
                    f"(cos {r['cos_layouts']:.6f})",
                )


class TestKimiK3MonoLayer(CustomTestCase):
    """The whole-layer launch: o_proj + its all-reduce + the MLP seam + the MoE."""

    def test_layer_matches_torch_seam_and_aiter_moe(self):
        if torch.cuda.device_count() < TP_SIZE:
            self.skipTest(f"needs {TP_SIZE} GPUs")
        if not torch.version.hip:
            self.skipTest("ROCm only")
        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(_worker_k2, args=(_free_port(), results), nprocs=TP_SIZE, join=True)

        r0 = results.get(0) or {}
        if not r0.get("error"):
            for tag in ("accumulate", "reset"):
                r = r0[tag]
                print(
                    f"rank0 {tag}: cos prefix {r['cos_prefix']:.6f}, "
                    f"x {r['cos_x']:.6f}, moe {r['cos_moe']:.6f}"
                )
        for rank in range(TP_SIZE):
            r = results.get(rank)
            self.assertIsNotNone(r, f"rank {rank} produced no result")
            self.assertEqual(r["error"], "", f"rank {rank} failed")
            self.assertTrue(
                r["modes_differ"],
                f"rank {rank}: the two prefix modes agree, so neither checks the "
                "prefix -- the comparisons below prove nothing",
            )
            for tag in ("accumulate", "reset"):
                m = r[tag]
                self.assertTrue(m["finite"], f"rank {rank} {tag}: output not finite")
                self.assertTrue(m["nonzero"], f"rank {rank} {tag}: output all zero")
                self.assertTrue(
                    m["ranks_agree"], f"{tag}: TP ranks disagree after the reduce"
                )
                self.assertGreater(
                    m["cos_prefix"],
                    COS_MIN,
                    f"rank {rank} {tag}: the updated prefix differs "
                    f"(cos {m['cos_prefix']:.6f}) -- o_proj or its all-reduce",
                )
                self.assertGreater(
                    m["cos_x"],
                    COS_MIN,
                    f"rank {rank} {tag}: the MoE input differs "
                    f"(cos {m['cos_x']:.6f}) -- the attention-residual seam",
                )
                self.assertGreater(
                    m["cos_moe"],
                    COS_MIN,
                    f"rank {rank} {tag}: the MoE differs from the AITER stack "
                    f"(cos {m['cos_moe']:.6f}) given the kernel's own seam output",
                )


if __name__ == "__main__":
    import sys

    if "--guint" in sys.argv:
        sys.argv.remove("--guint")
        TestKimiK3MonoDecode.check_guint = True
    unittest.main()
