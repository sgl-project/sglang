"""Batch-invariance probe for the Qwen3.8 mono decode kernels (ROCm / FlyDSL).

One probe row keeps its inputs; only the step's other rows change (other
requests, zero pad rows, copies of the probe). The probe's outputs must stay
bit-identical, at every width 1 .. ``MAX_TOKENS`` and at the first and last
position, and equal to the probe alone (width 1). ATOM's MiniMax-M3 mono K4
failed the first check at 11 and 16 rows (ROCm/ATOM#2485): a down stage that
grouped an expert's partial sums by the expert's rank in the step's union.

``K1Test`` runs K1 (``gdn_pre``) on one GPU: core, res_out and the probe's
conv / SSM state. ``K2Test`` runs K2 (``layer_post``) on 8 GPUs with random
TP shards: out and res_out on every rank.

Run directly with ``python test/registered/amd/test_qwen3_5_mono_batch_invariance.py``;
``K2Test`` needs 8 GPUs.
"""

import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=600, suite="stage-c-test-large-8-gpu-amd-mi35x")

TP_SIZE = 8
BF = torch.bfloat16
KINDS = ("rand0", "rand1", "zero", "copy")


def _gfx950() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cases(max_tokens):
    """(width, probe position, kind of the other rows)."""
    for s in range(1, max_tokens + 1):
        for pos in sorted({0, s - 1}):
            for kind in KINDS if s > 1 else KINDS[:1]:
                yield s, pos, kind


def _rows(probe, s, pos, kind, rnd):
    """``s`` rows: ``probe`` at ``pos``, the others by ``kind``."""
    if kind == "copy":
        rows = probe.expand(s, *probe.shape[1:]).clone()
    elif kind == "zero":
        rows = torch.zeros(s, *probe.shape[1:], dtype=probe.dtype, device=probe.device)
    else:
        rows = rnd(s)
    rows[pos] = probe[0]
    return rows


def _diffs(outs):
    """The cases whose outputs differ from the same width and position with
    other random rows, and from the probe alone."""
    with_others, with_width = [], []
    for (s, pos, kind), got in outs.items():
        ref = outs[(s, pos, "rand0")]
        if not all(torch.equal(a, b) for a, b in zip(got, ref)):
            with_others.append((s, pos, kind))
        if kind == "rand0" and not all(
            torch.equal(a, b) for a, b in zip(got, outs[(1, 0, "rand0")])
        ):
            with_width.append((s, pos))
    return with_others, with_width


@unittest.skipUnless(_gfx950(), "needs FlyDSL on gfx950")
class K1Test(CustomTestCase):
    def test_probe_row(self):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import gdn
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl.layout import (
            BA,
            CONV,
            CORE,
            HD,
            HIDDEN,
            MAX_TOKENS,
            NV,
            QKVZ,
        )

        g = torch.Generator(device="cuda").manual_seed(0)

        def rnd(*shape, scale=1.0, dtype=BF):
            return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)

        w = {
            "ln_w": rnd(HIDDEN, scale=0.1),
            "w_qkvz": rnd(QKVZ, HIDDEN, scale=0.02),
            "w_ba": rnd(BA, HIDDEN, scale=0.05),
            "conv_w": rnd(CONV, 1, 4, scale=0.4),
            "a_log": rnd(NV, scale=0.5, dtype=torch.float32),
            "dt_bias": rnd(NV, scale=0.5),
            "norm_w": (1 + rnd(HD, scale=0.1, dtype=torch.float32)).to(BF),
        }
        slots, probe_slot = MAX_TOKENS + 2, 1
        conv0 = rnd(slots, CONV, 3)
        state0 = rnd(slots, NV, HD, HD, scale=0.05, dtype=torch.float32)
        probe_h, probe_r = rnd(1, HIDDEN, scale=2.0), rnd(1, HIDDEN, scale=2.0)
        epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
        # One scratch a width, as the glue has: a shared one is the vllm#60397 race.
        scratch = {
            s: torch.zeros(
                gdn.scratch_bytes(gdn.K1Build(tokens=s)),
                dtype=torch.uint8,
                device="cuda",
            )
            for s in range(1, MAX_TOKENS + 1)
        }
        outs = {}
        for s, pos, kind in _cases(MAX_TOKENS):
            g.manual_seed(1 + KINDS.index(kind))
            hidden = _rows(probe_h, s, pos, kind, lambda n: rnd(n, HIDDEN, scale=2.0))
            residual = _rows(probe_r, s, pos, kind, lambda n: rnd(n, HIDDEN, scale=2.0))
            others = list(range(2, slots))[: s - 1]
            idx = torch.tensor(
                others[:pos] + [probe_slot] + others[pos:],
                dtype=torch.int32,
                device="cuda",
            )
            conv, state = conv0.clone(), state0.clone()
            if kind == "zero":
                idx[torch.arange(s, device="cuda") != pos] = -1
            elif kind == "copy":
                conv[others], state[others] = conv[probe_slot], state[probe_slot]
            elif kind == "rand1":
                conv[others] = rnd(len(others), CONV, 3)
                state[others] = rnd(
                    len(others), NV, HD, HD, scale=0.05, dtype=torch.float32
                )
            core = torch.empty(s, CORE, dtype=BF, device="cuda")
            res_out = torch.empty(s, HIDDEN, dtype=BF, device="cuda")
            epoch.add_(1)
            gdn.gdn_pre(
                gdn.K1Build(tokens=s),
                **w,
                hidden=hidden,
                residual=residual,
                res_out=res_out,
                conv_state=conv,
                rstate=state,
                st_idx=idx,
                core=core,
                scratch=scratch[s],
                epoch=epoch,
                layer=0,
            )
            torch.cuda.synchronize()
            outs[(s, pos, kind)] = (
                core[pos],
                res_out[pos],
                conv[probe_slot],
                state[probe_slot],
            )
        with_others, with_width = _diffs(outs)
        self.assertEqual(with_others, [], "the probe changes with the other rows")
        self.assertEqual(with_width, [], "the probe changes with the width")


def _worker_k2(rank, port, results):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP_SIZE
    )
    peers = None
    try:
        from sglang.kernels.ops.moe.k3_mono_flydsl.common.peer_memory import (
            PeerBuffer,
            tp_agree,
        )
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import layer as k2
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl.layout import (
            CORE,
            E,
            HIDDEN,
            MAX_TOKENS,
            RI,
            SI,
        )

        g = torch.Generator(device=dev)

        def rnd(*shape, scale=1.0):
            return (torch.randn(*shape, generator=g, device=dev) * scale).to(BF)

        def codes(*shape, lo=0, hi=256):
            return torch.randint(
                lo, hi, shape, generator=g, device=dev, dtype=torch.uint8
            )

        # This rank's shards; random fp4 bytes, e8m0 scales near 2^-8.
        g.manual_seed(100 + rank)
        w = {
            "w_o": rnd(HIDDEN, CORE, scale=0.02),
            "w_sgu": rnd(2 * SI, HIDDEN, scale=0.02),
            "w_sd": rnd(HIDDEN, SI, scale=0.02),
            "w13": codes(E, 2 * RI, HIDDEN // 2),
            "w13s": codes(E * 2 * RI * HIDDEN // 32, lo=117, hi=121),
            "w2": codes(E, HIDDEN, RI // 2),
            "w2s": codes(E * HIDDEN * RI // 32, lo=117, hi=121),
        }
        # Replicated weights and the residual: one seed on every rank.
        g.manual_seed(0)
        w["ln_w"] = rnd(HIDDEN, scale=0.1)
        w["w_gate"] = rnd(E, HIDDEN, scale=0.02)
        w["w_sg"] = rnd(1, HIDDEN, scale=0.02)
        probe_r = rnd(1, HIDDEN)
        g.manual_seed(200 + rank)
        probe_c = rnd(1, CORE)

        peers = PeerBuffer(
            k2.peer_bytes(MAX_TOKENS), dist.group.WORLD, rank, TP_SIZE, dev
        )
        peers.bytes.zero_()
        epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        scratch = {
            s: torch.zeros(
                k2.scratch_bytes(k2.K2Build(tokens=s)), dtype=torch.uint8, device=dev
            )
            for s in range(1, MAX_TOKENS + 1)
        }
        # A rank that cannot build would leave the others spinning in the kernel.
        built = True
        try:
            for s in range(1, MAX_TOKENS + 1):
                k2.build(k2.K2Build(tokens=s))
        except Exception:
            built = False
            raise
        finally:
            if not tp_agree(built, dist.group.WORLD):
                raise RuntimeError("a rank failed to build K2")

        outs = {}
        for s, pos, kind in _cases(MAX_TOKENS):
            g.manual_seed(1 + KINDS.index(kind))
            residual = _rows(probe_r, s, pos, kind, lambda n: rnd(n, HIDDEN))
            g.manual_seed(1000 * rank + 1 + KINDS.index(kind))
            core = _rows(probe_c, s, pos, kind, lambda n: rnd(n, CORE))
            out = torch.empty(s, HIDDEN, dtype=BF, device=dev)
            res_out = torch.empty_like(out)
            epoch.add_(1)
            k2.layer_post(
                k2.K2Build(tokens=s),
                core=core,
                residual=residual,
                **w,
                out=out,
                res_out=res_out,
                scratch=scratch[s],
                peers=peers.addresses,
                rank=rank,
                epoch=epoch,
                layer=k2.TAG_SLOTS // 2,
            )
            torch.cuda.synchronize()
            outs[(s, pos, kind)] = (out[pos], res_out[pos])
        ref = outs[(1, 0, "rand0")][0]
        with_others, with_width = _diffs(outs)
        results[rank] = {
            "error": "",
            "finite": bool(torch.isfinite(ref).all()),
            "nonzero": bool(ref.abs().sum() > 0),
            "with_others": with_others,
            "with_width": with_width,
        }
    except Exception as error:  # surfaced by the parent as a test failure
        import traceback

        results[rank] = {
            "error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        }
    finally:
        if peers is not None:
            peers.close()
        dist.destroy_process_group()


@unittest.skipUnless(_gfx950(), "needs FlyDSL on gfx950")
class K2Test(CustomTestCase):
    def test_probe_row(self):
        if torch.cuda.device_count() < TP_SIZE:
            self.skipTest(f"needs {TP_SIZE} GPUs")
        results = mp.Manager().dict()
        mp.spawn(_worker_k2, args=(_free_port(), results), nprocs=TP_SIZE, join=True)
        for rank in range(TP_SIZE):
            r = results.get(rank)
            self.assertIsNotNone(r, f"rank {rank} produced no result")
            self.assertEqual(r["error"], "", f"rank {rank} failed")
            self.assertTrue(r["finite"] and r["nonzero"], f"rank {rank}: bad output")
            self.assertEqual(
                r["with_others"], [], f"rank {rank}: the probe changes with the others"
            )
            self.assertEqual(
                r["with_width"], [], f"rank {rank}: the probe changes with the width"
            )


if __name__ == "__main__":
    unittest.main()
