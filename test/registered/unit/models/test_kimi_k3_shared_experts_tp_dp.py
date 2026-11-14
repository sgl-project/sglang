"""Kimi-K3 shared-expert TP across attention-DP ranks (--shared-experts-tp-size under DP).

CPU, W simulated ranks as gloo processes:
- KimiK3MoE's shared-expert all-gather / reduce-scatter pads every rank's rows to the DP
  maximum, so ranks with different batch sizes, and idle ranks with 0 rows, line up. The
  TP-sharded shared MLP over the gathered rows, reduce-scattered back, equals each rank's
  replicated shared MLP on its own rows (up to fp32 summation order).
- A rank whose rows exceed the DP maximum raises.
- --shared-experts-tp-size accepts divisors of TP under attention DP with attention TP 1 and
  still rejects other widths (server-arg hook).
"""

import os
import socket
import types
import unittest
import unittest.mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

H, I = 16, 24  # hidden, shared intermediate (gate|up halves of I each)


class _Group:
    """GroupCoordinator stand-in over the default gloo group."""

    def __init__(self, world):
        self.world_size = world

    def all_gather_into_tensor(self, out, x):
        parts = list(out.chunk(self.world_size))
        dist.all_gather(parts, x.contiguous())
        out.copy_(torch.cat(parts))

    def reduce_scatter_tensor(self, out, x):
        full = x.clone()
        dist.all_reduce(full)
        r = dist.get_rank()
        out.copy_(full.chunk(self.world_size)[r])


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, rows, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from sglang.srt.models import kimi_k3

        torch.manual_seed(0)  # same full weights on every rank
        gate_up = torch.randn(2 * I, H, dtype=torch.float64)
        down = torch.randn(H, I, dtype=torch.float64)
        torch.manual_seed(100 + rank)
        x = torch.randn(rows[rank], H, dtype=torch.float64)

        def mlp(inp, gu, dn):
            g, u = (inp @ gu.t()).chunk(2, dim=-1)
            return (torch.nn.functional.silu(g) * u) @ dn.t()

        ref = mlp(x, gate_up, down)
        s = I // world  # this rank's column / row slice
        gu_r = torch.cat(
            [
                gate_up[rank * s : (rank + 1) * s],
                gate_up[I + rank * s : I + (rank + 1) * s],
            ]
        )
        dn_r = down[:, rank * s : (rank + 1) * s]

        moe = types.SimpleNamespace(
            _shared_experts_tp_group=_Group(world),
            _shared_experts_tp_across_dp=True,
            _shared_experts_tp_sp_comm=False,  # the GB300 push collectives need GPUs
        )
        moe._shared_tp_rows = types.MethodType(kimi_k3.KimiK3MoE._shared_tp_rows, moe)
        moe._shared_tp_sp_state = types.MethodType(
            kimi_k3.KimiK3MoE._shared_tp_sp_state, moe
        )
        with unittest.mock.patch.object(
            kimi_k3, "get_dp_global_num_tokens", lambda: list(rows)
        ):
            gathered = kimi_k3.KimiK3MoE._gather_shared_expert_inputs(moe, x)
            out = kimi_k3.KimiK3MoE._reduce_scatter_shared_experts(
                moe, mlp(gathered, gu_r, dn_r), x
            )
            too_many = None
            try:
                moe._shared_tp_rows(torch.zeros(max(rows) + 1, H))
            except RuntimeError as e:
                too_many = str(e)
        q.put(
            (
                rank,
                tuple(gathered.shape),
                tuple(out.shape),
                (out - ref).abs().max().item() if out.numel() else 0.0,
                too_many,
            )
        )
    finally:
        dist.destroy_process_group()


class TestSharedExpertsTpAcrossDp(unittest.TestCase):
    def _run(self, rows):
        world = len(rows)
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = [
            ctx.Process(target=_worker, args=(r, world, self.port, rows, q))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        res = sorted(q.get(timeout=300) for _ in range(world))
        for p in procs:
            p.join(60)
            self.assertEqual(p.exitcode, 0)
        return res

    def setUp(self):
        self.port = _free_port()

    def test_uneven_and_idle_ranks(self):
        rows = [3, 0, 5, 1]
        for rank, gshape, oshape, err, too_many in self._run(rows):
            self.assertEqual(gshape, (max(rows) * len(rows), H))
            self.assertEqual(oshape, (rows[rank], H))
            self.assertLess(err, 1e-9)
            self.assertIn("exceed the DP maximum", too_many)

    def test_equal_rows(self):
        for rank, gshape, oshape, err, _ in self._run([2, 2]):
            self.assertEqual(gshape, (4, H))
            self.assertEqual(oshape, (2, H))
            self.assertLess(err, 1e-9)


class TestServerArgHook(unittest.TestCase):
    def _check(self, *, size, tp=8, dp=8, a2a="megamoe"):
        from sglang.srt.arg_groups import parallel_hook

        cfg = types.SimpleNamespace(
            shared_experts_tp_size=size,
            tp_size=tp,
            attn_dp_size=dp,
            moe_a2a_backend=a2a,
            model_path="/m",
        )
        view = types.SimpleNamespace(attn_cp_size=1)
        hf = types.SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
        with unittest.mock.patch.multiple(
            parallel_hook,
            resolving_view=lambda _: cfg,
            resolved_view=lambda _: view,
            model_config_of=lambda _: types.SimpleNamespace(hf_config=hf),
            parse_connector_type=lambda _: None,
        ):
            parallel_hook.handle_shared_experts_tp(object())

    def test_divisors_of_tp_under_adp(self):
        for size in (2, 4, 8):
            self._check(size=size)

    def test_rejects_non_divisor(self):
        with self.assertRaisesRegex(ValueError, "must divide"):
            self._check(size=3)

    def test_attn_tp_rule_kept_without_dp(self):
        # No attention DP: attention TP is 8, so 4 still divides it and 16 does not.
        self._check(size=4, dp=1)
        with self.assertRaisesRegex(ValueError, "must divide"):
            self._check(size=16, dp=1)


if __name__ == "__main__":
    unittest.main()
