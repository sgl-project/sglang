# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Rank agreement of the K3.1 prefill custom all-gather plane (SGLANG_K31_PF_CUSTOM_AG).

The multicast all-gather pairs CTA windows and buffer regions by index across the TP group, so every rank has to
take the same path with the same CTA count, byte cap and buffer row width. A rank with different knobs (or a
different num_pull_blocks) would hang its peers or copy rows that a peer has not stored yet, and a rank whose
plane setup failed would sit in NCCL while its peers spin in the kernel. ``_plane`` all-gathers each rank's
(knobs, plane configuration) over the TP CPU group once and decides for every rank.

These tests run ``_plane`` itself on 4 gloo ranks, with the GPU pieces (multicast plane, symmetric buffer,
kernel) stubbed, and check that every rank reaches the same, expected decision:
- served form (SGLANG_K31_PF_CUSTOM_AG=1, default knobs, equal comm configs): custom path on every rank;
- one rank with another BLOCKS / PAD_MAX / MAX_MB, or another num_pull_blocks: every rank raises;
- one rank without a plane (no SP-tail comm, or an exception while building it): NCCL on every rank.
CPU-only.
"""

import os
import sys
import traceback
import types
import unittest
from unittest import mock

import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

WORLD = 4
KNOBS = (
    "SGLANG_K31_PF_CUSTOM_AG_BLOCKS",
    "SGLANG_K31_PF_CUSTOM_AG_MAX_MB",
    "SGLANG_K31_PF_CUSTOM_AG_PAD_MAX",
)

# name -> (per-rank env overrides, per-rank comm num_pull_blocks or None = no SP-tail comm,
#          rank whose symmetric-buffer setup raises or None)
SCENARIOS = {
    "served": ({}, {}, None),
    "blocks_differs": ({2: {"SGLANG_K31_PF_CUSTOM_AG_BLOCKS": "16"}}, {}, None),
    "pad_max_differs": ({1: {"SGLANG_K31_PF_CUSTOM_AG_PAD_MAX": "2.0"}}, {}, None),
    "max_mb_differs": ({3: {"SGLANG_K31_PF_CUSTOM_AG_MAX_MB": "512"}}, {}, None),
    "pull_blocks_differs": ({}, {3: 16}, None),
    "no_comm_on_one_rank": ({}, {0: None}, None),
    "setup_raises_on_one_rank": ({}, {}, 1),
}
# Expected decision, the same on every rank (from the protocol, not from the code under test).
EXPECT = {
    "served": "custom",
    "blocks_differs": "raise",
    "pad_max_differs": "raise",
    "max_mb_differs": "raise",
    "pull_blocks_differs": "raise",
    "no_comm_on_one_rank": "nccl",
    "setup_raises_on_one_rank": "nccl",
}


class _TP:
    """The TP GroupCoordinator surface _plane uses; all_gather_object is GroupCoordinator's own body."""

    def __init__(self, rank):
        self.ranks = list(range(WORLD))
        self.rank_in_group = rank
        self.world_size = WORLD

    def all_gather_object(self, obj):
        objs = [None] * self.world_size
        dist.all_gather_object(objs, obj)
        return objs


def _stubs(rank, tp, pull_blocks, setup_raises):
    sp = types.ModuleType("sp_collective_stub")
    sp.Tuning = lambda blocks, threads: (blocks, threads)
    sp.all_gather_direct = lambda *a, **k: None
    sp_pkg = types.ModuleType("communication_kernels_stub")
    sp_pkg.sp_collective = sp

    ar = types.ModuleType("k3_ar_fusion_stub")

    def max_rows():
        if setup_raises == rank:
            raise RuntimeError("symmetric buffer unavailable on this rank")
        return 8192

    ar.symm_buffer_max_rows = max_rows
    ar.symm_buffer = lambda *a, **k: None
    ar.get_mc_ptr = lambda t: 0

    spc = types.ModuleType("k3_sp_collective_stub")
    nb = pull_blocks.get(rank, 32)
    comm = (
        None
        if nb is None
        else types.SimpleNamespace(config=types.SimpleNamespace(num_pull_blocks=nb))
    )
    spc.prefill_sp_comm = lambda: comm

    rc = types.ModuleType("runtime_context_stub")
    rc.get_parallel = lambda: types.SimpleNamespace(tp_group=tp)
    return sp_pkg, ar, spc, rc


def _worker(rank, port, result_q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    import sglang.srt.layers.communication as comm_pkg
    from sglang.srt.layers.communication import k31_pf_custom_ag as m

    tp = _TP(rank)
    out = {}
    for name, (env_over, pull_blocks, setup_raises) in SCENARIOS.items():
        env = {k: v for k, v in os.environ.items() if k not in KNOBS}
        env["SGLANG_K31_PF_CUSTOM_AG"] = "1"
        env.update(env_over.get(rank, {}))
        sp_pkg, ar, spc, rc = _stubs(rank, tp, pull_blocks, setup_raises)
        m._PLANES.clear()
        m._ENABLED = None
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.dict(
                sys.modules,
                {
                    "sglang.kernels.ops.communication": sp_pkg,
                    "sglang.srt.runtime_context": rc,
                },
            ),
            mock.patch.object(comm_pkg, "k3_ar_fusion", ar, create=True),
            mock.patch.object(comm_pkg, "k3_sp_collective", spc, create=True),
        ):
            try:
                assert m.enabled()
                p = m._plane(tp)
                out[name] = ("nccl",) if p is None else ("custom", p[0], p[1], p[2])
            except RuntimeError as e:
                out[name] = ("raise", str(e)[:120])
            except Exception:
                out[name] = ("error", traceback.format_exc()[-400:])
        dist.barrier()
    result_q.put((rank, out))
    dist.destroy_process_group()


class TestPfCustomAgRankAgreement(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        port = 29600 + os.getpid() % 2000
        procs = [ctx.Process(target=_worker, args=(r, port, q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        cls.results = {}
        try:
            for _ in range(WORLD):
                rank, out = q.get(timeout=300)
                cls.results[rank] = out
        finally:
            for p in procs:
                p.join(timeout=60)
                if p.is_alive():
                    p.kill()

    def _decisions(self, name):
        return [self.results[r][name] for r in range(WORLD)]

    def _check(self, name):
        got = self._decisions(name)
        self.assertEqual(
            [g[0] for g in got],
            [EXPECT[name]] * WORLD,
            f"{name}: per-rank decisions {got}",
        )
        if EXPECT[name] == "custom":
            # every rank: world size, its own rank, the same byte cap
            self.assertEqual([g[1] for g in got], [WORLD] * WORLD)
            self.assertEqual([g[2] for g in got], list(range(WORLD)))
            self.assertEqual(len({g[3] for g in got}), 1)

    def test_served_form_takes_custom_path_on_every_rank(self):
        self._check("served")

    def test_blocks_mismatch_raises_on_every_rank(self):
        self._check("blocks_differs")

    def test_pad_max_mismatch_raises_on_every_rank(self):
        self._check("pad_max_differs")

    def test_max_mb_mismatch_raises_on_every_rank(self):
        self._check("max_mb_differs")

    def test_pull_blocks_mismatch_raises_on_every_rank(self):
        self._check("pull_blocks_differs")

    def test_missing_comm_on_one_rank_sends_every_rank_to_nccl(self):
        self._check("no_comm_on_one_rank")

    def test_setup_failure_on_one_rank_sends_every_rank_to_nccl(self):
        self._check("setup_raises_on_one_rank")


if __name__ == "__main__":
    unittest.main()
