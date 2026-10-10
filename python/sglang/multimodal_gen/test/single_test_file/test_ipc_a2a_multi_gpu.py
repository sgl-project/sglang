"""The copy-engine all-to-all and the pipelined Ulysses attention must match the
sequential exchange bit for bit.

Both only activate with SGLANG_DIFFUSION_IPC_A2A_MULTI / a pipeline group count
on a same-host group with peer-to-peer access, so nothing in the single-GPU
suite reaches them. Runs on 2 ranks, and on 4 when four GPUs are visible:

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_ipc_a2a_multi_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase


def _attend(q, k, v):
    # head-independent and exact under any head grouping; the flips and rolls
    # read across the whole gathered sequence, so a misplaced shard shows up
    return (q.float() * 2 + k.flip(0).float() - v.roll(1, 0).float()).to(q.dtype)


def _worker() -> int:
    import torch.distributed as dist

    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world, ulysses_degree=world
    )

    from sglang.multimodal_gen.runtime.distributed.device_communicators.ipc_a2a_multi import (
        IPC_A2A_MULTI,
        ulysses_pipelined_attention,
    )
    from sglang.multimodal_gen.runtime.layers.usp import (
        _usp_all_to_all_single,
        _usp_input_all_to_all_packed_qkv,
        _usp_output_all_to_all,
    )

    peers = [d for d in range(world) if d != rank]
    if not all(torch.cuda.can_device_access_peer(rank, d) for d in peers):
        print("SKIP no peer-to-peer access between the devices", flush=True)
        return 0

    failures = []

    def sequential(q, k, v):
        envs.SGLANG_DIFFUSION_IPC_A2A_MULTI = False
        qs, ks, vs = _usp_input_all_to_all_packed_qkv(q, k, v)
        return _usp_output_all_to_all(_attend(qs, ks, vs)[None], head_dim=2)[0]

    # (s_local, heads, head_dim, groups); heads % (world * groups) == 0
    cases = [(96, 16, 64, 2), (96, 16, 64, 4), (40, 32, 128, 2)]
    if world == 2:
        cases.append((128, 16, 64, 8))
    for s_local, heads, head_dim, groups in cases:
        # three calls in a row: the slots alternate and the counters advance
        for call in range(3):
            torch.manual_seed(
                1000 * call + s_local
            )  # identical full tensors on every rank
            full = [
                torch.randn(
                    s_local * world,
                    heads,
                    head_dim,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for _ in range(3)
            ]
            q, k, v = (t.narrow(0, rank * s_local, s_local).contiguous() for t in full)
            ref = sequential(q, k, v)
            got = ulysses_pipelined_attention(q, k, v, _attend, groups)
            if got is None:
                failures.append(
                    f"pipelined returned None for {(s_local, heads, head_dim, groups)}"
                )
            elif not torch.equal(ref, got):
                failures.append(
                    f"pipelined {(s_local, heads, head_dim, groups)} call {call}"
                )

    # fill: the caller writes each destination's q/k itself (H3's QK-norm does),
    # own blocks straight into this rank's receive slot; v moves on its own
    def transform(t, scale):
        return (t.float() * scale + 1).to(t.dtype)

    for s_local, heads, head_dim, groups in cases:
        for call in range(3):
            torch.manual_seed(2000 + 1000 * call + s_local)
            full = [
                torch.randn(
                    s_local * world,
                    heads,
                    head_dim,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for _ in range(3)
            ]
            q, k, v = (t.narrow(0, rank * s_local, s_local).contiguous() for t in full)

            def fill(head_start, head_count, q_dst, k_dst):
                h = slice(head_start, head_start + head_count)
                q_dst.copy_(transform(q[:, h], 2))
                k_dst.copy_(transform(k[:, h], 3))

            ref = sequential(transform(q, 2), transform(k, 3), v)
            got = ulysses_pipelined_attention(q, k, v, _attend, groups, fill=fill)
            if got is None:
                failures.append(
                    f"filled pipeline returned None for {(s_local, heads, head_dim, groups)}"
                )
            elif not torch.equal(ref, got):
                failures.append(
                    f"filled pipeline {(s_local, heads, head_dim, groups)} call {call}"
                )

    # groups=-1 picks a count that divides the heads per rank (4 for 16 heads at
    # 2 ranks, 2 at 4) and still matches
    torch.manual_seed(77)
    full = [
        torch.randn(64 * world, 16, 64, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]
    q, k, v = (t.narrow(0, rank * 64, 64).contiguous() for t in full)
    got = ulysses_pipelined_attention(q, k, v, _attend, -1)
    if got is None or not torch.equal(sequential(q, k, v), got):
        failures.append("auto group count")

    # a head count the groups cannot split must decline, not mis-shard
    q = torch.randn(32, 12, 64, dtype=torch.bfloat16, device="cuda")
    if ulysses_pipelined_attention(q, q, q, _attend, 4) is not None:
        failures.append("pipelined accepted 12 heads in 4 groups")

    # the plain N-rank exchange behind _usp_all_to_all_single
    for numel in (world * 1024, world * 4096 * 33):
        x = torch.randn(numel, dtype=torch.bfloat16, device="cuda")
        envs.SGLANG_DIFFUSION_IPC_A2A_MULTI = False
        ref = _usp_all_to_all_single(x.clone())
        envs.SGLANG_DIFFUSION_IPC_A2A_MULTI = True
        calls = IPC_A2A_MULTI.calls
        got = _usp_all_to_all_single(x.clone())
        envs.SGLANG_DIFFUSION_IPC_A2A_MULTI = False
        if IPC_A2A_MULTI.calls == calls:
            failures.append(f"exchange of {numel} elements never took the copy engine")
        elif not torch.equal(ref, got):
            failures.append(f"exchange of {numel} elements")

    # a comparison where every arm quietly fell back would pass while testing nothing
    if not IPC_A2A_MULTI.inited or IPC_A2A_MULTI.failed or not IPC_A2A_MULTI.staging:
        failures.append(
            f"transport never engaged (inited={IPC_A2A_MULTI.inited} "
            f"failed={IPC_A2A_MULTI.failed} staged={len(IPC_A2A_MULTI.staging)})"
        )

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} MISMATCH: {failures}", flush=True)
    if rank == 0:
        print(
            f"IPC_A2A_MULTI_PARITY {'FAIL' if verdict.item() else 'PASS'} world={world}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestIpcA2AMultiGpu(CustomTestCase):
    def _run(self, world: int, port: int):
        if not current_platform.is_cuda():
            self.skipTest("CUDA-IPC transport is unavailable on this platform")
        if torch.cuda.device_count() < world:
            self.skipTest(f"needs {world} GPUs")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={world}",
                f"--master-port={port}",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
        self.assertEqual(proc.returncode, 0, "copy-engine all-to-all diverged")
        if "SKIP" not in proc.stdout:
            self.assertIn(f"IPC_A2A_MULTI_PARITY PASS world={world}", proc.stdout)

    def test_two_ranks_match_sequential_bitwise(self):
        self._run(2, 29527)

    def test_four_ranks_match_sequential_bitwise(self):
        self._run(4, 29537)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()
