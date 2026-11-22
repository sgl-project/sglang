"""The copy-engine all-to-all and the pipelined Ulysses attention must match the
sequential exchange bit for bit.

Both only activate with SGLANG_DIFFUSION_IPC_A2A_MULTI / a pipeline group count
on a same-host group with peer-to-peer access, so nothing in the single-GPU
suite reaches them. Runs on 2 ranks, and on 4 when four GPUs are visible:

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_ipc_a2a_multi_gpu.py
"""

from __future__ import annotations

import ctypes
import errno
import math
import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase


def _attend(q, k, v):
    # head-independent and exact under any head grouping; the flips and rolls
    # read across the whole gathered sequence, so a misplaced shard shows up
    return (q.float() * 2 + k.flip(0).float() - v.roll(1, 0).float()).to(q.dtype)


def _attend_rows(q, k, v):
    """`_attend` for [batch, rows, heads, head_dim] parts whose k/v may carry more
    rows (a replicated KV prefix) and fewer heads (GQA) than q."""
    rep = q.shape[2] // k.shape[2]
    k = k.repeat_interleave(rep, dim=2).float()
    v = v.repeat_interleave(rep, dim=2).float()
    rows = q.shape[1]
    # elementwise only (a reduction could round differently per head count):
    # the leading and the trailing rows both read, so every row counts
    return (
        q.float() * 2
        + k[:, :rows]
        + k.flip(1)[:, :rows]
        - v.roll(1, 1)[:, -rows:]
        + v[:, :rows] * 0.5
    ).to(q.dtype)


def _deny_pidfd_getfd() -> None:
    """Fail pidfd_getfd with EPERM in this process, as containers without ptrace
    rights do; importing an expandable-segment IPC handle needs it."""

    class Insn(ctypes.Structure):  # struct sock_filter
        _fields_ = [
            ("code", ctypes.c_uint16),
            ("jt", ctypes.c_uint8),
            ("jf", ctypes.c_uint8),
            ("k", ctypes.c_uint32),
        ]

    class Prog(ctypes.Structure):  # struct sock_fprog
        _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(Insn))]

    insns = (Insn * 4)(
        Insn(0x20, 0, 0, 0),  # load the syscall number
        Insn(0x15, 0, 1, 438),  # pidfd_getfd, the same number on every arch
        Insn(0x06, 0, 0, 0x00050000 | errno.EPERM),  # SECCOMP_RET_ERRNO
        Insn(0x06, 0, 0, 0x7FFF0000),  # SECCOMP_RET_ALLOW
    )
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_NO_NEW_PRIVS, then PR_SET_SECCOMP with SECCOMP_MODE_FILTER
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(
        22, 2, ctypes.byref(Prog(len(insns), insns)), 0, 0
    ):
        raise OSError(ctypes.get_errno(), "could not install the seccomp filter")


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
        # three calls in a row: the slots are reused and the counters advance
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

    # groups=-1 picks the most groups that divide the heads per rank and whose
    # calls still fill three waves of 128-row tiles, and still matches; a shape too
    # small to fill them keeps the sequential exchange
    sms = torch.cuda.get_device_properties(rank).multi_processor_count
    s_auto = 3 * sms * 128 // world  # one head per group fills three waves
    torch.manual_seed(77)
    full = [
        torch.randn(s_auto * world, 16, 64, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]
    q, k, v = (t.narrow(0, rank * s_auto, s_auto).contiguous() for t in full)
    got = ulysses_pipelined_attention(q, k, v, _attend, -1)
    if got is None or not torch.equal(sequential(q, k, v), got):
        failures.append("auto group count")
    small = torch.randn(64, 16, 64, dtype=torch.bfloat16, device="cuda")
    if ulysses_pipelined_attention(small, small, small, _attend, -1) is not None:
        failures.append("auto pipelined a shape too small to fill the GPU")

    # sequential=: the first call per shape and signature also runs the
    # sequential exchange and keeps the pipeline only on a byte-for-byte match
    torch.manual_seed(78)
    full = [
        torch.randn(80 * world, 16, 64, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    ]
    q, k, v = (t.narrow(0, rank * 80, 80).contiguous() for t in full)
    ref = sequential(q, k, v)
    for call in range(2):
        got = ulysses_pipelined_attention(
            q,
            k,
            v,
            _attend,
            2,
            sequential=lambda: sequential(q, k, v),
            signature="same",
        )
        if got is None or not torch.equal(ref, got):
            failures.append(f"verified pipeline call {call}")
    off = ref + 1  # a "sequential" result the pipeline cannot reproduce
    got = ulysses_pipelined_attention(
        q, k, v, _attend, 2, sequential=lambda: off, signature="other"
    )
    if got is None or not torch.equal(got, off):
        failures.append("a mismatching first call did not return the sequential result")
    if (
        ulysses_pipelined_attention(
            q, k, v, _attend, 2, sequential=lambda: off, signature="other"
        )
        is not None
    ):
        failures.append("a mismatched shape kept pipelining")

    # a head count the groups cannot split must decline, not mis-shard
    q = torch.randn(32, 12, 64, dtype=torch.bfloat16, device="cuda")
    if ulysses_pipelined_attention(q, q, q, _attend, 4) is not None:
        failures.append("pipelined accepted 12 heads in 4 groups")

    # batch, replicated rows (prefix, suffix, KV-only prefix) and GQA: the output
    # rows match attention over the global layout, computed with every head
    def layout_case(batch, s_local, heads, kv_heads, rep, rep_kv, first, groups, seed):
        torch.manual_seed(seed)
        dims = 32
        shard = lambda h: torch.randn(
            batch, s_local * world, h, dims, device="cuda"
        ).bfloat16()
        full = [shard(heads), shard(kv_heads), shard(kv_heads)]
        reps = [
            torch.randn(batch, n, h, dims, device="cuda").bfloat16() if n else None
            for n, h in ((rep, heads), (rep_kv, kv_heads), (rep_kv, kv_heads))
        ]
        mine = [t[:, rank * s_local : (rank + 1) * s_local].contiguous() for t in full]

        def joined(t, r):
            if r is None:
                return t
            return torch.cat([r, t] if first else [t, r], dim=1)

        out_all = _attend_rows(*(joined(t, r) for t, r in zip(full, reps)))
        lo = (rep if first else 0) + rank * s_local
        own = out_all[:, lo : lo + s_local]
        if rep:
            rep_rows = out_all[:, :rep] if first else out_all[:, -rep:]
            own = torch.cat([rep_rows, own] if first else [own, rep_rows], dim=1)
        got = ulysses_pipelined_attention(
            *mine,
            _attend_rows,
            groups,
            replicated=tuple(reps),
            replicated_first=first,
        )
        label = f"layout b{batch} h{heads}/{kv_heads} rep {rep}/{rep_kv} first={first} g{groups}"
        if got is None:
            failures.append(f"{label} returned None")
        elif not torch.equal(got, own):
            failures.append(label)

    for call in range(2):  # every layout reuses the one pool
        layout_case(2, 48, 16, 16, 0, 0, True, 2, 300 + call)
        layout_case(1, 40, 16, 16, 9, 9, True, 2, 310 + call)
        layout_case(2, 40, 16, 16, 7, 7, False, 2, 320 + call)
        layout_case(1, 40, 16, 16, 0, 11, True, 2, 330 + call)
        layout_case(2, 40, 16, 8, 5, 5, True, 2, 340 + call)

    # the auto-group case grows both buffers to their widest plan plus 25% headroom
    pools = IPC_A2A_MULTI.pools
    if list(pools) != [torch.bfloat16]:
        failures.append(f"pipeline pools per dtype: {list(pools)}")
    else:
        slots = list(IPC_A2A_MULTI.slots.values())
        expected = (
            math.ceil(max(s.in_total for s in slots) * 1.25),
            math.ceil(max(s.out_total for s in slots) * 1.25),
            max(s.plan.groups for s in slots),
        )
        if pools[torch.bfloat16].sizes != expected:
            failures.append(f"pool sizes {pools[torch.bfloat16].sizes} != {expected}")

    # one set of slots is enough: a peer cannot write a call's rows or output
    # blocks before this rank has read the previous call's, however far ahead
    # it runs. Rank 0 attends slowly; alternating plans share the pool.
    def slow_attend(q, k, v):
        if rank == 0:
            torch.cuda._sleep(20_000_000)
        return _attend(q, k, v)

    issued = []
    for call in range(6):  # queued back to back; compared only afterwards
        s_local, heads, groups = ((96, 16, 2), (40, 32, 4))[call % 2]
        torch.manual_seed(4000 + call)
        full = [
            torch.randn(s_local * world, heads, 64, dtype=torch.bfloat16, device="cuda")
            for _ in range(3)
        ]
        parts = [t.narrow(0, rank * s_local, s_local).contiguous() for t in full]
        issued.append((parts, ulysses_pipelined_attention(*parts, slow_attend, groups)))
    for call, (parts, got) in enumerate(issued):
        if got is None or not torch.equal(sequential(*parts), got):
            failures.append(f"skewed back-to-back call {call}")

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


def _timeout_worker() -> int:
    """An expired peer wait: the last rank arrives three budgets late, so its
    peers' waits expire after SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS and
    check_timeout fails the request on every rank, naming who waited for whom.
    The plain exchange, then the head-group pipeline: its input rows; its
    output rows, with the late rank stalled inside its first attention, on the
    group streams and on the gathering stream."""
    import time

    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        get_sp_group,
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
        TIMEOUT_EXCHANGE,
        TIMEOUT_PIPELINE_INPUT,
        TIMEOUT_PIPELINE_OUTPUT,
        IpcPeerTimeoutError,
        ipc_a2a_multi_ready,
    )

    group = get_sp_group().ulysses_group
    if not ipc_a2a_multi_ready(group, enabled=True):
        print("SKIP no peer-to-peer copy-engine transport", flush=True)
        return 0
    state = IPC_A2A_MULTI
    budget_ns = state.budget_ns
    budget_s = budget_ns / 1e9
    late = world - 1
    failures = []
    # the late rank's stall inside its next attention call (head-group pipeline)
    stall_in_attend = []

    def run_late(name, call, expected, waited, max_budgets, in_attend=False):
        # the first call sets up the buffers (a collective) under a budget no
        # setup skew reaches; then the late rank sleeps through three budgets,
        # before the calls or, with its input rows sent, inside the first one
        state.budget_ns = 60 * 10**9
        if call() is None:
            failures.append(f"{name}: the transport declined the call")
        torch.cuda.synchronize()
        dist.barrier()
        state.budget_ns = budget_ns
        if rank == late:
            if in_attend:
                stall_in_attend.append(3 * budget_s)
            else:
                time.sleep(3 * budget_s)
        start = time.monotonic()
        for _ in range(4):
            call()
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if state.timed_out.item() != expected:
            failures.append(f"{name}: code {state.timed_out.item()} != {expected}")
        if rank != late and not budget_s <= elapsed < max_budgets * budget_s:
            failures.append(f"{name}: four calls took {elapsed:.3f} s")
        try:
            state.check_timeout()
            failures.append(f"{name}: check_timeout passed")
        except IpcPeerTimeoutError as e:
            if f"after {budget_s:g} s: {waited}." not in str(e):
                failures.append(f"{name}: {str(e)[:300]}")
        if not state.failed:
            failures.append(f"{name}: the transport stayed in use")

    # the plain exchange: a spin flags both sides, so every rank holds the
    # code, and each later spin on the late rank spends its own budget until
    # that rank's rows arrive
    send = torch.randn(world, 4096, dtype=torch.bfloat16, device="cuda")
    everyone = ", ".join(map(str, range(world)))
    run_late(
        "exchange",
        lambda: state.exchange(send),
        TIMEOUT_EXCHANGE,
        f"ranks {everyone} waited for a peer in the copy-engine all-to-all",
        max_budgets=5,
    )

    # the head-group pipeline: only the waiting ranks hold the code, which names
    # the late rank; the first wait expires and every later bounded wait gives
    # up at once, while the later groups' front-end input waits end when the
    # late rank's rows arrive
    def attend(q, k, v):
        if stall_in_attend:
            time.sleep(stall_in_attend.pop())
        return _attend_rows(q, k, v)

    def attend_into(q, k, v, out):
        if out is not None:
            out.copy_(attend(q, k, v))
        return True

    s_local, heads, head_dim = 384, 2 * world, 64
    q, k, v = (
        torch.randn(1, s_local, heads, head_dim, dtype=torch.bfloat16, device="cuda")
        for _ in range(3)
    )
    waiting = ", ".join(map(str, range(late)))
    waiting = f"rank{'s' if late > 1 else ''} {waiting} waited for rank {late}"
    for name, kwargs, code, rows, in_attend in (
        ("pipeline in", {}, TIMEOUT_PIPELINE_INPUT, "input", False),
        ("pipeline out", {}, TIMEOUT_PIPELINE_OUTPUT, "output", True),
        (
            "pipeline gather",
            {"attend_into": attend_into},
            TIMEOUT_PIPELINE_OUTPUT,
            "output",
            True,
        ),
    ):
        state.reset()
        if not ipc_a2a_multi_ready(group, enabled=True):
            failures.append(f"{name}: the transport did not come back")
        run_late(
            name,
            lambda: state.pipelined_attention(q, k, v, attend, 2, **kwargs),
            0 if rank == late else code + late,
            f"{waiting} in the head-group pipeline ({rows} rows)",
            max_budgets=4,
            in_attend=in_attend,
        )

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} MISMATCH: {failures[:8]}", flush=True)
    if rank == 0:
        print(
            f"PEER_TIMEOUT {'FAIL' if verdict.item() else 'PASS'} world={world}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


def _run_ranks(test, world: int, port: int, marker: str, *worker_args: str, **env: str):
    if not current_platform.is_cuda():
        test.skipTest("CUDA-IPC transport is unavailable on this platform")
    if torch.cuda.device_count() < world:
        test.skipTest(f"needs {world} GPUs")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={world}",
            f"--master-port={port}",
            __file__,
            *worker_args,
        ],
        capture_output=True,
        text=True,
        timeout=1200,
        env={**os.environ, **env},
    )
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
    test.assertEqual(proc.returncode, 0, "copy-engine all-to-all diverged")
    if "SKIP" not in proc.stdout:
        test.assertIn(f"{marker} PASS world={world}", proc.stdout)


class TestIpcA2AMultiGpu(CustomTestCase):
    def _run(self, world: int, port: int, *worker_args: str, **env: str):
        _run_ranks(
            self, world, port, "IPC_A2A_MULTI_PARITY", "--worker", *worker_args, **env
        )

    def test_two_ranks_match_sequential_bitwise(self):
        self._run(2, 29527)

    def test_four_ranks_match_sequential_bitwise(self):
        self._run(4, 29537)

    def test_two_ranks_engage_under_expandable_segments_without_pidfd_getfd(self):
        self._run(
            2,
            29547,
            "--deny-pidfd-getfd",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )


class TestPeerTimeoutMultiGpu(CustomTestCase):
    """An expired peer wait fails the request on every rank and names the rank
    it waited for; the bounded waits after it give up at once."""

    def _run(self, world: int, port: int):
        _run_ranks(
            self,
            world,
            port,
            "PEER_TIMEOUT",
            "--timeout-worker",
            SGLANG_DIFFUSION_IPC_A2A_TIMEOUT_MS="300",
        )

    def test_two_ranks(self):
        self._run(2, 62657)

    def test_four_ranks(self):
        self._run(4, 62662)

    def test_eight_ranks(self):
        self._run(8, 62667)


class TestAutoPipelineGroups(CustomTestCase):
    def test_group_counts_at_known_shapes(self):
        """148 SMs: MiniMax-H3 at Ulysses 4 runs 7 two-head groups at 1280x704
        and 1344x768; Wan 1.3B at Ulysses 4 (one head per group) stays sequential."""
        from sglang.multimodal_gen.runtime.distributed.device_communicators import (
            ipc_a2a_multi,
        )

        def groups(rows, heads):
            plan = ipc_a2a_multi._PipelinePlan(
                batch=1,
                s_local=rows // 4,
                heads=(heads,) * 3,
                head_dim=128,
                replicated=(0, 0, 0),
                replicated_first=True,
                world=4,
                groups=-1,
            )
            return ipc_a2a_multi._auto_pipeline_groups(plan)

        with (
            mock.patch.object(ipc_a2a_multi, "_sm_count", return_value=148),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
        ):
            self.assertEqual(groups(33024, 56), 7)
            self.assertEqual(groups(37120, 56), 7)
            self.assertEqual(groups(32768, 12), 0)


class TestPlainExchangeTransport(CustomTestCase):
    def test_unset_switch_picks_the_copy_engine_where_it_measured_faster(self):
        """Unset: three- and four-rank groups on Blackwell from the payload each
        measured ahead at. Set: the switch decides for every exchange."""
        from sglang.multimodal_gen import envs
        from sglang.multimodal_gen.runtime import platforms
        from sglang.multimodal_gen.runtime.distributed.device_communicators import (
            ipc_a2a_multi,
        )

        mib = 1 << 20

        def takes_copy_engine(world, nbytes, switch=None, blackwell=True):
            platform = SimpleNamespace(is_blackwell=lambda: blackwell)
            with (
                mock.patch.object(envs, "SGLANG_DIFFUSION_IPC_A2A_MULTI", switch),
                mock.patch.object(platforms, "current_platform", platform),
                mock.patch.object(
                    ipc_a2a_multi.dist, "get_world_size", return_value=world
                ),
                mock.patch.object(
                    ipc_a2a_multi,
                    "ipc_a2a_multi_ready",
                    side_effect=lambda group, enabled: enabled,
                ),
            ):
                return ipc_a2a_multi.plain_exchange_ready(None, nbytes)

        self.assertTrue(takes_copy_engine(4, 128 * mib))
        self.assertFalse(takes_copy_engine(4, 128 * mib - 1))
        self.assertTrue(takes_copy_engine(3, 48 * mib))
        self.assertFalse(takes_copy_engine(3, 48 * mib - 1))
        # pairs have their own transport; larger groups are not measured
        self.assertFalse(takes_copy_engine(2, 1024 * mib))
        self.assertFalse(takes_copy_engine(8, 1024 * mib))
        self.assertFalse(takes_copy_engine(4, 1024 * mib, blackwell=False))
        self.assertTrue(takes_copy_engine(2, 1, switch=True, blackwell=False))
        self.assertFalse(takes_copy_engine(4, 1024 * mib, switch=False))


if __name__ == "__main__":
    if "--timeout-worker" in sys.argv:
        raise SystemExit(_timeout_worker())
    if "--worker" in sys.argv:
        if "--deny-pidfd-getfd" in sys.argv:
            _deny_pidfd_getfd()
        raise SystemExit(_worker())
    unittest.main()
