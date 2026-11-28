"""The ROCm copy-engine Ulysses head-group pipeline gives the bytes of the
RCCL head-group pipeline and of the sequential packed exchange.

Needs ROCm GPUs with peer-to-peer access; runs on 2 ranks, and on 4 (also at
MiniMax-H3's attention shape with AITer) when four GPUs are visible:

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_ipc_a2a_rocm_multi_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

_ENV = "SGLANG_DIFFUSION_ROCM_CE_A2A"


def _worker() -> int:
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1, sp_size=world, ulysses_degree=world
    )
    from sglang.multimodal_gen.runtime.distributed.device_communicators import (
        ipc_a2a_rocm as ce,
    )
    from sglang.multimodal_gen.runtime.layers import usp

    if not all(
        torch.cuda.can_device_access_peer(rank, d) for d in range(world) if d != rank
    ):
        print("SKIP no peer-to-peer access between the devices", flush=True)
        return 0

    heads, dim = 56, 128
    failures, ran = [], []

    def cumsum_attend(q, k, v):
        # head-independent but row-order dependent, like attention
        return torch.cumsum(q.float() * k.flip(0).float(), dim=0).to(q.dtype) + v

    def shards(rows, seed):
        torch.manual_seed(seed + rank)
        qkv = torch.randn(rows, 3 * heads * dim, device="cuda").bfloat16()
        # q/k/v are views of a fused projection, as in MiniMax-H3
        return [t.view(rows, heads, dim) for t in qkv.split(heads * dim, dim=-1)]

    def pipelined(copy_engine, q, k, v, attend, groups):
        os.environ[_ENV] = "1" if copy_engine else "0"
        with torch.inference_mode():
            out = usp.ulysses_head_group_pipelined_attention(q, k, v, attend, groups)
        os.environ[_ENV] = "0"
        return out.clone()

    def check(label, got, want):
        ran.append(label)
        if not torch.equal(got, want):
            failures.append(label)

    for rows in (37, 1024):
        q, k, v = shards(rows, 10 * rows)
        with torch.inference_mode():
            gathered = usp._usp_input_all_to_all_packed_qkv(q, k, v)
            out = cumsum_attend(*gathered)[None]
            sequential = usp._usp_output_all_to_all(out, head_dim=2)[0]
        for groups in (2, 7):
            rccl = pipelined(False, q, k, v, cumsum_attend, groups)
            check(f"rccl {rows} rows {groups} groups", rccl, sequential)
            for group_pack in (False, True):
                ce._GROUP_PACK = group_pack
                got = pipelined(True, q, k, v, cumsum_attend, groups)
                check(f"ce {rows} rows {groups} groups pack={group_pack}", got, rccl)
        ce._GROUP_PACK = False

    if world == 4:
        # MiniMax-H3 video: 9440 rows per rank, 37757 live of 37760; the 2-head
        # calls of 7 groups plan their tail split for the 7-head calls
        from sglang.multimodal_gen.runtime.layers.attention.backends import (
            aiter as backend,
        )

        rows, used = 9440, 37757
        bounds = (0, used, world * rows)
        cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device="cuda")
        impl = backend.AITerImpl(
            num_heads=heads // world,
            head_size=dim,
            softmax_scale=dim**-0.5,
            packed_trailing_padding=True,
        )

        def aiter_attend(q, k, v):
            return impl.forward_varlen(
                q,
                k,
                v,
                cu_seqlens=cu_seqlens,
                max_seqlen=used,
                cu_seqlens_host=bounds,
            )

        backend._TAIL_SPLIT_HEADS = 7
        inputs = [shards(rows, 100 + i) for i in range(4)]
        refs = [pipelined(False, *qkv, aiter_attend, 2) for qkv in inputs]
        for groups in (2, 7):
            got = pipelined(True, *inputs[0], aiter_attend, groups)
            check(f"ce aiter {groups} groups vs rccl 2 groups", got, refs[0])
        # back to back without a host sync: the slots are reused across calls
        os.environ[_ENV] = "1"
        with torch.inference_mode():
            outs = [
                usp.ulysses_head_group_pipelined_attention(
                    *inputs[i % 4], aiter_attend, 7
                ).clone()
                for i in range(12)
            ]
        os.environ[_ENV] = "0"
        for i, out in enumerate(outs):
            check(f"ce aiter back-to-back call {i}", out, refs[i % 4])
        backend._TAIL_SPLIT_HEADS = 0

    # a comparison that quietly fell back to RCCL would pass while testing nothing
    if not (ce.IPC_A2A_ROCM.inited and ce.IPC_A2A_ROCM.pools):
        failures.append("the copy-engine pipeline never ran")

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} MISMATCH: {failures}", flush=True)
    if rank == 0:
        print(f"ran: {ran}", flush=True)
        print(
            f"ROCM_CE_PARITY {'FAIL' if verdict.item() else 'PASS'} world={world}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestIpcA2ARocmMultiGpu(CustomTestCase):
    def _run(self, world: int, port: int):
        if torch.version.hip is None:
            self.skipTest("the copy-engine pipeline is ROCm-only")
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
        self.assertEqual(proc.returncode, 0, "copy-engine pipeline diverged")
        if "SKIP" not in proc.stdout:
            self.assertIn(f"ROCM_CE_PARITY PASS world={world}", proc.stdout)

    def test_two_ranks_match_rccl_bitwise(self):
        self._run(2, 29577)

    def test_four_ranks_match_rccl_bitwise(self):
        self._run(4, 29587)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()
