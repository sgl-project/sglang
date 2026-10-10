"""Every Ulysses path of USPAttention gives the same bytes with the exchange
pipelined over head groups as with the sequential exchange.

The pipeline only engages on a same-host Ulysses group with peer-to-peer access,
so nothing in the single-GPU suite reaches it. Runs on 2 ranks, and on 4 when
four GPUs are visible:

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_usp_pipelined_attention_multi_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase


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
    )
    from sglang.multimodal_gen.runtime.layers.attention.layer import (
        PIPELINED_ATTENTION_BACKENDS,
        USPAttention,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args

    set_global_server_args(_make_unit_server_args())
    peers = [d for d in range(world) if d != rank]
    if not all(torch.cuda.can_device_access_peer(rank, d) for d in peers):
        print("SKIP no peer-to-peer access between the devices", flush=True)
        return 0

    heads, kv_heads, dim, s_local, n_rep = 16, 8, 64, 96, 24
    failures, ran = [], []
    backends = [b for b in PIPELINED_ATTENTION_BACKENDS]

    def shard(batch, rows, h, seed):
        torch.manual_seed(seed)  # identical full tensors on every rank
        full = torch.randn(batch, rows * world, h, dim, device="cuda").bfloat16()
        return full[:, rank * rows : (rank + 1) * rows].contiguous()

    def replicated(batch, rows, h, seed):
        torch.manual_seed(seed)
        return torch.randn(batch, rows, h, dim, device="cuda").bfloat16()

    def both(label, call):
        """`call()` sequentially and pipelined (two forced groups) must agree."""
        outs = []
        for groups in (0, 2):
            envs.SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS = groups
            with torch.inference_mode(), set_forward_context(0, None, None):
                outs.append(call())
        envs.SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS = -1
        if not torch.equal(outs[0], outs[1]):
            failures.append(label)
        ran.append(label)

    for backend in backends:
        try:
            layer = USPAttention(
                num_heads=heads,
                head_size=dim,
                causal=False,
                required_attention_backend=backend,
                prefix=f"test.{backend.name}",
            )
        except Exception as e:  # noqa: BLE001 - a backend this GPU cannot build
            print(f"backend {backend.name} unavailable here: {e}", flush=True)
            continue
        if layer.backend is not backend:
            print(
                f"backend {backend.name} resolved to {layer.backend.name}", flush=True
            )
            continue
        name = backend.name
        for batch in (1, 2):
            q, k, v = (shard(batch, s_local, heads, 10 + i) for i in range(3))
            both(f"{name} dense b{batch}", lambda: layer(q, k, v))

            rep = [replicated(batch, n_rep, heads, 20 + i) for i in range(3)]
            joined = [torch.cat([r, t], dim=1) for r, t in zip(rep, (q, k, v))]
            both(
                f"{name} replicated prefix b{batch}",
                lambda: layer(*joined, num_replicated_prefix=n_rep),
            )
            mask = torch.ones(batch, n_rep + s_local, dtype=torch.bool, device="cuda")
            mask[:, n_rep // 2 : n_rep] = False  # padded text tokens
            both(
                f"{name} replicated prefix + key mask b{batch}",
                lambda: layer(*joined, num_replicated_prefix=n_rep, attn_mask=mask),
            )
            trailing = [torch.cat([t, r], dim=1) for r, t in zip(rep, (q, k, v))]
            both(
                f"{name} replicated suffix b{batch}",
                lambda: layer(*trailing, num_replicated_suffix=n_rep),
            )
            k_rep, v_rep = (replicated(batch, n_rep, heads, 30 + i) for i in range(2))
            both(
                f"{name} replicated KV prefix b{batch}",
                lambda: layer.forward_with_replicated_kv_prefix(q, k_rep, v_rep, k, v),
            )
            key_mask = torch.ones(batch, s_local, dtype=torch.bool, device="cuda")
            key_mask[:, -s_local // 4 :] = rank % 2 == 0  # ragged across ranks
            both(
                f"{name} gathered key mask b{batch}",
                lambda: layer(q, k, v, attn_mask=key_mask),
            )
        gqa = USPAttention(
            num_heads=heads,
            head_size=dim,
            num_kv_heads=kv_heads,
            causal=False,
            required_attention_backend=backend,
            prefix=f"test.gqa.{backend.name}",
        )
        q = shard(1, s_local, heads, 40)
        k, v = (shard(1, s_local, kv_heads, 41 + i) for i in range(2))
        both(f"{name} GQA dense", lambda: gqa(q, k, v))

    # a comparison where every arm quietly fell back would pass while testing
    # nothing: the pipeline must have run and verified every layout
    if not ran:
        failures.append("no pipelined backend could be built")
    if IPC_A2A_MULTI.mismatched:
        failures.append(
            f"pipeline disagreed with the sequential exchange: {IPC_A2A_MULTI.mismatched}"
        )
    if len(IPC_A2A_MULTI.verified) < len({label.split(" b")[0] for label in ran}):
        failures.append(
            f"pipeline verified {len(IPC_A2A_MULTI.verified)} layouts for {len(ran)} cases"
        )

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} MISMATCH: {failures}", flush=True)
    if rank == 0:
        print(f"ran: {ran}", flush=True)
        print(
            f"USP_PIPELINE_PARITY {'FAIL' if verdict.item() else 'PASS'} world={world}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestUspPipelinedAttentionMultiGpu(CustomTestCase):
    def _run(self, world: int, port: int):
        if not current_platform.is_cuda():
            self.skipTest("the copy-engine pipeline is CUDA-only")
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
        self.assertEqual(proc.returncode, 0, "pipelined USPAttention diverged")
        if "SKIP" not in proc.stdout:
            self.assertIn(f"USP_PIPELINE_PARITY PASS world={world}", proc.stdout)

    def test_two_ranks_match_sequential_bitwise(self):
        self._run(2, 29547)

    def test_four_ranks_match_sequential_bitwise(self):
        self._run(4, 29557)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()
