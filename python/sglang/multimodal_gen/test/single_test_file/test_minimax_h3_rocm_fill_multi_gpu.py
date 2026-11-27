"""MiniMax-H3 attention with the QK RMSNorm + RoPE filled into the ROCm
copy-engine exchange gives the bytes of the norm kernel followed by the
copy-engine pipeline, at the model's video attention shape on 4 ranks.

Needs 4 ROCm GPUs with peer-to-peer access:

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_minimax_h3_rocm_fill_multi_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase

_WORLD = 4


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
    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.distributed.device_communicators import (
        ipc_a2a_rocm as ce,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends import aiter as backend
    from sglang.multimodal_gen.runtime.models.dits import minimax_h3 as h3
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    if not all(
        torch.cuda.can_device_access_peer(rank, d) for d in range(world) if d != rank
    ):
        print("SKIP no peer-to-peer access between the devices", flush=True)
        return 0

    # 9440 rows per rank, 37757 live of 37760, 56 heads of 128, 96 RoPE dims
    rows, heads, dim, hidden, used = 9440, 56, 128, 5376, 37757
    total = world * rows
    os.environ["SGLANG_DIFFUSION_ROCM_CE_A2A"] = "1"
    envs.SGLANG_DIFFUSION_ULYSSES_PIPELINE_GROUPS = 7
    backend._TAIL_SPLIT_HEADS = 7
    torch.manual_seed(1000 + rank)
    gen = torch.Generator(device="cuda").manual_seed(5)  # weights alike on every rank

    def rms_norm():
        norm = torch.nn.RMSNorm(dim, eps=1e-6, device="cuda", dtype=torch.bfloat16)
        norm.weight.data.copy_(1 + 0.1 * torch.randn(dim, device="cuda", generator=gen))
        return norm

    attention = SimpleNamespace(
        q_norm=rms_norm(),
        k_norm=rms_norm(),
        head_dim=dim,
        num_heads=heads,
        _use_fused_qknorm_rope=False,
        rotary_emb=None,
        _attention_backend_enum=AttentionBackendEnum.AITER,
        _attention_impl=backend.AITerImpl(
            num_heads=heads // world,
            head_size=dim,
            softmax_scale=dim**-0.5,
            packed_trailing_padding=True,
        ),
        to_gate_compress=None,
        bcg_breakpoint=False,
        prefix="blocks.0.attn",
    )
    inv = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, device="cuda") / 32))
    idx = torch.arange(rank * rows, (rank + 1) * rows, device="cuda")
    pos = torch.stack((idx // (34 * 60), (idx // 60) % 34, idx % 60), dim=-1).float()
    half = (pos[..., None] * inv).flatten(1)
    rope_cache = (
        h3._rope_cos_sin_cache(torch.cat((half, half), dim=-1), dtype=torch.bfloat16),
        torch.arange(rows, device="cuda"),
    )
    bounds = (0, used, total)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32, device="cuda")
    w_qkv = (
        torch.randn(hidden, 3 * heads * dim, device="cuda", generator=gen) * 0.02
    ).bfloat16()

    def attend(x, filled):
        os.environ["SGLANG_DIFFUSION_ROCM_CE_FILL"] = "1" if filled else "0"
        # q/k/v are views of the fused projection, as in the model
        qkv = torch.mm(x, w_qkv)
        q, k, v = (t.view(rows, heads, dim) for t in qkv.split(heads * dim, dim=-1))
        return h3.MiniMaxH3Attention._attend(
            attention,
            x,
            q,
            k,
            v,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=bounds,
            max_seqlen=used,
            subblock_sparse_query_block_mask=None,
            ulysses_active=True,
            ring_active=False,
        )

    failures = []
    xs = [torch.randn(rows, hidden, device="cuda").bfloat16() for _ in range(4)]
    variants = {
        "one fill": {},
        "two in flight": {"_MAX_CONC": 2},
        "group 0 first": {"_FILL_FIRST": True},
        "merge per group": {"_GROUP_MERGE": True},
        "merged output": {"_OUT_MERGED": True},
    }
    with torch.inference_mode():
        attend(xs[0], False)  # the first sight verifies the norm kernel
        if not h3._ATEN_HEAD_RMSNORM_ROPE.verified:
            failures.append("the norm kernel the fill replicates was not verified")
        refs = [attend(x, False).clone() for x in xs]
        for name, knobs in variants.items():
            for key, value in knobs.items():
                setattr(ce, key, value)
            outs = [attend(xs[i % 4], True).clone() for i in range(8)]  # no host sync
            for key in knobs:
                setattr(ce, key, {"_MAX_CONC": 0}.get(key, False))
            bad = [i for i, out in enumerate(outs) if not torch.equal(out, refs[i % 4])]
            if bad:
                failures.append(f"{name}: calls {bad} differ")
    os.environ["SGLANG_DIFFUSION_ROCM_CE_FILL"] = "0"
    # a fill that quietly fell back would pass while testing nothing
    pool = ce.IPC_A2A_ROCM.pools.get(torch.bfloat16)
    if pool is None or pool.send is None:
        failures.append("the fill never ran")

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} MISMATCH: {failures}", flush=True)
    if rank == 0:
        print(f"ROCM_FILL_PARITY {'FAIL' if verdict.item() else 'PASS'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestMiniMaxH3RocmFillMultiGpu(CustomTestCase):
    def test_four_ranks_match_the_unfilled_path_bitwise(self):
        if torch.version.hip is None:
            self.skipTest("the copy-engine fill is ROCm-only")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={_WORLD}",
                "--master-port=29597",
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
        self.assertEqual(proc.returncode, 0, "the filled exchange diverged")
        if "SKIP" not in proc.stdout:
            self.assertIn("ROCM_FILL_PARITY PASS", proc.stdout)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()
