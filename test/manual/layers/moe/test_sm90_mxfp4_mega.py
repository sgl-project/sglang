# Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""SGLang SM90 MegaMoE warmup and workspace reuse against a dequantized oracle.

Example (choose two idle Hopper GPUs):
    CUDA_VISIBLE_DEVICES=1,2 NVSHMEM_SYMMETRIC_SIZE=256M NCCL_DEBUG=WARN \
      timeout 420s torchrun --standalone --nproc-per-node=2 \
      test/manual/layers/moe/test_sm90_mxfp4_mega.py

Requires the CP FlashInfer fused staging/conversion operators. This exercises
SGLang startup and steady-state execution with real NVSHMEM peer buffers.
Two routes and a repeat check workspace and counter reuse. The independent oracle decodes packed E2M1,
performs FP32 matmul, SwiGLU, router weighting, K64 FP8 quantization and summation;
it imports no implementation helpers. Tolerances allow native WGMMA rounding.
"""

import json
import os


def run_ep_oracle():
    from types import SimpleNamespace
    from unittest.mock import patch

    import nvshmem.core
    import torch
    import torch.distributed as dist
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.weights import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.weights import PrequantizedMoEWeights

    from sglang.srt.layers.moe import sm90_mxfp4_mega as runtime
    from sglang.srt.runtime_context import get_context
    from sglang.srt.server_args import ServerArgs

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2, "Run this test with torchrun --nproc-per-node=2"
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    device_group = dist.new_group(backend="nccl")
    group = SimpleNamespace(device_group=device_group, cpu_group=dist.group.WORLD)
    get_context().set_server_args(
        ServerArgs(
            model_path="dummy",
            tp_size=world,
            ep_size=world,
            attn_cp_size=world,
            chunked_prefill_size=512,
            max_prefill_tokens=512,
            max_running_requests=1,
            disable_flashinfer_autotune=True,
        )
    )
    E, H, I, T, K = 4, 512, 256, 8, 4
    LOCAL = E // world
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])

    def raw_weights(owner):
        g = torch.Generator().manual_seed(397 + owner)
        a = torch.randint(
            0, 256, (LOCAL, 2 * I, H // 2), generator=g, dtype=torch.uint8
        )
        b = torch.randint(0, 256, (LOCAL, H, I // 2), generator=g, dtype=torch.uint8)
        return a, b

    def dequant(a):
        code = torch.stack((a & 15, a >> 4), -1).flatten(-2).long()
        return lut[code] * (2.0**-6)

    wa, wb = raw_weights(rank)
    pack = PrequantizedMoEWeights(
        wa.cuda(),
        wb.cuda(),
        torch.full((LOCAL, 2 * I, H // 32), 121, dtype=torch.uint8, device="cuda"),
        torch.full((LOCAL, H, I // 32), 121, dtype=torch.uint8, device="cuda"),
    )
    l1, l2 = preprocess_mega_weights(pack, hidden_size=H, intermediate_size=I)
    global_w = [tuple(dequant(w) for w in raw_weights(r)) for r in range(world)]

    def inputs(launch):
        g = torch.Generator().manual_seed(125 + 97 * launch + rank)
        x = (torch.randn(T, H, generator=g) * 0.25).bfloat16()
        ids = (torch.arange(T)[:, None] + torch.arange(K)[None, :] + rank + launch) % E
        p = torch.tensor([0.125, 0.25, 0.25, 0.375]).expand(T, K).contiguous()
        return x, ids, p

    def reference(x, ids, p):
        # Independent dequantized matmul + router + SwiGLU + K64 FP8 quantization.
        xf = x.float()
        scale = (xf.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
        xq = (xf / scale).to(torch.float8_e4m3fn).float() * scale
        terms = torch.empty(x.shape[0], K, H)
        for i in range(x.shape[0]):
            for j in range(K):
                e = int(ids[i, j])
                owner, local = divmod(e, LOCAL)
                a, b = global_w[owner]
                first = xq[i] @ a[local].T
                gate, up = first[:I], first[I:]
                h = torch.nn.functional.silu(gate) * up * p[i, j]
                groups = h.reshape(I // 64, 64)
                sf = (groups.abs().amax(-1, keepdim=True) / 448).clamp_min(1e-30)
                hq = (groups / sf).to(torch.float8_e4m3fn).float() * sf
                terms[i, j] = (hq.flatten() @ b[local].T).bfloat16().float()
        return terms.sum(1)

    results = []
    experts = SimpleNamespace(
        hidden_size=H,
        intermediate_size_per_partition=I,
        num_experts=E,
        top_k=K,
        mega_l1_weights=l1,
        mega_l2_weights=l2,
    )
    try:
        with patch("sglang.srt.distributed.get_moe_ep_group", return_value=group):
            runtime.warmup_experts(
                experts,
                device=torch.device("cuda", rank),
                dtype=torch.bfloat16,
                context_len=512,
            )
        prev = None
        for iteration in range(3):
            x, ids, p = inputs(min(iteration, 1))
            x_device = x.cuda()
            workspace = runtime.prepare_workspace(
                experts, x_device, num_experts=E, top_k=K
            )
            actual = (
                runtime.run_experts(
                    experts,
                    x_device,
                    SimpleNamespace(topk_ids=ids.cuda(), topk_weights=p.cuda()),
                    workspace,
                )
                .float()
                .cpu()
            )
            torch.cuda.synchronize()
            dist.barrier()
            expected = reference(x, ids, p)
            err = actual - expected
            rel = (err.norm() / expected.norm()).item()
            torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)
            assert rel < 0.025, (rank, iteration, rel)
            if iteration == 2:
                assert torch.equal(actual, prev)
            prev = actual.clone()
            result = dict(
                rank=rank,
                iteration=iteration,
                relative_l2=rel,
                max_abs=err.abs().max().item(),
            )
            results.append(result)
            print("FORWARD_PASS", json.dumps(result), flush=True)
        # Exercise many layers without host synchronization between NCCL halo
        # exchange and MegaMoE. Empty CP peers still participate in every call.
        x, ids, probabilities = inputs(1)
        rows = 1 if rank == 0 else 0
        x, ids, probabilities = x[:rows], ids[:rows], probabilities[:rows]
        x_device, ids_device, probabilities_device = (
            x.cuda(),
            ids.cuda(),
            probabilities.cuda(),
        )
        halo = torch.full((12288, 3), float(rank), device="cuda", dtype=torch.bfloat16)
        queued = []
        for layer in range(64):
            tails = [torch.empty_like(halo) for _ in range(world)]
            dist.all_gather(tails, halo, group=device_group)
            workspace = runtime.prepare_workspace(
                experts, x_device, num_experts=E, top_k=K
            )
            queued.append(
                runtime.run_experts(
                    experts,
                    x_device,
                    SimpleNamespace(
                        topk_ids=ids_device, topk_weights=probabilities_device
                    ),
                    workspace,
                ).clone()
            )
        print("RAGGED_LAYERS_QUEUED", rank, flush=True)
        torch.cuda.synchronize()
        expected = reference(x, ids, probabilities)
        for output in queued:
            torch.testing.assert_close(
                output.float().cpu(), expected, atol=0.002, rtol=0.03
            )
        print("RAGGED_LAYERS_COMPLETE", rank, len(queued), flush=True)
        print("EP2_COMPLETE", rank, flush=True)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        for buf in runtime._BUFFERS.values():
            buf.destroy()
        runtime._BUFFERS.clear()
        nvshmem.core.finalize()
        dist.destroy_process_group()


def test_ep_oracle():
    import pytest

    if os.environ.get("WORLD_SIZE") != "2":
        pytest.skip("Requires torchrun with two Hopper GPUs")
    run_ep_oracle()


if __name__ == "__main__":
    run_ep_oracle()
