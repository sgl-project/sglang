# SPDX-License-Identifier: Apache-2.0
"""SM120 RDMA integration test with real NVFP4 or native MXFP4 CUTLASS experts.

Run with an RDMA-capable FlashInfer/NCCL installation:
    torchrun --standalone --nproc-per-node=4 test/manual/ep/test_flashinfer_rdma.py

Use torchrun's node/rank rendezvous options for multi-node EP. The test derives
its expert group from WORLD_SIZE; local rank only selects the CUDA device.

Set TEST_NATIVE_MXFP4=1 to test native MXFP4 weights with an explicit
SGLANG_FLASHINFER_RDMA_NVFP4_GLOBAL_SCALE (for example, 1.0). Set
SGLANG_FLASHINFER_RDMA_WIRE_DTYPE=bf16 only for the explicit BF16 baseline.

Tests non-unit routing weights, uneven/empty ranks, eager prefill fallback and
CUDA-graph replay with changing inputs, routes and weights. Source-order direct
FlashInfer compute plus explicit routing is independent of dispatch compaction.
"""

import gc
import json
import math
import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist


def _decode_nvfp4(packed, block_scales, global_scale):
    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device
    )
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    values = codebook[(codes & 7).long()] * torch.where(codes & 8 != 0, -1.0, 1.0)
    scales = block_scales.view(torch.float8_e4m3fn).float() * global_scale.reciprocal()
    return values * scales.repeat_interleave(16, dim=-1)


def main():
    ep = int(os.environ.get("WORLD_SIZE", "1"))
    if ep < 2:
        raise ValueError("Run with torchrun and at least two ranks")
    native_mxfp4 = os.environ.get("TEST_NATIVE_MXFP4") == "1"
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    from flashinfer import (
        mxfp4_quantize,
        mxfp8_quantize,
        nvfp4_batched_quantize,
        nvfp4_block_scale_interleave,
    )
    from flashinfer.fused_moe import cutlass_fused_moe
    from flashinfer.fused_moe.core import ActivationType

    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        FlashInferCutlassMoeQuantInfo,
        FlashInferCutlassMxfp4MoeQuantInfo,
        fused_experts_flashinfer_to_flashinfer_cutlass,
        fused_experts_rdma_to_flashinfer_mxfp4,
    )
    from sglang.srt.layers.moe.token_dispatcher import flashinfer_rdma as rdma
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
    from sglang.srt.runtime_context import get_context, get_parallel
    from sglang.srt.server_args import ServerArgs

    get_context().set_server_args(
        ServerArgs(model_path="unused", tp_size=ep, attn_dp_size=ep, ep_size=ep)
    )
    get_parallel().override_permanently(
        tp_rank=rank, attn_dp_rank=rank, moe_ep_rank=rank
    )
    group = GroupCoordinator(
        [list(range(ep))],
        local_rank,
        "nccl",
        True,
        False,
        False,
        False,
        False,
        False,
        group_name="native_rdma_gate",
    )
    get_parallel().override_permanently(tp_group=group, moe_ep_group=group)
    parallel = SimpleNamespace(
        tp_size=ep,
        attn_dp_size=ep,
        attn_tp_size=1,
        attn_cp_size=1,
        attn_dcp_size=1,
        moe_ep_size=ep,
        moe_tp_size=1,
        moe_dp_size=1,
        tp_rank=rank,
        attn_dp_rank=rank,
        moe_ep_rank=rank,
        tp_group=group,
        moe_ep_group=group,
    )
    hidden, experts, intermediate, topk, capacity = (
        (5120, 384, 384, 6, 54) if native_mxfp4 else (4096, 288, 128, 8, 68)
    )
    if experts % ep:
        raise ValueError(
            f"The test's {experts} experts must divide evenly across {ep} ranks"
        )

    def quantize_weights(src):
        if native_mxfp4:
            pairs = [mxfp4_quantize(row) for row in src]
            return tuple(torch.stack([pair[i] for pair in pairs]) for i in range(2))
        return nvfp4_batched_quantize(src, scale.reshape(1))

    local_experts = experts // ep
    scale = torch.tensor(256.0, device=device)
    torch.manual_seed(20261115 + rank)
    src = torch.randn(
        (local_experts, 2 * intermediate, hidden), device=device, dtype=torch.bfloat16
    ) / math.sqrt(hidden)
    w13, sf13 = quantize_weights(src)
    del src
    src = torch.randn(
        (local_experts, hidden, intermediate), device=device, dtype=torch.bfloat16
    ) / math.sqrt(intermediate)
    w2, sf2 = quantize_weights(src)
    del src
    if native_mxfp4:
        weight_global_scale = torch.ones(local_experts, device=device)
        quant = FlashInferCutlassMxfp4MoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_weight_scale=sf13,
            w2_weight_scale=sf2,
            mxfp4_weight_global_scale=weight_global_scale,
        )
        expert_fn = fused_experts_rdma_to_flashinfer_mxfp4
    else:
        sf13 = sf13.view(torch.uint8).reshape(
            local_experts, 2 * intermediate, hidden // 16
        )
        sf2 = sf2.view(torch.uint8).reshape(local_experts, hidden, intermediate // 16)
        alpha = torch.full((local_experts,), 1.0 / 65536.0, device=device)
        quant = FlashInferCutlassMoeQuantInfo(
            "fp4",
            w13.view(torch.uint8),
            w2.view(torch.uint8),
            [scale, sf13, alpha, scale, sf2, alpha],
            torch.bfloat16,
            apply_routed_scaling_factor=False,
        )
        expert_fn = fused_experts_flashinfer_to_flashinfer_cutlass
    config = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=local_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=topk,
        activation="silu",
        is_gated=True,
        routed_scaling_factor=2.5,
        runner_backend=(
            MoeRunnerBackend.FLASHINFER_MXFP4
            if native_mxfp4
            else MoeRunnerBackend.FLASHINFER_CUTLASS
        ),
        layer=SimpleNamespace(
            moe_tp_size=1, moe_tp_rank=0, moe_ep_size=ep, moe_ep_rank=rank
        ),
    )
    with patch.object(rdma, "get_parallel", return_value=parallel):
        dispatcher = rdma.FlashinferRdmaDispatcher(
            group.device_group,
            topk,
            experts,
            local_experts,
            hidden,
            config,
            capacity,
            torch.bfloat16,
        )
    dispatcher.set_quant_config({} if native_mxfp4 else {"input_global_scale": scale})
    direct_scales = (
        [
            sf13.view(torch.int32),
            weight_global_scale,
            sf2.view(torch.int32),
            weight_global_scale,
        ]
        if native_mxfp4
        else [scale, sf13.view(torch.int32), alpha, scale, sf2.view(torch.int32), alpha]
    )

    def direct(x, sf, ids, weights):
        out = torch.zeros((x.shape[0], hidden), device=device, dtype=torch.bfloat16)
        if x.shape[0]:
            if native_mxfp4:
                x, sf = mxfp8_quantize(x, is_sf_swizzled_layout=True, alignment=32)
            cutlass_fused_moe(
                output=out,
                input=x,
                token_selected_experts=torch.where(ids < 0, experts, ids),
                token_final_scales=weights,
                fc1_expert_weights=w13.view(torch.long),
                fc2_expert_weights=w2.view(torch.long),
                output_dtype=torch.bfloat16,
                input_sf=sf,
                quant_scales=direct_scales,
                ep_size=ep,
                ep_rank=rank,
                tp_size=1,
                tp_rank=0,
                tune_max_num_tokens=4096,
                activation_type=ActivationType.Swiglu,
                enable_alltoall=False,
                use_mxfp8_act_scaling=native_mxfp4,
                use_fused_finalize=True,
            )
            dist.all_reduce(out, group=group.device_group)
        return out

    def oracle(global_x, routes, weights, decode):
        n = global_x.shape[0]
        if not n:
            return torch.empty((0, hidden), device=device, dtype=torch.bfloat16)
        if not decode:
            return direct(global_x, None, routes, weights)
        if native_mxfp4:
            if dispatcher.wire_dtype == "nvfp4":
                globals_ = torch.full(
                    (n, 1), dispatcher.wire_global_scale, device=device
                )
                wire, blocks = dispatcher.state.buffer.quantize_bf16_to_nvfp4(
                    global_x, globals_
                )
                x = _decode_nvfp4(wire, blocks, globals_).to(torch.bfloat16)
            else:
                x = global_x
            q, sf = x.repeat_interleave(topk, dim=0), None
        else:
            q, sf = fp4_quantize(global_x, scale, is_sf_swizzled_layout=False)
            q = q.repeat_interleave(topk, dim=0)
            sf = nvfp4_block_scale_interleave(sf.repeat_interleave(topk, dim=0))
        expert = direct(
            q, sf, routes.reshape(-1, 1), torch.ones((n * topk, 1), device=device)
        )
        if dispatcher.wire_dtype == "nvfp4":
            amax = expert.float().abs().amax(-1, keepdim=True)
            output_scale = torch.where(amax > 0, 2688.0 / amax.clamp_min(1e-20), 1.0)
            wire, blocks = dispatcher.state.buffer.quantize_bf16_to_nvfp4(
                expert, output_scale
            )
            decoded = _decode_nvfp4(wire, blocks, output_scale)
        else:
            decoded = expert.float()
        expected = torch.zeros((n, hidden), device=device)
        for k in range(topk):
            expected = torch.addcmul(
                expected,
                decoded.reshape(n, topk, hidden)[:, k],
                weights[:, k, None] * (routes[:, k] >= 0)[:, None],
            )
        return expected.to(torch.bfloat16)

    def check(actual, expected, name, phase):
        torch.cuda.synchronize()
        error = (
            float((actual.float() - expected.float()).abs().max())
            if actual.numel()
            else 0.0
        )
        if not torch.isfinite(actual).all():
            raise AssertionError("non-finite output")
        # Compacted and source-order grouped GEMMs can select different tactics;
        # communication is additionally NVFP4. Bound ordinary rounding while
        # the non-unit routing weights reject duplicate or missing weighting.
        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.006)
        gathered = [None] * ep
        dist.all_gather_object(gathered, error)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "case": name,
                        "phase": phase,
                        "max_abs_error": max(gathered),
                        "real_cutlass": True,
                        "native_dispatcher": True,
                        "ep_size": ep,
                        "weight_dtype": "mxfp4" if native_mxfp4 else "nvfp4",
                        "wire_dtype": dispatcher.wire_dtype,
                        "wire_global_scale": dispatcher.wire_global_scale,
                    }
                ),
                flush=True,
            )

    try:
        for name, sizes in [
            ("uniform", [8] * ep),
            ("uneven_empty_rank", [(0, 1, 5, 8)[r % 4] for r in range(ep)]),
            ("all_invalid", [8] * ep),
            ("all_empty", [0] * ep),
        ]:
            torch.manual_seed(1234)
            all_x = torch.randn(
                (sum(sizes), hidden), device=device, dtype=torch.bfloat16
            )
            logits = torch.rand((sum(sizes), experts), device=device)
            all_ids = logits.topk(topk, dim=-1).indices.to(torch.int32)
            all_weights = (
                torch.rand((sum(sizes), topk), device=device).softmax(-1) * 2.5
            )
            all_weights[:, 0] = 0
            if name == "all_invalid":
                all_ids.fill_(-1)
            begin, end = sum(sizes[:rank]), sum(sizes[: rank + 1])
            x, ids, weights = (
                all_x[begin:end].clone(),
                all_ids[begin:end].clone(),
                all_weights[begin:end].clone(),
            )
            top = StandardTopKOutput(weights, ids, None)

            def cycle():
                d = dispatcher.dispatch(x, top)
                y = expert_fn(d, quant, config)
                return dispatcher.combine(y)

            with (
                patch.object(rdma, "get_is_extend_in_batch", return_value=False),
                patch(
                    "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._cutlass_moe_tune_max_num_tokens",
                    return_value=4096,
                ),
            ):
                gold = oracle(all_x, all_ids, all_weights, True)[begin:end]
                actual = cycle()
                check(actual, gold, name, "decode_eager")
                cycle()
                torch.cuda.synchronize()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = cycle()
                torch.cuda.current_stream().wait_stream(stream)
                for phase in range(2):
                    all_x.neg_()
                    x.neg_()
                    all_weights.mul_(0.75)
                    weights.mul_(0.75)
                    if name not in ("all_invalid", "all_empty"):
                        all_ids.add_(1).remainder_(experts)
                        ids.copy_(all_ids[begin:end])
                    gold = oracle(all_x, all_ids, all_weights, True)[begin:end]
                    graph.replay()
                    check(captured, gold, name, f"decode_graph_{phase}")
                graph.reset()
                del graph, captured
            if sum(sizes):
                with (
                    patch.object(rdma, "get_is_extend_in_batch", return_value=True),
                    patch.object(rdma, "get_dp_global_num_tokens", return_value=sizes),
                    patch(
                        "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._cutlass_moe_tune_max_num_tokens",
                        return_value=4096,
                    ),
                ):
                    gold = oracle(all_x, all_ids, all_weights, False)[begin:end]
                    actual = cycle()
                    check(actual, gold, name, "prefill_eager")
        if rank == 0:
            print("NATIVE_RDMA_CUTLASS_GATE_PASS", flush=True)
    finally:
        torch.cuda.synchronize()
        gc.collect()
        rdma.close_flashinfer_rdma_buffers()
        group.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
