# SPDX-License-Identifier: Apache-2.0
"""SM120 RDMA integration test with real NVFP4 or native MXFP4 CUTLASS experts.

Run with an RDMA-capable FlashInfer/NCCL installation:
    torchrun --standalone --nproc-per-node=4 test/manual/ep/test_flashinfer_rdma.py

Use torchrun's node/rank rendezvous options for multi-node EP. The test derives
its expert group from WORLD_SIZE; local rank only selects the CUDA device.

Set TEST_NATIVE_MXFP4=1 to test native MXFP4 weights with an explicit
SGLANG_FLASHINFER_RDMA_NVFP4_GLOBAL_SCALE (for example, 1.0). Set
SGLANG_FLASHINFER_RDMA_WIRE_DTYPE=bf16 for BF16 combine; set
SGLANG_FLASHINFER_RDMA_DISPATCH_DTYPE=mxfp8 for native MXFP8 raw dispatch.
Without the override, WIRE_DTYPE controls both directions.

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


def _check_bf16_sum(actual, partials):
    """Check BF16 sum against independently gathered inputs, not another NCCL sum.

    Any reduction tree of n BF16 inputs uses at most n-1 rounded additions.
    With u=2^-8, |fl(sum)-sum| <= gamma_(n-1) * sum(abs(inputs)).
    The final BF16 store is part of those additions, not an extra tolerance.
    """
    assert actual.dtype == partials.dtype == torch.bfloat16
    assert partials.shape[1:] == actual.shape and partials.shape[0] >= 2
    assert bool(torch.isfinite(actual).all() and torch.isfinite(partials).all())
    n = partials.shape[0]
    u = 2.0**-8
    assert (n - 1) * u < 1
    wide = partials.double()
    reference = wide.sum(0)
    absolute_sum = wide.abs().sum(0)
    infinity = torch.full_like(absolute_sum, float("inf"))

    def upward(value):
        return torch.nextafter(value, infinity)

    gamma = math.nextafter(((n - 1) * u) / (1 - (n - 1) * u), math.inf)
    u64 = 2.0**-53
    gamma64 = math.nextafter(((n - 1) * u64) / (1 - (n - 1) * u64), math.inf)
    # FP64 sum is an independent, high-precision reference, not assumed exact.
    # Recover an upper bound on the true sum(abs(inputs)), then cover both the
    # BF16 reduction and FP64 reference rounding. Every budget operation rounds
    # outward; there is no fixed epsilon, atol or extra BF16 final-cast budget.
    denominator = math.nextafter(1.0 - gamma64, -math.inf)
    absolute_sum_upper = upward(absolute_sum / denominator)
    bound = upward(
        upward(gamma * absolute_sum_upper) + upward(gamma64 * absolute_sum_upper)
    )
    error = (actual.double() - reference).abs()
    # The subtraction also rounds in FP64. Its next lower value cannot exceed
    # the true difference, so this comparison does not reject valid roundoff.
    error_lower = torch.nextafter(error, -infinity).clamp_min(0)
    if bool((error_lower > bound).any()):
        raise AssertionError(
            "BF16 reduction exceeds its input-dependent roundoff bound"
        )
    return float(error.max()) if error.numel() else 0.0


def _check_bf16_sum_negative_controls(ep):
    # Disjoint rank signatures and row powers of two: each nonzero output is
    # exactly representable. Wrong owners/rows and dropped/duplicated inputs
    # change a signature by much more than gamma*sumabs, even at EP32.
    rows = 2 * ep
    parts = torch.zeros((ep, rows, ep), dtype=torch.bfloat16)
    for rank in range(ep):
        parts[rank, :, rank] = torch.pow(2.0, torch.arange(rows) % 16 + rank % 16).to(
            torch.bfloat16
        )
    exact = parts.double().sum(0).to(torch.bfloat16)
    _check_bf16_sum(exact, parts)
    corruptions = {
        # Two consecutive rows belong to one source owner in this fixture.
        "wrong_owner": exact.roll(2, 0),
        "wrong_row": exact.roll(1, 0),
        "missing_rank": (exact.double() - parts[ep // 2].double()).to(torch.bfloat16),
        "duplicate_rank": (exact.double() + parts[ep // 2].double()).to(torch.bfloat16),
    }
    for name, corrupted in corruptions.items():
        try:
            _check_bf16_sum(corrupted, parts)
        except AssertionError:
            continue
        raise AssertionError(f"BF16 sum negative control was accepted: {name}")


def main():
    ep = int(os.environ.get("WORLD_SIZE", "1"))
    if ep < 2:
        raise ValueError("Run with torchrun and at least two ranks")
    _check_bf16_sum_negative_controls(ep)
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
    from sglang.srt.environ import envs
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
    capacity = int(os.environ.get("TEST_CAPACITY", capacity))
    intermediate = int(os.environ.get("TEST_INTERMEDIATE", intermediate))
    tokens = int(os.environ.get("TEST_TOKENS", "8"))
    if not 1 <= tokens <= capacity:
        raise ValueError("TEST_TOKENS must be positive and fit TEST_CAPACITY")
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

    def direct(x, sf, ids, weights, *, reduce=True):
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
                use_fused_finalize=envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get(),
            )
            if reduce:
                dist.all_reduce(out, group=group.device_group)
        return out

    def decode_oracle(global_x, routes, weights):
        n = global_x.shape[0]
        if not n:
            return torch.empty((0, hidden), device=device, dtype=torch.bfloat16)
        if native_mxfp4:
            if dispatcher.dispatch_dtype == "nvfp4":
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
        if dispatcher.combine_dtype == "nvfp4":
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

    latest_dispatch = None
    latest_partial = None

    def check_mxfp8_bytes(global_x, sizes):
        if dispatcher.dispatch_dtype != "mxfp8":
            return
        state = dispatcher.state
        counts = state.outputs[2].cpu().tolist()
        source_ids = state.outputs[3].cpu()
        ranges = state.outputs[4].cpu()
        reference = torch.zeros_like(state.packed, dtype=torch.bfloat16)
        offset = 0
        for expert, count in enumerate(counts):
            for source in range(ep):
                encoded = int(ranges[expert, source])
                length, begin = encoded & 0xFFFFFFFF, encoded >> 32
                if length:
                    source_rows = (
                        source_ids[expert, begin : begin + length].long().to(device)
                    )
                    source_rows += sum(sizes[:source])
                    reference[offset + begin : offset + begin + length] = global_x[
                        source_rows
                    ]
            offset += count
        q, sf = mxfp8_quantize(reference, is_sf_swizzled_layout=True, alignment=32)
        torch.testing.assert_close(
            state.packed[:offset], q.view(torch.uint8)[:offset], rtol=0, atol=0
        )
        mask = torch.zeros_like(state.packed_sf)
        mask[:offset] = 1
        mask = nvfp4_block_scale_interleave(mask).bool()
        torch.testing.assert_close(
            latest_dispatch.hidden_states_scale[mask], sf[mask], rtol=0, atol=0
        )
        assert torch.count_nonzero(state.packed[offset:]) == 0
        assert torch.count_nonzero(state.packed_sf[offset:]) == 0

    def check(actual, expected, name, phase, *, partials=None):
        torch.cuda.synchronize()
        if phase.startswith("decode"):
            check_mxfp8_bytes(all_x, sizes)
        if partials is not None:
            # Prefill error is relative to independent FP64 sum, rather than
            # another BF16 collective with a different reduction tree.
            error = _check_bf16_sum(actual, partials)
        else:
            error = (
                float((actual.float() - expected.float()).abs().max())
                if actual.numel()
                else 0.0
            )
            if not torch.isfinite(actual).all():
                raise AssertionError("non-finite output")
            # Compacted and source-order grouped GEMMs can select different
            # tactics; NVFP4 communication adds quantization error.
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
                        "dispatch_dtype": dispatcher.dispatch_dtype,
                        "combine_dtype": dispatcher.combine_dtype,
                        "wire_global_scale": dispatcher.wire_global_scale,
                    }
                ),
                flush=True,
            )

    try:
        for name, sizes in [
            ("uniform", [tokens] * ep),
            (
                "uneven_empty_rank",
                [min((0, 1, 5, 8)[r % 4], tokens) for r in range(ep)],
            ),
            ("all_invalid", [tokens] * ep),
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
                nonlocal latest_dispatch, latest_partial
                d = dispatcher.dispatch(x, top)
                latest_dispatch = d
                y = expert_fn(d, quant, config)
                if rdma.get_is_extend_in_batch():
                    latest_partial = y.hidden_states.clone()
                return dispatcher.combine(y)

            with (
                patch.object(rdma, "get_is_extend_in_batch", return_value=False),
                patch(
                    "sglang.srt.layers.moe.moe_runner.flashinfer_cutlass._cutlass_moe_tune_max_num_tokens",
                    return_value=4096,
                ),
            ):
                gold = decode_oracle(all_x, all_ids, all_weights)[begin:end]
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
                    all_x.mul_(-1.75)
                    x.mul_(-1.75)
                    all_weights.mul_(0.75)
                    weights.mul_(0.75)
                    if name not in ("all_invalid", "all_empty"):
                        all_ids.add_(1).remainder_(experts)
                        ids.copy_(all_ids[begin:end])
                    gold = decode_oracle(all_x, all_ids, all_weights)[begin:end]
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
                    actual = cycle()
                    reference_partial = direct(
                        all_x, None, all_ids, all_weights, reduce=False
                    )
                    torch.testing.assert_close(
                        latest_partial, reference_partial, rtol=0, atol=0
                    )
                    pieces = [torch.empty_like(reference_partial) for _ in range(ep)]
                    # Transfer independent source-order partials unchanged;
                    # the reference never uses production combine arithmetic.
                    dist.all_gather(pieces, reference_partial, group=group.device_group)
                    check(
                        actual,
                        None,
                        name,
                        "prefill_eager",
                        partials=torch.stack(pieces)[:, begin:end],
                    )
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
