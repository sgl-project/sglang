"""Two-GPU real NCCL EP + Triton + shared MLP acceptance gate.

Model/attention initialization and routing are controlled fixtures. The outer
ModelRunner branch, input registry, Graph backend, EP calls and GEMMs are real.
Run only after the single-GPU SM120 gate passes.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .dispatcher import dispatchers_for, initialize
from .ep_audit import EpAudit
from .oracle import RoutingBatch, assert_dispatch_matches, dequantize_fp8
from .runner_inputs import SyntheticDecodeRunner, SyntheticEagerRunner, input_batch
from .sglang_graph import close_runtime, runner_fixture, runner_routing
from .shared_compute import make_shared_mlp, shared_reference
from .triton_compute import (
    _gemm_cpu,
    _quantize_rows_cpu,
    configure_compute,
    make_compute_fixture,
)


def expected_output(batch, rank, quant, shared, scale, *, compute_backend="cpu"):
    """CPU routing/weighting with CPU or independent unpadded Triton experts.

    The Triton variant never calls the EP adapter or uses its receive buffers.
    Structured inputs can land on BF16/FP8 midpoints; CPU arithmetic is also
    reported, but cannot universally reproduce that amplified device rounding.
    """
    if compute_backend not in ("cpu", "triton"):
        raise ValueError(compute_backend)
    x = batch.tokens[rank]
    active = (batch.expert_ids[rank] >= 0).any(-1)
    result = torch.zeros_like(x, dtype=torch.float32)
    if not active.any():
        return result
    # NCCL post-quantization, adapter dequantization, then Triton quantization.
    wire = _quantize_rows_cpu(x.bfloat16()).bfloat16()
    for expert in range(batch.num_experts):
        if compute_backend == "cpu":
            first = _gemm_cpu(
                wire, quant.w13_weight[expert].cpu(), quant.w13_scale[expert].cpu()
            )
            gate, up = first.chunk(2, -1)
            intermediate = (F.silu(gate) * up).bfloat16()
            out = _gemm_cpu(
                intermediate,
                quant.w2_weight[expert].cpu(),
                quant.w2_scale[expert].cpu(),
            )
        else:
            from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
                fused_experts_impl,
            )

            out = (
                fused_experts_impl(
                    wire.cuda(),
                    quant.w13_weight[expert : expert + 1],
                    quant.w2_weight[expert : expert + 1],
                    torch.ones(len(x), 1, device="cuda"),
                    torch.zeros(len(x), 1, device="cuda", dtype=torch.int32),
                    use_fp8_w8a8=True,
                    w1_scale=quant.w13_scale[expert : expert + 1],
                    w2_scale=quant.w2_scale[expert : expert + 1],
                    block_shape=[128, 128],
                    no_combine=True,
                )
                .view(len(x), -1)
                .cpu()
                .float()
            )
        factor = torch.where(
            batch.expert_ids[rank] == expert, batch.weights[rank], 0
        ).sum(-1)
        result += out * factor[:, None]
    shared_output = (
        shared_reference(shared, x)
        if compute_backend == "cpu"
        else shared(x.cuda()).cpu().float()
    )
    result = result.bfloat16().float() * scale + shared_output
    result[~active] = 0
    return result.bfloat16().float()


def exercise(*, replays=1000):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.layers.moe.moe_runner.nccl_ep_triton import run_nccl_ep_triton
    from sglang.srt.layers.moe.token_dispatcher.nccl_ep_admission import (
        NcclEpGraphAdmission,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    rank, coordinator, bindings = initialize(64, graph_enabled=True)
    configure_compute(graph_enabled=True)
    coordinator.cpu_group = dist.group.WORLD
    _, global_quant, config = make_compute_fixture(
        hidden=2048, intermediate=128, experts=4
    )
    local_quant = replace(
        global_quant,
        w13_weight=global_quant.w13_weight[rank * 2 : rank * 2 + 2],
        w2_weight=global_quant.w2_weight[rank * 2 : rank * 2 + 2],
        w13_scale=global_quant.w13_scale[rank * 2 : rank * 2 + 2],
        w2_scale=global_quant.w2_scale[rank * 2 : rank * 2 + 2],
    )
    config = replace(config, num_experts=4, num_local_experts=2)
    observed = {}
    selected, checked, idle_receives = [], 0, 0
    admission_ms, forward_host_ms = [], []
    cpu_composed_max_abs_error, cpu_composed_allclose = 0.0, True

    class TimedAdmission(NcclEpGraphAdmission):
        def decide(self, **kwargs):
            started = time.perf_counter()
            result = super().decide(**kwargs)
            self.last_ms = (time.perf_counter() - started) * 1000
            return result

    with patch(
        "sglang.srt.layers.linear.get_tp_group", return_value=coordinator
    ), EpAudit() as audit:
        dispatchers = dispatchers_for(coordinator, 2)
        shared = make_shared_mlp(shared_experts=2)

        def forward(batch):
            x, ids, weights = runner_routing(batch, rank)
            x = x / 64
            snapshots, outputs = [], []
            for dispatcher in dispatchers:
                shared_output = shared(x)
                dispatched = dispatcher.dispatch(
                    x, StandardTopKOutput(weights, ids, None)
                )
                computed = run_nccl_ep_triton(dispatched, local_quant, config)
                snapshots.append(
                    (
                        dequantize_fp8(
                            dispatched.hidden_states, dispatched.hidden_states_scale
                        ),
                        dispatched.masked_m.clone(),
                    )
                )
                combined = dispatcher.combine(computed)
                outputs.append(shared_output.add(combined, alpha=2.5))
            observed[batch.batch_size] = snapshots
            return LogitsProcessorOutput(next_token_logits=torch.cat(outputs, dim=1))

        runner = SyntheticDecodeRunner(
            forward,
            coordinator,
            buckets=(1, 8, 16, 32),
            backend_factory=lambda owner: FullCudaGraphBackend(
                owner, nccl_ep_capacity=32
            ),
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )
        runner.require_mlp_sync = True
        outer = SimpleNamespace(
            device="cuda",
            attn_backend=runner.attn_backend,
            server_args=SimpleNamespace(enable_nccl_ep_cuda_graph=True),
            decode_cuda_graph_runner=runner,
            _nccl_ep_graph_admission=TimedAdmission(dist.group.WORLD),
            hisparse_coordinator=None,
            _prepare_eager_forward_batch=lambda batch: None,
            _maybe_execute_deferred_mamba_cow_and_clear=lambda batch: None,
            prefill_cuda_graph_runner=None,
            eager_runner=SyntheticEagerRunner(forward, 64),
        )
        cases = ((8, 8), (0, 16), (32, 0), (0, 33), (9, 1), (1, 17), (8, 8))
        try:
            for generation in range(2):
                for step in range(len(cases) + replays):
                    counts = (
                        cases[step % len(cases)]
                        if step < len(cases)
                        else ((0, 16), (32, 0), (9, 1))[step % 3]
                    )
                    values = tuple(
                        [[1, 2, 4, 8][(row + step + peer) % 4] for row in range(count)]
                        for peer, count in enumerate(counts)
                    )
                    mode = (
                        CaptureHiddenMode.FULL
                        if generation == 0 and rank == 0
                        else CaptureHiddenMode.NULL
                    )
                    incoming = input_batch(values[rank], hidden_mode=mode)
                    incoming.forward_mode = (
                        ForwardMode.IDLE if counts[rank] == 0 else ForwardMode.DECODE
                    )
                    incoming.can_run_dp_cuda_graph = True
                    started = time.perf_counter()
                    result = ModelRunner._forward_raw(outer, incoming, None)
                    # Exclude validation cases and the first step after their
                    # CPU oracle work, which can leave unequal host arrival times.
                    if step > len(cases):
                        admission_ms.append(outer._nccl_ep_graph_admission.last_ms)
                        forward_host_ms.append((time.perf_counter() - started) * 1000)
                    expected_graph = max(counts) <= 32
                    assert result.can_run_graph == expected_graph
                    local_bucket = (
                        next(
                            (
                                bucket
                                for bucket in (1, 8, 16, 32)
                                if bucket >= counts[rank]
                            ),
                            counts[rank],
                        )
                        if expected_graph
                        else counts[rank]
                    )
                    selected.append((counts[rank], local_bucket, result.can_run_graph))
                    if step < len(cases) or step == len(cases) + replays - 1:
                        torch.cuda.synchronize()
                        fixture = runner_fixture(max(1, max(counts)), values, counts)
                        fixture = RoutingBatch(
                            tuple(x / 64 for x in fixture.tokens),
                            fixture.expert_ids,
                            fixture.weights,
                            4,
                        )
                        for received, counters in observed[local_bucket]:
                            assert_dispatch_matches(fixture, rank, received, counters)
                            if not counts[rank]:
                                idle_receives += int(counters.sum().item() > 0)
                        wanted = expected_output(
                            fixture,
                            rank,
                            global_quant,
                            shared,
                            2.5,
                            compute_backend="triton",
                        )[: counts[rank]]
                        cpu_wanted = expected_output(
                            fixture, rank, global_quant, shared, 2.5
                        )[: counts[rank]]
                        actual = result.logits_output.next_token_logits
                        assert actual.shape == (counts[rank], 4096)
                        for layer in range(2):
                            layer_output = (
                                actual[:, layer * 2048 : (layer + 1) * 2048]
                                .cpu()
                                .float()
                            )
                            torch.testing.assert_close(
                                layer_output,
                                wanted,
                                rtol=0.02,
                                atol=0.02,
                            )
                            if counts[rank]:
                                cpu_composed_max_abs_error = max(
                                    cpu_composed_max_abs_error,
                                    (layer_output - cpu_wanted).abs().max().item(),
                                )
                                cpu_composed_allclose &= torch.allclose(
                                    layer_output, cpu_wanted, rtol=0.02, atol=0.02
                                )
                        checked += 1
            assert runner.capture_generations == 2
            assert idle_receives > 0
        finally:
            runner.backend.cleanup()
            resources = close_runtime(coordinator, audit)
    return {
        "gate": "native_pair_followups",
        "rank": rank,
        "passed": True,
        "checked": checked,
        "idle_receives": idle_receives,
        "capture_generations": runner.capture_generations,
        "replays_per_generation": replays,
        "admission_host_ms_mean": (
            sum(admission_ms) / len(admission_ms) if admission_ms else None
        ),
        "steady_forward_host_ms_mean": (
            sum(forward_host_ms) / len(forward_host_ms) if forward_host_ms else None
        ),
        "timing_scope": "Host wall time including peer wait; not GPU kernel latency",
        "selected": selected[: len(cases)],
        "bindings": bindings,
        "resources": resources,
        "real_ep": True,
        "real_triton_gemm": True,
        "real_shared_mlp": True,
        "arithmetic_acceptance_reference": "Independent unpadded Triton experts and standalone shared MLP; CPU routing and weighting",
        "cpu_composed_diagnostic": {
            "max_abs_error": cpu_composed_max_abs_error,
            "allclose_at_0_02": cpu_composed_allclose,
        },
        "full_model": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=1000)
    args = parser.parse_args()
    if args.replays <= 0:
        parser.error("replays must be positive")
    result = exercise(replays=args.replays)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / f"pair-rank{result['rank']}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
