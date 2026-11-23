"""Run the production CP FP8 preparation path on real NCCL ranks.

torchrun --standalone --nproc-per-node=4 this_file.py
Tests preparation (not MoE GEMMs); full-model coverage is separate.
"""

import json
import math
import os
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.quantization.per_token_group_quant import per_token_group_quant
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers.cp.interleave import attn_cp_interleave_gather
from sglang.srt.layers.cp.moe_fp8_transport import (
    CpMoeFp8Binding,
    CpMoeFp8Invocation,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.runtime_context import get_parallel


class RouterFixture:
    """Synthetic weights, production GLM gate and quantization kernels."""

    def __init__(self):
        torch.manual_seed(42)
        self.weight = torch.randn(288, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
        self.bias = torch.zeros(288, device="cuda", dtype=torch.float32)
        self.experts = SimpleNamespace(
            quant_method=SimpleNamespace(
                runner=SimpleNamespace(
                    runner_backend=SimpleNamespace(is_deep_gemm=lambda: False)
                )
            )
        )

    def _compute_cp_moe_fp8_transport_enabled(self):
        return True, "synthetic integration fixture"

    def gate(self, x):
        return linear_bf16_fp32(x, self.weight)

    def topk(self, x, logits, num_token_non_padded=None):
        weights, ids = moe_fused_gate(
            logits,
            self.bias,
            topk=9,
            scoring_func="sigmoid",
            num_fused_shared_experts=1,
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_routed_scaling_factor_on_output=False,
            num_expert_group=1,
            topk_group=1,
        )
        return StandardTopKOutput(weights, ids, None)


@torch.inference_mode()
def run_transport_chain():
    rank = int(os.environ["LOCAL_RANK"])
    size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    coord = ps.init_world_group(list(range(size)), rank, "nccl")
    get_parallel().override_permanently(
        world_group=coord,
        tp_group=coord,
        attn_cp_group=coord,
        tp_size=size,
        tp_rank=rank,
        attn_cp_size=size,
        attn_cp_rank=rank,
        attn_tp_size=1,
        attn_dp_size=1,
    )
    layer = RouterFixture()
    binding = CpMoeFp8Binding(layer)
    # Construct the declared CP rows with the actual configured group widths.
    from sglang.srt.layers.layer_boundary.layout import Layout, SumGroup, TokenAxis
    from sglang.srt.layers.layer_boundary.row_transport import TransportRows

    source = Layout(frozenset({TokenAxis.ATTN_CP}))
    rows = TransportRows(source, Layout(frozenset()), source, SumGroup.MOE_OUTPUT)
    batch = SimpleNamespace(
        num_token_non_padded=None, global_num_token_non_padded_cpu=7
    )
    torch.cuda.set_stream(torch.cuda.Stream())
    from sglang.srt.distributed.device_communicators.pynccl import PyNcclCommunicator

    comm = PyNcclCommunicator(coord.cpu_group, device=rank)
    coord.pynccl_comm = comm
    try:
        with comm.change_state(enable=True):
            for total in (4096, 8192, 17408, 32768, 65536):
                torch.manual_seed(1234 + rank)
                x = torch.randn(
                    total // size, 4096, device="cuda", dtype=torch.bfloat16
                )

                def baseline(value):
                    full = attn_cp_interleave_gather(value)
                    topk = layer.topk(full, layer.gate(full))
                    q, scale = per_token_group_quant(full, group_size=128)
                    return q, scale, topk.topk_weights, topk.topk_ids

                def optimized(value):
                    invocation = CpMoeFp8Invocation(binding, value, batch, rows)
                    assert invocation.prepared is not None
                    return invocation.prepared

                for updated in (False, True):
                    if updated:
                        x.mul_(0.5).add_(0.1)
                    ref, actual = baseline(x), optimized(x)
                    torch.cuda.synchronize()
                    assert torch.equal(
                        ref[0].view(torch.uint8), actual[0].view(torch.uint8)
                    )
                    torch.testing.assert_close(ref[1], actual[1], rtol=0, atol=0)
                    assert torch.equal(ref[3], actual[3])
                    torch.testing.assert_close(ref[2], actual[2], rtol=1e-5, atol=1e-6)
                # Ineligible row counts must select the BF16 helper, not quantize.
                small = x[:3]
                fallback = CpMoeFp8Invocation(binding, small, batch, rows)
                assert fallback.prepared is None
                torch.testing.assert_close(
                    fallback.input, attn_cp_interleave_gather(small)
                )
                with fallback.scope():
                    assert batch.global_num_token_non_padded_cpu == 3 * size
                assert batch.global_num_token_non_padded_cpu == 7
                partial = torch.full(
                    (total, 8), rank + 1, device="cuda", dtype=torch.bfloat16
                )
                returned = fallback.return_output(partial[: 3 * size].clone())
                assert returned.shape == (3, 8)
                torch.testing.assert_close(
                    returned, torch.full_like(returned, size * (size + 1) / 2)
                )
                ordinary = TransportRows(
                    Layout(frozenset()),
                    Layout(frozenset()),
                    Layout(frozenset()),
                    SumGroup.MOE_OUTPUT,
                )
                non_cp = CpMoeFp8Invocation(binding, x, batch, ordinary)
                assert non_cp.input is x and non_cp.prepared is None
                fallback.release()

                # Rotate local reads over >5x L2 and retain graph outputs, so
                # cache-hot repeated-input timing is not reported as DRAM timing.
                l2 = torch.cuda.get_device_properties(rank).L2_cache_size
                rotations = max(2, math.ceil(5 * l2 / x.nbytes) + 1)
                locals_ = [x] + [x.clone() for _ in range(rotations - 1)]
                graphs, held = {}, []
                for name, fn in (("bf16", baseline), ("fp8", optimized)):
                    for _ in range(3):
                        fn(x)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for local in locals_:
                            held.append(fn(local))
                    graphs[name] = graph
                # Replay with new input verifies no captured stale routing/values.
                x.add_(0.1)
                for graph in graphs.values():
                    graph.replay()
                expected = baseline(x)
                torch.cuda.synchronize()
                for actual in held[::rotations]:
                    assert torch.equal(
                        actual[0].view(torch.uint8), expected[0].view(torch.uint8)
                    )
                    ref_ids, ref_order = expected[3].sort(dim=-1)
                    new_ids, new_order = actual[3].sort(dim=-1)
                    assert torch.equal(new_ids, ref_ids), (
                        "expert sets changed in replay"
                    )
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual[2].gather(1, new_order),
                        expected[2].gather(1, ref_order),
                        rtol=1e-5,
                        atol=1e-6,
                    )
                times = {"bf16": [], "fp8": []}
                for trial in range(6):
                    for name in ("bf16", "fp8") if trial % 2 == 0 else ("fp8", "bf16"):
                        dist.barrier()
                        start, end = [
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        ]
                        start.record()
                        for _ in range(20):
                            graphs[name].replay()
                        end.record()
                        end.synchronize()
                        sample = torch.tensor(
                            start.elapsed_time(end) / 20 / rotations, device="cuda"
                        )
                        dist.all_reduce(
                            sample, op=dist.ReduceOp.MAX, group=coord.device_group
                        )
                        times[name].append(sample.item())
                record = dict(
                    total_rows=total,
                    bf16_ms=statistics.median(times["bf16"]),
                    fp8_ms=statistics.median(times["fp8"]),
                    trials=times,
                    rotations=rotations,
                    input_source="synthetic BF16 and router weights",
                )
                if rank == 0:
                    print("RESULT", json.dumps(record), flush=True)
                del graphs, held, ref, actual, expected, locals_
        if rank == 0:
            print("PRODUCTION_PREPARATION_INTEGRATION_PASS", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_transport_chain()
