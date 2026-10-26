"""Benchmark one greedy EAGLE draft step's LM head + top-1: the logits gather vs
the vocab-shard argmax partials exchange.

Both providers run LogitsProcessor.forward on SGLang's TP group (set up as the
server's bootstrap does) with a 154880-token vocab and hidden size 6144:
  gather:   LM head GEMM, all-gather of the logits, fp32 copy,
            draft_topk1_postprocess
  exchange: LM head GEMM, draft_topk1_shard_partials, all-reduce of the fp32
            partials, draft_topk1_postprocess_from_partials
Times are CUDA-graph replays (16 steps per graph), max over ranks. The LM head
shard (300-600 MB) far exceeds L2, and the logits the top-1 reads were just
written by the GEMM, which is the serving pattern, so no L2 flush is added.

Usage::

    # one node, 4 GPUs
    python test/registered/kernels/benchmark/speculative/bench_draft_argmax_exchange.py --num-gpu 4
    # TP8 over two GB300 trays (MNNVL): launch under torchrun on both nodes
    torchrun --nnodes 2 --nproc-per-node 4 --rdzv-backend c10d \\
        --rdzv-endpoint <node0>:29500 bench_draft_argmax_exchange.py
"""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.kernels.jit.benchmark.utils import multigpu_bench_main
from sglang.kernels.ops.speculative.topk1 import (
    draft_topk1_postprocess,
    draft_topk1_postprocess_from_partials,
)
from sglang.srt.distributed import parallel_state as ps
from sglang.srt.layers import flashinfer_comm_fusion as fi
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.layers.layer_boundary import FUSE_ALLREDUCE_MAX_BATCH_SIZE
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_exec, get_forward, get_server_args
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import publish_build_topology

register_cuda_ci(
    est_time=120,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
    disabled="requires multi-GPU, self-skips in CI",
)

BF16 = torch.bfloat16
HIDDEN = 6144
VOCAB = 154880
ROWS = (1, 2, 4, 8, 16)
STEPS_PER_GRAPH = 16


def _init_tp_group():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    nnodes = world // int(os.environ.get("LOCAL_WORLD_SIZE", world))
    torch.cuda.set_device(local_rank)
    ps.init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
    )
    blackwell = torch.cuda.get_device_capability()[0] >= 10
    publish_build_topology(
        world_rank=rank,
        tp_size=world,
        nnodes=nnodes,
        device="cuda",
        flashinfer_allreduce_fusion_backend="auto" if blackwell else None,
    )
    fusion = get_exec().comm.flashinfer_allreduce_fusion_backend
    ps.set_custom_all_reduce(not get_exec().comm.disable_custom_all_reduce)
    ps.set_flashinfer_allreduce_only(
        fusion is not None and not fi.uses_cutedsl_ar_fusion()
    )
    ps.initialize_model_parallel()
    ps._tag_groups_for_flashinfer_allreduce_only()
    initialize_dp_attention(server_args=get_server_args())
    if fusion is not None:
        fi.pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE, hidden_dim=HIDDEN, dtype=BF16
        )
    return ps.get_tp_group()


def _graph_us(step, tp) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(STEPS_PER_GRAPH):
            step()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(40):
        dist.barrier(group=tp.cpu_group)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / STEPS_PER_GRAPH)
    del graph
    median = torch.tensor([sorted(times)[len(times) // 2]], device="cuda")
    dist.all_reduce(median, op=dist.ReduceOp.MAX)
    return median.item()


def main() -> None:
    tp = _init_tp_group()
    proc = LogitsProcessor(SimpleNamespace(vocab_size=VOCAB))
    assert proc.enable_draft_argmax_exchange(), "processor not eligible"
    with torch.device("cuda"):
        head = ParallelLMHead(VOCAB, HIDDEN, params_dtype=BF16)
    torch.nn.init.normal_(head.weight, std=0.02)
    rows_out = []
    for rows in ROWS:
        hidden = torch.randn(rows, HIDDEN, device="cuda").to(BF16)
        input_ids = torch.zeros(rows, dtype=torch.long, device="cuda")
        pos = torch.zeros(rows, dtype=torch.long, device="cuda")
        toks = torch.zeros(rows, 6, dtype=torch.long, device="cuda")
        metadata = LogitsMetadata(forward_mode=ForwardMode.DECODE)

        def gather():
            out = proc(input_ids, hidden, head, metadata)
            draft_topk1_postprocess(out.next_token_logits, pos, toks, 1)

        def exchange():
            with get_forward().scoped(draft_argmax_only=True):
                out = proc(input_ids, hidden, head, metadata)
            draft_topk1_postprocess_from_partials(
                out.next_token_argmax_partials, pos, toks, 1
            )

        rows_out.append((rows, _graph_us(gather, tp), _graph_us(exchange, tp)))
    if dist.get_rank() == 0:
        method = tp.device_all_reduce_method(
            torch.empty(1, 64, dtype=torch.float32, device="cuda")
        )
        print(f"TP{tp.world_size}, vocab {VOCAB}, all-reduce {method}")
        print("| rows | gather us | exchange us | delta |")
        print("|---|---|---|---|")
        for rows, old, new in rows_out:
            print(f"| {rows} | {old:.2f} | {new:.2f} | {new - old:+.2f} |")


if __name__ == "__main__":
    if (
        "_IS_BENCH_MULTIGPU_SGLANG_JIT_KERNEL" not in os.environ
        and "RANK" in os.environ
    ):
        main()  # already under an external (multi-node) torchrun
    else:
        multigpu_bench_main(
            name=__name__, file=__file__, num_gpus=(2, 4, 8), main_fn=main
        )
