"""Frozen C=0 operator parity against an independently built vLLM OPP."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu

from sglang.kernels.ops.attention.npu_copy_sfa import (
    fused_scatter_copy_sparse_flash_attention,
    load_copy_sfa_operator,
)


def compare(width, graph):
    torch.manual_seed(7100 + width)
    batch, heads = 4, 8
    capacity = max(8192, width * 2048)
    pages = capacity // 128
    length_values = [width, 127, 2049, capacity - 1]
    cache = torch.randn(batch * pages, 128, 1, 576, device="npu", dtype=torch.bfloat16)
    split_kv = cache[..., :512].contiguous()
    split_rope = cache[..., 512:].contiguous()
    table = torch.randperm(batch * pages, device="npu", dtype=torch.int32).view(
        batch, pages
    )
    query = torch.randn(batch * width, heads, 512, device="npu", dtype=torch.bfloat16)
    rope = torch.randn(batch * width, heads, 64, device="npu", dtype=torch.bfloat16)
    lengths = torch.tensor(length_values, dtype=torch.int32, device="npu")
    ends = torch.arange(1, batch + 1, dtype=torch.int32, device="npu") * width
    zeros = torch.zeros(batch, dtype=torch.int32, device="npu")
    topk = torch.zeros(batch * width, 1, 2048, dtype=torch.int32, device="npu")
    topk_misses = torch.zeros(batch * width, dtype=torch.int32, device="npu")
    miss = torch.zeros(batch, 32768, dtype=torch.int32, device="npu")
    dram_table = torch.zeros(batch, 1, dtype=torch.int32, device="npu")
    # C=0 and miss_count=0 never read DRAM. Both operators receive valid
    # addressable dummies, while every real KV row comes from the frozen HBM.
    dram_kv = torch.zeros(1, 128, 512, device="npu", dtype=torch.bfloat16)
    dram_rope = torch.zeros(1, 128, 64, device="npu", dtype=torch.bfloat16)
    out = torch.zeros_like(query)
    reference = torch.zeros_like(query)
    native = load_copy_sfa_operator()
    original = torch.ops.vllm_copy_reference.run.default
    scale = 576**-0.5

    def ours():
        fused_scatter_copy_sparse_flash_attention(
            query,
            rope,
            cache,
            ends,
            lengths,
            zeros,
            topk,
            topk,
            topk_misses,
            miss,
            miss,
            zeros,
            table,
            dram_table,
            dram_kv.data_ptr(),
            out,
            scale,
            operator=native,
            full_visible=True,
            resident_refresh_tail=width > 1,
        )

    def theirs():
        original(
            rope,
            query,
            ends,
            lengths,
            zeros,
            topk,
            topk,
            topk_misses,
            miss,
            miss,
            zeros,
            table,
            dram_table,
            split_rope,
            split_kv,
            dram_rope,
            dram_kv,
            scale,
            reference,
        )

    ours()
    theirs()
    torch.npu.synchronize()
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.002)
    max_pair = (out.float() - reference.float()).abs().max().item()
    cpu_query, cpu_rope, cpu_cache, cpu_table = (
        query.cpu().float(),
        rope.cpu().float(),
        cache.cpu(),
        table.cpu().long(),
    )
    max_abs = [0.0, 0.0]
    for row, length in enumerate(length_values):
        payload = cpu_cache[cpu_table[row]].view(-1, 576).float()
        for col in range(width):
            t = row * width + col
            visible = length - width + col + 1
            scores = (
                cpu_query[t] @ payload[:visible, :512].T
                + cpu_rope[t] @ payload[:visible, 512:].T
            ) * scale
            expected = scores.softmax(-1) @ payload[:visible, :512]
            for index, output in enumerate((out, reference)):
                actual = output[t].cpu().float()
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.025)
                max_abs[index] = max(
                    max_abs[index], (actual - expected).abs().max().item()
                )
    samples = {}
    for name, function in (("sglang", ours), ("vllm", theirs)):
        for _ in range(5):
            function()
        iterations = 50
        unroll = 10 if graph else 1
        if graph:
            captured = torch.npu.NPUGraph()
            with torch.npu.graph(captured):
                for _ in range(unroll):
                    function()
            step = captured.replay
        else:
            step = function
        timings = []
        for _ in range(5):
            begin, end = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(iterations):
                step()
            end.record()
            end.synchronize()
            timings.append(begin.elapsed_time(end) * 1000 / iterations / unroll)
        samples[name] = timings
    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.002)
    return dict(
        query_width=width,
        graph=graph,
        bitwise_equal=torch.equal(
            out.cpu().view(torch.int16), reference.cpu().view(torch.int16)
        ),
        lengths=length_values,
        heads=heads,
        max_abs_sglang_vllm=max_pair,
        max_abs_sglang_fp32=max_abs[0],
        max_abs_vllm_fp32=max_abs[1],
        samples_us=samples,
        median_us={name: statistics.median(values) for name, values in samples.items()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--reference-commit", default="unspecified")
    parser.add_argument("--widths", default="1,2,3,4,5,8,14")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.ops.load_library(str(args.reference_library.resolve()))
    results = []
    for graph in (False, True):
        for width in map(int, args.widths.split(",")):
            result = compare(width, graph)
            print(json.dumps(result), flush=True)
            results.append(result)
            args.output.write_text(
                json.dumps(
                    dict(
                        scope="C=0 dense short-row attention only; excludes LIM, residency and D2H; graph timings unroll 10",
                        reference_commit=args.reference_commit,
                        torch=torch.__version__,
                        torch_npu=torch_npu.__version__,
                        cases=results,
                    ),
                    indent=2,
                )
                + "\n"
            )
