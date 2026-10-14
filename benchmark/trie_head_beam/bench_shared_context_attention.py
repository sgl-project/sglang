import argparse
import math
import sys

import torch
from common import require_single_visible_gpu

from sglang.srt.beam_search.shared_context_attention_kernel import (
    shared_context_attention,
)


def make_inputs(batch, beam_width, context_len, decode_len):
    torch.manual_seed(0)
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    context_capacity = batch * context_len
    decode_capacity = batch * decode_len * beam_width
    num_slots = context_capacity + decode_capacity

    q = torch.randn(
        batch,
        beam_width,
        q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_buffer = torch.randn(
        num_slots,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_buffer = torch.randn_like(k_buffer)
    context_slots = torch.arange(context_capacity, device="cuda", dtype=torch.int64)
    context_cu_seqlens = (
        torch.arange(batch + 1, device="cuda", dtype=torch.int32) * context_len
    )
    decode_slots = torch.arange(
        context_capacity,
        context_capacity + decode_capacity,
        device="cuda",
        dtype=torch.int64,
    ).view(batch, decode_len * beam_width)
    decode_lens = torch.full(
        (batch,),
        decode_len,
        device="cuda",
        dtype=torch.int32,
    )
    return (
        q,
        k_buffer,
        v_buffer,
        context_slots,
        context_cu_seqlens,
        decode_slots,
        decode_lens,
    )


def load_cute(root):
    sys.path.insert(0, root)
    import interface

    return interface


def cute_call(interface, inputs, softmax_scale):
    (
        q,
        k_buffer,
        v_buffer,
        context_slots,
        context_cu_seqlens,
        decode_slots,
        decode_lens,
    ) = inputs
    batch, beam_width, q_heads, _ = q.shape
    decode_len = decode_slots.shape[1] // beam_width
    context_k = k_buffer[context_slots]
    context_v = v_buffer[context_slots]
    beam_k = k_buffer[decode_slots]
    beam_v = v_buffer[decode_slots]
    ancestry = (
        torch.arange(
            decode_len * beam_width,
            device=q.device,
            dtype=torch.int32,
        )
        .view(1, 1, 1, decode_len, beam_width)
        .expand(batch, 1, q_heads, decode_len, beam_width)
        .contiguous()
    )
    output, _ = interface.BeamDecodeAttn.forward(
        None,
        q[:, None],
        context_k,
        context_v,
        beam_k,
        beam_v,
        ancestry,
        decode_len,
        softmax_scale,
        "fused",
        None,
        context_cu_seqlens,
    )
    return output[:, 0]


def native_call(inputs, softmax_scale, block_m, block_n, context_len):
    (
        q,
        k_buffer,
        v_buffer,
        context_slots,
        context_cu_seqlens,
        decode_slots,
        decode_lens,
    ) = inputs
    return shared_context_attention(
        q,
        k_buffer,
        v_buffer,
        context_slots,
        context_cu_seqlens,
        decode_slots,
        decode_lens,
        max_context_len=context_len,
        softmax_scale=softmax_scale,
        block_m=block_m,
        block_n=block_n,
    )


def benchmark(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return {
        "median_ms": times[len(times) // 2],
        "min_ms": times[0],
        "max_ms": times[-1],
        "mean_ms": sum(times) / len(times),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare native shared-context attention with a reference kernel."
    )
    parser.add_argument("--cute-root", required=True)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 10])
    parser.add_argument("--beam-width", type=int, default=2000)
    parser.add_argument("--context-len", type=int, default=384)
    parser.add_argument("--decode-len", type=int, default=3)
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args()
    require_single_visible_gpu(args.physical_gpu)

    interface = load_cute(args.cute_root)
    scale = 1.0 / math.sqrt(128)

    correctness_inputs = make_inputs(2, 64, 97, args.decode_len)
    cute_output = cute_call(interface, correctness_inputs, scale)
    print("correctness")
    for block_m in (32, 64, 128):
        native_output = native_call(correctness_inputs, scale, block_m, 64, 97)
        difference = (native_output.float() - cute_output.float()).abs()
        print(
            {
                "block_m": block_m,
                "max_abs": difference.max().item(),
                "mean_abs": difference.mean().item(),
                "allclose_5e-2": torch.allclose(
                    native_output.float(),
                    cute_output.float(),
                    rtol=5e-2,
                    atol=5e-2,
                ),
            }
        )

    print("performance")
    for batch in args.batch:
        inputs = make_inputs(
            batch,
            args.beam_width,
            args.context_len,
            args.decode_len,
        )
        results = {
            "batch": batch,
            "cute": benchmark(lambda: cute_call(interface, inputs, scale)),
        }
        for block_m in (32, 64, 128):
            for block_n in (32, 64, 128):
                results[f"triton_m{block_m}_n{block_n}"] = benchmark(
                    lambda bm=block_m, bn=block_n: native_call(
                        inputs,
                        scale,
                        bm,
                        bn,
                        args.context_len,
                    )
                )
        print(results)


if __name__ == "__main__":
    main()
