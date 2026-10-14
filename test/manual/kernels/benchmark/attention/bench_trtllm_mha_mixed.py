"""Compare split and unsplit mixed-extend attention on SM100-family GPUs.

Times forward_extend with CUDA graphs and cold L2, excluding cache flushes,
projection, KV writes and metadata construction. Run from the repository root
with PYTHONPATH=python.
"""

import argparse
import gc
import json
import unittest

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.common import is_sm100_supported
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
)


@torch.no_grad()
def benchmark(args, num_decode):
    case = DenseAttentionCase(
        name="mixed_extend",
        backend="trtllm_mha",
        forward_mode=ForwardMode.MIXED,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        page_size=64,
        prefix_lens=(0,) + (args.kv_len - 1,) * num_decode,
        extend_lens=(args.prefill_tokens,) + (1,) * num_decode,
    )
    fixture = build_dense_attention_fixture(
        unittest.TestCase(),
        case,
        head_dim=args.head_dim,
        hidden_size=128,
        max_context_len=((max(args.kv_len, args.prefill_tokens) + 63) // 64) * 64,
        dtype=torch.bfloat16,
    )
    runner = fixture.runner
    layer = fixture.actual_module.attn
    layer.sliding_window_size = args.window_size
    if args.kv_cache_dtype == "fp8":
        runner.kv_cache_dtype = torch.float8_e4m3fn
        runner.token_to_kv_pool = MHATokenToKVPool(
            size=runner.token_to_kv_pool.size,
            page_size=64,
            dtype=runner.kv_cache_dtype,
            head_num=args.num_kv_heads,
            head_dim=args.head_dim,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            enable_alt_stream=False,
        )
        runner.init_kv_index_translator()
        fixture.backend = TRTLLMHAAttnBackend(runner)
        for req, prefix in enumerate(fixture.prefix_hidden):
            if prefix.shape[0]:
                _, k, v = fixture.actual_module.project_qkv(prefix)
                runner.token_to_kv_pool.set_kv_buffer(
                    layer,
                    runner.req_to_token_pool.req_to_token[req, : len(prefix)].long(),
                    k.view(-1, args.num_kv_heads, args.head_dim),
                    v.view(-1, args.num_kv_heads, args.head_dim),
                )
    backend = fixture.backend
    if not is_sm100_supported():
        raise RuntimeError("The decode-tail split requires an SM100-family GPU")
    q, k, v = fixture.actual_module.project_qkv(fixture.input_hidden)
    batch = fixture.forward_batch
    backend.init_forward_metadata(batch)
    backend.forward_extend(q, k, v, layer, batch)

    def run():
        return backend.forward_extend(q, None, None, layer, batch, save_kv_cache=False)

    outputs, timings = {}, {}
    for split in (False, True):
        backend.init_forward_metadata(batch)
        if not split:
            backend.forward_metadata.mixed_prefill_reqs = -1
        outputs[split] = run().clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Flush before each replay, not just before a graph of repeated calls.
        timings[split] = (
            marker.do_bench(
                graph.replay,
                use_cuda_graph=False,
                estimated_time_ms=200,
                disable_log_bandwidth=True,
            ).times[0]
            * 1e6
        )
    torch.testing.assert_close(outputs[True], outputs[False], atol=1e-2, rtol=1e-2)
    delta = outputs[True].float() - outputs[False].float()
    return {
        "decode_requests": num_decode,
        "prefill_tokens": args.prefill_tokens,
        "kv_len": args.kv_len,
        "kv_cache_dtype": args.kv_cache_dtype,
        "window_size": args.window_size,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "head_dim": args.head_dim,
        "unsplit_us": timings[False],
        "split_us": timings[True],
        "speedup": timings[False] / timings[True],
        "max_abs_diff": delta.abs().max().item(),
        "relative_rms_diff": (delta.norm() / outputs[False].float().norm()).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decode-requests", type=int, nargs="+", default=[0, 1, 32, 128, 256, 511]
    )
    parser.add_argument("--prefill-tokens", type=int, default=8192)
    parser.add_argument("--kv-len", type=int, default=8192)
    parser.add_argument("--kv-cache-dtype", choices=["bf16", "fp8"], default="fp8")
    parser.add_argument("--window-size", type=int, default=-1)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    for num_decode in args.decode_requests:
        print(json.dumps(benchmark(args, num_decode)), flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
