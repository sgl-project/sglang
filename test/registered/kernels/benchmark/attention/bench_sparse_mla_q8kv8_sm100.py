"""SM100 Q8KV8 sparse MLA against FlashInfer's TRT-LLM-gen sparse MLA on the same FP8 inputs.

Both kernels get the inputs the DSA `trtllm` backend builds: merged e4m3 Q, a 576 B e4m3 KV pool in
pages of 64 tokens, and token-level top-k slots. Each case runs s_q query tokens against a pool of
s_kv tokens; s_q = 4096 stands for a prefill chunk, small s_q for decode and MTP verify batches.
"""

import functools
import math

import torch
from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.sparse_mla_q8kv8_sm100 import (
    D_QK,
    D_V,
    sparse_mla_q8kv8_sm100_trtllm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="nightly", runner_config="4-gpu-b200")

H_Q = 64
PAGE = 64
SCALING = 1.0 / math.sqrt(192 + 64)


@functools.cache
def _workspace() -> torch.Tensor:
    return torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")


@functools.lru_cache(maxsize=1)
def _inputs(s_q: int, s_kv: int, topk: int):
    g = torch.Generator(device="cuda").manual_seed(0)
    query = (torch.randn(s_q, 1, H_Q, D_QK, generator=g, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    kv_cache = (torch.randn(s_kv, D_QK, generator=g, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    block_tables = torch.empty(s_q, 1, topk, dtype=torch.int32, device="cuda")
    for start in range(0, s_q, 256):
        rows = min(256, s_q - start)
        keys = torch.rand(rows, s_kv, generator=g, device="cuda")
        block_tables[start : start + rows, 0] = keys.topk(topk, dim=-1).indices
    seq_lens = torch.full((s_q,), s_kv, dtype=torch.int32, device="cuda")
    return query, kv_cache.view(-1, 1, PAGE, D_QK), block_tables, seq_lens


def _q8kv8_sm100(query, kv_cache, block_tables, seq_lens):
    return sparse_mla_q8kv8_sm100_trtllm(
        query, kv_cache, block_tables, SCALING, seq_lens=seq_lens
    )


def _trtllm_gen(query, kv_cache, block_tables, seq_lens):
    return trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=_workspace(),
        qk_nope_head_dim=192,
        kv_lora_rank=D_V,
        qk_rope_head_dim=D_QK - D_V,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=kv_cache.shape[0] * PAGE,
        sparse_mla_top_k=block_tables.shape[-1],
        bmm1_scale=SCALING,
        backend="trtllm-gen",
    )


FN_MAP = {"q8kv8_sm100": _q8kv8_sm100, "trtllm_gen": _trtllm_gen}


@marker.parametrize("topk", [512, 2048], [2048])
@marker.parametrize(
    "s_q,s_kv",
    [(1, 65536), (16, 65536), (64, 65536), (256, 65536), (4096, 4096), (4096, 65536)],
    [(16, 65536), (4096, 4096)],
)
@marker.benchmark("impl", list(FN_MAP))
def benchmark(topk: int, s_q: int, s_kv: int, impl: str) -> marker.BenchResult:
    args = _inputs(s_q, s_kv, topk)
    return marker.do_bench(
        FN_MAP[impl],
        input_args=args,
        # Q, KV and the slots are cloned per graph iteration, so every call starts with a cold L2.
        graph_clone_args=(0, 1, 2),
        disable_log_bandwidth=True,
        flops=2.0 * s_q * topk * H_Q * (D_QK + D_V),
    )


if __name__ == "__main__":
    benchmark.run()
