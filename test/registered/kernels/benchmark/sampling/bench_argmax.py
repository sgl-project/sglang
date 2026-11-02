import torch
import triton
import triton.language as tl

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.sampling.argmax_jit import argmax as jit_argmax
from sglang.kernels.ops.speculative.row_argmax import _medium_row_argmax
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def torch_argmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


# The Triton split-vocab argmax DSpark verify used before the JIT kernel replaced it
@triton.jit
def _argmax_partial_kernel(
    X, OUTV, OUTI, N, SX, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    per = tl.cdiv(N, SPLITS)
    start = part * per
    best_v = float("-inf")
    best_i = N
    for off in tl.range(start, tl.minimum(start + per, N), BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < tl.minimum(start + per, N)
        v = tl.load(X + row * SX + idx, m, float("-inf"))
        cur_v = tl.max(v, 0)
        # lowest index among the maxima of this tile
        cur_i = tl.min(tl.where(v == cur_v, idx, N), 0)
        take = (cur_v > best_v) | ((cur_v == best_v) & (cur_i < best_i))
        best_i = tl.where(take, cur_i, best_i)
        best_v = tl.where(take, cur_v, best_v)
    tl.store(OUTV + row * SPLITS + part, best_v)
    tl.store(OUTI + row * SPLITS + part, best_i)


@triton.jit
def _argmax_final_kernel(INV, INI, OUT, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    o = tl.arange(0, BLOCK)
    m = o < SPLITS
    v = tl.load(INV + row * SPLITS + o, m, float("-inf"))
    i = tl.load(INI + row * SPLITS + o, m, 0x7FFFFFFF)
    best_v = tl.max(v, 0)
    best_i = tl.min(tl.where(v == best_v, i, 0x7FFFFFFF), 0)
    tl.store(OUT + row, best_i.to(tl.int64))


def triton_argmax(logits: torch.Tensor) -> torch.Tensor:
    rows, n = logits.shape
    if rows > 64:
        return _medium_row_argmax(logits)
    splits = 64
    out = torch.empty((rows,), dtype=torch.int64, device=logits.device)
    pv = torch.empty((rows, splits), dtype=torch.float32, device=logits.device)
    pi = torch.empty((rows, splits), dtype=torch.int32, device=logits.device)
    _argmax_partial_kernel[(rows, splits)](
        logits, pv, pi, n, logits.stride(0), SPLITS=splits, BLOCK=2048, num_warps=8
    )
    _argmax_final_kernel[(rows,)](pv, pi, out, SPLITS=splits, BLOCK=64, num_warps=2)
    return out


FN_MAP = {"jit": jit_argmax, "triton": triton_argmax, "torch": torch_argmax}


# fp32 is what the sampler sees: `LogitsProcessor._copy_logits_to_buffer` upcasts
# before `Sampler.forward` reaches its greedy branch.
@marker.parametrize("vocab_size", [32000, 129280, 151936, 256000], [129280])
@marker.parametrize("batch_size", [2**n for n in range(0, 12)], [1, 64, 1024])
@marker.benchmark("impl", ["jit", "triton", "torch"])
def benchmark(vocab_size: int, batch_size: int, impl: str):
    logits = create_random(batch_size, vocab_size, dtype=torch.float32)
    return marker.do_bench(FN_MAP[impl], input_args=(logits,))


if __name__ == "__main__":
    benchmark.run()
