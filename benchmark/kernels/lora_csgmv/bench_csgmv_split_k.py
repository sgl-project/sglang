"""Compare CSGMV early-return expand, compact expand, and split-K on one GPU.

PYTHONPATH=python python benchmark/kernels/lora_csgmv/bench_csgmv_split_k.py
Timings cover shrink + expand under CUDA graphs; routing is excluded.
"""

import argparse
import itertools
import json
import math

import torch
import triton

from sglang.kernels.ops.gemm.chunked_sgmv_expand import chunked_sgmv_lora_expand_forward
from sglang.kernels.ops.gemm.chunked_sgmv_shrink import chunked_sgmv_lora_shrink_forward
from sglang.srt.environ import envs
from sglang.srt.lora.utils import LoRABatchInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[32, 128, 8192])
    parser.add_argument("--adapters", type=int, default=8)
    args = parser.parse_args()
    torch.manual_seed(0)
    envs.SGLANG_CSGMV_SPLIT_K.set(False)
    envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.set(False)
    print(
        json.dumps(
            {"gpu": torch.cuda.get_device_name(), "dtype": "bfloat16", "rank": 32}
        )
    )
    for total in args.tokens:
        ids = torch.arange(total) % args.adapters
        permutation = torch.argsort(ids, stable=True).int().cuda()
        counts = torch.bincount(ids, minlength=args.adapters).tolist()
        lengths = [min(16, n - j) for n in counts for j in range(0, n, 16)]
        adapter_ids = [a for a, n in enumerate(counts) for _ in range(0, n, 16)]
        info = LoRABatchInfo(
            use_cuda_graph=False,
            bs=total,
            num_segments=len(lengths),
            max_len=16,
            seg_lens=None,
            permutation=permutation,
            seg_indptr=torch.tensor(
                [0, *itertools.accumulate(lengths)], device="cuda", dtype=torch.int32
            ),
            weight_indices=torch.tensor(adapter_ids, device="cuda", dtype=torch.int32),
            lora_ranks=torch.full(
                (args.adapters,), 32, device="cuda", dtype=torch.int32
            ),
            scalings=torch.ones(args.adapters, device="cuda"),
        )
        for name, k, offsets in [
            ("qkv", 4096, [0, 8192, 9216, 10240]),
            ("gate_up", 4096, [0, 12288, 24576]),
            ("down", 12288, [0, 4096]),
        ]:
            slices, n = len(offsets) - 1, offsets[-1]
            x = torch.randn(total, k, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(
                args.adapters, slices * 32, k, device="cuda", dtype=torch.bfloat16
            ) / math.sqrt(k)
            b = torch.randn(
                args.adapters, n, 32, device="cuda", dtype=torch.bfloat16
            ) / math.sqrt(32)
            base = torch.zeros(total, n, device="cuda", dtype=torch.bfloat16)
            cpu_offsets = torch.tensor(offsets, dtype=torch.int32)
            gpu_offsets = cpu_offsets.cuda()
            maxwidth = max(hi - lo for lo, hi in zip(offsets, offsets[1:]))
            splits = (16 if k == 12288 else 8) if total <= 128 else 1
            result = dict(
                tokens=total, adapters=args.adapters, projection=name, split_k=splits
            )
            for label, split, compact in [
                ("early_return", 1, False),
                ("compact", 1, True),
                ("compact_split_k", splits, True),
            ]:

                def run():
                    h = chunked_sgmv_lora_shrink_forward(
                        x, a, info, slices, split_k=split
                    )
                    chunked_sgmv_lora_expand_forward(
                        h,
                        b,
                        info,
                        gpu_offsets,
                        maxwidth,
                        base,
                        cpu_offsets if compact else None,
                    )

                result[label + "_us"] = (
                    triton.testing.do_bench_cudagraph(run, rep=100) * 1000
                )
            result["speedup"] = result["early_return_us"] / result["compact_split_k_us"]
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
