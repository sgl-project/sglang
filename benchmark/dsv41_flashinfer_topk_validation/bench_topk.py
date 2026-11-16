"""DeepSeek V4.1 index top-k benchmark, mirroring sgl-project/sglang#47960.

One process runs one variant over every shape: a CUDA graph of R x [PDL producer
-> top-k], R sized so a replay takes ~3 ms, replays separated by 5 ms idle gaps,
median microseconds per call. Inputs are seeded random rows, identical for every
variant. The producer copies the row lengths the top-k reads and triggers PDL
after its stores.

Variants (each checks the environment it needs):
  sgl-kernel-base       base-commit V4.1 call paths (run with the base tree on PYTHONPATH)
  sgl-kernel            branch TopKSelector(sgl-kernel), default top-k v2 mode
  sgl-kernel-mode2      branch TopKSelector(sgl-kernel) with SGLANG_TOPK_V2_MODE=2 (top-k v2
                        exact and deterministic, ties to the smaller index; DeepSelect and
                        the bf16 sparse kernel do not read it)
  flashinfer            branch TopKSelector(flashinfer), fused, default modes
  flashinfer-small      branch TopKSelector(flashinfer), fused, SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK=small
  flashinfer-small-det  as flashinfer-small plus SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC=1

Suites:
  paged    47960's workloads through the V4.1 decode paged transform (fp32 rows,
           page size 64): GLM-5.2 k 2048, V4-Pro k 1024, V4.1-Flash k 512,
           8K-256K x bs 1-128, plus 1 x 960K and mixed batches
  paged16  V4.1-Flash k 512 with bf16 rows, the SM100 decode logits dtype
  sparse   V4.1-Flash candidate consumer: bf16 rows of 2048 blocks x 8, k 512
  blocks   V4.1-Flash candidate source block top-k (block maxima, k 2048)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import zlib
from contextlib import ExitStack
from types import SimpleNamespace

import torch
import triton
import triton.language as tl

K_BY_MODEL = {"GLM-5.2": 2048, "V4-Pro": 1024, "V4.1-Flash": 512}
CONTEXTS = [8192, 16384, 32768, 65536, 131072, 262144]
BATCHES = [1, 4, 16, 32, 64, 128]
PAGE_SIZE = 64
BLOCK = 8
TOPK_BLOCKS = 2048
TARGET_REPLAY_MS = 3.0
IDLE_GAP_S = 0.005


@triton.jit
def _producer_kernel(SRC, DST, N, BLOCK_N: tl.constexpr, USE_PDL: tl.constexpr):
    offs = tl.arange(0, BLOCK_N)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    for start in range(0, N, BLOCK_N):
        mask = start + offs < N
        tl.store(DST + start + offs, tl.load(SRC + start + offs, mask=mask), mask=mask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def produce(src: torch.Tensor, dst: torch.Tensor) -> None:
    _producer_kernel[(1,)](src, dst, src.numel(), 256, True, num_warps=4, launch_pdl=True)


def shapes(suite: str):
    """(label, model, k, lens list) per shape; lens are rows' valid lengths."""
    out = []
    if suite in ("paged", "paged16"):
        models = K_BY_MODEL if suite == "paged" else {"V4.1-Flash": 512}
        for model, k in models.items():
            for ctx in CONTEXTS:
                for bs in BATCHES:
                    out.append((f"{model} {bs} x {ctx // 1024}K", model, k, [ctx] * bs))
            out.append((f"{model} 1 x 960K", model, k, [960 * 1024]))
            out.append((f"{model} mixed 64 x 1K-256K", model, k, mixed(64, 1024, 262144, model)))
            out.append((f"{model} mixed 16 x 8K-128K", model, k, mixed(16, 8192, 131072, model)))
    elif suite == "sparse":
        width = TOPK_BLOCKS * BLOCK
        for bs in BATCHES + [256]:
            out.append((f"V4.1-Flash sparse {bs} x {width}", "V4.1-Flash", 512, [width] * bs))
        out.append(("V4.1-Flash sparse mixed 64", "V4.1-Flash", 512, mixed(64, 1024, width, "sparse")))
    elif suite == "blocks":
        for ctx in (32768, 65536, 131072, 262144):
            for bs in (1, 4, 16, 64, 128):
                out.append((f"V4.1-Flash blocks {bs} x {ctx // 1024}K", "V4.1-Flash", TOPK_BLOCKS, [ctx] * bs))
    else:
        raise ValueError(suite)
    return out


def mixed(rows: int, low: int, high: int, salt: str):
    g = torch.Generator().manual_seed(zlib.crc32(f"mixed:{rows}:{low}:{high}:{salt}".encode()))
    return torch.randint(low, high + 1, (rows,), generator=g).tolist()


def seed_of(*parts) -> int:
    return zlib.crc32(":".join(map(str, parts)).encode())


def make_inputs(suite: str, label: str, k: int, lens: list):
    dev = "cuda"
    rows = len(lens)
    if suite == "sparse":
        width = TOPK_BLOCKS * BLOCK
    else:
        width = -(-max(lens) // 64) * 64
    g = torch.Generator(device=dev).manual_seed(seed_of(suite, label))
    lens_t = torch.tensor(lens, dtype=torch.int32, device=dev)
    inputs = SimpleNamespace(k=k, lens_src=lens_t.clone(), lens=lens_t, rows=rows, width=width)
    if suite in ("paged", "paged16"):
        dtype = torch.float32 if suite == "paged" else torch.bfloat16
        inputs.scores = torch.randn(rows, width, generator=g, device=dev).to(dtype)
        pages = width // PAGE_SIZE
        inputs.page_table = torch.stack(
            [torch.randperm(4 * pages, generator=g, device=dev)[:pages] for _ in range(rows)]
        ).to(torch.int32)
        inputs.out = torch.empty(rows, k, dtype=torch.int32, device=dev)
    elif suite == "sparse":
        inputs.scores = torch.randn(rows, width, generator=g, device=dev).to(torch.bfloat16)
        blocks = torch.stack(
            [
                torch.randperm(4 * TOPK_BLOCKS, generator=g, device=dev)[:TOPK_BLOCKS].sort().values
                for _ in range(rows)
            ]
        ).to(torch.int32)
        inputs.blocks = blocks
        inputs.out = torch.empty(rows, k, dtype=torch.int32, device=dev)
    elif suite == "blocks":
        inputs.scores = torch.randn(rows, width, generator=g, device=dev).to(torch.bfloat16)
        inputs.nblocks = (lens_t + BLOCK - 1) // BLOCK
    return inputs


def build_call(variant: str, suite: str, inputs):
    """A zero-argument callable launching the variant's selection on ``inputs``."""
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

    if suite in ("paged", "paged16"):
        metadata = SimpleNamespace(
            compressed_seq_lens=inputs.lens,
            page_table=inputs.page_table,
            compressed_page_size=PAGE_SIZE,
            use_topk_v2=True,
            topk_metadata=plan_topk_v2(inputs.lens),
        )
        if variant == "sgl-kernel-base":
            from sglang.srt.layers.attention.dsv4.indexer import (
                topk_transform_paged_from_metadata,
            )

            return lambda: topk_transform_paged_from_metadata(inputs.scores, metadata, inputs.out)
        selector = make_selector(variant)
        return lambda: selector.paged_from_metadata(inputs.scores, metadata, inputs.out)
    if suite == "sparse":
        if variant == "sgl-kernel-base":
            from sglang.kernels.ops.attention.dsv4.topk import topk_transform_sparse

            return lambda: topk_transform_sparse(inputs.scores, inputs.lens, inputs.blocks, inputs.out)
        selector = make_selector(variant)
        return lambda: selector.sparse(inputs.scores, inputs.lens, [(inputs.blocks, inputs.out)])
    if suite == "blocks":
        if variant == "sgl-kernel-base":
            from sglang.kernels.ops.attention.dsv4.candidate_blocks import amax_topk_blocks

            return lambda: amax_topk_blocks(inputs.scores, inputs.lens, inputs.nblocks, TOPK_BLOCKS)
        selector = make_selector(variant)
        return lambda: selector.amax_blocks(inputs.scores, inputs.lens, inputs.nblocks, TOPK_BLOCKS)
    raise ValueError(suite)


_SELECTORS = {}


def make_selector(variant: str):
    if variant in _SELECTORS:
        return _SELECTORS[variant]
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
    from sglang.srt.layers.attention.dsv4.v41_indexer.selector import TopKSelector

    from sglang.srt.environ import envs

    mode = envs.SGLANG_TOPK_V2_MODE.get()
    tie_break = envs.SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK.get()
    deterministic = envs.SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC.get()
    if variant in ("sgl-kernel", "sgl-kernel-mode2"):
        want_mode = 2 if variant == "sgl-kernel-mode2" else 1
        assert mode == want_mode, f"{variant} needs SGLANG_TOPK_V2_MODE={want_mode}"
        selector = TopKSelector(DSATopKBackend.SGL_KERNEL)
    elif variant in ("flashinfer", "flashinfer-small", "flashinfer-small-det"):
        want = {
            "flashinfer": (None, False),
            "flashinfer-small": ("small", False),
            "flashinfer-small-det": ("small", True),
        }[variant]
        assert (tie_break, deterministic) == want, f"{variant} needs tie-break/deterministic {want}"
        selector = TopKSelector(DSATopKBackend.FLASHINFER)
        assert selector.flashinfer_fused, "SGLANG_DSA_FUSE_TOPK must stay on"
    else:
        raise ValueError(variant)
    _SELECTORS[variant] = selector
    return selector


def capture(call, inputs, repeats: int) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            produce(inputs.lens_src, inputs.lens)
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            produce(inputs.lens_src, inputs.lens)
            call()
    return graph


def replay_us(graph, repeats: int, rounds: int):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        time.sleep(IDLE_GAP_S)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / repeats)
    return samples


def run_shape(variant, suite, label, k, lens, rounds):
    inputs = make_inputs(suite, label, k, lens)
    call = build_call(variant, suite, inputs)
    probe_repeats = 8
    probe = capture(call, inputs, probe_repeats)
    per_call_us = statistics.median(replay_us(probe, probe_repeats, 5))
    del probe
    repeats = max(1, min(4000, round(TARGET_REPLAY_MS * 1000.0 / per_call_us)))
    graph = capture(call, inputs, repeats)
    samples = replay_us(graph, repeats, rounds)
    del graph
    return repeats, samples


def environment(tree: str):
    import flashinfer

    props = torch.cuda.get_device_properties(0)
    commit = ""
    try:
        commit = open(os.path.join(tree, "COMMIT")).read().strip()
    except OSError:
        pass
    return {
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": flashinfer.__version__,
        "triton": triton.__version__,
        "tree": tree,
        "tree_commit": commit,
        "env": {
            key: os.environ.get(key)
            for key in (
                "SGLANG_DSA_FUSE_TOPK",
                "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK",
                "SGLANG_DSA_TOPK_FLASHINFER_DETERMINISTIC",
                "SGLANG_TOPK_V2_MODE",
                "SGLANG_OPT_USE_TOPK_V2",
                "FLASHINFER_TOPK_ALGO",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--suite", nargs="+", default=["paged", "paged16", "sparse", "blocks"])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--only", nargs="*", default=None, help="labels to run")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tree", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    with open(args.out, "a") as f:
        f.write(json.dumps({"variant": args.variant, "environment": environment(args.tree)}) + "\n")
        for suite in args.suite:
            for label, model, k, lens in shapes(suite):
                if args.only is not None and label not in args.only:
                    continue
                repeats, samples = run_shape(args.variant, suite, label, k, lens, args.rounds)
                record = {
                    "variant": args.variant,
                    "suite": suite,
                    "label": label,
                    "model": model,
                    "k": k,
                    "rows": len(lens),
                    "max_len": max(lens),
                    "repeats": repeats,
                    "rounds": args.rounds,
                    "samples_us": samples,
                    "median_us": statistics.median(samples),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"{args.variant:21s} {suite:7s} {label:36s} R={repeats:5d} {record['median_us']:9.3f} us", flush=True)


if __name__ == "__main__":
    main()
