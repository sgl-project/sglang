"""Compare paged attention implementations on short multi-token queries."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import time
from unittest.mock import patch
from pathlib import Path

import flashinfer
import torch


def measure(functions, repeats, replays):
    for fn in functions:
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in functions:
            fn()
    for _ in range(4):
        graph.replay()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / replays / len(functions))
    return samples


def reference(q, k, v, length):
    q64 = q.double().transpose(0, 1)
    k64 = k[:, 0].reshape(-1, k.shape[-1])[:length].double()
    v64 = v[:, 0].reshape(-1, v.shape[-1])[:length].double()
    scores = torch.matmul(q64, k64.T) / math.sqrt(q.shape[-1])
    pos = torch.arange(length, device=q.device)
    ends = length - q.shape[0] + torch.arange(q.shape[0], device=q.device)
    scores.masked_fill_(pos[None, None, :] > ends[None, :, None], -float("inf"))
    return torch.matmul(scores.softmax(-1), v64).transpose(0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", default="32768,49152,65536,131072,168960")
    parser.add_argument("--heads", default="4,8")
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--other-length", type=int)
    parser.add_argument("--qlen", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-pages", action="store_true")
    parser.add_argument("--arms", default="stock")
    parser.add_argument("--kernel")
    parser.add_argument("--vendor")
    parser.add_argument("--vendor-tile", type=int, default=64)
    parser.add_argument(
        "--vendor-reduction", choices=("gmem", "separate", "cga"), default="gmem"
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    vendor = None
    vendor_fingerprint = None
    if args.vendor:
        spec = importlib.util.spec_from_file_location("attention_vendor", args.vendor)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        vendor, vendor_fingerprint = module.build_variant()
    torch.manual_seed(args.seed)
    workspace = torch.empty(256 * 1024**2, device="cuda", dtype=torch.uint8)
    counter = torch.zeros(8 * 1024**2, device="cuda", dtype=torch.uint8)
    records = []
    metadata = dict(
        torch=torch.__version__,
        flashinfer=flashinfer.__version__,
        device=torch.cuda.get_device_name(),
        mode="static microbenchmark",
        workspace_reset_before_arm=True,
        script_md5=hashlib.md5(Path(__file__).read_bytes()).hexdigest(),
        vendor_fingerprint=vendor_fingerprint,
        kernel_md5=hashlib.md5(Path(args.kernel).read_bytes()).hexdigest()
        if args.kernel
        else None,
        args=vars(args),
        started=time.time(),
    )
    for heads in map(int, args.heads.split(",")):
        for length in map(int, args.lengths.split(",")):
            lengths = [length] * args.batch
            if args.other_length is not None:
                lengths[1:] = [args.other_length] * (args.batch - 1)
            page_counts = [(v + 63) // 64 for v in lengths]
            pages = max(page_counts)
            total_pages = sum(page_counts)
            q = torch.randn(args.batch * args.qlen, heads, 256, device="cuda").to(
                torch.float8_e4m3fn
            )
            banks = [
                (
                    torch.randn(total_pages, 1, 64, 256, device="cuda").to(q.dtype),
                    torch.randn(total_pages, 1, 64, 256, device="cuda").to(q.dtype),
                )
                for _ in range(args.banks)
            ]
            table = torch.zeros(args.batch, pages, device="cuda", dtype=torch.int32)
            page_order = (
                torch.randperm(total_pages, device="cuda", dtype=torch.int32)
                if args.shuffle_pages
                else torch.arange(total_pages, device="cuda", dtype=torch.int32)
            )
            offset = 0
            reference_rows = []
            for index, count in enumerate(page_counts):
                selected = page_order[offset : offset + count]
                table[index, :count] = selected
                if args.reference:
                    reference_rows.append(
                        reference(
                            q[index * args.qlen : (index + 1) * args.qlen],
                            banks[0][0].index_select(0, selected.long()),
                            banks[0][1].index_select(0, selected.long()),
                            lengths[index],
                        )
                    )
                offset += count
            seq = torch.tensor(lengths, device="cuda", dtype=torch.int32)
            ref = torch.cat(reference_rows) if args.reference else None
            arms = []
            if "stock" in args.arms.split(","):
                for maximum in sorted(set((max(lengths), 262144))):
                    outputs = [
                        torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                        for _ in banks
                    ]
                    functions = [
                        lambda kv=kv, o=o, maximum=maximum: (
                            flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                                q,
                                kv,
                                workspace,
                                table,
                                seq,
                                maximum,
                                bmm1_scale=1 / 16,
                                bmm2_scale=1.0,
                                out=o,
                                q_len_per_req=args.qlen,
                                multi_ctas_kv_counter_buffer=counter,
                            )
                        )
                        for kv, o in zip(banks, outputs)
                    ]
                    arms.append((f"trtllm_max{maximum}", functions, outputs))
            for arm in args.arms.split(","):
                if arm == "stock":
                    continue
                try:
                    if arm.startswith(
                        (
                            "split:",
                            "split-t:",
                            "split-b:",
                            "split-e:",
                            "split-s:",
                            "split-a:",
                        )
                    ):
                        _, split, bm, bn, *tuning = arm.split(":")
                        split, bm, bn = int(split), int(bm), int(bn)
                        tuning_kwargs = (
                            dict(num_stages=int(tuning[0]), num_warps=int(tuning[1]))
                            if tuning
                            else {}
                        )
                        tuning_kwargs["transposed"] = arm.startswith("split-t:")
                        tuning_kwargs["balanced"] = arm.startswith("split-b:")
                        tuning_kwargs["skip_empty"] = arm.startswith(
                            ("split-e:", "split-s:", "split-a:")
                        )
                        tuning_kwargs["short_merge"] = arm.startswith(
                            ("split-s:", "split-a:")
                        )
                        tuning_kwargs["short_tiles"] = arm.startswith("split-a:")
                        if len(tuning) > 2:
                            tuning_kwargs.update(
                                merge_tile=int(tuning[2]), merge_warps=int(tuning[3])
                            )
                        spec = importlib.util.spec_from_file_location(
                            "split_attention_kernel", args.kernel
                        )
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        scratch = module.allocate_scratch(
                            q, args.batch, split, args.qlen
                        )
                        outputs = [
                            torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                            for _ in banks
                        ]
                        functions = [
                            lambda kv=kv, o=o, scratch=scratch, split=split, bm=bm, bn=bn, module=module, tuning_kwargs=tuning_kwargs: (
                                module.split_kv_verify(
                                    q,
                                    kv,
                                    table,
                                    seq,
                                    o,
                                    scratch,
                                    splits=split,
                                    qlen=args.qlen,
                                    block_m=bm,
                                    block_n=bn,
                                    **tuning_kwargs,
                                )
                            )
                            for kv, o in zip(banks, outputs)
                        ]
                        arms.append((arm, functions, outputs))
                        continue
                    backend, split_text, *variant_config = arm.split(":")
                    split = int(split_text)
                    if backend in ("trtllm-bf16", "trtllm-k-only", "xqa", "cake"):
                        q16 = q.to(torch.bfloat16)
                        selected_backend = (
                            backend if backend in ("xqa", "cake") else "trtllm-gen"
                        )
                        transform_mode = (
                            "k_only" if backend == "trtllm-k-only" else None
                        )
                        probe_mask = None
                        if selected_backend == "xqa" and args.qlen > 1:
                            words = 2 * math.ceil(args.qlen / 32)
                            probe_mask = (
                                torch.tensor(
                                    [
                                        [
                                            (1 << min(max(row + 1 - 16 * word, 0), 16))
                                            - 1
                                            for word in range(words)
                                        ]
                                        for row in range(args.qlen)
                                    ],
                                    device="cuda",
                                    dtype=torch.uint16,
                                )[None]
                                .expand(args.batch, -1, -1)
                                .contiguous()
                            )
                        outputs = [
                            torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                            for _ in banks
                        ]
                        functions = [
                            lambda kv=kv, o=o, q16=q16, selected_backend=selected_backend, probe_mask=probe_mask, transform_mode=transform_mode: (
                                flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                                    q16,
                                    kv,
                                    workspace,
                                    table,
                                    seq,
                                    262144,
                                    bmm1_scale=1 / 16,
                                    bmm2_scale=1.0,
                                    out=o,
                                    q_len_per_req=args.qlen,
                                    backend=selected_backend,
                                    mask=probe_mask,
                                    bf16q_fp8kv_transform_mode=transform_mode,
                                    multi_ctas_kv_counter_buffer=counter,
                                )
                            )
                            for kv, o in zip(banks, outputs)
                        ]
                        arms.append((arm, functions, outputs))
                        continue
                    if backend == "cap":
                        outputs = [
                            torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                            for _ in banks
                        ]

                        def run_cap(kv, out, cap):
                            os.environ["Q35MF_ATTENTION_SPLIT_CAP"] = str(cap)
                            with patch.object(
                                flashinfer.decode,
                                "get_trtllm_gen_fmha_module",
                                return_value=vendor,
                            ):
                                return (
                                    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                                        q,
                                        kv,
                                        workspace,
                                        table,
                                        seq,
                                        262144,
                                        bmm1_scale=1 / 16,
                                        bmm2_scale=1.0,
                                        out=out,
                                        q_len_per_req=args.qlen,
                                        multi_ctas_kv_counter_buffer=counter,
                                    )
                                )

                        functions = [
                            lambda kv=kv, o=o, cap=split: run_cap(kv, o, cap)
                            for kv, o in zip(banks, outputs)
                        ]
                        arms.append((arm, functions, outputs))
                        continue
                    if backend == "vendor":
                        outputs = [
                            torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                            for _ in banks
                        ]

                        tile, mode = (
                            map(int, variant_config)
                            if variant_config
                            else (
                                args.vendor_tile,
                                {"gmem": 1, "separate": 2, "cga": 3}[
                                    args.vendor_reduction
                                ],
                            )
                        )

                        def run_vendor(kv, out, budget, tile, mode):
                            os.environ["SGLANG_Q35MF_ATTN_PROBE_TILE"] = str(tile)
                            os.environ["SGLANG_Q35MF_ATTN_PROBE_REDUCTION"] = str(mode)
                            with patch.object(
                                flashinfer.decode,
                                "get_trtllm_gen_fmha_module",
                                return_value=vendor,
                            ):

                                def call():
                                    return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                                        q,
                                        kv,
                                        workspace,
                                        table,
                                        seq,
                                        262144,
                                        bmm1_scale=1 / 16,
                                        out=out,
                                        q_len_per_req=args.qlen,
                                        multi_ctas_kv_counter_buffer=counter,
                                    )

                                if budget:
                                    with patch.object(
                                        flashinfer.decode,
                                        "get_device_sm_count",
                                        return_value=budget,
                                    ):
                                        return call()
                                return call()

                        functions = [
                            lambda kv=kv, o=o, split=split, tile=tile, mode=mode: (
                                run_vendor(kv, o, split, tile, mode)
                            )
                            for kv, o in zip(banks, outputs)
                        ]
                        arms.append((arm, functions, outputs))
                        continue
                    if args.batch != 1:
                        raise NotImplementedError(
                            "This backend arm supports batch one only"
                        )
                    if backend == "token-batch":
                        outputs = [
                            torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                            for _ in banks
                        ]
                        token_table = table.expand(args.qlen, -1).contiguous()
                        token_lengths = (
                            seq
                            - args.qlen
                            + torch.arange(
                                1, args.qlen + 1, device="cuda", dtype=torch.int32
                            )
                        )

                        def run_tokens(kv, out, budget):
                            def call():
                                return (
                                    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                                        q,
                                        kv,
                                        workspace,
                                        token_table,
                                        token_lengths,
                                        262144,
                                        bmm1_scale=1 / 16,
                                        out=out,
                                        q_len_per_req=1,
                                        multi_ctas_kv_counter_buffer=counter,
                                    )
                                )

                            if budget:
                                with patch.object(
                                    flashinfer.decode,
                                    "get_device_sm_count",
                                    return_value=budget,
                                ):
                                    return call()
                            return call()

                        functions = [
                            lambda kv=kv, o=o, split=split: run_tokens(kv, o, split)
                            for kv, o in zip(banks, outputs)
                        ]
                        arms.append((arm, functions, outputs))
                        continue
                    if backend == "trtllm-sm":
                        outputs = [
                            torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                            for _ in banks
                        ]

                        def run_budget(kv, out, budget):
                            with patch.object(
                                flashinfer.decode,
                                "get_device_sm_count",
                                return_value=budget,
                            ):
                                return (
                                    flashinfer.decode.trtllm_batch_decode_with_kv_cache(
                                        q,
                                        kv,
                                        workspace,
                                        table,
                                        seq,
                                        262144,
                                        bmm1_scale=1 / 16,
                                        out=out,
                                        q_len_per_req=args.qlen,
                                        multi_ctas_kv_counter_buffer=counter,
                                    )
                                )

                        functions = [
                            lambda kv=kv, o=o, split=split: run_budget(kv, o, split)
                            for kv, o in zip(banks, outputs)
                        ]
                        arms.append((arm, functions, outputs))
                        continue
                    qo_indptr = torch.tensor(
                        [0, args.qlen], device="cuda", dtype=torch.int32
                    )
                    indptr = torch.tensor([0, pages], device="cuda", dtype=torch.int32)
                    last = torch.tensor(
                        [(length - 1) % 64 + 1], device="cuda", dtype=torch.int32
                    )
                    q16 = q.to(torch.bfloat16)
                    if backend == "cute-dsl":
                        from flashinfer.cute_dsl.attention.wrappers.batch_decode import (
                            BatchDecodePagedCuteDSLWrapper,
                        )

                        wrapper = BatchDecodePagedCuteDSLWrapper(workspace)
                        wrapper.plan(
                            indptr,
                            table[0],
                            seq,
                            heads,
                            1,
                            256,
                            64,
                            q_data_type=q.dtype,
                            kv_data_type=q.dtype,
                            o_data_type=torch.bfloat16,
                            q_len_per_req=args.qlen,
                            kv_splits=split,
                            reduction="kernel",
                            max_kv_len=length,
                        )
                        q16 = q
                        outputs = [
                            torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                            for _ in banks
                        ]
                        functions = [
                            lambda kv=kv, o=o, wrapper=wrapper: wrapper.run(
                                q,
                                kv[0].permute(0, 2, 1, 3),
                                kv[1].permute(0, 2, 1, 3),
                                out=o,
                            )
                            for kv, o in zip(banks, outputs)
                        ]
                        arms.append((arm, functions, outputs))
                        continue
                    elif backend.startswith("prefill-"):
                        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                            workspace,
                            kv_layout="HND",
                            backend=backend.removeprefix("prefill-"),
                        )
                        wrapper.plan(
                            qo_indptr,
                            indptr,
                            table[0],
                            last,
                            heads,
                            1,
                            256,
                            64,
                            causal=True,
                            q_data_type=q16.dtype,
                            kv_data_type=q.dtype,
                            o_data_type=torch.bfloat16,
                            fixed_split_size=split if split > 0 else None,
                        )
                    else:
                        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                            workspace,
                            kv_layout="HND",
                            use_tensor_cores=True,
                            backend=backend,
                        )
                        kwargs = dict(
                            q_data_type=q16.dtype,
                            kv_data_type=q.dtype,
                            o_data_type=torch.bfloat16,
                            q_len_per_req=args.qlen,
                        )
                        if backend == "cute-dsl":
                            kwargs.update(
                                kv_splits=split if split > 0 else None,
                                reduction="kernel",
                            )
                        elif split > 0:
                            kwargs["fixed_split_size"] = split
                        wrapper.plan(
                            indptr, table[0], last, heads, 1, 256, 64, **kwargs
                        )
                    outputs = [
                        torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
                        for _ in banks
                    ]
                    functions = [
                        lambda kv=kv, o=o, wrapper=wrapper, q16=q16: wrapper.run(
                            q16, kv, out=o
                        )
                        for kv, o in zip(banks, outputs)
                    ]
                    arms.append((arm, functions, outputs))
                except Exception as exc:
                    print(
                        json.dumps(
                            dict(
                                arm=arm,
                                length=length,
                                heads=heads,
                                error=str(exc),
                                verdict="UNAVAILABLE",
                            )
                        ),
                        flush=True,
                    )
            for name, functions, outputs in arms:
                try:
                    workspace.zero_()
                    counter.zero_()
                    samples = measure(functions, args.repeats, args.replays)
                except Exception as exc:
                    row = dict(
                        heads=heads,
                        length=length,
                        arm=name,
                        error=str(exc),
                        verdict="UNAVAILABLE",
                    )
                    records.append(row)
                    print(json.dumps(row), flush=True)
                    Path(args.output).write_text(
                        json.dumps(dict(metadata=metadata, records=records), indent=2)
                    )
                    continue
                median = statistics.median(samples)
                row = dict(
                    heads=heads,
                    kv_heads=1,
                    head_dim=256,
                    q_len=args.qlen,
                    length=length,
                    arm=name,
                    median_us=median,
                    min_us=min(samples),
                    max_us=max(samples),
                    samples_us=samples,
                    logical_kv_TB_s=sum(lengths) * 512 / median / 1e6,
                    batch=args.batch,
                    lengths=lengths,
                    verdict="PENDING",
                    banks=args.banks,
                )
                if ref is not None:
                    error = outputs[0].double() - ref
                    row["fp64_rel_l2"] = (error.norm() / ref.norm()).item()
                    row["fp64_max_abs"] = error.abs().max().item()
                    row["fp64_mean_row_rel_l2"] = (
                        (error.norm(dim=-1) / ref.norm(dim=-1)).mean().item()
                    )
                records.append(row)
                print(json.dumps(row), flush=True)
                Path(args.output).write_text(
                    json.dumps(dict(metadata=metadata, records=records), indent=2)
                )


if __name__ == "__main__":
    main()
