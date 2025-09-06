#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare dense vs sparse attention serving on the same model (e.g., MiniCPM-4).

Usage examples
--------------
# 1) 两个服务端分别跑在 30000(dense), 30001(sparse)，模型相同（MiniCPM-4）：
python3 bench_sparse_vs_dense.py \
  --dense-base-url http://127.0.0.1:30000 \
  --sparse-base-url http://127.0.0.1:30001 \
  --model OpenBMB/MiniCPM-4 \
  --dataset-name random --num-prompts 200 \
  --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
  --request-rate inf --max-concurrency 64

# 2) 复用 ShareGPT：
python3 bench_sparse_vs_dense.py \
  --dense-base-url http://127.0.0.1:30000 \
  --sparse-base-url http://127.0.0.1:30001 \
  --model OpenBMB/MiniCPM-4 \
  --dataset-name sharegpt --sharegpt-output-len 256 --num-prompts 500

Notes
-----
- 你需要提前启动两个 sglang 服务端：
  Dense 端：不打开 sparse（例如未设置 SGLANG_INFLLM_ENABLE，或设为 0）
  Sparse 端：开启你实现的稀疏注意力（例如 SGLANG_INFLLM_ENABLE=1 等）
- 本脚本只作为客户端对比，同一机器同一 GPU 资源更有可比性。
"""

import argparse
import json
import math
from types import SimpleNamespace
from typing import Dict, Any, Tuple

# 直接复用 sglang 的单次基准逻辑
from sglang.bench_serving import run_benchmark as _run_one


def _mk_args(
    base_url: str,
    backend: str,
    model: str | None,
    tokenizer: str | None,
    dataset_name: str,
    num_prompts: int,
    sharegpt_output_len: int | None,
    sharegpt_context_len: int | None,
    random_input_len: int,
    random_output_len: int,
    random_range_ratio: float,
    request_rate: float,
    max_concurrency: int | None,
    disable_tqdm: bool,
    disable_stream: bool,
    return_logprob: bool,
    seed: int,
    disable_ignore_eos: bool,
    extra_request_body: str | None,
    apply_chat_template: bool,
    lora_names: list[str] | None,
    prompt_suffix: str,
    profile: bool,
    pd_separated: bool,
    flush_cache: bool,
    warmup_requests: int,
    output_file: str | None,
    output_details: bool,
    dataset_path: str,
    random_image_num_images: int,
    random_image_resolution: str,
    tokenize_prompt: bool,
    gsp_num_groups: int,
    gsp_prompts_per_group: int,
    gsp_system_prompt_len: int,
    gsp_question_len: int,
    gsp_output_len: int,
) -> SimpleNamespace:
    # run_benchmark 期望的参数集合（尽量与 bench_serving 的 argparse 对齐）
    return SimpleNamespace(
        backend=backend,
        base_url=base_url,
        host="0.0.0.0",
        port=None,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        model=model,
        tokenizer=tokenizer,
        num_prompts=num_prompts,
        sharegpt_output_len=sharegpt_output_len,
        sharegpt_context_len=sharegpt_context_len,
        random_input_len=random_input_len,
        random_output_len=random_output_len,
        random_range_ratio=random_range_ratio,
        request_rate=request_rate,
        max_concurrency=max_concurrency,
        output_file=output_file,
        output_details=output_details,
        disable_tqdm=disable_tqdm,
        disable_stream=disable_stream,
        return_logprob=return_logprob,
        seed=seed,
        disable_ignore_eos=disable_ignore_eos,
        extra_request_body=extra_request_body,
        apply_chat_template=apply_chat_template,
        profile=profile,
        lora_name=lora_names,
        prompt_suffix=prompt_suffix,
        pd_separated=pd_separated,
        flush_cache=flush_cache,
        warmup_requests=warmup_requests,
        # random-image
        random_image_num_images=random_image_num_images,
        random_image_resolution=random_image_resolution,
        # tokenize prompt
        tokenize_prompt=tokenize_prompt,
        # generated-shared-prefix
        gsp_num_groups=gsp_num_groups,
        gsp_prompts_per_group=gsp_prompts_per_group,
        gsp_system_prompt_len=gsp_system_prompt_len,
        gsp_question_len=gsp_question_len,
        gsp_output_len=gsp_output_len,
    )


def _safe_get(d: Dict[str, Any], k: str, default=None):
    return d.get(k, default) if isinstance(d, dict) else default


def _fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, (float, int)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "-"
        return f"{x:.{nd}f}"
    return str(x)


def _speedup(new, base):
    if new is None or base is None:
        return "-"
    if base == 0:
        return "-"
    return f"{(new/base):.2f}×"


def _print_side_by_side(title: str, rows: Tuple[Tuple[str, Any, Any]]):
    w = 78
    print("\n" + "=" * w)
    print(f"{title}".center(w))
    print("=" * w)
    print(f"{'Metric':<35} {'Dense':>12} {'Sparse':>12} {'Speedup':>12}")
    print("-" * w)
    for name, dv, sv in rows:
        print(f"{name:<35} { _fmt(dv):>12} { _fmt(sv):>12} { _speedup(sv, dv):>12}")
    print("=" * w)


def main():
    ap = argparse.ArgumentParser("Dense vs Sparse attention benchmark (client-side)")
    # 两个服务端地址
    ap.add_argument("--dense-base-url", type=str, required=True,
                    help="Dense server base URL, e.g. http://127.0.0.1:30000")
    ap.add_argument("--sparse-base-url", type=str, required=True,
                    help="Sparse server base URL, e.g. http://127.0.0.1:30001")

    # 模型与分词器
    ap.add_argument("--model", type=str, default=None,
                    help="Model id/path; if None, the server /v1/models will be used.")
    ap.add_argument("--tokenizer", type=str, default=None)

    # 数据集/流量
    ap.add_argument("--dataset-name", type=str, default="random",
                    choices=["sharegpt","random","random-ids","generated-shared-prefix","mmmu","random-image"])
    ap.add_argument("--dataset-path", type=str, default="")
    ap.add_argument("--num-prompts", type=int, default=1000)
    ap.add_argument("--sharegpt-output-len", type=int, default=None)
    ap.add_argument("--sharegpt-context-len", type=int, default=None)
    ap.add_argument("--random-input-len", type=int, default=1024)
    ap.add_argument("--random-output-len", type=int, default=1024)
    ap.add_argument("--random-range-ratio", type=float, default=0.5)
    ap.add_argument("--request-rate", type=float, default=float("inf"))
    ap.add_argument("--max-concurrency", type=int, default=64)

    # 其他通用选项（保持与 bench_serving 一致）
    ap.add_argument("--disable-tqdm", action="store_true")
    ap.add_argument("--disable-stream", action="store_true")
    ap.add_argument("--return-logprob", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--disable-ignore-eos", action="store_true")
    ap.add_argument("--extra-request-body", type=str, default=None,
                    help='JSON string to append to request payload, e.g. \'{"temperature":0}\'')
    ap.add_argument("--apply-chat-template", action="store_true")
    ap.add_argument("--lora-name", type=str, nargs="*", default=None)
    ap.add_argument("--prompt-suffix", type=str, default="")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--pd-separated", action="store_true")
    ap.add_argument("--flush-cache", action="store_true")
    ap.add_argument("--warmup-requests", type=int, default=1)
    ap.add_argument("--output-file", type=str, default=None)
    ap.add_argument("--output-details", action="store_true")
    ap.add_argument("--random-image-num-images", type=int, default=1)
    ap.add_argument("--random-image-resolution", type=str, default="1080p")
    ap.add_argument("--tokenize-prompt", action="store_true")

    # generated-shared-prefix
    ap.add_argument("--gsp-num-groups", type=int, default=64)
    ap.add_argument("--gsp-prompts-per-group", type=int, default=16)
    ap.add_argument("--gsp-system-prompt-len", type=int, default=2048)
    ap.add_argument("--gsp-question-len", type=int, default=128)
    ap.add_argument("--gsp-output-len", type=int, default=256)

    # 导出
    ap.add_argument("--save-json", type=str, default=None,
                    help="Optional: save both results & comparison as JSON to this file")

    args = ap.parse_args()

    backend = "sglang"  # 对齐 bench_serving 默认

    # dense run
    dense_ns = _mk_args(
        base_url=args.dense_base_url,
        backend=backend,
        model=args.model,
        tokenizer=args.tokenizer,
        dataset_name=args.dataset_name,
        num_prompts=args.num_prompts,
        sharegpt_output_len=args.sharegpt_output_len,
        sharegpt_context_len=args.sharegpt_context_len,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        random_range_ratio=args.random_range_ratio,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        disable_tqdm=True if not args.profile else False,  # 开 profiler 时保留输出
        disable_stream=args.disable_stream,
        return_logprob=args.return_logprob,
        seed=args.seed,
        disable_ignore_eos=args.disable_ignore_eos,
        extra_request_body=args.extra_request_body,
        apply_chat_template=args.apply_chat_template,
        lora_names=args.lora_name,
        prompt_suffix=args.prompt_suffix,
        profile=args.profile,
        pd_separated=args.pd_separated,
        flush_cache=args.flush_cache,
        warmup_requests=args.warmup_requests,
        output_file=None,  # 不让两次 run 写同一文件
        output_details=False,
        dataset_path=args.dataset_path,
        random_image_num_images=args.random_image_num_images,
        random_image_resolution=args.random_image_resolution,
        tokenize_prompt=args.tokenize_prompt,
        gsp_num_groups=args.gsp_num_groups,
        gsp_prompts_per_group=args.gsp_prompts_per_group,
        gsp_system_prompt_len=args.gsp_system_prompt_len,
        gsp_question_len=args.gsp_question_len,
        gsp_output_len=args.gsp_output_len,
    )
    print("\n====== DENSE run ======")
    dense_res: Dict[str, Any] = _run_one(dense_ns)

    # sparse run
    sparse_ns = _mk_args(
        base_url=args.sparse_base_url,
        backend=backend,
        model=args.model,
        tokenizer=args.tokenizer,
        dataset_name=args.dataset_name,
        num_prompts=args.num_prompts,
        sharegpt_output_len=args.sharegpt_output_len,
        sharegpt_context_len=args.sharegpt_context_len,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        random_range_ratio=args.random_range_ratio,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        disable_tqdm=True if not args.profile else False,
        disable_stream=args.disable_stream,
        return_logprob=args.return_logprob,
        seed=args.seed,
        disable_ignore_eos=args.disable_ignore_eos,
        extra_request_body=args.extra_request_body,
        apply_chat_template=args.apply_chat_template,
        lora_names=args.lora_name,
        prompt_suffix=args.prompt_suffix,
        profile=args.profile,
        pd_separated=args.pd_separated,
        flush_cache=args.flush_cache,
        warmup_requests=args.warmup_requests,
        output_file=None,
        output_details=False,
        dataset_path=args.dataset_path,
        random_image_num_images=args.random_image_num_images,
        random_image_resolution=args.random_image_resolution,
        tokenize_prompt=args.tokenize_prompt,
        gsp_num_groups=args.gsp_num_groups,
        gsp_prompts_per_group=args.gsp_prompts_per_group,
        gsp_system_prompt_len=args.gsp_system_prompt_len,
        gsp_question_len=args.gsp_question_len,
        gsp_output_len=args.gsp_output_len,
    )
    print("\n====== SPARSE run ======")
    sparse_res: Dict[str, Any] = _run_one(sparse_ns)

    # 取关键信息
    def K(d: Dict[str, Any], k: str): return _safe_get(d, k)

    rows = (
        ("Request throughput (req/s)",  K(dense_res, "request_throughput"), K(sparse_res, "request_throughput")),
        ("Input tok/s",                  K(dense_res, "input_throughput"),   K(sparse_res, "input_throughput")),
        ("Output tok/s",                 K(dense_res, "output_throughput"),  K(sparse_res, "output_throughput")),
        ("Total tok/s",                  K(dense_res, "total_throughput"),   K(sparse_res, "total_throughput")),
        ("Mean TTFT (ms)",               K(dense_res, "mean_ttft_ms"),       K(sparse_res, "mean_ttft_ms")),
        ("P99 TTFT (ms)",                K(dense_res, "p99_ttft_ms"),        K(sparse_res, "p99_ttft_ms")),
        ("Mean ITL (ms)",                K(dense_res, "mean_itl_ms"),        K(sparse_res, "mean_itl_ms")),
        ("P95 ITL (ms)",                 K(dense_res, "p95_itl_ms"),         K(sparse_res, "p95_itl_ms")),
        ("Mean E2E Latency (ms)",        K(dense_res, "mean_e2e_latency_ms"),K(sparse_res, "mean_e2e_latency_ms")),
        ("Concurrency",                  K(dense_res, "concurrency"),        K(sparse_res, "concurrency")),
        ("Accept length",                K(dense_res, "accept_length"),      K(sparse_res, "accept_length")),
        ("Completed requests",           K(dense_res, "completed"),          K(sparse_res, "completed")),
        ("Total output tokens",          K(dense_res, "total_output_tokens"),K(sparse_res, "total_output_tokens")),
    )
    _print_side_by_side("Dense vs Sparse — Summary", rows)

    # 可选保存 JSON
    if args.save_json:
        payload = {
            "dense": dense_res,
            "sparse": sparse_res,
            "compare": {name: {"dense": dv, "sparse": sv} for (name, dv, sv) in rows},
            "meta": {
                "dense_base_url": args.dense_base_url,
                "sparse_base_url": args.sparse_base_url,
                "model": args.model,
                "dataset": args.dataset_name,
                "num_prompts": args.num_prompts,
                "request_rate": args.request_rate,
                "max_concurrency": args.max_concurrency,
            },
        }
        with open(args.save_json, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved comparison JSON to: {args.save_json}")


if __name__ == "__main__":
    main()



# dense
# CUDA_VISIBLE_DEVICES=0 \
# SGLANG_INFLLM_ENABLE=0 \
# python3 -m sglang.launch_server \
#   --model OpenBMB/MiniCPM-4 \
#   --port 30000


# infllmv2
# CUDA_VISIBLE_DEVICES=0 \
# SGLANG_INFLLM_ENABLE=1 \
# SGLANG_INFLLM_BACKEND=triton \
# SGLANG_INFLLM_TOPK=8 \
# SGLANG_INFLLM_BLOCK=64 \
# SGLANG_INFLLM_SW_SPAN=2048 \
# SGLANG_INFLLM_SINK_LEN=64 \
# python3 -m sglang.launch_server \
#   --model OpenBMB/MiniCPM-4 \
#   --port 30001


# vs 示例随机数据集 200 条、1024/1024
# python3 bench_sparse_vs_dense.py \
#   --dense-base-url http://127.0.0.1:30000 \
#   --sparse-base-url http://127.0.0.1:30001 \
#   --model OpenBMB/MiniCPM-4 \
#   --dataset-name random --num-prompts 200 \
#   --random-input-len 1024 --random-output-len 1024 --random-range-ratio 0.5 \
#   --request-rate inf --max-concurrency 64 \
#   --save-json cmp_minicpm4_random_1024x1024.json
