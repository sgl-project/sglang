"""DSpark candidate sampling microbenchmark (synthetic weights, TP=1).

This does not measure an LM head, target verification, serving throughput, or
checkpoint acceptance rate. Shapes must be supplied from inspected checkpoint
metadata. See dspark_markov_candidates_results.md for reproduction and limits.
"""

import argparse
import importlib.metadata
import inspect
import json
import math
import platform
import statistics
import subprocess
import sys
import time

import torch


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _metadata(args):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    try:
        driver = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {
        "command": sys.argv,
        "commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": _version("triton"),
        "flashinfer": _version("flashinfer-python"),
        "gpu": torch.cuda.get_device_name(),
        "device_capability": torch.cuda.get_device_capability(),
        "driver_versions": driver,
        "arguments": vars(args),
        "weights": "synthetic; not a checkpoint evaluation",
        "sampling": {
            "temperature": args.temperature,
            "target_top_k": -1,
            "target_top_p": 1.0,
        },
        "scope": "base logits to proposal IDs and cache; excludes LM head and verifier",
    }


def _summary(samples):
    samples = sorted(float(x) for x in samples)
    return {
        "p50_us": statistics.median(samples),
        "p95_us": samples[min(len(samples) - 1, math.ceil(len(samples) * 0.95) - 1)],
        "samples": len(samples),
    }


def _measure(fn, args, flush):
    # Compilations and lazy allocations finish before capture or timed calls.
    for _ in range(args.warmup):
        fn()
    torch.cuda.synchronize()
    graph = None
    capture_ms = None
    if args.graph:
        graph = torch.cuda.CUDAGraph()
        started = time.perf_counter()
        with torch.cuda.graph(graph):
            fn()
        torch.cuda.synchronize()
        capture_ms = (time.perf_counter() - started) * 1000
        fn = graph.replay
    timing = args.timing
    unavailable = None
    if timing in ("auto", "cupti"):
        try:
            from flashinfer.testing import bench_gpu_time_with_cupti

            parameters = inspect.signature(bench_gpu_time_with_cupti).parameters
            kwargs = {}
            # Require explicit cache policy control; never label an unknown
            # CUPTI default as a cold-cache measurement.
            if "cold_l2_cache" not in parameters:
                raise RuntimeError(
                    "installed CUPTI harness has no cold_l2_cache control"
                )
            kwargs["cold_l2_cache"] = args.cache == "cold"
            if "use_cuda_graph" in parameters:
                kwargs["use_cuda_graph"] = (
                    False  # We capture the complete wrapper above.
                )
            samples = bench_gpu_time_with_cupti(fn, **kwargs)
            return {
                **_summary([float(x) * 1000 for x in samples]),
                "timer": "FlashInfer CUPTI",
                "capture_ms": capture_ms,
            }
        except (ImportError, RuntimeError) as exc:
            if timing == "cupti":
                raise
            unavailable = str(exc)
    samples = []
    for _ in range(args.repeats):
        if flush is not None:
            flush.zero_()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return {
        **_summary(samples),
        "timer": "CUDA events",
        "capture_ms": capture_ms,
        "cupti_unavailable_reason": unavailable,
    }


def _staged_candidates(
    base, anchor, temps, greedy, w1, w2, static_ids, static_bias, topk, alpha
):
    """Vectorized Torch candidate baseline; same union and static-side authority.

    This intentionally measures the dense cache allocation/publication it uses;
    it is a functional baseline, not claimed to be an optimal implementation.
    """
    batch, gamma, vocab = base.shape
    values, ids = torch.topk(base, topk, dim=-1)
    cache = torch.full(
        (batch, gamma, vocab), -torch.inf, dtype=torch.float32, device=base.device
    )
    outputs = []
    prev = anchor
    for step in range(gamma):
        a = ids[:, step]
        h = static_ids[prev].long()
        a_bias = (w1[prev].float()[:, None, :] * w2[a].float()).sum(-1)
        a_scores = values[:, step].float() + alpha * a_bias
        if h.shape[1]:
            duplicate = (a[:, :, None] == h[:, None, :]).any(-1)
            a_scores = a_scores.masked_fill(duplicate, -torch.inf)
            h_scores = base[:, step].gather(1, h).float() + alpha * static_bias[prev]
            candidates = torch.cat((a, h), dim=1)
            scores = torch.cat((a_scores, h_scores), dim=1)
        else:
            candidates, scores = a, a_scores
        noise = -torch.log(-torch.log(torch.rand_like(scores).clamp_(1e-7, 1 - 1e-7)))
        keys = torch.where(greedy[:, None], scores, scores / temps[:, None] + noise)
        best = keys.max(-1, keepdim=True).values
        prev = torch.where(keys == best, candidates, vocab).min(-1).values
        outputs.append(prev)
        cache[:, step].scatter_reduce_(
            1, candidates, scores, reduce="amax", include_self=True
        )
    return torch.stack(outputs, dim=1), cache


def _dense_baseline(base, anchor, temps, greedy, w1, w2, alpha, all_greedy):
    from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
        MarkovGreedyStep,
        SampleStepTokens,
    )

    prev = anchor
    outputs, cache = [], []
    noise = torch.empty(base.shape[0], base.shape[-1], device=base.device)
    for step in range(base.shape[1]):
        embedding = w1[prev]
        if all_greedy:
            prev = MarkovGreedyStep.execute(
                base_logits=base[:, step],
                prev_embeds=embedding * alpha,
                w2_weight=w2,
            )
        else:
            scores = base[:, step] + alpha * torch.nn.functional.linear(embedding, w2)
            noise.exponential_()
            prev = SampleStepTokens.execute(
                step_logits=scores,
                temperatures=temps,
                greedy_mask=greedy,
                exp_noise=noise,
            )
            cache.append(scores)
        outputs.append(prev)
    return torch.stack(outputs, dim=1), torch.stack(cache, dim=1) if cache else None


def _check_case(base, anchor, temps, w1, w2, sampler, args):
    greedy = torch.ones(base.shape[0], dtype=torch.bool, device="cuda")
    actual = sampler.sample(base, anchor, temps, greedy)
    expected_ids, expected_scores = _staged_candidates(
        base,
        anchor,
        temps,
        greedy,
        w1,
        w2,
        sampler.static_ids,
        sampler.static_bias,
        args.k,
        args.alpha,
    )
    torch.testing.assert_close(actual.tokens, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        actual.corrected_logits, expected_scores, rtol=2e-4, atol=2e-4
    )
    # This numerical check supplements (not replaces) the independent CPU and
    # statistical GPU suite. It must pass on the exact benchmark shape.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab", type=int, required=True, help="actual draft output vocabulary"
    )
    parser.add_argument("--rank", type=int, required=True, help="actual Markov rank")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--base-dtype", choices=("float16", "bfloat16", "float32"), default="float32"
    )
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16, 32, 64, 128])
    parser.add_argument("--gamma", type=int, default=8)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--sampling",
        choices=("greedy", "probabilistic", "mixed"),
        default="probabilistic",
    )
    parser.add_argument(
        "--topk-backend", choices=("auto", "torch", "flashinfer"), default="auto"
    )
    parser.add_argument("--timing", choices=("auto", "cupti", "events"), default="auto")
    parser.add_argument("--cache", choices=("cold", "hot"), default="cold")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.cuda is None:
        parser.error("NVIDIA CUDA is required; no performance result was produced")
    if min(args.vocab, args.rank, args.gamma, args.k, *args.batch) <= 0:
        parser.error("vocab/rank/gamma/k/batch must be positive")
    if args.m < 0 or max(args.k, args.m) > args.vocab or args.temperature <= 0:
        parser.error("invalid candidate budgets or temperature")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("use at least one warmup and two timed samples")
    from sglang.kernels.ops.speculative.dspark.dspark_markov_topk import (
        MarkovCandidateSampler,
    )

    torch.manual_seed(20260919)
    dtype = getattr(torch, args.dtype)
    w1 = (torch.randn(args.vocab, args.rank, device="cuda") / math.sqrt(args.rank)).to(
        dtype
    )
    w2 = torch.randn(args.vocab, args.rank, device="cuda").to(dtype)
    # Flush well beyond L2; excluded from events. CUPTI handles its own policy.
    flush = (
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if args.cache == "cold"
        else None
    )
    print(json.dumps({"metadata": _metadata(args)}), flush=True)
    for batch in args.batch:
        base = torch.randn(
            batch,
            args.gamma,
            args.vocab,
            device="cuda",
            dtype=getattr(torch, args.base_dtype),
        )
        anchor = torch.randint(args.vocab, (batch,), device="cuda")
        temps = torch.full((batch,), args.temperature, device="cuda")
        greedy = torch.zeros(batch, dtype=torch.bool, device="cuda")
        if args.sampling == "greedy":
            greedy.fill_(True)
        elif args.sampling == "mixed":
            greedy[::2] = True
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        sampler = MarkovCandidateSampler(
            w1,
            w2,
            alpha=args.alpha,
            topk=args.k,
            bias_topk=args.m,
            target_vocab_size=args.vocab,
            gamma=args.gamma,
            capacity=batch,
            topk_backend=args.topk_backend,
            logits_dtype=base.dtype,
        )
        torch.cuda.synchronize()
        initialization_ms = (time.perf_counter() - started) * 1000
        if sampler.path != "triton":
            raise RuntimeError(f"candidate fast path not enabled: {sampler.path}")
        memory = {
            "persistent_bytes": torch.cuda.memory_allocated() - memory_before,
            "initialization_peak_extra_bytes": torch.cuda.max_memory_allocated()
            - memory_before,
            "proposal_bytes": sampler.corrected_logits.numel() * 4,
            "table_bytes": sampler.static_ids.numel() * 4
            + sampler.static_bias.numel() * 4,
        }
        _check_case(base, anchor, temps, w1, w2, sampler, args)
        prepared_values, prepared_ids = sampler.prepare_topk(base)
        prepared_seeds = torch.randint(0, 2**31, (batch, args.gamma), device="cuda")
        functions = {
            (
                "existing_dense_markov_greedy"
                if args.sampling == "greedy"
                else "existing_dense_sampling"
            ): lambda: _dense_baseline(
                base,
                anchor,
                temps,
                greedy,
                w1,
                w2,
                args.alpha,
                args.sampling == "greedy",
            ),
            "staged_candidate_torch": lambda: _staged_candidates(
                base,
                anchor,
                temps,
                greedy,
                w1,
                w2,
                sampler.static_ids,
                sampler.static_bias,
                args.k,
                args.alpha,
            ),
            "production_candidate_wrapper": lambda: sampler.sample(
                base, anchor, temps, greedy
            ),
            "production_topk_only": lambda: sampler.prepare_topk(base),
            "production_walk_and_sparse_cache_only_ablation": lambda: (
                sampler.sample_prepared(
                    base,
                    prepared_values,
                    prepared_ids,
                    anchor,
                    temps,
                    greedy,
                    seeds=prepared_seeds,
                )
            ),
            "torch_topk_only_ablation": lambda: torch.topk(base, args.k, dim=-1),
            "dense_q_softmax_only_ablation": lambda: (
                sampler.corrected_logits / temps[:, None, None]
            ).softmax(-1),
        }
        # Alternating order distributes thermal drift across implementations.
        names = list(functions)
        if args.batch.index(batch) % 2:
            names.reverse()
        for name in names:
            result = _measure(functions[name], args, flush)
            print(
                json.dumps(
                    {
                        "batch": batch,
                        "implementation": name,
                        **result,
                        "memory": memory,
                        "initialization_ms": initialization_ms,
                        "cache": args.cache,
                        "graph": args.graph,
                        "path": sampler.path,
                        "static_matmul_precision": sampler.static_precision,
                        "static_allow_tf32": sampler.static_allow_tf32,
                        "effective_topk_backend": getattr(
                            sampler, "topk_backend", "unreported"
                        ),
                        "correctness": "exact-shape greedy staged parity; run independent suite separately",
                    }
                ),
                flush=True,
            )
        del sampler


if __name__ == "__main__":
    main()
