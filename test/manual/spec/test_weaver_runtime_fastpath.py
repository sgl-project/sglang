"""CUDA parity and replay-latency check for Weaver frontier materialization."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import types
from pathlib import Path

import torch

sglang = types.ModuleType("sglang")
sglang.__path__ = [str(Path(__file__).parents[3] / "python" / "sglang")]
sys.modules["sglang"] = sglang

from sglang.srt.speculative.dflash_tfm import DFlashTfmWorker, Weaver  # noqa: E402


def make_worker(model: Weaver, budget: int) -> DFlashTfmWorker:
    worker = object.__new__(DFlashTfmWorker)
    worker.weaver = model
    worker.tree_budget = budget
    worker._weaver_compiled_indexed_step_fns = {}
    worker._weaver_tree_cuda_graphs = {}
    return worker


def snapshot(tree) -> tuple[torch.Tensor, ...]:
    return (
        tree.draft_tokens.clone(),
        tree.parent_indices.clone(),
        tree.depths.clone(),
        tree.node_mask.clone(),
        tree.draft_logprobs.clone(),
    )


def assert_same_tree(reference: tuple[torch.Tensor, ...], actual, label: str) -> None:
    names = ("tokens", "parents", "depths", "mask", "logprobs")
    for name, expected, observed in zip(
        names, reference, snapshot(actual), strict=True
    ):
        if torch.equal(expected, observed):
            continue
        if expected.is_floating_point():
            finite = torch.isfinite(expected) & torch.isfinite(observed)
            max_error = float((expected[finite] - observed[finite]).abs().max())
            raise AssertionError(f"{label}: {name} differs; max_error={max_error}")
        mismatch = (expected != observed).nonzero()[0].tolist()
        raise AssertionError(f"{label}: {name} differs first at {mismatch}")


def build(worker: DFlashTfmWorker, kwargs: dict, optimized: bool):
    os.environ["SGLANG_WEAVER_FUSED_FRONTIER_MATERIALIZE"] = "1" if optimized else "0"
    return worker._build_tree(**kwargs)


def timed_graph_us(worker: DFlashTfmWorker, optimized: bool, iterations: int) -> float:
    graph_state = next(
        state
        for key, state in worker._weaver_tree_cuda_graphs.items()
        if key[2] is optimized
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph_state.graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def profile_graph(worker: DFlashTfmWorker, optimized: bool, iterations: int) -> None:
    graph_state = next(
        state
        for key, state in worker._weaver_tree_cuda_graphs.items()
        if key[2] is optimized
    )
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        for _ in range(iterations):
            graph_state.graph.replay()
    label = "optimized" if optimized else "baseline"
    print(f"PROFILE {label}")
    print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", type=int, nargs="+", default=[49, 64, 128, 129])
    parser.add_argument("--depth", type=int, default=15)
    parser.add_argument("--candidate-pool", type=int, default=512)
    parser.add_argument("--rank", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--mlp-dim", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--random-cases", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile-iterations", type=int, default=0)
    parser.add_argument("--skip-timing", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This manual test requires a CUDA GPU.")
    torch.set_grad_enabled(False)
    torch.manual_seed(123)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    vocab = 2 + args.batch_size * args.depth * args.candidate_pool
    model = Weaver(
        d_model=args.hidden_size,
        d_embed=args.hidden_size,
        d_rank=args.rank,
        num_layers=args.num_layers,
        num_heads=args.heads,
        mlp_dim=args.mlp_dim,
        K=args.depth,
        candidate_pool_size=args.candidate_pool,
    ).to(device=device, dtype=dtype)
    model.eval()

    root_ids = torch.arange(
        1,
        1 + args.batch_size,
        device=device,
    )
    output_norm = torch.randn(
        (args.batch_size, args.hidden_size),
        device=device,
        dtype=dtype,
    )
    proposal_features = torch.randn(
        (args.batch_size, args.depth, args.hidden_size),
        device=device,
        dtype=dtype,
    )
    candidate_ids = torch.arange(2, vocab, device=device).reshape(
        args.batch_size,
        args.depth,
        args.candidate_pool,
    )
    candidate_weights = torch.empty(
        (
            args.batch_size,
            args.depth,
            args.candidate_pool,
            args.rank,
        ),
        device=device,
        dtype=dtype,
    )
    candidate_scores = torch.empty(
        (args.batch_size, args.depth, args.candidate_pool),
        device=device,
        dtype=torch.float32,
    )
    token_embed = torch.randn((vocab, args.rank), device=device, dtype=dtype)
    kwargs = {
        "root_ids": root_ids,
        "output_norm": output_norm,
        "candidate_ids": candidate_ids,
        "candidate_weights": candidate_weights,
        "candidate_scores": candidate_scores,
        "proposal_features": proposal_features,
        "token_embed": token_embed,
    }

    with torch.inference_mode():
        for budget in args.budgets:
            torch.compiler.reset()
            worker = make_worker(model, budget)
            for seed in range(args.random_cases):
                generator = torch.Generator(device=device).manual_seed(seed)
                candidate_weights.normal_(generator=generator)
                candidate_scores.normal_(generator=generator)
                candidate_scores.copy_(
                    candidate_scores.sort(dim=-1, descending=True).values
                )
                reference = snapshot(build(worker, kwargs, optimized=False))
                assert_same_tree(
                    reference,
                    build(worker, kwargs, optimized=True),
                    f"budget={budget} random_seed={seed}",
                )

            candidate_weights.zero_()
            candidate_scores.zero_()
            reference = snapshot(build(worker, kwargs, optimized=False))
            assert_same_tree(
                reference,
                build(worker, kwargs, optimized=True),
                f"budget={budget} equal_score_tie",
            )

            candidate_weights.zero_()
            duplicate_scores = torch.arange(
                args.candidate_pool,
                device=device,
                dtype=torch.float32,
            ).remainder(7)
            candidate_scores.copy_(
                duplicate_scores[None, None]
                .expand_as(candidate_scores)
                .sort(dim=-1, descending=True)
                .values
            )
            reference = snapshot(build(worker, kwargs, optimized=False))
            assert_same_tree(
                reference,
                build(worker, kwargs, optimized=True),
                f"budget={budget} duplicate_score_ties",
            )

            candidate_ids.fill_(-1)
            candidate_ids[:, :, 0] = torch.arange(2, 2 + args.depth, device=device)
            reference = snapshot(build(worker, kwargs, optimized=False))
            assert_same_tree(
                reference,
                build(worker, kwargs, optimized=True),
                f"budget={budget} frontier_exhaustion",
            )

            candidate_ids.copy_(
                torch.arange(2, vocab, device=device).reshape_as(candidate_ids)
            )
            if args.skip_timing:
                print(
                    f"PASS budget={budget} batch_size={args.batch_size} "
                    f"random_cases={args.random_cases} ties=2 exhaustion=1"
                )
                continue

            candidate_weights.normal_()
            candidate_scores.normal_()
            candidate_scores.copy_(
                candidate_scores.sort(dim=-1, descending=True).values
            )
            build(worker, kwargs, optimized=False)
            build(worker, kwargs, optimized=True)
            baseline = [
                timed_graph_us(worker, False, args.iterations)
                for _ in range(args.repeats)
            ]
            optimized = [
                timed_graph_us(worker, True, args.iterations)
                for _ in range(args.repeats)
            ]
            baseline_median = statistics.median(baseline)
            optimized_median = statistics.median(optimized)
            print(
                f"PASS budget={budget} baseline_us={baseline_median:.2f} "
                f"optimized_us={optimized_median:.2f} "
                f"speedup={baseline_median / optimized_median:.3f}x "
                f"batch_size={args.batch_size} random_cases={args.random_cases} "
                "ties=2 exhaustion=1"
            )
            if args.profile_iterations:
                profile_graph(worker, False, args.profile_iterations)
                profile_graph(worker, True, args.profile_iterations)


if __name__ == "__main__":
    main()
