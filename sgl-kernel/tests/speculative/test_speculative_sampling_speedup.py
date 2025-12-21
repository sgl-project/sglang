import argparse
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sgl_kernel import (
    tree_speculative_sampling_target_only,
    tree_speculative_sampling_target_only_rejmask,
)


@dataclass
class BenchConfig:
    bs: int
    num_draft_tokens: int
    topk: int
    vocab_size: int
    iters: int
    warmup: int
    threshold_single: float
    threshold_acc: float
    deterministic: bool
    seed: int


def _build_kary_tree_structure(num_nodes: int, topk: int, device: str):
    """
    Build a k-ary tree structure in BFS order using two "linked-list" arrays:
      - retrive_next_token: parent -> first child (or -1)
      - retrive_next_sibling: child -> next sibling (or -1)

    Node 0 is treated as the root.
    """
    if num_nodes <= 0:
        raise ValueError(f"num_nodes must be > 0, got {num_nodes}")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}")

    next_token = torch.full((num_nodes,), -1, dtype=torch.int64, device=device)
    next_sibling = torch.full((num_nodes,), -1, dtype=torch.int64, device=device)

    # Simple BFS construction on CPU-side indices, then write into CUDA tensors.
    # This cost is negligible compared to kernel benchmarking, and keeps logic easy to audit.
    queue: list[int] = [0]
    next_free = 1
    while queue and next_free < num_nodes:
        parent = queue.pop(0)
        remaining = num_nodes - next_free
        num_children = min(topk, remaining)
        if num_children <= 0:
            continue

        first_child = next_free
        next_token[parent] = first_child
        prev_child = -1
        for _ in range(num_children):
            child = next_free
            if prev_child != -1:
                next_sibling[prev_child] = child
            prev_child = child
            queue.append(child)
            next_free += 1

    return next_token.contiguous(), next_sibling.contiguous()


def _build_inputs(cfg: BenchConfig, device: str):
    """
    Build a topk=k tree structure:
      - Node 0 is the root (token id is unused by kernels)
      - retrive_next_token / retrive_next_sibling define the tree topology
      - retrive_index maps each node to a flat output position in predicts
    """
    bs = cfg.bs
    n = cfg.num_draft_tokens
    d = cfg.vocab_size

    # candidates: [bs, n], token ids in [0, d)
    candidates = torch.randint(0, d, (bs, n), dtype=torch.int64, device=device)
    # Root token id is unused by the kernels, but keep it valid and stable.
    candidates[:, 0] = 0

    # retrive_index: [bs, n] maps to a flat predicts buffer of length bs*n
    base = (torch.arange(bs, device=device, dtype=torch.int64) * n).unsqueeze(1)
    retrive_index = base + torch.arange(n, device=device, dtype=torch.int64).unsqueeze(
        0
    )

    next_token_1d, next_sibling_1d = _build_kary_tree_structure(n, cfg.topk, device)
    retrive_next_token = next_token_1d.unsqueeze(0).repeat(bs, 1).contiguous()
    retrive_next_sibling = next_sibling_1d.unsqueeze(0).repeat(bs, 1).contiguous()

    # Uniform samples: [bs, n] and [bs]
    coins = torch.rand((bs, n), dtype=torch.float32, device=device)
    coins_for_final_sampling = torch.rand((bs,), dtype=torch.float32, device=device)

    # target_probs: [bs, n, d] float32 probabilities
    # Keep it moderately "peaky" to resemble real logits but avoid extreme underflow.
    logits = torch.randn((bs, n, d), dtype=torch.float32, device=device)
    target_probs = F.softmax(logits, dim=-1)

    return (
        candidates.contiguous(),
        retrive_index.contiguous(),
        retrive_next_token,
        retrive_next_sibling.contiguous(),
        coins.contiguous(),
        coins_for_final_sampling.contiguous(),
        target_probs.contiguous(),
    )


def _alloc_outputs(cfg: BenchConfig, device: str):
    bs = cfg.bs
    n = cfg.num_draft_tokens
    num_spec_tokens = n  # allow at most n-1 draft tokens accepted
    predicts = torch.full((bs * n,), -1, dtype=torch.int32, device=device)
    accept_index = torch.full(
        (bs, num_spec_tokens), -1, dtype=torch.int32, device=device
    )
    accept_token_num = torch.zeros((bs,), dtype=torch.int32, device=device)
    return predicts, accept_index, accept_token_num


def _reset_outputs(
    predicts: torch.Tensor, accept_index: torch.Tensor, accept_token_num: torch.Tensor
):
    predicts.fill_(-1)
    accept_index.fill_(-1)
    accept_token_num.zero_()


def _time_kernel_cuda_events(fn, iters: int, pre_fn=None) -> list[float]:
    """
    Returns per-iter GPU time in milliseconds using CUDA events.

    If provided, `pre_fn()` runs before the timed region (e.g. reset output buffers).
    """
    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if pre_fn is not None:
            pre_fn()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _summarize(name: str, times_ms: list[float]) -> str:
    times_sorted = sorted(times_ms)
    mean = statistics.mean(times_ms)
    p50 = times_sorted[len(times_sorted) // 2]
    p95 = times_sorted[int(len(times_sorted) * 0.95)]
    return f"{name}: mean={mean:.3f}ms  p50={p50:.3f}ms  p95={p95:.3f}ms  iters={len(times_ms)}"


def _run_torch_profiler(
    *,
    name: str,
    fn,
    steps: int,
    trace_dir: str,
    wait: int,
    warmup: int,
    active: int,
    repeat: int,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
):
    """
    Run PyTorch Profiler for `steps` iterations of `fn()` and write traces to `trace_dir`.
    View with:
      tensorboard --logdir <trace_dir>
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0, got {steps}")
    os.makedirs(trace_dir, exist_ok=True)

    # Import lazily to keep default benchmark path lean.
    from torch.profiler import (  # type: ignore
        ProfilerActivity,
        profile,
        schedule,
        tensorboard_trace_handler,
    )

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print("=== PyTorch Profiler ===")
    print(
        f"name={name} steps={steps} schedule(wait={wait}, warmup={warmup}, active={active}, repeat={repeat}) "
        f"trace_dir={trace_dir}"
    )

    # Avoid capturing unrelated work.
    torch.cuda.synchronize()
    with profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=tensorboard_trace_handler(trace_dir),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        for _ in range(steps):
            fn()
            prof.step()
    torch.cuda.synchronize()
    print(f"Profiler traces written to: {trace_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark speculative sampling kernels with configurable topk tree structure (CUDA events timing)."
    )
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--num-draft-tokens", type=int, default=4)
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Branching factor (number of siblings per node) for the draft token tree.",
    )
    parser.add_argument("--vocab-size", type=int, default=120000)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threshold-single", type=float, default=1.0)
    parser.add_argument("--threshold-acc", type=float, default=1.0)
    # Keep deterministic by default (matches typical server settings), but allow opting out.
    parser.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_false",
        help="Disable deterministic sampling paths (may be faster but non-reproducible).",
    )
    parser.set_defaults(deterministic=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--profile",
        type=str,
        default="none",
        choices=["none", "tree", "tree_rejmask"],
        help="Enable PyTorch Profiler for a chosen kernel.",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        default=False,
        help="If set, run profiler (if enabled) and exit without timing summary.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="/tmp/sgl_speculative_sampling_profiler",
        help="Output directory for tensorboard traces.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=50,
        help="How many iterations to run under profiler.",
    )
    parser.add_argument("--profile-wait", type=int, default=1)
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--profile-active", type=int, default=10)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--profile-record-shapes", action="store_true", default=False)
    parser.add_argument("--profile-memory", action="store_true", default=False)
    parser.add_argument("--profile-with-stack", action="store_true", default=False)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    cfg = BenchConfig(
        bs=args.bs,
        num_draft_tokens=args.num_draft_tokens,
        topk=args.topk,
        vocab_size=args.vocab_size,
        iters=args.iters,
        warmup=args.warmup,
        threshold_single=args.threshold_single,
        threshold_acc=args.threshold_acc,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = "cuda"

    (
        candidates,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        coins,
        coins_for_final_sampling,
        target_probs,
    ) = _build_inputs(cfg, device)

    # Pre-allocate outputs for both kernels.
    predicts_tree, accept_index_tree, accept_token_num_tree = _alloc_outputs(
        cfg, device
    )
    predicts_tree_rejmask, accept_index_tree_rejmask, accept_token_num_tree_rejmask = (
        _alloc_outputs(cfg, device)
    )

    def call_tree():
        tree_speculative_sampling_target_only(
            predicts=predicts_tree,
            accept_index=accept_index_tree,
            accept_token_num=accept_token_num_tree,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            threshold_single=cfg.threshold_single,
            threshold_acc=cfg.threshold_acc,
            deterministic=cfg.deterministic,
        )

    def call_tree_rejmask():
        tree_speculative_sampling_target_only_rejmask(
            predicts=predicts_tree_rejmask,
            accept_index=accept_index_tree_rejmask,
            accept_token_num=accept_token_num_tree_rejmask,
            candidates=candidates,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            threshold_single=cfg.threshold_single,
            threshold_acc=cfg.threshold_acc,
            deterministic=cfg.deterministic,
        )

    # Warmup (also ensures kernels are JIT-loaded and caches are populated).
    for _ in range(cfg.warmup):
        _reset_outputs(predicts_tree, accept_index_tree, accept_token_num_tree)
        call_tree()
        _reset_outputs(
            predicts_tree_rejmask,
            accept_index_tree_rejmask,
            accept_token_num_tree_rejmask,
        )
        call_tree_rejmask()
    torch.cuda.synchronize()

    # Optional profiling (kept outside the timed region).
    if args.profile != "none":
        if args.profile == "tree":
            _reset_outputs(predicts_tree, accept_index_tree, accept_token_num_tree)
            prof_fn = call_tree
        elif args.profile == "tree_rejmask":
            _reset_outputs(
                predicts_tree_rejmask,
                accept_index_tree_rejmask,
                accept_token_num_tree_rejmask,
            )
            prof_fn = call_tree_rejmask

        _run_torch_profiler(
            name=args.profile,
            fn=prof_fn,
            steps=args.profile_steps,
            trace_dir=args.profile_dir,
            wait=args.profile_wait,
            warmup=args.profile_warmup,
            active=args.profile_active,
            repeat=args.profile_repeat,
            record_shapes=args.profile_record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.profile_with_stack,
        )
        if args.profile_only:
            return

    # Make sure clocks are stable before timing.
    torch.cuda.synchronize()
    time.sleep(0.05)

    # Time only kernel call (exclude _reset_outputs cost).
    tree_ms = _time_kernel_cuda_events(
        call_tree,
        cfg.iters,
        pre_fn=lambda: _reset_outputs(
            predicts_tree, accept_index_tree, accept_token_num_tree
        ),
    )
    tree_rejmask_ms = _time_kernel_cuda_events(
        call_tree_rejmask,
        cfg.iters,
        pre_fn=lambda: _reset_outputs(
            predicts_tree_rejmask,
            accept_index_tree_rejmask,
            accept_token_num_tree_rejmask,
        ),
    )

    print("=== Config ===")
    print(
        f"bs={cfg.bs} num_draft_tokens={cfg.num_draft_tokens} topk={cfg.topk} vocab_size={cfg.vocab_size} "
        f"warmup={cfg.warmup} iters={cfg.iters} threshold_single={cfg.threshold_single} "
        f"threshold_acc={cfg.threshold_acc} deterministic={cfg.deterministic} seed={cfg.seed}"
    )
    print("=== Results (GPU kernel time) ===")
    print(_summarize("tree", tree_ms))
    print(_summarize("tree_rejmask", tree_rejmask_ms))
    tree_rejmask_speedup = statistics.mean(tree_ms) / max(
        statistics.mean(tree_rejmask_ms), 1e-12
    )
    print(f"speedup(tree/tree_rejmask) = {tree_rejmask_speedup:.2f}x")

    # Optional correctness spot-check:
    # This is not exhaustive and is deliberately outside the timed region.
    _reset_outputs(predicts_tree, accept_index_tree, accept_token_num_tree)
    _reset_outputs(
        predicts_tree_rejmask, accept_index_tree_rejmask, accept_token_num_tree_rejmask
    )
    call_tree()
    call_tree_rejmask()
    torch.cuda.synchronize()
    ok_tree_rejmask = (
        torch.equal(predicts_tree, predicts_tree_rejmask)
        and torch.equal(accept_index_tree, accept_index_tree_rejmask)
        and torch.equal(accept_token_num_tree, accept_token_num_tree_rejmask)
    )
    print(f"spot_check_equal(tree, tree_rejmask) = {ok_tree_rejmask}")


if __name__ == "__main__":
    main()
