"""
SGLang Score API — Stacked-Embedding CUDA-IPC Benchmark (Engine API)

Companion to ``bench_score.py`` (which load-tests the HTTP ``/v1/score`` endpoint).
The zero-copy stacked-embedding transport passes GPU tensor views plus a CUDA IPC
handle to the scheduler, which cannot travel over the HTTP/JSON API — so this
benchmark drives the in-process ``sglang.Engine`` directly.

What it does
------------
1. Launches a single-process ``Engine`` (tp_size must be 1 for zero-copy IPC).
2. Synthesizes (query, items) token sequences with ``--placeholder-token-id``
   markers where embeddings are injected (seeded; no external dataset).
3. Allocates ONE shared GPU buffer (the "pool"), stacks each sequence's override
   vectors into a contiguous view, exports the pool's CUDA IPC handle, and scores
   via the stacked path.
4. Scores the same inputs via the by-value ``query_/item_embed_overrides`` path and
   asserts the returned scores match (within --atol/--rtol) -> correctness proof.
   Both paths inject the SAME vectors, so their values don't matter to the proof.
5. Times both paths over the dataset and prints latency percentiles + throughput.

Usage
-----
    python benchmark/prefill_only/bench_score_stacked_ipc.py \
        --model-path Qwen/Qwen3-4B \
        --num-samples 64 --items-per-query 8 --iters 20 --warmup 5

    # correctness only:
    python benchmark/prefill_only/bench_score_stacked_ipc.py \
        --model-path Qwen/Qwen3-0.6B --skip-perf
"""

import argparse
import statistics
import sys
import time
from typing import List, Optional, Tuple

import torch

import sglang as sgl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model-path", default="Qwen/Qwen3-4B", help="HF model path/id for the Engine."
    )
    p.add_argument(
        "--placeholder-token-id",
        type=int,
        default=50,
        help="Token id marking positions replaced by an override vector.",
    )
    p.add_argument(
        "--label-token-ids",
        type=int,
        nargs="+",
        default=None,
        help="label_token_ids for the score call. Defaults to two valid mid-vocab ids.",
    )
    p.add_argument(
        "--hidden-size",
        type=int,
        default=None,
        help="Override vector dim. Default: read from the model config.",
    )
    # Synthetic dataset controls.
    p.add_argument(
        "--num-samples", type=int, default=32, help="Number of (query, items) samples."
    )
    p.add_argument("--items-per-query", type=int, default=8, help="Items per query.")
    p.add_argument("--query-len", type=int, default=32, help="Query token length.")
    p.add_argument("--item-len", type=int, default=24, help="Item token length.")
    p.add_argument(
        "--query-placeholders",
        type=int,
        default=2,
        help="# override positions per query.",
    )
    p.add_argument(
        "--item-placeholders",
        type=int,
        default=1,
        help="# override positions per item.",
    )
    # Benchmark controls.
    p.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Timed iterations over the dataset per path.",
    )
    p.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations per path (not timed)."
    )
    p.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for score equivalence.",
    )
    p.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for score equivalence.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--mem-fraction-static", type=float, default=0.6)
    p.add_argument(
        "--skip-perf", action="store_true", help="Only run the equivalence check."
    )
    return p.parse_args()


def resolve_hidden_size(
    engine: "sgl.Engine", model_path: str, override: Optional[int]
) -> int:
    if override is not None:
        return override
    for getter in (
        lambda: engine.tokenizer_manager.model_config.hf_config.hidden_size,
        lambda: engine.tokenizer_manager.model_config.hidden_size,
    ):
        try:
            return int(getter())
        except Exception:
            pass
    from transformers import AutoConfig

    return int(AutoConfig.from_pretrained(model_path).hidden_size)


def resolve_vocab_size(engine: "sgl.Engine") -> int:
    try:
        return int(engine.tokenizer_manager.tokenizer.vocab_size)
    except Exception:
        return int(engine.tokenizer_manager.model_config.vocab_size)


def synth_dataset(args: argparse.Namespace, vocab_size: int) -> List[dict]:
    """Deterministically build samples with placeholder tokens at fixed positions."""
    g = torch.Generator().manual_seed(args.seed)
    ph = args.placeholder_token_id

    def rand_tokens(n: int) -> List[int]:
        toks = torch.randint(0, vocab_size, (n,), generator=g).tolist()
        return [t if t != ph else (t + 1) % vocab_size for t in toks]

    samples = []
    for _ in range(args.num_samples):
        q = rand_tokens(args.query_len)
        for pos in range(min(args.query_placeholders, len(q))):
            q[pos] = ph
        items = []
        for _ in range(args.items_per_query):
            it = rand_tokens(args.item_len)
            for pos in range(min(args.item_placeholders, len(it))):
                it[pos] = ph
            items.append(it)
        samples.append({"query": q, "items": items})
    return samples


def count_placeholders(seq: List[int], ph: int) -> int:
    return sum(1 for t in seq if t == ph)


class OverrideVectors:
    """Deterministic per-placeholder override vectors, shared by both paths."""

    def __init__(self, hidden: int, seed: int):
        self.hidden = hidden
        self.g = torch.Generator().manual_seed(seed)

    def make(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.hidden, generator=self.g, dtype=torch.float32)


def build_byvalue_inputs(sample: dict, ph: int, ov: OverrideVectors):
    """Return (query_embed_overrides, item_embed_overrides, masters) for one sample."""
    q_ph = count_placeholders(sample["query"], ph)
    q_master = ov.make(q_ph) if q_ph else torch.empty(0, ov.hidden)
    query_embed_overrides = [q_master[i].clone() for i in range(q_ph)] if q_ph else None

    item_embed_overrides: List[Optional[List[torch.Tensor]]] = []
    item_masters = []
    for it in sample["items"]:
        m = count_placeholders(it, ph)
        if m == 0:
            item_embed_overrides.append(None)
            item_masters.append(torch.empty(0, ov.hidden))
            continue
        im = ov.make(m)
        item_masters.append(im)
        item_embed_overrides.append([im[i].clone() for i in range(m)])
    if all(x is None for x in item_embed_overrides):
        item_embed_overrides = [None] * len(sample["items"])
    return query_embed_overrides, item_embed_overrides, (q_master, item_masters)


def pool_rows_needed(sample: dict, ph: int) -> int:
    q_ph = count_placeholders(sample["query"], ph)
    total = 0
    for it in sample["items"]:
        m = count_placeholders(it, ph)
        if q_ph + m > 0:
            total += q_ph + m
    return total


def build_stacked_pool(
    sample: dict, masters, pool: torch.Tensor
) -> Tuple[List[Optional[torch.Tensor]], object]:
    """Pack per-item [Q+M_i, hidden] stacked views into `pool` (one shared buffer).

    Row order per item = query rows first, then that item's rows (matches
    TokenizerManagerScoreMixin._build_token_id_inputs_stacked). Returns
    (stacked_list, ipc_handle).
    """
    q_master, item_masters = masters
    q_ph = q_master.shape[0]

    stacked: List[Optional[torch.Tensor]] = []
    offset = 0
    for i in range(len(sample["items"])):
        m = item_masters[i].shape[0]
        rows = q_ph + m
        if rows == 0:
            stacked.append(None)
            continue
        view = pool[offset : offset + rows]  # a view into the single shared buffer
        if q_ph:
            view[:q_ph].copy_(q_master.to(pool.device))
        if m:
            view[q_ph:].copy_(item_masters[i].to(pool.device))
        stacked.append(view)
        offset += rows

    handle = pool.untyped_storage()._share_cuda_()  # ONE handle for the whole pool
    torch.cuda.synchronize()  # producer-sync contract: writes complete before handoff
    return stacked, handle


def scores_close(a, b, atol: float, rtol: float) -> Tuple[bool, float]:
    ta = torch.tensor(a, dtype=torch.float64)
    tb = torch.tensor(b, dtype=torch.float64)
    if ta.shape != tb.shape:
        return False, float("inf")
    max_abs = (ta - tb).abs().max().item() if ta.numel() else 0.0
    return torch.allclose(ta, tb, atol=atol, rtol=rtol), max_abs


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"[init] launching Engine: {args.model_path} (tp_size=1)")
    engine = sgl.Engine(
        model_path=args.model_path,
        tp_size=1,
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=True,
        disable_radix_cache=True,
    )

    try:
        hidden = resolve_hidden_size(engine, args.model_path, args.hidden_size)
        vocab = resolve_vocab_size(engine)
        ph = args.placeholder_token_id
        label_ids = args.label_token_ids or [vocab // 3, vocab // 2]
        assert all(t < vocab for t in label_ids), "label_token_ids out of vocab"
        print(
            f"[init] hidden={hidden} vocab={vocab} placeholder={ph} labels={label_ids}"
        )

        samples = synth_dataset(args, vocab)
        print(
            f"[data] synthesized {len(samples)} samples ({args.items_per_query} items/query)"
        )

        ov = OverrideVectors(hidden, args.seed)

        prepared = []
        max_rows = 1
        for s in samples:
            qbv, ibv, masters = build_byvalue_inputs(s, ph, ov)
            prepared.append((s, qbv, ibv, masters))
            max_rows = max(max_rows, pool_rows_needed(s, ph))

        pool = torch.empty(max_rows, hidden, dtype=torch.float32, device="cuda")
        print(
            f"[pool] shared GPU buffer [{max_rows}, {hidden}] float32 "
            f"({pool.numel() * 4 / 1e6:.2f} MB); one CUDA IPC handle reused per call"
        )

        def run_byvalue(s, qbv, ibv):
            return engine.score(
                query=s["query"],
                items=s["items"],
                label_token_ids=label_ids,
                embed_override_token_id=ph,
                query_embed_overrides=qbv,
                item_embed_overrides=ibv,
                apply_softmax=False,
            )

        def run_stacked(s, masters):
            stacked, handle = build_stacked_pool(s, masters, pool)
            return engine.score(
                query=s["query"],
                items=s["items"],
                label_token_ids=label_ids,
                embed_override_token_id=ph,
                stacked_query_item_embed_overrides=stacked,
                stacked_query_item_embed_ipc_handle=handle,
                apply_softmax=False,
            )

        # ---- Equivalence ----
        print("\n[equivalence] comparing by-value vs stacked-IPC scores ...")
        n_ok, worst = 0, 0.0
        for idx, (s, qbv, ibv, masters) in enumerate(prepared):
            r_bv = run_byvalue(s, qbv, ibv)
            r_ipc = run_stacked(s, masters)
            ok, max_abs = scores_close(r_bv.scores, r_ipc.scores, args.atol, args.rtol)
            worst = max(worst, max_abs)
            n_ok += int(ok)
            if not ok:
                print(f"  sample {idx}: MISMATCH max_abs_diff={max_abs:.3e}")
        all_ok = n_ok == len(prepared)
        print(
            f"[equivalence] {n_ok}/{len(prepared)} matched "
            f"(max abs diff {worst:.3e}) -> {'PASS' if all_ok else 'FAIL'}"
        )

        # ---- Perf ----
        if not args.skip_perf:

            def bench(fn) -> List[float]:
                for _ in range(args.warmup):
                    for s, qbv, ibv, masters in prepared:
                        fn(s, qbv, ibv, masters)
                torch.cuda.synchronize()
                out = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    for s, qbv, ibv, masters in prepared:
                        fn(s, qbv, ibv, masters)
                    torch.cuda.synchronize()
                    out.append((time.perf_counter() - t0) * 1e3)
                return out

            print(
                f"\n[perf] warmup={args.warmup} iters={args.iters} "
                f"({len(prepared)} score calls / iter)"
            )
            bv_ms = bench(lambda s, qbv, ibv, masters: run_byvalue(s, qbv, ibv))
            ipc_ms = bench(lambda s, qbv, ibv, masters: run_stacked(s, masters))

            def summarize(name, ms):
                srt = sorted(ms)
                p50 = statistics.median(srt)
                p90 = srt[min(len(srt) - 1, int(0.9 * len(srt)))]
                calls = len(prepared) * len(ms)
                print(
                    f"  {name:<12} mean={statistics.mean(ms):7.2f}ms  "
                    f"p50={p50:7.2f}ms  p90={p90:7.2f}ms  "
                    f"throughput={calls / (sum(ms) / 1e3):8.1f} calls/s"
                )
                return statistics.mean(ms)

            print("[perf] per-iteration wall time over the dataset:")
            m_bv = summarize("by-value", bv_ms)
            m_ipc = summarize("stacked-ipc", ipc_ms)
            if m_ipc > 0:
                print(f"[perf] stacked-ipc speedup vs by-value: {m_bv / m_ipc:.2f}x")

        print("\n[result] EQUIVALENCE " + ("PASS" if all_ok else "FAIL"))
        return 0 if all_ok else 1
    finally:
        engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
