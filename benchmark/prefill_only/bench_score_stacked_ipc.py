"""
SGLang Score API: stacked GPU embedding overrides vs per-placeholder CPU vectors (Engine API)

Companion to ``bench_score.py`` (which load-tests the HTTP ``/v1/score`` endpoint).
GPU tensors cannot travel over the HTTP/JSON API, so this benchmark drives the
in-process ``sglang.Engine`` directly.

Scores synthetic token-id samples with ``query_/item_embed_overrides`` (CPU
vectors) and ``stacked_query_item_embed_overrides`` (GPU tensors), checks the scores
match, and times both.

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
    p.add_argument("--tp-size", type=int, default=1)
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
    def __init__(self, hidden: int, seed: int):
        self.hidden = hidden
        self.g = torch.Generator().manual_seed(seed)

    def make(self, n: int) -> torch.Tensor:
        return torch.randn(n, self.hidden, generator=self.g, dtype=torch.float32)


def build_byvalue_inputs(sample: dict, ph: int, ov: OverrideVectors):
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


def build_stacked_inputs(masters) -> List[Optional[torch.Tensor]]:
    q_master, item_masters = masters
    stacked: List[Optional[torch.Tensor]] = []
    for im in item_masters:
        rows = torch.cat([q_master, im])
        stacked.append(rows.cuda() if rows.shape[0] else None)
    return stacked


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

    print(f"[init] launching Engine: {args.model_path} (tp_size={args.tp_size})")
    engine = sgl.Engine(
        model_path=args.model_path,
        tp_size=args.tp_size,
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
        for s in samples:
            qbv, ibv, masters = build_byvalue_inputs(s, ph, ov)
            prepared.append((s, qbv, ibv, build_stacked_inputs(masters)))

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

        def run_stacked(s, stacked):
            return engine.score(
                query=s["query"],
                items=s["items"],
                label_token_ids=label_ids,
                embed_override_token_id=ph,
                stacked_query_item_embed_overrides=stacked,
                apply_softmax=False,
            )

        print("\n[equivalence] comparing by-value vs stacked scores ...")
        n_ok, worst = 0, 0.0
        for idx, (s, qbv, ibv, stacked) in enumerate(prepared):
            r_bv = run_byvalue(s, qbv, ibv)
            r_stacked = run_stacked(s, stacked)
            ok, max_abs = scores_close(
                r_bv.scores, r_stacked.scores, args.atol, args.rtol
            )
            worst = max(worst, max_abs)
            n_ok += int(ok)
            if not ok:
                print(f"  sample {idx}: MISMATCH max_abs_diff={max_abs:.3e}")
        all_ok = n_ok == len(prepared)
        print(
            f"[equivalence] {n_ok}/{len(prepared)} matched "
            f"(max abs diff {worst:.3e}) -> {'PASS' if all_ok else 'FAIL'}"
        )

        if not args.skip_perf:

            def bench(fn) -> List[float]:
                for _ in range(args.warmup):
                    for s, qbv, ibv, stacked in prepared:
                        fn(s, qbv, ibv, stacked)
                torch.cuda.synchronize()
                out = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    for s, qbv, ibv, stacked in prepared:
                        fn(s, qbv, ibv, stacked)
                    torch.cuda.synchronize()
                    out.append((time.perf_counter() - t0) * 1e3)
                return out

            print(
                f"\n[perf] warmup={args.warmup} iters={args.iters} "
                f"({len(prepared)} score calls / iter)"
            )
            bv_ms = bench(lambda s, qbv, ibv, stacked: run_byvalue(s, qbv, ibv))
            stacked_ms = bench(lambda s, qbv, ibv, stacked: run_stacked(s, stacked))

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
            m_stacked = summarize("stacked", stacked_ms)
            if m_stacked > 0:
                print(f"[perf] stacked speedup vs by-value: {m_bv / m_stacked:.2f}x")

        print("\n[result] EQUIVALENCE " + ("PASS" if all_ok else "FAIL"))
        return 0 if all_ok else 1
    finally:
        engine.shutdown()


if __name__ == "__main__":
    sys.exit(main())
