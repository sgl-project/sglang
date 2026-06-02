"""Loop-7 fast MMLU 5-shot runner (AC-3 re-anchor). Uses the AC-12 harness
helpers: 5-shot "Answer:" prompt + raw /generate max_new_tokens=4 (no reasoning
chain, so it is fast on the reasoning model), deterministic example set (same
questions across servers for a paired DSA-vs-DS-hybrid delta)."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test", "manual"))
import test_double_sparsity_v32 as h  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--data-dir", default=os.environ.get("AC12_MMLU_DATA_DIR", "/root/ac12_mmlu_data"))
    ap.add_argument("--label", default="ds")
    ap.add_argument("--op-point", default="graph-mode int8 / mem 0.7 / TP=8 (or DSA native-NSA)")
    ap.add_argument("--out", default="development/loop7/mmlu_result.json")
    args = ap.parse_args()
    url = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")

    dev_dir, test_dir = h._ensure_mmlu_data_dir(args.data_dir)
    examples, _ = h._load_mmlu_examples(dev_dir, test_dir, max_examples=args.num)
    n = len(examples)
    correct = 0
    t0 = time.time()
    for i, ex in enumerate(examples):
        prompt = h._make_mmlu_5shot_prompt(ex["dev"], ex["subject"], ex["row"])
        resp, _ = h._generate(url, prompt, max_new_tokens=4)
        pred = h._parse_mmlu_letter((resp or "").strip())
        gold = str(ex["row"][5]).strip().upper()
        correct += int(pred == gold)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{n} acc={correct/(i+1)*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    acc = correct / n if n else 0.0
    out = {"label": args.label, "op_point": args.op_point, "graph_mode": "cuda_graph",
           "transport": "raw /generate, 5-shot Answer: prompt, max_new_tokens=4",
           "example_seed": "deterministic (_load_mmlu_examples seed=0xAC12) -> same questions across servers",
           "data_dir": args.data_dir, "num_examples": n, "hits": correct,
           "score_pct": round(acc * 100, 2), "elapsed_s": round(time.time() - t0, 1)}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n{args.label}: {correct}/{n} = {acc*100:.2f}%  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
