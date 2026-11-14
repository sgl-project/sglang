#!/usr/bin/env python3
"""MMLU 5-shot accuracy against an SGLang server, using the dataset staged on the box.

Why this exists: lm-eval could not be installed in the meetup image (PyPI blocked), and V4 has no Jinja
chat template, so chat-completions harnesses do not fit. This uses only pandas + requests and the local
cais/mmlu parquet copy, scoring the four letter options by next-token logprob.

Usage:
  python3 bench/mmlu_eval.py --per-subject 20 --shots 5 --concurrency 8 \
      --out results/accuracy/mmlu_shot.json

Measured on 4x Ascend950PR, 2026-11-14: 1140 asked, 1015 answered, acc = 89.04% (904/1015);
79.3% if unanswered are counted wrong.
"""
import argparse, glob, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

DATA = os.environ.get("MMLU_DIR", "/workspace/shared_assets/datasets/cais/mmlu")
MODEL = os.environ.get("SERVED_MODEL", "/workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731")
URL = os.environ.get("COMPLETIONS_URL", "http://127.0.0.1:30000/v1/completions")
LET = "ABCD"


def load_split(name):
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, "*", f"{name}-*.parquet"))):
        d = os.path.basename(os.path.dirname(f))
        if d in ("all", "auxiliary_train"):
            continue
        df = pd.read_parquet(f)
        df["__dir"] = d
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fmt(q, ch, a):
    body = "\n".join(f"{LET[i]}. {c}" for i, c in enumerate(ch))
    return f"{str(q).strip()}\n{body}\nAnswer: {LET[int(a)]}\n\n"


def build(subj, shots, q, ch):
    head = f"The following are multiple choice questions (with answers) about {subj.replace('_', ' ')}.\n\n"
    body = "".join(fmt(r.question, list(r.choices), r.answer) for r in shots)
    body += f"{str(q).strip()}\n" + "\n".join(f"{LET[i]}. {c}" for i, c in enumerate(ch)) + "\nAnswer:"
    return head + body


def ask(prompt):
    payload = {"model": MODEL, "prompt": prompt, "max_tokens": 1,
               "temperature": 0, "logprobs": 20, "echo": False}
    for attempt in range(3):
        try:
            r = requests.post(URL, json=payload, timeout=180)
            r.raise_for_status()
            lp = r.json()["choices"][0].get("logprobs") or {}
            cands = {}
            top = lp.get("top_logprobs") or []
            if top and isinstance(top[-1], dict):
                cands = top[-1]
            else:
                toks, sc = lp.get("tokens") or [], lp.get("token_logprobs") or []
                if toks and sc:
                    cands = {toks[-1]: sc[-1]}
            best, best_lp = None, -1e9
            for i, L in enumerate(LET):
                for v in (L, " " + L, L + ".", " " + L + ".", L + ")", " " + L + ")"):
                    if v in cands and cands[v] > best_lp:
                        best, best_lp = i, cands[v]
            return best
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-subject", type=int, default=20)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", default="results/accuracy/mmlu_shot.json")
    args = ap.parse_args()

    test, dev = load_split("test"), load_split("dev")
    if test.empty:
        raise SystemExit(f"no MMLU test parquet under {DATA}")

    dev_by = {}
    if args.shots and not dev.empty:
        for s, g in dev.groupby("__dir"):
            dev_by[s] = g.head(args.shots)

    tasks, meta = [], []
    for subj, g in test.groupby("__dir"):
        for row in g.head(args.per_subject).itertuples():
            shots = list(dev_by.get(subj).itertuples()) if subj in dev_by else []
            tasks.append(build(subj, shots, row.question, list(row.choices)))
            meta.append((subj, int(row.answer)))

    print(f"questions={len(tasks)} shots={args.shots} concurrency={args.concurrency}")
    preds, t0 = [None] * len(tasks), time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(ask, p): i for i, p in enumerate(tasks)}
        done = 0
        for fu in as_completed(futs):
            preds[futs[fu]] = fu.result()
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)} {time.time() - t0:.0f}s")

    ok = [i for i, p in enumerate(preds) if p is not None]
    correct = sum(1 for i in ok if preds[i] == meta[i][1])
    acc = correct / len(ok) if ok else 0.0
    per = {}
    for i in ok:
        s = meta[i][0]
        per.setdefault(s, [0, 0])
        per[s][1] += 1
        per[s][0] += int(preds[i] == meta[i][1])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({
        "acc": acc, "correct": correct, "answered": len(ok), "total": len(tasks),
        "shots": args.shots, "model": MODEL, "endpoint": URL,
        "seconds": round(time.time() - t0, 1),
        "per_subject": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in sorted(per.items())},
    }, open(args.out, "w"), indent=2)
    print(f"\nMMLU {args.shots}-shot: acc={acc * 100:.2f}%  ({correct}/{len(ok)} answered, {len(tasks)} asked)")
    for k, v in sorted(per.items(), key=lambda kv: kv[1][0] / kv[1][1])[:5]:
        print(f"  worst {k}: {v[0] / v[1] * 100:.0f}% ({v[1]})")
    print("saved", args.out)


if __name__ == "__main__":
    main()
