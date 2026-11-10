#!/usr/bin/env python3
"""Build the per-(layer, ep-rank) cold-expert table from
expert-distribution recorder dumps (``--expert-distribution-recorder-mode stat``).

Output: ``{"cold_ids": int64 [layers, ep, S]}`` with *local* routed expert ids
in coldness order (coldest first), as JSON when ``--out`` ends in ``.json``
(the form shipped in the repo), else as a torch file. The server takes the first
``n_spilled`` per (layer, rank) via ``SGLANG_DSV41_EXPERT_SPILL_COLD_SET=<out>``.

If more than one dump is given, the last one is held out and the table built
from the rest is scored on it (out-of-sample spilled hits per token), so the
number printed is what the server will see on traffic like the held-out dump.

    python scripts/dsv41_cold_set_from_dumps.py review.pt code.pt agent.pt heldout.pt \
        --out scripts/dsv41_flash_cold_set_ep8.json --ep 8 --local 48 --spill 18
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import List

import torch


def load_counts(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    lc = obj["logical_count"] if isinstance(obj, dict) else obj
    lc = torch.as_tensor(lc)
    # [buffer, layers, experts] (stat mode) or [layers, experts]
    return (lc.sum(0) if lc.ndim == 3 else lc).to(torch.float64)


def cold_order(cnt: torch.Tensor, ep: int, local: int) -> torch.Tensor:
    """[layers, ep, local] local ids sorted by ascending hit count."""
    layers = cnt.shape[0]
    if cnt.shape[1] != ep * local:
        raise SystemExit(f"experts {cnt.shape[1]} != ep*local {ep*local}")
    per_rank = cnt.view(layers, ep, local)
    # stable sort so ties keep ascending id order (deterministic across runs)
    return per_rank.argsort(dim=-1, stable=True).to(torch.int64)


def spilled_hits_per_token(cnt: torch.Tensor, order: torch.Tensor, spill: int, topk: int) -> float:
    layers, ep, local = order.shape
    per_rank = cnt.view(layers, ep, local)
    cold = order[..., :spill]
    share = per_rank.gather(-1, cold).sum() / per_rank.sum().clamp(min=1)
    return float(share) * topk * layers


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="recorder .pt dumps or globs, oldest first")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ep", type=int, default=8)
    ap.add_argument("--local", type=int, default=48, help="local routed experts per rank")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--spill", type=int, nargs="+", default=[14], help="n_spilled values to report")
    ap.add_argument("--no-holdout", action="store_true", help="use every dump for the table")
    args = ap.parse_args(argv)

    files: List[str] = []
    for d in args.dumps:
        files.extend(sorted(glob.glob(d)) or [d])
    if not files:
        raise SystemExit("no dumps")
    counts = [load_counts(f) for f in files]
    tot = sum(int(c.sum()) for c in counts)
    print(f"{len(files)} dumps, {tot} routed hits, {counts[0].shape[0]} layers x {counts[0].shape[1]} experts")

    train = counts if (args.no_holdout or len(counts) < 2) else counts[:-1]
    fit = torch.stack(train).sum(0)
    order = cold_order(fit, args.ep, args.local)
    layers = fit.shape[0]
    tail = torch.arange(args.local - 1, -1, -1).view(1, 1, -1).expand(layers, args.ep, -1)
    for s in args.spill:
        fit_hits = spilled_hits_per_token(fit, order, s, args.topk)
        line = f"spill {s:2d}/{args.local}: in-sample {fit_hits:5.1f} spilled hits/tok"
        if train is not counts:
            hold = counts[-1]
            line += (f"   held-out {spilled_hits_per_token(hold, order, s, args.topk):5.1f}"
                     f"   (tail placement {spilled_hits_per_token(hold, tail, s, args.topk):5.1f},"
                     f" uniform {s / args.local * args.topk * layers:5.1f})")
        print(line)
    if train is not counts:
        # ship the table fitted on everything; the held-out score above is the honest estimate
        order = cold_order(torch.stack(counts).sum(0), args.ep, args.local)
    meta = {
        # File names only: a shipped table must not carry local paths.
        "source": [os.path.basename(f) for f in files],
        "routed_hits": tot,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ep": args.ep,
        "local": args.local,
    }
    if args.out.endswith(".json"):
        # Plain data for the repo copy; one line per layer keeps diffs readable.
        layers_json = ",\n".join("  " + json.dumps(layer, separators=(",", ":")) for layer in order.tolist())
        with open(args.out, "w") as f:
            f.write(json.dumps(meta)[:-1] + ',\n"cold_ids": [\n' + layers_json + "\n]}\n")
    else:
        torch.save({"cold_ids": order, **meta}, args.out)
    print(f"wrote {args.out}: cold_ids {tuple(order.shape)} (coldest first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
