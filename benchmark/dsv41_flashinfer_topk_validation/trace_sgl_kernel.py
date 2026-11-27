"""Check that one tree's top-k selections launch what another tree's launch.

    PYTHONPATH=<tree>/python python3 trace_sgl_kernel.py record --variant <bench_topk.py variant> \
        --tree <tree> --out <variant>.jsonl
    python3 trace_sgl_kernel.py compare base.jsonl branch.jsonl [--repeat base-again.jsonl] \
        [--names base branch]

record: for every shape of bench_topk.py's four suites (same seeded inputs), runs the
variant's selection once to warm up, then once under torch.profiler, and stores each
launched CUDA kernel's name, grid, block, dynamic shared memory and registers per thread,
a sha256 of the selection's output (exact bytes, and with every row sorted), and whether
the output is a valid top-k: the selected entries map back (through the page table, the
candidate block table or the block ids) to distinct in-range columns whose scores, sorted,
equal torch.topk's values over the row's valid columns (for ``blocks``: block maxima, the
newest block's key +inf as amax_topk_blocks defines it).
compare: per suite, how many shapes launch the identical kernel sequence, produce
identical outputs, and are valid top-k on each side.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_topk  # noqa: E402

SUITES = ("paged", "paged16", "sparse", "blocks")


def output_of(suite, inputs, result):
    if suite == "blocks":
        # amax_topk_blocks returns the selected block ids
        return result if torch.is_tensor(result) else result[0]
    return inputs.out


def digest(tensor):
    exact = hashlib.sha256(tensor.contiguous().cpu().numpy().tobytes()).hexdigest()
    rows = tensor.reshape(tensor.shape[0], -1).sort(dim=1).values
    sorted_rows = hashlib.sha256(rows.contiguous().cpu().numpy().tobytes()).hexdigest()
    return exact, sorted_rows


def _check(values, selected_cols, lens_cols, k):
    """``values`` [rows, n] float scores per candidate column, ``selected_cols`` [rows, k]
    selected column ids (-1 padding), ``lens_cols`` [rows] valid column counts."""
    rows, n = values.shape
    col = torch.arange(n, device=values.device)
    masked = values.float().masked_fill(col[None, :] >= lens_cols[:, None], float("-inf"))
    want = torch.clamp(lens_cols, max=k)
    ref = masked.topk(min(k, n), dim=1).values
    for r in range(rows):
        sel = selected_cols[r]
        sel = sel[sel >= 0]
        if sel.numel() != int(want[r]) or sel.unique().numel() != sel.numel():
            return False
        if bool((sel >= lens_cols[r]).any()):
            return False
        got = masked[r, sel.long()].sort(descending=True).values
        if not torch.equal(got, ref[r, : int(want[r])]):
            return False
    return True


def valid_topk(suite, inputs, output):
    lens = inputs.lens.long()
    if suite in ("paged", "paged16"):
        table = inputs.page_table.long()
        page = bench_topk.PAGE_SIZE
        inverse = torch.full((table.shape[0], int(table.max()) + 1), -1, dtype=torch.long, device=table.device)
        inverse.scatter_(1, table, torch.arange(table.shape[1], device=table.device).expand_as(table).contiguous())
        slots = output.long()
        pages = inverse.gather(1, (slots.clamp_min(0) // page))
        cols = torch.where((slots >= 0) & (pages >= 0), pages * page + slots % page, -1)
        return _check(inputs.scores, cols, lens, inputs.k)
    if suite == "sparse":
        blocks = inputs.blocks.long()
        slots = output.long()
        block = slots.clamp_min(0) // bench_topk.BLOCK
        pos = torch.searchsorted(blocks, block)
        found = blocks.gather(1, pos.clamp_max(blocks.shape[1] - 1)) == block
        cols = torch.where((slots >= 0) & found, pos * bench_topk.BLOCK + slots % bench_topk.BLOCK, -1)
        return _check(inputs.scores, cols, lens, inputs.k)
    if suite == "blocks":
        rows, width = inputs.scores.shape
        col = torch.arange(width, device=lens.device)
        masked = inputs.scores.float().masked_fill(col[None, :] >= lens[:, None], float("-inf"))
        maxima = masked.view(rows, width // bench_topk.BLOCK, bench_topk.BLOCK).amax(-1)
        # amax_topk_blocks always keeps the newest block: its key is +inf
        nblocks = inputs.nblocks.long()
        newest = (nblocks - 1).clamp_min(0)[:, None]
        maxima.scatter_(1, newest, torch.where(nblocks[:, None] > 0, float("inf"), maxima.gather(1, newest)))
        return _check(maxima, output.long(), nblocks, bench_topk.TOPK_BLOCKS)
    raise ValueError(suite)


def kernels_of(call):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        result = call()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    prof.export_chrome_trace(path)
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    os.unlink(path)
    launched = []
    for event in sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"]):
        args = event.get("args", {})
        launched.append(
            [
                event["name"],
                args.get("grid"),
                args.get("block"),
                args.get("shared memory"),
                args.get("registers per thread"),
            ]
        )
    return launched, result


def record(args):
    with open(args.out, "w") as out:
        for suite in SUITES:
            for label, _model, k, lens in bench_topk.shapes(suite):
                inputs = bench_topk.make_inputs(suite, label, k, lens)
                call = bench_topk.build_call(args.variant, suite, inputs)
                call()
                torch.cuda.synchronize()
                if suite != "blocks":
                    inputs.out.fill_(-7)
                launched, result = kernels_of(call)
                output = output_of(suite, inputs, result)
                exact, sorted_rows = digest(output)
                valid = valid_topk(suite, inputs, output)
                out.write(
                    json.dumps(
                        {
                            "suite": suite,
                            "label": label,
                            "variant": args.variant,
                            "kernels": launched,
                            "out_sha256": exact,
                            "out_sorted_rows_sha256": sorted_rows,
                            "valid_topk": valid,
                        }
                    )
                    + "\n"
                )
                out.flush()
                print(suite, label, len(launched), "kernels", "valid" if valid else "INVALID", flush=True)
                del inputs, call, result
                torch.cuda.empty_cache()


def compare(args):
    def load(path):
        with open(path) as f:
            return {(r["suite"], r["label"]): r for r in map(json.loads, f)}

    def identity_rows(a, b):
        rows = []
        for suite in SUITES:
            keys = [k for k in a if k[0] == suite and k in b]
            rows.append(
                (
                    suite,
                    len(keys),
                    sum(a[k]["kernels"] == b[k]["kernels"] for k in keys),
                    sum(a[k]["out_sha256"] == b[k]["out_sha256"] for k in keys),
                    sum(a[k]["out_sorted_rows_sha256"] == b[k]["out_sorted_rows_sha256"] for k in keys),
                    sum(bool(a[k].get("valid_topk")) for k in keys),
                    sum(bool(b[k].get("valid_topk")) for k in keys),
                )
            )
        return rows

    base, branch = load(args.base), load(args.branch)
    header = (
        "| suite | shapes | identical kernel sequence (name, grid, block, smem, regs) | identical output | "
        "identical output, rows sorted | valid top-k, {a} | valid top-k, {b} |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header.format(a=args.names[0], b=args.names[1]), sep]
    for row in identity_rows(base, branch):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    if args.repeat:
        repeat = load(args.repeat)
        lines += ["", "Base run twice (same tree, same inputs):", "", header.format(a="base", b="base again"), sep]
        for row in identity_rows(base, repeat):
            lines.append("| " + " | ".join(map(str, row)) + " |")
    missing = sorted(set(base) ^ set(branch))
    print("\n".join(lines))
    print(f"\nshapes missing on one side: {len(missing)}")
    kernel_names = sorted({kern[0] for r in base.values() for kern in r["kernels"]})
    print(f"kernels launched by the {args.names[0]} variant:", json.dumps(kernel_names, indent=1))
    mismatches = [k for k in base if k in branch and base[k]["kernels"] != branch[k]["kernels"]]
    for k in mismatches[:5]:
        print("kernel mismatch", k, json.dumps(base[k]["kernels"]), json.dumps(branch[k]["kernels"]))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument(
        "--variant",
        required=True,
        choices=[
            "sgl-kernel-base",
            "sgl-kernel",
            "sgl-kernel-mode2",
            "flashinfer",
            "flashinfer-small",
            "flashinfer-small-det",
        ],
    )
    r.add_argument("--tree", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("branch")
    c.add_argument("--repeat", help="a second record of the base variant (A/A)")
    c.add_argument("--names", nargs=2, default=["base", "branch"], help="column labels")
    args = parser.parse_args()
    {"record": record, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
