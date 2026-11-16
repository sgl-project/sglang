"""Per TopKSelector entry point, the kernels one tree launches and what it selects.

    PYTHONPATH=<tree>/python [variant env] python3 trace_entry_points.py record \
        --variant <bench_topk.py branch variant> --out <name>.jsonl
    python3 trace_entry_points.py compare a.jsonl b.jsonl [--names a b]

record: seeded inputs for every TopKSelector call the V4.1 indexer makes, beyond the
decode shapes trace_sgl_kernel.py covers: prefill-shaped ``ragged`` on padded bf16 and
fp32 rows (own top-k plus offsets, and the fp32 block keys), ``raw`` with its top-k v2
plan, ``dense`` unsorted and sorted, one- and two-output ``sparse``, ``order_candidates``,
``paged_from_metadata`` on bf16 rows, top-k v2, a row chunk with its own plan and
top-k v1, ``amax_blocks``, and the ``candidate_blocks`` helpers that take
``topk_op=selector.dense``. Each call runs once to warm up and once under torch.profiler;
the record keeps every launched kernel (name, grid, block, dynamic shared memory,
registers per thread) and sha256 digests of the outputs, exact and with each row sorted.
compare: per entry point, whether both records launch the same kernel sequence and
produce the same outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_topk  # noqa: E402
from trace_sgl_kernel import digest, kernels_of  # noqa: E402

DEV = "cuda"
K = 512
BLOCK = 8


def _gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def _rows(rows, width, dtype, seed, pad=64):
    """Seeded random score rows as a padded view, like DeepGEMM's logits."""
    storage = torch.randn(rows, width + pad, generator=_gen(seed), device=DEV).to(dtype)
    return storage[:, :width]


def _lens(rows, low, high, seed):
    return torch.randint(low, high + 1, (rows,), generator=_gen(seed), device=DEV, dtype=torch.int32)


def _table(rows, pages, seed):
    return torch.randperm(rows * pages, generator=_gen(seed), device=DEV).view(rows, pages).to(torch.int32)


def _sparse_tables(valid_lens, topk_blocks, seed):
    """Ascending logical block ids (``-1`` past each row's blocks) and pool blocks."""
    rows = valid_lens.shape[0]
    logical = torch.sort(torch.rand(rows, 4 * topk_blocks, generator=_gen(seed), device=DEV).argsort(-1)[:, :topk_blocks], -1).values
    used = (valid_lens + BLOCK - 1) // BLOCK
    logical = torch.where(torch.arange(topk_blocks, device=DEV)[None, :] < used[:, None], logical, -1).to(torch.int32)
    physical = torch.where(logical >= 0, (logical * 7 + 3) % (8 * topk_blocks), -1).to(torch.int32)
    return logical, physical


def _metadata(lens, table, page_size, use_v2):
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

    return SimpleNamespace(
        compressed_seq_lens=lens,
        page_table=table,
        compressed_page_size=page_size,
        use_topk_v2=use_v2,
        topk_metadata=plan_topk_v2(lens) if use_v2 else torch.empty((0,)),
    )


def entry_points(selector):
    """(name, zero-argument call returning the output tensors to digest)."""
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
        candidate_topk,
        select_candidate_block_ids,
        topk_among_blocks,
    )
    from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

    points = []

    for dtype, name in ((torch.bfloat16, "bf16"), (torch.float32, "fp32")):
        for rows, width, seed in ((64, 8192, 1), (8, 65536, 2)):
            scores = _rows(rows, width, dtype, seed)
            lens = _lens(rows, 1, width, seed + 100)
            offsets = _lens(rows, 0, 1 << 20, seed + 200)
            out = torch.empty(rows, K, dtype=torch.int32, device=DEV)

            def ragged(scores=scores, lens=lens, offsets=offsets, out=out):
                selector.ragged(scores, lens, out=out, offsets=offsets)
                return [out]

            points.append((f"ragged {name} {rows}x{width}", ragged))

    keys = _rows(32, 4096, torch.float32, 3)
    key_lens = _lens(32, 1, 4096, 103)
    blocks_out = torch.empty(32, 2048, dtype=torch.int32, device=DEV)
    zero = torch.zeros(32, dtype=torch.int32, device=DEV)

    def ragged_block_keys():
        selector.ragged(keys, key_lens, out=blocks_out, offsets=zero)
        return [blocks_out]

    points.append(("ragged fp32 block keys 32x4096 k2048", ragged_block_keys))

    raw_scores = _rows(32, 16384, torch.float32, 4)
    raw_lens = _lens(32, 1, 16384, 104)
    raw_out = torch.empty(32, K, dtype=torch.int32, device=DEV)

    def raw():
        selector.raw(raw_scores, raw_lens, out=raw_out, plan=plan_topk_v2(raw_lens))
        return [raw_out]

    points.append(("raw fp32 32x16384", raw))

    dense_scores = torch.randn(32, 3000, generator=_gen(5), device=DEV)
    for sorted_ in (False, True):

        def dense(sorted_=sorted_):
            values, indices = selector.dense(dense_scores, K, sorted=sorted_)
            return [values, indices]

        points.append((f"dense fp32 32x3000 sorted={sorted_}", dense))

    for outputs in (1, 2):
        topk_blocks = 2048
        logits = _rows(64, topk_blocks * BLOCK, torch.bfloat16, 6, pad=0).contiguous()
        valid_lens = _lens(64, 0, topk_blocks * BLOCK, 106)
        logical, physical = _sparse_tables(valid_lens, topk_blocks, 206)
        outs = [torch.empty(64, K, dtype=torch.int32, device=DEV) for _ in range(outputs)]

        def sparse(logits=logits, valid_lens=valid_lens, logical=logical, physical=physical, outs=outs):
            pairs = [(logical, outs[0])] + ([(physical, outs[1])] if len(outs) > 1 else [])
            selector.sparse(logits, valid_lens, pairs)
            return outs

        points.append((f"sparse bf16 64x16384 outputs={outputs}", sparse))

    published = torch.randint(-1, 4096, (64, 64), generator=_gen(7), device=DEV, dtype=torch.int32)

    def order_candidates():
        return [selector.order_candidates(published)]

    points.append(("order_candidates 64x64", order_candidates))

    for dtype, use_v2, chunk, label in (
        (torch.bfloat16, True, False, "bf16"),
        (torch.float32, True, False, "fp32 v2"),
        (torch.float32, True, True, "fp32 v2 row chunk + plan"),
        (torch.float32, False, False, "fp32 v1"),
    ):
        rows, width, page_size = 32, 16384, 64
        scores = _rows(rows, width, dtype, 8)
        lens = _lens(rows, 1, width, 108)
        meta = _metadata(lens, _table(rows, width // page_size, 208), page_size, use_v2)
        out = torch.empty(rows, K, dtype=torch.int32, device=DEV)

        def paged(scores=scores, meta=meta, out=out, chunk=chunk):
            if chunk:
                rows_ = slice(8, 24)
                selector.paged_from_metadata(
                    scores[rows_], meta, out, rows=rows_, plan=plan_topk_v2(meta.compressed_seq_lens[rows_])
                )
            else:
                selector.paged_from_metadata(scores, meta, out)
            return [out]

        points.append((f"paged {label} {rows}x{width}", paged))

    amax_logits = _rows(16, 65536, torch.bfloat16, 9)
    amax_lens = _lens(16, 1, 65536, 109)

    def amax_blocks():
        return [selector.amax_blocks(amax_logits, amax_lens, (amax_lens + 7) // 8, 2048)]

    points.append(("amax_blocks bf16 16x65536 k2048", amax_blocks))

    block_logits = torch.randn(16, 4000, generator=_gen(10), device=DEV)
    compress_lens = _lens(16, 1, 4000, 110)

    def candidate_block_ids():
        return [select_candidate_block_ids(block_logits, compress_lens[:, None], 64, BLOCK, topk_op=selector.dense)]

    points.append(("select_candidate_block_ids 16x4000", candidate_block_ids))

    cand_scores = torch.randn(16, 4000, generator=_gen(11), device=DEV)
    cand_lens = _lens(16, 1, 4000, 111)
    cand_blocks = selector.order_candidates(
        torch.randint(-1, 500, (16, 96), generator=_gen(211), device=DEV, dtype=torch.int32)
    )

    def among_blocks():
        return [topk_among_blocks(cand_scores, cand_lens, cand_blocks, K, block_size=BLOCK, topk_op=selector.dense)]

    points.append(("topk_among_blocks 16x4000 n96", among_blocks))

    flat = torch.randn(16, 96 * BLOCK, generator=_gen(12), device=DEV)

    def cand_topk():
        return [candidate_topk(flat, cand_blocks, K, BLOCK, topk_op=selector.dense)]

    points.append(("candidate_topk 16x768", cand_topk))
    return points


def record(args):
    selector = bench_topk.make_selector(args.variant)
    with open(args.out, "w") as out, torch.inference_mode():
        for name, call in entry_points(selector):
            call()
            torch.cuda.synchronize()
            launched, outputs = kernels_of(call)
            out.write(
                json.dumps(
                    {
                        "entry": name,
                        "variant": args.variant,
                        "kernels": launched,
                        "out_sha256": [digest(t)[0] for t in outputs],
                        "out_sorted_rows_sha256": [digest(t)[1] for t in outputs],
                    }
                )
                + "\n"
            )
            out.flush()
            print(name, len(launched), "kernels", flush=True)


def compare(args):
    def load(path):
        with open(path) as f:
            return {r["entry"]: r for r in map(json.loads, f)}

    a, b = load(args.a), load(args.b)
    lines = [
        f"| entry point | kernels, {args.names[0]} | identical kernel sequence | identical output | "
        "identical output, rows sorted |",
        "|---|---:|---|---|---|",
    ]
    for entry in a:
        if entry not in b:
            lines.append(f"| {entry} | {len(a[entry]['kernels'])} | missing in {args.names[1]} | | |")
            continue
        x, y = a[entry], b[entry]
        lines.append(
            f"| {entry} | {len(x['kernels'])} | {'yes' if x['kernels'] == y['kernels'] else 'NO'} | "
            f"{'yes' if x['out_sha256'] == y['out_sha256'] else 'no'} | "
            f"{'yes' if x['out_sorted_rows_sha256'] == y['out_sorted_rows_sha256'] else 'no'} |"
        )
    same = sum(a[e]["kernels"] == b.get(e, {}).get("kernels") for e in a)
    lines += ["", f"entry points with identical kernel sequences: {same} / {len(a)}; only in {args.names[1]}: {sorted(set(b) - set(a))}"]
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument(
        "--variant",
        required=True,
        choices=["sgl-kernel", "sgl-kernel-mode2", "flashinfer", "flashinfer-small", "flashinfer-small-det"],
    )
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--names", nargs=2, default=["a", "b"], help="column labels")
    args = parser.parse_args()
    {"record": record, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
