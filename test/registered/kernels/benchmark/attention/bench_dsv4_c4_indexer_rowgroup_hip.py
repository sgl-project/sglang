"""DeepSeek-V4 C4 FP4 indexer logits on gfx950: today's schedule kernels against AITER's
row-group kernel (SGLANG_HIP_DSV4_C4_INDEXER_ROWGROUP) for decode (1 row a request) and
DSpark target-verify (4 or 7 rows a request).

Shapes are the serving ones: 64 heads, 64-slot index-K pages, a page table as wide as the
captured graph's (``--max-context``, the server's context length), every request at the
same context. Each timed launch reads its own pages (``--rotate-mb`` of index K in all), so
the keys come from HBM as they do between layers.

* ``sched_kern``: per layer, today's kernel with its workspace built: the varctx decode
  kernel (decode) or the prefill kernel (verify).
* ``sched_ws``: per step, today's workspace build in the graph: the decode schedule or
  the prefill schedule (DSpark verify builds it in the graph, upgraded from raw metadata).
* ``rg_grp`` / ``rg_row``: per layer, the row-group kernel, a request's rows one sequence /
  each row its own (decode: one row a request either way, shown as ``rg_row``).
* ``rg_ws``: per step, the row-group workspace build (in the graph).
* ``step_old`` / ``step_new``: per step over ``--c4-layers`` C4 layers (DeepSeek-V4-Pro:
  30), workspace included; ``step_new`` is what the backend picks: decode on ``rg_row``,
  verify on ``rg_grp`` from ``C4_ROWGROUP_MIN_VERIFY_ROWS`` rows, else today's path.

Kernel and workspace times are CUDA-graph replays divided by the launches in them.
"""

from __future__ import annotations

import argparse

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    C4_ROWGROUP_MIN_VERIFY_ROWS,
    aiter_fp4_paged_mqa_logits,
    prepare_fp4_decode_workspace,
    prepare_fp4_prefill_workspace,
    prepare_fp4_rowgroup_decode_workspace,
    rowgroup_table_columns,
)
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(
    est_time=300,
    stage="jit-kernel-benchmark",
    runner_config="amd",
    disabled="manual microbenchmark",
)

HEADS = 64
HEAD_DIM = 128
PAGE = 64
RATIO = 4
PAGE_BYTES = PAGE * HEAD_DIM // 2 + PAGE * HEAD_DIM // 32


def _graph_time_us(fn, launches: int, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3 / iters / launches


def _case(bs: int, ctx: int, rows: int, max_context: int, rotate_mb: int):
    dev = "cuda"
    keys = ctx // RATIO
    pages_per_seq = -(-keys // PAGE)
    step_pages = bs * pages_per_seq
    n_rot = max(1, min(32, rotate_mb * 2**20 // (step_pages * PAGE_BYTES)))
    num_pages = n_rot * step_pages
    payload = torch.randint(
        0, 256, (num_pages, 1, 4, PAGE, 16), dtype=torch.uint8, device=dev
    )
    scale = torch.randint(
        110, 130, (num_pages, 1, 4, PAGE), dtype=torch.uint8, device=dev
    )
    width = rowgroup_table_columns(max_context, RATIO) - 4
    n = bs * rows
    pos = torch.tensor(
        [ctx - rows + r for _ in range(bs) for r in range(rows)], device=dev
    )
    row_ends = ((pos + 1) // RATIO).to(torch.int32)
    steps = []
    perm = torch.randperm(num_pages, device=dev).to(torch.int32)
    for i in range(n_rot):
        table = torch.zeros(bs, width, dtype=torch.int32, device=dev)
        table[:, :pages_per_seq] = perm[i * step_pages : (i + 1) * step_pages].view(
            bs, pages_per_seq
        )
        steps.append(
            dict(
                q_fp4=torch.randint(
                    0, 256, (n, HEADS, HEAD_DIM // 2), dtype=torch.uint8, device=dev
                ),
                q_scale=torch.randint(
                    120, 128, (n, 1, 4, 16, 4), dtype=torch.uint8, device=dev
                ),
                weights=torch.randn(n, HEADS, device=dev, dtype=torch.bfloat16),
                page_table=table.repeat_interleave(rows, dim=0).contiguous(),
            )
        )
    return payload, scale, row_ends, steps


def _logits(payload, scale, row_ends, step, is_decode, decode_ws=None, prefill_ws=None):
    return aiter_fp4_paged_mqa_logits(
        q_fp4=step["q_fp4"],
        q_scale=step["q_scale"],
        k_payload=payload,
        k_scale=scale,
        weights=step["weights"],
        page_table=step["page_table"],
        c4_seq_lens=row_ends,
        weight_scale=0.01,
        is_decode=is_decode,
        decode_workspace=decode_ws,
        prefill_workspace=prefill_ws,
    )


def bench(bs, ctx, rows, args):
    payload, scale, row_ends, steps = _case(
        bs, ctx, rows, args.max_context, args.rotate_mb
    )
    columns = rowgroup_table_columns(args.max_context, RATIO)
    launches = len(steps)
    out = {}

    def rg_ws(step, **kw):
        kw.setdefault("min_grouped_seqs", 1)
        return prepare_fp4_rowgroup_decode_workspace(
            step["page_table"], kw.pop("rows", rows), table_columns=columns, **kw
        )

    for name, kw in (("rg_grp", {}), ("rg_row", {"rows": 1})):
        if name == "rg_grp" and rows == 1:
            continue
        wss = [rg_ws(s, **dict(kw)) for s in steps]

        def run(wss=wss):
            for s, ws in zip(steps, wss):
                _logits(payload, scale, row_ends, s, True, decode_ws=ws)

        out[name] = _graph_time_us(run, launches, args.iters)
    ws_reps = 8

    def rg_ws_build():
        for _ in range(ws_reps):
            rg_ws(steps[0])

    out["rg_ws"] = _graph_time_us(rg_ws_build, ws_reps, args.iters)

    if rows == 1:
        wss = [prepare_fp4_decode_workspace(s["page_table"], row_ends) for s in steps]

        def run_sched():
            for s, ws in zip(steps, wss):
                _logits(payload, scale, row_ends, s, True, decode_ws=ws)

        def sched_ws_build():
            for _ in range(ws_reps):
                prepare_fp4_decode_workspace(steps[0]["page_table"], row_ends)

        out["sched_ws"] = _graph_time_us(sched_ws_build, ws_reps, args.iters)
    else:
        wss = [prepare_fp4_prefill_workspace(s["page_table"], row_ends) for s in steps]

        def run_sched():
            for s, ws in zip(steps, wss):
                _logits(payload, scale, row_ends, s, False, prefill_ws=ws)

        ws0 = wss[0]

        def sched_ws_build():
            for _ in range(ws_reps):
                prepare_fp4_prefill_workspace(
                    steps[0]["page_table"], row_ends, workspace=ws0
                )

        out["sched_ws"] = _graph_time_us(sched_ws_build, ws_reps, args.iters)
    out["sched_kern"] = _graph_time_us(run_sched, launches, args.iters)
    layers = args.c4_layers
    out["step_old"] = layers * out["sched_kern"] + out["sched_ws"]
    if rows == 1:
        out["step_new"] = layers * out["rg_row"] + out["rg_ws"]
    elif bs * rows >= C4_ROWGROUP_MIN_VERIFY_ROWS:
        out["step_new"] = layers * out["rg_grp"] + out["rg_ws"]
    else:
        out["step_new"] = out["step_old"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,2,4,8,16,32,48")
    parser.add_argument("--ctx", default="8192,32768,65536,131072,200000")
    parser.add_argument("--rows", default="1,4,7")
    parser.add_argument("--max-context", type=int, default=1 << 20)
    parser.add_argument("--rotate-mb", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--c4-layers", type=int, default=30)
    args = parser.parse_args()
    cols = (
        "sched_kern",
        "rg_grp",
        "rg_row",
        "sched_ws",
        "rg_ws",
        "step_old",
        "step_new",
    )
    print(
        f"min_verify_rows={C4_ROWGROUP_MIN_VERIFY_ROWS} max_context={args.max_context}"
    )
    print("| rows | bs | ctx | " + " | ".join(cols) + " | step x |")
    print("|---" * (len(cols) + 4) + "|")
    for rows in map(int, args.rows.split(",")):
        for bs in map(int, args.batches.split(",")):
            for ctx in map(int, args.ctx.split(",")):
                r = bench(bs, ctx, rows, args)
                cells = " | ".join(f"{r[c]:.1f}" if c in r else "-" for c in cols)
                print(
                    f"| {rows} | {bs} | {ctx} | {cells} | "
                    f"{r['step_old'] / r['step_new']:.3f} |",
                    flush=True,
                )
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
