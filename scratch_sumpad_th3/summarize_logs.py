from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)

_KV_RE = re.compile(r"(\w+)=(\[[^\]]*\]|[^\s]+)")


def _parse_kv(line: str) -> dict[str, str]:
    return {key: value for key, value in _KV_RE.findall(line)}


def _parse_int_list(text: str) -> list[int]:
    inner = text.strip("[]")
    if not inner:
        return []
    return [int(part) for part in inner.split(",")]


@app.command()
def main(
    log_path: Annotated[Path, typer.Argument()],
    rank_filter: Annotated[int, typer.Option()] = -1,
) -> None:
    """Summarize [DPPAD] / [PCG] / [KPROBE] lines of one server log."""
    dppad_modes: Counter[str] = Counter()
    extend_modes: Counter[str] = Counter()
    real_tokens = 0
    padded_tokens = 0
    forced_max_steps = 0
    bcg_ok_steps = 0
    pcg_buckets: Counter[int] = Counter()
    pcg_raw_vs_bucket: list[tuple[int, int]] = []
    kprobe_rows: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    kprobe_real = 0
    kprobe_graph_rows = 0
    kprobe_graph_steps = 0
    kprobe_eager_steps = 0

    for raw_line in log_path.read_text(errors="replace").splitlines():
        if "[DPPAD]" in raw_line:
            fields = _parse_kv(raw_line[raw_line.index("[DPPAD]") :])
            rank = int(fields["rank"])
            if rank_filter >= 0 and rank != rank_filter:
                continue
            dppad_modes[fields["mode"]] += 1
            if fields["is_extend_in_batch"] == "True":
                extend_modes[fields["mode"]] += 1
                real_tokens += int(fields["real_local"])
                padded_tokens += int(fields["padded_local"])
                forced_max_steps += fields["forced_max_by_prefill_cg"] == "True"
                bcg_ok_steps += fields["bcg_ok"] == "True"
        elif "[PCG]" in raw_line:
            fields = _parse_kv(raw_line[raw_line.index("[PCG]") :])
            bucket = int(fields["bucket"])
            pcg_buckets[bucket] += 1
            pcg_raw_vs_bucket.append((int(fields["raw_num_tokens"]), bucket))
        elif "[KPROBE]" in raw_line:
            fields = _parse_kv(raw_line[raw_line.index("[KPROBE]") :])
            rank = int(fields["rank"])
            if rank < 0:
                continue
            if rank_filter >= 0 and rank != rank_filter:
                continue
            mode = int(fields["dp_pad_mode"])
            used_graph = int(fields["used_prefill_graph"])
            graph_rows = int(fields["graph_rows"])
            kprobe_rows[(mode, used_graph)][graph_rows] += 1
            kprobe_real += int(fields["real_local_tokens"])
            kprobe_graph_rows += graph_rows
            if used_graph:
                kprobe_graph_steps += 1
            else:
                kprobe_eager_steps += 1

    print(f"log: {log_path}")
    print(f"rank_filter: {rank_filter}")
    print(f"[DPPAD] all-mode counts       : {dict(dppad_modes)}")
    print(f"[DPPAD] extend-only counts    : {dict(extend_modes)}")
    print(f"[DPPAD] extend steps forced_max_by_prefill_cg: {forced_max_steps}")
    print(f"[DPPAD] extend steps bcg_ok   : {bcg_ok_steps}")
    print(f"[DPPAD] real local tokens     : {real_tokens}")
    print(f"[DPPAD] padded local tokens   : {padded_tokens}")
    if real_tokens:
        print(
            f"[DPPAD] local pad waste       : "
            f"{(padded_tokens - real_tokens) / real_tokens * 100:.2f}%"
        )
    print(f"[PCG] bucket histogram        : {dict(pcg_buckets)}")
    print(f"[PCG] replays                 : {sum(pcg_buckets.values())}")
    print(f"[KPROBE] graphed / eager steps: {kprobe_graph_steps} / {kprobe_eager_steps}")
    print(f"[KPROBE] sum real local tokens: {kprobe_real}")
    print(f"[KPROBE] sum attn rows        : {kprobe_graph_rows}")
    if kprobe_real:
        print(
            f"[KPROBE] attn row waste       : "
            f"{(kprobe_graph_rows - kprobe_real) / kprobe_real * 100:.2f}%"
        )
    for key in sorted(kprobe_rows):
        mode, used_graph = key
        print(
            f"[KPROBE] mode={mode} used_graph={used_graph} rows histogram: "
            f"{dict(sorted(kprobe_rows[key].items()))}"
        )


if __name__ == "__main__":
    app()
