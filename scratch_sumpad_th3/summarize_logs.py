from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Optional

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


class _Bucket:
    def __init__(self) -> None:
        self.steps = 0
        self.modes: Counter[str] = Counter()
        self.real_tokens = 0
        self.padded_tokens = 0
        self.forced_max_steps = 0
        self.bcg_ok_steps = 0

    def waste_pct(self) -> Optional[float]:
        if not self.real_tokens:
            return None
        return (self.padded_tokens - self.real_tokens) / self.real_tokens * 100

    def render(self, label: str) -> str:
        waste = self.waste_pct()
        waste_text = "n/a" if waste is None else f"{waste:.2f}%"
        return (
            f"[DPPAD] {label:<12s} steps={self.steps:<5d} modes={dict(self.modes)} "
            f"forced_max_steps={self.forced_max_steps} bcg_ok_steps={self.bcg_ok_steps} "
            f"real={self.real_tokens} padded={self.padded_tokens} waste={waste_text}"
        )


@app.command()
def main(
    log_path: Annotated[Path, typer.Argument()],
) -> None:
    """Summarize [DPPAD] / [PCG] / [KPROBE] lines of one server log.

    Steps are counted from rank-0 lines only; token sums cover every rank. Extend steps
    are split by whether any DP rank was idle, because that is what decides whether
    MAX_LEN padding fabricates work.
    """
    all_active = _Bucket()
    has_idle = _Bucket()
    decode_steps = 0
    pcg_buckets: Counter[int] = Counter()
    kprobe_rows: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    capture_rows: Counter[int] = Counter()
    kprobe_real = 0
    kprobe_rows_total = 0
    kprobe_graph_lines = 0
    kprobe_eager_lines = 0

    for raw_line in log_path.read_text(errors="replace").splitlines():
        if "[DPPAD]" in raw_line:
            fields = _parse_kv(raw_line[raw_line.index("[DPPAD]") :])
            rank = int(fields["rank"])
            is_extend = fields["is_extend_in_batch"] == "True"
            if not is_extend:
                if rank == 0:
                    decode_steps += 1
                continue
            raw_tokens = _parse_int_list(fields["raw"])
            bucket = has_idle if (not raw_tokens or min(raw_tokens) == 0) else all_active
            bucket.real_tokens += int(fields["real_local"])
            bucket.padded_tokens += int(fields["padded_local"])
            if rank == 0:
                bucket.steps += 1
                bucket.modes[fields["mode"]] += 1
                bucket.forced_max_steps += fields["forced_max_by_prefill_cg"] == "True"
                bucket.bcg_ok_steps += fields["bcg_ok"] == "True"
        elif "[PCG]" in raw_line:
            fields = _parse_kv(raw_line[raw_line.index("[PCG]") :])
            pcg_buckets[int(fields["bucket"])] += 1
        elif "[KPROBE]" in raw_line:
            fields = _parse_kv(raw_line[raw_line.index("[KPROBE]") :])
            if int(fields["rank"]) < 0:
                continue
            mode = int(fields["dp_pad_mode"])
            graph_rows = int(fields["graph_rows"])
            if mode == 0:
                capture_rows[graph_rows] += 1
                continue
            used_graph = int(fields["used_prefill_graph"])
            kprobe_rows[(mode, used_graph)][graph_rows] += 1
            kprobe_real += int(fields["real_local_tokens"])
            kprobe_rows_total += graph_rows
            if used_graph:
                kprobe_graph_lines += 1
            else:
                kprobe_eager_lines += 1

    print(f"log: {log_path}")
    print("note: steps are global (rank-0 lines); token sums cover all ranks")
    print(all_active.render("all-active"))
    print(has_idle.render("has-idle"))
    print(f"[DPPAD] decode-only steps   : {decode_steps}")
    print(f"[PCG] replay lines (rank-local): {sum(pcg_buckets.values())} {dict(sorted(pcg_buckets.items()))}")
    print(f"[KPROBE] capture launches   : {sum(capture_rows.values())}")
    print(f"[KPROBE] graphed / eager lines: {kprobe_graph_lines} / {kprobe_eager_lines}")
    print(f"[KPROBE] real tokens / attn rows: {kprobe_real} / {kprobe_rows_total}")
    if kprobe_real:
        print(
            f"[KPROBE] attn row waste     : "
            f"{(kprobe_rows_total - kprobe_real) / kprobe_real * 100:.2f}%"
        )
    print("[KPROBE] warning: mode reported here is live metadata; a replayed prefill graph")
    print("[KPROBE]          always executes the MAX_LEN collectives it was captured with")
    for key in sorted(kprobe_rows):
        mode, used_graph = key
        print(
            f"[KPROBE] mode={mode} used_graph={used_graph} rows histogram: "
            f"{dict(sorted(kprobe_rows[key].items()))}"
        )


if __name__ == "__main__":
    app()
