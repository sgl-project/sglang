"""Summarize e2e run logs pulled into evidence/runs/<run>/test.log.

Usage: python3 summarize_runs.py [--logs DIR] [--split] [run ...]   (default: every run in DIR)

--split reports each server launch ("Launching server" line) of a whole-file run as its own
row, labeled by the test method that follows it; verdict and wall time stay file-level.

Per run: test verdict, exit code, wall time, GSM8K score, EAGLE accept length, the
target prefill graph memory printed by the test, prefill batches that replayed a
CUDA graph versus ran eagerly (summed over DP-rank log lines), the EXTEND autotune
pass line, and the target/draft prefill capture lines from TP rank 0.
"""

import re
import sys
from pathlib import Path

DEFAULT_LOGS = Path(__file__).resolve().parent.parent / "evidence" / "runs"


def _first(pattern, text, group=1, default="-"):
    match = re.search(pattern, text, re.M)
    return match.group(group) if match else default


def summarize(run_dir: Path) -> dict:
    text = (run_dir / "test.log").read_text(errors="replace").replace("\r", "\n")
    return summarize_text(run_dir.name, text)


def split_segments(run_dir: Path) -> list[dict]:
    text = (run_dir / "test.log").read_text(errors="replace").replace("\r", "\n")
    starts = [m.start() for m in re.finditer(r"^.*Launching server.*$", text, re.M)]
    if not starts:
        return [summarize_text(run_dir.name, text)]
    tail = text[starts[-1] :]
    method_re = re.compile(r"\[CI Test Method\] (\w+)\.")
    first_method = method_re.search(text)
    # A test that launches inside its method logs the method line before the
    # launch; one that launches in setUpClass logs it after.
    launches_in_method = first_method is not None and first_method.start() < starts[0]
    rows = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        segment = text[start:end]
        if launches_in_method:
            previous = text[(starts[index - 1] if index else 0) : start]
            before = method_re.findall(previous)
            method = before[-1] if before else "-"
        else:
            method = _first(method_re.pattern, segment)
        row = summarize_text(f"{run_dir.name}#{index + 1}:{method}", segment)
        # Only the file's last segment carries its verdict and wall time.
        row["result"] = _first(r"^(OK|FAILED[^\n]*)$", tail)
        row["exit"] = _first(r"^exit=(\d+)", tail)
        row["seconds"] = "-"
        rows.append(row)
    return rows


def summarize_text(name: str, text: str) -> dict:
    prefill_lines = re.findall(r"Prefill batch.*cuda graph: (True|False)", text)
    captures = re.findall(
        r"\[[^\]]*TP0[^\]]*\] Capture (target|draft) prefill CUDA graph end\. "
        r"elapsed=([\d.]+) s, mem usage=([\d.]+) GB",
        text,
    )
    disabled = re.findall(r"\] (Disable prefill CUDA graph[^\n]*)", text)
    return {
        "run": name,
        "result": _first(r"^(OK|FAILED[^\n]*)$", text),
        "exit": _first(r"^exit=(\d+)", text),
        "seconds": _first(r"^Ran \d+ tests? in ([\d.]+)s", text),
        "score": _first(r"'score': np\.float64\(([\d.]+)\)", text),
        "accept_len": _first(r"avg_spec_accept_length=([\d.]+)", text),
        "prefill_graph_gb": _first(r"prefill_graph_gb=([\d.]+)", text),
        "prefill_replayed": prefill_lines.count("True"),
        "prefill_eager": prefill_lines.count("False"),
        "extend_pass": _first(
            r"FlashInfer autotune: extra EXTEND pass at ([^\n]*)", text
        ),
        "captures": "; ".join(
            f"{role} {elapsed}s/{mem}GB" for role, elapsed, mem in captures
        )
        or "-",
        "disabled": "; ".join(sorted(set(disabled))) or "-",
    }


def main() -> None:
    args = sys.argv[1:]
    logs = DEFAULT_LOGS
    if args[:1] == ["--logs"]:
        logs, args = Path(args[1]), args[2:]
    split = args[:1] == ["--split"]
    if split:
        args = args[1:]
    names = args or sorted(p.name for p in logs.iterdir() if (p / "test.log").is_file())
    if split:
        rows = [row for name in names for row in split_segments(logs / name)]
    else:
        rows = [summarize(logs / name) for name in names]
    keys = list(rows[0])
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for row in rows:
        print("| " + " | ".join(str(row[k]) for k in keys) + " |")


if __name__ == "__main__":
    main()
