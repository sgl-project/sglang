from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(add_completion=False)

_BOB_ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")


def _parse_bob(path: Path) -> Optional[dict[str, float]]:
    if not path.exists():
        return None
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = _BOB_ROW_RE.match(line.strip())
        if match:
            rows.append(
                {
                    "batch_size": float(match.group(1)),
                    "input_len": float(match.group(2)),
                    "latency_s": float(match.group(3)),
                    "input_tput": float(match.group(4)),
                }
            )
    return rows[-1] if rows else None


def _parse_serving(path: Path) -> Optional[dict[str, float]]:
    if not path.exists():
        return None
    result: dict[str, float] = {}
    for line in path.read_text(errors="replace").splitlines():
        for label, key in (
            ("Input token throughput (tok/s):", "input_tput"),
            ("Request throughput (req/s):", "req_tput"),
            ("Mean TTFT (ms):", "mean_ttft_ms"),
            ("Successful requests:", "ok_requests"),
            ("Benchmark duration (s):", "duration_s"),
        ):
            if line.strip().startswith(label):
                result[key] = float(line.split(":")[1].strip())
    return result or None


@app.command()
def main(
    root: Annotated[Path, typer.Argument()],
) -> None:
    """Collect one phase directory (<root>/<variant>-<graph>/) into a markdown table."""
    print(
        "| variant | graph | bob_uniform lat(s) | bob_uniform tok/s | bob_skew lat(s) | "
        "bob_skew tok/s | serve_uniform tok/s | serve_uniform TTFT(ms) | "
        "serve_skew tok/s | serve_skew TTFT(ms) | notes |"
    )
    print("| --- " * 11 + "|")
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        variant, _, graph = run_dir.name.rpartition("-")
        bob_uniform = _parse_bob(run_dir / "bob_uniform.log")
        bob_skew = _parse_bob(run_dir / "bob_skew.log")
        serve_uniform = _parse_serving(run_dir / "serving_uniform.log")
        serve_skew = _parse_serving(run_dir / "serving_skew.log")

        notes = []
        if (run_dir / "RESULT_NOTREADY").exists():
            notes.append("server never became ready")
        if (run_dir / "SKIPPED").exists():
            notes.append((run_dir / "SKIPPED").read_text().strip().replace("\n", "; "))
        if not (run_dir / "DONE").exists():
            notes.append("incomplete")

        def fmt(data: Optional[dict[str, float]], key: str) -> str:
            if data is None or key not in data:
                return "-"
            return f"{data[key]:.2f}" if data[key] < 1000 else f"{data[key]:.0f}"

        print(
            f"| {variant} | {graph} | {fmt(bob_uniform, 'latency_s')} | "
            f"{fmt(bob_uniform, 'input_tput')} | {fmt(bob_skew, 'latency_s')} | "
            f"{fmt(bob_skew, 'input_tput')} | {fmt(serve_uniform, 'input_tput')} | "
            f"{fmt(serve_uniform, 'mean_ttft_ms')} | {fmt(serve_skew, 'input_tput')} | "
            f"{fmt(serve_skew, 'mean_ttft_ms')} | {' / '.join(notes)} |"
        )


if __name__ == "__main__":
    app()
