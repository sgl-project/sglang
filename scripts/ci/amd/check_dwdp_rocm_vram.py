#!/usr/bin/env python3

import argparse
import csv
import io
import subprocess
import sys


GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject MI355X nodes with occupied or unbalanced VRAM."
    )
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--max-used-gib", type=float, default=4.0)
    parser.add_argument("--max-skew-gib", type=float, default=2.0)
    return parser.parse_args()


def query_vram() -> list[tuple[str, int, int]]:
    result = subprocess.run(
        ["rocm-smi", "--showmeminfo", "vram", "--csv"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"rocm-smi failed ({result.returncode}): {detail}")

    lines = result.stdout.splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith("device,")
        )
    except StopIteration as exc:
        raise RuntimeError(f"rocm-smi returned no CSV header: {result.stdout!r}") from exc

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    rows = []
    for row in reader:
        device = (row.get("device") or "").strip()
        if not device.startswith("card"):
            continue
        total = int(row["VRAM Total Memory (B)"])
        used = int(row["VRAM Total Used Memory (B)"])
        rows.append((device, total, used))
    return sorted(rows)


def main() -> int:
    args = parse_args()
    try:
        rows = query_vram()
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError, KeyError) as exc:
        print(f"VRAM preflight failed: {exc}", file=sys.stderr)
        return 2

    if len(rows) != args.expected_gpus:
        print(
            f"VRAM preflight failed: expected {args.expected_gpus} GPUs, got {len(rows)}",
            file=sys.stderr,
        )
        return 3

    used_gib = [used / GIB for _, _, used in rows]
    free_gib = [(total - used) / GIB for _, total, used in rows]
    max_used = max(used_gib)
    skew = max(used_gib) - min(used_gib)
    print(
        "VRAM preflight: "
        f"used_gib=[{', '.join(f'{value:.2f}' for value in used_gib)}], "
        f"min_free_gib={min(free_gib):.2f}, "
        f"max_used_gib={max_used:.2f}, skew_gib={skew:.2f}"
    )

    violations = []
    if max_used > args.max_used_gib:
        violations.append(
            f"max used {max_used:.2f} GiB exceeds {args.max_used_gib:.2f} GiB"
        )
    if skew > args.max_skew_gib:
        violations.append(f"skew {skew:.2f} GiB exceeds {args.max_skew_gib:.2f} GiB")
    if violations:
        print("VRAM preflight rejected node: " + "; ".join(violations), file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
