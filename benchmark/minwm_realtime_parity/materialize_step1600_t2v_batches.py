#!/usr/bin/env python3
"""Materialize the archived 8-case and reconstructed 6-case baseline inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

EARLIER_TRAJECTORIES = (
    (1, "w*181"),
    (2, "W*60,k*30,w*60,d*31"),
    (3, "a*80,d*81"),
    (5, "W*60,a*60,w*61"),
    (6, "w*181"),
    (8, "w*181"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = Path(args.prompts).read_text(encoding="utf-8").splitlines()
    trajectories = Path(args.trajectories).read_text(encoding="utf-8").splitlines()
    if len(prompts) != 8 or len(trajectories) != 8:
        raise ValueError(
            f"expected 8 archived prompt/trajectory rows, got "
            f"{len(prompts)}/{len(trajectories)}"
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_lines(output / "later8_prompts.txt", prompts)
    write_lines(output / "later8_trajectories.txt", trajectories)
    write_lines(
        output / "earlier6_prompts.txt",
        [prompts[line_number - 1] for line_number, _ in EARLIER_TRAJECTORIES],
    )
    write_lines(
        output / "earlier6_trajectories.txt",
        [trajectory for _, trajectory in EARLIER_TRAJECTORIES],
    )


if __name__ == "__main__":
    main()
