"""Invoke the pinned scikit-build-core backend without an ambient frontend."""

import sys
from pathlib import Path

from scikit_build_core.build import build_wheel


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: bazel_pep517_build.py OUTPUT_DIR BUILD_DIR")

    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel = build_wheel(
        str(output_dir),
        {"build-dir": sys.argv[2]},
    )
    if not (output_dir / wheel).is_file():
        raise RuntimeError(f"backend reported missing wheel: {wheel}")


if __name__ == "__main__":
    main()
