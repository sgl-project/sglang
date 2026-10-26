#!/usr/bin/env python3
"""List the diffusion pytest targets of one NPU shard (coverage collection only).

This is an NPU CI helper: it imports the shared multimodal_gen runner
READ-ONLY. The shard composition must match what
``python/sglang/multimodal_gen/test/run_suite.py`` would execute, so the LPT
assignment is reused instead of reimplemented, but no platform-shared test
code is modified.

Output: one pytest target per line, with absolute paths, e.g.
    <workspace>/python/sglang/multimodal_gen/test/server/ascend/test_server_1_npu.py::TestDiffusionServerOneNpu::test_diffusion_generation[ernie_image_t2i_1npu]
    <workspace>/python/sglang/multimodal_gen/test/server/ascend/test_glm_image_distributed.py

Absolute paths matter: run_tests_with_coverage.sh encodes them into the
coverage directory name and test_selector.py recovers the repo-relative test
path from the ``__sglang__sglang__`` workspace marker.

Usage:
    python3 list_multimodal_tests.py --suite 1-npu \
        --partition-id 0 --total-partitions 2 -o /tmp/selected_tests.txt
"""

import argparse
import sys
from pathlib import Path

# Repo layout: this script lives at <repo>/scripts/ci/npu/precise-test/.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

# The runner imports sglang (torch / torch_npu), so make the checkout importable
# even when the package is not installed in editable mode.
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.multimodal_gen.test.runner import (  # noqa: E402
    diffusion_suite_runner as runner,
)
from sglang.multimodal_gen.test.runner.pytest_runner import (  # noqa: E402
    collect_test_items,
    partition_items_by_index,
)


def _case_files(suite: str, server_dir: Path) -> dict:
    """Map case id -> absolute test file for a parametrized diffusion suite."""
    mapping = {}
    for filename, case_group in runner.PARAMETRIZED_CASE_GROUPS.get(suite, []):
        file_path = server_dir / filename
        for case in case_group:
            mapping[case.id] = file_path
    return mapping


def _node_targets(file_path: Path, case_ids: list) -> list:
    """Expand case ids into absolute pytest node-id targets for one file."""
    filter_expr = " or ".join(f"[{case_id}]" for case_id in case_ids)
    targets = []
    for item in collect_test_items([str(file_path)], filter_expr):
        # collect_test_items reports node ids relative to pytest's rootdir,
        # which is not the checkout root. Rebuild them from the absolute file
        # so the coverage collector can encode the workspace layout.
        node = item.partition("::")[2]
        targets.append(f"{file_path}::{node}" if node else str(file_path))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Diffusion suite, e.g. 1-npu")
    parser.add_argument("--partition-id", type=int, default=0)
    parser.add_argument("--total-partitions", type=int, default=1)
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Write the pytest targets (one per line) to this file.",
    )
    args = parser.parse_args()

    test_root = Path(runner.__file__).resolve().parents[1]
    server_dir = test_root / "server"

    if args.suite in runner.PARAMETRIZED_CASE_GROUPS:
        assignment = runner.build_local_partition_assignment(
            args.suite, args.partition_id, args.total_partitions
        )
        case_files = _case_files(args.suite, server_dir)
        missing = [cid for cid in assignment.case_ids if cid not in case_files]
        if missing:
            print(f"::warning::Cases absent from the {args.suite} config: {missing}")

        targets = []
        for file_path in sorted(
            {case_files[cid] for cid in assignment.case_ids if cid in case_files}
        ):
            case_ids = [
                cid for cid in assignment.case_ids if case_files.get(cid) == file_path
            ]
            targets.extend(_node_targets(file_path, case_ids))
        for standalone_rel in assignment.standalone_files:
            targets.append(str(server_dir / standalone_rel))
    elif args.suite in runner.FILE_SUITES:
        files = [str(server_dir / rel) for rel in runner.FILE_SUITES[args.suite]]
        targets = partition_items_by_index(
            files, args.partition_id, args.total_partitions
        )
    else:
        parser.error(f"Unknown diffusion suite: {args.suite!r}")

    unique_targets = []
    seen = set()
    for target in targets:
        if target not in seen:
            seen.add(target)
            unique_targets.append(target)

    with open(args.output, "w", encoding="utf-8") as f:
        for target in unique_targets:
            f.write(target + "\n")

    print(f"Listed {len(unique_targets)} test target(s) to {args.output}:")
    for target in unique_targets:
        print(f"  - {target}")


if __name__ == "__main__":
    main()
