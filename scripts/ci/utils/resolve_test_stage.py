#!/usr/bin/env python3
"""Print the CUDA CI stage(s) a registered test file declares, one per line.

Used by rerun-test.yml so a manual /rerun-test resolves a file's stage and
applies the same per-stage env as the gated PR test (e.g. base-a runs with
async assert off). Pure stdlib (no GitHub / network deps) so it runs in the
rerun-test container, and AST-based so it only reads `register_cuda_ci(...)`
calls (ignoring `register_amd_ci` / `register_cpu_ci`) and tolerates
multi-line registrations.

Usage: resolve_test_stage.py <test_file>
"""

import ast
import sys


def cuda_stages(path: str) -> list:
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    stages = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "register_cuda_ci":
            continue
        for kw in node.keywords:
            if kw.arg == "stage" and isinstance(kw.value, ast.Constant):
                stages.append(kw.value.value)
    return stages


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: resolve_test_stage.py <test_file>")
    for stage in cuda_stages(sys.argv[1]):
        print(stage)
