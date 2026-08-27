#!/usr/bin/env python3
"""
Pre-commit hook: reject CI-registered tests that live inside the importable
`sglang` package (python/sglang/).

Registered tests and benchmarks must live under test/registered/ (e.g.
test/registered/jit/ for JIT kernel tests and test/registered/jit/benchmark/
for JIT kernel benchmarks) so they are not shipped in the wheel and are
collected by run_suite.py's registered glob. A registered file placed inside
the package would be shipped to users AND silently dropped by run_suite.py
(which no longer globs the package) -- it would never run in CI. This guard
turns that silent skip into a hard failure.

Reuses ut_parse_one_file() from ci_register.py (AST-based) so the registry
detection matches run_suite.py's collect_tests() exactly.
"""

import glob
import importlib.util
import os
import sys

# Markers whose mere presence in the source is worth an AST parse. Anything
# without one of these strings cannot register a test, so we skip parsing it.
_MARKERS = (
    "register_cuda_ci",
    "register_amd_ci",
    "register_cpu_ci",
    "register_npu_ci",
    "register_xpu_ci",
    "register_musa_ci",
)


def main() -> int:
    # Import ci_register directly to avoid pulling in all of sglang.
    spec = importlib.util.spec_from_file_location(
        "ci_register",
        os.path.join("python", "sglang", "test", "ci", "ci_register.py"),
    )
    ci_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_register)

    offenders = []
    for f in sorted(glob.glob("python/sglang/**/*.py", recursive=True)):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                source = fh.read()
        except (OSError, UnicodeDecodeError):
            continue
        if not any(marker in source for marker in _MARKERS):
            continue
        try:
            registries, _has_main_entry = ci_register.ut_parse_one_file(f)
        except Exception:
            # A malformed register call still indicates a misplaced test.
            offenders.append(f)
            continue
        if registries:
            offenders.append(f)

    if offenders:
        print(
            "ERROR: CI-registered test(s)/benchmark(s) found inside the sglang package:"
        )
        print(
            "  Registered tests and benchmarks must live under test/registered/\n"
            "  (e.g. test/registered/jit/ for JIT kernel tests and\n"
            "  test/registered/jit/benchmark/ for JIT kernel benchmarks) so they\n"
            "  are not shipped in the wheel and are collected by run_suite.py.\n"
        )
        for f in offenders:
            print(f"  {f}")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
