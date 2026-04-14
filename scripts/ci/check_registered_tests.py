#!/usr/bin/env python3
"""
Pre-commit hook: validate that all Python test files under test/registered/
contain a CI registry call (register_cuda_ci, register_amd_ci, etc.).

Reuses ut_parse_one_file() from ci_register.py (AST-based parsing)
to match the same logic used by run_suite.py's collect_tests().
"""

import glob
import importlib.util
import os
import sys


def main() -> int:
    # Import ci_register directly to avoid pulling in all of sglang
    spec = importlib.util.spec_from_file_location(
        "ci_register",
        os.path.join("python", "sglang", "test", "ci", "ci_register.py"),
    )
    ci_register = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ci_register)

    # Same filter as run_suite.py: skip conftest.py and __init__.py
    files = sorted(
        f
        for f in glob.glob("test/registered/**/*.py", recursive=True)
        if not f.endswith("/conftest.py") and not f.endswith("/__init__.py")
    )
    if not files:
        return 0

    errors = []
    for f in files:
        try:
            registries = ci_register.ut_parse_one_file(f)
            if len(registries) == 0:
                errors.append(f)
        except Exception:
            # Skip files that can't be parsed (syntax errors, etc.)
            pass

    if errors:
        print("ERROR: Files in test/registered/ missing CI registry call:")
        print("  Move manual-only tests to test/manual/.\n")
        for f in errors:
            print(f"  {f}")
        print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
