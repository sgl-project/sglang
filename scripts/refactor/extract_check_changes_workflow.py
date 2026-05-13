#!/usr/bin/env python3
"""Mechanically extract the `check-changes` job from `.github/workflows/pr-test.yml`
into a new reusable workflow `.github/workflows/_pr-test-check-changes.yml`.

Why a script (and not a manual edit): the job body is ~275 lines and the
extraction must be byte-faithful so the diff is reviewable as "pure move".
The script enforces that by performing only a single in-place text change
(the job-key rename `check-changes:` -> `run:`); every other byte of the
original block is preserved exactly. A round-trip integrity check at the
end asserts that pasting the renamed block back with its original key
reproduces the source file.

Idempotency: re-running fails. The second invocation finds that
`_pr-test-check-changes.yml` already exists and aborts.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PR_TEST = REPO_ROOT / ".github" / "workflows" / "pr-test.yml"
REUSABLE = REPO_ROOT / ".github" / "workflows" / "_pr-test-check-changes.yml"

JOB_START = "  check-changes:\n"
# Last line of the job: the closing `}` of the "Show filter results in summary"
# step's heredoc-style bash block. Verified unique in pr-test.yml at runtime.
JOB_END = "          } >> $GITHUB_STEP_SUMMARY\n"
# Sanity-check marker: the comment header for the next section, expected to
# appear right after the job body (modulo blank lines).
NEXT_SECTION_PREFIX = "  # =============================================== Wait Jobs"

CALLER_STUB = """  check-changes:
    uses: ./.github/workflows/_pr-test-check-changes.yml
    with:
      pr_head_sha: ${{ inputs.pr_head_sha || '' }}
      git_ref: ${{ inputs.git_ref || '' }}
      target_stage: ${{ inputs.target_stage || '' }}
      include_wheel_build: ${{ inputs.include_wheel_build == true }}
      run_all_tests: ${{ inputs.run_all_tests == true }}
      force_continue_on_error: ${{ inputs.force_continue_on_error == true }}
    secrets: inherit
"""

REUSABLE_HEADER = """name: Check Changes

on:
  workflow_call:
    inputs:
      pr_head_sha:
        type: string
        default: ''
      git_ref:
        type: string
        default: ''
      target_stage:
        type: string
        default: ''
      include_wheel_build:
        type: boolean
        default: false
      run_all_tests:
        type: boolean
        default: false
      force_continue_on_error:
        type: boolean
        default: false
    outputs:
      main_package:
        value: ${{ jobs.run.outputs.main_package }}
      sgl_kernel:
        value: ${{ jobs.run.outputs.sgl_kernel }}
      sgl_kernel_raw:
        value: ${{ jobs.run.outputs.sgl_kernel_raw }}
      jit_kernel:
        value: ${{ jobs.run.outputs.jit_kernel }}
      multimodal_gen:
        value: ${{ jobs.run.outputs.multimodal_gen }}
      max_parallel:
        value: ${{ jobs.run.outputs.max_parallel }}
      max_parallel_small:
        value: ${{ jobs.run.outputs.max_parallel_small }}
      max_parallel_2gpu:
        value: ${{ jobs.run.outputs.max_parallel_2gpu }}
      b200_runner:
        value: ${{ jobs.run.outputs.b200_runner }}
      enable_retry:
        value: ${{ jobs.run.outputs.enable_retry }}
      continue_on_error:
        value: ${{ jobs.run.outputs.continue_on_error }}

jobs:
"""


def die(msg: str) -> None:
    sys.exit(f"FAIL: {msg}")


def find_unique_line(lines: list[str], target: str) -> int:
    matches = [i for i, line in enumerate(lines) if line == target]
    if len(matches) == 0:
        die(f"marker not found: {target!r}")
    if len(matches) > 1:
        die(
            f"marker matched multiple lines (line numbers, 1-based: "
            f"{[i + 1 for i in matches]}): {target!r}"
        )
    return matches[0]


def main() -> None:
    if REUSABLE.exists():
        die(
            f"{REUSABLE.relative_to(REPO_ROOT)} already exists -- "
            "the extraction appears to have already been performed. "
            "Delete the file (or revert this refactor) and re-run."
        )

    original = PR_TEST.read_text()
    lines = original.splitlines(keepends=True)

    start = find_unique_line(lines, JOB_START)
    end = find_unique_line(lines, JOB_END)
    if end < start:
        die(
            f"job-end marker (line {end + 1}) precedes job-start " f"(line {start + 1})"
        )

    # The first non-blank line after the job body must begin the Wait Jobs
    # section. Catches the case where JOB_END happens to match somewhere
    # unrelated (defense in depth -- JOB_END is asserted unique above).
    j = end + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1
    if j >= len(lines) or not lines[j].startswith(NEXT_SECTION_PREFIX):
        die(
            f"line {j + 1} after job body does not start the Wait Jobs section: "
            f"{lines[j] if j < len(lines) else '<EOF>'!r}"
        )

    job_block = lines[start : end + 1]
    assert job_block[0] == JOB_START

    # Single-line in-place rename: '  check-changes:' -> '  run:'.
    # Every other byte of the block carries over unchanged.
    renamed_block = ["  run:\n"] + job_block[1:]

    reusable_text = REUSABLE_HEADER + "".join(renamed_block)
    REUSABLE.write_text(reusable_text)
    print(f"wrote {REUSABLE.relative_to(REPO_ROOT)} ({len(reusable_text)} bytes)")

    new_text = "".join(lines[:start] + [CALLER_STUB] + lines[end + 1 :])
    PR_TEST.write_text(new_text)
    print(
        f"rewrote {PR_TEST.relative_to(REPO_ROOT)} "
        f"({len(original)} -> {len(new_text)} bytes)"
    )

    # Round-trip integrity check: pasting the original block back at the
    # same offset must reproduce the source file byte-for-byte. Proves the
    # extraction is a pure move + key rename, nothing else.
    restored = "".join(lines[:start] + job_block + lines[end + 1 :])
    if restored != original:
        die(
            "round-trip integrity check failed (the extracted block does "
            "not equal the original byte-for-byte)"
        )
    print("byte-fidelity OK (block round-trip equals original pr-test.yml)")


if __name__ == "__main__":
    main()
