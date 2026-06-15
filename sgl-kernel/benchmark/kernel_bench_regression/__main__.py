"""CLI for the SGLang kernel benchmark regression harness.

Examples
--------
Generate a ground-truth file (run on the nightly GPU):

    python -m kernel_bench_regression generate --out ground_truth.json

Compare a fresh run on a PR against a ground truth, failing on >5% regression:

    python -m kernel_bench_regression compare --gt-file ground_truth.json

    python -m kernel_bench_regression compare \\
        --gt-url https://raw.githubusercontent.com/sgl-project/ci-data/main/kernel-bench/sm90.json

List the registered cases (no GPU required):

    python -m kernel_bench_regression list
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# Allow both ``python -m kernel_bench_regression`` (run from the benchmark dir)
# and package-style execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kernel_bench_regression import compare as compare_mod
    from kernel_bench_regression import registry
else:
    from . import compare as compare_mod
    from . import registry


def _load_ground_truth(args) -> dict:
    if args.gt_file:
        with open(args.gt_file, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.gt_url:
        print(f"Fetching ground truth from {args.gt_url}")
        with urllib.request.urlopen(args.gt_url, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    raise SystemExit("compare requires --gt-file or --gt-url")


def _cmd_list(args) -> int:
    cases = registry.get_cases(args.cases)
    print(f"{len(cases)} kernel benchmark case(s):\n")
    for c in cases:
        cap = ".".join(map(str, c.min_compute_capability))
        direction = "higher=better" if c.higher_is_better else "lower=better"
        tags = ",".join(c.tags)
        print(f"  {c.case_id}  [{tags}]")
        print(f"          {c.component}")
        print(
            f"          file={c.bench_file} provider={c.provider!r} "
            f"metric={c.metric} ({direction}) min_cc={cap}"
        )
    return 0


def _cmd_generate(args) -> int:
    from kernel_bench_regression import runner  # noqa: WPS433 (lazy: needs torch)

    cases = registry.get_cases(args.cases)
    payload = runner.generate(
        cases, repeat=args.repeat, tolerance=args.tolerance, commit=args.commit
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    n_meas = sum(len(c["measurements"]) for c in payload["cases"].values())
    print(
        f"\nWrote {len(payload['cases'])} case(s), {n_meas} measurement(s) to {out_path}"
    )
    return 0


def _cmd_compare(args) -> int:
    from kernel_bench_regression import runner  # noqa: WPS433 (lazy: needs torch)

    ground_truth = _load_ground_truth(args)
    tolerance = (
        args.tolerance
        if args.tolerance is not None
        else ground_truth.get("tolerance", 0.05)
    )
    cases = registry.get_cases(args.cases)
    measured = runner.generate(
        cases, repeat=args.repeat, tolerance=tolerance, commit=args.commit
    )

    if args.measured_out:
        with open(args.measured_out, "w", encoding="utf-8") as f:
            json.dump(measured, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote measured results to {args.measured_out}")

    report = compare_mod.compare_results(ground_truth, measured, tolerance)
    print("\n" + compare_mod.format_report(report, tolerance))

    if not report.passed and not args.warn_only:
        print(
            "\nThis gate is allowed to be flaky on non-isolated CI GPUs: "
            "if you believe the regression is noise, simply re-run the job.",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="kernel_bench_regression")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common(p):
        p.add_argument(
            "--cases",
            nargs="*",
            default=None,
            help="Subset of case_ids to run (default: all).",
        )
        p.add_argument(
            "--repeat",
            type=int,
            default=3,
            help="Best-of-N repeats per config to suppress noise (default: 3).",
        )
        p.add_argument(
            "--commit", default="", help="Commit SHA to stamp into the results meta."
        )
        p.add_argument(
            "--no-ci-config",
            action="store_true",
            help="Do NOT force SGLANG_IS_IN_CI=true (use each benchmark's full "
            "config set). Ground truth and PR runs must use the same setting.",
        )

    p_list = sub.add_parser("list", help="List registered cases (no GPU needed).")
    p_list.add_argument("--cases", nargs="*", default=None)

    p_gen = sub.add_parser("generate", help="Run benchmarks and write a results JSON.")
    _add_common(p_gen)
    p_gen.add_argument("--out", required=True, help="Output JSON path.")
    p_gen.add_argument("--tolerance", type=float, default=0.05)

    p_cmp = sub.add_parser(
        "compare", help="Run benchmarks and compare to ground truth."
    )
    _add_common(p_cmp)
    p_cmp.add_argument("--gt-file", default=None, help="Local ground-truth JSON path.")
    p_cmp.add_argument("--gt-url", default=None, help="Remote ground-truth JSON URL.")
    p_cmp.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Relative tolerance (default: value stored in the ground truth, or 0.05).",
    )
    p_cmp.add_argument(
        "--measured-out", default=None, help="Optional path to dump the fresh run."
    )
    p_cmp.add_argument(
        "--warn-only",
        action="store_true",
        help="Report regressions but always exit 0.",
    )

    args = parser.parse_args()

    if args.command == "list":
        return _cmd_list(args)

    # Force the benchmarks' CI config path unless explicitly disabled, so that the
    # ground truth and the PR run measure identical config keys.
    if not getattr(args, "no_ci_config", False):
        os.environ.setdefault("SGLANG_IS_IN_CI", "true")

    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "compare":
        return _cmd_compare(args)
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
