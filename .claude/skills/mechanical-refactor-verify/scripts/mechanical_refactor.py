"""Unified entry point for the mechanical-refactor toolchain: split / construct / verify.

Pure dispatch — each subcommand delegates to the existing module unchanged:

- ``split``     -> mechanical_refactor_proof_generator (single-commit mode)
- ``construct`` -> mechanical_refactor_proof_generator (range mode)
- ``verify``    -> mechanical_refactor_reproduction_cli (whole-chain verifier)
"""

import argparse
import os
import sys

import mechanical_refactor_proof_generator as _generator
import mechanical_refactor_reproduction_cli as _chain_cli


def main(argv: "list[str]") -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "split":
        return _run_split(args)
    if args.command == "construct":
        return _run_construct(args)
    return _run_verify(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mechanical_refactor.py",
        description=(
            "Mechanical-refactor toolchain. split: analyze one commit's provable "
            "relocation content. construct: generate the chain's proof folder. "
            "verify: run EVERY proof in the chain (never a sample)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser(
        "split",
        help=(
            "analyze one commit: print the inferred relocation script and run it; "
            "PASS means the commit is a pure relocation, RESIDUAL/UNSUPPORTED "
            "shows what must be split out (guide-split.md)"
        ),
    )
    split.add_argument("commit", help="the commit to analyze")
    split.add_argument(
        "--repo-root", default=None, help="repo root (default: cwd's repo)"
    )

    construct = subparsers.add_parser(
        "construct",
        help=(
            "generate the proof folder for a chain: one auditable script per "
            "matching commit, plus verdicts (guide-construct-proof.md)"
        ),
    )
    construct.add_argument(
        "rev_range", metavar="<base>..<tip>", help="the chain's commit range"
    )
    construct.add_argument(
        "--match",
        default=None,
        help="regex a commit subject must match to get a proof "
        "(e.g. '(?<!_)mechanical_provable'); default: every commit in the range",
    )
    construct.add_argument("--out", required=True, help="output folder for the proof")
    construct.add_argument(
        "--repo-root", default=None, help="repo root (default: cwd's repo)"
    )

    verify = subparsers.add_parser(
        "verify",
        help=(
            "verify the whole chain: classify every commit and run every provable "
            "commit's proof — all of them, never a sample (guide-verify-proof.md)"
        ),
    )
    verify.add_argument("--base", required=True, help="base commit of the chain")
    verify.add_argument("--branch", required=True, help="PR branch name (chain tip)")
    verify.add_argument("--proof", required=True, help="proof folder path")
    verify.add_argument("--repo-root", default=None, help="repo root (default: cwd's)")
    verify.add_argument(
        "--report", default=None, help="report file path (default: inside --proof)"
    )
    verify.add_argument(
        "--jobs", type=int, default=None, help="max concurrent proof runs"
    )
    verify.add_argument(
        "--skip-passed",
        action="store_true",
        help="reuse this machine's earlier PASS verdicts for unchanged proofs",
    )

    return parser


def _run_split(args: argparse.Namespace) -> int:
    _maybe_chdir(args.repo_root)
    return _generator._main([args.commit])


def _run_construct(args: argparse.Namespace) -> int:
    _maybe_chdir(args.repo_root)
    generator_argv = [args.rev_range, "--out", args.out]
    if args.match is not None:
        generator_argv += ["--match", args.match]
    return _generator._main(generator_argv)


def _run_verify(args: argparse.Namespace) -> int:
    chain_argv = [
        "--base",
        args.base,
        "--branch",
        args.branch,
        "--proof",
        args.proof,
    ]
    if args.repo_root is not None:
        chain_argv += ["--repo-root", args.repo_root]
    if args.report is not None:
        chain_argv += ["--report", args.report]
    if args.jobs is not None:
        chain_argv += ["--jobs", str(args.jobs)]
    if args.skip_passed:
        chain_argv += ["--skip-passed"]
    return _chain_cli.main(chain_argv)


def _maybe_chdir(repo_root: "str | None") -> None:
    if repo_root is not None:
        os.chdir(repo_root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
