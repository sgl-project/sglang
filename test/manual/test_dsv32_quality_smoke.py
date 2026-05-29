"""Lightweight quality smoke for Double Sparsity on DeepSeek-V3.2 (AC-Q).

Compares DS-on vs the DSA baseline on 20 deterministic prompts (+5 NIAH-mini)
with four gates:

1. ``prefix_match_rate >= 0.80`` — DS output matches DSA's first 32 chars.
2. ``mean_rouge_l >= 0.85`` — mean ROUGE-L F across the 20 prompts.
3. ``niah_mini_recall >= 4/5`` — needle-in-haystack mini recall on DS.
4. ``first_8_tokens_divergence == 0`` — no prompt whose first 8 tokens are
   entirely different between DS and DSA.

Two TP=8 servers cannot co-reside on one 8-GPU node (plan DEC-2), so the
primary workflow is **single-node sequential**: capture the DSA references
first, shut DSA down, boot DS, then compare. Use the CLI:

    # 1. with ONLY the DSA server up:
    python test/manual/test_dsv32_quality_smoke.py capture \
        --dsa-url http://127.0.0.1:30030 \
        --out runs/20260528_dsv32_mvp/dsa_quality_refs.json

    # 2. shut DSA down, boot DS, then with ONLY the DS server up:
    python test/manual/test_dsv32_quality_smoke.py compare \
        --ds-url http://127.0.0.1:30030 \
        --refs runs/20260528_dsv32_mvp/dsa_quality_refs.json \
        --out runs/20260528_dsv32_mvp/dsv32_quality_smoke.json
    # exit 0 iff all four gates pass.

A legacy simultaneous mode is kept for environments that CAN run both
servers at once (e.g. two nodes): set ``DS_BASE_URL`` and ``DSA_BASE_URL``
and run the unittest. If either is unset, the unittest skips cleanly.

The shared prompt fixtures, generation path, and gate math live in
``_dsv32_quality_smoke_lib.py`` so the CPU regression
(``test/registered/unit/manual/test_dsv32_quality_smoke_sequential.py``)
exercises the exact same ``compute_gates`` code under CI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys
import time
import unittest
from typing import Optional


# ----- Load the shared library by path (test/manual is not a package) ---

_LIB_PATH = pathlib.Path(__file__).resolve().parent / "_dsv32_quality_smoke_lib.py"


def _load_lib():
    spec = importlib.util.spec_from_file_location(
        "_dsv32_quality_smoke_lib", str(_LIB_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dsv32_quality_smoke_lib"] = mod
    spec.loader.exec_module(mod)
    return mod


_lib = _load_lib()


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


# ----- Legacy simultaneous-server unittest -----------------------------


@unittest.skipUnless(
    _env("DS_BASE_URL") and _env("DSA_BASE_URL"),
    "DS_BASE_URL and DSA_BASE_URL must both point at running servers "
    "(simultaneous mode; use the capture/compare CLI for single-node).",
)
class TestDSv32QualitySmoke(unittest.TestCase):
    """AC-Q quality smoke, simultaneous-servers mode. Manual hardware test."""

    @classmethod
    def setUpClass(cls):
        cls.ds_url = _env("DS_BASE_URL")
        cls.dsa_url = _env("DSA_BASE_URL")

    def test_quality_smoke(self):
        """Run all four AC-Q gates against both live servers."""
        smoke_pairs = []
        for p in _lib.SMOKE_PROMPTS:
            dsa = _lib.generate(self.dsa_url, p, max_new_tokens=_lib.SMOKE_MAX_NEW_TOKENS)
            ds = _lib.generate(self.ds_url, p, max_new_tokens=_lib.SMOKE_MAX_NEW_TOKENS)
            smoke_pairs.append((p, dsa, ds))
        niah_pairs = []
        for p, needle in _lib.NIAH_MINI_PROMPTS:
            ds = _lib.generate(self.ds_url, p, max_new_tokens=_lib.NIAH_MAX_NEW_TOKENS)
            niah_pairs.append((needle, ds))

        result = _lib.compute_gates(smoke_pairs, niah_pairs)
        _record_artifact(result, ds_url=self.ds_url, dsa_url=self.dsa_url)

        g = result["gates"]
        self.assertTrue(g["prefix_match_rate"]["pass"], g["prefix_match_rate"])
        self.assertTrue(g["mean_rouge_l"]["pass"], g["mean_rouge_l"])
        self.assertTrue(g["niah_mini_recall"]["pass"], g["niah_mini_recall"])
        self.assertTrue(g["first_8_tokens_divergence"]["pass"], g["first_8_tokens_divergence"])


def _record_artifact(result: dict, *, ds_url: str, dsa_url: str) -> None:
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "development", "results")
    )
    try:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = os.path.join(out_dir, f"dsv32_quality_smoke_{ts}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    except OSError:
        pass


# ----- Single-node sequential CLI (capture / compare) ------------------


def _cmd_capture(args: argparse.Namespace) -> int:
    refs = _lib.capture_reference_outputs(args.dsa_url)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(refs, fh, indent=2)
    print(
        f"captured {len(refs['smoke'])} smoke + {len(refs['niah'])} NIAH DSA "
        f"references (dsa_commit={refs.get('dsa_commit_sha')}) -> {args.out}"
    )
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    with open(args.refs, "r", encoding="utf-8") as fh:
        refs = json.load(fh)
    result = _lib.evaluate_against_references(args.ds_url, refs)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"AC-Q quality smoke -> {args.out}")
    for name, g in result["gates"].items():
        verdict = "PASS" if g["pass"] else "FAIL"
        detail = g.get("value", f"{g.get('hits')}/{g.get('total')}")
        print(f"  [{verdict}] {name}: {detail} (threshold {g['threshold']})")
    overall = "PASS" if result["all_pass"] else "FAIL"
    print(f"AC-Q overall: {overall}")
    return 0 if result["all_pass"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="test_dsv32_quality_smoke.py",
        description="AC-Q quality smoke: single-node sequential capture/compare.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cap = sub.add_parser("capture", help="Generate DSA reference outputs (DSA server up).")
    p_cap.add_argument("--dsa-url", required=True)
    p_cap.add_argument("--out", required=True, help="Path to write the reference JSON.")
    p_cap.set_defaults(func=_cmd_capture)

    p_cmp = sub.add_parser("compare", help="Generate DS outputs and score gates (DS server up).")
    p_cmp.add_argument("--ds-url", required=True)
    p_cmp.add_argument("--refs", required=True, help="Reference JSON from `capture`.")
    p_cmp.add_argument("--out", required=True, help="Path to write the gate-result JSON.")
    p_cmp.set_defaults(func=_cmd_compare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    # No subcommand -> behave like the unittest entrypoint for back-compat.
    if len(sys.argv) > 1 and sys.argv[1] in ("capture", "compare"):
        sys.exit(main())
    unittest.main()
