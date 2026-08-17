"""Build a one-file override dir that forces one table change, for A/B runs.

Point ``SGLANG_LORA_MOE_CONFIG_DIR`` at the output: lookup falls back per
file, so the directory holds ONLY the file being varied and everything else
still comes from the package.  Rows are matched by NAME, never by position —
inserting a rule must not silently retarget anyone's tooling.

Three variant kinds, one per invocation:

  # Force one tile-rule's sites onto a whole row (kills its ladder):
  python make_config_variant.py tiles --arch gb300 --out /tmp/v \
      --row decode.per_expert --from-rule 'max_tokens=16'

  # Drop a row's tile rules entirely (serve the built-in heuristics):
  python make_config_variant.py tiles --arch gb300 --out /tmp/v \
      --row decode.per_expert --drop

  # Swap one plan row's base-GEMM row order (the vendor is a server flag,
  # --moe-lora-base-gemm, not a table value):
  python make_config_variant.py plans --arch gb300 --out /tmp/v \
      --row prefill.serial --base-gemm-rows expert_major
"""

from __future__ import annotations

import argparse
import copy
import json
import os

# repo root = three levels up from benchmark/kernels/lora_moe/
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_CONFIG_DIR = os.path.join(
    _REPO_ROOT, "python", "sglang", "srt", "lora", "moe", "configs"
)


def _load(arch: str, kind: str) -> dict:
    with open(os.path.join(_CONFIG_DIR, f"{arch}.{kind}.json")) as handle:
        return json.load(handle)


def _match_rule(rules: list[dict], spec: str) -> dict:
    """Find exactly one rule by predicate, e.g. 'max_tokens=16' or 'wildcard'."""
    if spec == "wildcard":
        want = {}
    else:
        want = {
            key: int(value)
            for key, value in (part.split("=") for part in spec.split(","))
        }
    hits = [
        rule
        for rule in rules
        if {k: v for k, v in rule.items() if k != "sites"} == want
    ]
    if len(hits) != 1:
        preds = [{k: v for k, v in r.items() if k != "sites"} for r in rules]
        raise SystemExit(f"predicate {want} matched {len(hits)} of {preds}")
    return hits[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("tiles", "plans"))
    parser.add_argument("--arch", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--row", required=True, help="plan row name, exact")
    parser.add_argument("--from-rule", help="tiles: predicate of the rule to force")
    parser.add_argument("--drop", action="store_true", help="tiles: delete the row")
    parser.add_argument(
        "--base-gemm-rows",
        choices=("expert_major", "route_major"),
        help="plans: new base-GEMM row order for the row",
    )
    args = parser.parse_args()

    table = copy.deepcopy(_load(args.arch, args.kind))
    if args.kind == "tiles":
        if args.drop == bool(args.from_rule):
            raise SystemExit("tiles: pass exactly one of --from-rule / --drop")
        if args.row not in table["rules"]:
            raise SystemExit(f"no tile rules for row {args.row!r}")
        if args.drop:
            del table["rules"][args.row]
            tag = "builtin"
        else:
            rule = _match_rule(table["rules"][args.row], args.from_rule)
            table["rules"][args.row] = [{"sites": copy.deepcopy(rule["sites"])}]
            tag = args.from_rule.replace("=", "").replace(",", "_")
    else:
        if not args.base_gemm_rows:
            raise SystemExit("plans: --base-gemm-rows is required")
        rows = [
            row
            for row in table["scenarios"] + table.get("fallback", [])
            if row["name"] == args.row
        ]
        if not rows:
            raise SystemExit(f"no plan row named {args.row!r}")
        before = {row["base_gemm_rows"] for row in rows}
        for row in rows:
            row["base_gemm_rows"] = args.base_gemm_rows
        tag = args.base_gemm_rows
        print(f"{args.row}: {'/'.join(sorted(before))} -> {args.base_gemm_rows}")

    out = os.path.join(args.out, f"{args.row}__{tag}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{args.arch}.{args.kind}.json")
    with open(path, "w") as handle:
        json.dump(table, handle, indent=1)
        handle.write("\n")
    print(f"wrote {path}\n  SGLANG_LORA_MOE_CONFIG_DIR={out}")


if __name__ == "__main__":
    main()
