"""Build a named-row override for A/B runs via SGLANG_LORA_MOE_CONFIG_DIR."""

from __future__ import annotations

import argparse
import copy
import json
import os

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
