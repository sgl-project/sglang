#!/usr/bin/env python3
"""Join a mixed-phase and a reference-phase lifecycle probe by checkpoint depth.

For every depth P claimed in the mixed phase: was the claimed slot tracked in a
forward that also carried one-token decode tails (a MIXED forward), what did
the donated slot contain (fingerprint), and does it equal the reference
phase's checkpoint at the same depth? Then: did R / P2 restore the same
content as their reference counterparts, and were the answers right?

The result carries a ``status``: ``complete`` only when the mixed phase
claimed at least one depth, every claimed depth has a donation with a valid
fingerprint on both sides, and both R and P2 restored on both sides with
valid fingerprints. Anything else is ``incomplete`` (with the reasons listed)
and never yields an equality verdict. A valid fingerprint is exactly five
finite numbers. Exit status: 0 complete, 2 incomplete or invalid.
"""

import argparse
import json
import math
import sys

FINGERPRINT_LEN = 5


def valid_fingerprint(fp) -> bool:
    if not isinstance(fp, (list, tuple)) or len(fp) != FINGERPRINT_LEN:
        return False
    for x in fp:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return False
        if not math.isfinite(x):
            return False
    return True


def fp_close(a, b, rel=1e-6) -> bool:
    return all(abs(x - y) <= rel * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


def donations_by_depth(chain):
    """Depth -> the donation that actually entered the tree (mamba_exist is
    False, or unknown when the tree impl reports no result)."""
    out = {}
    for d in chain.get("donations", []):
        if d.get("mamba_exist") is True or d.get("depth") is None:
            continue
        out[d["depth"]] = d
    return out


def compare(mixed: dict, ref: dict) -> dict:
    reasons = []
    pm = mixed.get("chains", {}).get("P-mixed")
    pr = ref.get("chains", {}).get("P-ref")
    if pm is None or pr is None:
        return {"status": "incomplete", "reasons": ["missing P chain in one phase"]}

    claimed = sorted({c["claim"]["depth"] for c in pm.get("claims", [])})
    if not claimed:
        reasons.append("mixed phase claimed no checkpoint depth")
    dm, dr = donations_by_depth(pm), donations_by_depth(pr)

    rows = []
    for depth in claimed:
        a, b = dm.get(depth), dr.get(depth)
        claim = next(c for c in pm["claims"] if c["claim"]["depth"] == depth)
        row = {
            "depth": depth,
            "mixed_slot": a["slot"] if a else None,
            "mixed_kind": a["kind"] if a else None,
            "mixed_claim_tracked_in_mixed_forward": any(
                f["one_token_rows"] for f in claim["tracked_in_forwards"]
            ),
            "mixed_fp": a["fp"] if a else None,
            "ref_fp": b["fp"] if b else None,
            "mixed_equals_ref": None,
            "mixed_is_all_zero": None,
        }
        if a is None:
            reasons.append(f"depth {depth}: no mixed-phase donation")
        elif not valid_fingerprint(a["fp"]):
            reasons.append(f"depth {depth}: invalid mixed fingerprint {a['fp']!r}")
        else:
            row["mixed_is_all_zero"] = all(x == 0.0 for x in a["fp"])
        if b is None:
            reasons.append(f"depth {depth}: no reference donation")
        elif not valid_fingerprint(b["fp"]):
            reasons.append(f"depth {depth}: invalid reference fingerprint {b['fp']!r}")
        if a and b and valid_fingerprint(a["fp"]) and valid_fingerprint(b["fp"]):
            row["mixed_equals_ref"] = fp_close(a["fp"], b["fp"])
        rows.append(row)
    extra_ref = sorted(set(dr) - set(claimed))
    if extra_ref:
        reasons.append(
            f"reference donated at depths not claimed in the mixed phase: {extra_ref}"
        )

    restores = {}
    for k in ("R", "P2"):
        rm = mixed["chains"].get(f"{k}-mixed", {}).get("restores", [])
        rr = ref["chains"].get(f"{k}-ref", {}).get("restores", [])
        entry = {
            "mixed": rm[-1] if rm else None,
            "ref": rr[-1] if rr else None,
            "same_depth": None,
            "content_equal": None,
        }
        if not rm:
            reasons.append(f"{k}: no mixed-phase restore")
        elif not valid_fingerprint(rm[-1]["fp"]):
            reasons.append(f"{k}: invalid mixed restore fingerprint {rm[-1]['fp']!r}")
        if not rr:
            reasons.append(f"{k}: no reference restore")
        elif not valid_fingerprint(rr[-1]["fp"]):
            reasons.append(
                f"{k}: invalid reference restore fingerprint {rr[-1]['fp']!r}"
            )
        if (
            rm
            and rr
            and valid_fingerprint(rm[-1]["fp"])
            and valid_fingerprint(rr[-1]["fp"])
        ):
            entry["same_depth"] = rm[-1]["matched"] == rr[-1]["matched"]
            entry["content_equal"] = fp_close(rm[-1]["fp"], rr[-1]["fp"])
        restores[k] = entry

    status = "complete" if not reasons else "incomplete"
    result = {
        "status": status,
        "reasons": reasons,
        "mixed_tag": mixed.get("tag"),
        "ref_tag": ref.get("tag"),
        "P_mixed_batches": pm.get("mixed_batches"),
        "P_mixed_mask_rows": pm.get("mixed_mask_for_rid"),
        "claimed_depths": claimed,
        "depths": rows,
        "restores": restores,
        "answers_mixed": {
            k: (v["text"][:24], v["correct"], v.get("cached_tokens"))
            for k, v in mixed.get("answers", {}).items()
        },
        "answers_ref": {
            k: (v["text"][:24], v["correct"], v.get("cached_tokens"))
            for k, v in ref.get("answers", {}).items()
        },
    }
    if status == "complete":
        depths_equal = all(r["mixed_equals_ref"] for r in rows)
        restores_equal = all(
            v["same_depth"] and v["content_equal"] for v in restores.values()
        )
        result["verdict"] = "equal" if depths_equal and restores_equal else "mismatch"
        result["all_mixed_depths_equal_ref"] = depths_equal
        result["all_restores_equal_ref"] = restores_equal
    else:
        result["verdict"] = None
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mixed", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    with open(args.mixed) as f:
        mixed = json.load(f)
    with open(args.ref) as f:
        ref = json.load(f)
    result = compare(mixed, ref)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=1, default=str))
    if result["status"] != "complete":
        print("INCOMPLETE: " + "; ".join(result["reasons"]), file=sys.stderr)
        return 2
    print(f"VERDICT: {result['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
