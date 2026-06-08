"""Offline DS-vs-DSA accuracy gate (MMLU + NIAH) — sequential collect, then compare.

GLM-5.1 cannot run two TP=8 servers at once on 8 GPUs (weights > 140 GB/rank), and
it cannot run at TP=4 (weights ~2x exceed a single H200), so the paired in-process
harness (`test/manual/test_double_sparsity_v32.py`, which needs DS_BASE_URL AND
DSA_BASE_URL live together) cannot drive the DS-vs-DSA accuracy comparison. This
module provides the sequential path:

    AC12_MODE=collect AC12_SIDE=dsa AC12_BASE_URL=http://127.0.0.1:30000 \
        python development/loop8/accuracy_gate.py        # boot DSA-native, score, write artifact
    # shut DSA down, boot DS with the 256 mask, then:
    AC12_MODE=collect AC12_SIDE=ds  AC12_BASE_URL=http://127.0.0.1:30000 \
        python development/loop8/accuracy_gate.py
    AC12_MODE=compare AC12_DSA_ARTIFACT=... AC12_DS_ARTIFACT=... \
        python development/loop8/accuracy_gate.py        # offline, no server

`collect` reuses the TUNED scoring from `test_double_sparsity_v32.py` (the
`_parse_mmlu_letter` two-tier parser, the deterministic `_make_niah_prompt` /
`_niah_needle` / `_niah_recall_hits`, and `_make_mmlu_5shot_prompt` /
`_load_mmlu_examples`) so a side's score is identical to the paired harness.
`compare` is pure (operates on the two artifact dicts), validates that both sides
used the SAME prompt set (run_id + per-test prompt hashes + index_topk + length set),
and FAILS CLOSED on any mismatch or missing field. Thresholds mirror the paired
gate: MMLU DS within 1.0 pp of DSA (mandatory); within-budget NIAH DS within 5.0 pp
of DSA (mandatory); beyond-budget NIAH is characterization-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "ac12_accuracy_side_v1"
MMLU_TOLERANCE_PP = 1.0          # DS MMLU must be within 1.0 pp of DSA
NIAH_WITHIN_BUDGET_TOLERANCE_PP = 5.0  # within-budget DS NIAH recall within 5.0 pp


# ----- shared helpers ----------------------------------------------------


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def prompt_set_run_id(
    *,
    index_topk: int,
    niah_lengths: List[int],
    niah_num_prompts: int,
    niah_seed: int,
    mmlu_subjects: List[str],
    mmlu_example_ids: List[str],
) -> str:
    """Deterministic id of the exact prompt set both sides must share.

    Two collect runs that produce different run_ids used different prompts and
    MUST NOT be compared — `compare` fails closed on a run_id mismatch.
    """
    payload = json.dumps(
        {
            "index_topk": int(index_topk),
            "niah_lengths": [int(x) for x in niah_lengths],
            "niah_num_prompts": int(niah_num_prompts),
            "niah_seed": int(niah_seed),
            "mmlu_subjects": sorted(mmlu_subjects),
            "mmlu_example_ids": list(mmlu_example_ids),
        },
        sort_keys=True,
    )
    return _sha(payload)


# ----- compare (pure; offline; fail-closed) ------------------------------


class GateError(ValueError):
    """Raised when the two artifacts are incomparable or a mandatory gate fails."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise GateError(msg)


def _pct(hits: int, total: int) -> float:
    return 100.0 * hits / total if total else 0.0


def compare(dsa: Dict[str, Any], ds: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a DSA-native and a DS accuracy artifact. Fail closed.

    Validates the two sides are the SAME prompt set (schema, run_id, index_topk,
    NIAH length set + per-length prompt hashes, MMLU example set), then applies
    the mandatory thresholds. Returns a verdict dict; raises GateError on an
    incomparable pair or a mandatory-gate failure with no usable data.
    """
    for name, art in (("dsa", dsa), ("ds", ds)):
        _require(isinstance(art, dict), f"{name} artifact is not a dict")
        _require(art.get("schema") == SCHEMA, f"{name} artifact schema != {SCHEMA!r}")
    _require(dsa.get("side") == "dsa", "dsa artifact side != 'dsa'")
    _require(ds.get("side") == "ds", "ds artifact side != 'ds'")
    _require(
        dsa.get("run_id") and dsa.get("run_id") == ds.get("run_id"),
        f"run_id mismatch: dsa={dsa.get('run_id')!r} ds={ds.get('run_id')!r} "
        "(the two sides did not score the same prompt set)",
    )
    _require(
        int(dsa.get("index_topk", -1)) == int(ds.get("index_topk", -2)),
        f"index_topk mismatch: dsa={dsa.get('index_topk')} ds={ds.get('index_topk')}",
    )

    # MMLU (mandatory): same example set, DS within tolerance of DSA.
    dmm, smm = dsa.get("mmlu") or {}, ds.get("mmlu") or {}
    _require(
        bool(dmm) and bool(smm), "missing mmlu block in one or both artifacts"
    )
    _require(
        dmm.get("prompt_set_hash") and dmm.get("prompt_set_hash") == smm.get("prompt_set_hash"),
        "MMLU prompt-set hash mismatch (different example set/prompts)",
    )
    _require(int(dmm.get("total", 0)) > 0 and int(smm.get("total", 0)) > 0,
             "MMLU total is zero in one or both sides (fail closed)")
    dsa_mmlu = _pct(int(dmm["hits"]), int(dmm["total"]))
    ds_mmlu = _pct(int(smm["hits"]), int(smm["total"]))
    mmlu_delta = dsa_mmlu - ds_mmlu  # positive = DS lost accuracy
    mmlu_pass = abs(mmlu_delta) <= MMLU_TOLERANCE_PP

    # NIAH within-budget (mandatory): per matching length, DS recall within tol.
    d_niah = {int(e["length_words"]): e for e in (dsa.get("niah_within_budget") or [])}
    s_niah = {int(e["length_words"]): e for e in (ds.get("niah_within_budget") or [])}
    _require(
        d_niah and set(d_niah) == set(s_niah),
        f"NIAH within-budget length set mismatch: dsa={sorted(d_niah)} ds={sorted(s_niah)}",
    )
    niah_rows: List[Dict[str, Any]] = []
    niah_pass = True
    for L in sorted(d_niah):
        de, se = d_niah[L], s_niah[L]
        _require(
            de.get("prompt_set_hash") and de.get("prompt_set_hash") == se.get("prompt_set_hash"),
            f"NIAH prompt-set hash mismatch at length {L}",
        )
        _require(int(de.get("num_prompts", 0)) > 0,
                 f"NIAH num_prompts zero at length {L} (fail closed)")
        dr, sr = _pct(int(de["hits"]), int(de["num_prompts"])), _pct(int(se["hits"]), int(se["num_prompts"]))
        delta = dr - sr
        ok = abs(delta) <= NIAH_WITHIN_BUDGET_TOLERANCE_PP
        niah_pass = niah_pass and ok
        niah_rows.append({"length_words": L, "dsa_recall_pct": round(dr, 2),
                          "ds_recall_pct": round(sr, 2), "delta_pp": round(delta, 2),
                          "within_tolerance": ok})

    # Beyond-budget NIAH: characterization only (recorded, never a pass/fail).
    d_beyond = {int(e["length_words"]): e for e in (dsa.get("niah_beyond_budget") or [])}
    s_beyond = {int(e["length_words"]): e for e in (ds.get("niah_beyond_budget") or [])}
    beyond_rows = []
    for L in sorted(set(d_beyond) & set(s_beyond)):
        de, se = d_beyond[L], s_beyond[L]
        dr = _pct(int(de.get("hits", 0)), max(int(de.get("num_prompts", 0)), 1))
        sr = _pct(int(se.get("hits", 0)), max(int(se.get("num_prompts", 0)), 1))
        beyond_rows.append({"length_words": L, "dsa_recall_pct": round(dr, 2),
                            "ds_recall_pct": round(sr, 2), "delta_pp": round(dr - sr, 2)})

    return {
        "run_id": dsa["run_id"],
        "index_topk": int(dsa["index_topk"]),
        "mmlu": {"dsa_pct": round(dsa_mmlu, 2), "ds_pct": round(ds_mmlu, 2),
                 "delta_pp": round(mmlu_delta, 2), "tolerance_pp": MMLU_TOLERANCE_PP,
                 "pass": mmlu_pass},
        "niah_within_budget": {"tolerance_pp": NIAH_WITHIN_BUDGET_TOLERANCE_PP,
                               "rows": niah_rows, "pass": niah_pass},
        "niah_beyond_budget_characterization": beyond_rows,
        # MMLU within-tol is mandatory (DEC-2); within-budget NIAH is mandatory;
        # beyond-budget is characterization-only and never gates.
        "mandatory_pass": bool(mmlu_pass and niah_pass),
    }


# ----- collect (reuses the tuned harness scoring; needs one live server) --


def _load_harness():
    """Import test/manual/test_double_sparsity_v32.py by path (not a package)."""
    import importlib.util

    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "test", "manual",
                     "test_double_sparsity_v32.py")
    )
    spec = importlib.util.spec_from_file_location("_ac12_harness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ac12_harness"] = mod  # before exec (dataclass contract)
    spec.loader.exec_module(mod)
    return mod


def collect(base_url: str, side: str, *, out_path: Optional[str] = None) -> Dict[str, Any]:
    """Score one live server (DSA-native or DS) for MMLU + NIAH, write a per-side artifact.

    Reuses the harness's deterministic prompt-gen + tuned parser + recall scorer so
    the side's score matches the paired gate. Fail-closed: raises if the artifact
    cannot be written.
    """
    if side not in ("dsa", "ds"):
        raise GateError(f"AC12_SIDE must be 'dsa' or 'ds', got {side!r}")
    H = _load_harness()
    index_topk = H._env_int("AC12_INDEX_TOPK", 2048)
    niah_num = H._env_int("AC12_NIAH_NUM_PROMPTS", 20)
    niah_max_new = H._env_int("AC12_NIAH_MAX_NEW_TOKENS", 64)
    niah_seed = H._env_int("AC12_NIAH_SEED", 1234)
    # Within-budget = the tokenized prompt fits the DS selection budget (<= index_topk).
    # Use word-length bins comfortably under index_topk for within-budget.
    within_lengths = [int(x) for x in os.environ.get(
        "AC12_NIAH_WITHIN_LENGTHS", "256,512,1024").split(",")]

    # ---- NIAH within-budget ----
    niah_within: List[Dict[str, Any]] = []
    for L in within_lengths:
        needles = [H._niah_needle(L, i) for i in range(niah_num)]
        prompts = [H._make_niah_prompt(L, seed=niah_seed + i, needle=needles[i])
                   for i in range(niah_num)]
        attempts = [H._generate_attempt(base_url, p, max_new_tokens=niah_max_new, use_chat=True)
                    for p in prompts]
        responses = [a.text for a in attempts]
        served, first_err = H._summarize_attempts(attempts)
        hits = H._niah_recall_hits(needles, responses)
        niah_within.append({
            "length_words": L, "num_prompts": niah_num, "served": served,
            "hits": hits, "recall_pct": round(_pct(hits, niah_num), 2),
            "prompt_set_hash": _sha("|".join(prompts)), "first_error": first_err,
        })

    # ---- MMLU 5-shot ----
    data_dir = os.environ.get("AC12_MMLU_DATA_DIR", "")
    subjects_env = os.environ.get("AC12_MMLU_SUBJECTS", "all")
    max_examples = H._env_int("AC12_MMLU_NUM_EXAMPLES", 200)
    dev_dir, test_dir = H._ensure_mmlu_data_dir(data_dir)
    subjects = None if subjects_env == "all" else subjects_env.split(",")
    examples, _per_subject_totals = H._load_mmlu_examples(
        dev_dir, test_dir, subjects=subjects, max_examples=max_examples)
    mmlu_hits = 0
    mmlu_prompts: List[str] = []
    example_ids: List[str] = []
    for ex in examples:
        prompt = H._make_mmlu_5shot_prompt(ex["dev"], ex["subject"], ex["row"])
        mmlu_prompts.append(prompt)
        example_ids.append(f"{ex['subject']}:{_sha(str(ex['row']))}")
        att = H._generate_attempt(base_url, prompt, max_new_tokens=4, use_chat=False)
        pred = H._parse_mmlu_letter(att.text.strip()) if att.ok else None
        gold = str(ex["row"][5]).strip().upper()
        if pred is not None and pred == gold:
            mmlu_hits += 1
    run_id = prompt_set_run_id(
        index_topk=index_topk, niah_lengths=within_lengths, niah_num_prompts=niah_num,
        niah_seed=niah_seed, mmlu_subjects=(subjects or ["all"]), mmlu_example_ids=example_ids,
    )
    artifact = {
        "schema": SCHEMA, "side": side, "run_id": run_id, "index_topk": index_topk,
        "base_url": base_url,
        "mmlu": {"hits": mmlu_hits, "total": len(examples),
                 "num_examples": max_examples,
                 "prompt_set_hash": _sha("|".join(mmlu_prompts))},
        "niah_within_budget": niah_within,
        "niah_beyond_budget": [],  # populated by a separate beyond-budget run if desired
    }
    out_path = out_path or os.environ.get("AC12_OUTPUT_PATH") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results",
                     f"ac12_accuracy_{side}_{run_id}.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:   # fail-closed: no try/except
        json.dump(artifact, fh, indent=2)
    print(f"wrote {out_path} (run_id={run_id} mmlu={mmlu_hits}/{len(examples)})")
    return artifact


def main(argv: List[str]) -> int:
    mode = os.environ.get("AC12_MODE", "")
    if mode == "collect":
        collect(os.environ["AC12_BASE_URL"], os.environ.get("AC12_SIDE", ""))
        return 0
    if mode == "compare":
        with open(os.environ["AC12_DSA_ARTIFACT"]) as f:
            dsa = json.load(f)
        with open(os.environ["AC12_DS_ARTIFACT"]) as f:
            ds = json.load(f)
        verdict = compare(dsa, ds)
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["mandatory_pass"] else 1
    print("set AC12_MODE=collect (AC12_SIDE, AC12_BASE_URL) or "
          "AC12_MODE=compare (AC12_DSA_ARTIFACT, AC12_DS_ARTIFACT)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
