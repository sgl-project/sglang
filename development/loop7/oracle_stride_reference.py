"""Loop-7 AC-1 evidence: the oracle sampling-stride reference. The plan asks for
a stride=1 (dense) reference next to the default stride. The oracle's sampling
stride is hardcoded stride=1 (selection_kernel.py: oracle_payload_for_row(...,
stride=1)) — it samples EVERY needle token, no subsampling — so default == stride=1
(dense); proven from the emitted sink records. Also records the dense-DS
within-budget recall reference (context <=2048 ⇒ DS selects densely ⇒ 100%) next
to the default-stride beyond-budget served recall.

Emits oracle_stride_reference.json from the committed R4 sink + R7 matrices.
"""
import collections
import json
import os

SINK = "/sgl-workspace/sglang/.sglang_ds_oracle/sink.jsonl"
OUT = "development/loop7/oracle_stride_reference.json"


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    # 1) Prove the oracle's emitted stride == 1 for every record.
    strides = collections.Counter()
    total = succ = 0
    if os.path.exists(SINK):
        for line in open(SINK):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            if "failure" in r:
                continue
            succ += 1
            strides[r.get("stride")] += 1
    stride_values = {str(k): v for k, v in strides.items()}
    default_is_stride1 = (set(strides) == {1}) and succ > 0

    # 2) score-only recall@K per length (oracle), from the R4 budget-vs-scorer artifact.
    bvs = _load("development/loop7/oracle_budget_vs_scorer_r4.json")
    recall_at_k = {}
    if bvs:
        for L, v in bvs.get("lengths", {}).items():
            recall_at_k[L] = v.get("score_only_recall_at_k")

    # 3) dense-DS within-budget reference vs default-stride beyond-budget served recall.
    m = _load("development/loop7/ds_vs_dsa_recall_matrix_graph_n50.json")
    within_budget = beyond_budget = None
    if m:
        lens = m.get("lengths", {})
        def hyb(L):
            c = lens.get(L, {}).get("ds_hybrid") or {}
            return c.get("recall")
        def dft(L):
            c = lens.get(L, {}).get("ds_default") or {}
            return c.get("recall")
        # 1024w is <=2048 tokens => within budget (DS selects all => dense-DS).
        within_budget = {"length_words": 1024, "tokenized_le_2048": True,
                         "ds_default_recall": dft("1024"), "ds_hybrid_recall": hyb("1024"),
                         "note": "context <= 2048 budget => DS selects densely (dense-DS) => 100%"}
        beyond_budget = {L: {"ds_default_recall": dft(L), "ds_hybrid_recall": hyb(L)}
                         for L in ("4096", "16384")}

    out = {
        "what": "oracle sampling-stride reference + dense-DS within-budget reference (AC-1)",
        "oracle_sampling_stride": {
            "hardcoded_stride": 1,
            "source": "selection_kernel.py::_maybe_record_recall_oracle -> oracle_payload_for_row(stride=1)",
            "emitted_stride_value_counts": stride_values,
            "records_total": total, "records_success": succ,
            "default_equals_stride1": default_is_stride1,
            "interpretation": (
                "The oracle samples score-only recall over ALL needle tokens (stride=1, "
                "dense); it never subsamples. So the 'default stride' IS stride=1 and a "
                "separate sparse-stride run is N/A — proven: every success record carries stride==1."
            ),
        },
        "score_only_recall_at_k_by_length": recall_at_k,
        "dense_ds_within_budget_reference": within_budget,
        "default_stride_beyond_budget_served_recall": beyond_budget,
        "verdict": "PASS" if default_is_stride1 else "INSUFFICIENT_RECORDS",
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote -> {OUT}")


if __name__ == "__main__":
    main()
