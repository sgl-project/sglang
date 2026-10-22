import csv
import hashlib
import json
import random
from pathlib import Path

from sgl_eval._vendored.nemo_skills.dataset.gpqa.prepare import format_entry
from sgl_eval.evals._loader import _CACHE_ROOT, _commit_with_marker, load_via_prepare
from sgl_eval.registry import get

root = Path("/task/results")
spec = get("gpqa")
print("SPEC", spec, flush=True)
csv_path = Path("/task/inputs/gpqa_diamond.csv")
if csv_path.exists():
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 198, len(rows)
    random.seed(42)
    prepared = root / "gpqa-diamond-prepared.jsonl"
    prepared.write_text("".join(json.dumps(format_entry(r)) + "\n" for r in rows))
    cached = _CACHE_ROOT / "gpqa/diamond.jsonl"
    cached.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    staged = cached.with_name(".diamond.staged.jsonl")
    shutil.copyfile(prepared, staged)
    _commit_with_marker(staged, cached)
    print(
        "SOURCE_CSV_SHA256",
        hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        flush=True,
    )
examples = load_via_prepare("gpqa", ("diamond",), {"random_seed": 42})(None)
assert len(examples) == 198, len(examples)
data = [
    {
        "id": e.id,
        "problem": e.inputs["problem"],
        "expected_answer": e.target,
        "meta": e.meta,
    }
    for e in examples
]
out = root / "gpqa-diamond-198-seed42.json"
out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print("DATASET", len(data), hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)
