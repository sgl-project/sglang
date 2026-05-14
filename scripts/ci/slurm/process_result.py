"""Process a raw srt-slurm benchmark result JSON into an aggregated format.

Usage (called once per result file):
    RESULT_FILENAME=<path_without_.json> PREFILL_GPUS=<n> DECODE_GPUS=<n> \\
        RECIPE_FILE=<path_to_recipe.yaml> python3 process_result.py

Required env vars:
    RESULT_FILENAME   - path to the result file without the .json extension
    FRAMEWORK         - e.g. dynamo-sglang
    PRECISION         - e.g. fp8, fp4
    MODEL_PREFIX      - short model label, e.g. dsr1
    ISL               - input sequence length
    OSL               - output sequence length
    PREFILL_GPUS      - number of prefill GPUs (extracted from result filename)
    DECODE_GPUS       - number of decode GPUs (extracted from result filename)

Optional env vars:
    RECIPE_FILE       - path to the srt-slurm recipe YAML; if set, topology
                        fields (TP, EP, DP, workers) are parsed from it
"""

import json
import os
import sys
from pathlib import Path


def require(var):
    val = os.environ.get(var)
    if val is None:
        print(f"ERROR: Missing required env var: {var}", file=sys.stderr)
        sys.exit(1)
    return val


result_filename = require("RESULT_FILENAME")
framework = require("FRAMEWORK")
precision = require("PRECISION")
model_prefix = require("MODEL_PREFIX")
isl = int(require("ISL"))
osl = int(require("OSL"))
prefill_gpus = int(require("PREFILL_GPUS"))
decode_gpus = int(require("DECODE_GPUS"))

with open(f"{result_filename}.json") as f:
    raw = json.load(f)

# ---------------------------------------------------------------------------
# Topology — parse from recipe YAML if available, otherwise default to 0/"-"
# ---------------------------------------------------------------------------
prefill_tp = prefill_ep = prefill_dp_attn = 0
prefill_num_workers = decode_tp = decode_ep = decode_dp_attn = decode_num_workers = 0

recipe_file = os.environ.get("RECIPE_FILE")
if recipe_file and Path(recipe_file).exists():
    import yaml

    with open(recipe_file) as f:
        recipe = yaml.safe_load(f)

    res = recipe.get("resources", {})
    prefill_num_workers = res.get("prefill_workers", 0)
    decode_num_workers = res.get("decode_workers", 0)

    sgl = recipe.get("backend", {}).get("sglang_config", {})
    p = sgl.get("prefill", {})
    d = sgl.get("decode", {})

    prefill_tp = p.get("tensor-parallel-size", 0)
    prefill_ep = p.get("expert-parallel-size", 0)
    prefill_dp_attn = p.get("data-parallel-size", "-")
    decode_tp = d.get("tensor-parallel-size", 0)
    decode_ep = d.get("expert-parallel-size", 0)
    decode_dp_attn = d.get("data-parallel-size", "-")

total_gpus = prefill_gpus + decode_gpus

data = {
    "hw": "gb200",
    "conc": int(raw["max_concurrency"]),
    "model": raw["model_id"],
    "infmax_model_prefix": model_prefix,
    "framework": framework,
    "precision": precision,
    "isl": isl,
    "osl": osl,
    "is_multinode": True,
    "disagg": True,
    "num_prefill_gpu": prefill_gpus,
    "num_decode_gpu": decode_gpus,
    "prefill_num_workers": prefill_num_workers,
    "prefill_tp": prefill_tp,
    "prefill_ep": prefill_ep,
    "prefill_dp_attention": prefill_dp_attn,
    "decode_num_workers": decode_num_workers,
    "decode_tp": decode_tp,
    "decode_ep": decode_ep,
    "decode_dp_attention": decode_dp_attn,
    "tput_per_gpu": float(raw["total_token_throughput"]) / total_gpus,
    "output_tput_per_gpu": float(raw["output_throughput"]) / decode_gpus,
    "input_tput_per_gpu": (
        float(raw["total_token_throughput"]) - float(raw["output_throughput"])
    )
    / prefill_gpus,
}

for key, value in raw.items():
    if key.endswith("_ms"):
        data[key.replace("_ms", "")] = float(value) / 1000.0
    if "tpot" in key:
        data[key.replace("_ms", "").replace("tpot", "intvty")] = 1000.0 / float(value)

out_path = Path(result_filename).parent / f"agg_{Path(result_filename).name}.json"
with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"Written: {out_path}")
