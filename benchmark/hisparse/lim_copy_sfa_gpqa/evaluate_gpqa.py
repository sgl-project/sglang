"""Complete GPQA Diamond, sgl-eval scoring, fixed A/B generation settings."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import requests
from sglang.test.sgl_eval_utils import run_sgl_eval

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--num-examples", type=int, default=198)
p.add_argument("--threads", type=int, default=8)
args = p.parse_args()
root = Path("/task/results")
info = requests.get("http://127.0.0.1:8810/server_info", timeout=30).json()
root.joinpath(args.tag + ".server-info.json").write_text(
    json.dumps(info, indent=2) + "\n"
)
config = info.get("server_args", info)
assert config.get("model_path"), "Missing server configuration"
assert "speculative_algorithm" in config, "Missing speculative decoding configuration"
assert config.get("speculative_algorithm") in (None, "NONE"), config.get(
    "speculative_algorithm"
)
assert not config.get("speculative_draft_model_path"), config.get(
    "speculative_draft_model_path"
)
config_args = SimpleNamespace(
    eval_name="gpqa",
    model="GLM-5.2-w4a8",
    base_url="http://127.0.0.1:8810/v1",
    num_examples=args.num_examples,
    num_threads=args.threads,
    repeat=1,
    max_tokens=32768,
    temperature=0.0,
    top_p=0.95,
    seed=42,
    thinking_mode="glm-45",
    chat_template_kwargs={"enable_thinking": True},
    sgl_eval_out_dir=str(root / args.tag),
)
record = {
    "dataset_sha256": hashlib.sha256(
        (root / "gpqa-diamond-198-seed42.json").read_bytes()
    ).hexdigest(),
    "sgl_eval_version": "0.1.3",
    "repeat": 1,
    "seed": 42,
    "temperature": 0.0,
    "max_tokens": 32768,
    "thinking": True,
    "num_examples": args.num_examples,
    "num_threads": args.threads,
    "mtp": False,
    "metrics": run_sgl_eval(config_args),
}
root.joinpath(args.tag + ".gpqa.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2), flush=True)
