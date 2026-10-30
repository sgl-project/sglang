#!/usr/bin/env python3
"""Tune sglang's Triton fused-MoE kernel for the MoE FFN of a DFlash/DSpark draft export.

Temporary helper for sgl-project/sglang#47181, to be removed before merge: sglang already ships the tuner
(benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py). This script only feeds it the draft's expert
shape, read the way the server reads it (``parse_dflash_moe_spec``), and the token counts a DSpark server runs
the draft FFN with (block size x concurrent requests), then moves the result where the runtime loader looks.

Example:
    python scripts/tune_dflash_moe_draft.py --draft /path/to/draft-export --block-size 8
    # -> python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_<ver>/E=8,N=1536,device_name=NVIDIA_H200.json

With --output-dir DIR the file goes to DIR/configs/triton_<ver>/ and the server needs SGLANG_MOE_CONFIG_DIR=DIR.
Needs an sglang source checkout (editable install) and ray; uses every visible GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def tuner_dir(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_dir() else path.parent
    import sglang

    repo = Path(sglang.__file__).resolve().parents[2]  # <repo>/python/sglang/__init__.py
    path = repo / "benchmark" / "kernels" / "fused_moe_triton"
    if not (path / "tuning_fused_moe_triton.py").is_file():
        sys.exit(f"tuner not found at {path}; pass --tuner (needs an sglang source checkout)")
    return path


def in_tree_config_root() -> Path:
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config

    return Path(fused_moe_triton_config.__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--draft", required=True, help="DFlash/DSpark draft export (config.json + weights)")
    parser.add_argument(
        "--block-size", type=int, default=8,
        help="the server's --speculative-dspark-block-size: draft tokens per request per step",
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=DEFAULT_CONCURRENCY,
        help="concurrent requests to cover; tokens per step = block size x concurrency",
    )
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "fp8_w8a8", "int8_w8a16", "int8_w8a8", "int4_w4a16"]
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="config root; default is sglang's in-tree one (picked up automatically), "
        "anything else needs SGLANG_MOE_CONFIG_DIR=<dir> on the server",
    )
    parser.add_argument("--tuner", default=None, help="tuning_fused_moe_triton.py or its directory")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the tuner command only")
    args = parser.parse_args()

    import triton
    from sglang.srt.models.dflash import parse_dflash_moe_spec
    from sglang.srt.utils.hf_transformers_utils import get_config

    config = get_config(args.draft, trust_remote_code=True)
    spec = parse_dflash_moe_spec(config)
    if spec is None:
        sys.exit(f"{args.draft}: the config has no routed experts (dense draft), nothing to tune")
    hidden_size = int(config.hidden_size)
    batch_sizes = sorted({args.block_size * c for c in args.concurrency})
    version_dir = f"triton_{triton.__version__.replace('.', '_')}"
    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else in_tree_config_root()
    dest_dir = root / "configs" / version_dir
    tdir = tuner_dir(args.tuner)

    # The tuner takes the expert shape from an HF config by architecture name, so hand it a stub
    # of a known MoE architecture that carries the draft's shape.
    stub = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_size": hidden_size,
        "intermediate_size": hidden_size,
        "moe_intermediate_size": spec.intermediate_size,
        "num_experts": spec.num_experts,
        "num_experts_per_tok": spec.top_k,
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "torch_dtype": "bfloat16",
    }
    cmd = [
        sys.executable, "tuning_fused_moe_triton.py", "--model", "<stub>",
        "--tp-size", str(args.tp_size), "--dtype", args.dtype, "--tune",
        "--batch-sizes", *map(str, batch_sizes),
    ]
    print(
        f"draft {args.draft}: E={spec.num_experts} top-k={spec.top_k} N={spec.intermediate_size} "
        f"hidden={hidden_size} shared-expert={spec.shared_intermediate_size or 'none'}"
    )
    print(f"tokens per step to tune: {batch_sizes}")
    print(f"tuner: {tdir}")
    print(f"destination: {dest_dir}/E={spec.num_experts},N={spec.intermediate_size},device_name=<gpu>.json")
    print("command (cwd = tuner dir):", " ".join(cmd))
    if args.dry_run:
        return 0

    with tempfile.TemporaryDirectory(prefix="dflash-moe-stub-") as stub_dir:
        (Path(stub_dir) / "config.json").write_text(json.dumps(stub, indent=2) + "\n")
        cmd[cmd.index("<stub>")] = stub_dir
        before = {p: p.stat().st_mtime for p in tdir.glob("E=*.json")}
        subprocess.run(cmd, cwd=tdir, check=True)
        produced = [
            p for p in tdir.glob(f"E={spec.num_experts},N={spec.intermediate_size},*.json")
            if p not in before or p.stat().st_mtime > before[p]
        ]
        if not produced:
            sys.exit("the tuner finished but wrote no E=*,N=*.json next to itself")
        src = max(produced, key=lambda p: p.stat().st_mtime)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        shutil.move(str(src), dest)
    print(f"wrote {dest}")
    if args.output_dir:
        print(f"start the server with SGLANG_MOE_CONFIG_DIR={root}")
    else:
        print("in-tree: the next server start uses it (the 'Using default MoE kernel config' warning goes away)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
