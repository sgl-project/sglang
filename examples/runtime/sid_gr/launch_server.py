"""Launch the opt-in SID numerical reference example with normal SGLang HTTP."""

import argparse
import json
import os
import sys
from pathlib import Path

from .attention import BACKEND


def validate_configuration(config):
    required = {
        "enable_gr": True,
        "dtype": "bfloat16",
        "page_size": 1,
        "pp_size": 1,
        "ep_size": 1,
        "dp_size": 1,
        "attn_dp_size": 1,
        "attn_cp_size": 1,
        "attention_backend": BACKEND,
        "chunked_prefill_size": -1,
        "disable_radix_cache": True,
        "disable_overlap_schedule": True,
        "cuda_graph_backend_decode": "disabled",
        "cuda_graph_backend_prefill": "disabled",
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(
                f"SID reference requires {name}={expected!r}, got {config.get(name)!r}"
            )
    if config.get("tp_size") not in (1, 2, 4):
        raise ValueError("SID reference supports TP=1, 2 or 4")
    if config.get("dcp_size", 1) != 1:
        raise ValueError("SID reference requires dcp_size=1")
    for name in (
        "quantization",
        "speculative_algorithm",
        "enable_torch_compile",
        "enable_lora",
        "enable_layernorm_sp",
        "enable_fp32_lm_head",
    ):
        if config.get(name):
            raise ValueError(f"SID reference does not support {name}")
    if config.get("kv_cache_dtype") not in ("auto", "bfloat16"):
        raise ValueError("SID reference requires BF16 KV")
    for name in ("prefill_attention_backend", "decode_attention_backend"):
        if config.get(name) not in (None, BACKEND):
            raise ValueError(f"SID reference requires {name}={BACKEND}")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gr-kernel-path", type=Path, required=True)
    local, remaining = parser.parse_known_args(argv)
    kernel_path = local.gr_kernel_path.resolve(strict=True)
    if not (kernel_path / "interface.py").is_file():
        parser.error("--gr-kernel-path must contain the pinned GR interface.py")
    os.environ["SGLANG_SID_GR_KERNEL_PATH"] = str(kernel_path)
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "examples.runtime.sid_gr"

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    defaults = [
        "--enable-gr",
        "--dtype",
        "bfloat16",
        "--page-size",
        "1",
        "--attention-backend",
        BACKEND,
        "--chunked-prefill-size",
        "-1",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--json-model-override-args",
        json.dumps({"architectures": ["SIDQwen3ForCausalLM"]}),
    ]
    args = prepare_server_args(defaults + remaining)
    config = args.resolved_dict()
    validate_configuration(config)
    overrides = json.loads(config["json_model_override_args"])
    if overrides.get("architectures") != ["SIDQwen3ForCausalLM"]:
        raise ValueError("SID reference requires the SIDQwen3ForCausalLM architecture")
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main(sys.argv[1:])
