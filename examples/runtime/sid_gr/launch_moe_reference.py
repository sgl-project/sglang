"""Launch the optional strict MoE/EP numerical reference, separate from native MoE."""

import argparse
import json
import os
from pathlib import Path

from .attention import BACKEND
from .launch_server import validate_configuration as validate_dense_configuration


def validate_configuration(config):
    if config.get("tp_size") != config.get("ep_size"):
        raise ValueError("MoE reference requires TP=EP")
    # The remaining dense reference safety/geometry constraints also apply.
    validate_dense_configuration({**config, "ep_size": 1})
    if (
        config.get("moe_a2a_backend") != "none"
        or config.get("moe_runner_backend") != "triton"
    ):
        raise ValueError("MoE reference requires standard unquantized expert storage")
    if config.get("enable_eplb") or config.get("ep_num_redundant_experts", 0):
        raise ValueError("MoE reference requires static contiguous expert ownership")


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gr-kernel-path", type=Path, required=True)
    local, remaining = parser.parse_known_args(argv)
    kernel = local.gr_kernel_path.resolve(strict=True)
    if not (kernel / "interface.py").is_file():
        parser.error("Missing pinned GR kernel")
    os.environ["SGLANG_SID_GR_KERNEL_PATH"] = str(kernel)
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
        "--moe-a2a-backend",
        "none",
        "--moe-runner-backend",
        "triton",
        "--json-model-override-args",
        json.dumps({"architectures": ["SIDQwen3MoeForCausalLM"]}),
    ]
    args = prepare_server_args(defaults + remaining)
    config = args.resolved_dict()
    validate_configuration(config)
    if json.loads(config["json_model_override_args"]).get("architectures") != [
        "SIDQwen3MoeForCausalLM"
    ]:
        raise ValueError("MoE reference requires SIDQwen3MoeForCausalLM")
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
