"""tune deepgemm int8 tile config"""

import argparse
import json

from sglang.srt.layers import deep_gemm_tuner as tuner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="anonymous")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tuned-config", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tp-size", "--tp", type=int, default=2)
    parser.add_argument(
        "--mnk-configs",
        "-f",
        help="json file with dumped [m, n, k, n_groups] test cases",
        type=str,
        default=tuner.get_default_deep_gemm_dump_file(),
    )
    parser.add_argument(
        "--disable-oob",
        help="directly save tuned config to where sglang can find",
        action="store_true",
    )
    args = parser.parse_args()

    if args.model:
        tuner.tuning_deepgemm_model_config(args)
    else:
        # load mnk file
        with open(args.mnk_configs, "r") as f:
            test_cases = json.load(f)
        tuner.tuning_deepgemm_config_entrypoint(
            test_cases,
            args.tp_size,
            args.seed,
            args.model,
            out_of_box=not args.disable_oob,
        )
