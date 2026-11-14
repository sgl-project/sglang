"""
Reads nightly-configs.yaml and generates one matrix entry per recipe YAML,
where each srt-slurm recipe runs its full concurrency sweep as a single Slurm job.

conc-list in the config is documentation only and is not used to split jobs.

Output: JSON array written to stdout, consumed by the workflow setup job as
a dynamic matrix via fromJson(needs.setup.outputs.matrix).

Usage:
    python3 generate_matrix.py <path-to-nightly-configs.yaml> --runner <label> [--filter NAMES]

Example:
    python3 generate_matrix.py scripts/ci/slurm/nightly-configs.yaml --runner gb200
    python3 generate_matrix.py scripts/ci/slurm/nightly-configs.yaml --runner gb200 \\
        --filter dsr1-fp8-1k1k-max-tpt,dsr1-fp4-1k1k-mid-curve
"""

import argparse
import json
import sys

import yaml


def seq_len_str(isl, osl):
    def fmt(n):
        return f"{n // 1024}k" if n % 1024 == 0 else str(n)

    return f"{fmt(isl)}{fmt(osl)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to nightly-configs.yaml")
    parser.add_argument(
        "--runner",
        required=True,
        help="Filter configs by runner label (e.g. gb200, b200)",
    )
    parser.add_argument(
        "--filter",
        default="",
        help=(
            "Optional comma-separated list of matrix entry names to include "
            "(e.g. 'dsr1-fp8-1k1k-max-tpt'). Names must match exactly."
        ),
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        data = yaml.safe_load(f)

    matrix = []
    for exp_name, exp in data.items():
        if exp["runner"] != args.runner:
            continue

        for seq_cfg in exp["seq-len-configs"]:
            isl, osl = seq_cfg["isl"], seq_cfg["osl"]
            sl = seq_len_str(isl, osl)

            for entry in seq_cfg["search-space"]:
                config_file = entry["config_file"]
                topology = config_file.rsplit("/", 1)[-1].replace(".yaml", "")

                matrix.append(
                    {
                        "name": f"{exp['model-prefix']}-{exp['precision']}-{sl}-{topology}",
                        "exp_name": exp_name,
                        "model": exp["model"],
                        "model_prefix": exp["model-prefix"],
                        "precision": exp["precision"],
                        "isl": str(isl),
                        "osl": str(osl),
                        "config_file": config_file,
                    }
                )

    wanted = [n.strip() for n in args.filter.split(",") if n.strip()]
    if wanted:
        available = [e["name"] for e in matrix]
        unknown = [n for n in wanted if n not in available]
        if unknown:
            print(
                f"ERROR: unknown config name(s): {', '.join(unknown)}. "
                f"Available for runner '{args.runner}': {', '.join(available)}",
                file=sys.stderr,
            )
            sys.exit(1)
        matrix = [e for e in matrix if e["name"] in wanted]

    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
