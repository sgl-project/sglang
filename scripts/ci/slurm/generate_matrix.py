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

MIN_EVAL_CONC = 8

# Only this (ISL, OSL) is eligible for eval auto-marking. 1k1k recipes have
# context budgets too small for gsm8k's chain-of-thought outputs to fit; 8k1k
# server context (~9600) gives comfortable headroom.
EVAL_SEQ_LEN = (8192, 1024)


def seq_len_str(isl, osl):
    def fmt(n):
        return f"{n // 1024}k" if n % 1024 == 0 else str(n)

    return f"{fmt(isl)}{fmt(osl)}"


def _pick_eval_entry(search_space):
    """Within a (precision, isl, osl) group, pick the entry with the highest
    max-concurrency (eligible concs only) and compute its eval concurrency.

    Returns (best_index, eval_conc) or (-1, None) if no entry is eligible.
    eval_conc is the upper-median of the chosen entry's eligible conc list —
    keeps eval-phase GPU memory comfortable without dropping concurrency so
    low that the eval drags on.
    """
    best_idx = -1
    best_max = -1
    best_eligible: list[int] = []
    for i, entry in enumerate(search_space):
        eligible = sorted(c for c in entry.get("conc-list", []) if c >= MIN_EVAL_CONC)
        if not eligible:
            continue
        if eligible[-1] > best_max:
            best_max = eligible[-1]
            best_idx = i
            best_eligible = eligible
    if best_idx < 0:
        return -1, None
    return best_idx, best_eligible[len(best_eligible) // 2]


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
            search_space = seq_cfg["search-space"]

            # Auto-select one entry per (exp, isl, osl) group to also run
            # lm-eval after the perf sweep. Picks the recipe with the highest
            # max-conc; lm-eval then runs at the upper-median of its concs.
            # Only the 8k1k seq-len is eval-eligible.
            if (isl, osl) == EVAL_SEQ_LEN:
                eval_idx, eval_conc = _pick_eval_entry(search_space)
            else:
                eval_idx, eval_conc = -1, None

            for i, entry in enumerate(search_space):
                config_file = entry["config_file"]
                topology = config_file.rsplit("/", 1)[-1].replace(".yaml", "")

                is_eval_entry = i == eval_idx
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
                        # Caps eval request size to fit the server's context.
                        "max_model_len": str(isl + osl + 256),
                        # Eval flags forwarded as env vars to srt-slurm's
                        # do_sweep.py. Strings so the GH Actions env: block
                        # copies them verbatim.
                        "run_eval": "true" if is_eval_entry else "false",
                        "eval_only": "false",
                        "eval_conc": str(eval_conc) if is_eval_entry else "",
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
