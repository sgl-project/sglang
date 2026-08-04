"""
Reads nightly-configs.yaml and generates one matrix entry per recipe YAML,
where each srt-slurm recipe runs its full concurrency sweep as a single Slurm job.

conc-list in the config is documentation only and is not used to split jobs.

Each entry carries `nodes`, the size of the allocation the leg will salloc. A
caller can split its matrix on that with --nodes-max/--nodes-min so legs that
fit on the cluster side by side run in parallel while legs that need the whole
cluster run alone.

Output: JSON array written to stdout, consumed by the workflow setup job as
a dynamic matrix via fromJson(needs.setup.outputs.matrix).

Usage:
    python3 generate_matrix.py <path-to-nightly-configs.yaml> --runner <label> [--filter NAMES]

Example:
    python3 generate_matrix.py scripts/ci/slurm/nightly-configs.yaml --runner gb200
    python3 generate_matrix.py scripts/ci/slurm/nightly-configs.yaml --runner gb200 \\
        --filter dsr1-fp8-1k1k-max-tpt,dsr1-fp4-1k1k-mid-curve
    python3 generate_matrix.py scripts/ci/slurm/nightly-configs.yaml --runner mi355x \\
        --nodes-max 2
"""

import argparse
import json
import sys

import yaml

# Mirrors ${GPUS_PER_NODE:-8} in launch_mi355x.sh.
GPUS_PER_NODE = 8

# Node count for a recipe that is not checked into this repo -- the gb200
# entries point at recipes hosted in NVIDIA/srt-slurm, so their allocation size
# cannot be read here.
NODES_UNKNOWN = 0


def seq_len_str(isl, osl):
    def fmt(n):
        return f"{n // 1024}k" if n % 1024 == 0 else str(n)

    return f"{fmt(isl)}{fmt(osl)}"


def recipe_nodes(config_file):
    """Nodes the leg will salloc; mirrors TOTAL_NODES in launch_mi355x.sh.

    An engine whose TP exceeds one node's GPU count spans ceil(TP/GPUS_PER_NODE)
    nodes, so a role contributes workers * nodes-per-engine. Returns
    NODES_UNKNOWN when the recipe lives outside this repo. A recipe that IS
    present but malformed raises, because silently mis-sizing an allocation is
    worse than a loud failure in the setup job.
    """
    try:
        with open(config_file) as f:
            recipe = yaml.safe_load(f)
    except FileNotFoundError:
        return NODES_UNKNOWN

    resources = recipe.get("resources") or {}
    backend = recipe["backend"]["sglang_config"]

    def role_nodes(role, workers_key):
        tp = backend[role]["tensor-parallel-size"]
        nodes_per_engine = -(-tp // GPUS_PER_NODE)
        return resources.get(workers_key, 1) * nodes_per_engine

    return role_nodes("prefill", "prefill_workers") + role_nodes(
        "decode", "decode_workers"
    )


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
    parser.add_argument(
        "--nodes-max",
        type=int,
        help="Keep only entries whose allocation is at most N nodes.",
    )
    parser.add_argument(
        "--nodes-min",
        type=int,
        help="Keep only entries whose allocation is at least N nodes.",
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
                        "model_path": exp.get("model_path", ""),
                        "precision": exp["precision"],
                        "isl": str(isl),
                        "osl": str(osl),
                        "config_file": config_file,
                        "nodes": recipe_nodes(config_file),
                    }
                )

    # Name filter runs first so an unknown name is reported against the full set
    # for this runner rather than against whichever node-count slice is active.
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

    if args.nodes_max is not None or args.nodes_min is not None:
        # Refuse to slice a set we cannot size: dropping such an entry would
        # silently remove a leg from the nightly, and keeping it would hand the
        # runner an allocation the caller did not budget for.
        unsized = [e["name"] for e in matrix if e["nodes"] == NODES_UNKNOWN]
        if unsized:
            print(
                "ERROR: cannot filter by node count -- recipe not checked into "
                f"this repo for: {', '.join(unsized)}",
                file=sys.stderr,
            )
            sys.exit(1)
        lo = args.nodes_min if args.nodes_min is not None else 0
        hi = args.nodes_max if args.nodes_max is not None else sys.maxsize
        matrix = [e for e in matrix if lo <= e["nodes"] <= hi]

    print(json.dumps(matrix))


if __name__ == "__main__":
    main()
