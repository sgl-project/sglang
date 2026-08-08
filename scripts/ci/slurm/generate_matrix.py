"""
Reads nightly-configs.yaml and generates one matrix entry per recipe YAML,
where each srt-slurm recipe runs its full concurrency sweep as a single Slurm job.

conc-list in the config is documentation only and is not used to split jobs.

With --with-nodes, each entry also carries `nodes`, the size of the Slurm
allocation the leg will make. The MI355X nightly stages its matrix on that value
so legs that fit on the cluster side by side run in parallel while legs that
need the whole cluster run alone. The flag is opt-in so callers that do not
stage on it (gb200) keep byte-identical output -- matrix values show up in
GitHub job names, so an extra field would rename their jobs for no reason.

Output: JSON array written to stdout, consumed by the workflow setup job as
a dynamic matrix via fromJson(needs.setup.outputs.matrix).

Usage:
    python3 generate_matrix.py <path-to-nightly-configs.yaml> --runner <label> [--filter NAMES]

Example:
    python3 generate_matrix.py scripts/ci/slurm/nightly-configs.yaml --runner gb200
    python3 generate_matrix.py scripts/ci/slurm/nightly-configs.yaml --runner gb200 \\
        --filter dsr1-fp8-1k1k-max-tpt,dsr1-fp4-1k1k-mid-curve
    python3 generate_matrix.py scripts/ci/slurm/nightly-configs.yaml --runner mi355x \\
        --with-nodes
"""

import argparse
import json
import sys

import yaml

# Mirrors ${GPUS_PER_NODE:-8} in launch_mi355x.sh.
GPUS_PER_NODE = 8

# Node count for a recipe that is not checked into this repo -- the gb200
# entries point at recipes hosted in NVIDIA/srt-slurm, so their allocation size
# cannot be read here. Consumers that stage on node count must reject this
# rather than guess.
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
        "--with-nodes",
        action="store_true",
        help=(
            "Add a `nodes` field per entry with the size of the Slurm "
            "allocation the leg will make. Opt-in so runners that do not stage "
            "on it keep byte-identical output."
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

                extra = {"nodes": recipe_nodes(config_file)} if args.with_nodes else {}
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
                        **extra,
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
