#!/usr/bin/env python3
"""Generate the AMD nightly serving-benchmark job matrix.

Reads ``scripts/ci/amd/benchmark-configs.yaml`` and prints a JSON array (one
entry per ``config x variant``) suitable for a GitHub Actions
``strategy.matrix``. Each emitted row carries everything
``scripts/ci/amd/run_benchmark.sh`` needs.

Filtering / overrides come from environment variables so the workflow can pass
optional ``workflow_dispatch`` inputs without fragile shell quoting:

  CONFIGS_FILTER      comma list of config names to keep   (empty = all)
  VARIANTS_FILTER     comma list of variants to keep        (empty = per-config)
  OVERRIDE_ISL        override isl for every row            (empty = per-config)
  OVERRIDE_OSL        override osl for every row
  OVERRIDE_DP_ATTN    override dp-attn for every row (true|false)
  OVERRIDE_CONC_LIST  override conc-list for every row
  OVERRIDE_MODEL      override model id/path for every row

To add a model, edit ``benchmark-configs.yaml`` -- not this script or the
workflow YAML.
"""

import json
import os
import sys

import yaml


def _csv(value):
    return [x for x in (value or "").replace(" ", "").split(",") if x]


def main():
    if len(sys.argv) < 2:
        print("usage: generate_benchmark_matrix.py <configs.yaml>", file=sys.stderr)
        return 2

    with open(sys.argv[1]) as f:
        doc = yaml.safe_load(f) or {}
    configs = doc.get("configs", []) or []

    name_filter = set(_csv(os.environ.get("CONFIGS_FILTER")))
    variant_filter = set(_csv(os.environ.get("VARIANTS_FILTER")))
    ov_isl = os.environ.get("OVERRIDE_ISL", "").strip()
    ov_osl = os.environ.get("OVERRIDE_OSL", "").strip()
    ov_dp = os.environ.get("OVERRIDE_DP_ATTN", "").strip()
    ov_conc = os.environ.get("OVERRIDE_CONC_LIST", "").strip()
    ov_model = os.environ.get("OVERRIDE_MODEL", "").strip()

    rows = []
    for cfg in configs:
        name = cfg["name"]
        if name_filter and name not in name_filter:
            continue
        for variant in cfg.get("variants", ["base"]):
            if variant_filter and variant not in variant_filter:
                continue
            rows.append(
                {
                    "name": name,
                    "model_prefix": cfg["model-prefix"],
                    "precision": cfg["precision"],
                    "runner": cfg["runner"],
                    "framework": cfg["framework"],
                    "model": ov_model or cfg["model"],
                    "runs_on": cfg["runs-on"],
                    "variant": variant,
                    "tp": str(cfg.get("tp", 8)),
                    "ep": str(cfg.get("ep", 1)),
                    "isl": ov_isl or str(cfg.get("isl", 1024)),
                    "osl": ov_osl or str(cfg.get("osl", 1024)),
                    "dp_attn": (ov_dp or str(cfg.get("dp-attn", False))).lower(),
                    "conc_list": ov_conc or str(cfg.get("conc-list", "32")),
                }
            )

    if not rows:
        print("ERROR: no benchmark configs matched the filters", file=sys.stderr)
        return 1

    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
