#!/usr/bin/env python3
"""Generate the AMD nightly serving-benchmark job matrix (or model groups).

Reads ``scripts/ci/amd/benchmark-configs.yaml``.

Default mode prints a JSON array (one entry per ``config x variant``) for a
GitHub Actions ``strategy.matrix``. Each row carries the override-able per-run
knobs plus a base64-encoded ``launch_spec`` (env + server/client args) that
``scripts/ci/amd/run_benchmark.sh`` decodes to launch the sglang server itself.

With ``--list-groups`` it instead prints a JSON array of the unique model
groups (``model-prefix`` values). The parent workflow uses this to fan out one
nested (reusable-workflow) job per model, and each calls back here with
``GROUP_FILTER`` set to expand only that model's configs.

The base/mtp launch spec is merged here (so the launcher stays variant-agnostic)
and base64-encoded so it survives ``-e`` / ``docker exec`` without quoting issues.

Filtering / overrides come from environment variables so the workflow can pass
optional inputs without fragile shell quoting:

  GROUP_FILTER        comma list of model-prefixes to keep (empty = all)
  CONFIGS_FILTER      comma list of config names to keep    (empty = all)
  VARIANTS_FILTER     comma list of variants to keep         (empty = per-config)
  OVERRIDE_ISL        override isl for every row             (empty = per-config)
  OVERRIDE_OSL        override osl for every row
  OVERRIDE_DP_ATTN    override dp-attn for every row (true|false)
  OVERRIDE_CONC_LIST  override conc-list for every row
  OVERRIDE_MODEL      override model id/path for every row

To add a model, edit ``benchmark-configs.yaml`` -- not this script or the
workflow YAML.
"""

import base64
import json
import os
import sys

import yaml


def _csv(value):
    return [x for x in (value or "").replace(" ", "").split(",") if x]


def _join(*parts):
    """Join non-empty arg strings with single spaces."""
    return " ".join(p.strip() for p in parts if p and p.strip())


def _launch_spec_b64(cfg, variant):
    """Merge the base + variant launch spec and return it base64(JSON)-encoded."""
    env = dict(cfg.get("env") or {})
    server_args = cfg.get("server-args", "") or ""
    client_args = cfg.get("client-args", "") or ""

    if variant == "mtp":
        env.update(cfg.get("mtp-env") or {})
        server_args = _join(server_args, cfg.get("mtp-server-args", ""))
        client_args = _join(client_args, cfg.get("mtp-client-args", ""))

    spec = {
        "env": {str(k): str(v) for k, v in env.items()},
        "server_args": server_args,
        "client_args": client_args,
        "chat_template": str(cfg.get("chat-template", "") or ""),
        "bench_backend": str(cfg.get("bench-backend", "vllm") or "vllm"),
    }
    return base64.b64encode(json.dumps(spec).encode()).decode()


def main():
    argv = [a for a in sys.argv[1:] if a != "--list-groups"]
    list_groups = "--list-groups" in sys.argv[1:]
    if not argv:
        print(
            "usage: generate_benchmark_matrix.py <configs.yaml> [--list-groups]",
            file=sys.stderr,
        )
        return 2

    with open(argv[0]) as f:
        doc = yaml.safe_load(f) or {}
    configs = doc.get("configs", []) or []

    group_filter = set(_csv(os.environ.get("GROUP_FILTER")))
    name_filter = set(_csv(os.environ.get("CONFIGS_FILTER")))
    variant_filter = set(_csv(os.environ.get("VARIANTS_FILTER")))
    ov_isl = os.environ.get("OVERRIDE_ISL", "").strip()
    ov_osl = os.environ.get("OVERRIDE_OSL", "").strip()
    ov_dp = os.environ.get("OVERRIDE_DP_ATTN", "").strip()
    ov_conc = os.environ.get("OVERRIDE_CONC_LIST", "").strip()
    ov_model = os.environ.get("OVERRIDE_MODEL", "").strip()

    def keep(cfg):
        if group_filter and cfg["model-prefix"] not in group_filter:
            return False
        if name_filter and cfg["name"] not in name_filter:
            return False
        return True

    if list_groups:
        groups = []
        for cfg in configs:
            if keep(cfg) and cfg["model-prefix"] not in groups:
                groups.append(cfg["model-prefix"])
        if not groups:
            print("ERROR: no benchmark configs matched the filters", file=sys.stderr)
            return 1
        print(json.dumps(groups))
        return 0

    rows = []
    for cfg in configs:
        if not keep(cfg):
            continue
        for variant in cfg.get("variants", ["base"]):
            if variant_filter and variant not in variant_filter:
                continue
            rows.append(
                {
                    "name": cfg["name"],
                    "model_prefix": cfg["model-prefix"],
                    "precision": cfg["precision"],
                    "runner": cfg["runner"],
                    "framework": cfg["framework"],
                    "model": ov_model or cfg["model"],
                    "runs_on": cfg["runs-on"],
                    "rocm_version": str(cfg.get("rocm-version", "rocm720")),
                    "variant": variant,
                    "tp": str(cfg.get("tp", 8)),
                    "ep": str(cfg.get("ep", 1)),
                    "isl": ov_isl or str(cfg.get("isl", 1024)),
                    "osl": ov_osl or str(cfg.get("osl", 1024)),
                    "dp_attn": (ov_dp or str(cfg.get("dp-attn", False))).lower(),
                    "conc_list": ov_conc or str(cfg.get("conc-list", "32")),
                    "launch_spec_b64": _launch_spec_b64(cfg, variant),
                }
            )

    if not rows:
        print("ERROR: no benchmark configs matched the filters", file=sys.stderr)
        return 1

    print(json.dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
