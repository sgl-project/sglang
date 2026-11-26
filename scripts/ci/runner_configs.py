"""Emit runner_config setup for GitHub Actions $GITHUB_OUTPUT.

runner_configs.py <runner_config> <stage>
    Per-field `key=value` lines (install / artifact_version /
    install_timeout / rdma_devices) for the `{stage}-test-{runner_config}`
    suite, from resolve(). `runs_on` is intentionally omitted — it carries
    the `$b200_runner` sentinel and is resolved via --map. Called per stage
    by _pr-test-stage.yml and _diffusion-pr-test-stage.yml.

runner_configs.py --map <b200_runner_label>
    `runs_on_map={json}` — flat dict {runner_config: runs_on}, with
    `$b200_runner` substituted. Called once by _pr-test-check-changes.yml.
"""

import json
import os
import sys

import yaml

_YAML_PATH = os.path.join(os.path.dirname(__file__), "runner_configs.yml")
_B200_SENTINEL = "$b200_runner"


def _read() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)


def load() -> dict:
    return _read()["runner_configs"]


def _stage_kind(stage: str) -> str:
    """The stage less its tier and stage letter: `base-b-diffusion-unit` ->
    `diffusion-unit`; `base-b` and `nightly` give ''."""
    _tier, _, rest = stage.partition("-")
    letter, _, kind = rest.partition("-")
    return kind if len(letter) == 1 else rest


def resolve(runner_config: str, stage: str) -> dict:
    """The setup of the `{stage}-test-{runner_config}` suite: the
    runner_config's fields, with `install` replaced by the entry in
    `kind_installs` for the first word of the stage's kind, when it has one.
    Raises ValueError for an unknown runner_config."""
    data = _read()
    if runner_config not in data["runner_configs"]:
        raise ValueError(f"unknown runner_config: {runner_config!r}")
    cfg = dict(data["runner_configs"][runner_config])
    family = _stage_kind(stage).partition("-")[0]
    install = data["kind_installs"].get(family)
    if install is not None:
        cfg["install"] = install
    return cfg


def _emit_single(rc: str, stage: str) -> None:
    try:
        cfg = resolve(rc, stage)
    except ValueError as e:
        sys.exit(str(e))
    # runs_on goes through --map (resolves $b200_runner). Suppress it here so a
    # consumer can't accidentally read the raw sentinel value.
    for key, value in cfg.items():
        if key == "runs_on":
            continue
        print(f"{key}={value}")


def _emit_map(b200_runner: str) -> None:
    runs_on = {
        name: (b200_runner if cfg.get("runs_on") == _B200_SENTINEL else cfg["runs_on"])
        for name, cfg in load().items()
    }
    print(f"runs_on_map={json.dumps(runs_on, separators=(',', ':'))}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--map":
        _emit_map(args[1])
    elif len(args) == 2:
        _emit_single(args[0], args[1])
    else:
        sys.exit(
            "usage:\n"
            "  runner_configs.py <runner_config> <stage>\n"
            "  runner_configs.py --map <b200_runner_label>"
        )
