#!/usr/bin/env python3
"""Check that the rust-ext cache key agrees across the sites that build it.

A consumer derives its prefix from the runner it lands on; a producer is told one
by its caller. Neither file can see the other, and a disagreement costs no error
at runtime - just a miss, and a job that quietly source-builds.

`stage_rust_ext_modules.sh` checks a producer's prefix against the modules it
built, which is the authority. This runs before any runner is queued, and covers
the half that check cannot see: which prefix a consumer stage will look up.
"""

import pathlib
import re
import sys

import msgspec
import yaml

BUILD_WORKFLOW = ".github/workflows/_pr-test-rust-ext-build.yml"
DOWNLOAD_ACTION = ".github/actions/download-rust-ext/action.yml"
STAGE_WORKFLOW = "_pr-test-stage.yml"
RUNNER_CONFIGS = "scripts/ci/runner_configs.yml"
WORKFLOW_GLOB = ".github/workflows/*.yml"

# Declared, not read off the label text: `4-gpu-gb300` is Grace and names no arm,
# and a missing label is an error asking for an entry rather than a guess.
RUNNER_ARCHES = {
    "ubuntu-latest": "x86_64",
    "ubuntu-22.04": "x86_64",
    "ubuntu-24.04": "x86_64",
    "ubuntu-22.04-arm": "aarch64",
    "ubuntu-24.04-arm": "aarch64",
    "x64-kernel-build-node": "x86_64",
    "arm-kernel-build-node": "aarch64",
    "1-gpu-5090": "x86_64",
    "1-gpu-h100": "x86_64",
    "2-gpu-h100": "x86_64",
    "4-gpu-h100": "x86_64",
    "8-gpu-h20": "x86_64",
    "8-gpu-h200": "x86_64",
    "8-gpu-b200": "x86_64",
    "8-gpu-b300": "x86_64",
    "4-gpu-b200": "x86_64",
    "4-gpu-gb300": "aarch64",
    # check-changes substitutes a 4-gpu-b200 variant here, all of them x86_64.
    "$b200_runner": "x86_64",
}

_UNAME = "$(uname -m)"
_HASH_FILES = re.compile(r"hashFiles\(([^)]*)\)")
_QUOTED = re.compile(r"'([^']*)'")
# The one line the action spells the prefix out on, e.g.
#   resolved="rust-ext-$(uname -m)-cp310-cp312"
_DERIVED_PREFIX = re.compile(r'"([^"]*\$\(uname -m\)[^"]*)"')
_NEEDS_JOB = re.compile(r"needs\.([A-Za-z0-9_.-]+)\.outputs")
_MATRIX_REF = re.compile(r"^\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}$")


class _Unparsable(Exception):
    """A YAML file this check has to read does not parse."""


class _Producer(msgspec.Struct, frozen=True):
    """One caller of the build workflow: what it saves under, and from where."""

    file: str
    job: str
    prefix: str
    runs_on: str
    collect_runs_on: str


class _StageCaller(msgspec.Struct, frozen=True):
    """One test stage on one pool. `runner_config` is None when it is an
    expression this check cannot resolve to a pool."""

    file: str
    job: str
    declared: str
    runner_config: str | None
    producer_job: str | None
    skips: bool


def _load(path: pathlib.Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as error:
        raise _Unparsable(f"ERROR: cannot read {path}: {error}") from error


def _jobs(workflow: dict) -> dict:
    return {
        name: job
        for name, job in (workflow.get("jobs") or {}).items()
        if isinstance(job, dict)
    }


def _workflow_call_inputs(workflow: dict) -> dict:
    # yaml 1.1 parses the `on:` key as boolean True
    triggers = workflow.get("on", workflow.get(True)) or {}
    return triggers["workflow_call"]["inputs"]


def _hashed_inputs(path: pathlib.Path) -> list[tuple[str, ...]]:
    """The argument tuple of every ``hashFiles(...)`` cache key in a file."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return [tuple(_QUOTED.findall(args)) for args in _HASH_FILES.findall(text)]


def _derived_prefix(action: pathlib.Path) -> str | None:
    """The prefix literal the action derives, `$(uname -m)` still in it."""
    with open(action, encoding="utf-8") as f:
        found = _DERIVED_PREFIX.findall(f.read())
    return found[0] if len(found) == 1 else None


def _prefix_pattern(literal: str) -> "re.Pattern[str]":
    """Matches a producer prefix, capturing its arch and its interpreter tail."""
    head, _, tail = literal.partition(_UNAME)
    arches = "|".join(re.escape(arch) for arch in sorted(set(RUNNER_ARCHES.values())))
    # The tail is captured, not fixed, so a prefix naming the wrong interpreters
    # still yields an arch and does not strand its consumers behind a second error.
    return re.compile(
        f"^{re.escape(head)}({arches})-(.+)$"
        if tail.startswith("-")
        else f"^{re.escape(head)}({arches}){re.escape(tail)}$"
    )


def _restore_key_uses_derived_prefix(action: pathlib.Path) -> bool:
    """Whether the cache restore key interpolates the resolve step's output."""
    # The literal existing in the file proves nothing: a key pointed back at an
    # input leaves it dead, every consumer looking up a prefix no producer saves.
    steps = _load(action).get("runs", {}).get("steps", [])
    resolve_ids = [
        step.get("id")
        for step in steps
        if isinstance(step, dict) and _DERIVED_PREFIX.search(str(step.get("run", "")))
    ]
    if len(resolve_ids) != 1 or not resolve_ids[0]:
        return False
    wanted = f"steps.{resolve_ids[0]}.outputs"
    return any(
        wanted in str((step.get("with") or {}).get("key", ""))
        for step in steps
        if isinstance(step, dict) and "cache/restore" in str(step.get("uses", ""))
    )


def _interpreter_tokens(root: pathlib.Path) -> str:
    """The compile matrix's interpreters as a prefix spells them, e.g. cp310-cp312."""
    versions = _jobs(_load(root / BUILD_WORKFLOW))["compile"]["strategy"]["matrix"][
        "python-version"
    ]
    # Lexicographic, matching how stage_rust_ext_modules.sh sorts the suffix set.
    return "-".join(sorted(f"cp{str(v).replace('.', '')}" for v in versions))


def _producers(root: pathlib.Path) -> list[_Producer]:
    inputs = _workflow_call_inputs(_load(root / BUILD_WORKFLOW))
    build_workflow_name = BUILD_WORKFLOW.rsplit("/", 1)[-1]
    found = []
    for path in sorted(root.glob(WORKFLOW_GLOB)):
        for name, job in _jobs(_load(path)).items():
            if not str(job.get("uses", "")).endswith(build_workflow_name):
                continue
            with_ = job.get("with") or {}
            found.append(
                _Producer(
                    file=str(path.relative_to(root)),
                    job=name,
                    prefix=with_.get(
                        "cache_key_prefix", inputs["cache_key_prefix"]["default"]
                    ),
                    runs_on=str(with_.get("runs_on", "")),
                    collect_runs_on=str(
                        with_.get(
                            "collect_runs_on", inputs["collect_runs_on"]["default"]
                        )
                    ),
                )
            )
    return found


def _matrix_values(job: dict, key: str) -> list[str]:
    """Every value `matrix.<key>` takes, across `include:` rows and plain lists."""
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    return [str(value) for value in matrix.get(key, [])] + [
        str(row[key])
        for row in matrix.get("include", [])
        if isinstance(row, dict) and key in row
    ]


def _runner_configs_of(job: dict, declared: str) -> list[str]:
    reference = _MATRIX_REF.match(declared)
    if reference is None:
        return [declared] if "${{" not in declared else []
    return _matrix_values(job=job, key=reference.group(1))


def _stage_callers(root: pathlib.Path) -> list[_StageCaller]:
    """Every test stage a workflow calls, one entry per pool it resolves to."""
    found = []
    for path in sorted(root.glob(WORKFLOW_GLOB)):
        for name, job in _jobs(_load(path)).items():
            with_ = job.get("with") or {}
            if not str(job.get("uses", "")).endswith(STAGE_WORKFLOW):
                continue
            if "runner_config" not in with_:
                continue
            declared = str(with_["runner_config"])
            producer = _NEEDS_JOB.search(str(with_.get("rust_ext_artifact", "")))
            # One entry per pool, since it is the pool's arch that decides which
            # prefix that install looks up; None marks one this check cannot resolve.
            resolved = _runner_configs_of(job=job, declared=declared)
            found += [
                _StageCaller(
                    file=str(path.relative_to(root)),
                    job=name,
                    declared=declared,
                    runner_config=runner_config,
                    producer_job=producer.group(1) if producer else None,
                    skips=bool(with_.get("skip_prebuilt_rust_ext")),
                )
                for runner_config in (resolved or [None])
            ]
    return found


def _runner_config_labels(root: pathlib.Path) -> dict[str, str]:
    configs = _load(root / RUNNER_CONFIGS)["runner_configs"]
    return {name: str(config["runs_on"]) for name, config in configs.items()}


def _unknown_label(where: str, label: str) -> str:
    return (
        f"ERROR: {where}: runner label '{label}' has no architecture on record. Add "
        f"it to RUNNER_ARCHES in {pathlib.Path(__file__).name} - the arch is half "
        f"the cache key and cannot be read off the label text."
    )


def _check_producers(
    *, producers: list[_Producer], pattern: "re.Pattern[str]", abis: str
) -> tuple[list[str], dict[str, str]]:
    """Problems, plus the arch each well-formed producer builds for."""
    problems = []
    arch_by_job = {}
    for producer in producers:
        where = f"{producer.file}: job '{producer.job}'"
        match = pattern.match(producer.prefix)
        if not match:
            problems.append(
                f"ERROR: {where} saves under '{producer.prefix}', which no consumer "
                f"can derive ({pattern.pattern}). Producer and consumer must agree, "
                f"or that pool source-builds."
            )
            continue
        arch = match.group(1)
        arch_by_job[producer.job] = arch
        if match.lastindex == 2 and match.group(2) != abis:
            problems.append(
                f"ERROR: {where} saves under '{producer.prefix}', but the compile "
                f"matrix builds {abis}. The interpreter set is part of the key "
                f"because no crate sets abi3, so no consumer would derive this one."
            )
        for field, label in (
            ("runs_on", producer.runs_on),
            ("collect_runs_on", producer.collect_runs_on),
        ):
            label_arch = RUNNER_ARCHES.get(label)
            if label_arch is None:
                problems.append(_unknown_label(where=f"{where} ({field})", label=label))
            elif label_arch != arch:
                problems.append(
                    f"ERROR: {where} saves under '{producer.prefix}' but {field} is "
                    f"'{label}' ({label_arch}). The compile job must be the arch it "
                    f"builds for, and the collect job too - its GLIBC gate runs "
                    f"objdump on the .so."
                )
    return problems, arch_by_job


def _check_consumers(
    *,
    callers: list[_StageCaller],
    labels: dict[str, str],
    arch_by_job: dict[str, str],
) -> list[str]:
    problems = []
    served = set()
    for caller in callers:
        where = f"{caller.file}: job '{caller.job}'"
        if caller.runner_config is None:
            problems.append(
                f"ERROR: {where}: cannot resolve runner_config '{caller.declared}' "
                f"to a pool, so its architecture is unknown. Pass a literal or a "
                f"matrix reference this check can expand."
            )
            continue
        label = labels.get(caller.runner_config)
        if label is None:
            problems.append(
                f"ERROR: {where}: runner_config '{caller.runner_config}' is not in "
                f"{RUNNER_CONFIGS}."
            )
            continue
        arch = RUNNER_ARCHES.get(label)
        if arch is None:
            problems.append(_unknown_label(where=where, label=label))
            continue
        if not caller.skips:
            served.add(arch)
        if caller.producer_job is None:
            continue
        producer_arch = arch_by_job.get(caller.producer_job)
        if producer_arch is None:
            problems.append(
                f"ERROR: {where} takes its artifact from '{caller.producer_job}', "
                f"which is not a producer in that file."
            )
        elif producer_arch != arch:
            problems.append(
                f"ERROR: {where} runs on {arch} ({label}) but takes its artifact "
                f"from '{caller.producer_job}', which builds {producer_arch}. The "
                f"modules would be ignored and the stage would compile during "
                f"install."
            )

    orphaned = served - set(arch_by_job.values())
    if orphaned:
        problems.append(
            f"ERROR: stages run on {sorted(orphaned)} with no producer for it, so "
            f"they can never get a prebuild. Add a caller of {BUILD_WORKFLOW} for "
            f"that arch, or set skip_prebuilt_rust_ext on those stages."
        )
    return problems


def _check_hashed_inputs(sites: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    if not sites:
        return ["ERROR: no hashFiles(...) cache key found; this check is dead."]
    if len({args for _, args in sites}) == 1:
        return []
    detail = "\n".join(f"  {path}: {list(args)}" for path, args in sites)
    return [
        "ERROR: rust-ext cache key inputs do not match. Every lookup/save/restore "
        f"site must hash the same inputs.\n{detail}"
    ]


def check(root: pathlib.Path) -> list[str]:
    """Every problem found, one message per entry; empty means the sites agree."""
    action = root / DOWNLOAD_ACTION
    try:
        literal = _derived_prefix(action=action)
        if literal is None:
            return [
                f"ERROR: expected exactly one `rust-ext-{_UNAME}-...` literal in "
                f"{DOWNLOAD_ACTION}; this check cannot tell what consumers restore."
            ]
        abis = _interpreter_tokens(root=root)
        producers = _producers(root=root)
        callers = _stage_callers(root=root)
        labels = _runner_config_labels(root=root)
        sites = [
            (BUILD_WORKFLOW, args) for args in _hashed_inputs(root / BUILD_WORKFLOW)
        ]
        sites += [(DOWNLOAD_ACTION, args) for args in _hashed_inputs(action)]
    except _Unparsable as error:
        return [str(error)]
    except (KeyError, TypeError) as error:
        return [
            f"ERROR: {BUILD_WORKFLOW} or {RUNNER_CONFIGS} is not shaped as this check "
            f"expects ({error!r}); update the check with the change."
        ]

    problems = []
    if not _restore_key_uses_derived_prefix(action=action):
        problems.append(
            f"ERROR: {DOWNLOAD_ACTION} derives '{literal}' but its cache restore key "
            f"does not interpolate that step's output, so consumers look up a prefix "
            f"no producer saves."
        )
    if not producers:
        return problems + [
            f"ERROR: no caller of {BUILD_WORKFLOW} found; this check is dead."
        ]

    producer_problems, arch_by_job = _check_producers(
        producers=producers, pattern=_prefix_pattern(literal), abis=abis
    )
    return (
        problems
        + producer_problems
        + _check_consumers(callers=callers, labels=labels, arch_by_job=arch_by_job)
        + _check_hashed_inputs(sites)
    )


def main() -> int:
    problems = check(pathlib.Path("."))
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
